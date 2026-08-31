"""Qwen3-VL SFT from a pre-extracted frame cache (idea_3a_cached).

Same plain Qwen3-VL baseline as idea_3a (no auxiliary 3D encoder), but video
frames are read from the pre-extracted cache instead of decoded on every epoch.
Build the cache once with the 64-frame extraction:

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i/pre_extract_videos.py \\
      --max_frames 64 \\
      --cache_root <data_root>/../vsi_590k_frame_cache_64f
  python idea_3a_cached/train.py --output_dir ./results/idea_3a_cached

Overfitting sanity run (train == a few VSI-Bench eval rows, so a healthy setup
drives train loss -> 0 and VSI-Bench accuracy -> 1.0):

  python idea_3a_cached/train.py --overfit_on_eval --output_dir ./results/idea_3a_cached_overfit
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig
from transformers import AutoProcessor, HfArgumentParser, Qwen3VLForConditionalGeneration
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# Put idea_3a_cached-local modules (collator, callbacks) ahead of the common
# package so they shadow the common versions. Remove any existing entry first and
# re-insert: running as a script auto-adds _THIS_DIR, so a plain "not in sys.path"
# guard would skip it and let _COMMON land in front instead.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    VsibenchEvalArguments,
    DEFAULT_HF_HOME,
)
from callbacks import VsibenchMetricsCallback
from collator import make_collator
from data import (
    build_train_dataset,
    build_vsibench_eval_dataset,
    is_scannet_record,
    load_jsonl,
)

logging.getLogger("decord").disabled = True


@dataclass
class OverfitArguments:
    """Opt-in overfitting sanity run (train == a few VSI-Bench eval rows)."""

    overfit_on_eval: bool = field(
        default=False,
        metadata={
            "help": "Train on the first N VSI-Bench eval rows (memorization test) "
            "instead of the full ScanNet++ train jsonl."
        },
    )
    overfit_num_samples: int = field(
        default=20,
        metadata={"help": "Number of eval rows to overfit on when overfit_on_eval is set."},
    )


@dataclass
class CacheArguments:
    """Location of the pre-extracted frame cache (see idea_3i/pre_extract_videos.py)."""

    cache_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Frame cache dir. Defaults to <data_root>/../vsi_590k_frame_cache_64f "
            "(the 64-frame cache)."
        },
    )


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def configure_hf_home(training_args: TrainingArguments) -> None:
    if training_args.hf_home:
        os.environ["HF_HOME"] = training_args.hf_home
    else:
        os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)


def build_lora_config(model_args: ModelArguments) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def main() -> None:
    setup_logging()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            VsibenchEvalArguments,
            OverfitArguments,
            CacheArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        vsibench_args,
        overfit_args,
        cache_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    configure_hf_home(training_args)

    output_dir = Path(training_args.output_dir)
    data_root = Path(data_args.data_root)
    cache_root = (
        Path(cache_args.cache_root)
        if cache_args.cache_root
        else data_root.parent / "vsi_590k_frame_cache_64f"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the VSI-Bench eval set up front when it's needed -- either as the
    # overfit training set or for the in-training metrics callback.
    eval_dataset = None
    if overfit_args.overfit_on_eval or vsibench_args.vsibench_eval_enable:
        eval_dataset = build_vsibench_eval_dataset(
            hf_home=training_args.hf_home or os.environ.get("HF_HOME"),
            max_samples=(
                overfit_args.overfit_num_samples
                if overfit_args.overfit_on_eval
                else vsibench_args.vsibench_max_eval_samples
            ),
        )
        logging.info(
            "VSI-Bench eval rows: %s | frame cache: %s", len(eval_dataset), cache_root
        )

    if overfit_args.overfit_on_eval:
        # Memorization sanity run: the training set *is* the eval set.
        train_dataset = eval_dataset
        logging.info(
            "OVERFIT MODE: training on %d VSI-Bench eval rows (train == eval).",
            len(train_dataset),
        )
    else:
        records = load_jsonl(Path(data_args.jsonl_path))
        train_rows = [r for r in records if is_scannet_record(r)]
        logging.info(
            "Train: %s scannetpp rows from %s total",
            len(train_rows),
            len(records),
        )
        train_dataset = build_train_dataset(data_args.jsonl_path)

    dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    peft_config = build_lora_config(model_args) if model_args.lora_enable else None

    collator = make_collator(
        processor,
        data_root=data_root,
        cache_root=cache_root,
        image_patch_size=data_args.image_patch_size,
    )

    callbacks = []
    if vsibench_args.vsibench_eval_enable:
        eval_log = (
            Path(vsibench_args.vsibench_eval_log)
            if vsibench_args.vsibench_eval_log
            else output_dir / "vsibench_eval.txt"
        )
        callbacks.append(
            VsibenchMetricsCallback(
                model=model,
                processor=processor,
                eval_dataset=eval_dataset,
                data_root=data_root,
                cache_root=cache_root,
                log_path=eval_log,
                eval_steps=vsibench_args.vsibench_eval_steps,
                image_patch_size=data_args.image_patch_size,
                max_new_tokens=vsibench_args.vsibench_max_new_tokens,
                max_eval_samples=vsibench_args.vsibench_max_eval_samples,
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=callbacks,
    )

    for cb in callbacks:
        if isinstance(cb, VsibenchMetricsCallback):
            cb.model = trainer.model

    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt == "latest":
        resume_ckpt = True

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
