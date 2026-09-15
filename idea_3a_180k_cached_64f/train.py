"""Plain Qwen3-VL LoRA SFT on the 180k VQA mix (idea_3a_180k_cached_64f).

Fork of idea_3i_180k_cached_64f_lat512/train.py with VGGT / Perceiver removed: the base
Qwen3VLForConditionalGeneration is loaded directly, LoRA on the LLM only, no placeholder
tokens, no VGGT cache. Frames come from the same frame cache (64 frames, 128 tok/frame),
so this is the no-3D-context control for the lat512 run.

  python idea_3a_180k_cached_64f/train.py --jsonl_path <mix> --frame_cache_root <cache> \
      --output_dir ./results/idea_3a_180k_cached_64f --overfit_on_eval False ...
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig
from datasets import Dataset
from transformers import AutoProcessor, HfArgumentParser, Qwen3VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# Put fork-local modules (collator, callbacks) ahead of the common package so they
# shadow the common versions. Remove any existing entry first and re-insert: running as
# a script auto-adds _THIS_DIR, so a plain "not in sys.path" guard would skip it and let
# _COMMON land in front instead.
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
from data import build_vsibench_eval_dataset

logging.getLogger("decord").disabled = True


@dataclass
class JointArguments:
    """Knobs that exist only because the mix format allows two objectives."""

    box_noise: float = field(
        default=0.005,
        metadata={"help": "Scale each det box value by 1 +/- U(box_noise). det rows only "
                          "(none in the 180k mix); 0.0 disables."},
    )

    frame_cache_root: str = field(
        default="",
        metadata={"help": "pre_extract_frames.py cache; vqa rows read pre-decoded Qwen frames "
                          "from it instead of decoding video (a miss raises). Empty = decode live."},
    )


@dataclass
class OverfitArguments:
    """Controls the overfitting sanity run (train == a few eval rows)."""

    overfit_on_eval: bool = field(
        default=True,
        metadata={"help": "Train on the first N VSI-Bench eval rows (memorization test)."},
    )
    overfit_num_samples: int = field(
        default=20,
        metadata={"help": "Number of eval rows to overfit on when overfit_on_eval is set."},
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


def load_mix(path: Path) -> Dataset:
    """The build_mix.py manifest, as-is.

    A plain jsonl read rather than data.build_train_dataset: that helper filters VSI-590K
    by source and knows nothing about det rows, and the mix is already exactly the corpus
    to train on. Kept as a datasets.Dataset so the Trainer's sampler/resume behave as they
    do everywhere else in this repo.
    """
    rows = [json.loads(line) for line in open(path)]
    if not rows:
        raise ValueError(f"{path} is empty. Run build_mix.py first.")
    missing = {r.get("task") for r in rows} - {"vqa", "det"}
    if missing:
        raise ValueError(f"{path} has rows with unknown task {missing}; rebuild the mix.")
    # Column-per-key, so absent fields (a vqa row has no `images`) become None instead of
    # a schema clash.
    keys = {k for r in rows for k in r}
    return Dataset.from_dict({k: [r.get(k) for r in rows] for k in keys})


def build_lora_config(model_args: ModelArguments) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


class SFTTrainerCpuBatches(SFTTrainer):
    """SFTTrainer whose dataloader keeps micro-batches on the CPU until their own step."""

    def get_train_dataloader(self):
        """Keep micro-batches on the CPU until their own training_step.

        Trainer.get_batch_samples pulls every micro-batch of the accumulation window
        before running any of them, and a device-placed dataloader lands each one on
        the GPU as it is pulled (accum 128 x pixel_values_videos per rank). Prepared
        without device placement, each batch is moved by training_step's
        _prepare_inputs instead, so only one is ever on the GPU.
        """
        placement = self.accelerator.device_placement
        self.accelerator.device_placement = False
        try:
            dl = super().get_train_dataloader()
        finally:
            self.accelerator.device_placement = placement
        logging.info("train dataloader: %s device=%s (device_placement forced off)",
                     type(dl).__name__, getattr(dl, "device", None))
        return dl


def main() -> None:
    setup_logging()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            VsibenchEvalArguments,
            JointArguments,
            OverfitArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        vsibench_args,
        joint_args,
        overfit_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    configure_hf_home(training_args)

    output_dir = Path(training_args.output_dir)
    data_root = Path(data_args.data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset = build_vsibench_eval_dataset(
        hf_home=training_args.hf_home or os.environ.get("HF_HOME"),
        max_samples=(
            overfit_args.overfit_num_samples
            if overfit_args.overfit_on_eval
            else vsibench_args.vsibench_max_eval_samples
        ),
        chosen_dataset="scannet"
    )
    logging.info(
        "VSI-Bench eval rows: %s | live decode (%s fps, max %s frames)",
        len(eval_dataset), data_args.video_fps, data_args.video_max_frames,
    )

    if overfit_args.overfit_on_eval:
        # VSI-Bench rows carry no `task`, and the collator refuses to guess one. The eval
        # callback stamps "vqa" per row as it reads; the train side has to stamp it here,
        # on a copy, so eval_dataset stays exactly what the callback expects.
        train_dataset = eval_dataset.add_column("task", ["vqa"] * len(eval_dataset))
        logging.info("OVERFIT MODE: training on %d VSI-Bench eval rows (train == eval).", len(train_dataset))
    else:
        # build_mix.py already did the source filtering, the budgeting and the shuffle,
        # so there is nothing to select here -- the mix IS the corpus. Rows arrive
        # pre-stamped with `task`, which the collator dispatches on.
        train_dataset = load_mix(Path(data_args.jsonl_path))
        counts = Counter(r["task"] for r in train_dataset)
        logging.info(
            "Train rows: %d | mix=%s | shares=%s",
            len(train_dataset), dict(counts),
            {k: f"{100 * v / len(train_dataset):.1f}%" for k, v in counts.items()},
        )
        # 180k mix: rows carry `source` (vsi590k / vlm3r); show both are in the stream.
        logging.info("Train sources: %s", dict(Counter(r.get("source") for r in train_dataset)))
        logging.info("Train question types: %s", dict(Counter(r.get("question_type") for r in train_dataset)))
        logging.info("First rows (source, type, video): %s",
                     [(r.get("source"), r.get("question_type"), Path(r["video"]).name) for r in train_dataset.select(range(5))])

    dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    peft_config = build_lora_config(model_args)

    collator = make_collator(
        processor,
        data_root=data_root,
        video_fps=data_args.video_fps,
        video_max_frames=data_args.video_max_frames,
        image_patch_size=data_args.image_patch_size,
        box_noise=joint_args.box_noise,
        frame_cache_root=Path(joint_args.frame_cache_root) if joint_args.frame_cache_root else None,
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
                log_path=eval_log,
                eval_steps=vsibench_args.vsibench_eval_steps,
                image_patch_size=data_args.image_patch_size,
                max_new_tokens=vsibench_args.vsibench_max_new_tokens,
                max_eval_samples=vsibench_args.vsibench_max_eval_samples,
                video_fps=data_args.video_fps,
                video_max_frames=data_args.video_max_frames,
            )
        )

    trainer = SFTTrainerCpuBatches(
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

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logging.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    # "latest"/True: resume from the newest checkpoint if one exists, else start
    # fresh. Plain True makes HF raise on the first run (empty output_dir).
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt in (True, "latest", "True"):
        resume_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
