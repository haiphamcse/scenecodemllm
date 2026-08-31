"""Qwen3-VL SFT with VGGT-Perceiver placeholder scatter (idea_3i).

VGGTOmega (frozen) features are compressed by two Perceiver IO encoders and
scattered into ``<|quad_start|>``/``<|quad_end|>`` placeholder tokens
(see model_with_vggt.py). Video frames are read from the pre-extracted cache
(run idea_3i/pre_extract_videos.py first). Trainable: the two Perceivers
(``vggt_projector``) + LoRA adapters on the LLM.

By default this runs an *overfitting sanity check*: it trains on the first
``--overfit_num_samples`` VSI-Bench (ScanNet++) eval rows and evaluates on those
same rows, so a healthy setup should drive train loss toward zero and VSI-Bench
accuracy toward 1.0.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i/pre_extract_videos.py          # once, to build the frame cache
  python idea_3i/train.py --output_dir ./results/idea_3i
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig
from transformers import AutoProcessor, HfArgumentParser
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# Put idea_3i-local modules (collator, callbacks, model_with_vggt) ahead of the
# common package so they shadow the common versions. Remove any existing entry
# first and re-insert: running as a script auto-adds _THIS_DIR, so a plain
# "not in sys.path" guard would skip it and let _COMMON land in front instead.
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
from collator import make_collator, resolve_video_path
from frame_cache import cache_paths_for
from safetensors.torch import load_file
from data import build_train_dataset, build_vsibench_eval_dataset, is_scannet_record, load_jsonl
from transformers.trainer_utils import get_last_checkpoint
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

logging.getLogger("decord").disabled = True

_DEFAULT_VGGT_CHECKPOINT = (
    os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
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


@dataclass
class CacheArguments:
    """Location of the pre-extracted frame cache (see pre_extract_videos.py)."""

    init_projector: str = field(
        default="",
        metadata={"help": "vggt_projector.safetensors from merge_lora.py. Loaded AFTER "
                          "initialize_vggt, which builds a fresh random projector."},
    )
    cache_root: Optional[str] = field(
        default=None,
        metadata={"help": "Frame cache dir. Defaults to <data_root>/../vsi_590k_frame_cache."},
    )


@dataclass
class VggtPerceiverArguments:
    """VGGT encoder + Perceiver latent / placeholder settings."""

    vggt_checkpoint: str = field(default=_DEFAULT_VGGT_CHECKPOINT)
    vggt_embed_dim: int = field(default=2048, metadata={"help": "VGGTOmega token dim."})
    frame_num_latents: int = field(default=128, metadata={"help": "frame_encoder latents (= placeholders)."})
    camera_num_latents: int = field(default=32, metadata={"help": "camera_encoder latents (= placeholders)."})
    frame_placeholder_token: str = field(default="<|quad_start|>")
    camera_placeholder_token: str = field(default="<|quad_end|>")
    frame_widening_factor: int = field(default=1, metadata={"help": "frame_encoder cross+self MLP widening (default 1 = baseline)."})
    perceiver_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate LR for the Perceivers (vggt_projector). None = use --learning_rate for all."},
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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Train and checkpoint the two Perceivers alongside the LoRA adapter
        # (otherwise PEFT saves only LoRA keys and the projector is dropped).
        modules_to_save=["vggt_projector"],
    )


def state_dict_without_vggt_encoder(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop frozen VGGTOmega encoder weights; keep Qwen VL + LoRA + vggt_projector."""
    return {k: v for k, v in state_dict.items() if "vggt_model" not in k.split(".")}


class SFTTrainerSaveLlMAndProjectorOnly(SFTTrainer):
    """Omit frozen VGGTOmega weights from checkpoints; optional separate Perceiver LR."""

    def __init__(self, *args, perceiver_lr: Optional[float] = None, **kwargs):
        self.perceiver_lr = perceiver_lr
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict: Dict[str, Any] | None = None) -> None:
        if state_dict is None:
            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            state_dict = unwrapped.state_dict()
        state_dict = state_dict_without_vggt_encoder(state_dict)
        return super()._save(output_dir, state_dict=state_dict)

    def create_optimizer(self):
        """Same as HF Trainer, but put vggt_projector params in their own LR group."""
        if self.perceiver_lr is None or self.optimizer is not None:
            return super().create_optimizer()

        opt_model = self.model
        decay = set(self.get_decay_parameter_names(opt_model))
        is_perc = lambda n: "vggt_projector" in n
        named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
        groups = [
            {"params": [p for n, p in named if n in decay and not is_perc(n)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in named if n not in decay and not is_perc(n)],
             "weight_decay": 0.0},
            {"params": [p for n, p in named if n in decay and is_perc(n)],
             "weight_decay": self.args.weight_decay, "lr": self.perceiver_lr},
            {"params": [p for n, p in named if n not in decay and is_perc(n)],
             "weight_decay": 0.0, "lr": self.perceiver_lr},
        ]
        groups = [g for g in groups if g["params"]]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        base_lr = optimizer_kwargs.get("lr", self.args.learning_rate)
        logging.info(
            "Optimizer LR groups: %s (base_lr=%s, perceiver_lr=%s)",
            [(g.get("lr", base_lr), sum(p.numel() for p in g["params"])) for g in groups],
            base_lr, self.perceiver_lr,
        )
        return self.optimizer


def keep_cached_rows(dataset, data_root: Path, cache_root: Path):
    """Drop rows whose frame cache is missing.

    load_cached_frames() does a bare np.load, and train.py never checked the cache, so an
    uncached row raises FileNotFoundError mid-epoch rather than being skipped. The cache
    here covers 856 of 5963 videos, so most rows would hit that.

    Resolution goes through the same resolve_video_path / cache_paths_for the collator
    uses -- reimplementing the mirroring here would silently diverge from what is actually
    loaded. One existence check per DISTINCT video, not per row: 374k rows share ~6k
    videos, so this is ~6k stats instead of 374k.
    """
    seen: Dict[str, bool] = {}

    def has_cache(video: str) -> bool:
        if video not in seen:
            npy_path, json_path = cache_paths_for(
                resolve_video_path(data_root, video), cache_root
            )
            seen[video] = npy_path.exists() and json_path.exists()
        return seen[video]

    keep = [i for i, v in enumerate(dataset["video"]) if has_cache(v)]
    logging.info(
        "Frame cache: %d/%d rows usable (%d/%d distinct videos cached).",
        len(keep), len(dataset), sum(seen.values()), len(seen),
    )
    if not keep:
        raise RuntimeError(f"No row has a frame cache under {cache_root}.")
    return dataset.select(keep)


def main() -> None:
    setup_logging()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            VsibenchEvalArguments,
            OverfitArguments,
            CacheArguments,
            VggtPerceiverArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        vsibench_args,
        overfit_args,
        cache_args,
        vggt_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    configure_hf_home(training_args)

    output_dir = Path(training_args.output_dir)
    data_root = Path(data_args.data_root)
    cache_root = (
        Path(cache_args.cache_root)
        if cache_args.cache_root
        else data_root.parent / "vsi_590k_frame_cache"
    )
    frame_placeholder_text = vggt_args.frame_placeholder_token * vggt_args.frame_num_latents
    camera_placeholder_text = vggt_args.camera_placeholder_token * vggt_args.camera_num_latents
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset = build_vsibench_eval_dataset(
        hf_home=training_args.hf_home or os.environ.get("HF_HOME"),
        max_samples=(
            overfit_args.overfit_num_samples
            if overfit_args.overfit_on_eval
            else vsibench_args.vsibench_max_eval_samples
        ),
    )
    logging.info("VSI-Bench eval rows: %s | frame cache: %s", len(eval_dataset), cache_root)

    if overfit_args.overfit_on_eval:
        train_dataset = eval_dataset
        logging.info("OVERFIT MODE: training on %d VSI-Bench eval rows (train == eval).", len(train_dataset))
    else:
        records = load_jsonl(Path(data_args.jsonl_path))
        train_rows = [r for r in records if is_scannet_record(r)]
        logging.info("Train: %s scannetpp rows from %s total", len(train_rows), len(records))
        train_dataset = build_train_dataset(data_args.jsonl_path)
        train_dataset = keep_cached_rows(train_dataset, data_root, cache_root)

    dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model.initialize_vggt(
        vggt_args.vggt_checkpoint,
        tokenizer=processor.tokenizer,
        vggt_embed_dim=vggt_args.vggt_embed_dim,
        frame_num_latents=vggt_args.frame_num_latents,
        camera_num_latents=vggt_args.camera_num_latents,
        frame_placeholder_token=vggt_args.frame_placeholder_token,
        camera_placeholder_token=vggt_args.camera_placeholder_token,
        frame_widening_factor=vggt_args.frame_widening_factor,
    )

    if cache_args.init_projector:
        # initialize_vggt() has just replaced the projector with fresh random weights, so
        # this has to come after it. Putting these tensors in the merged model dir instead
        # would be a silent no-op: they do not exist at from_pretrained() time.
        proj = load_file(cache_args.init_projector)
        missing, unexpected = model.load_state_dict(proj, strict=False)
        landed = [k for k in proj if k not in unexpected]
        if not landed:
            raise RuntimeError(
                f"init_projector matched nothing in the model: {list(proj)[:3]}"
            )
        logging.info("init_projector %s: loaded %d/%d tensors.",
                     cache_args.init_projector, len(landed), len(proj))

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    # idea_3i trains the Perceivers (modules_to_save) + LoRA adapters on the LLM.
    peft_config = build_lora_config(model_args)

    collator = make_collator(
        processor,
        data_root=data_root,
        cache_root=cache_root,
        frame_placeholder_text=frame_placeholder_text,
        camera_placeholder_text=camera_placeholder_text,
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
                frame_placeholder_text=frame_placeholder_text,
                camera_placeholder_text=camera_placeholder_text,
                log_path=eval_log,
                eval_steps=vsibench_args.vsibench_eval_steps,
                image_patch_size=data_args.image_patch_size,
                max_new_tokens=vsibench_args.vsibench_max_new_tokens,
                max_eval_samples=vsibench_args.vsibench_max_eval_samples,
            )
        )

    trainer = SFTTrainerSaveLlMAndProjectorOnly(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=callbacks,
        perceiver_lr=vggt_args.perceiver_lr,
    )

    for cb in callbacks:
        if isinstance(cb, VsibenchMetricsCallback):
            cb.model = trainer.model

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logging.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    # "latest" must resolve to an actual path, or None on a fresh output dir. Mapping it to
    # True (as idea_3i did) makes HF Trainer raise "No valid checkpoint found" on the very
    # first run, since True means "resume, and fail if you cannot".
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt in (True, "latest", "True"):
        resume_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        logging.info("resume_from_checkpoint=latest -> %s", resume_ckpt or "fresh start")

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
