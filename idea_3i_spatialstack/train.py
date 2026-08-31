"""Qwen3-VL SFT with SpatialStack layered VGGT fusion (idea_3i_spatialstack).

VGGTOmega (frozen) features from several aggregator layers are projected by one
merger each and added into the LLM decoder's video-token hidden states at several
decoder layers (see model_with_vggt.py). Video frames are read from the
pre-extracted cache shared with idea_3i (run idea_3i/pre_extract_videos.py first).
Trainable: the per-layer mergers (``vggt_projector``) + LoRA adapters on the LLM.

By default this runs an *overfitting sanity check*: it trains on the first
``--overfit_num_samples`` VSI-Bench (ScanNet++) eval rows and evaluates on those
same rows, so a healthy setup should drive train loss toward zero and VSI-Bench
accuracy toward 1.0.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i/pre_extract_videos.py          # once, to build the frame cache
  bash scripts/idea_3i_spatialstack/overfit.sh
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig
from transformers import AutoProcessor, HfArgumentParser
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# Put the local modules (collator, callbacks, model_with_vggt) ahead of the common
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
from data import build_train_dataset, build_vsibench_eval_dataset, is_scannet_record, load_jsonl
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

logging.getLogger("decord").disabled = True

_DEFAULT_VGGT_CHECKPOINT = (
    "/home/ducpham/scratch/Working/cache/hub/models--facebook--VGGT-Omega/"
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
    """Location of the pre-extracted frame cache (see idea_3i/pre_extract_videos.py)."""

    cache_root: Optional[str] = field(
        default=None,
        metadata={"help": "Frame cache dir. Defaults to <data_root>/../vsi_590k_frame_cache."},
    )


@dataclass
class VggtFusionArguments:
    """VGGT encoder + SpatialStack layered fusion settings."""

    vggt_checkpoint: str = field(default=_DEFAULT_VGGT_CHECKPOINT)
    vggt_embed_dim: int = field(default=2048, metadata={"help": "VGGTOmega token dim."})
    geometry_encoder_layers: List[int] = field(
        default_factory=lambda: [11, 17, 23],
        metadata={"help": "VGGT aggregator layers to read (only 4/11/17/23 are cached)."},
    )
    geometry_fusion_layers: List[int] = field(
        default_factory=lambda: [0, 1, 2],
        metadata={"help": "LLM decoder layers each VGGT layer is fused into (same length)."},
    )
    merger_hidden_dim: int = field(default=4096, metadata={"help": "Hidden width of each merger MLP."})
    fusion_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate LR for the mergers (vggt_projector). None = use --learning_rate."},
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
        # Train and checkpoint the mergers alongside the LoRA adapter (otherwise
        # PEFT saves only LoRA keys and the fusion modules are dropped).
        modules_to_save=["vggt_projector"],
    )


def state_dict_without_vggt_encoder(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop frozen VGGTOmega encoder weights; keep Qwen VL + LoRA + vggt_projector."""
    return {k: v for k, v in state_dict.items() if "vggt_model" not in k.split(".")}


class SFTTrainerSaveLlMAndProjectorOnly(SFTTrainer):
    """Omit frozen VGGTOmega weights from checkpoints; optional separate fusion LR."""

    def __init__(self, *args, fusion_lr: Optional[float] = None, **kwargs):
        self.fusion_lr = fusion_lr
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict: Dict[str, Any] | None = None) -> None:
        if state_dict is None:
            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            state_dict = unwrapped.state_dict()
        state_dict = state_dict_without_vggt_encoder(state_dict)
        return super()._save(output_dir, state_dict=state_dict)

    def create_optimizer(self):
        """Same as HF Trainer, but put vggt_projector params in their own LR group."""
        if self.fusion_lr is None or self.optimizer is not None:
            return super().create_optimizer()

        opt_model = self.model
        decay = set(self.get_decay_parameter_names(opt_model))
        is_fusion = lambda n: "vggt_projector" in n
        named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
        groups = [
            {"params": [p for n, p in named if n in decay and not is_fusion(n)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in named if n not in decay and not is_fusion(n)],
             "weight_decay": 0.0},
            {"params": [p for n, p in named if n in decay and is_fusion(n)],
             "weight_decay": self.args.weight_decay, "lr": self.fusion_lr},
            {"params": [p for n, p in named if n not in decay and is_fusion(n)],
             "weight_decay": 0.0, "lr": self.fusion_lr},
        ]
        groups = [g for g in groups if g["params"]]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        base_lr = optimizer_kwargs.get("lr", self.args.learning_rate)
        logging.info(
            "Optimizer LR groups: %s (base_lr=%s, fusion_lr=%s)",
            [(g.get("lr", base_lr), sum(p.numel() for p in g["params"])) for g in groups],
            base_lr, self.fusion_lr,
        )
        return self.optimizer


def main() -> None:
    setup_logging()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            VsibenchEvalArguments,
            OverfitArguments,
            CacheArguments,
            VggtFusionArguments,
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
        vggt_embed_dim=vggt_args.vggt_embed_dim,
        geometry_encoder_layers=vggt_args.geometry_encoder_layers,
        geometry_fusion_layers=vggt_args.geometry_fusion_layers,
        merger_hidden_dim=vggt_args.merger_hidden_dim,
    )
    logging.info(
        "Layered fusion: VGGT layers %s -> decoder layers %s",
        vggt_args.geometry_encoder_layers, vggt_args.geometry_fusion_layers,
    )

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    peft_config = build_lora_config(model_args)

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

    trainer = SFTTrainerSaveLlMAndProjectorOnly(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=callbacks,
        fusion_lr=vggt_args.fusion_lr,
    )

    for cb in callbacks:
        if isinstance(cb, VsibenchMetricsCallback):
            cb.model = trainer.model

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logging.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt == "latest":
        resume_ckpt = True

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
