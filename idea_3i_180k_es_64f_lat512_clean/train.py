"""Qwen3-VL LoRA SFT with VGGT-Perceiver placeholder scatter, cached VGGT features, det_es frames.

Reads a VQA mix (build_mix_190k.py, default vqa_train_190k_es_clean.jsonl). Qwen frames are each clip's det_es
JPEGs; VGGT features come from the export_vggt_features.py cache (a miss raises).
Trainable: the two Perceivers (``vggt_projector``) + LoRA on the LLM.

  python idea_3i_180k_es_64f_lat512_clean/train.py --jsonl_path <mix> --vggt_cache_root <cache> \
      --output_dir results/idea_3i_180k_es_64f_lat512_clean ...
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
from transformers import AutoProcessor, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# This fork's modules must shadow common's (collator, model_with_vggt). Re-insert rather than
# "if not in sys.path": running as a script already adds _THIS_DIR, behind _COMMON otherwise.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    DEFAULT_HF_HOME,
)
from collator import make_collator
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

_DEFAULT_VGGT_CHECKPOINT = (
    os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)


@dataclass
class VggtPerceiverArguments:
    """VGGT cache + Perceiver latent / placeholder settings."""

    vggt_cache_root: str = field(
        default="",
        metadata={"help": "REQUIRED. Root of the export_vggt_features.py cache."},
    )
    vggt_checkpoint: str = field(
        default=_DEFAULT_VGGT_CHECKPOINT,
        metadata={"help": "NOT loaded -- part of the cache key, so a run cannot silently "
                          "consume features from a different VGGT checkpoint."},
    )
    vggt_embed_dim: int = field(default=2048, metadata={"help": "VGGTOmega token dim."})
    frame_num_latents: int = field(default=128, metadata={"help": "frame_encoder latents (= placeholders)."})
    camera_num_latents: int = field(default=32, metadata={"help": "camera_encoder latents (= placeholders)."})
    frame_placeholder_token: str = field(default="<|quad_start|>")
    camera_placeholder_token: str = field(default="<|quad_end|>")
    frame_widening_factor: int = field(default=1, metadata={"help": "frame_encoder cross+self MLP widening (default 1 = baseline)."})
    vggt_image_resolution: int = field(
        default=512,
        metadata={"help": "Balanced-resize target the cache was exported at. Checked "
                          "against every entry's metadata; a mismatch raises."},
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
    """The mix jsonl as-is (already filtered, budgeted and shuffled)."""
    rows = [json.loads(line) for line in open(path)]
    if not rows:
        raise ValueError(f"{path} is empty. Run build_mix_180k.py first.")
    bad = {r.get("task") for r in rows} - {"vqa"}
    if bad:
        raise ValueError(f"{path} has rows with task {bad}; this fork trains vqa rows only.")
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
        # Train and checkpoint the Perceivers with the adapter (else PEFT drops them).
        modules_to_save=["vggt_projector"],
    )


class CpuBatchSFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        """Keep micro-batches on the CPU until their own training_step.

        Trainer.get_batch_samples pulls the whole accumulation window before running any of
        it; a device-placed loader parks all of it (with ~70-270 MB of cached VGGT features
        per sample) on the GPU -- job 1967515 OOM'd at 77.5 GiB. Without device placement,
        _prepare_inputs moves one batch at a time.
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
        (ModelArguments, DataArguments, VggtPerceiverArguments, TrainingArguments)
    )
    model_args, data_args, vggt_args, training_args = parser.parse_args_into_dataclasses()

    configure_hf_home(training_args)

    # Checked before the model load.
    if not vggt_args.vggt_cache_root:
        raise ValueError(
            "--vggt_cache_root is required; build the cache with export_vggt_features.py first."
        )
    if not Path(vggt_args.vggt_cache_root).is_dir():
        raise NotADirectoryError(
            f"--vggt_cache_root {vggt_args.vggt_cache_root} does not exist. "
            "Run export_vggt_features.py first."
        )

    output_dir = Path(training_args.output_dir)
    data_root = Path(data_args.data_root)
    frame_placeholder_text = vggt_args.frame_placeholder_token * vggt_args.frame_num_latents
    camera_placeholder_text = vggt_args.camera_placeholder_token * vggt_args.camera_num_latents
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_mix(Path(data_args.jsonl_path))
    logging.info("Train rows: %d", len(train_dataset))
    logging.info("Train sources: %s", dict(Counter(r.get("source") for r in train_dataset)))
    logging.info("Train question types: %s", dict(Counter(r.get("question_type") for r in train_dataset)))
    logging.info("First rows (source, type, video): %s",
                 [(r.get("source"), r.get("question_type"), Path(r["video"]).name) for r in train_dataset.select(range(5))])

    dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model.initialize_projector(
        tokenizer=processor.tokenizer,
        vggt_embed_dim=vggt_args.vggt_embed_dim,
        frame_num_latents=vggt_args.frame_num_latents,
        camera_num_latents=vggt_args.camera_num_latents,
        frame_placeholder_token=vggt_args.frame_placeholder_token,
        camera_placeholder_token=vggt_args.camera_placeholder_token,
        frame_widening_factor=vggt_args.frame_widening_factor,
    )

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    collator = make_collator(
        processor,
        data_root=data_root,
        vggt_cache_root=Path(vggt_args.vggt_cache_root),
        vggt_checkpoint=vggt_args.vggt_checkpoint,
        frame_placeholder_text=frame_placeholder_text,
        camera_placeholder_text=camera_placeholder_text,
        video_fps=data_args.video_fps,
        video_max_frames=data_args.video_max_frames,
        image_patch_size=data_args.image_patch_size,
        vggt_image_resolution=vggt_args.vggt_image_resolution,
    )

    trainer = CpuBatchSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=build_lora_config(model_args),
        processing_class=processor,
    )

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
