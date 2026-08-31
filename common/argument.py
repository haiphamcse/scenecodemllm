"""CLI dataclasses for Qwen-VL finetuning (shared across experiments)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig


# Paths come from the environment so the same code runs unmodified on this machine
# and on Jean Zay, where the data lives on $SCRATCH and the code on $WORK. The
# literals below stay as fallbacks, so behaviour here is unchanged when the vars
# are unset. jeanzay/sites/{local,jeanzay}.env define them.
DEFAULT_JSONL = os.environ.get(
    "SR_JSONL", "/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl"
)
DEFAULT_DATA_ROOT = os.environ.get(
    "SR_DATA_ROOT", "/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
)
DEFAULT_HF_HOME = os.environ.get(
    "SR_HF_HOME", "/home/ducpham/scratch/Working/cache"
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-4B-Instruct",
        metadata={"help": "Hugging Face model id or local path."},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention backend, e.g. sdpa or flash_attention_2."},
    )
    lora_enable: bool = field(default=False, metadata={"help": "Train with PEFT LoRA."})
    freeze_llm_bottom_layers: int = field(
        default=0,
        metadata={"help": "Full-FT only: freeze token embeddings + the bottom N LLM decoder layers (0 = train all)."},
    )
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    jsonl_path: str = field(
        default=DEFAULT_JSONL,
        metadata={"help": "VSI-590K jsonl; filtered to scannetppv2 in-domain rows."},
    )
    data_root: str = field(
        default=DEFAULT_DATA_ROOT,
        metadata={"help": "Root directory for relative video paths in jsonl."},
    )
    train_sources: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated VSI-590K source dirs (e.g. scannet,scannetppv2,arkitscenes). "
                    "None = legacy scannetppv2-only filter."
        },
    )
    video_fps: float = field(default=1.0)
    video_max_frames: int = field(default=32)
    image_patch_size: int = field(default=16)


@dataclass
class VsibenchEvalArguments:
    vsibench_eval_enable: bool = field(
        default=True,
        metadata={"help": "Run generation-based VSI-Bench eval via TrainerCallback."},
    )
    vsibench_eval_steps: int = field(
        default=20,
        metadata={"help": "Run VSI-Bench callback every N training steps."},
    )
    vsibench_eval_log: Optional[str] = field(
        default=None,
        metadata={"help": "Append-only log (default: <output_dir>/vsibench_eval.txt)."},
    )
    vsibench_max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Cap ScanNet++ debiased eval rows (None = all ~752). Use e.g. 50 for debug."
        },
    )
    vsibench_max_new_tokens: int = field(default=16)


@dataclass
class TrainingArguments(SFTConfig):
    """TRL SFT config; extra HF cache helper."""

    cache_dir: Optional[str] = field(default=None)
    hf_home: Optional[str] = field(
        default=None,
        metadata={"help": f"If set, exports HF_HOME (else default {DEFAULT_HF_HOME})."},
    )
