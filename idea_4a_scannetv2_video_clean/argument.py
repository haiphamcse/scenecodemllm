"""CLI dataclasses for this run.

Trimmed copy of finetuning/common/argument.py: only the fields train.py actually reads,
so this directory runs without the common/ package on sys.path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig


# Paths come from the environment so the same code runs unmodified on this machine
# and on Jean Zay, where the data lives on $SCRATCH and the code on $WORK.
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
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.1)


@dataclass
class DataArguments:
    image_patch_size: int = field(default=16)


@dataclass
class VsibenchEvalArguments:
    vsibench_eval_enable: bool = field(
        default=True,
        metadata={"help": "Run the generation-based scene-graph eval via TrainerCallback."},
    )
    vsibench_eval_steps: int = field(
        default=20,
        metadata={"help": "Run the eval callback every N training steps."},
    )
    vsibench_eval_log: Optional[str] = field(
        default=None,
        metadata={"help": "Append-only log (default: <output_dir>/scannet_eval.txt)."},
    )
    vsibench_max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Cap eval rows (None = all). Use e.g. 50 for debug."},
    )


@dataclass
class TrainingArguments(SFTConfig):
    """TRL SFT config; extra HF cache helper."""

    cache_dir: Optional[str] = field(default=None)
    hf_home: Optional[str] = field(
        default=None,
        metadata={"help": f"If set, exports HF_HOME (else default {DEFAULT_HF_HOME})."},
    )
