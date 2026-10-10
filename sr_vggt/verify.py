"""Check an experiment can train and evaluate, with exact counts (login node, CPU, offline).

Every video of the mix must exist under --data_root; VSI-Bench full must load with
5,130 rows and all its videos; the weights must resolve inside HF_HOME. Exits 1 on any problem.

    python -m sr_vggt.verify <config flags> --jsonl_path <mix> --data_root <VSI-590K root> \
        --vsibench_dir <dir> --model_name_or_path <qwen>
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from transformers import HfArgumentParser

from . import config as cfg
from .data import load_mix, video_path
from .eval import build_rows


@dataclass
class VerifyArguments:
    jsonl_path: str = ""
    data_root: str = ""
    vsibench_dir: str = ""
    model_name_or_path: str = ""


def main() -> int:
    a, va = HfArgumentParser((VerifyArguments, cfg.VggtArguments)).parse_args_into_dataclasses()
    problems = []

    from huggingface_hub import snapshot_download
    try:
        snapshot_download(a.model_name_or_path, allow_patterns=["config.json"])
        ckpt = va.checkpoint()
        print(f"weights: {a.model_name_or_path} ok, VGGT {ckpt} ({os.path.getsize(ckpt) / 1e9:.1f} GB)")
    except Exception as exc:
        problems.append(f"weights: {type(exc).__name__}: {exc} (run scripts/download_weights.sh)")

    rows = load_mix(a.jsonl_path)
    videos = sorted({video_path(r, a.data_root) for r in rows})
    missing = [v for v in videos if not os.path.isfile(v)]
    problems += [f"missing video: {v}" for v in missing]
    print(f"mix {a.jsonl_path}: {len(rows)} rows, {len(videos)} videos, {len(missing)} missing under {a.data_root}")

    try:
        eval_rows = build_rows(a.vsibench_dir)
        print(f"VSI-Bench full: {len(eval_rows)} rows, {len({r['video'] for r in eval_rows})} videos, all on disk")
        if len(eval_rows) != 5130:
            problems.append(f"VSI-Bench full has {len(eval_rows)} rows, expected 5130")
    except Exception as exc:
        problems.append(f"VSI-Bench: {type(exc).__name__}: {exc}")

    print(f"{len(problems)} problems")
    for p in problems[:20]:
        print("  ", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
