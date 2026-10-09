"""Full-scan check that an experiment can train and evaluate, with exact counts.

Every video of the mix must have det_es frames and a fresh cache entry whose tensors hold one
row per frame; the VSI-Bench full split must load with every video on disk; the weights must
resolve offline. Prints the counts and the first problems; exits 1 on any problem.

    python -m sr_vggt.verify <config flags> --jsonl_path <mix> --data_root <root> \
        --vggt_cache_root <cache> --vsibench_videos <dir> --model_name_or_path <qwen>
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from transformers import HfArgumentParser

from . import config as cfg
from . import vggt_cache
from .data import det_es_dir, load_mix, video_path
from .eval import build_rows

CAMERA_TOKENS = {"omega": 17, "vggt1b": 5}


@dataclass
class VerifyArguments:
    jsonl_path: str = ""
    data_root: str = ""
    vggt_cache_root: str = ""
    vsibench_videos: str = ""
    model_name_or_path: str = ""


def main() -> int:
    a, va = HfArgumentParser((VerifyArguments, cfg.VggtArguments)).parse_args_into_dataclasses()
    problems = []
    data_root = a.data_root or None

    from huggingface_hub import snapshot_download
    snapshot_download(a.model_name_or_path, allow_patterns=["config.json"])  # raises if not in HF_HOME
    ckpt = va.checkpoint()
    print(f"weights: {a.model_name_or_path} ok, VGGT {ckpt} ({os.path.getsize(ckpt) / 1e9:.1f} GB)")

    rows = load_mix(a.jsonl_path)
    videos = sorted({vggt_cache.clip_key(r, data_root): video_path(r, data_root) for r in rows}.items())
    print(f"mix {a.jsonl_path}: {len(rows)} rows, {len(videos)} distinct videos")
    params = va.cache_params()
    for key, path in videos:
        n = len(list(det_es_dir(path).glob("frame*.jpg")))
        if not n:
            problems.append(f"no det_es frames: {path}")
            continue
        entry = vggt_cache.cache_path(a.vggt_cache_root, key)
        if not vggt_cache.is_fresh(a.vggt_cache_root, key, params):
            problems.append(f"cache missing/stale: {entry}")
            continue
        shapes = vggt_cache.shapes(entry)
        want = ({f"patch_{i}" for i in range(len(va.vggt_layers))} if va.vggt_frames == "native"
                else {"patch", "camera"})
        if set(shapes) != want or any(s[0] != n for s in shapes.values()) or (
                "camera" in shapes and shapes["camera"][1] != CAMERA_TOKENS[va.vggt_backbone]):
            problems.append(f"shape mismatch ({n} det_es frames): {entry} {shapes}")
    print(f"cache {a.vggt_cache_root}: {len(videos) - sum(p.startswith(('cache', 'shape')) for p in problems)}"
          f"/{len(videos)} entries fresh with matching shapes")

    try:
        eval_rows = build_rows(a.vsibench_videos)
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
