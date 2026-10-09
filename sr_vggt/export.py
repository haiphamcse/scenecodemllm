"""Run the frozen VGGT backbone once per clip of the mix(es) and write its tokens to the cache.

    python -m sr_vggt.export <configs/caches/*.env flags> --cache_root <dir> \
        --mix_jsonl <mix> [<mix> ...] --data_root <VSI-590K root> --num_shards 10 --shard_index 0

Frames are each clip's det_es JPEGs, so T == det_es frame count. Resumable: entries that are
already fresh (same parameters) are skipped. One process per GPU, each taking a disjoint slice of
the sorted clip list.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from typing import List

import torch
from transformers import HfArgumentParser

from . import config as cfg
from . import encoders, vggt_cache
from .data import det_es_frames, video_path


@dataclass
class ExportArguments:
    cache_root: str = ""
    mix_jsonl: List[str] = field(default_factory=list)
    data_root: str = ""
    num_shards: int = 1
    shard_index: int = 0
    store_dtype: str = field(default="bf16", metadata={"help": "On-disk dtype: bf16 | fp32."})
    dtype: str = field(default="fp32", metadata={"help": "VGGT compute dtype (V100: fp32)."})
    device: str = "cuda"
    dry_run: bool = False


def distinct_clips(mixes, data_root):
    """Sorted (clip_key, row) pairs, one per distinct video: the same split on every shard."""
    seen = {}
    for mix in mixes:
        for line in open(mix):
            row = json.loads(line)
            seen.setdefault(vggt_cache.clip_key(row, data_root or None), row)
    return sorted(seen.items())


def main() -> int:
    ea, va = HfArgumentParser((ExportArguments, cfg.VggtArguments)).parse_args_into_dataclasses()
    if not 0 <= ea.shard_index < ea.num_shards:
        raise SystemExit(f"--shard_index {ea.shard_index} outside [0, {ea.num_shards})")
    params = va.cache_params()
    clips = distinct_clips(ea.mix_jsonl, ea.data_root)
    mine = clips[ea.shard_index::ea.num_shards]
    todo = [(k, r) for k, r in mine if not vggt_cache.is_fresh(ea.cache_root, k, params)]
    print(f"shard {ea.shard_index}/{ea.num_shards}: {len(clips)} clips total, {len(mine)} here, "
          f"{len(mine) - len(todo)} fresh, {len(todo)} to export into {ea.cache_root}\n  params {params}",
          flush=True)
    if ea.dry_run or not todo:
        return 0

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[ea.dtype]
    store = {"bf16": torch.bfloat16, "fp32": torch.float32}[ea.store_dtype]
    aggregator = encoders.load_aggregator(va.vggt_backbone, va.checkpoint(), ea.device, dtype)
    start, failed, total_bytes = time.time(), 0, 0
    for i, (key, row) in enumerate(todo, 1):
        try:
            frames = det_es_frames(video_path(row, ea.data_root or None))
            video = encoders.frames_to_video(frames, va.vggt_frames, va.vggt_image_resolution, va.vggt_backbone)
            tensors = encoders.encode(aggregator, video, ea.device, dtype, layers=va.vggt_layers)
            out = vggt_cache.save(ea.cache_root, key, {k: v.to(store) for k, v in tensors.items()}, params)
        except Exception as exc:  # one bad clip must not sink the shard
            failed += 1
            print(f"FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        total_bytes += out.stat().st_size
        shapes = {k: tuple(v.shape) for k, v in tensors.items()}
        print(f"  [{i}/{len(todo)}] {key} {shapes} ({time.time() - start:.0f}s)", flush=True)
    print(f"exported={len(todo) - failed} failed={failed} bytes={total_bytes / 1e9:.2f}GB "
          f"elapsed={time.time() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
