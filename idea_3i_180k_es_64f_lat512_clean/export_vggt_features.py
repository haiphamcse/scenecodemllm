"""Run the frozen VGGT-Omega once per clip of the mix and write its tokens to the cache.

The only training-side script that imports VGGT-Omega (eval.py borrows
extract_raw_video_tensors). Frames come from collator.load_frames (the clip's det_es JPEGs), so
T == det_es frame count and entries are stamped frames=det_es (vggt_cache.params_from).
Resumable: fresh entries are skipped.

  python idea_3i_180k_es_64f_lat512_clean/export_vggt_features.py --cache_root <cache> \
      --mix_jsonl <vqa_train_190k_es_clean.jsonl> --vggt_image_resolution 256 --video_max_frames 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)

import vggt_cache  # noqa: E402
from collator import VGGT_IMAGE_RESOLUTION, load_frames  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms as TF  # noqa: E402
from vggt_omega.utils.load_fn import (  # noqa: E402
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
)

VGGT_PATCH_SIZE = 16
_TO_TENSOR = TF.ToTensor()


def extract_raw_video_tensors(
    frames,
    image_resolution: int = VGGT_IMAGE_RESOLUTION,
    patch_size: int = VGGT_PATCH_SIZE,
):
    """Frames -> ``[tensor [T, C, H, W]]`` in [0, 1], preprocessed the way VGGT-Omega asks.

    load_fn.load_and_preprocess_images(mode="balanced") minus its file-opening step: crop the
    aspect ratio into [0.5, 2.0], resize (BICUBIC) to whichever h x w keeps the patch count
    near ``(image_resolution / patch_size)**2``, ToTensor to [0, 1]. One-element list: one clip.
    """
    tensors = []
    for frame in frames:
        image = _crop_to_supported_aspect_ratio(frame)
        width, height = image.size
        target_h, target_w = _balanced_target_shape(
            height / max(width, 1), image_resolution, patch_size
        )
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensors.append(_TO_TENSOR(image))
    # All frames of a clip share one size; torch.stack fails loudly if not.
    return [torch.stack(tensors)]


DEFAULT_VGGT_CHECKPOINT = (
    os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)


def load_vggt(checkpoint: str, device: str, dtype: torch.dtype):
    """The frozen aggregator, built as model_with_vggt_live.initialize_vggt builds it."""
    from vggt_omega.models import VGGTOmega

    vggt = VGGTOmega(enable_alignment=False)
    vggt.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    aggregator = vggt.aggregator.to(device=device, dtype=dtype).eval()
    for param in aggregator.parameters():
        param.requires_grad = False
    return aggregator


@torch.no_grad()
def encode(aggregator, frames, resolution: int, device: str, dtype: torch.dtype):
    """Frames -> ``(patch [T, Np, d], camera [T, 17, d])`` in fp32, ready to store."""
    video = extract_raw_video_tensors(frames, resolution)[0]  # [T, C, H, W] in [0, 1]
    video = video.to(device=device, dtype=dtype).unsqueeze(0)
    tokens_list, patch_token_start = aggregator(video)
    final = tokens_list[-1][0]  # [T, 17 + Np, d]; batch is always 1 clip here
    camera = final[:, :patch_token_start, :]
    patch = final[:, patch_token_start:, :]
    return patch.float().cpu(), camera.float().cpu()


def build_rows(args):
    """The mix's distinct clips, as sorted (clip_key, example) pairs."""
    rows = [json.loads(line) for line in open(args.mix_jsonl)]
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(vggt_cache.clip_key(row, args.data_root), row)
    # Sorted, so the shard split is the same on every rank and across restarts.
    return sorted(seen.items())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--mix_jsonl", required=True, help="Training mix (vqa_train_190k_es_clean.jsonl).")
    # One process per GPU, each taking a disjoint slice. The freshness check still runs
    # inside the slice, so a re-run resumes rather than re-exporting.
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--data_root",
                   default="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K")
    p.add_argument("--hf_home", default="/home/ducpham/scratch/Working/cache")
    p.add_argument("--vggt_checkpoint", default=DEFAULT_VGGT_CHECKPOINT)
    p.add_argument("--vggt_image_resolution", type=int, default=VGGT_IMAGE_RESOLUTION)
    p.add_argument("--video_fps", type=float, default=1.0)
    p.add_argument("--video_max_frames", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--store_dtype", choices=("fp32", "bf16"), default="fp32",
                   help="On-disk dtype. bf16 halves the file; the model casts to its compute dtype anyway.")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Compute dtype for the VGGT forward (storage: --store_dtype).")
    p.add_argument("--overwrite", action="store_true", help="Re-export entries that are fresh.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    params = vggt_cache.params_from(
        args.vggt_image_resolution, args.video_fps, args.video_max_frames,
        args.vggt_checkpoint,
    )

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit(f"--shard_index {args.shard_index} outside [0, {args.num_shards})")

    all_clips = build_rows(args)
    clips = all_clips[args.shard_index::args.num_shards]
    todo = [(k, r) for k, r in clips
            if args.overwrite or not vggt_cache.is_fresh(args.cache_root, k, params)]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(all_clips)} distinct clips total, "
          f"{len(clips)} in this shard, {len(clips) - len(todo)} fresh, {len(todo)} to export "
          f"at resolution {args.vggt_image_resolution} into {args.cache_root}", flush=True)
    if args.dry_run:
        for k, _ in todo[:5]:
            print(f"  {k} -> {vggt_cache.cache_path(args.cache_root, k)}")
        return 0
    if not todo:
        return 0

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    aggregator = load_vggt(args.vggt_checkpoint, args.device, dtype)

    start = time.time()
    total_bytes = 0
    failed = 0
    for i, (key, row) in enumerate(todo, 1):
        try:
            frames = load_frames(row, args.data_root)
            patch, camera = encode(aggregator, frames, args.vggt_image_resolution,
                                   args.device, dtype)
            store = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.store_dtype]
            out = vggt_cache.save(args.cache_root, key, patch.to(store), camera.to(store), params)
        except Exception as exc:  # one bad clip must not sink the export
            failed += 1
            print(f"FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
            continue
        total_bytes += out.stat().st_size
        print(f"  [{i}/{len(todo)}] {key} patch{tuple(patch.shape)} camera{tuple(camera.shape)} "
              f"{out.stat().st_size / 1e6:.0f}MB ({time.time() - start:.0f}s)", flush=True)

    print(f"exported={len(todo) - failed} failed={failed} "
          f"bytes={total_bytes / 1e9:.2f}GB elapsed={time.time() - start:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
