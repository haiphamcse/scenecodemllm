"""Run the frozen VGGT-Omega once per clip and write its tokens to disk.

This is the ONLY script in idea_3i_590k_joint_cached that imports VGGT-Omega. Training
and eval read what it writes (see vggt_cache.py) and hard-error on a miss, so this must
run first for every clip a run will touch.

Why cache at all: VGGT is frozen, so its output for a clip is a constant. Recomputing it
every epoch, on every rank, is the largest fixed cost in the joint run -- and at
--vggt_image_resolution 512 (this fork's default, VGGT-Omega's own native size) it is
four times what the 256-res runs paid. Caching turns it into one pass.

Frames are decoded and resized by the SAME collator functions the trainer used to call,
so cached features are bit-for-bit what an uncached run would have computed:
decode_video_frames / load_example_video, then extract_raw_video_tensors.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i_590k_joint_cached/export_vggt_features.py \
      --cache_root /home/ducpham/scratch/Working/dataset/vggt_cache_512 --overfit_num_samples 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

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

    This is load_fn.load_and_preprocess_images(mode="balanced") minus its file-opening
    step: crop the aspect ratio into [0.5, 2.0], resize (BICUBIC) to whichever h x w keeps
    the patch count near ``(image_resolution / patch_size)**2``, and scale to [0, 1]. Its
    top-level function takes paths; the frames are already decoded here, so the two
    shape helpers are called directly.

    The [0, 1] scaling used to live in model_with_vggt._extract_vggt_tokens as a
    ``max() > 1`` guard. It belongs here: ToTensor already produces the range VGGT's
    aggregator expects, and the guard could not tell a genuinely dark [0, 255] clip from
    an already-normalised one.

    Returned as a one-element list because a row is one clip; the batch is the
    concatenation over rows.
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
    # Every frame of a ScanNet clip shares one camera, so the shapes always agree and
    # load_fn's pad-to-common-size branch has nothing to do; torch.stack says so loudly.
    return [torch.stack(tensors)]


DEFAULT_VGGT_CHECKPOINT = (
    os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)


def load_vggt(checkpoint: str, device: str, dtype: torch.dtype):
    """The frozen aggregator, exactly as model_with_vggt.initialize_vggt used to build it.

    enable_alignment=False builds (and then drops) the text_alignment_head so
    load_state_dict accepts the aligned checkpoint's key set; only the aggregator is
    kept, which is the part that produced every feature this project has trained on.
    """
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
    # fp32 on disk: the exporter runs in bf16 to fit VGGT at 512, but the stored copy
    # is widened so the cache is not pinned to the compute dtype of this machine.
    return patch.float().cpu(), camera.float().cpu()


def build_rows(args):
    """The rows to export, as (clip_key, example) pairs, deduplicated by clip.

    VSI-Bench asks ~18 questions per video, and a mix row repeats its clip many times
    over, so the distinct-clip count is what the export actually costs.
    """
    import json

    from data import build_vsibench_eval_dataset

    rows = []
    if args.mix_jsonl:
        rows += [json.loads(line) for line in open(args.mix_jsonl)]

    # Additive, not either/or: a training run reads the mix, and its eval callback and
    # the standalone eval read VSI-Bench. Both raise VggtCacheMiss if not exported.
    if args.include_vsibench or not args.mix_jsonl:
        vsib = list(
            build_vsibench_eval_dataset(
                hf_home=args.hf_home,
                max_samples=None if args.mix_jsonl else args.overfit_num_samples,
                chosen_dataset=args.chosen_dataset,
            )
        )
        # train.py forces every VSI-Bench row to the vqa turn; load_frames needs it too.
        rows += [dict(r, task="vqa") for r in vsib]

    # Every video file under a directory, config-free. build_vsibench_eval_dataset loads
    # the DEBIASED config (274 videos); eval.py --split full reads the FULL config (512
    # videos on disk), so 238 clips were missing and shards died on VggtCacheMiss. A row
    # is keyed on its absolute path (vggt_cache.clip_key), which is exactly what eval.py
    # looks up, so globbing the files is the one source that cannot drift from a config.
    if args.video_glob:
        import glob
        rows += [{"video": v, "task": "vqa"} for v in sorted(glob.glob(args.video_glob, recursive=True))]

    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(vggt_cache.clip_key(row, args.data_root), row)
    # Sorted, so the shard split is the same on every rank and across restarts.
    return sorted(seen.items())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--mix_jsonl", default=None,
                   help="build_mix.py manifest. Omit to export the overfit subset instead.")
    p.add_argument("--overfit_num_samples", type=int, default=20,
                   help="Rows of the VSI-Bench overfit subset (train.py's own default).")
    p.add_argument("--chosen_dataset", default="scannet",
                   help='Substring filter on the VSI-Bench clip path. "" keeps every source.')
    p.add_argument("--video_glob", default=None,
                   help='Also export every video matching this glob, e.g. "$SCRATCH/dataset/vsib/vsibench/*/*.mp4".')
    p.add_argument("--include_vsibench", action="store_true",
                   help="Add the VSI-Bench clips to --mix_jsonl instead of exporting only the mix.")
    # One process per GPU, each taking a disjoint slice. The freshness check still runs
    # inside the slice, so a re-run resumes rather than re-exporting.
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--data_root",
                   default="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K")
    p.add_argument("--hf_home", default="/home/ducpham/scratch/Working/cache")
    p.add_argument("--vggt_checkpoint", default=DEFAULT_VGGT_CHECKPOINT)
    # 512 is VGGT-Omega's native size and this fork's default, unlike the 256 the
    # uncached joint runs used. Features from the two are NOT interchangeable, which is
    # why the resolution is recorded in every cache entry and checked on load.
    p.add_argument("--vggt_image_resolution", type=int, default=VGGT_IMAGE_RESOLUTION)
    p.add_argument("--video_fps", type=float, default=1.0)
    p.add_argument("--video_max_frames", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--store_dtype", choices=("fp32", "bf16"), default="fp32",
                   help="On-disk dtype. bf16 halves the file; the model casts to its compute dtype anyway.")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Compute dtype for the VGGT forward. Storage is always fp32.")
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
            frames = load_frames(row, args.data_root, args.video_fps, args.video_max_frames)
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
