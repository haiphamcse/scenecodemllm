"""Pre-extract VSI-590K / VSI-Bench videos into cached resized uint8 frames.

Video decode + resize is the dominant CPU cost in the idea_3i dataloader and is
repeated every epoch. This script runs the real ``fetch_video`` once per video
(decode + the exact resize tail, identical to training), stores the final frames
as uint8 ``.npy`` + a JSON sidecar, so the collator can skip decoding at train
time (see idea_3i/collator.py, path B).

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i/pre_extract_videos.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from qwen_vl_utils import fetch_video

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
for _p in (_COMMON, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from argument import DEFAULT_DATA_ROOT, DEFAULT_HF_HOME, DEFAULT_JSONL
from data import build_vsibench_eval_dataset, filter_train_records
from typing import Sequence

from frame_cache import cache_paths_for, save_cached_frames

# Must match the training config (collator + data args) so the cached frames are
# exactly what training would otherwise decode on the fly.
VIDEO_FPS = 1.0
VIDEO_MAX_FRAMES = 32
IMAGE_PATCH_SIZE = 16


def resolve_video_path(data_root: Path, video_field: str) -> str:
    if os.path.isabs(video_field):
        return video_field
    return str(data_root / video_field)


def collect_video_paths(
    data_root: Path,
    jsonl_path: Path,
    hf_home: str | None,
    sources: Sequence[str] | None = None,
) -> list[str]:
    """Resolved absolute paths of every train + eval video.

    ``sources=None`` keeps the legacy scannetppv2-only train filter.
    """
    paths: set[str] = set()

    for record in filter_train_records(jsonl_path, sources):
        video = record.get("video")
        if video:
            paths.add(resolve_video_path(data_root, video))

    eval_dataset = build_vsibench_eval_dataset(hf_home=hf_home)
    for record in eval_dataset:
        paths.add(resolve_video_path(data_root, record["video"]))

    return sorted(paths)


def extract_one(
    video_abs: str,
    cache_root: Path,
    video_fps: float = VIDEO_FPS,
    video_max_frames: int = VIDEO_MAX_FRAMES,
) -> str:
    npy_path, json_path = cache_paths_for(video_abs, cache_root)
    if npy_path.exists() and json_path.exists():
        return "skip"

    ele = {"video": video_abs, "fps": video_fps, "max_frames": video_max_frames}
    (video, metadata), _sample_fps = fetch_video(
        ele,
        image_patch_size=IMAGE_PATCH_SIZE,
        return_video_sample_fps=True,
        return_video_metadata=True,
    )
    # `video` is the final resized float tensor (T, C, H, W) in [0, 255].
    frames = video.clamp(0, 255).round().to(torch.uint8).cpu().numpy()
    num_frames, _, height, width = frames.shape
    # Store the *real* decode metadata so the collator can reproduce the exact
    # per-frame timestamps the live-decode path (and eval) would produce.
    meta = {
        "fps": float(metadata["fps"]),
        "frames_indices": [int(x) for x in metadata["frames_indices"]],
        "total_num_frames": int(metadata["total_num_frames"]),
        "resized_height": int(height),
        "resized_width": int(width),
        "num_frames": int(num_frames),
    }
    save_cached_frames(npy_path, json_path, frames, meta)
    return "done"


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("decord").disabled = True

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--hf_home", type=str, default=DEFAULT_HF_HOME)
    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(Path(DEFAULT_DATA_ROOT).parent / "vsi_590k_frame_cache"),
        help="Where cached frames are written (sibling of data_root by default).",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=VIDEO_FPS,
        help="Sampling fps; must match the training/collator config.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=VIDEO_MAX_FRAMES,
        help="Max frames per video; must match the training/collator config.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated VSI-590K source dirs (e.g. scannet). None = legacy scannetppv2 filter.",
    )
    parser.add_argument(
        "--video_list",
        type=str,
        default=None,
        help="Read video paths from this file instead of scanning the jsonl (which costs ~5 min).",
    )
    parser.add_argument(
        "--dump_list",
        type=str,
        default=None,
        help="Write the resolved video paths to this file and exit (feed it to --video_list).",
    )
    parser.add_argument("--shard", type=int, default=0, help="This shard index, 0-based.")
    parser.add_argument("--num_shards", type=int, default=1, help="Total shards; each takes paths[shard::num_shards].")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)

    if args.video_list:
        video_paths = [ln.strip() for ln in Path(args.video_list).read_text().splitlines() if ln.strip()]
    else:
        sources = tuple(args.sources.split(",")) if args.sources else None
        video_paths = collect_video_paths(data_root, Path(args.jsonl_path), args.hf_home, sources)

    if args.dump_list:
        Path(args.dump_list).write_text("\n".join(video_paths) + "\n")
        logging.info("Wrote %d video paths to %s", len(video_paths), args.dump_list)
        return

    if args.num_shards > 1:
        video_paths = video_paths[args.shard :: args.num_shards]
    logging.info(
        "Videos to extract: %d (shard %d/%d) -> %s (fps=%s max_frames=%d)",
        len(video_paths), args.shard, args.num_shards, cache_root, args.video_fps, args.max_frames,
    )

    done = skipped = failed = 0
    for i, video_abs in enumerate(video_paths, 1):
        try:
            status = extract_one(video_abs, cache_root, args.video_fps, args.max_frames)
            done += status == "done"
            skipped += status == "skip"
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failed += 1
            logging.warning("FAILED %s: %s", video_abs, exc)
        if i % 50 == 0 or i == len(video_paths):
            logging.info("[%d/%d] done=%d skip=%d fail=%d", i, len(video_paths), done, skipped, failed)


if __name__ == "__main__":
    main()
