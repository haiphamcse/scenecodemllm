"""Shared helpers for the idea_3i pre-extracted frame cache.

Each video is cached as the *final resized* uint8 video tensor ``(T, C, H, W)``
that ``qwen_vl_utils.fetch_video`` would produce, plus a small JSON sidecar with
the fps/resolution needed to rebuild an equivalent chat message at train time
(path B: the frames are fed back through the existing list branch of
``fetch_video`` as a ``list[PIL.Image]``).

The cache is keyed by the *resolved absolute* video path mirrored under
``cache_root`` (with the leading anchor stripped). This works for both the
relative train-jsonl paths and the absolute VSI-Bench eval paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def cache_paths_for(video_abs: str, cache_root: Path) -> Tuple[Path, Path]:
    """Return ``(npy_path, json_path)`` for a resolved absolute video path."""
    resolved = Path(video_abs).resolve()
    mirrored = cache_root / resolved.relative_to(resolved.anchor)
    npy_path = mirrored.with_suffix(mirrored.suffix + ".npy")
    json_path = mirrored.with_suffix(mirrored.suffix + ".json")
    return npy_path, json_path


def save_cached_frames(
    npy_path: Path,
    json_path: Path,
    frames_uint8: np.ndarray,
    meta: Dict[str, Any],
) -> None:
    """Write ``(T, C, H, W)`` uint8 frames + JSON sidecar."""
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, frames_uint8)
    json_path.write_text(json.dumps(meta))


def load_cached_frames(
    npy_path: Path,
    json_path: Path,
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """Load cached frames as a ``list[PIL.Image]`` (RGB) plus the sidecar dict."""
    frames = np.load(npy_path)  # (T, C, H, W) uint8
    meta = json.loads(json_path.read_text())
    pil_frames = [Image.fromarray(np.transpose(f, (1, 2, 0))) for f in frames]
    return pil_frames, meta
