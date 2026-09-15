"""The VGGT feature cache: paths, format, and the errors that guard it.

This fork removes VGGT-Omega from training and eval entirely. The frozen encoder runs
ONCE, offline, in export_vggt_features.py; everything downstream reads its output from
disk. So this module is the contract between the two halves, and it is deliberately
strict: a miss or a parameter mismatch raises, it never falls back to running VGGT.
A silent fallback is exactly what would make a cached run quietly cost the same as an
uncached one, or -- worse -- train on features from the wrong resolution.

Layout mirrors the source clip's own path, so a cache entry is findable by eye:

    /home/.../vsib/vsibench/scannet/scene0277_02.mp4
      -> <cache_root>/home/.../vsib/vsibench/scannet/scene0277_02.safetensors

det rows have no single file -- their clip IS a directory of JPEGs -- so they key on
that directory with the same suffix swap:

    /home/.../scannet/scene0000_00/000.jpg
      -> <cache_root>/home/.../scannet/scene0000_00.safetensors

Stored per clip (fp32, as exported):

    patch  [T, Np, vggt_dim]   VGGT patch tokens, one group per frame
    camera [T, 17, vggt_dim]   camera token + 16 register tokens

Both are the LAST aggregator layer, which is the only one model_with_vggt ever read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file, save_file

# Every parameter a cached tensor depends on. Checked on load; a mismatch is an error,
# not a warning, because the shapes still line up and the run would look healthy.
CACHE_PARAM_KEYS = (
    "vggt_image_resolution",
    "video_fps",
    "video_max_frames",
    "vggt_checkpoint",
)


class VggtCacheMiss(FileNotFoundError):
    """No cache entry for this clip. Never recoverable at train time -- export first."""


def clip_key(example: Dict[str, Any], data_root=None) -> str:
    """The absolute path identifying a clip, whichever corpus the row came from.

    Resolves relative video paths against ``data_root`` with the same rule
    collator.load_frames uses, so the exporter and the trainer cannot disagree about
    which file a row means.
    """
    images = example.get("images")
    if images:
        # One entry per frame WINDOW, not per scene: the det json holds ~43 different
        # 32-frame windows per ScanNet scene, and keying by scene made every row but the
        # first train on another window's tokens (found 2026-09-11).
        digest = hashlib.sha1("\n".join(images).encode()).hexdigest()[:16]
        return str(Path(images[0]).parent.resolve() / f"win_{digest}")
    video = example.get("video")
    if not video:
        raise ValueError(f"row has neither `video` nor `images`: {sorted(example)}")
    path = Path(video)
    if data_root is not None and not path.is_absolute():
        path = Path(data_root) / path
    return str(path.resolve())


def cache_path(cache_root, key: str) -> Path:
    """``<cache_root>/<key without its leading />.safetensors``."""
    path = Path(key)
    return Path(cache_root) / path.relative_to(path.anchor).with_suffix(".safetensors")


def save(cache_root, key: str, patch: torch.Tensor, camera: torch.Tensor,
         params: Dict[str, Any]) -> Path:
    """Write one clip's tokens. Atomic: build beside the target, then rename in."""
    out = cache_path(cache_root, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".safetensors.tmp")
    meta = {"clip_key": key, **{k: params[k] for k in CACHE_PARAM_KEYS}}
    save_file(
        {"patch": patch.contiguous(), "camera": camera.contiguous()},
        str(tmp),
        metadata={"vggt_cache": json.dumps(meta)},
    )
    tmp.rename(out)
    return out


def _read_meta(path: Path) -> Dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        raw = (f.metadata() or {}).get("vggt_cache")
    if not raw:
        raise ValueError(f"{path} carries no vggt_cache metadata; re-export it.")
    return json.loads(raw)


def is_fresh(cache_root, key: str, params: Dict[str, Any]) -> bool:
    """True if an entry exists AND was exported under these parameters."""
    path = cache_path(cache_root, key)
    if not path.exists():
        return False
    try:
        meta = _read_meta(path)
    except (OSError, ValueError):
        return False
    return all(meta.get(k) == params[k] for k in CACHE_PARAM_KEYS)


def load(cache_root, key: str, params: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(patch [T, Np, d], camera [T, 17, d])`` or raise. Never runs VGGT."""
    path = cache_path(cache_root, key)
    if not path.exists():
        raise VggtCacheMiss(
            f"No VGGT cache for {key}\n  expected: {path}\n"
            f"  This fork runs only on cached features. Export them first:\n"
            f"    python idea_3i_590k_joint_cached/export_vggt_features.py "
            f"--cache_root {cache_root} ..."
        )
    meta = _read_meta(path)
    mismatch = {k: (meta.get(k), params[k]) for k in CACHE_PARAM_KEYS if meta.get(k) != params[k]}
    if mismatch:
        raise ValueError(
            f"VGGT cache {path} was exported under different parameters "
            f"(cached, requested): {mismatch}. Re-export or fix the training flags."
        )
    tensors = load_file(str(path))
    return tensors["patch"], tensors["camera"]


def params_from(vggt_image_resolution: int, video_fps: float, video_max_frames: int,
                vggt_checkpoint: str) -> Dict[str, Any]:
    """The dict every function above expects, built once by the caller."""
    return {
        "vggt_image_resolution": int(vggt_image_resolution),
        "video_fps": float(video_fps),
        "video_max_frames": int(video_max_frames),
        "vggt_checkpoint": str(vggt_checkpoint),
    }


if __name__ == "__main__":
    # Self-check: the path mapping for both corpora, and a save/load round trip that
    # must reject a parameter change.
    import tempfile

    params = params_from(512, 1.0, 32, "/models/vggt_omega_1b_512.pt")

    vqa = {"video": "/data/vsib/vsibench/scannet/scene0277_02.mp4"}
    det = {"images": ["/data/scannet/scene0000_00/000.jpg",
                      "/data/scannet/scene0000_00/001.jpg"]}
    rel = {"video": "scannet/xxx.mp4"}

    assert clip_key(vqa) == "/data/vsib/vsibench/scannet/scene0277_02.mp4"
    win = hashlib.sha1("\n".join(det["images"]).encode()).hexdigest()[:16]
    assert clip_key(det) == f"/data/scannet/scene0000_00/win_{win}"
    assert clip_key(dict(det, images=det["images"][::-1])) != clip_key(det)
    assert clip_key(rel, data_root="/data") == "/data/scannet/xxx.mp4"

    with tempfile.TemporaryDirectory() as d:
        assert cache_path(d, clip_key(vqa)) == Path(d) / (
            "data/vsib/vsibench/scannet/scene0277_02.safetensors")
        assert cache_path(d, clip_key(det)) == Path(d) / f"data/scannet/scene0000_00/win_{win}.safetensors"

        patch = torch.randn(2, 4, 8)
        camera = torch.randn(2, 17, 8)
        assert not is_fresh(d, clip_key(vqa), params)
        save(d, clip_key(vqa), patch, camera, params)
        assert is_fresh(d, clip_key(vqa), params)

        got_patch, got_camera = load(d, clip_key(vqa), params)
        assert torch.equal(got_patch, patch) and torch.equal(got_camera, camera)
        assert got_patch.dtype == torch.float32

        try:
            load(d, clip_key(vqa), params_from(256, 1.0, 32, params["vggt_checkpoint"]))
        except ValueError as exc:
            assert "different parameters" in str(exc)
        else:
            raise AssertionError("resolution mismatch must raise")

        try:
            load(d, clip_key(det), params)
        except VggtCacheMiss:
            pass
        else:
            raise AssertionError("a missing entry must raise VggtCacheMiss")

    print("vggt_cache selftest ok")
