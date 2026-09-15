"""The VGGT feature cache: paths, format, and the errors that guard it.

export_vggt_features.py writes it once; the trainer reads it. A miss or a parameter mismatch
raises, never falls back to running VGGT.

Layout mirrors the source clip's path:

    /.../VSI-590K/scannet/scene0277_02.mp4
      -> <cache_root>/.../VSI-590K/scannet/scene0277_02.safetensors

Stored per clip (fp32 or bf16, as exported), last aggregator layer:

    patch  [T, Np, vggt_dim]   VGGT patch tokens, one group per frame
    camera [T, 17, vggt_dim]   camera token + 16 register tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors.torch import load_file, save_file

# Every parameter a cached tensor depends on. A mismatch is an error: shapes still line up.
CACHE_PARAM_KEYS = (
    "vggt_image_resolution",
    "video_fps",
    "video_max_frames",
    "vggt_checkpoint",
    "frames",   # "det_es": features of the det_es JPEGs, never of decoded video
)


class VggtCacheMiss(FileNotFoundError):
    """No cache entry for this clip. Export first."""


def clip_key(example: Dict[str, Any], data_root=None) -> str:
    """Absolute video path of a row (relative paths resolved against ``data_root``)."""
    video = example.get("video")
    if not video:
        raise ValueError(f"row has no `video`: {sorted(example)}")
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
            f"  Export it first:\n"
            f"    python idea_3i_180k_es_64f_lat512_clean/export_vggt_features.py "
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
                vggt_checkpoint: str, frames: str = "det_es") -> Dict[str, Any]:
    """The dict every function above expects, built once by the caller."""
    return {
        "vggt_image_resolution": int(vggt_image_resolution),
        "video_fps": float(video_fps),
        "video_max_frames": int(video_max_frames),
        "vggt_checkpoint": str(vggt_checkpoint),
        "frames": str(frames),
    }


if __name__ == "__main__":
    # Self-check: path mapping, and a save/load round trip that must reject a parameter change.
    import tempfile

    params = params_from(512, 1.0, 32, "/models/vggt_omega_1b_512.pt")
    vqa = {"video": "/data/vsib/vsibench/scannet/scene0277_02.mp4"}

    assert clip_key(vqa) == "/data/vsib/vsibench/scannet/scene0277_02.mp4"
    assert clip_key({"video": "scannet/xxx.mp4"}, data_root="/data") == "/data/scannet/xxx.mp4"

    with tempfile.TemporaryDirectory() as d:
        assert cache_path(d, clip_key(vqa)) == Path(d) / (
            "data/vsib/vsibench/scannet/scene0277_02.safetensors")

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
            load(d, "/data/missing.mp4", params)
        except VggtCacheMiss:
            pass
        else:
            raise AssertionError("a missing entry must raise VggtCacheMiss")

    print("vggt_cache selftest ok")
