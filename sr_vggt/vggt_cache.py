"""The VGGT feature cache: one safetensors file per training clip.

export.py writes it once; training reads it. A miss or a parameter mismatch raises; nothing
ever falls back to running VGGT during training.

Layout mirrors the clip's resolved absolute video path:

    /data/VSI-590K/scannet/scene0277_02.mp4
      -> <cache_root>/data/VSI-590K/scannet/scene0277_02.safetensors

Tensors per clip (fp32 or bf16, as exported), see encoders.encode:

    last-layer caches   patch [T, Np, d], camera [T, S, d]   (S = camera + register tokens)
    layered caches      patch_0, patch_1, ... [T, Np, d]     (one per entry of vggt_layers)

Every entry carries its export parameters (encoders.cache_params) in the safetensors metadata;
``load`` compares all of them, so a cache exported under another recipe cannot be consumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


class VggtCacheMiss(FileNotFoundError):
    """No cache entry for this clip. Run scripts/export_vggt.sh first."""


def clip_key(example: Dict[str, Any], data_root=None) -> str:
    """Absolute, resolved video path of a row (relative paths resolved against ``data_root``)."""
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


def save(cache_root, key: str, tensors: Dict[str, torch.Tensor], params: Dict[str, Any]) -> Path:
    """Write one clip's tensors. Atomic: written beside the target, then renamed in."""
    out = cache_path(cache_root, key)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".safetensors.tmp")
    meta = {"clip_key": key, **params}
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(tmp),
              metadata={"vggt_cache": json.dumps(meta)})
    tmp.rename(out)
    return out


def read_meta(path: Path) -> Dict[str, Any]:
    with safe_open(str(path), framework="pt") as f:
        raw = (f.metadata() or {}).get("vggt_cache")
    if not raw:
        raise ValueError(f"{path} carries no vggt_cache metadata; re-export it.")
    return json.loads(raw)


def _mismatch(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (meta.get(k), v) for k, v in params.items() if meta.get(k) != v}


def is_fresh(cache_root, key: str, params: Dict[str, Any]) -> bool:
    """True if an entry exists AND was exported under these parameters."""
    path = cache_path(cache_root, key)
    if not path.exists():
        return False
    try:
        return not _mismatch(read_meta(path), params)
    except (OSError, ValueError):
        return False


def load(cache_root, key: str, params: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """The clip's tensors, or raise. Never runs VGGT."""
    path = cache_path(cache_root, key)
    if not path.exists():
        raise VggtCacheMiss(f"No VGGT cache for {key}\n  expected: {path}\n"
                            f"  Run scripts/export_vggt.sh for this cache first.")
    mismatch = _mismatch(read_meta(path), params)
    if mismatch:
        raise ValueError(f"VGGT cache {path} was exported under different parameters "
                         f"(cached, requested): {mismatch}. Re-export or fix the config.")
    return load_file(str(path))


def shapes(path: Path) -> Dict[str, tuple]:
    """Tensor shapes of one entry, read from the header only."""
    with safe_open(str(path), framework="pt") as f:
        return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}


if __name__ == "__main__":
    import tempfile

    params = {"vggt_image_resolution": 256, "video_fps": 1.0, "video_max_frames": 64,
              "vggt_checkpoint": "/models/x.pt", "frames": "det_es"}
    row = {"video": "/data/vsi/scannet/scene0277_02.mp4"}
    assert clip_key(row) == "/data/vsi/scannet/scene0277_02.mp4"
    assert clip_key({"video": "scannet/x.mp4"}, data_root="/data") == "/data/scannet/x.mp4"
    with tempfile.TemporaryDirectory() as d:
        key = clip_key(row)
        assert cache_path(d, key) == Path(d) / "data/vsi/scannet/scene0277_02.safetensors"
        t = {"patch": torch.randn(2, 4, 8), "camera": torch.randn(2, 17, 8)}
        assert not is_fresh(d, key, params)
        save(d, key, t, params)
        assert is_fresh(d, key, params)
        got = load(d, key, params)
        assert all(torch.equal(got[k], t[k]) for k in t)
        assert shapes(cache_path(d, key)) == {"patch": (2, 4, 8), "camera": (2, 17, 8)}
        for bad in ({**params, "vggt_image_resolution": 512}, {**params, "vggt_patch_size": 14}):
            assert not is_fresh(d, key, bad)
            try:
                load(d, key, bad)
            except ValueError as exc:
                assert "different parameters" in str(exc)
            else:
                raise AssertionError("a parameter mismatch must raise")
        try:
            load(d, "/data/missing.mp4", params)
        except VggtCacheMiss:
            pass
        else:
            raise AssertionError("a missing entry must raise VggtCacheMiss")
    print("vggt_cache selftest ok")
