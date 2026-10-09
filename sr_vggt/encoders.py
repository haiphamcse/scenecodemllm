"""Frozen VGGT backbones: checkpoint lookup, frame preprocessing, token extraction.

Shared by export.py (writes the training cache from det_es JPEGs) and eval.py (runs the same
encoder live on decoded VSI-Bench video), so training and eval see identically computed tokens.

Backbones
  omega   facebook/VGGT-Omega aggregator (patch 16, 1 camera + 16 register tokens per frame)
  vggt1b  facebook/VGGT-1B aggregator, vendored in ablations/vggt1b/vggt (patch 14, 1 + 4)

Frame modes
  balanced  VGGT-Omega's balanced resize to ~(resolution/patch)^2 patches (Perceiver, VLM-3R)
  native    the det_es / Qwen pixels untouched, so VGGT's patch grid equals Qwen's (SpatialStack)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms as TF

PATCH_SIZE = {"omega": 16, "vggt1b": 14}
_TO_TENSOR = TF.ToTensor()
_VGGT1B_DIR = Path(__file__).resolve().parent / "ablations" / "vggt1b"


def resolve_checkpoint(repo: str, filename: str, revision: Optional[str] = None) -> str:
    """Absolute path of a weight file inside HF_HOME (download_weights.sh fetched it).

    This exact string is stamped into every cache entry (``vggt_checkpoint``), so a cache is
    only reusable under the same HF_HOME.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, revision=revision)


def cache_params(backbone: str, checkpoint: str, frames: str, video_fps: float,
                 video_max_frames: int, resolution: Optional[int] = None,
                 layers: Optional[List[int]] = None) -> Dict:
    """Everything a cached tensor depends on. The key sets match the caches the original
    experiments trained on, so those remain loadable."""
    params = {"video_fps": float(video_fps), "video_max_frames": int(video_max_frames),
              "vggt_checkpoint": str(checkpoint)}
    if frames == "native":
        params["frames"] = "det_es_native"
        params["vggt_layers"] = [int(l) for l in layers]
    else:
        params["frames"] = "det_es"
        params["vggt_image_resolution"] = int(resolution)
    if backbone == "vggt1b":
        params["vggt_patch_size"] = PATCH_SIZE["vggt1b"]
    return params


def load_aggregator(backbone: str, checkpoint: str, device, dtype: torch.dtype):
    """The frozen aggregator of the given backbone, in eval mode, no grads."""
    if backbone == "omega":
        from vggt_omega.models import VGGTOmega

        state = torch.load(checkpoint, map_location="cpu")
        # The text-aligned checkpoint carries text_alignment_head weights: build the head so
        # the strict load accepts them. Only the aggregator is kept either way.
        vggt = VGGTOmega(enable_alignment=any(k.startswith("text_alignment_head") for k in state))
        vggt.load_state_dict(state)
        aggregator = vggt.aggregator
    elif backbone == "vggt1b":
        from safetensors import safe_open

        if str(_VGGT1B_DIR) not in sys.path:
            sys.path.insert(0, str(_VGGT1B_DIR))
        from vggt.models.aggregator import Aggregator  # vendored, see ablations/vggt1b/vggt

        with safe_open(checkpoint, framework="pt") as f:
            state = {k[len("aggregator."):]: f.get_tensor(k)
                     for k in f.keys() if k.startswith("aggregator.")}
        aggregator = Aggregator()  # VGGT-1B config: img 518, patch 14, embed 1024
        aggregator.load_state_dict(state)
        assert aggregator.patch_size == PATCH_SIZE["vggt1b"], aggregator.patch_size
        assert aggregator.patch_start_idx == 5, aggregator.patch_start_idx
    else:
        raise ValueError(f"unknown VGGT backbone {backbone!r}")
    aggregator = aggregator.to(device=device, dtype=dtype).eval()
    for p in aggregator.parameters():
        p.requires_grad = False
    return aggregator


def preprocess(frames: List[Image.Image], resolution: int, patch_size: int) -> torch.Tensor:
    """Balanced mode: frames -> ``[T, C, H, W]`` in [0, 1], the way VGGT-Omega asks.

    load_fn.load_and_preprocess_images(mode="balanced") minus the file opening: crop the aspect
    ratio into [0.5, 2.0], BICUBIC resize to whichever h x w keeps the patch count near
    ``(resolution / patch_size)**2``, ToTensor.
    """
    from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio

    tensors = []
    for frame in frames:
        image = _crop_to_supported_aspect_ratio(frame)
        width, height = image.size
        target_h, target_w = _balanced_target_shape(height / max(width, 1), resolution, patch_size)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensors.append(_TO_TENSOR(image))
    return torch.stack(tensors)  # one size per clip; stack fails loudly otherwise


@torch.no_grad()
def encode(aggregator, video: torch.Tensor, device, dtype: torch.dtype,
           layers: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
    """``[T, C, H, W]`` in [0, 1] -> cache tensors, fp32 on the CPU.

    layers=None: last layer, ``patch [T, Np, d]`` + ``camera [T, S, d]`` (camera + registers).
    layers=[...]: patch tokens of each listed aggregator layer, ``patch_<i>``.
    """
    tokens_list, patch_start = aggregator(video.to(device=device, dtype=dtype).unsqueeze(0))
    if layers is None:
        final = tokens_list[-1][0]  # [T, S + Np, d]
        return {"patch": final[:, patch_start:, :].float().cpu(),
                "camera": final[:, :patch_start, :].float().cpu()}
    return {f"patch_{i}": tokens_list[l][0][:, patch_start:, :].float().cpu()
            for i, l in enumerate(layers)}


def frames_to_video(frames: List[Image.Image], mode: str, resolution: Optional[int],
                    backbone: str) -> torch.Tensor:
    """det_es frames -> the VGGT input tensor of a cache's frame mode."""
    if mode == "native":
        return torch.stack([_TO_TENSOR(f) for f in frames])
    return preprocess(frames, resolution, PATCH_SIZE[backbone])
