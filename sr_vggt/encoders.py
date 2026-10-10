"""Frozen VGGT backbones and the frames they see. Training and eval share every function here.

Backbones (vendored in third_party/)
  omega   facebook/VGGT-Omega aggregator (patch 16, 1 camera + 16 register tokens per frame)
  vggt1b  facebook/VGGT-1B aggregator (patch 14, 1 camera + 4 register tokens per frame)

VGGT input (--vggt_frames)
  balanced  VGGT-Omega's balanced resize of the decoded frames to ~(resolution/patch)^2 patches
            (Perceiver, VLM-3R, VGGT-1B)
  native    Qwen's own processed pixels, so VGGT's patch grid equals Qwen's video grid (SpatialStack)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

PATCH_SIZE = {"omega": 16, "vggt1b": 14}
_TO_TENSOR = TF.ToTensor()


def resolve_checkpoint(repo: str, filename: str, revision: Optional[str] = None) -> str:
    """Absolute path of a weight file inside HF_HOME (download_weights.sh fetched it)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo, filename=filename, revision=revision)


def load_aggregator(backbone: str, checkpoint: str):
    """The backbone's aggregator with the checkpoint's ``aggregator.*`` weights (strict), on CPU,
    eval mode, frozen. The checkpoints' heads are never built."""
    if backbone == "omega":
        from vggt_omega.models.aggregator import Aggregator

        state = torch.load(checkpoint, map_location="cpu")
        aggregator = Aggregator(patch_size=16, embed_dim=1024)  # = VGGTOmega()'s aggregator
    elif backbone == "vggt1b":
        from safetensors.torch import load_file
        from vggt.models.aggregator import Aggregator

        state = load_file(checkpoint)
        aggregator = Aggregator()  # VGGT-1B config: img 518, patch 14, embed 1024
    else:
        raise ValueError(f"unknown VGGT backbone {backbone!r}")
    aggregator.load_state_dict({k[len("aggregator."):]: v for k, v in state.items()
                                if k.startswith("aggregator.")})
    for p in aggregator.parameters():
        p.requires_grad = False
    return aggregator.eval()


def decode_video_frames(path: str, video_fps: float, video_max_frames: int) -> List[Image.Image]:
    """Video -> PIL frames at native resolution: ``video_fps`` sampling, uniform over the clip,
    capped at ``video_max_frames`` (qwen_vl_utils.fetch_video's rule). The one sampling rule for
    training and eval."""
    import decord

    reader = decord.VideoReader(path, num_threads=3)
    total = len(reader)
    native_fps = reader.get_avg_fps() or 30.0
    n_frames = max(1, min(int(round(total / native_fps * video_fps)), video_max_frames, total))
    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    return [Image.fromarray(f) for f in reader.get_batch(idx).asnumpy()]


def video_meta(path: str, video_fps: float, video_max_frames: int):
    """(frame count, (W, H)) decode_video_frames would return, decoding one frame only."""
    import decord

    reader = decord.VideoReader(path, num_threads=1)
    total = len(reader)
    n_frames = max(1, min(int(round(total / (reader.get_avg_fps() or 30.0) * video_fps)), video_max_frames, total))
    h, w = reader[0].shape[:2]
    return n_frames, (w, h)


def preprocess(frames: List[Image.Image], resolution: int, patch_size: int) -> torch.Tensor:
    """Balanced mode: frames -> ``[T, C, H, W]`` in [0, 1] (load_fn.load_and_preprocess_images
    minus the file opening): crop the aspect into [0.5, 2.0], BICUBIC resize to whichever h x w
    keeps the patch count near ``(resolution / patch_size)**2``."""
    from vggt_omega.utils.load_fn import _balanced_target_shape, _crop_to_supported_aspect_ratio

    tensors = []
    for frame in frames:
        image = _crop_to_supported_aspect_ratio(frame)
        width, height = image.size
        target_h, target_w = _balanced_target_shape(height / max(width, 1), resolution, patch_size)
        tensors.append(_TO_TENSOR(image.resize((target_w, target_h), Image.Resampling.BICUBIC)))
    return torch.stack(tensors)


def vggt_input(vggt_frames: str, frames, qwen_video: torch.Tensor, resolution: int, backbone: str) -> torch.Tensor:
    """The ``[T, C, H, W]`` tensor in [0, 1] VGGT encodes for one clip."""
    if vggt_frames == "native":
        video = qwen_video.float()
        return video / 255.0 if video.max() > 1.0 else video
    return preprocess(frames, resolution, PATCH_SIZE[backbone])
