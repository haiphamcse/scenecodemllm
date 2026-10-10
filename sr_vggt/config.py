"""The flags that define an experiment's VGGT features and fusion, shared by export, train, eval
and verify (configs/*.env pass them), plus the per-architecture wiring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import encoders

ARCHS = ("perceiver", "spatialstack", "vlm3r")


@dataclass
class VggtArguments:
    # --- architecture / prompt ---
    arch: str = field(default="perceiver", metadata={"help": f"One of {ARCHS}."})
    prompt: str = field(default="clean", metadata={"help": "text256 | clean | plain | reason (data.py)."})
    # --- VGGT features (= the cache recipe; configs/caches/*.env) ---
    vggt_backbone: str = field(default="omega", metadata={"help": "omega | vggt1b"})
    vggt_repo: str = field(default="facebook/VGGT-Omega")
    vggt_file: str = field(default="vggt_omega_1b_512.pt")
    vggt_revision: str = field(default="", metadata={"help": "HF snapshot sha (default: main)."})
    vggt_checkpoint: str = field(default="", metadata={"help": "Explicit weight path; default: "
                                 "<vggt_repo>/<vggt_file> resolved inside HF_HOME."})
    vggt_frames: str = field(default="balanced", metadata={"help": "balanced | native (encoders.py)."})
    vggt_image_resolution: int = field(default=256, metadata={"help": "Balanced-resize target."})
    vggt_layers: Optional[List[int]] = field(default=None, metadata={"help": "native caches: "
                                             "aggregator layers stored, shallow to deep."})
    video_fps: float = 1.0
    video_max_frames: int = 64
    image_patch_size: int = 16
    vggt_embed_dim: int = 2048
    # --- perceiver ---
    frame_num_latents: int = 512
    camera_num_latents: int = 32
    frame_widening_factor: int = 2
    frame_placeholder_token: str = "<|quad_start|>"
    camera_placeholder_token: str = "<|quad_end|>"
    # --- spatialstack ---
    geometry_encoder_layers: List[int] = field(default_factory=lambda: [11, 17, 23])
    geometry_fusion_layers: List[int] = field(default_factory=lambda: [0, 1, 2])
    merger_hidden_dim: int = 4096
    # --- vlm3r ---
    fusion_num_heads: int = 16
    fusion_dropout: float = 0.1

    def checkpoint(self) -> str:
        return self.vggt_checkpoint or encoders.resolve_checkpoint(
            self.vggt_repo, self.vggt_file, self.vggt_revision or None)

    def cache_params(self) -> dict:
        return encoders.cache_params(self.vggt_backbone, self.checkpoint(), self.vggt_frames,
                                     self.video_fps, self.video_max_frames,
                                     self.vggt_image_resolution, self.vggt_layers)

    def placeholder_texts(self) -> tuple[str, str]:
        if self.arch != "perceiver":
            return "", ""
        return (self.frame_placeholder_token * self.frame_num_latents,
                self.camera_placeholder_token * self.camera_num_latents)


def model_class(args: VggtArguments):
    if args.arch == "spatialstack":
        from .ablations.spatialstack import SpatialStackForConditionalGeneration
        return SpatialStackForConditionalGeneration
    if args.arch == "vlm3r":
        from .ablations.vlm3r import Vlm3rForConditionalGeneration
        return Vlm3rForConditionalGeneration
    from .model import Qwen3VLWithVggtForConditionalGeneration
    return Qwen3VLWithVggtForConditionalGeneration


def batch_vggt_fn(args: VggtArguments):
    """Collator hook: cache entries -> the model's VGGT kwargs."""
    if args.arch == "spatialstack":
        from .ablations.spatialstack import batch_vggt
    elif args.arch == "vlm3r":
        from .ablations.vlm3r import batch_vggt
    else:
        from .model import batch_vggt
    return batch_vggt


def initialize_projector(model, args: VggtArguments, tokenizer) -> None:
    """Build the trainable VGGT module. Must run before LoRA/PEFT wrapping."""
    if args.arch == "spatialstack":
        model.initialize_projector(vggt_embed_dim=args.vggt_embed_dim, cached_layers=args.vggt_layers,
                                   geometry_encoder_layers=args.geometry_encoder_layers,
                                   geometry_fusion_layers=args.geometry_fusion_layers,
                                   merger_hidden_dim=args.merger_hidden_dim)
    elif args.arch == "vlm3r":
        model.initialize_projector(vggt_embed_dim=args.vggt_embed_dim,
                                   fusion_num_heads=args.fusion_num_heads,
                                   fusion_dropout=args.fusion_dropout)
    else:
        model.initialize_projector(tokenizer=tokenizer, vggt_embed_dim=args.vggt_embed_dim,
                                   frame_num_latents=args.frame_num_latents,
                                   camera_num_latents=args.camera_num_latents,
                                   frame_placeholder_token=args.frame_placeholder_token,
                                   camera_placeholder_token=args.camera_placeholder_token,
                                   frame_widening_factor=args.frame_widening_factor)
