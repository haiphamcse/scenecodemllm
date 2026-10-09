"""SpatialStack ablation: layered VGGT fusion through Qwen3-VL's own deepstack slots.

- No placeholder tokens (--prompt plain). The cache holds patch tokens of VGGT layers 11/17/23,
  computed on the det_es pixels Qwen itself sees (frame mode ``native``), so VGGT's patch grid
  equals Qwen's video grid.
- Each fused layer gets a ``VggtGeometryMerger`` (RMSNorm -> m x m patch-block merge -> MLP to the
  text hidden size, last Linear zero-init), giving one geometry vector per Qwen video token.
- Those are added to ``deepstack_visual_embeds[layer]`` for decoder layers ``--geometry_fusion_layers``
  (default 0 1 2): Qwen3-VL's text model already adds that list to the video-token hidden states
  after each of those layers, which is exactly SpatialStack's ``deepstack_language_add``.
- Trainable: the mergers (``vggt_projector``) + LoRA.
"""

from __future__ import annotations

import torch
from torch import nn

from ..model import Qwen3VLWithVggtForConditionalGeneration, Qwen3VLWithVggtModel


def group_frames(feats: torch.Tensor, grid_t: int) -> torch.Tensor:
    """``[T, ...]`` per-frame features -> ``[grid_t, ...]``, packed the way Qwen packs frames.

    Qwen3-VL's video processor repeats the LAST frame when T is not a multiple of
    temporal_patch_size and folds tps frames into one token grid (70 of the 4,878 clips have an
    odd det_es frame count). Averaging each group turns per-frame features into per-token ones.
    """
    t = feats.shape[0]
    if t == grid_t:
        return feats
    tps = -(-t // grid_t)
    pad = grid_t * tps - t
    if not 0 <= pad < tps:
        raise ValueError(f"{t} VGGT frames do not map onto {grid_t} Qwen temporal groups.")
    if pad:
        feats = torch.cat([feats, feats[-1:].expand(pad, *feats.shape[1:])], dim=0)
    return feats.view(grid_t, tps, *feats.shape[1:]).mean(dim=1)


def batch_vggt(cached, batch):
    """Collator hook: fold each clip's frames into Qwen's temporal groups in the worker (exact,
    the mergers' RMSNorm runs after; halves the shared-memory payload)."""
    return {"vggt_patch_tokens": [
        [group_frames(c[f"patch_{i}"], int(grid[0])) for i in range(len(c))]
        for c, grid in zip(cached, batch["video_grid_thw"])
    ]}


class VggtGeometryMerger(nn.Module):
    """VGGT patch features of one layer -> one vector per Qwen video token (zero-init output)."""

    def __init__(self, vggt_dim: int, hidden_dim: int, spatial_merge_size: int = 2,
                 merger_hidden_dim: int = 4096) -> None:
        super().__init__()
        self.vggt_dim = vggt_dim
        self.merge_size = spatial_merge_size
        self.ln_q = nn.RMSNorm(vggt_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(vggt_dim * spatial_merge_size**2, merger_hidden_dim),
            nn.GELU(),
            nn.Linear(merger_hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``[T, H, W, d]`` on Qwen's patch grid -> ``[T*(H/m)*(W/m), hidden]``."""
        m = self.merge_size
        t, h, w, _ = feats.shape
        if h % m or w % m:
            raise ValueError(f"Geometry grid {(h, w)} is not divisible by merge size {m}.")
        x = self.ln_q(feats)
        # Row-major over (t, h//m, w//m), block flattened last: Qwen's patch-merger order.
        x = x.view(t, h // m, m, w // m, m, self.vggt_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.vggt_dim * m * m)
        return self.mlp(x)


class VggtLayeredFusion(nn.Module):
    """One merger per fused layer. A plain module: PEFT modules_to_save refuses a bare ModuleList."""

    def __init__(self, num_layers: int, **merger_kwargs) -> None:
        super().__init__()
        self.mergers = nn.ModuleList(VggtGeometryMerger(**merger_kwargs) for _ in range(num_layers))

    def forward(self, feats_per_layer):
        if len(feats_per_layer) != len(self.mergers):
            raise ValueError(f"Got {len(feats_per_layer)} layers for {len(self.mergers)} mergers.")
        return [merger(feats) for merger, feats in zip(self.mergers, feats_per_layer)]


class SpatialStackModel(Qwen3VLWithVggtModel):
    def __init__(self, config):
        super().__init__(config)
        self.geometry_cache_indices: list[int] = []
        self.geometry_fusion_layers: list[int] = []

    def initialize_projector(self, vggt_embed_dim: int, cached_layers: list[int],
                             geometry_encoder_layers: list[int], geometry_fusion_layers: list[int],
                             merger_hidden_dim: int = 4096) -> None:
        if len(geometry_encoder_layers) != len(geometry_fusion_layers):
            raise ValueError("geometry_encoder_layers and geometry_fusion_layers need one entry per merger.")
        n_dec = int(self.config.text_config.num_hidden_layers)
        if any(not 0 <= l < n_dec for l in geometry_fusion_layers):
            raise ValueError(f"fusion layers {geometry_fusion_layers} outside the {n_dec} decoder layers.")
        missing = [l for l in geometry_encoder_layers if l not in cached_layers]
        if missing:
            raise ValueError(f"VGGT layers {missing} are not in the cache (holds {cached_layers}).")
        self.geometry_cache_indices = [cached_layers.index(l) for l in geometry_encoder_layers]
        self.geometry_fusion_layers = list(geometry_fusion_layers)
        self.vggt_projector = VggtLayeredFusion(
            num_layers=len(geometry_encoder_layers),
            vggt_dim=vggt_embed_dim,
            hidden_dim=int(self.config.text_config.hidden_size),
            spatial_merge_size=int(self.config.vision_config.spatial_merge_size),
            merger_hidden_dim=merger_hidden_dim,
        ).to(device=self.device, dtype=self.dtype)

    def _geometry_layer_embeds(self, patch_tokens, grid_t, grid_h, grid_w, dtype, device):
        """One ``[num_video_tokens, hidden]`` geometry tensor per fusion layer, for one clip."""
        feats_per_layer = []
        for idx in self.geometry_cache_indices:
            patches = patch_tokens[idx].to(device=device, dtype=dtype)
            if patches.shape[1] != grid_h * grid_w:
                raise ValueError(f"VGGT entry has {patches.shape[1]} patch tokens per frame but Qwen's "
                                 f"video grid is {grid_h}x{grid_w}; the features came from other pixels.")
            # No-op when the collator already grouped (training); real work at eval.
            feats_per_layer.append(group_frames(patches, grid_t).view(grid_t, grid_h, grid_w, -1))
        # Through the module, not .mergers, so PEFT's modules_to_save wrapper stays on the path.
        return self.vggt_projector(feats_per_layer)

    def fuse_deepstack(self, deepstack_visual_embeds, deepstack_video_embeds, pixel_values,
                       vggt_patch_tokens, video_grid_thw, dtype, device):
        # Prefill only: decode steps carry no video, hence no deepstack list to extend.
        if self.vggt_projector is None or vggt_patch_tokens is None or deepstack_video_embeds is None:
            return deepstack_visual_embeds
        if pixel_values is not None:
            raise ValueError("Layered geometry fusion assumes video-only inputs.")
        if video_grid_thw is None or len(vggt_patch_tokens) != len(video_grid_thw):
            raise ValueError("Need one VGGT entry per video grid.")
        per_video = [self._geometry_layer_embeds(p, *(int(v) for v in g), dtype, device)
                     for p, g in zip(vggt_patch_tokens, video_grid_thw)]
        geo_per_layer = [torch.cat(layer, dim=0) for layer in zip(*per_video)]
        fused = list(deepstack_visual_embeds)
        # ponytail: fusion layers past Qwen's deepstack depth (3) are zero-padded (a clone + add
        # per padded layer); fork the text-model loop if deep fusion layers become the default.
        while len(fused) < max(self.geometry_fusion_layers) + 1:
            fused.append(torch.zeros_like(fused[0]))
        for layer_idx, geo in zip(self.geometry_fusion_layers, geo_per_layer):
            if geo.shape != fused[layer_idx].shape:
                raise ValueError(f"Geometry {tuple(geo.shape)} != visual block "
                                 f"{tuple(fused[layer_idx].shape)} at fusion layer {layer_idx}.")
            fused[layer_idx] = fused[layer_idx] + geo.to(fused[layer_idx].dtype)
        return fused


class SpatialStackForConditionalGeneration(Qwen3VLWithVggtForConditionalGeneration):
    inner_model_class = SpatialStackModel


if __name__ == "__main__":
    # CPU self-check: zero-init no-op and Qwen's patch-block order (the two silent failure modes).
    torch.manual_seed(0)
    t, h, w, d, m = 3, 4, 6, 8, 2
    merger = VggtGeometryMerger(vggt_dim=d, hidden_dim=16, spatial_merge_size=m, merger_hidden_dim=32)
    assert torch.all(merger(torch.randn(t, h, w, d)) == 0), "zero-init broken"
    tagged = torch.zeros(t, h, w, d)
    tagged[..., 0] = torch.arange(t * h * w, dtype=torch.float32).view(t, h, w)
    merger.ln_q, merger.mlp = nn.Identity(), nn.Identity()
    want = torch.tensor([ti * h * w + by * m * w + bx * m for ti in range(t)
                         for by in range(h // m) for bx in range(w // m)], dtype=torch.float32)
    assert torch.equal(merger(tagged)[:, 0], want), "block order mismatch"
    for total, groups in ((64, 32), (55, 28)):
        x = torch.arange(total, dtype=torch.float32).view(total, 1)
        got = group_frames(x, groups)[:, 0]
        exp = torch.tensor([(min(2 * g, total - 1) + min(2 * g + 1, total - 1)) / 2 for g in range(groups)])
        assert torch.equal(got, exp), (total, got)
    print("spatialstack selftest ok")
