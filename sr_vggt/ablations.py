"""The two ablations with their own fusion module. Each subclasses the core model (model.py),
picks the VGGT tokens it needs (``vggt_features``) and overrides one fusion hook.

SpatialStack ablation: layered VGGT fusion through Qwen3-VL's own deepstack slots.

- No placeholder tokens (--prompt plain). VGGT runs on the pixels Qwen itself sees
  (--vggt_frames native), so its patch grid equals Qwen's video grid; the patch tokens of layers
  --vggt_layers (11/17/23) are kept.
- Each fused layer gets a ``VggtGeometryMerger`` (RMSNorm -> m x m patch-block merge -> MLP to the
  text hidden size, last Linear zero-init), giving one geometry vector per Qwen video token.
- Those are added to ``deepstack_visual_embeds[layer]`` for decoder layers ``--geometry_fusion_layers``
  (default 0 1 2): Qwen3-VL's text model already adds that list to the video-token hidden states
  after each of those layers, which is exactly SpatialStack's ``deepstack_language_add``.
- Trainable: the mergers (``vggt_projector``) + LoRA.

VLM-3R ablation: Qwen's video embeddings cross-attend to the clip's VGGT tokens.

- No placeholder tokens (--prompt plain). Per frame ``Z_3D = concat(camera token, patch tokens)``
  of VGGT's last layer; the register tokens between them are dropped, as VLM-3R does.
- Post-merger, just before the scatter into ``inputs_embeds``:
      H_v' = H_v + out_proj(LN(CrossAttention(Q=LN(H_v), K,V=LN(Z_3D))))
  (VLM-3R's CrossAttentionFusion; LayerNorm moved before the zero-init out_proj so step 0 is an
  exact no-op; 16 heads since 18 does not divide 2048.) Sequence length is unchanged.
- Attention is per Qwen temporal group: group g's tokens see only the frames packed into g.
- Trainable: the fusion block (``vggt_projector``) + LoRA.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import Qwen3VLWithVggtForConditionalGeneration, Qwen3VLWithVggtModel


# ================================ SpatialStack ================================ #

def group_frames(feats: torch.Tensor, grid_t: int) -> torch.Tensor:
    """``[T, ...]`` per-frame features -> ``[grid_t, ...]``, packed the way Qwen packs frames.

    Qwen3-VL's video processor repeats the LAST frame when T is not a multiple of
    temporal_patch_size and folds tps frames into one token grid (clips can have an odd frame
    count). Averaging each group turns per-frame features into per-token ones.
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
        self.vggt_layers: list[int] = []
        self.geometry_layer_indices: list[int] = []
        self.geometry_fusion_layers: list[int] = []

    def vggt_features(self, raw_videos) -> dict:
        """Patch tokens of each of --vggt_layers, per clip (one frame per VGGT frame)."""
        out = []
        for video in raw_videos:
            tokens, ps = self.run_vggt(video)
            out.append([tokens[l][:, ps:] for l in self.vggt_layers])
        return {"vggt_patch_tokens": out}

    def initialize_projector(self, vggt_embed_dim: int, vggt_layers: list[int],
                             geometry_encoder_layers: list[int], geometry_fusion_layers: list[int],
                             merger_hidden_dim: int = 4096) -> None:
        if len(geometry_encoder_layers) != len(geometry_fusion_layers):
            raise ValueError("geometry_encoder_layers and geometry_fusion_layers need one entry per merger.")
        n_dec = int(self.config.text_config.num_hidden_layers)
        if any(not 0 <= l < n_dec for l in geometry_fusion_layers):
            raise ValueError(f"fusion layers {geometry_fusion_layers} outside the {n_dec} decoder layers.")
        missing = [l for l in geometry_encoder_layers if l not in vggt_layers]
        if missing:
            raise ValueError(f"geometry layers {missing} are not among --vggt_layers {vggt_layers}.")
        self.vggt_layers = list(vggt_layers)
        self.geometry_layer_indices = [vggt_layers.index(l) for l in geometry_encoder_layers]
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
        for idx in self.geometry_layer_indices:
            patches = patch_tokens[idx].to(device=device, dtype=dtype)
            if patches.shape[1] != grid_h * grid_w:
                raise ValueError(f"VGGT gives {patches.shape[1]} patch tokens per frame but Qwen's "
                                 f"video grid is {grid_h}x{grid_w}; VGGT saw other pixels.")
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


# ================================ VLM-3R ================================ #

class VggtCrossAttentionFusion(nn.Module):
    def __init__(self, d_visual: int, d_spatial: int, d_attn: int, num_heads: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if d_attn % num_heads:
            raise ValueError(f"d_attn {d_attn} is not divisible by num_heads {num_heads}.")
        self.visual_norm = nn.LayerNorm(d_visual)
        self.spatial_norm = nn.LayerNorm(d_spatial)
        self.q_proj = nn.Linear(d_visual, d_attn)
        self.k_proj = nn.Linear(d_spatial, d_attn)
        self.v_proj = nn.Linear(d_spatial, d_attn)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d_attn)
        self.out_proj = nn.Linear(d_attn, d_visual)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, visual: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """``[B, Nv, d_visual]`` queries over ``[B, N3d, d_spatial]`` -> ``[B, Nv, d_visual]``."""
        q = self.q_proj(self.visual_norm(visual))
        spatial_norm = self.spatial_norm(spatial)
        attn, _ = self.cross_attention(query=q, key=self.k_proj(spatial_norm),
                                       value=self.v_proj(spatial_norm), need_weights=False)
        return visual + self.dropout(self.out_proj(self.out_norm(attn)))


class Vlm3rModel(Qwen3VLWithVggtModel):
    def vggt_features(self, raw_videos) -> dict:
        """Per clip ``Z_3D [T, 1 + Np, d]``: last-layer camera token + patch tokens."""
        out = []
        for video in raw_videos:
            tokens, ps = self.run_vggt(video)
            out.append(torch.cat([tokens[-1][:, :1], tokens[-1][:, ps:]], dim=1))
        return {"vggt_tokens": out}

    def initialize_projector(self, vggt_embed_dim: int, fusion_num_heads: int = 16,
                             fusion_dropout: float = 0.1) -> None:
        hidden = int(self.config.text_config.hidden_size)
        self.vggt_projector = VggtCrossAttentionFusion(
            d_visual=hidden, d_spatial=vggt_embed_dim, d_attn=hidden,
            num_heads=fusion_num_heads, dropout=fusion_dropout,
        ).to(device=self.device, dtype=self.dtype)

    def fuse_video_embeds(self, video_embeds, video_grid_thw, vggt_tokens):
        if self.vggt_projector is None:
            return video_embeds
        if vggt_tokens is None:
            raise ValueError("The batch carries a video but no `vggt_tokens`.")
        if video_grid_thw is None or len(vggt_tokens) != len(video_grid_thw):
            raise ValueError("Need one Z_3D per video grid.")
        merge = int(self.config.vision_config.spatial_merge_size)
        tps = int(self.config.vision_config.temporal_patch_size)
        fused, offset = [], 0
        for z, grid in zip(vggt_tokens, video_grid_thw):
            grid_t, grid_h, grid_w = (int(v) for v in grid)
            n = grid_t * (grid_h // merge) * (grid_w // merge)
            block = video_embeds[offset:offset + n]
            offset += n
            if block.shape[0] != n or z.ndim != 3:
                raise ValueError(f"grid {(grid_t, grid_h, grid_w)} / Z_3D {tuple(z.shape)} mismatch.")
            z = z.to(device=video_embeds.device, dtype=video_embeds.dtype)
            # Qwen repeats the LAST frame to fill the final temporal group; pad Z_3D the same way.
            pad = grid_t * tps - z.shape[0]
            if not 0 <= pad < tps:
                raise ValueError(f"{z.shape[0]} VGGT frames do not fill {grid_t} groups of {tps}.")
            if pad:
                z = torch.cat([z, z[-1:].expand(pad, -1, -1)], dim=0)
            spatial = z.reshape(grid_t, -1, z.shape[-1])
            queries = block.view(grid_t, n // grid_t, -1)
            fused.append(self.vggt_projector(queries, spatial).reshape(n, -1))
        if offset != video_embeds.shape[0]:
            raise ValueError(f"Video grids cover {offset} tokens, got {video_embeds.shape[0]}.")
        return torch.cat(fused, dim=0)


class Vlm3rForConditionalGeneration(Qwen3VLWithVggtForConditionalGeneration):
    inner_model_class = Vlm3rModel


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
