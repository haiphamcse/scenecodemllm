"""VLM-3R ablation: Qwen's video embeddings cross-attend to the clip's VGGT tokens.

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

from ..model import Qwen3VLWithVggtForConditionalGeneration, Qwen3VLWithVggtModel


def z3d(cached) -> torch.Tensor:
    """``{patch [T, Np, d], camera [T, 17, d]}`` -> ``Z_3D [T, 1 + Np, d]`` in bf16."""
    return torch.cat([cached["camera"][:, :1], cached["patch"]], dim=1).to(torch.bfloat16)


def batch_vggt(cached, batch):
    return {"vggt_tokens": [z3d(c) for c in cached]}


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
