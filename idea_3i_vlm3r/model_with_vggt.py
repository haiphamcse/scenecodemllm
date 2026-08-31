"""Qwen3-VL with VLM-3R-style Spatial-Visual-View cross-attention fusion (idea_3i_vlm3r).

Architecture
------------
- Qwen3-VL encodes the video as usual; VGGTOmega (frozen) runs once on
  ``raw_videos`` ([T, C, H, W] in [0, 255]) and its last aggregator layer supplies
  the 3D tokens.
- Per frame the 3D token set is ``Z_3D = concat(camera_token, patch_tokens)`` --
  VLM-3R's ``spatial_tower_select_feature="all"`` (``llava_arch.py:333``), where the
  camera token carries the view pose and the patch tokens the geometry. VGGT-Omega's
  16 register tokens sit between the two and are dropped, matching VLM-3R's
  ``camera=[:, 0:1]`` / ``patches=[:, ps_idx:]`` split
  (``vggt_spatial_encoder.py:100``).
- Qwen's video embeddings ``H_v`` are then enriched in place:

      H_attn = CrossAttention(Q=LN(H_v), K,V=LN(Z_3D))
      H_v'   = H_v + out_proj(LN(H_attn))

  which is VLM-3R's ``CrossAttentionFusion``
  (``multimodal_fusion_block/builder.py:9``) and the paper's Spatial-Visual-View
  Fusion. No tokens are added to the sequence: the video block keeps its length and
  only its contents change.

Deviations from VLM-3R, all forced by the backbone
--------------------------------------------------
- **Placement.** VLM-3R fuses the raw CLIP/SigLIP features and *then* runs the
  ``mm_projector``. Qwen3-VL has no exposed projector -- its patch merger lives
  inside the vision tower and also feeds the deepstack features -- so fusion runs
  on the post-merger embeddings (already at the text hidden size) just before they
  are scattered into ``inputs_embeds``. Same residual cross-attention, one stage
  later.
- **Heads.** VLM-3R hardcodes 18 heads for SigLIP's d=1152. 18 does not divide
  Qwen's 2048, so the default is 16 (head dim 128).
- **LayerNorm position.** VLM-3R normalises *after* ``out_proj``; here it is before,
  so ``out_proj`` can be zero-initialised into an exact no-op without the residual
  branch's LayerNorm sitting on a zero-variance input (whose backward blows up by
  1/sqrt(eps)). Pre-norm on the residual branch is the standard arrangement anyway.
- **Attention scope.** Per Qwen temporal group: the video tokens of one group attend
  only to the ``Z_3D`` of the frames in that group, mirroring VLM-3R, which batches
  frames so attention never crosses a frame boundary.

Contrast with siblings: idea_3i compresses VGGT into Perceiver latents scattered
into placeholder tokens; idea_3i_spatialstack adds per-layer geometry into Qwen's
deepstack slots. This one leaves token count and decoder untouched and rewrites the
video embeddings themselves.

- VGGTOmega is frozen; only ``vggt_projector`` (the fusion block) trains, alongside
  LoRA adapters on the LLM.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithDeepstackFeatures,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.utils import can_return_tuple
from vggt_omega.models import VGGTOmega

# Aggregator blocks whose activations are kept (aggregator.py:30); anything else
# comes back as None. VLM-3R reads only the last layer.
VGGT_CACHED_LAYERS = (4, 11, 17, 23)


class VggtCrossAttentionFusion(nn.Module):
    """VLM-3R's Spatial-Visual-View fusion: visual tokens cross-attend to 3D tokens.

    Port of ``CrossAttentionFusion`` (vlm-3r ``multimodal_fusion_block/builder.py:9``).
    ``out_proj`` is ZERO-INITIALISED, so at step 0 ``H_v' == H_v`` exactly and the
    model is plain Qwen3-VL on video; VLM-3R trains this block from random init, but
    a random branch writing into every visual token from step 0 is a worse start when
    the backbone is already instruction-tuned.
    """

    def __init__(
        self,
        d_visual: int,
        d_spatial: int,
        d_attn: int,
        num_heads: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_attn % num_heads:
            raise ValueError(f"d_attn {d_attn} is not divisible by num_heads {num_heads}.")
        self.visual_norm = nn.LayerNorm(d_visual)
        self.spatial_norm = nn.LayerNorm(d_spatial)
        self.q_proj = nn.Linear(d_visual, d_attn)
        self.k_proj = nn.Linear(d_spatial, d_attn)
        self.v_proj = nn.Linear(d_spatial, d_attn)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_attn, num_heads=num_heads, batch_first=True
        )
        self.out_norm = nn.LayerNorm(d_attn)
        self.out_proj = nn.Linear(d_attn, d_visual)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, visual: torch.Tensor, spatial: torch.Tensor) -> torch.Tensor:
        """``[B, Nv, d_visual]`` queries over ``[B, N3d, d_spatial]`` -> ``[B, Nv, d_visual]``."""
        q = self.q_proj(self.visual_norm(visual))
        spatial_norm = self.spatial_norm(spatial)
        attn, _ = self.cross_attention(
            query=q, key=self.k_proj(spatial_norm), value=self.v_proj(spatial_norm),
            need_weights=False,
        )
        return visual + self.dropout(self.out_proj(self.out_norm(attn)))


class Qwen3VLWithVggtModel(Qwen3VLModel):
    """Qwen3-VL backbone whose video embeddings cross-attend to VGGT 3D tokens."""

    def __init__(self, config):
        super().__init__(config)
        self.vggt_model: VGGTOmega | None = None
        self.vggt_projector: VggtCrossAttentionFusion | None = None
        self.geometry_encoder_layer: int = 23

    def train(self, mode: bool = True):
        # Keep the frozen VGGT encoder in eval mode even when the trainer flips
        # the whole model to train mode.
        super().train(mode)
        if self.vggt_model is not None:
            self.vggt_model.eval()
        return self

    def initialize_vggt(
        self,
        checkpoint_path: str,
        vggt_embed_dim: int,
        geometry_encoder_layer: int = 23,
        fusion_num_heads: int = 16,
        fusion_dropout: float = 0.1,
    ) -> None:
        if geometry_encoder_layer % 24 not in VGGT_CACHED_LAYERS:
            raise ValueError(
                f"VGGT layer {geometry_encoder_layer} is not cached by the aggregator "
                f"(cached: {VGGT_CACHED_LAYERS}); its activations come back as None."
            )
        self.geometry_encoder_layer = int(geometry_encoder_layer)

        vggt = VGGTOmega()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        vggt.load_state_dict(state_dict)
        del state_dict

        self.vggt_model = vggt.aggregator.to(device=self.device, dtype=self.dtype)
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad = False

        hidden = int(self.config.text_config.hidden_size)
        self.vggt_projector = VggtCrossAttentionFusion(
            d_visual=hidden,
            d_spatial=vggt_embed_dim,
            d_attn=hidden,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        ).to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _extract_vggt_tokens(self, raw_video: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Frozen VGGT forward -> ``Z_3D`` of shape ``[T, 1 + Np, d]`` (camera + patches).

        The 16 register tokens between them are dropped, as in VLM-3R.
        """
        if raw_video.ndim != 4:
            raise ValueError(f"Expected raw video [T, C, H, W], got shape {tuple(raw_video.shape)}")

        video = raw_video.to(device=device, dtype=self.dtype)
        if video.max() > 1.0:
            video = video / 255.0

        aggregated_tokens_list, patch_token_start = self.vggt_model(video.unsqueeze(0))
        tokens = aggregated_tokens_list[self.geometry_encoder_layer]  # [1, T, 17 + Np, d]
        if tokens is None:
            raise ValueError(f"VGGT layer {self.geometry_encoder_layer} was not cached.")
        camera = tokens[0, :, 0:1, :]
        patches = tokens[0, :, patch_token_start:, :]
        return torch.cat([camera, patches], dim=1)

    def _fuse_video_embeds(
        self,
        video_embeds: torch.Tensor,
        raw_videos: list[torch.Tensor],
        video_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Rewrite each video's token block by cross-attending it to that video's Z_3D."""
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is required to align VGGT features with video tokens.")
        if len(raw_videos) != len(video_grid_thw):
            raise ValueError(f"Got {len(raw_videos)} raw videos for {len(video_grid_thw)} video grids.")

        merge_size = int(self.config.vision_config.spatial_merge_size)
        fused_blocks = []
        offset = 0
        for raw_video, grid in zip(raw_videos, video_grid_thw):
            grid_t, grid_h, grid_w = (int(v) for v in grid)
            num_tokens = grid_t * (grid_h // merge_size) * (grid_w // merge_size)
            block = video_embeds[offset : offset + num_tokens]
            offset += num_tokens
            if block.shape[0] != num_tokens:
                raise ValueError(
                    f"Video grid {(grid_t, grid_h, grid_w)} implies {num_tokens} tokens but only "
                    f"{block.shape[0]} embeddings remain."
                )

            spatial = self._extract_vggt_tokens(raw_video, device=video_embeds.device)
            t_frames = spatial.shape[0]
            if t_frames % grid_t:
                raise ValueError(
                    f"{t_frames} VGGT frames do not divide into {grid_t} Qwen temporal groups."
                )
            # Per-group cross-attention: group g's video tokens see only the Z_3D of
            # the frames Qwen packed into group g.
            spatial = spatial.reshape(grid_t, -1, spatial.shape[-1]).to(video_embeds.dtype)
            queries = block.view(grid_t, num_tokens // grid_t, -1)
            fused_blocks.append(self.vggt_projector(queries, spatial).reshape(num_tokens, -1))

        if offset != video_embeds.shape[0]:
            raise ValueError(
                f"Video grids account for {offset} tokens but got {video_embeds.shape[0]} embeddings."
            )
        return torch.cat(fused_blocks, dim=0)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        raw_videos=None,
        **kwargs,
    ) -> tuple | Qwen3VLModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # ----- Qwen3-VL vision scatter (parent behaviour) --------------- #
        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithDeepstackFeatures = self.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True
            )
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            # ----- VLM-3R Spatial-Visual-View fusion --------------------- #
            # Prefill only: on decode steps pixel_values_videos is None, so the
            # already-fused states are reused from the KV cache.
            if self.vggt_projector is not None and raw_videos is not None:
                video_embeds = self._fuse_video_embeds(video_embeds, raw_videos, video_grid_thw)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # ----- Deepstack visual position mask aggregation -------------- #
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLWithVggtForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """LM head + loss on top of ``Qwen3VLWithVggtModel``."""

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLWithVggtModel(config)
        self.post_init()

    def initialize_vggt(
        self,
        checkpoint_path: str,
        vggt_embed_dim: int,
        geometry_encoder_layer: int = 23,
        fusion_num_heads: int = 16,
        fusion_dropout: float = 0.1,
    ) -> None:
        self.model.initialize_vggt(
            checkpoint_path,
            vggt_embed_dim=vggt_embed_dim,
            geometry_encoder_layer=geometry_encoder_layer,
            fusion_num_heads=fusion_num_heads,
            fusion_dropout=fusion_dropout,
        )

    @property
    def vggt_projector(self) -> VggtCrossAttentionFusion | None:
        return self.model.vggt_projector

    @property
    def vggt_model(self) -> VGGTOmega | None:
        return self.model.vggt_model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        raw_videos=None,
        **kwargs,
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            raw_videos=raw_videos,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size
            )

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


def _self_check() -> None:
    """CPU self-check for the two things that silently break this fusion.

    (a) the branch not actually starting as a no-op, and (b) the per-group query/KV
    split pairing a group's video tokens with the wrong frames' geometry -- which
    would look exactly like "geometry doesn't help".
    """
    torch.manual_seed(0)
    b, nv, n3d, d_vis, d_sp = 3, 5, 7, 16, 24
    fusion = VggtCrossAttentionFusion(
        d_visual=d_vis, d_spatial=d_sp, d_attn=d_vis, num_heads=4, dropout=0.0
    ).eval()
    visual = torch.randn(b, nv, d_vis)
    spatial = torch.randn(b, n3d, d_sp)

    with torch.no_grad():
        out = fusion(visual, spatial)
    assert out.shape == (b, nv, d_vis), out.shape
    assert torch.equal(out, visual), "zero-init out_proj must make fusion an exact no-op"

    # After breaking the zero-init, output must change and must depend on the
    # spatial tokens (i.e. the KV path is actually wired in).
    with torch.no_grad():
        fusion.out_proj.weight.normal_(std=0.1)
        out_a = fusion(visual, spatial)
        out_b = fusion(visual, torch.randn(b, n3d, d_sp))
    assert not torch.allclose(out_a, visual), "fusion should change the visual tokens once trained"
    assert not torch.allclose(out_a, out_b), "output must depend on the 3D tokens"

    # Per-group batching: group g must be independent of every other group's Z_3D.
    with torch.no_grad():
        spatial_perturbed = spatial.clone()
        spatial_perturbed[1] = torch.randn(n3d, d_sp)
        out_c = fusion(visual, spatial_perturbed)
    assert torch.allclose(out_a[0], out_c[0], atol=1e-6), "group 0 leaked into group 1's Z_3D"
    assert torch.allclose(out_a[2], out_c[2], atol=1e-6), "group 2 leaked into group 1's Z_3D"
    assert not torch.allclose(out_a[1], out_c[1]), "group 1 ignored its own Z_3D"

    print("ok: zero-init no-op, shape, KV dependence, per-group isolation")


if __name__ == "__main__":
    _self_check()
