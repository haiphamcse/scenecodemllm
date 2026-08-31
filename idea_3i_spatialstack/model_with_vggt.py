"""Qwen3-VL with SpatialStack-style layered VGGT geometry fusion (idea_3i_spatialstack).

Architecture
------------
- Qwen3-VL encodes the video as usual (``pixel_values_videos`` -> video tokens);
  the native visual stream is untouched.
- VGGTOmega (frozen) runs once on ``raw_videos`` ([T, C, H, W] in [0, 255]). Its
  aggregator caches layers ``(4, 11, 17, 23)``; we read ``--geometry_encoder_layers``
  (default ``11 17 23``, SpatialStack's default) instead of only the last one.
- Each selected VGGT layer gets its own ``VggtGeometryMerger`` (RMSNorm ->
  m x m patch-block merge -> MLP to the text hidden size, last Linear zero-init),
  producing one geometry vector per Qwen video token.
- Those are **added into the LLM decoder hidden states at video-token positions**
  at decoder layers ``--geometry_fusion_layers`` (default ``0 1 2``). That is
  SpatialStack's ``deepstack_language_add``
  (``feature_fusion.py:434``, ``modeling_qwen3_5.py:363``): geometry re-enters the
  language stream repeatedly rather than once before the decoder, so shallow VGGT
  layers meet shallow decoder layers.

Why no custom decoder loop
--------------------------
Qwen3-VL's text model already does exactly this injection for its own vision
deepstack features: ``deepstack_visual_embeds[layer_idx]`` is added to
``hidden_states[visual_pos_masks]`` after decoder layer ``layer_idx``
(``modeling_qwen3_vl.py:929``). Geometry is therefore summed into that same list,
which keeps the fusion identical to SpatialStack's while forking nothing.

Contrast with siblings: idea_3i compresses VGGT into Perceiver latents scattered
into placeholder tokens (single stage, latent-only); idea_4a_sg_vgadd adds
geometry once onto the video embeds before the decoder (single stage). This is the
layered variant.

- VGGTOmega is frozen; only ``vggt_projector`` (the per-layer mergers) trains,
  alongside LoRA adapters on the LLM.
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

# VGGTOmega's patch stride (aggregator.py:23), which equals Qwen3-VL's, hence no
# grid interpolation anywhere below.
VGGT_PATCH_SIZE = 16
# Aggregator blocks whose activations are kept (aggregator.py:30); anything else
# comes back as None.
VGGT_CACHED_LAYERS = (4, 11, 17, 23)


class VggtGeometryMerger(nn.Module):
    """VGGT patch features of one layer -> one vector per Qwen video token.

    Port of ``GeometryFeatureMerger`` / the ``deepstack_language_add`` branch of
    SpatialStack (``feature_fusion.py:320``): RMSNorm the VGGT features, group them
    into the same ``merge_size x merge_size`` blocks Qwen uses for its own patches,
    and project the concatenated block to the text hidden size.

    The last Linear is ZERO-INITIALISED (SpatialStack's
    ``reset_residual_branches_to_noop``), so at step 0 every fusion layer is an
    exact no-op and the model is plain Qwen3-VL on video.
    """

    def __init__(
        self,
        vggt_dim: int,
        hidden_dim: int,
        spatial_merge_size: int = 2,
        merger_hidden_dim: int = 4096,
    ) -> None:
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
        """``[T, H, W, vggt_dim]`` on Qwen's patch grid -> ``[T*(H/m)*(W/m), hidden]``."""
        m = self.merge_size
        t, h, w, _ = feats.shape
        if h % m or w % m:
            raise ValueError(f"Geometry grid {(h, w)} is not divisible by merge size {m}.")
        x = self.ln_q(feats)
        # Group each m x m patch block into one token. The permute matches Qwen's
        # patch merger ordering: row-major over (t, h//m, w//m), block flattened last.
        x = x.view(t, h // m, m, w // m, m, self.vggt_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.vggt_dim * m * m)
        return self.mlp(x)


class VggtLayeredFusion(nn.Module):
    """One ``VggtGeometryMerger`` per fused VGGT layer.

    A plain module rather than a bare ``nn.ModuleList`` because PEFT's
    ``modules_to_save`` refuses to wrap container types.
    """

    def __init__(self, num_layers: int, **merger_kwargs) -> None:
        super().__init__()
        self.mergers = nn.ModuleList(VggtGeometryMerger(**merger_kwargs) for _ in range(num_layers))

    def forward(self, feats_per_layer: list[torch.Tensor]) -> list[torch.Tensor]:
        """One ``[T, H, W, vggt_dim]`` per fused layer -> one ``[tokens, hidden]`` each."""
        if len(feats_per_layer) != len(self.mergers):
            raise ValueError(f"Got {len(feats_per_layer)} layers for {len(self.mergers)} mergers.")
        return [merger(feats) for merger, feats in zip(self.mergers, feats_per_layer)]


class Qwen3VLWithVggtModel(Qwen3VLModel):
    """Qwen3-VL backbone with layered VGGT geometry fusion in the decoder."""

    def __init__(self, config):
        super().__init__(config)
        self.vggt_model: VGGTOmega | None = None
        self.vggt_projector: VggtLayeredFusion | None = None
        self.geometry_encoder_layers: list[int] = []
        self.geometry_fusion_layers: list[int] = []

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
        geometry_encoder_layers: list[int],
        geometry_fusion_layers: list[int],
        merger_hidden_dim: int = 4096,
    ) -> None:
        if len(geometry_encoder_layers) != len(geometry_fusion_layers):
            raise ValueError(
                f"geometry_encoder_layers {geometry_encoder_layers} and geometry_fusion_layers "
                f"{geometry_fusion_layers} must have the same length (one merger per pair)."
            )
        num_decoder_layers = int(self.config.text_config.num_hidden_layers)
        for layer in geometry_fusion_layers:
            if not 0 <= layer < num_decoder_layers:
                raise ValueError(f"Fusion layer {layer} outside the {num_decoder_layers} decoder layers.")
        for layer in geometry_encoder_layers:
            if layer % 24 not in VGGT_CACHED_LAYERS:
                raise ValueError(
                    f"VGGT layer {layer} is not cached by the aggregator "
                    f"(cached: {VGGT_CACHED_LAYERS}); its activations come back as None."
                )
        self.geometry_encoder_layers = list(geometry_encoder_layers)
        self.geometry_fusion_layers = list(geometry_fusion_layers)

        vggt = VGGTOmega()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        vggt.load_state_dict(state_dict)
        del state_dict

        self.vggt_model = vggt.aggregator.to(device=self.device, dtype=self.dtype)
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad = False

        self.vggt_projector = VggtLayeredFusion(
            num_layers=len(self.geometry_encoder_layers),
            vggt_dim=vggt_embed_dim,
            hidden_dim=int(self.config.text_config.hidden_size),
            spatial_merge_size=int(self.config.vision_config.spatial_merge_size),
            merger_hidden_dim=merger_hidden_dim,
        ).to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _extract_vggt_layer_patches(
        self,
        raw_video: torch.Tensor,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Frozen VGGT forward -> one ``[T, Np, d]`` patch-token tensor per requested layer."""
        if raw_video.ndim != 4:
            raise ValueError(f"Expected raw video [T, C, H, W], got shape {tuple(raw_video.shape)}")

        video = raw_video.to(device=device, dtype=self.dtype)
        if video.max() > 1.0:
            video = video / 255.0

        aggregated_tokens_list, patch_token_start = self.vggt_model(video.unsqueeze(0))
        layers = []
        for idx in self.geometry_encoder_layers:
            tokens = aggregated_tokens_list[idx]  # [1, T, 17 + Np, d] or None
            if tokens is None:
                raise ValueError(f"VGGT layer {idx} was not cached by the aggregator.")
            layers.append(tokens[0, :, patch_token_start:, :])
        return layers

    def _geometry_layer_embeds(
        self,
        raw_video: torch.Tensor,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """One ``[num_video_tokens, hidden]`` geometry tensor per fusion layer, for one scene."""
        # VGGT (frozen) runs under no_grad; the mergers below are trainable.
        layer_patches = self._extract_vggt_layer_patches(raw_video, device=device)

        height, width = raw_video.shape[-2:]
        hv, wv = height // VGGT_PATCH_SIZE, width // VGGT_PATCH_SIZE
        t_frames, n_patches = layer_patches[0].shape[0], layer_patches[0].shape[1]
        if hv * wv != n_patches:
            raise ValueError(
                f"VGGT returned {n_patches} patch tokens for a {height}x{width} frame, "
                f"but stride {VGGT_PATCH_SIZE} implies {hv}x{wv}={hv * wv}."
            )
        if (hv, wv) != (grid_h, grid_w):
            raise ValueError(
                f"VGGT patch grid {(hv, wv)} != Qwen video grid {(grid_h, grid_w)}; the "
                f"two strides were assumed equal."
            )

        feats_per_layer = []
        for patches in layer_patches:
            feats = patches.view(t_frames, hv, wv, -1).to(dtype=dtype)
            # Qwen3-VL packs temporal_patch_size frames into one token grid while
            # VGGT stays per-frame, so average each group to line the two up.
            if t_frames != grid_t:
                if t_frames % grid_t:
                    raise ValueError(
                        f"{t_frames} VGGT frames do not divide into {grid_t} Qwen temporal groups."
                    )
                feats = feats.view(grid_t, t_frames // grid_t, grid_h, grid_w, -1).mean(dim=1)
            feats_per_layer.append(feats)
        # Called through the module (not self.vggt_projector.mergers) so PEFT's
        # modules_to_save wrapper stays on the path.
        return self.vggt_projector(feats_per_layer)

    def _fuse_into_deepstack(
        self,
        deepstack_visual_embeds: list[torch.Tensor],
        raw_videos: list[torch.Tensor],
        video_grid_thw: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Sum per-layer geometry into Qwen's deepstack list (SpatialStack layered add)."""
        if video_grid_thw is None:
            raise ValueError("video_grid_thw is required to align VGGT features with video tokens.")
        if len(raw_videos) != len(video_grid_thw):
            raise ValueError(f"Got {len(raw_videos)} raw videos for {len(video_grid_thw)} video grids.")

        # [num_fusion_layers][num_videos] -> concatenated over videos, matching the
        # row order of Qwen's own video features.
        per_video = [
            self._geometry_layer_embeds(raw_video, *(int(v) for v in grid), dtype, device)
            for raw_video, grid in zip(raw_videos, video_grid_thw)
        ]
        geo_per_layer = [torch.cat(layer, dim=0) for layer in zip(*per_video)]

        fused = list(deepstack_visual_embeds)
        # ponytail: layers past Qwen's own deepstack depth are padded with zero
        # tensors, which costs a clone + zero-add per padded decoder layer. Fine for
        # the default fusion layers (0 1 2, no padding); fork the text-model loop if
        # deep fusion layers ever become the main setting.
        needed = max(self.geometry_fusion_layers) + 1
        while len(fused) < needed:
            fused.append(torch.zeros_like(fused[0]))
        for layer_idx, geo in zip(self.geometry_fusion_layers, geo_per_layer):
            if geo.shape != fused[layer_idx].shape:
                raise ValueError(
                    f"Geometry tokens {tuple(geo.shape)} do not match the visual token block "
                    f"{tuple(fused[layer_idx].shape)} at fusion layer {layer_idx}."
                )
            fused[layer_idx] = fused[layer_idx] + geo.to(fused[layer_idx].dtype)
        return fused

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

        # ----- SpatialStack layered geometry fusion -------------------- #
        # Prefill only: on decode steps pixel_values_videos is None, so there is no
        # deepstack list to extend and the cached fused states are reused.
        if (
            self.vggt_projector is not None
            and raw_videos is not None
            and deepstack_video_embeds is not None
        ):
            if pixel_values is not None:
                raise ValueError(
                    "Layered geometry fusion assumes video-only inputs; interleaved images "
                    "would break the row alignment of the deepstack visual block."
                )
            deepstack_visual_embeds = self._fuse_into_deepstack(
                deepstack_visual_embeds, raw_videos, video_grid_thw,
                dtype=inputs_embeds.dtype, device=inputs_embeds.device,
            )

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
        geometry_encoder_layers: list[int],
        geometry_fusion_layers: list[int],
        merger_hidden_dim: int = 4096,
    ) -> None:
        self.model.initialize_vggt(
            checkpoint_path,
            vggt_embed_dim=vggt_embed_dim,
            geometry_encoder_layers=geometry_encoder_layers,
            geometry_fusion_layers=geometry_fusion_layers,
            merger_hidden_dim=merger_hidden_dim,
        )

    @property
    def vggt_projector(self) -> VggtLayeredFusion | None:
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
    """CPU self-check: merger zero-init no-op, output shape, patch-block ordering.

    The two things that silently break this fusion are (a) the geometry branch not
    actually starting as a no-op, and (b) grouping patches in an order that does not
    match Qwen's patch merger -- which would add geometry to the wrong token and look
    exactly like "geometry doesn't help".
    """
    torch.manual_seed(0)
    t, h, w, d, hidden, m = 3, 4, 6, 8, 16, 2
    merger = VggtGeometryMerger(vggt_dim=d, hidden_dim=hidden, spatial_merge_size=m,
                                merger_hidden_dim=32).eval()
    feats = torch.randn(t, h, w, d)

    with torch.no_grad():
        out = merger(feats)
    assert out.shape == (t * (h // m) * (w // m), hidden), out.shape
    assert torch.all(out == 0), "zero-init broken: fusion is not a no-op at step 0"

    # Once the last Linear is non-zero the branch must actually contribute.
    with torch.no_grad():
        merger.mlp[-1].weight.normal_()
        assert merger(feats).abs().sum() > 0

    # Block ordering: tag every patch with its flat (t, y, x) id and make the mlp
    # return the first of the m*m concatenated features, so each output token is the
    # top-left patch of its block. Expected order is row-major over (t, h//m, w//m).
    tagged = torch.zeros(t, h, w, d)
    for ti in range(t):
        for y in range(h):
            for x in range(w):
                tagged[ti, y, x, 0] = ti * h * w + y * w + x
    merger.ln_q = nn.Identity()
    merger.mlp = nn.Identity()
    got = merger(tagged)[:, 0]
    want = torch.tensor([
        ti * h * w + (by * m) * w + (bx * m)
        for ti in range(t) for by in range(h // m) for bx in range(w // m)
    ], dtype=got.dtype)
    assert torch.equal(got, want), f"block order mismatch:\n got {got}\nwant {want}"

    # Layered injection: geometry lands on the requested decoder layers only, and
    # layers past Qwen's own deepstack depth get zero-padded rather than dropped.
    model = Qwen3VLWithVggtModel.__new__(Qwen3VLWithVggtModel)
    model.geometry_fusion_layers = [0, 2, 4]
    native = [torch.ones(5, hidden) for _ in range(3)]
    geo = [torch.full((5, hidden), float(i + 1)) for i in range(3)]
    model._geometry_layer_embeds = lambda *a, **k: geo
    fused = model._fuse_into_deepstack(native, [None], torch.tensor([[1, 2, 2]]),
                                       torch.float32, torch.device("cpu"))
    assert len(fused) == 5, len(fused)
    assert torch.all(fused[0] == 2) and torch.all(fused[2] == 3) and torch.all(fused[4] == 3)
    assert torch.all(fused[1] == 1) and torch.all(fused[3] == 0)
    print("ok: zero-init no-op, shape, row-major block ordering, layered injection")


if __name__ == "__main__":
    _self_check()
