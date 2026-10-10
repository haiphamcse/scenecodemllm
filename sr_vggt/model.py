"""Qwen3-VL + VGGT features. Core architecture: two Perceiver IO encoders whose latents are
scattered into placeholder tokens (main and the Perceiver ablations).

- Qwen3-VL encodes the video as usual (``pixel_values_videos`` -> video tokens).
- The batch carries the clip's VGGT tokens: ``vggt_patch_tokens`` ``[T, Np, d]`` and
  ``vggt_camera_tokens`` ``[T, S, d]`` (camera + register tokens), one tensor per row; from the
  cache at training time, from the live encoder at eval (encoders.py).
- ``frame_encoder``  : ``[1, T*Np, d]`` -> ``[1, L, hidden]``  scattered into L ``<|quad_start|>``
  ``camera_encoder`` : ``[1, T*S, d]``  -> ``[1, M, hidden]``  scattered into M ``<|quad_end|>``
  the way Qwen3-VL scatters its own video features, so generation needs no extra plumbing.
- Trainable: the Perceivers (``vggt_projector``) + LoRA.

Ablations (ablations/) subclass ``Qwen3VLWithVggtModel`` and override one of the two fusion
hooks (``fuse_video_embeds``, ``fuse_deepstack``); their ``vggt_projector`` replaces the
Perceivers and the placeholder scatter stays off (no placeholder ids).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.utils import can_return_tuple

from perceiver.model.core.adapter import InputAdapter
from perceiver.model.core.modules import PerceiverEncoder


class FeatureInputAdapter(InputAdapter):
    """Pass-through adapter for pre-extracted features (PerceiverEncoder has no rotary PE)."""

    def forward(self, x):
        return x


class VggtPerceiver(nn.Module):
    """Two Perceiver IO encoders: a heavy one over patch tokens, a light one over camera tokens."""

    def __init__(self, vggt_dim: int, hidden_dim: int, frame_num_latents: int,
                 camera_num_latents: int, frame_widening_factor: int = 1) -> None:
        super().__init__()
        self.frame_encoder = PerceiverEncoder(
            input_adapter=FeatureInputAdapter(num_input_channels=vggt_dim),
            num_latents=frame_num_latents,
            num_latent_channels=hidden_dim,
            num_cross_attention_heads=16,
            num_cross_attention_layers=4,
            cross_attention_widening_factor=frame_widening_factor,
            num_self_attention_heads=16,
            num_self_attention_layers_per_block=4,
            num_self_attention_blocks=4,
            self_attention_widening_factor=frame_widening_factor,
        )
        self.camera_encoder = PerceiverEncoder(
            input_adapter=FeatureInputAdapter(num_input_channels=vggt_dim),
            num_latents=camera_num_latents,
            num_latent_channels=hidden_dim,
            num_cross_attention_heads=4,
            num_cross_attention_layers=2,
            num_self_attention_heads=4,
            num_self_attention_layers_per_block=2,
            num_self_attention_blocks=2,
        )

    def forward(self, patch_tokens, camera_tokens):
        """``[B, T*Np, d]``, ``[B, T*S, d]`` -> ``[B, L, hidden]``, ``[B, M, hidden]``."""
        return self.frame_encoder(patch_tokens), self.camera_encoder(camera_tokens)


def batch_vggt(cached, batch):
    """Collator hook: cache entries -> the model's VGGT kwargs (one tensor per row)."""
    return {"vggt_patch_tokens": [c["patch"] for c in cached],
            "vggt_camera_tokens": [c["camera"] for c in cached]}


def _single_token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Placeholder '{token}' must tokenize to exactly one token, got {ids}.")
    return int(ids[0])


class Qwen3VLWithVggtModel(Qwen3VLModel):
    """Qwen3-VL backbone with the VGGT fusion points."""

    def __init__(self, config):
        super().__init__(config)
        self.vggt_projector: nn.Module | None = None
        self.frame_placeholder_token: str | None = None
        self.camera_placeholder_token: str | None = None
        self.frame_placeholder_token_id: int | None = None
        self.camera_placeholder_token_id: int | None = None

    def initialize_projector(self, tokenizer, vggt_embed_dim: int, frame_num_latents: int,
                             camera_num_latents: int, frame_placeholder_token: str,
                             camera_placeholder_token: str, frame_widening_factor: int = 1) -> None:
        """Build the two Perceivers and resolve the placeholder ids."""
        self.frame_placeholder_token = frame_placeholder_token
        self.camera_placeholder_token = camera_placeholder_token
        self.frame_placeholder_token_id = _single_token_id(tokenizer, frame_placeholder_token)
        self.camera_placeholder_token_id = _single_token_id(tokenizer, camera_placeholder_token)
        self.vggt_projector = VggtPerceiver(
            vggt_dim=vggt_embed_dim,
            hidden_dim=int(self.config.text_config.hidden_size),
            frame_num_latents=frame_num_latents,
            camera_num_latents=camera_num_latents,
            frame_widening_factor=frame_widening_factor,
        ).to(device=self.device, dtype=self.dtype)

    # ----- fusion hooks (identity here; ablations override one) ----------------------- #
    def fuse_video_embeds(self, video_embeds, video_grid_thw, vggt_tokens):
        """Rewrite Qwen's video embeddings before they are scattered (VLM-3R)."""
        return video_embeds

    def fuse_deepstack(self, deepstack_visual_embeds, deepstack_video_embeds, pixel_values,
                       vggt_patch_tokens, video_grid_thw, dtype, device):
        """Add to Qwen's deepstack features before the decoder (SpatialStack)."""
        return deepstack_visual_embeds

    # ----- Perceiver placeholder scatter ---------------------------------------------- #
    def _extract_vggt_latents(self, patch_tokens_per_clip, camera_tokens_per_clip, dtype, device):
        """Per clip ``[T, Np, d]`` / ``[T, S, d]`` -> stacked ``[B, L, hidden]``, ``[B, M, hidden]``."""
        if len(patch_tokens_per_clip) != len(camera_tokens_per_clip):
            raise ValueError(f"Got {len(patch_tokens_per_clip)} patch and "
                             f"{len(camera_tokens_per_clip)} camera tensors; need one of each per clip.")
        frame_latents, camera_latents = [], []
        for patch, camera in zip(patch_tokens_per_clip, camera_tokens_per_clip):
            if patch.ndim != 3 or camera.ndim != 3:
                raise ValueError(f"Expected tokens [T, N, d]; got patch {tuple(patch.shape)} "
                                 f"camera {tuple(camera.shape)}.")
            patch = patch.to(device=device, dtype=dtype).flatten(0, 1).unsqueeze(0)    # [1, T*Np, d]
            camera = camera.to(device=device, dtype=dtype).flatten(0, 1).unsqueeze(0)  # [1, T*S, d]
            frame_lat, camera_lat = self.vggt_projector(patch, camera)
            frame_latents.append(frame_lat.squeeze(0))
            camera_latents.append(camera_lat.squeeze(0))
        return torch.stack(frame_latents, dim=0), torch.stack(camera_latents, dim=0)

    @staticmethod
    def _scatter_latents(inputs_embeds, input_ids, token_id, latents, token_str):
        """Scatter ``latents [B, K, hidden]`` into every ``token_id`` position."""
        mask_1d = input_ids == token_id
        per_row = mask_1d.sum(dim=1)
        if not torch.all(per_row == latents.shape[1]):
            raise ValueError(f"Each row must contain exactly {latents.shape[1]} `{token_str}` "
                             f"placeholders, got per-row counts {per_row.tolist()}.")
        mask_3d = mask_1d.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(mask_3d, latents.to(inputs_embeds.dtype))

    def scatter_vggt(self, inputs_embeds, input_ids, vggt_patch_tokens, vggt_camera_tokens):
        # Only when input_ids holds placeholders: decode steps skip the Perceiver.
        if input_ids is None or self.frame_placeholder_token_id is None:
            return inputs_embeds
        has_frame = (input_ids == self.frame_placeholder_token_id).any()
        has_camera = (input_ids == self.camera_placeholder_token_id).any()
        if not (has_frame or has_camera):
            return inputs_embeds
        if vggt_patch_tokens is None or vggt_camera_tokens is None:
            raise ValueError("Input contains VGGT placeholder tokens but `vggt_patch_tokens` / "
                             "`vggt_camera_tokens` were not provided.")
        frame_latents, camera_latents = self._extract_vggt_latents(
            vggt_patch_tokens, vggt_camera_tokens, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds = self._scatter_latents(inputs_embeds, input_ids, self.frame_placeholder_token_id,
                                              frame_latents, self.frame_placeholder_token)
        return self._scatter_latents(inputs_embeds, input_ids, self.camera_placeholder_token_id,
                                     camera_latents, self.camera_placeholder_token)

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
        vggt_patch_tokens=None,
        vggt_camera_tokens=None,
        vggt_tokens=None,
        **kwargs,
    ) -> tuple | Qwen3VLModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # ----- Qwen3-VL vision scatter (parent behaviour) ----- #
        image_mask = video_mask = None
        deepstack_image_embeds = deepstack_video_embeds = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                      image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = video_outputs.pooler_output
            deepstack_video_embeds = video_outputs.deepstack_features
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            # Prefill only: decode steps carry no pixels and reuse the fused KV cache.
            video_embeds = self.fuse_video_embeds(video_embeds, video_grid_thw, vggt_tokens)
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds,
                                                      video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        inputs_embeds = self.scatter_vggt(inputs_embeds, input_ids, vggt_patch_tokens, vggt_camera_tokens)

        # ----- Deepstack visual position mask aggregation ----- #
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
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
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

        deepstack_visual_embeds = self.fuse_deepstack(
            deepstack_visual_embeds, deepstack_video_embeds, pixel_values, vggt_patch_tokens,
            video_grid_thw, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                past_key_values=past_key_values)

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
        return Qwen3VLModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)


class Qwen3VLWithVggtForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """LM head + loss on top of ``inner_model_class`` (ablations swap it)."""

    inner_model_class = Qwen3VLWithVggtModel

    def __init__(self, config):
        super().__init__(config)
        self.model = self.inner_model_class(config)
        self.post_init()

    def initialize_projector(self, **kwargs) -> None:
        self.model.initialize_projector(**kwargs)

    @property
    def vggt_projector(self):
        return self.model.vggt_projector

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
        vggt_patch_tokens=None,
        vggt_camera_tokens=None,
        vggt_tokens=None,
        skip_logits: bool | None = None,
        return_token_accuracy: bool = False,
        **kwargs,
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:
        from liger_kernel.transformers.model.loss_utils import (
            LigerForCausalLMLoss, unpack_cross_entropy_result)
        from liger_kernel.transformers.model.qwen3_vl import LigerQwen3VLCausalLMOutputWithPast

        # TRL injects these under --use_liger_kernel; they are ours, not self.model's.
        shift_labels = kwargs.pop("shift_labels", None)
        return_predicted_tokens = kwargs.pop("return_predicted_tokens", False)
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
            vggt_patch_tokens=vggt_patch_tokens,
            vggt_camera_tokens=vggt_camera_tokens,
            vggt_tokens=vggt_tokens,
            **kwargs,
        )
        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        kept_hidden_states = hidden_states[:, slice_indices, :]

        # Liger fused linear CE never materialises the [seq, vocab] logits. Liger's monkey-patch
        # hits the parent class, which this override shadows, so the fused path is called here
        # (mirrors liger_kernel/transformers/model/qwen3_vl.py::lce_forward).
        if skip_logits is None:
            skip_logits = self.training and (labels is not None or shift_labels is not None)

        logits = loss = token_accuracy = predicted_tokens = None
        if skip_logits:
            result = LigerForCausalLMLoss(
                hidden_states=kept_hidden_states,
                lm_head_weight=self.lm_head.weight,
                labels=labels,
                shift_labels=shift_labels,
                hidden_size=self.config.text_config.hidden_size,
                return_token_accuracy=return_token_accuracy,
                return_predicted_tokens=return_predicted_tokens,
                **kwargs,
            )
            loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
        else:
            logits = self.lm_head(kept_hidden_states)
            if labels is not None or shift_labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, shift_labels=shift_labels,
                                          vocab_size=self.config.text_config.vocab_size, **kwargs)

        return LigerQwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            token_accuracy=token_accuracy,
            predicted_tokens=predicted_tokens,
        )
