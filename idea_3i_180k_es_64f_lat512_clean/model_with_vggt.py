"""Qwen3-VL with CACHED VGGTOmega features compressed by two Perceiver IO encoders and
scattered into placeholder tokens. VGGT-Omega itself is never run here.

- Qwen3-VL encodes the video as usual (``pixel_values_videos`` -> video tokens).
- The collator loads the clip's cached tokens: ``vggt_patch_tokens`` ``[T, Np, vggt_dim]``
  and ``vggt_camera_tokens`` ``[T, 17, vggt_dim]`` (camera + 16 register tokens).
- Two Perceiver IO encoders compress them (``num_latent_channels`` = Qwen hidden size):
    * ``frame_encoder``  : ``[1, T*Np, vggt_dim]`` -> ``[1, L, hidden]``
    * ``camera_encoder`` : ``[1, T*17, vggt_dim]`` -> ``[1, M, hidden]``
- ``forward`` scatters the frame latents into the ``L`` ``<|quad_start|>`` and the camera
  latents into the ``M`` ``<|quad_end|>`` placeholder positions, the way Qwen3-VL scatters
  its video features, so generation needs no extra plumbing.
- Trainable: the Perceivers (``vggt_projector``) + LoRA.
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
from liger_kernel.transformers.model.loss_utils import (
    LigerForCausalLMLoss,
    unpack_cross_entropy_result,
)
from liger_kernel.transformers.model.qwen3_vl import LigerQwen3VLCausalLMOutputWithPast

from perceiver.model.core.adapter import InputAdapter
from perceiver.model.core.modules import PerceiverEncoder


class FeatureInputAdapter(InputAdapter):
    """Pass-through adapter for pre-extracted continuous features.

    PerceiverEncoder does not use rotary position encoding, so a plain
    InputAdapter that reports the input feature dim is all that's needed.
    """

    def forward(self, x):
        return x


class VggtPerceiver(nn.Module):
    """Two Perceiver IO encoders compressing VGGT tokens into latent tokens.

    ``frame_encoder`` consumes the (many) VGGT patch tokens; ``camera_encoder``
    is a lighter encoder over the small camera+register token set.
    """

    def __init__(
        self,
        vggt_dim: int,
        hidden_dim: int,
        frame_num_latents: int,
        camera_num_latents: int,
        frame_widening_factor: int = 1,
    ) -> None:
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
    def forward(
        self,
        patch_tokens: torch.Tensor,
        camera_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, T*Np, vggt_dim]``, ``[B, T*17, vggt_dim]`` -> ``[B, L, hidden]``, ``[B, M, hidden]``."""
        return self.frame_encoder(patch_tokens), self.camera_encoder(camera_tokens)


def _single_token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Placeholder '{token}' must tokenize to exactly one token, got {ids}.")
    return int(ids[0])


class Qwen3VLWithVggtModel(Qwen3VLModel):
    """Qwen3-VL backbone that scatters VGGT-Perceiver latents into placeholder positions."""

    def __init__(self, config):
        super().__init__(config)
        self.vggt_projector: VggtPerceiver | None = None
        self.frame_placeholder_token: str | None = None
        self.camera_placeholder_token: str | None = None
        self.frame_placeholder_token_id: int | None = None
        self.camera_placeholder_token_id: int | None = None

    def initialize_projector(
        self,
        tokenizer,
        vggt_embed_dim: int,
        frame_num_latents: int,
        camera_num_latents: int,
        frame_placeholder_token: str,
        camera_placeholder_token: str,
        frame_widening_factor: int = 1,
    ) -> None:
        """Build the two Perceivers and resolve the placeholder ids."""
        if tokenizer is None:
            raise ValueError("`tokenizer` is required to resolve the placeholder token ids.")
        self.frame_placeholder_token = frame_placeholder_token
        self.camera_placeholder_token = camera_placeholder_token
        self.frame_placeholder_token_id = _single_token_id(tokenizer, frame_placeholder_token)
        self.camera_placeholder_token_id = _single_token_id(tokenizer, camera_placeholder_token)

        qwen_hidden_dim = int(self.config.text_config.hidden_size)
        self.vggt_projector = VggtPerceiver(
            vggt_dim=vggt_embed_dim,
            hidden_dim=qwen_hidden_dim,
            frame_num_latents=frame_num_latents,
            camera_num_latents=camera_num_latents,
            frame_widening_factor=frame_widening_factor,
        ).to(device=self.device, dtype=self.dtype)

    def _extract_vggt_latents(
        self,
        patch_tokens_per_clip: list[torch.Tensor],
        camera_tokens_per_clip: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached tokens -> ``(frame_latents [B, L, hidden], camera_latents [B, M, hidden])``.

        Per clip ``[T, Np, d]`` / ``[T, 17, d]`` off disk; T is flattened into the token axis
        and cast to the compute dtype, as the live-encoder model does.
        """
        if self.vggt_projector is None:
            raise RuntimeError(
                "VGGT projector is not initialized. Call `initialize_projector` first."
            )
        if len(patch_tokens_per_clip) != len(camera_tokens_per_clip):
            raise ValueError(
                f"Got {len(patch_tokens_per_clip)} patch and "
                f"{len(camera_tokens_per_clip)} camera token tensors; must be one of each per clip."
            )

        frame_latents: list[torch.Tensor] = []
        camera_latents: list[torch.Tensor] = []
        for patch, camera in zip(patch_tokens_per_clip, camera_tokens_per_clip):
            if patch.ndim != 3 or camera.ndim != 3:
                raise ValueError(
                    f"Expected cached tokens [T, N, d]; got patch {tuple(patch.shape)} "
                    f"camera {tuple(camera.shape)}."
                )
            patch = patch.to(device=device, dtype=dtype).flatten(0, 1).unsqueeze(0)    # [1, T*Np, d]
            camera = camera.to(device=device, dtype=dtype).flatten(0, 1).unsqueeze(0)  # [1, T*17, d]
            frame_lat, camera_lat = self.vggt_projector(patch, camera)
            frame_latents.append(frame_lat.squeeze(0))
            camera_latents.append(camera_lat.squeeze(0))

        return torch.stack(frame_latents, dim=0), torch.stack(camera_latents, dim=0)

    def _scatter_latents(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        latents: torch.Tensor,
        token_str: str,
    ) -> torch.Tensor:
        """Scatter ``latents [B, K, hidden]`` into every ``token_id`` position."""
        mask_1d = input_ids == token_id  # [B, S]
        per_row = mask_1d.sum(dim=1)
        expected = latents.shape[1]
        if not torch.all(per_row == expected):
            raise ValueError(
                f"Each batch row must contain exactly {expected} `{token_str}` "
                f"placeholder tokens, got per-row counts {per_row.tolist()}."
            )
        mask_3d = mask_1d.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(mask_3d, latents.to(inputs_embeds.dtype))

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

        # ----- VGGT-Perceiver scatter ---------------------------------- #
        # Only when input_ids contains placeholders; decode steps skip the Perceiver.
        if (
            input_ids is not None
            and self.vggt_projector is not None
            and self.frame_placeholder_token_id is not None
        ):
            has_frame = (input_ids == self.frame_placeholder_token_id).any()
            has_camera = (input_ids == self.camera_placeholder_token_id).any()
            if has_frame or has_camera:
                if vggt_patch_tokens is None or vggt_camera_tokens is None:
                    raise ValueError(
                        "Input contains VGGT placeholder tokens but `vggt_patch_tokens` / "
                        "`vggt_camera_tokens` were not provided (see vggt_cache.load)."
                    )
                frame_latents, camera_latents = self._extract_vggt_latents(
                    vggt_patch_tokens, vggt_camera_tokens,
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device,
                )
                inputs_embeds = self._scatter_latents(
                    inputs_embeds, input_ids, self.frame_placeholder_token_id,
                    frame_latents, self.frame_placeholder_token,
                )
                inputs_embeds = self._scatter_latents(
                    inputs_embeds, input_ids, self.camera_placeholder_token_id,
                    camera_latents, self.camera_placeholder_token,
                )

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

    def initialize_projector(
        self,
        tokenizer,
        vggt_embed_dim: int,
        frame_num_latents: int,
        camera_num_latents: int,
        frame_placeholder_token: str,
        camera_placeholder_token: str,
        frame_widening_factor: int = 1,
    ) -> None:
        self.model.initialize_projector(
            tokenizer=tokenizer,
            vggt_embed_dim=vggt_embed_dim,
            frame_num_latents=frame_num_latents,
            camera_num_latents=camera_num_latents,
            frame_placeholder_token=frame_placeholder_token,
            camera_placeholder_token=camera_placeholder_token,
            frame_widening_factor=frame_widening_factor,
        )

    @property
    def vggt_projector(self) -> VggtPerceiver | None:
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
        skip_logits: bool | None = None,
        return_token_accuracy: bool = False,
        **kwargs,
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:
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
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        kept_hidden_states = hidden_states[:, slice_indices, :]

        # Liger fused linear CE never materialises the [seq, 151936] logits. Liger's
        # monkey-patch hits the parent class, which this override shadows, so the fused
        # path is called here (mirrors liger_kernel/transformers/model/qwen3_vl.py::lce_forward).
        if skip_logits is None:
            skip_logits = self.training and (labels is not None or shift_labels is not None)

        logits = None
        loss = None
        token_accuracy = None
        predicted_tokens = None
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
                loss = self.loss_function(
                    logits=logits, labels=labels, shift_labels=shift_labels,
                    vocab_size=self.config.text_config.vocab_size, **kwargs,
                )

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

