"""Qwen3-VL with VGGTOmega features compressed by two Perceiver IO encoders and
scattered into placeholder tokens.

Architecture
------------
- Qwen3-VL encodes the video as usual (``pixel_values_videos`` -> video tokens).
- VGGTOmega (frozen) runs on ``raw_videos`` ([T, C, H, W] in [0, 255]) and yields
  per-frame tokens ``[1, T, 17 + Np, vggt_dim]``: token 0 is the camera token,
  tokens 1..16 are register tokens (the "17" group), tokens 17.. are patch tokens
  (the "Np" group).
- Two Perceiver IO encoders compress those into a fixed latent budget:
    * ``frame_encoder``  : VGGT patch tokens ``[1, T*Np, vggt_dim]`` -> ``[1, L, hidden]``
    * ``camera_encoder`` : VGGT camera+register ``[1, T*17, vggt_dim]`` -> ``[1, M, hidden]``
  ``num_latent_channels`` equals the Qwen text hidden size, so latents need no
  extra projection before scatter.
- The chat template inserts ``L`` ``<|quad_start|>`` + ``M`` ``<|quad_end|>``
  placeholder tokens (see collator.py; counts/tokens are train.py args). ``forward`` scatters
  the frame latents into ``<|quad_start|>`` positions and the camera latents into
  ``<|quad_end|>`` positions, exactly as Qwen3-VL scatters its own video features
  -- so HF generation handles cache length / M-RoPE / attention mask with no extra
  plumbing.
- VGGTOmega is frozen; only the two Perceiver encoders (``vggt_projector``) train.
"""

from __future__ import annotations

import os

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
        frame_latents = self.frame_encoder(patch_tokens)
        camera_latents = self.camera_encoder(camera_tokens)
        return frame_latents, camera_latents


def _single_token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Placeholder '{token}' must tokenize to exactly one token, got {ids}.")
    return int(ids[0])


class Qwen3VLWithVggtModel(Qwen3VLModel):
    """Qwen3-VL backbone that scatters VGGT-Perceiver latents into placeholder positions."""

    def __init__(self, config):
        super().__init__(config)
        self.vggt_model: VGGTOmega | None = None
        self.vggt_projector: VggtPerceiver | None = None
        self.frame_placeholder_token: str | None = None
        self.camera_placeholder_token: str | None = None
        self.frame_placeholder_token_id: int | None = None
        self.camera_placeholder_token_id: int | None = None

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
        tokenizer,
        vggt_embed_dim: int,
        frame_num_latents: int,
        camera_num_latents: int,
        frame_placeholder_token: str,
        camera_placeholder_token: str,
        frame_widening_factor: int = 1,
    ) -> None:
        if tokenizer is None:
            raise ValueError("`tokenizer` is required to resolve the placeholder token ids.")
        self.frame_placeholder_token = frame_placeholder_token
        self.camera_placeholder_token = camera_placeholder_token
        self.frame_placeholder_token_id = _single_token_id(tokenizer, frame_placeholder_token)
        self.camera_placeholder_token_id = _single_token_id(tokenizer, camera_placeholder_token)

        vggt = VGGTOmega()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        vggt.load_state_dict(state_dict)
        del state_dict

        self.vggt_model = vggt.aggregator.to(device=self.device, dtype=self.dtype)
        self.vggt_model.eval()
        for param in self.vggt_model.parameters():
            param.requires_grad = False

        qwen_hidden_dim = int(self.config.text_config.hidden_size)
        self.vggt_projector = VggtPerceiver(
            vggt_dim=vggt_embed_dim,
            hidden_dim=qwen_hidden_dim,
            frame_num_latents=frame_num_latents,
            camera_num_latents=camera_num_latents,
            frame_widening_factor=frame_widening_factor,
        ).to(device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def _extract_vggt_tokens(
        self,
        raw_video: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Frozen VGGT forward -> ``(camera_tokens [1, T, 17, d], patch_tokens [1, T, Np, d])``."""
        if raw_video.ndim != 4:
            raise ValueError(f"Expected raw video [T, C, H, W], got shape {tuple(raw_video.shape)}")

        video = raw_video.to(device=device, dtype=self.dtype)
        if video.max() > 1.0:
            video = video / 255.0
        video = video.unsqueeze(0)  # [1, T, C, H, W]

        aggregated_tokens_list, patch_token_start = self.vggt_model(video)
        final_tokens = aggregated_tokens_list[-1]  # [1, T, 17 + Np, d]
        camera_tokens = final_tokens[:, :, :patch_token_start, :]
        patch_tokens = final_tokens[:, :, patch_token_start:, :]
        return camera_tokens, patch_tokens

    def _extract_vggt_latents(
        self,
        raw_videos: list[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(frame_latents [B, L, hidden], camera_latents [B, M, hidden])``."""
        if self.vggt_projector is None:
            raise RuntimeError("VGGT projector is not initialized. Call `initialize_vggt` first.")

        frame_latents: list[torch.Tensor] = []
        camera_latents: list[torch.Tensor] = []
        for video in raw_videos:
            # VGGT (frozen) runs under no_grad; the projector below is trainable.
            camera_tokens, patch_tokens = self._extract_vggt_tokens(video, device=device)
            camera_tokens = camera_tokens.flatten(1, 2).to(dtype=dtype)  # [1, T*17, d]
            patch_tokens = patch_tokens.flatten(1, 2).to(dtype=dtype)    # [1, T*Np, d]
            frame_lat, camera_lat = self.vggt_projector(patch_tokens, camera_tokens)
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

        # ----- VGGT-Perceiver scatter ---------------------------------- #
        # Only runs when input_ids actually contains placeholder tokens. On
        # decode steps the new token is not a placeholder, so this is a cheap
        # no-op (no VGGT / Perceiver call).
        if (
            input_ids is not None
            and self.vggt_projector is not None
            and self.frame_placeholder_token_id is not None
        ):
            has_frame = (input_ids == self.frame_placeholder_token_id).any()
            has_camera = (input_ids == self.camera_placeholder_token_id).any()
            if has_frame or has_camera:
                if raw_videos is None:
                    raise ValueError(
                        "Input contains VGGT placeholder tokens but `raw_videos` "
                        "was not provided to the model."
                    )
                frame_latents, camera_latents = self._extract_vggt_latents(
                    raw_videos, dtype=inputs_embeds.dtype, device=inputs_embeds.device
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

    def initialize_vggt(
        self,
        checkpoint_path: str,
        tokenizer,
        vggt_embed_dim: int,
        frame_num_latents: int,
        camera_num_latents: int,
        frame_placeholder_token: str,
        camera_placeholder_token: str,
        frame_widening_factor: int = 1,
    ) -> None:
        self.model.initialize_vggt(
            checkpoint_path,
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


if __name__ == "__main__":
    import argparse

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    _DEFAULT_VGGT_CKPT = (
        os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
        "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
    )

    parser = argparse.ArgumentParser(
        description="Smoke test model.generate with VGGT-Perceiver placeholder scatter."
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vggt_ckpt", type=str, default=_DEFAULT_VGGT_CKPT)
    parser.add_argument("--vggt_embed_dim", type=int, default=2048)
    parser.add_argument("--frame_num_latents", type=int, default=128)
    parser.add_argument("--camera_num_latents", type=int, default=32)
    parser.add_argument("--frame_placeholder_token", type=str, default="<|quad_start|>")
    parser.add_argument("--camera_placeholder_token", type=str, default="<|quad_end|>")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="What's in this video?")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto"
    ).to(args.device)
    model.initialize_vggt(
        args.vggt_ckpt,
        tokenizer=processor.tokenizer,
        vggt_embed_dim=args.vggt_embed_dim,
        frame_num_latents=args.frame_num_latents,
        camera_num_latents=args.camera_num_latents,
        frame_placeholder_token=args.frame_placeholder_token,
        camera_placeholder_token=args.camera_placeholder_token,
    )
    model.eval()

    frame_placeholder_text = args.frame_placeholder_token * args.frame_num_latents
    camera_placeholder_text = args.camera_placeholder_token * args.camera_num_latents
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
                {"type": "text", "text": "This is the 3D scene context\n"},
                {"type": "video", "video": args.video_path, "fps": 1.0, "max_frames": 32},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _, video_inputs, processed_video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
        image_patch_size=16,
    )
    if not video_inputs:
        raise RuntimeError("No video inputs extracted from the provided --video_path.")

    video_tensors, video_metadata = zip(*video_inputs)
    model_inputs = processor(
        text=[prompt_text],
        videos=list(video_tensors),
        video_metadata=list(video_metadata),
        return_tensors="pt",
        padding=True,
        **processed_video_kwargs,
    ).to(args.device)

    model_inputs["raw_videos"] = [t.to(args.device) for t in video_tensors]
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(output_text)
