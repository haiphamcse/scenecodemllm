"""Qwen3-VL with CACHED VGGTOmega features compressed by two Perceiver IO encoders and
scattered into placeholder tokens.

VGGT-Omega itself is not imported, loaded, or run anywhere in this module. Its output is
exported once by export_vggt_features.py and arrives here as two tensors per clip; there
is no code path that falls back to computing them. That is the whole point of this fork:
the frozen encoder was ~1 B parameters of GPU memory and the largest fixed cost per step,
for a value that never changes.

Architecture
------------
- Qwen3-VL encodes the video as usual (``pixel_values_videos`` -> video tokens).
- The collator loads this clip's cached VGGTOmega tokens: ``vggt_patch_tokens``
  ``[T, Np, vggt_dim]`` and ``vggt_camera_tokens`` ``[T, 17, vggt_dim]``. In the
  original encoder output those were one ``[T, 17 + Np, vggt_dim]`` block -- token 0 the
  camera token, tokens 1..16 the register tokens, tokens 17.. the patch tokens -- and
  the exporter splits them at the aggregator's own ``patch_token_start``.
- Two Perceiver IO encoders compress those into a fixed latent budget:
    * ``frame_encoder``  : VGGT patch tokens ``[1, T*Np, vggt_dim]`` -> ``[1, L, hidden]``,
      applied once per cached VGGT depth (multi-scale, Map-Det3D style) with the latents
      concatenated -> ``[1, S*L, hidden]``. The encoder weights are SHARED across scales.
    * ``camera_encoder`` : VGGT camera+register ``[1, T*17, vggt_dim]`` -> ``[1, M, hidden]``
  ``num_latent_channels`` equals the Qwen text hidden size, so latents need no
  extra projection before scatter.
- The chat template inserts ``L`` ``<|quad_start|>`` + ``M`` ``<|quad_end|>``
  placeholder tokens (see collator.py; counts/tokens are train.py args). ``forward`` scatters
  the frame latents into ``<|quad_start|>`` positions and the camera latents into
  ``<|quad_end|>`` positions, exactly as Qwen3-VL scatters its own video features
  -- so HF generation handles cache length / M-RoPE / attention mask with no extra
  plumbing.
- Only the two Perceiver encoders (``vggt_projector``) train.
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
        patch_tokens_per_scale: list[torch.Tensor],
        camera_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``S x [B, T*Np, vggt_dim]``, ``[B, T*17, vggt_dim]`` -> ``[B, S*L, hidden]``, ``[B, M, hidden]``.

        One shared frame_encoder is applied to each VGGT depth in turn and the latents are
        concatenated, so the scales are told apart by content alone -- no per-scale weights
        and no scale embedding.
        """
        frame_latents = [self.frame_encoder(tokens) for tokens in patch_tokens_per_scale]
        return torch.cat(frame_latents, dim=1), self.camera_encoder(camera_tokens)


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
        """Build the two Perceivers and resolve the placeholder ids. No encoder to load.

        Was ``initialize_vggt``, which also loaded a ~1 B-parameter VGGTOmega. The
        rename is deliberate: a caller still passing a checkpoint path should fail
        loudly rather than have it silently ignored.
        """
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
        """Cached tokens -> ``(frame_latents [B, S*L, hidden], camera_latents [B, M, hidden])``.

        Per clip the inputs are ``S x [T, Np, d]`` and ``[T, 17, d]`` straight off disk (fp32,
        see vggt_cache). Flattening T into the token axis and casting to the compute dtype
        here reproduces exactly what the live-encoder version fed the Perceivers.
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
        for patches, camera in zip(patch_tokens_per_clip, camera_tokens_per_clip):
            if any(p.ndim != 3 for p in patches) or camera.ndim != 3:
                raise ValueError(
                    f"Expected cached tokens S x [T, N, d]; got patch {[tuple(p.shape) for p in patches]} "
                    f"camera {tuple(camera.shape)}."
                )
            patches = [p.to(device=device, dtype=dtype).flatten(0, 1).unsqueeze(0) for p in patches]  # S x [1, T*Np, d]
            camera = camera.to(device=device, dtype=dtype).flatten(0, 1).unsqueeze(0)  # [1, T*17, d]
            frame_lat, camera_lat = self.vggt_projector(patches, camera)
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
        # Only runs when input_ids actually contains placeholder tokens. On
        # decode steps the new token is not a placeholder, so this is a cheap
        # no-op (no Perceiver call).
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
                        "Input contains VGGT placeholder tokens but cached features were "
                        "not provided. This fork never computes them: pass "
                        "`vggt_patch_tokens` and `vggt_camera_tokens` from the cache "
                        "(see vggt_cache.load)."
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

        # Only the scene-graph span is supervised; the collator masks everything
        # else to -100 in `labels`. Standard causal-LM CE over that single span.
        #
        # The full logits tensor [seq, 151936] in fp32 is what OOM'd every V100 run
        # (4.70 GiB at seq ~8300, plus its gradient). Liger's fused linear CE chunks
        # lm_head and the loss together and never materialises it. Liger's monkey-patch
        # cannot reach this method -- it patches the parent class and this override
        # shadows it -- so the fused path is called here, mirroring
        # liger_kernel/transformers/model/qwen3_vl.py::lce_forward.
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


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vggt_cache

    parser = argparse.ArgumentParser(
        description="Smoke test model.generate with cached VGGT-Perceiver placeholder scatter."
    )
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vggt_cache_root", type=str, required=True)
    parser.add_argument("--vggt_image_resolution", type=int, default=512)
    parser.add_argument("--vggt_checkpoint", type=str, default="")
    parser.add_argument("--vggt_embed_dim", type=int, default=2048)
    parser.add_argument("--frame_num_latents", type=int, default=128)
    parser.add_argument("--frame_num_scales", type=int, default=4)
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
    model.initialize_projector(
        tokenizer=processor.tokenizer,
        vggt_embed_dim=args.vggt_embed_dim,
        frame_num_latents=args.frame_num_latents,
        camera_num_latents=args.camera_num_latents,
        frame_placeholder_token=args.frame_placeholder_token,
        camera_placeholder_token=args.camera_placeholder_token,
    )
    model.eval()

    frame_placeholder_text = args.frame_placeholder_token * (
        args.frame_num_latents * args.frame_num_scales
    )
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

    # The cache is the only source of VGGT features here; a miss raises rather than
    # falling back to an encoder this module no longer imports.
    patch, camera = vggt_cache.load(
        args.vggt_cache_root,
        vggt_cache.clip_key({"video": args.video_path}),
        vggt_cache.params_from(args.vggt_image_resolution, 1.0, 32, args.vggt_checkpoint),
    )
    model_inputs["vggt_patch_tokens"] = [[p.to(args.device) for p in patch]]
    model_inputs["vggt_camera_tokens"] = [camera.to(args.device)]
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )

    output_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    print(output_text)
