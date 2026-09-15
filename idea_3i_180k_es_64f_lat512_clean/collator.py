"""VQA collator: det_es frames for Qwen, cached VGGT features for the Perceiver branch.

Each row's frames are the clip's det_es JPEGs (<video without .mp4>/det_es/frames/frameNN.jpg),
never decoded video. ``batch["vggt_patch_tokens"]`` / ``["vggt_camera_tokens"]`` are read from
vggt_cache (exported from the same JPEGs by export_vggt_features.py); a miss raises.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

import vggt_cache
from transformers import AutoProcessor
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index

# Default VGGT balanced-resize target. Part of the cache key; the 256 runs pass it explicitly.
VGGT_IMAGE_RESOLUTION = 512

SYSTEM_PROMPT = ("You are a spatial reasoning assistant. You are shown 3D scene context and the "
                 "RGB frames of a scene. Answer the question about the scene.")

# 128 tok/frame cap (96 actual on 4:3 -> 384x256). min_pixels must be set too: the default
# min exceeds this max and smart_resize asserts max >= min.
_LLM_MIN_PIXELS = 64 * 32 * 32
_LLM_MAX_PIXELS = 128 * 32 * 32


def det_es_frames(video_path: str) -> List[Image.Image]:
    """The det_es frames of one VSI-590K clip: <root>/<source>/<scene>/det_es/frames/frameNN.jpg.

    T = files present (64 for most clips). Already at the LLM budget size, so the processor's
    smart_resize is a no-op (asserted once in the collator).
    """
    d = Path(video_path).with_suffix("") / "det_es" / "frames"
    files = sorted(d.glob("frame*.jpg"))
    if not files:
        raise FileNotFoundError(f"no det_es frames for {video_path}: {d}")
    return [Image.open(f).convert("RGB") for f in files]


def load_frames(example, data_root=None) -> List[Image.Image]:
    path = example["video"]
    if data_root is not None and not os.path.isabs(path):
        path = os.path.join(str(data_root), path)
    return det_es_frames(path)


def task_messages(example, frames, frame_placeholder_text, camera_placeholder_text,
                  target_text: str, prompt_only: bool = False):
    """System + user (+ assistant unless prompt_only). VGGT placeholders precede the video;
    model_with_vggt.py scatters the Perceiver latents into them."""
    user = [
        {"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
        {"type": "text", "text": "This is the 3D scene context.\n"},
        {"type": "video", "video": frames,
         "min_pixels": _LLM_MIN_PIXELS, "max_pixels": _LLM_MAX_PIXELS},
        {"type": "text", "text": example["conversations"][0]["value"]},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user},
    ]
    if not prompt_only:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]})
    return messages


def make_collator(
    processor: AutoProcessor,
    data_root: Path,
    vggt_cache_root: Path,
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    vggt_checkpoint: str,
    image_patch_size: int = 16,
    vggt_image_resolution: int = VGGT_IMAGE_RESOLUTION,
    video_fps: float = 1.0,
    video_max_frames: int = 32,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """One supervised assistant span (the answer) per row."""
    cache_params = vggt_cache.params_from(
        vggt_image_resolution, video_fps, video_max_frames, vggt_checkpoint
    )
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id

    def _unexpanded_len(text: str) -> int:
        return len(tokenizer(text, return_attention_mask=False)["input_ids"])

    resize_checked = [False]

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        patch_tokens: List[Any] = []
        camera_tokens: List[Any] = []
        llm_video_tensors: List[Any] = []
        llm_video_metadata: List[Any] = []
        merged_video_kwargs: Dict[str, Any] = {}
        spans: List[tuple] = []

        for example in examples:
            frames = load_frames(example, data_root)
            messages = task_messages(
                example, frames, frame_placeholder_text, camera_placeholder_text,
                example["conversations"][1]["value"],
            )
            t1 = processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)
            t2 = processor.apply_chat_template(messages, tokenize=False)
            spans.append((_unexpanded_len(t1), _unexpanded_len(t2)))
            full_texts.append(t2)

            patch, camera = vggt_cache.load(
                vggt_cache_root, vggt_cache.clip_key(example, data_root), cache_params)
            patch_tokens.append(patch)
            camera_tokens.append(camera)

            _, llm_videos, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
            )
            if not llm_videos:
                raise ValueError("process_vision_info returned no video for the MLLM branch.")
            tensors, metadata = zip(*llm_videos)
            if not resize_checked[0]:
                # det_es JPEGs are stored at the LLM budget: the processor must not resize them.
                assert tuple(tensors[0].shape[-2:]) == frames[0].size[::-1], (tensors[0].shape, frames[0].size)
                resize_checked[0] = True
            llm_video_tensors.extend(tensors)
            llm_video_metadata.extend(metadata)
            for key, value in video_kwargs.items():
                merged_video_kwargs.setdefault(key, value)

        batch = processor(
            text=full_texts,
            videos=llm_video_tensors,
            video_metadata=llm_video_metadata,
            return_tensors="pt",
            padding=True,
            **merged_video_kwargs,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        expanded_lens = attention_mask.sum(dim=1).tolist()

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        _, seq_len = input_ids.shape

        for i, (start, end) in enumerate(spans):
            # delta = tokens the processor added over the plain tokenizer count (<|video_pad|>
            # expansion). Correct only because all expansion sits in the user turn, ahead of
            # the answer span, and padding is on the right.
            delta = expanded_lens[i] - end
            g0 = max(0, min(start + delta, seq_len))
            g1 = max(0, min(end + delta, seq_len))
            if g1 > g0:
                labels[i, g0:g1] = input_ids[i, g0:g1]

        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        batch["labels"] = labels
        batch["vggt_patch_tokens"] = patch_tokens
        batch["vggt_camera_tokens"] = camera_tokens
        return batch

    return collate_fn
