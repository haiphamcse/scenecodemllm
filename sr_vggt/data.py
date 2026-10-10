"""Training data: mix rows, det_es frames, prompts, and the collator.

Qwen sees each clip's det_es JPEGs (``<video without .mp4>/det_es/frames/frameNN.jpg``, stored at
the 128 tok/frame budget, so the processor never resizes them). The VGGT branch reads the same
clip's tokens from the cache (vggt_cache); the architecture decides how they enter the batch
(``batch_vggt`` in model.py / ablations).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers.trainer_pt_utils import LabelSmoother

from . import vggt_cache

IGNORE_INDEX = LabelSmoother.ignore_index

# 128 tok/frame cap (96 actual on 4:3 -> 384x256). min_pixels must be set too: the default
# min exceeds this max and smart_resize asserts max >= min.
LLM_MIN_PIXELS = 64 * 32 * 32
LLM_MAX_PIXELS = 128 * 32 * 32

_SHORT_SYSTEM = ("You are a spatial reasoning assistant. You are shown 3D scene context and the "
                 "RGB frames of a scene. Answer the question about the scene.")
_TEXT256_SYSTEM = ("You are a spatial reasoning assistant. You are given three inputs about one "
                   "indoor scene: (1) per-frame geometry features, compressed from a 3D "
                   "reconstruction model's patch tokens of every video frame; (2) camera and scene "
                   "features, compressed from the same model's per-frame camera tokens, which encode "
                   "where each frame was taken; (3) the RGB video frames. Answer the question about "
                   "the scene.")

# --prompt: the system prompt + how the VGGT placeholders are laid out before the video.
#   clean    bare placeholders + "This is the 3D scene context." (main, lora128)
#   text256  named placeholder blocks (text256, 4b, lat3072, vggt1b)
#   plain    no placeholders (spatialstack, vlm3r: geometry enters through the video tokens)
#   reason   text256 + reasoning-format instructions (extensions/reasoning.py)
PROMPTS = ("text256", "clean", "plain", "reason")


def system_prompt(prompt: str) -> str:
    if prompt == "reason":
        from .extensions.reasoning import SYSTEM_PROMPT
        return SYSTEM_PROMPT
    return _TEXT256_SYSTEM if prompt == "text256" else _SHORT_SYSTEM


def load_mix(path) -> List[Dict[str, Any]]:
    """The mix jsonl as-is (already filtered, budgeted and shuffled). VQA rows only."""
    rows = [json.loads(line) for line in open(path)]
    if not rows:
        raise ValueError(f"{path} is empty.")
    bad = {r.get("task") for r in rows} - {"vqa"}
    if bad:
        raise ValueError(f"{path} has rows with task {bad}; only vqa rows are supported.")
    for r in rows:
        r.pop("meta", None)  # the reasoning traces' provenance block, not a training field
    return rows


def det_es_dir(video_path: str) -> Path:
    return Path(video_path).with_suffix("") / "det_es" / "frames"


def det_es_frames(video_path: str) -> List[Image.Image]:
    """All det_es frames of one clip (64 for most clips)."""
    files = sorted(det_es_dir(video_path).glob("frame*.jpg"))
    if not files:
        raise FileNotFoundError(f"no det_es frames for {video_path}: {det_es_dir(video_path)}")
    return [Image.open(f).convert("RGB") for f in files]


def video_path(example, data_root=None) -> str:
    path = example["video"]
    if data_root is not None and not os.path.isabs(path):
        path = os.path.join(str(data_root), path)
    return path


def task_messages(example, frames, prompt: str, frame_placeholder_text: str,
                  camera_placeholder_text: str, target_text: str, prompt_only: bool = False):
    """System + user (+ assistant unless prompt_only)."""
    video = {"type": "video", "video": frames,
             "min_pixels": LLM_MIN_PIXELS, "max_pixels": LLM_MAX_PIXELS}
    if prompt in ("text256", "reason"):
        head = [{"type": "text", "text": "Per-frame geometry features: " + frame_placeholder_text
                 + "\nCamera and scene features: " + camera_placeholder_text + "\n"}]
    elif prompt == "clean":
        head = [{"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
                {"type": "text", "text": "This is the 3D scene context.\n"}]
    elif prompt == "plain":
        head = [{"type": "text", "text": "This is the 3D scene context.\n"}]
    else:
        raise ValueError(f"unknown prompt {prompt!r}; one of {PROMPTS}")
    user = head + [video, {"type": "text", "text": example["conversations"][0]["value"]}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt(prompt)}]},
        {"role": "user", "content": user},
    ]
    if not prompt_only:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
    return messages


def make_collator(
    processor,
    data_root,
    prompt: str,
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    vggt_cache_root,
    cache_params: Dict[str, Any],
    batch_vggt: Callable[[List[Dict[str, torch.Tensor]], Dict[str, Any]], Dict[str, Any]],
    image_patch_size: int = 16,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """One supervised span (the assistant answer) per row; VGGT tokens from the cache."""
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id

    def _unexpanded_len(text: str) -> int:
        return len(tokenizer(text, return_attention_mask=False)["input_ids"])

    resize_checked = [False]

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        cached: List[Dict[str, torch.Tensor]] = []
        llm_video_tensors: List[Any] = []
        llm_video_metadata: List[Any] = []
        merged_video_kwargs: Dict[str, Any] = {}
        spans: List[tuple] = []

        for example in examples:
            frames = det_es_frames(video_path(example, data_root))
            messages = task_messages(example, frames, prompt, frame_placeholder_text,
                                     camera_placeholder_text, example["conversations"][1]["value"])
            t1 = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            t2 = processor.apply_chat_template(messages, tokenize=False)
            spans.append((_unexpanded_len(t1), _unexpanded_len(t2)))
            full_texts.append(t2)

            cached.append(vggt_cache.load(
                vggt_cache_root, vggt_cache.clip_key(example, data_root), cache_params))

            _, llm_videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, image_patch_size=image_patch_size,
                return_video_metadata=True)
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

        batch = processor(text=full_texts, videos=llm_video_tensors, video_metadata=llm_video_metadata,
                          return_tensors="pt", padding=True, **merged_video_kwargs)

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
        batch.update(batch_vggt(cached, batch))
        return batch

    return collate_fn
