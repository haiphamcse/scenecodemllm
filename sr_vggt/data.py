"""Training data: mix rows, prompts, and the collator.

Every row names a VSI-590K clip (``video``, relative to SR_VSI590K). The collator decodes it with
the shared sampling rule (encoders.decode_video_frames: 1 fps, at most 64 frames), lets Qwen's
processor resize the frames to the 128 tok/frame budget, and hands the model the clip's VGGT input
(``raw_videos``); the model runs the frozen VGGT itself. eval.py builds its inputs with the same
functions, so a clip looks identical to the model at training and at eval.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Callable, Dict, List

import torch
from qwen_vl_utils import process_vision_info
from transformers.trainer_pt_utils import LabelSmoother

from . import encoders

IGNORE_INDEX = LabelSmoother.ignore_index

# 128 tok/frame cap (96 actual on 4:3). min_pixels must be set too: the default min exceeds this
# max and smart_resize asserts max >= min.
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
    """The mix (.jsonl or .jsonl.gz) as-is: already filtered, budgeted and shuffled. VQA rows only."""
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as f:
        rows = [json.loads(line) for line in f]
    if not rows:
        raise ValueError(f"{path} is empty.")
    bad = {r.get("task") for r in rows} - {"vqa"}
    if bad:
        raise ValueError(f"{path} has rows with task {bad}; only vqa rows are supported.")
    for r in rows:
        r.pop("meta", None)  # the reasoning traces' provenance block, not a training field
    return rows


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


def clip_inputs(va, messages, frames):
    """Qwen video tensors + metadata + kwargs, and the VGGT input, of the one clip in ``messages``."""
    _, vision, video_kwargs = process_vision_info(messages, return_video_kwargs=True,
                                                  image_patch_size=va.image_patch_size,
                                                  return_video_metadata=True)
    if not vision:
        raise ValueError("process_vision_info returned no video.")
    videos, metadata = map(list, zip(*vision))
    raw = encoders.vggt_input(va.vggt_frames, frames, videos[0], va.vggt_image_resolution, va.vggt_backbone)
    return videos, metadata, video_kwargs, raw


def make_collator(processor, data_root, va) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """One supervised span (the assistant answer) per row. ``va``: config.VggtArguments."""
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    frame_ph, camera_ph = va.placeholder_texts()

    def _unexpanded_len(text: str) -> int:
        return len(tokenizer(text, return_attention_mask=False)["input_ids"])

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts, raw_videos, llm_videos, llm_metadata, spans = [], [], [], [], []
        merged_video_kwargs: Dict[str, Any] = {}
        for example in examples:
            frames = encoders.decode_video_frames(video_path(example, data_root), va.video_fps, va.video_max_frames)
            messages = task_messages(example, frames, va.prompt, frame_ph, camera_ph,
                                     example["conversations"][1]["value"])
            t1 = processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            t2 = processor.apply_chat_template(messages, tokenize=False)
            spans.append((_unexpanded_len(t1), _unexpanded_len(t2)))
            full_texts.append(t2)
            videos, metadata, video_kwargs, raw = clip_inputs(va, messages, frames)
            llm_videos.extend(videos)
            llm_metadata.extend(metadata)
            raw_videos.append(raw)
            for key, value in video_kwargs.items():
                merged_video_kwargs.setdefault(key, value)

        batch = processor(text=full_texts, videos=llm_videos, video_metadata=llm_metadata,
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
        batch["raw_videos"] = raw_videos
        return batch

    return collate_fn
