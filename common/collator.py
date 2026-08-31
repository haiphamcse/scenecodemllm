"""Qwen-VL chat messages and supervised collator with prompt masking."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index


def resolve_video_path(data_root: Path, video_field: str) -> str:
    if os.path.isabs(video_field):
        return video_field
    return str(data_root / video_field)


def training_messages(
    example: Dict[str, Any],
    data_root: Path,
    video_fps: float,
    video_max_frames: int,
) -> List[Dict[str, Any]]:
    video_path = resolve_video_path(data_root, example["video"])
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": video_fps,
                    "max_frames": video_max_frames,
                },
                {"type": "text", "text": example["conversations"][0]["value"]},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["conversations"][1]["value"]}],
        },
    ]


def user_only_messages(
    example: Dict[str, Any],
    data_root: Path,
    video_fps: float,
    video_max_frames: int,
) -> List[Dict[str, Any]]:
    video_path = resolve_video_path(data_root, example["video"])
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": video_fps,
                    "max_frames": video_max_frames,
                },
                {"type": "text", "text": example["conversations"][0]["value"]},
            ],
        }
    ]


def make_collator(
    processor: AutoProcessor,
    data_root: Path,
    video_fps: float = 1.0,
    video_max_frames: int = 32,
    image_patch_size: int = 16,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Mask prompt tokens; supervise assistant spans only."""

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        prompt_texts: List[str] = []
        all_video_tensors: List[Any] = []
        all_video_metadata: List[Any] = []
        merged_video_kwargs: Dict[str, Any] = {}

        for example in examples:
            full_messages = training_messages(
                example, data_root, video_fps, video_max_frames
            )
            full_text = processor.apply_chat_template(full_messages, tokenize=False)
            prompt_text = processor.apply_chat_template(
                [full_messages[0]], tokenize=False, add_generation_prompt=True
            )

            _, video_inputs, processed_video_kwargs = process_vision_info(
                full_messages,
                return_video_kwargs=True,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
            )
            if not video_inputs:
                raise ValueError("process_vision_info returned no video_inputs for an example.")

            video_tensors, video_metadata = zip(*video_inputs)
            all_video_tensors.extend(video_tensors)
            all_video_metadata.extend(video_metadata)
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

            for key, value in processed_video_kwargs.items():
                if key not in merged_video_kwargs:
                    merged_video_kwargs[key] = value

        batch = processor(
            text=full_texts,
            videos=all_video_tensors,
            video_metadata=all_video_metadata,
            return_tensors="pt",
            padding=True,
            **merged_video_kwargs,
        )

        # `tokenizer(prompt_text)` underestimates the prompt length because the
        # plain tokenizer leaves `<|video_pad|>` as a single placeholder, while the
        # processor expands it to hundreds/thousands of tokens to match the video
        # grid. Video placeholders only occur in the user/prompt span though, so
        # the assistant-turn suffix has identical token length before and after
        # expansion: subtracting that (unexpanded) suffix length from the actual
        # expanded sequence length yields the true, expanded prompt length.
        prompt_tok = tokenizer(
            prompt_texts, padding=True, return_tensors="pt", return_attention_mask=True,
        )
        full_tok = tokenizer(
            full_texts, padding=True, return_tensors="pt", return_attention_mask=True,
        )
        prompt_unexpanded_lens = prompt_tok["attention_mask"].sum(dim=1).tolist()
        full_unexpanded_lens = full_tok["attention_mask"].sum(dim=1).tolist()
        assistant_suffix_lens = [
            f - p for f, p in zip(full_unexpanded_lens, prompt_unexpanded_lens)
        ]

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        expanded_lens = attention_mask.sum(dim=1).tolist()

        labels = input_ids.clone()
        batch_size, seq_len = labels.shape

        for i in range(batch_size):
            prompt_len = expanded_lens[i] - assistant_suffix_lens[i]
            prompt_len = max(0, min(prompt_len, seq_len))
            labels[i, :prompt_len] = IGNORE_INDEX

        if pad_token_id is not None:
            labels[input_ids == pad_token_id] = IGNORE_INDEX
        labels[attention_mask == 0] = IGNORE_INDEX

        batch["labels"] = labels
        batch["raw_videos"] = all_video_tensors
        return batch

    return collate_fn
