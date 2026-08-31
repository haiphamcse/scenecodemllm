"""Qwen-VL supervised collator (idea_3i_spatialstack) reading pre-extracted frame caches.

Fork of idea_3i/collator.py with the VGGT placeholder blocks removed: layered
geometry fusion injects into the decoder's video-token positions, so the prompt
needs no placeholder tokens at all. Everything else is unchanged -- video frames
come from the pre-extracted cache (idea_3i/pre_extract_videos.py, cache shared)
and are fed as a ``list[PIL.Image]`` so ``fetch_video`` takes its list branch with
no decoding.

The cached ``resized_height``/``resized_width`` are passed in the message so the
list branch's resize is an identity op. The list branch fabricates video
metadata (``frames_indices = range(n)``), which drifts the per-frame timestamps
vs the decode path; since eval still decodes live, the collator overwrites the
fabricated metadata with the *real* decode metadata stored in the cache sidecar,
so ``_calculate_timestamps`` produces timestamps identical to the decode path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.trainer_pt_utils import LabelSmoother

from frame_cache import cache_paths_for, load_cached_frames

IGNORE_INDEX = LabelSmoother.ignore_index


def resolve_video_path(data_root: Path, video_field: str) -> str:
    if os.path.isabs(video_field):
        return video_field
    return str(data_root / video_field)


def load_example_video(
    example: Dict[str, Any],
    data_root: Path,
    cache_root: Path,
):
    """Load cached frames (list[PIL.Image]) + sidecar metadata for an example."""
    video_abs = resolve_video_path(data_root, example["video"])
    npy_path, json_path = cache_paths_for(video_abs, cache_root)
    return load_cached_frames(npy_path, json_path)


def _video_content(frames, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "video",
        "video": frames,  # list[PIL.Image] -> fetch_video list branch (no decode)
        "resized_height": meta["resized_height"],
        "resized_width": meta["resized_width"],
    }


def real_video_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Decode-equivalent metadata to replace the list branch's fabricated one."""
    return {
        "fps": meta["fps"],
        "frames_indices": meta["frames_indices"],
        "total_num_frames": meta["total_num_frames"],
    }


def _user_content(frames, meta: Dict[str, Any], example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        _video_content(frames, meta),
        {"type": "text", "text": example["conversations"][0]["value"]},
    ]


def training_messages(frames, meta: Dict[str, Any], example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": _user_content(frames, meta, example)},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["conversations"][1]["value"]}],
        },
    ]


def user_only_messages(frames, meta: Dict[str, Any], example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": _user_content(frames, meta, example)}]


def make_collator(
    processor: AutoProcessor,
    data_root: Path,
    cache_root: Path,
    image_patch_size: int = 16,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Mask prompt tokens; supervise assistant spans only.

    ``fps``/``max_frames`` are intentionally absent: frame sampling is baked into
    the cache at extraction time. ``image_patch_size`` must match the value used
    by pre_extract_videos.py so the list branch's resize stays an identity op.
    """

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        prompt_texts: List[str] = []
        all_video_tensors: List[Any] = []
        all_video_metadata: List[Any] = []
        merged_video_kwargs: Dict[str, Any] = {}

        for example in examples:
            frames, meta = load_example_video(example, data_root, cache_root)
            full_messages = training_messages(frames, meta, example)
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

            video_tensors, _fabricated_metadata = zip(*video_inputs)
            # One cached video per example; replace fabricated metadata with the
            # real decode metadata so timestamps match the live-decode path.
            all_video_tensors.extend(video_tensors)
            all_video_metadata.extend([real_video_metadata(meta)] * len(video_tensors))
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
