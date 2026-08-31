"""Qwen-VL supervised collator (idea_3i) reading pre-extracted frame caches.

Fork of idea_3d/collator.py. Same prompt-masking logic, but instead of pointing
the chat message at a video *path* (which triggers a decode + resize on every
epoch), it loads the pre-extracted resized uint8 frames from the cache (see
idea_3i/pre_extract_videos.py) and feeds them as a ``list[PIL.Image]``. That
routes ``fetch_video`` through its list branch with no video decoding.

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


def _user_content(
    frames,
    meta: Dict[str, Any],
    example: Dict[str, Any],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
) -> List[Dict[str, Any]]:
    # Prefix the user turn with the VGGT placeholder blocks (frame then camera);
    # model_with_vggt.py scatters the Perceiver latents into these positions.
    return [
        {"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
        {"type": "text", "text": "This is the 3D scene context\n"},
        _video_content(frames, meta),
        {"type": "text", "text": example["conversations"][0]["value"]},
    ]


def training_messages(
    frames,
    meta: Dict[str, Any],
    example: Dict[str, Any],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": _user_content(
                frames, meta, example, frame_placeholder_text, camera_placeholder_text
            ),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["conversations"][1]["value"]}],
        },
    ]


def user_only_messages(
    frames,
    meta: Dict[str, Any],
    example: Dict[str, Any],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": _user_content(
                frames, meta, example, frame_placeholder_text, camera_placeholder_text
            ),
        }
    ]


def make_collator(
    processor: AutoProcessor,
    data_root: Path,
    cache_root: Path,
    frame_placeholder_text: str,
    camera_placeholder_text: str,
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
            full_messages = training_messages(
                frames, meta, example, frame_placeholder_text, camera_placeholder_text
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


if __name__ == "__main__":
    # Smoke run: collate every *cached* train and validation row through the real
    # processor, skipping (and counting) rows whose frame cache is missing. Also
    # checks that the first OVERFIT_NUM_SAMPLES val rows -- the set idea_3i's
    # overfit job trains on -- are fully cached, since the training collator hard
    # -fails on a missing cache.
    import sys

    _FINETUNING_ROOT = Path(__file__).resolve().parent.parent
    _COMMON = _FINETUNING_ROOT / "common"
    _THIS_DIR = Path(__file__).resolve().parent
    for _p in (_THIS_DIR, _COMMON):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    from data import build_train_dataset, build_vsibench_eval_dataset
    from tqdm import tqdm

    model_path = "Qwen/Qwen3-VL-2B-Instruct"
    jsonl_path = os.environ.get("SR_JSONL", "/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl")
    data_root = Path(os.environ.get("SR_DATA_ROOT", "/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"))
    cache_root = Path(os.environ.get("SR_CACHE_ROOT", "/home/ducpham/scratch/Working/dataset/vsi_590k/vsi_590k_frame_cache"))
    hf_home = os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    frame_placeholder_text = "<|quad_start|>" * 128
    camera_placeholder_text = "<|quad_end|>" * 32
    OVERFIT_NUM_SAMPLES = 20

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    collate_fn = make_collator(
        processor,
        data_root=data_root,
        cache_root=cache_root,
        frame_placeholder_text=frame_placeholder_text,
        camera_placeholder_text=camera_placeholder_text,
    )

    def has_cache(example) -> bool:
        video_abs = resolve_video_path(data_root, example["video"])
        npy_path, json_path = cache_paths_for(video_abs, cache_root)
        return npy_path.exists() and json_path.exists()

    def smoke_split(name, dataset) -> int:
        # VSI-590K has many Q/A per video; collating one row per unique video
        # exercises every cached video without redundant re-collation.
        collated = skipped = dup = 0
        seen: set[str] = set()
        for example in tqdm(dataset, desc=f"smoke {name}"):
            video_abs = resolve_video_path(data_root, example["video"])
            if video_abs in seen:
                dup += 1
                continue
            seen.add(video_abs)
            if not has_cache(example):
                skipped += 1
                continue
            collate_fn([example])  # raises on any genuine collation failure
            collated += 1
        print(
            f"[{name}] collated={collated} unique videos  skipped(no cache)={skipped}  "
            f"dup rows(same video)={dup}  total={len(dataset)}"
        )
        return collated

    print("building datasets ...")
    train_dataset = build_train_dataset(jsonl_path)
    val_dataset = build_vsibench_eval_dataset(hf_home=hf_home)

    # Precondition for the overfit job: the first N val rows must all be cached.
    overfit_missing = [
        i for i in range(min(OVERFIT_NUM_SAMPLES, len(val_dataset)))
        if not has_cache(val_dataset[i])
    ]

    train_ok = smoke_split("train", train_dataset)
    val_ok = smoke_split("val", val_dataset)

    print(
        f"overfit precondition: first {OVERFIT_NUM_SAMPLES} val rows -> "
        f"{'ALL CACHED' if not overfit_missing else f'MISSING {overfit_missing}'}"
    )
    if train_ok == 0 or val_ok == 0:
        raise SystemExit(f"SMOKE FAILED: train collated={train_ok}, val collated={val_ok}")
    if overfit_missing:
        raise SystemExit(
            f"SMOKE FAILED: overfit rows missing cache {overfit_missing}; the training "
            f"collator will crash on these. Run pre_extract_videos.py for them first."
        )
    print(f"SMOKE OK: train collated={train_ok}, val collated={val_ok}")
