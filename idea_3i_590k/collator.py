"""Qwen-VL supervised collator (idea_3i_590k) decoding videos live.

Fork of idea_3i/collator.py with the frame cache removed. idea_3i pointed the
chat message at pre-extracted uint8 frames so ``fetch_video`` took its list
branch and never decoded; that made it fast but limited it to whatever had been
cached -- in practice the 856 scannetppv2 videos. This version points
``fetch_video`` at the video path, so any of VSI-590K's 374148 video rows works
with nothing to pre-build.

Two consequences of dropping the cache:

- **Decoding moves into the dataloader.** ``--dataloader_num_workers`` now
  carries real work, and step time rises accordingly.
- **The metadata fix-up is gone.** idea_3i had to overwrite ``fetch_video``'s
  fabricated list-branch metadata (``frames_indices = range(n)``) with the real
  decode metadata from the cache sidecar, so timestamps matched. The decode path
  produces real metadata directly, so there is nothing to correct.

Video rows only. VSI-590K also carries 216519 image rows (hypersim,
ytb_roomtour, robotics) keyed ``image`` rather than ``video``; those are
excluded upstream in train.py, and this collator raises rather than guessing if
one reaches it.
"""

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


def example_video_path(example: Dict[str, Any], data_root: Path) -> str:
    """Absolute path of an example's video.

    Raises on an image row rather than silently skipping it: the whole point of
    filtering upstream is that nothing should arrive here without a video, and a
    quiet skip is how this codebase has lost rows before.
    """
    if not example.get("video"):
        raise KeyError(
            f"row has no 'video' key (image row?): keys={sorted(example)}. "
            "idea_3i_590k trains on video rows only; filter upstream."
        )
    return resolve_video_path(data_root, example["video"])


def _video_content(video_path: str, video_fps: float, video_max_frames: int) -> Dict[str, Any]:
    return {
        "type": "video",
        "video": video_path,  # path -> fetch_video decodes
        "fps": video_fps,
        "max_frames": video_max_frames,
        # fetch_video's default is VIDEO_MAX_TOKEN_NUM(768) * (16*2)^2 = 786432 px/frame,
        # i.e. 24576 visual tokens over 32 frames -- the tensor that OOM'd every run on
        # 2026-09-04 (feature_extraction_utils.py maybe_to). SpatialStack's per-frame
        # budget instead (src/qwen_vl/train/argument.py:36-37).
        # min_pixels must be set too: the default min (131072) exceeds this max and
        # smart_resize asserts max >= min.
        # NOTE: unlike idea_3i_590k_joint, raw_videos IS this tensor, so VGGT sees this
        # resolution too. There is deliberately no separate VGGT resize here.
        "min_pixels": 128 * 32 * 32,
        "max_pixels": 256 * 32 * 32,
    }


def _user_content(
    video_path: str,
    example: Dict[str, Any],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    video_fps: float,
    video_max_frames: int,
) -> List[Dict[str, Any]]:
    # Prefix the user turn with the VGGT placeholder blocks (frame then camera);
    # model_with_vggt.py scatters the Perceiver latents into these positions.
    return [
        {"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
        {"type": "text", "text": "This is the 3D scene context\n"},
        _video_content(video_path, video_fps, video_max_frames),
        {"type": "text", "text": example["conversations"][0]["value"]},
    ]


def training_messages(
    video_path: str,
    example: Dict[str, Any],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    video_fps: float,
    video_max_frames: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": _user_content(
                video_path, example, frame_placeholder_text, camera_placeholder_text,
                video_fps, video_max_frames,
            ),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["conversations"][1]["value"]}],
        },
    ]


def user_only_messages(
    video_path: str,
    example: Dict[str, Any],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    video_fps: float,
    video_max_frames: int,
) -> List[Dict[str, Any]]:
    """Prompt half only -- used by the eval callback and eval.py for generation."""
    return [
        {
            "role": "user",
            "content": _user_content(
                video_path, example, frame_placeholder_text, camera_placeholder_text,
                video_fps, video_max_frames,
            ),
        }
    ]


def make_collator(
    processor: AutoProcessor,
    data_root: Path,
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    video_fps: float = 1.0,
    video_max_frames: int = 32,
    image_patch_size: int = 16,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Mask prompt tokens; supervise assistant spans only.

    ``video_fps``/``video_max_frames`` control frame sampling at decode time.
    They must match whatever eval uses, or train and eval see different frame
    counts for the same video.
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
            video_path = example_video_path(example, data_root)
            full_messages = training_messages(
                video_path, example, frame_placeholder_text, camera_placeholder_text,
                video_fps, video_max_frames,
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
                raise ValueError(
                    f"process_vision_info returned no video_inputs for {video_path}. "
                    "Unreadable or corrupt video?"
                )

            # Decode metadata is already real here, unlike the cached list branch.
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


if __name__ == "__main__":
    # Smoke run: collate a handful of rows per video source through the real
    # processor, decoding live. No cache, so the only way a row fails is an
    # unreadable video -- which is what this is checking for.
    import sys
    import time

    _FINETUNING_ROOT = Path(__file__).resolve().parent.parent
    _COMMON = _FINETUNING_ROOT / "common"
    _THIS_DIR = Path(__file__).resolve().parent
    for _p in (_THIS_DIR, _COMMON):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    from data import load_jsonl

    model_path = "Qwen/Qwen3-VL-2B-Instruct"
    jsonl_path = os.environ.get(
        "SR_JSONL",
        "/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl",
    )
    data_root = Path(os.environ.get(
        "SR_DATA_ROOT", "/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
    ))
    hf_home = os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    os.environ.setdefault("HF_HOME", hf_home)
    PER_SOURCE = int(os.environ.get("SMOKE_PER_SOURCE", "2"))

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    collate_fn = make_collator(
        processor,
        data_root=data_root,
        frame_placeholder_text="<|quad_start|>" * 128,
        camera_placeholder_text="<|quad_end|>" * 32,
    )

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for record in load_jsonl(Path(jsonl_path)):
        video = record.get("video")
        if not video:
            continue                       # image row -- excluded by design
        if not (data_root / video).exists():
            continue                       # source not downloaded on this machine
        rows = by_source.setdefault(video.split("/")[0], [])
        if len(rows) < PER_SOURCE:
            rows.append(record)

    print(f"sources with data on disk: {sorted(by_source)}\n")
    failures = 0
    for source, rows in sorted(by_source.items()):
        for record in rows:
            start = time.time()
            try:
                batch = collate_fn([record])
                supervised = int((batch["labels"] != IGNORE_INDEX).sum())
                print(
                    f"  {source:14} ok  seq={batch['input_ids'].shape[1]:>6} "
                    f"supervised={supervised:>3}  frames={batch['raw_videos'][0].shape[0]:>3} "
                    f"{time.time() - start:.1f}s  {record['video']}"
                )
                if supervised == 0:
                    print(f"  {source:14} WARNING: nothing supervised in this row")
                    failures += 1
            except Exception as exc:                       # noqa: BLE001
                failures += 1
                print(f"  {source:14} FAIL {type(exc).__name__}: {str(exc)[:90]}")

    print(f"\n{'SMOKE FAILED' if failures else 'SMOKE OK'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
