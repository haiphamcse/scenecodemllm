"""Plain Qwen3-VL collator for the 180k VQA mix (idea_3a_180k_cached_64f).

Fork of idea_3i_180k_cached_64f_lat512/collator.py with the VGGT branch removed: no
``vggt_patch_tokens`` / ``vggt_camera_tokens``, no placeholder text in the user turn.
The Qwen frames come from the same frame cache (``llm/`` JPEGs, 64 frames at the same
pixel budget), so the MLLM sees exactly the pixels the VGGT runs saw.

Rows carry a ``task`` field that build_mix.py stamps on:

  task="vqa"  VSI-590K / VLM-3R: ``video`` is a path; the question and answer come from
              the row's own ``conversations``.
  task="det"  ScanNet 3DOD (DEAD here: the 180k mix has no det rows; kept as-is minus the
              placeholder text).

The label masking is the same as the parent fork: a single user turn followed by a
single supervised assistant turn.
"""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from graph_vgllm import parse as _parse_boxes, render as _render_boxes
from transformers import AutoProcessor
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index

# Format exemplars for SCENE_GRAPH_QUESTION. Written to match graph_vgllm.render()
# exactly -- same fence, same tab indent, same count-first / largest-volume-first order --
# so the prompt cannot disagree with the target it is asking for. The numbers are
# hand-written at ScanNet-plausible magnitudes rather than copied from a training row, so
# no real scene is being memorised through the prompt.
_FEWSHOT_A = (
    "```json\n[\n"
    "\t{\"n\": 3},\n"
    "\t{\"label\": \"couch\", \"bbox_3d\": [0.82, -0.15, 2.34, 1.9, 0.88, 0.79, -1.57, 0.12, 1.98]},\n"
    "\t{\"label\": \"table\", \"bbox_3d\": [-0.24, 0.06, 1.71, 1.1, 0.62, 0.41, -1.57, 0.12, 1.98]},\n"
    "\t{\"label\": \"chair\", \"bbox_3d\": [1.36, 0.02, 1.12, 0.55, 0.53, 0.88, -1.57, 0.12, 1.98]}\n"
    "]```")
_FEWSHOT_B = (
    "```json\n[\n"
    "\t{\"n\": 3},\n"
    "\t{\"label\": \"door\", \"bbox_3d\": [-1.44, -0.31, 2.86, 0.09, 0.94, 2.03, -1.31, 0.07, 2.44]},\n"
    "\t{\"label\": \"desk\", \"bbox_3d\": [0.51, 0.22, 1.63, 1.42, 0.7, 0.05, -1.31, 0.07, 2.44]},\n"
    "\t{\"label\": \"monitor\", \"bbox_3d\": [0.44, -0.18, 1.55, 0.55, 0.18, 0.34, -1.31, 0.07, 2.44]}\n"
    "]```")

# VG-LLM's question (process_threedod.py:16-24) minus its <image> prefix, plus the two
# conventions graph_vgllm.canonicalize() imposes on the target: leading count, and
# objects ordered largest-first. Both exist to make sequence length learnable.
SCENE_GRAPH_QUESTION = (
    "Detect the 3D bounding boxes in the camera coordinate system of the first frame.\n"
    "Output a json list whose first entry is {\"n\": N} giving the number of objects, "
    "followed by one entry per object containing the object name in \"label\" and its "
    "3D bounding box in \"bbox_3d\".\n"
    "Order the objects from largest to smallest by volume.\n"
    "The 3D bounding box format should be [x_center, y_center, z_center, x_size, "
    "y_size, z_size, yaw, roll, pitch].\n"
    "Two examples of the required format, abridged to three objects each; the values are "
    "illustrative, not a prior on this scene:\n"
    + _FEWSHOT_A + "\n" + _FEWSHOT_B)


def scene_id_for(example: Dict[str, Any]) -> str:
    """ScanNet scene id of a row (logging / error messages only)."""
    return example.get("scene") or Path(example["images"][0]).parent.name


def load_example_video(example: Dict[str, Any], data_root=None, cache_root=None):
    """Load the row's frames (list[PIL.Image], RGB).

    data_root / cache_root are accepted (callbacks pass them) but unused: the row already
    holds absolute image paths in json order, and there is no npy cache.
    """
    return [Image.open(p).convert("RGB") for p in example["images"]]


TASKS = ("vqa", "det")

# One system turn per objective. No 3D context in this fork: the model only sees frames.
SYSTEM_PROMPTS = {
    "vqa": "You are a spatial reasoning assistant. You are shown the RGB frames of a "
           "scene. Answer the question about the scene.",
    "det": "You are a 3D object detector. You are shown the RGB frames of a scene. "
           "List every object with its 3D bounding box.",
}


def task_of(example: Dict[str, Any]) -> str:
    """Every mixed row is stamped by build_mix.py; refuse to guess if it is not."""
    task = example.get("task")
    if task not in TASKS:
        raise ValueError(f"row has task={task!r}, expected one of {TASKS}. "
                         "Rows must come from build_mix.py.")
    return task


def decode_video_frames(path: str, video_fps: float, video_max_frames: int):
    """VSI-590K video -> list[PIL.Image] at native resolution.

    Frame CHOICE follows fetch_video's rule -- uniform over the clip, ``video_fps``
    sampling capped at ``video_max_frames`` -- so live decode (eval) and the frame cache
    (training) pick the same frames.
    """
    import decord

    # 3 ffmpeg threads per worker: 8 workers x 3 = the 24 cores a rank owns on gpu_p6.
    # V100 nodes have 10 cores/GPU, so 8 workers x 3 oversubscribes there -- drop
    # workers, not this.
    reader = decord.VideoReader(path, num_threads=3)
    total = len(reader)
    native_fps = reader.get_avg_fps() or 30.0
    n_frames = int(round(total / native_fps * video_fps))
    n_frames = max(1, min(n_frames, video_max_frames, total))
    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    return [Image.fromarray(f) for f in reader.get_batch(idx).asnumpy()]


def load_frames(example, data_root=None, video_fps: float = 1.0,
                video_max_frames: int = 32):
    """Row -> list[PIL.Image], whichever corpus it came from."""
    if task_of(example) == "det":
        return load_example_video(example, data_root)
    path = example["video"]
    if data_root is not None and not os.path.isabs(path):
        path = os.path.join(str(data_root), path)
    return decode_video_frames(path, video_fps, video_max_frames)


def _vqa_user_content(frames, example):
    """VQA user turn: the video, then the question."""
    return [
        _video_content_llm(frames),
        {"type": "text", "text": example["conversations"][0]["value"]},
    ]


def task_messages(example, frames, target_text: str, prompt_only: bool = False):
    """The full (or prompt-only) conversation for either task, system turn included."""
    task = task_of(example)
    if task == "det":
        user = _graph_user_content(frames)
    else:
        user = _vqa_user_content(frames, example)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPTS[task]}]},
        {"role": "user", "content": user},
    ]
    if not prompt_only:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]})
    return messages


def target_text_for(example: Dict[str, Any]) -> str:
    """The supervised span: the scene graph for det, the gpt turn for vqa."""
    if task_of(example) == "det":
        text = load_scene_graph_text(example)
        if not text:
            raise ValueError(f"det row {scene_id_for(example)} carries no graph.")
        return text
    return example["conversations"][1]["value"]


def load_scene_graph_text(example: Dict[str, Any], scene_graph_root=None) -> Optional[str]:
    """Return the canonicalised VG-LLM json graph for a row (travels with the row)."""
    return example.get("graph") or None


def _video_content_llm(frames) -> Dict[str, Any]:
    """The MLLM's copy of the clip, at the fork's pixel budget.

    No resized_height/resized_width: passing them makes fetch_video honour them verbatim
    and bypasses its pixel budget. min_pixels must be set too: the default min (131072)
    exceeds this max and smart_resize asserts max >= min.
    """
    return {
        "type": "video",
        "video": frames,
        "min_pixels": _LLM_MIN_PIXELS,
        "max_pixels": _LLM_MAX_PIXELS,
    }


# 64-frame budget: 128 tok cap per frame (96 actual on 4:3 -> 384x256) so 64 frames cost
# about the visual tokens of the 32-frame runs at 256. Same numbers the frame cache was
# built with; applies to vqa rows and eval alike.
_LLM_MIN_PIXELS = 64 * 32 * 32
_LLM_MAX_PIXELS = 128 * 32 * 32


class FrameCacheMiss(FileNotFoundError):
    """No pre-decoded frames for this clip under these params. Never decoded live here:
    a silent fallback is how a run quietly goes back to 6 s/sample."""


def frame_cache_frames(root: Path, video_path: str, params: Dict[str, Any]) -> List[Image.Image]:
    """The Qwen-branch frames of one clip from pre_extract_frames.py's cache.

    Same key (sha1 of the mix's video string) and the same meta.json param stamp the
    extractor writes, so a cache built for another fps / frame count / pixel budget is a
    miss, not a wrong answer. Frames come back already at the LLM budget size, so the
    processor's smart_resize downstream is a no-op.
    """
    key = hashlib.sha1(video_path.encode()).hexdigest()
    d = root / key[:2] / key
    try:
        meta = json.loads((d / "meta.json").read_text())
    except (OSError, ValueError) as e:
        raise FrameCacheMiss(f"no frame cache entry for {video_path}: {e}") from e
    stale = {k: (meta.get(k), v) for k, v in params.items() if meta.get(k) != v}
    if stale:
        raise FrameCacheMiss(f"frame cache for {video_path} built with {stale} (meta vs run)")
    return [Image.open(d / "llm" / f"{i:03d}.jpg").convert("RGB") for i in range(meta["n_frames"])]


def _graph_user_content(frames):
    return [
        _video_content_llm(frames),
        {"type": "text", "text": "These are the RGB frames of the scene, in order.\n"},
        {"type": "text", "text": SCENE_GRAPH_QUESTION},
    ]


def perturb_graph_text(text: str, frac: float, rng) -> str:
    """Re-render the target with each of the 9 box values scaled by 1 +/- ``frac``.
    det rows only; see the parent fork for the rounding interaction."""
    objs = _parse_boxes(text)
    if not objs:
        return text
    for o in objs:
        box = np.asarray(o["bbox_3d"], dtype=float)
        o["bbox_3d"] = box * (1.0 + rng.uniform(-frac, frac, box.shape))
    return _render_boxes(objs)


def make_collator(
    processor: AutoProcessor,
    data_root: Path,
    image_patch_size: int = 16,
    box_noise: float = 0.0,
    video_fps: float = 1.0,
    video_max_frames: int = 32,
    frame_cache_root: Optional[Path] = None,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Build a batch: one supervised assistant span per row.

    ``box_noise`` applies to det rows only -- a VQA answer has nothing to perturb.
    """
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    # Entropy from the OS: dataloader workers are forked, so a fixed seed would hand every
    # worker the same perturbation stream.
    _noise_rng = np.random.default_rng()

    def _unexpanded_len(text: str) -> int:
        return len(tokenizer(text, return_attention_mask=False)["input_ids"])

    # The frame cache was built by the VGGT runs, so its meta.json still carries
    # vggt_image_resolution (256 for the 64f cache); it is part of the stamp, nothing else.
    frame_cache_params = {
        "video_fps": video_fps, "video_max_frames": video_max_frames,
        "vggt_image_resolution": 256,
        "min_pixels": _LLM_MIN_PIXELS, "max_pixels": _LLM_MAX_PIXELS,
    }

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        llm_video_tensors: List[Any] = []
        llm_video_metadata: List[Any] = []
        merged_video_kwargs: Dict[str, Any] = {}
        span_bounds: List[Dict[str, int]] = []

        for example in examples:
            if frame_cache_root is not None and task_of(example) != "det":
                # Pre-decoded Qwen frames (det rows are JPEG dirs already).
                frames = frame_cache_frames(frame_cache_root, example["video"], frame_cache_params)
            else:
                frames = load_frames(example, data_root, video_fps, video_max_frames)
            target = target_text_for(example)
            if task_of(example) == "det" and box_noise:
                target = perturb_graph_text(target, box_noise, _noise_rng)
            messages = task_messages(example, frames, target)
            # messages[:-1] is system + user: the prompt half, whichever task this is.
            t1 = processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)
            t2 = processor.apply_chat_template(messages, tokenize=False)
            u1, u2 = _unexpanded_len(t1), _unexpanded_len(t2)
            span_bounds.append({"graph_start": u1, "graph_end": u2, "full": u2})
            full_texts.append(t2)

            _, llm_videos, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
            )
            if not llm_videos:
                raise ValueError("process_vision_info returned no video for the MLLM branch.")
            tensors, metadata = zip(*llm_videos)
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

        graph_labels = torch.full_like(input_ids, IGNORE_INDEX)
        _, seq_len = input_ids.shape

        for i, bounds in enumerate(span_bounds):
            # delta absorbs every token the processor added over the plain tokenizer's
            # count (the expanded <|video_pad|> run). It stays correct only because all of
            # that expansion sits in the user turn, ahead of the assistant span; a video
            # block after the assistant turn would break this. (Right padding is likewise
            # assumed, as before.)
            delta = expanded_lens[i] - bounds["full"]
            g0 = max(0, min(bounds["graph_start"] + delta, seq_len))
            g1 = max(0, min(bounds["graph_end"] + delta, seq_len))
            if g1 > g0:
                graph_labels[i, g0:g1] = input_ids[i, g0:g1]

        if pad_token_id is not None:
            graph_labels[input_ids == pad_token_id] = IGNORE_INDEX
        graph_labels[attention_mask == 0] = IGNORE_INDEX

        batch["labels"] = graph_labels
        return batch

    return collate_fn


if __name__ == "__main__":
    # Smoke check: two vqa rows through the real collator, asserting the shapes and that
    # only the assistant span is supervised. Needs a built mix and its frame cache.
    import argparse
    import json

    from transformers import AutoProcessor as _AP

    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="jsonl from build_mix.py")
    ap.add_argument("--frame_cache_root", default=None)
    ap.add_argument("--video_max_frames", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--data_root", default="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K")
    args = ap.parse_args()

    rows = []
    for line in open(args.mix):
        row = json.loads(line)
        if row.get("task") == "vqa":
            rows.append(row)
        if len(rows) == 2:
            break
    assert len(rows) == 2, "mix has fewer than 2 vqa rows"

    processor = _AP.from_pretrained(args.model)
    collate = make_collator(
        processor, data_root=Path(args.data_root),
        frame_cache_root=Path(args.frame_cache_root) if args.frame_cache_root else None,
        video_max_frames=args.video_max_frames,
    )
    batch = collate(rows)
    assert not any(k.startswith("vggt") for k in batch), list(batch)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    assert batch["pixel_values_videos"].ndim == 2 and batch["video_grid_thw"].shape[0] == 2
    # Nothing in the prompt is supervised: the first label is at/after the generation prompt.
    for i, row in enumerate(rows):
        labels = batch["labels"][i]
        supervised = labels[labels != IGNORE_INDEX]
        first = int((labels != IGNORE_INDEX).nonzero()[0])
        prompt_len = len(processor.tokenizer(
            processor.apply_chat_template(task_messages(row, [], "")[:-1], tokenize=False,
                                          add_generation_prompt=True))["input_ids"])
        assert first >= prompt_len, (first, prompt_len)
        assert (labels[:first] == IGNORE_INDEX).all()
        text = processor.tokenizer.decode(supervised)
        assert target_text_for(row)[:40] in text, (text[:120], target_text_for(row)[:120])
        print(f"row {i}: input_ids {tuple(batch['input_ids'].shape)}, "
              f"pixel_values_videos {tuple(batch['pixel_values_videos'].shape)}, "
              f"grid {batch['video_grid_thw'][i].tolist()}, {len(supervised)} supervised tokens "
              f"from position {first}, target matches")
    print("collator smoke ok")
