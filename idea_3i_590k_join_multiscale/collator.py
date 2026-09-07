"""Joint VQA + 3DOD collator: one batch, two objectives, one frame pipeline.

Fork of idea_4a_scannetv2_video_clean/collator.py with VSI-590K VQA rows folded in.
Rows carry a ``task`` field that build_mix.py stamps on:

  task="vqa"  VSI-590K: ``video`` is a path; the question and answer come from the row's
              own ``conversations``. Frames are decoded here.
  task="det"  ScanNet 3DOD: ``images`` is a frame list; the target is the canonicalised
              scene graph in ``graph``, and the question is SCENE_GRAPH_QUESTION.

Both tasks converge on ``list[PIL.Image]`` and from there share ONE preprocessing path:
qwen_vl_utils' token budget for the MLLM, VGGT-Omega's balanced-512 for the frozen
encoder. That is idea_4a's convention, not idea_3i_590k's. It matters because a single
frozen VGGT and a single shared Perceiver see both streams -- feeding them two different
resolutions would make the Perceiver's job depend on which corpus a row came from.
The consequence: VQA numbers here are NOT comparable to earlier idea_3i_590k runs.

The tasks are separated by a system prompt (SYSTEM_PROMPTS) so the model can tell which
objective it is being asked for; the label masking is identical for both, since both are
a single user turn followed by a single supervised assistant turn.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torchvision import transforms as TF
from vggt_omega.utils.load_fn import (
    _balanced_target_shape,
    _crop_to_supported_aspect_ratio,
)

from graph_vgllm import parse as _parse_boxes, render as _render_boxes
from transformers import AutoProcessor
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_INDEX = LabelSmoother.ignore_index

# VGGT-Omega's own defaults: vggt_omega_1b_512.pt wants image_resolution 512 and its
# aggregator patchifies at 16, so "balanced" aims for (512/16)**2 = 1024 patch tokens per
# frame. The text-aligned checkpoint wants 256 (=256 patches) -- train.py picks per
# --vggt_variant and passes it down; this is only the default.
VGGT_IMAGE_RESOLUTION = 256
VGGT_PATCH_SIZE = 16
_TO_TENSOR = TF.ToTensor()

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

# One system turn per objective. The model is trained on both at once, so the task has to
# be stated somewhere the prompt cannot be confused about; the user turns differ too, but
# a VQA question about a scene and a detection request share too much surface to rely on
# that alone.
SYSTEM_PROMPTS = {
    "vqa": "You are a spatial reasoning assistant. You are shown 3D scene context and the "
           "RGB frames of a scene. Answer the question about the scene.",
    "det": "You are a 3D object detector. You are shown 3D scene context and the RGB "
           "frames of a scene. List every object with its 3D bounding box.",
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

    Deliberately NOT fetch_video: that returns frames already resized to Qwen's budget,
    which is the resolution VGGT must not see (see the module docstring). Decoding here
    keeps both branches downstream of one native-resolution frame list, exactly as the
    det rows are. Frame CHOICE still follows fetch_video's rule -- uniform over the clip,
    ``video_fps`` sampling capped at ``video_max_frames``.
    """
    import decord

    reader = decord.VideoReader(path, num_threads=1)
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


def _vqa_user_content(frames, example, frame_placeholder_text, camera_placeholder_text):
    """VQA user turn. Same placeholder-then-video ordering as the det turn."""
    return [
        {"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
        {"type": "text", "text": "This is the 3D scene context.\n"},
        _video_content_llm(frames),
        {"type": "text", "text": example["conversations"][0]["value"]},
    ]


def task_messages(example, frames, frame_placeholder_text, camera_placeholder_text,
                  target_text: str, prompt_only: bool = False):
    """The full (or prompt-only) conversation for either task, system turn included."""
    task = task_of(example)
    if task == "det":
        user = _graph_user_content(frames, frame_placeholder_text, camera_placeholder_text)
    else:
        user = _vqa_user_content(frames, example, frame_placeholder_text,
                                 camera_placeholder_text)
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
    """Return the canonicalised VG-LLM json graph for a row.

    train.py canonicalises once at load time (count first, objects largest-first), so the
    eval callback -- which calls this same function for its GT -- is scored against
    exactly what training saw. The ```json fence is part of the target, as VG-LLM trains
    it. scene_graph_root is ignored; the text travels with the row.
    """
    return example.get("graph") or None


def _video_content_llm(frames) -> Dict[str, Any]:
    """The MLLM's copy of the clip, resized the way idea_3i resizes.

    No resized_height/resized_width: passing them makes fetch_video honour them verbatim
    and bypasses its pixel budget, which at ScanNet's native 968x1296 costs ~604 tokens
    per frame (~2416 for a 4-frame row) against a ~1800-token sample. Omitting them is
    what qwen_vl_utils' own video budget wants, and it is the same computation that fixed
    idea_3i's cached frames at 480x640 (its collator then only re-pins that cached size).

    A SEPARATE tensor from the VGGT copy below: the two encoders want different
    resolutions -- qwen's token budget here, VGGT-Omega's patch budget there.
    """
    return {
        "type": "video",
        "video": frames,
        # fetch_video's default video budget is VIDEO_MAX_TOKEN_NUM(768) * (16*2)^2 =
        # 786432 px/frame -> 24576 visual tokens over 32 frames, which is the tensor that
        # OOM'd every run on 2026-09-04 (feature_extraction_utils.py maybe_to).
        # 256 tok/frame instead (8192 visual tokens over 32 frames). NOT SpatialStack's
        # numbers: their scripts use 1664*28*28, which exceeds this fetch_video cap and
        # would be clamped straight back to the default that OOM'd.
        # min_pixels must be set too: the default min (131072) exceeds this max and
        # smart_resize asserts max >= min.
        "min_pixels": 128 * 32 * 32,
        "max_pixels": 256 * 32 * 32,
    }


def extract_raw_video_tensors(
    frames,
    image_resolution: int = VGGT_IMAGE_RESOLUTION,
    patch_size: int = VGGT_PATCH_SIZE,
):
    """Frames -> ``[tensor [T, C, H, W]]`` in [0, 1], preprocessed the way VGGT-Omega asks.

    This is load_fn.load_and_preprocess_images(mode="balanced") minus its file-opening
    step: crop the aspect ratio into [0.5, 2.0], resize (BICUBIC) to whichever h x w keeps
    the patch count near ``(image_resolution / patch_size)**2``, and scale to [0, 1]. Its
    top-level function takes paths; the frames are already decoded here, so the two
    shape helpers are called directly.

    The [0, 1] scaling used to live in model_with_vggt._extract_vggt_tokens as a
    ``max() > 1`` guard. It belongs here: ToTensor already produces the range VGGT's
    aggregator expects, and the guard could not tell a genuinely dark [0, 255] clip from
    an already-normalised one.

    Returned as a one-element list because a row is one clip; the batch is the
    concatenation over rows.
    """
    tensors = []
    for frame in frames:
        image = _crop_to_supported_aspect_ratio(frame)
        width, height = image.size
        target_h, target_w = _balanced_target_shape(
            height / max(width, 1), image_resolution, patch_size
        )
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensors.append(_TO_TENSOR(image))
    # Every frame of a ScanNet clip shares one camera, so the shapes always agree and
    # load_fn's pad-to-common-size branch has nothing to do; torch.stack says so loudly.
    return [torch.stack(tensors)]


def _graph_user_content(frames, frame_placeholder_text, camera_placeholder_text):
    # Turn-1 user, idea_3i's ordering: VGGT placeholder blocks FIRST, then the video.
    # model_with_vggt.py scatters the Perceiver latents into the placeholder positions, so
    # the 3D tokens precede the pixels and the video is read as elaborating on them.
    return [
        {"type": "text", "text": frame_placeholder_text + camera_placeholder_text},
        {"type": "text", "text": "This is the 3D scene context.\n"},
        _video_content_llm(frames),
        {"type": "text", "text": "These are the RGB frames of the same scene, in the same order.\n"},
        {"type": "text", "text": "Use both the 3D scene context and the frames.\n"},
        {"type": "text", "text": SCENE_GRAPH_QUESTION},
    ]


def graph_only_messages(frames, frame_placeholder_text, camera_placeholder_text):
    """Turn-1 user only -- prompt to generate the scene graph (eval)."""
    return [
        {"role": "user",
         "content": _graph_user_content(frames, frame_placeholder_text, camera_placeholder_text)}
    ]


def graph_messages(frames, frame_placeholder_text, camera_placeholder_text, graph_text):
    """Full 2-turn conversation: graph question -> graph."""
    return [
        {"role": "user",
         "content": _graph_user_content(frames, frame_placeholder_text, camera_placeholder_text)},
        {"role": "assistant", "content": [{"type": "text", "text": graph_text}]},
    ]


def seq_color_jitter(rng):
    """DUSt3R/cambrian-p's SeqColorJitter: ONE draw of jitter params, applied to a whole clip.

    Ported from cambrian-p/cambrianp/datasets/utils/transforms.py at its exact strengths
    (brightness/contrast/saturation 0.5, hue 0.1) but returning PIL rather than a
    normalised tensor -- both branches downstream do their own resize and ToTensor.

    The "Seq" is the point. Drawing per frame would show the model 32 differently-tinted
    views of one room and teach the encoder that a scene's appearance drifts within a
    clip; one draw per clip varies appearance ACROSS the corpus while keeping each
    sequence photometrically consistent. The op order is shuffled per clip too, since
    these adjustments do not commute.
    """
    brightness = float(rng.uniform(0.5, 1.5))
    contrast = float(rng.uniform(0.5, 1.5))
    saturation = float(rng.uniform(0.5, 1.5))
    hue = float(rng.uniform(-0.1, 0.1))
    order = rng.permutation(4)

    def apply(frame):
        for fn_id in order:
            if fn_id == 0:
                frame = TF.functional.adjust_brightness(frame, brightness)
            elif fn_id == 1:
                frame = TF.functional.adjust_contrast(frame, contrast)
            elif fn_id == 2:
                frame = TF.functional.adjust_saturation(frame, saturation)
            else:
                frame = TF.functional.adjust_hue(frame, hue)
        return frame

    return apply


def perturb_graph_text(text: str, frac: float, rng) -> str:
    """Re-render the target with each of the 9 box values scaled by 1 +/- ``frac``.

    Multiplicative, so the perturbation is proportional to each value -- that is what
    "+/-0.5% of each value" means, and it keeps a 5 m coordinate and a 5 cm one perturbed
    on the same relative scale.

    Called per __getitem__, NOT at load time, so a clip gets a different perturbation every
    epoch: the model learns the target is uncertain instead of memorising one wrong value.

    NOTE the interaction with render()'s round(v, 2): at 0.005 the noise is smaller than the
    rounding quantum for most fields, so it lands as ~26% of values shifting by exactly
    +/-0.01 rather than as smooth jitter. Measured over 256k boxes; only 5.4% of boxes come
    out completely unchanged. Raising the noise or the precision changes that character.
    """
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
    cache_root: Optional[Path],
    frame_placeholder_text: str,
    camera_placeholder_text: str,
    scene_graph_root: Optional[Path] = None,
    image_patch_size: int = 16,
    box_noise: float = 0.0,
    color_jitter: float = 0.0,
    vggt_image_resolution: int = VGGT_IMAGE_RESOLUTION,
    video_fps: float = 1.0,
    video_max_frames: int = 32,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Build a mixed batch: one supervised assistant span per row, either task.

    ``box_noise`` and ``color_jitter`` apply to det rows only. A VQA answer has nothing
    to perturb geometrically, and a VSI-590K question can be about appearance -- jittering
    hue under a "what colour is the sofa" row would make its recorded answer wrong.
    ``color_jitter`` is the PROBABILITY of jittering a row, at fixed DUSt3R strength.
    """

    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    # Entropy from the OS: dataloader workers are forked, so a fixed seed would hand every
    # worker the same perturbation stream.
    _noise_rng = np.random.default_rng()

    def _unexpanded_len(text: str) -> int:
        return len(tokenizer(text, return_attention_mask=False)["input_ids"])

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_texts: List[str] = []
        all_video_tensors: List[Any] = []
        llm_video_tensors: List[Any] = []
        llm_video_metadata: List[Any] = []
        merged_video_kwargs: Dict[str, Any] = {}
        span_bounds: List[Dict[str, int]] = []

        for example in examples:
            frames = load_frames(example, data_root, video_fps, video_max_frames)
            target = target_text_for(example)
            if task_of(example) == "det":
                if box_noise:
                    target = perturb_graph_text(target, box_noise, _noise_rng)
                if color_jitter and _noise_rng.random() < color_jitter:
                    # Before both resizes, so the VGGT copy and the MLLM copy are the
                    # same augmented clip rather than two different-looking scenes.
                    jitter = seq_color_jitter(_noise_rng)
                    frames = [jitter(f) for f in frames]
            messages = task_messages(
                example, frames, frame_placeholder_text, camera_placeholder_text, target,
            )
            # messages[:-1] is system + user: the prompt half, whichever task this is.
            t1 = processor.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)
            t2 = processor.apply_chat_template(messages, tokenize=False)
            u1, u2 = _unexpanded_len(t1), _unexpanded_len(t2)
            span_bounds.append({"graph_start": u1, "graph_end": u2, "full": u2})
            full_texts.append(t2)

            # Two resizes of the same clip: VGGT-Omega's balanced patch budget for the
            # frozen encoder, qwen_vl_utils' token budget for the MLLM.
            all_video_tensors.extend(
                extract_raw_video_tensors(frames, vggt_image_resolution))

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
            # count -- now the expanded <|video_pad|> run as well as the quad placeholders.
            # It stays correct only because all of that expansion sits in the user turn,
            # ahead of the graph span; a video block after the assistant turn would break
            # this. (Right padding is likewise assumed, as before.)
            delta = expanded_lens[i] - bounds["full"]
            g0 = max(0, min(bounds["graph_start"] + delta, seq_len))
            g1 = max(0, min(bounds["graph_end"] + delta, seq_len))
            if g1 > g0:
                graph_labels[i, g0:g1] = input_ids[i, g0:g1]

        if pad_token_id is not None:
            graph_labels[input_ids == pad_token_id] = IGNORE_INDEX
        graph_labels[attention_mask == 0] = IGNORE_INDEX

        batch["labels"] = graph_labels
        batch["raw_videos"] = all_video_tensors
        return batch

    return collate_fn


if __name__ == "__main__":
    # Smoke check: one row of each task through the real collator, asserting the two
    # things a joint batch can silently get wrong -- VGGT range, and which span is
    # supervised. Needs a built mix (see build_mix.py).
    import argparse
    import json

    from transformers import AutoProcessor as _AP

    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="jsonl from build_mix.py")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--data_root", default="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.mix)]
    picked = {}
    for row in rows:
        picked.setdefault(row["task"], row)
        if len(picked) == len(TASKS):
            break
    missing = set(TASKS) - set(picked)
    assert not missing, f"mix has no rows for {missing}"

    processor = _AP.from_pretrained(args.model)
    collate = make_collator(
        processor, data_root=Path(args.data_root), cache_root=None,
        frame_placeholder_text="<|quad_start|>" * 256,
        camera_placeholder_text="<|quad_end|>" * 32,
    )
    for task, row in picked.items():
        batch = collate([row])
        raw = batch["raw_videos"][0]
        assert 0.0 <= float(raw.min()) and float(raw.max()) <= 1.0, (task, raw.min(), raw.max())
        supervised = batch["labels"][0][batch["labels"][0] != IGNORE_INDEX]
        text = processor.tokenizer.decode(supervised)
        expected = target_text_for(row)
        assert expected[:40] in text, (task, text[:120], expected[:120])
        print(f"{task}: raw_videos {tuple(raw.shape)} in [0,1], "
              f"input_ids {tuple(batch['input_ids'].shape)}, "
              f"{len(supervised)} supervised tokens, target matches")
    print("collator smoke ok")
