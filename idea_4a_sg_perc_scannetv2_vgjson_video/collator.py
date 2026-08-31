"""Qwen-VL collator for ScanNet-v2 VG-LLM-format 3D detection (latent-only).

Fork of idea_4a_sg_perc_ca1m_vgjson/collator.py with the SOURCE swapped: instead of a
per-scene corpus dir (``<root>/<scene>/frames/*.png`` + ``scene_graph.txt``), each row
comes straight from VG-LLM's ScanNet json (``scannet_det_{train,val}_4frames.json``),
which train.py has already parsed. A row carries:

    example["images"]  absolute paths to the 4 frames, json order (frame 0 = reference)
    example["graph"]   the canonicalised target text
    example["scene"]   scene id, for logging only

So the two loaders below read the row rather than the disk layout; their ``data_root`` /
``scene_graph_root`` arguments are kept only so callbacks.py works unchanged.

Frame order must be preserved: the boxes are expressed in the camera frame of
``images[0]``.

Each example is the same TWO-turn conversation as the CA-1M variant:

    turn 1  user       : [VGGT placeholders] "Detect the 3D bounding boxes..."
    turn 2  assistant  : <fenced 9-DoF json box list>  <- supervised

Only the graph span is supervised.
"""

from __future__ import annotations

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
    "y_size, z_size, yaw, roll, pitch].")


def scene_id_for(example: Dict[str, Any]) -> str:
    """ScanNet scene id of a row (logging / error messages only)."""
    return example.get("scene") or Path(example["images"][0]).parent.name


def load_example_video(example: Dict[str, Any], data_root=None, cache_root=None):
    """Load the row's frames (list[PIL.Image], RGB) + meta.

    data_root / cache_root are accepted (callbacks pass them) but unused: the row already
    holds absolute image paths in json order, and there is no npy cache.
    """
    frames = [Image.open(p).convert("RGB") for p in example["images"]]
    w, h = frames[0].size
    return frames, {"resized_height": h, "resized_width": w}


def load_scene_graph_text(example: Dict[str, Any], scene_graph_root=None) -> Optional[str]:
    """Return the canonicalised VG-LLM json graph for a row.

    train.py canonicalises once at load time (count first, objects largest-first), so the
    eval callback -- which calls this same function for its GT -- is scored against
    exactly what training saw. The ```json fence is part of the target, as VG-LLM trains
    it. scene_graph_root is ignored; the text travels with the row.
    """
    return example.get("graph") or None


def truncate_graph_text(text: str, tokenizer, max_graph_tokens: int) -> str:
    """Clamp the graph text to at most ``max_graph_tokens`` tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_graph_tokens:
        return text
    return tokenizer.decode(ids[:max_graph_tokens])


def _video_content(frames, meta: Dict[str, Any]) -> Dict[str, Any]:
    """VGGT's copy of the clip: native resolution, pinned explicitly.

    resized_* are the source jpg dimensions, so fetch_video's resize is an identity op
    and VGGT sees exactly the frames the idea_4a checkpoint's Perceiver was trained on.
    """
    return {
        "type": "video",
        "video": frames,  # list[PIL.Image] -> fetch_video list branch (no decode)
        "resized_height": meta["resized_height"],
        "resized_width": meta["resized_width"],
    }


def _video_content_llm(frames) -> Dict[str, Any]:
    """The MLLM's copy of the same clip.

    No resized_height/resized_width: passing them makes fetch_video honour them verbatim
    and bypasses its pixel budget, which at ScanNet's native 968x1296 costs ~604 tokens
    per frame (~2416 for a 4-frame row) against a ~1800-token sample. Omitting them lets
    qwen_vl_utils apply its own video budget, as idea_3i does.

    Deliberately a SEPARATE tensor from _video_content: idea_3i feeds one resize to both
    encoders, but here VGGT must keep native input or the latent pathway shifts at the
    same moment the video is introduced, and the two changes could not be told apart.
    """
    return {
        "type": "video",
        "video": frames,
    }


def extract_raw_video_tensors(frames, meta: Dict[str, Any], image_patch_size: int):
    """Frames -> VGGT input tensor(s) [T, C, H, W], via a video-only message.

    The MLLM never sees the video (latent-only variant); this tensor feeds only
    VGGTOmega, which the Perceiver compresses into the scattered latents.
    """
    video_messages = [{"role": "user", "content": [_video_content(frames, meta)]}]
    _, video_inputs, _ = process_vision_info(
        video_messages,
        return_video_kwargs=True,
        image_patch_size=image_patch_size,
        return_video_metadata=True,
    )
    if not video_inputs:
        return None
    video_tensors, _fabricated = zip(*video_inputs)
    return list(video_tensors)


def _graph_user_content(frames, meta, frame_placeholder_text, camera_placeholder_text):
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


def graph_only_messages(frames, meta, frame_placeholder_text, camera_placeholder_text):
    """Turn-1 user only -- prompt to generate the scene graph (eval)."""
    return [
        {"role": "user",
         "content": _graph_user_content(frames, meta, frame_placeholder_text, camera_placeholder_text)}
    ]


def graph_messages(frames, meta, frame_placeholder_text, camera_placeholder_text, graph_text):
    """Full 2-turn conversation: graph question -> graph."""
    return [
        {"role": "user",
         "content": _graph_user_content(frames, meta, frame_placeholder_text, camera_placeholder_text)},
        {"role": "assistant", "content": [{"type": "text", "text": graph_text}]},
    ]


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
    max_graph_tokens: int = 2048,
    image_patch_size: int = 16,
    box_noise: float = 0.0,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Build the graph-reconstruction batch (single supervised graph span)."""

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
            frames, meta = load_example_video(example, data_root, cache_root)
            graph_text = load_scene_graph_text(example, scene_graph_root)
            if not graph_text:
                raise ValueError(
                    f"No scene graph for {scene_id_for(example)}; graph-only training "
                    f"requires one for every row."
                )
            if box_noise:
                graph_text = perturb_graph_text(graph_text, box_noise, _noise_rng)
            graph_text = truncate_graph_text(graph_text, tokenizer, max_graph_tokens)
            messages = graph_messages(
                frames, meta, frame_placeholder_text, camera_placeholder_text, graph_text,
            )
            t1 = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            t2 = processor.apply_chat_template(messages, tokenize=False)
            u1, u2 = _unexpanded_len(t1), _unexpanded_len(t2)
            span_bounds.append({"graph_start": u1, "graph_end": u2, "full": u2})
            full_texts.append(t2)

            # Two resizes of the same clip. VGGT keeps native frames (unchanged from the
            # latent-only variant); the MLLM gets qwen_vl_utils' budgeted copy.
            video_tensors = extract_raw_video_tensors(frames, meta, image_patch_size)
            if not video_tensors:
                raise ValueError("extract_raw_video_tensors returned nothing for an example.")
            all_video_tensors.extend(video_tensors)

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
