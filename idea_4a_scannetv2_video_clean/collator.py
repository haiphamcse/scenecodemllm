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

The clip is preprocessed twice, once per encoder:

    raw_videos           VGGT-Omega's own balanced resize, [0, 1]  (extract_raw_video_tensors)
    pixel_values_videos  qwen_vl_utils' video token budget         (_video_content_llm)

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
VGGT_IMAGE_RESOLUTION = 512
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
    vggt_image_resolution: int = VGGT_IMAGE_RESOLUTION,
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
            frames = load_example_video(example, data_root, cache_root)
            graph_text = load_scene_graph_text(example, scene_graph_root)
            if not graph_text:
                raise ValueError(
                    f"No scene graph for {scene_id_for(example)}; graph-only training "
                    f"requires one for every row."
                )
            if box_noise:
                graph_text = perturb_graph_text(graph_text, box_noise, _noise_rng)
            messages = graph_messages(
                frames, frame_placeholder_text, camera_placeholder_text, graph_text,
            )
            t1 = processor.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
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
    # Corpus length survey. Collates every usable row of a scannet_det json exactly as
    # training does and dumps, per row, the expanded input_ids length and the number of
    # boxes in the target -- the two numbers a sequence-length budget is picked from, now
    # that neither the collator nor train.py clips a long target.
    #
    #   python collator.py                       # 32-frame train corpus
    #   python collator.py --json <other>.json --out lengths.txt
    import argparse
    import sys

    import numpy as np
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoProcessor as _AutoProcessor

    _THIS_DIR = Path(__file__).resolve().parent
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    from argument import DEFAULT_HF_HOME

    import os

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    # train.py imports this module, so the import lives here rather than at module scope.
    from train import _DATA, scannet_samples

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(_DATA / "vgllm_data/train/scannet_det_train_32frames_bi1.json"))
    parser.add_argument("--image_root", default=str(_DATA))
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--min_boxes", type=int, default=5)
    parser.add_argument("--frame_num_latents", type=int, default=128)
    parser.add_argument("--camera_num_latents", type=int, default=32)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", default=str(_THIS_DIR / "length_survey.txt"))
    cli = parser.parse_args()

    processor = _AutoProcessor.from_pretrained(cli.model, trust_remote_code=True)
    rows = scannet_samples(Path(cli.json), Path(cli.image_root), min_boxes=cli.min_boxes)
    collate = make_collator(
        processor,
        data_root=Path(cli.image_root),
        cache_root=Path(cli.image_root),
        frame_placeholder_text="<|quad_start|>" * cli.frame_num_latents,
        camera_placeholder_text="<|quad_end|>" * cli.camera_num_latents,
        image_patch_size=cli.image_patch_size,
    )

    # The jpg decode and the two resizes are the cost here, and they are per row and
    # independent -- exactly what worker processes are for. A plain list is already a
    # map-style Dataset, so the collator goes in as collate_fn unchanged. shuffle stays
    # off so the output lines keep corpus order and can be zipped back to `rows`.
    loader = DataLoader(
        rows,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        collate_fn=collate,
        shuffle=False,
    )

    lengths: List[int] = []
    n_boxes: List[int] = []
    with open(cli.out, "w") as out:
        out.write(f"# {Path(cli.json).name}\t{len(rows)} rows\tmodel={cli.model}\n")
        out.write("# scene\tn_boxes\tinput_ids\n")
        seen = 0
        for batch in tqdm(loader, unit="batch"):
            # attention_mask, not input_ids.shape: at batch_size > 1 the tensor is padded
            # to the longest row, and the padding is not part of any row's length.
            for n_tok in batch["attention_mask"].sum(dim=1).tolist():
                row = rows[seen]
                out.write(f"{row['scene']}\t{row['n_boxes']}\t{int(n_tok)}\n")
                lengths.append(int(n_tok))
                n_boxes.append(row["n_boxes"])
                seen += 1
            out.flush()  # the survey is long; a killed run still leaves usable lines

        out.write("\n# column\tmin\tmean\tmedian\tp95\tp99\tmax\n")
        for name, values in (("n_boxes", n_boxes), ("input_ids", lengths)):
            a = np.asarray(values)
            out.write(
                f"# {name}\t{a.min()}\t{a.mean():.1f}\t{np.median(a):.0f}\t"
                f"{np.percentile(a, 95):.0f}\t{np.percentile(a, 99):.0f}\t{a.max()}\n"
            )
    print(f"{len(rows)} rows -> {cli.out}")
