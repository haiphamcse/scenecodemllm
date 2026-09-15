"""Standalone VSI-Bench eval, LIVE: VGGT-Omega runs on the clip (no VGGT cache).

VSI-Bench clips are decoded from the video (fps sampling capped at --video_max_frames) and
fed to Qwen at the collator's 128 tok/frame budget; VGGT gets its own balanced resize of the
same frames. The model class is common/model_with_vggt_live.py (VGGT inside, same adapter
layout as this fork's cached model). Perceiver geometry flags must match the checkpoint.

  python idea_3i_180k_es_64f_lat512_clean/eval.py --adapter_path results/<run>/checkpoint-N \
      --split full --output_dir <dir> --frame_num_latents 512 --vggt_image_resolution 256 \
      --video_max_frames 64

--num_shards N / --shard_index i splits the split across N jobs; merge with
common/merge_eval_shards.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np
from datasets import Dataset, load_dataset
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# _THIS_DIR first so this fork's collator shadows the common one.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import DEFAULT_HF_HOME  # noqa: E402
from collator import VGGT_IMAGE_RESOLUTION, task_messages  # noqa: E402
# data.py inserts lmms-eval onto sys.path and re-exports the VSI-Bench helpers.
from data import (  # noqa: E402
    LMMS_DEFAULT_VSIBENCH_KW,
    vsibench_doc_to_text,
    vsibench_doc_to_visual,
)
from export_vggt_features import (  # noqa: E402
    DEFAULT_VGGT_CHECKPOINT as _DEFAULT_VGGT_CHECKPOINT, extract_raw_video_tensors,
)
from vsibench_metrics import AGGREGATORS, vsibench_process_results  # noqa: E402

# Loaded by path: this fork's model_with_vggt.py (cached variant) shares the class names.
_spec = importlib.util.spec_from_file_location(
    "model_with_vggt_live", _COMMON / "model_with_vggt_live.py")
_live = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _live  # transformers looks the class module up in sys.modules
_spec.loader.exec_module(_live)
Qwen3VLWithVggtForConditionalGeneration = _live.Qwen3VLWithVggtForConditionalGeneration

from qwen_vl_utils import process_vision_info  # noqa: E402


SPLIT_CONFIG = {
    # --split: (HF dataset config, keep ScanNet++ rows only)
    "debiased": ("debiased", True),
    "full": ("full", False),
}


def build_eval_dataset(
    split: str,
    hf_home: Optional[str],
    max_samples: Optional[int],
) -> Dataset:
    """Build a VSI-Bench eval split as rows consumable by ``task_messages``."""
    config, scannetpp_only = SPLIT_CONFIG[split]

    load_kwargs: Dict[str, Any] = {}
    if hf_home:
        load_kwargs["cache_dir"] = hf_home
    elif Path(DEFAULT_HF_HOME).exists():
        load_kwargs["cache_dir"] = DEFAULT_HF_HOME

    vsi_bench = load_dataset("nyu-visionx/VSI-Bench", config, **load_kwargs)["test"]
    eval_rows: List[Dict[str, Any]] = []
    for ex in vsi_bench:
        visual_paths = vsibench_doc_to_visual(ex)
        if not visual_paths:
            continue
        if scannetpp_only and "scannetpp" not in str(visual_paths[0]).lower():
            continue
        user_text = vsibench_doc_to_text(ex, lmms_eval_specific_kwargs=LMMS_DEFAULT_VSIBENCH_KW)
        gt = ex["ground_truth"]
        eval_rows.append(
            {
                "video": visual_paths[0],
                "conversations": [{"value": user_text}, {"value": str(gt)}],
                "question_type": ex.get("question_type"),
                "ground_truth": gt,
                "source": "VSI-Bench",
            }
        )
        if max_samples is not None and len(eval_rows) >= max_samples:
            break

    return Dataset.from_list(eval_rows)


def decode_video_frames(path: str, video_fps: float, video_max_frames: int):
    """Video -> list[PIL.Image] at native resolution: ``video_fps`` sampling, uniform over the
    clip, capped at ``video_max_frames`` (fetch_video's rule)."""
    import decord

    reader = decord.VideoReader(path, num_threads=3)
    total = len(reader)
    native_fps = reader.get_avg_fps() or 30.0
    n_frames = int(round(total / native_fps * video_fps))
    n_frames = max(1, min(n_frames, video_max_frames, total))
    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    return [Image.fromarray(f) for f in reader.get_batch(idx).asnumpy()]


def load_model(args: argparse.Namespace, processor: AutoProcessor, dtype: torch.dtype):
    """Base Qwen3-VL + VGGT/Perceiver modules + the trained LoRA/projector adapter.

    ``initialize_vggt`` must run *before* PeftModel wrapping so the adapter's
    ``modules_to_save=['vggt_projector']`` weights have a module to land in.
    """
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn_implementation
    ).to(args.device)
    model.initialize_vggt(
        args.vggt_checkpoint,
        tokenizer=processor.tokenizer,
        vggt_embed_dim=args.vggt_embed_dim,
        frame_num_latents=args.frame_num_latents,
        camera_num_latents=args.camera_num_latents,
        frame_placeholder_token=args.frame_placeholder_token,
        camera_placeholder_token=args.camera_placeholder_token,
        frame_widening_factor=args.frame_widening_factor,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    frame_placeholder_text = args.frame_placeholder_token * args.frame_num_latents
    camera_placeholder_text = args.camera_placeholder_token * args.camera_num_latents

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_model(args, processor, dtype)

    eval_dataset = build_eval_dataset(
        split=args.split, hf_home=args.hf_home, max_samples=args.max_samples
    )
    print(f"VSI-Bench split={args.split}: {len(eval_dataset)} eval rows | live decode "
          f"({args.video_fps} fps, max {args.video_max_frames} frames)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _sfx = "" if args.num_shards == 1 else f"_shard{args.shard_index}of{args.num_shards}"
    preds_path = out_dir / f"predictions_{args.split}{_sfx}.jsonl"
    results_path = out_dir / f"results_{args.split}{_sfx}.json"

    processed_docs: List[Dict[str, Any]] = []
    decoded_live = 0
    # Video-sorted rows let a one-entry memo decode each video once.
    order = sorted(range(len(eval_dataset)), key=lambda i: eval_dataset[i]["video"])
    if args.num_shards > 1:
        # Contiguous, not strided, so the decode memo keeps working.
        n = len(order)
        lo = n * args.shard_index // args.num_shards
        hi = n * (args.shard_index + 1) // args.num_shards
        order = order[lo:hi]
        print(f"shard {args.shard_index}/{args.num_shards}: rows [{lo}:{hi}] of {n}")
    last_vision: Dict[str, Any] = {}
    with preds_path.open("w", encoding="utf-8") as preds_file:
        for idx in tqdm(order, desc=f"VSI-Bench ({args.split})", unit="sample"):
            example = eval_dataset[idx]
            if last_vision.get("key") != example["video"]:
                decoded_live += 1
                last_vision.clear()
                last_vision.update(
                    key=example["video"],
                    val=decode_video_frames(example["video"], args.video_fps, args.video_max_frames),
                )
            frames = last_vision["val"]

            messages = task_messages(
                example, frames, frame_placeholder_text, camera_placeholder_text,
                target_text="", prompt_only=True,
            )
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            _, vision, processed_video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True,
                image_patch_size=args.image_patch_size,
                return_video_metadata=True,
            )
            if not vision:
                continue

            video_inputs, video_metadata_list = map(list, zip(*vision))
            inputs = processor(
                text=prompt,
                videos=video_inputs,
                video_metadata=video_metadata_list,
                return_tensors="pt",
                padding=True,
                **processed_video_kwargs,
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            # VGGT gets its own balanced resize, not the MLLM's budgeted tensors.
            inputs["raw_videos"] = [
                v.to(device) for v in
                extract_raw_video_tensors(frames, args.vggt_image_resolution)
            ]

            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[0, input_len:]
            prediction = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            gt = example["ground_truth"]
            processed_docs.append(
                vsibench_process_results(
                    {"question_type": example["question_type"], "ground_truth": gt},
                    [prediction],
                )["vsibench_overall"]
            )
            preds_file.write(
                json.dumps(
                    {
                        "index": idx,
                        "question_type": example["question_type"],
                        "prediction": prediction,
                        "ground_truth": str(gt),
                    }
                )
                + "\n"
            )

    scores = {tag: agg_fn(processed_docs) for tag, agg_fn in AGGREGATORS} if processed_docs else {}
    summary = {
        "split": args.split,
        "dtype": args.dtype,
        "adapter_path": args.adapter_path,
        "model_path": args.model_path,
        "frame_num_latents": args.frame_num_latents,
        "camera_num_latents": args.camera_num_latents,
        "frame_widening_factor": args.frame_widening_factor,
        "num_samples": len(processed_docs),
        "num_decoded_live": decoded_live,
        "scores": scores,
    }
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVSI-Bench results (split={args.split}, n={len(processed_docs)}, "
          f"decoded_live={decoded_live}):")
    for tag, value in scores.items():
        print(f"  {tag}\t{value}")
    print(f"\nPredictions: {preds_path}\nSummary: {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VSI-Bench eval for idea_3i (VGGT-Perceiver).")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="PEFT checkpoint dir (LoRA + vggt_projector).")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--split", type=str, default="full", choices=list(SPLIT_CONFIG),
                        help="debiased = debiased config, ScanNet++ only; full = full config, all sources.")
    parser.add_argument("--hf_home", type=str, default=DEFAULT_HF_HOME)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write predictions/results (default: <adapter_path>/eval).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--vggt_image_resolution", type=int, default=VGGT_IMAGE_RESOLUTION,
                        help="Balanced-resize target for the VGGT branch; must match training.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the eval across N independent jobs; merge with merge_eval_shards.py.")
    parser.add_argument("--shard_index", type=int, default=0, help="Which shard this job runs, 0-based.")
    # Must match the trained checkpoint / training-time video config.
    parser.add_argument("--vggt_checkpoint", type=str, default=_DEFAULT_VGGT_CHECKPOINT)
    parser.add_argument("--vggt_embed_dim", type=int, default=2048)
    parser.add_argument("--frame_num_latents", type=int, default=256)
    parser.add_argument("--camera_num_latents", type=int, default=32)
    parser.add_argument("--frame_widening_factor", type=int, default=2)
    parser.add_argument("--frame_placeholder_token", type=str, default="<|quad_start|>")
    parser.add_argument("--camera_placeholder_token", type=str, default="<|quad_end|>")
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--video_max_frames", type=int, default=32)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--device", type=str, default="cuda")
    # bf16 matches training and is what the H100/A100 want. V100 is sm_70: no bf16,
    # and fp16 overflows this stack -- every generation comes out "!!!!!!" (NaN logits
    # -> token 0), verified on job 1892043. Use fp32 on V100.
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.adapter_path) / "eval")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
