"""Standalone VSI-Bench evaluation for idea_3a_180k_cached_64f (plain Qwen3-VL + LoRA).

Same flow as the cached forks' eval.py minus VGGT: the base model plus the PEFT adapter,
the MLLM frame budget from THIS fork's collator (64 frames, 128 tok/frame cap), every
video decoded live at ``--video_fps``/``--video_max_frames`` (must match training).
Same sharding and output files, so merge_eval_shards.py works unchanged.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3a_180k_cached_64f/eval.py --adapter_path results/<fork>/checkpoint-300 \
      --split full --output_dir results/<fork>/eval_ckpt300_full --video_max_frames 64

--num_shards N / --shard_index i splits the split across N independent jobs; merge the
per-shard predictions afterwards with merge_eval_shards.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# _THIS_DIR first so this fork's collator (its frame budget) shadows the common one.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import DEFAULT_HF_HOME  # noqa: E402
from collator import load_frames, task_messages  # noqa: E402
# data.py inserts lmms-eval onto sys.path and re-exports the VSI-Bench helpers.
from data import (  # noqa: E402
    LMMS_DEFAULT_VSIBENCH_KW,
    vsibench_doc_to_text,
    vsibench_doc_to_visual,
)
from vsibench_metrics import AGGREGATORS, vsibench_process_results  # noqa: E402

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




def load_model(args: argparse.Namespace, dtype: torch.dtype):
    """Base Qwen3-VL + the trained LoRA adapter."""
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn_implementation
    ).to(args.device)
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
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_model(args, dtype)

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
    # VSI-Bench asks ~18 questions per video but does not group them, so walking
    # the rows in video order lets a one-entry memo skip all but one decode per
    # video (~25% of wall time on the live-decode split).
    order = sorted(range(len(eval_dataset)), key=lambda i: eval_dataset[i]["video"])
    if args.num_shards > 1:
        # Contiguous, NOT strided: `order` is video-sorted so the one-entry decode memo
        # below only works while rows for a video stay together. A strided shard would
        # make every shard decode every video.
        n = len(order)
        lo = n * args.shard_index // args.num_shards
        hi = n * (args.shard_index + 1) // args.num_shards
        order = order[lo:hi]
        print(f"shard {args.shard_index}/{args.num_shards}: rows [{lo}:{hi}] of {n}")
    last_vision: Dict[str, Any] = {}
    with preds_path.open("w", encoding="utf-8") as preds_file:
        for idx in tqdm(order, desc=f"VSI-Bench ({args.split})", unit="sample"):
            example = eval_dataset[idx]

            # VSI-Bench is a VQA benchmark, so force the vqa turn regardless of what
            # the training mix holds -- same as callbacks.py does.
            example = dict(example, task="vqa")
            if last_vision.get("key") != example["video"]:
                decoded_live += 1
                last_vision.clear()
                last_vision.update(
                    key=example["video"],
                    val=load_frames(example, None, args.video_fps, args.video_max_frames),
                )
            frames = last_vision["val"]

            messages = task_messages(example, frames, target_text="", prompt_only=True)
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
    parser = argparse.ArgumentParser(description="VSI-Bench eval for idea_3a (plain Qwen3-VL LoRA).")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="PEFT checkpoint dir (LoRA).")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--split", type=str, default="full", choices=list(SPLIT_CONFIG),
                        help="debiased = debiased config, ScanNet++ only; full = full config, all sources.")
    parser.add_argument("--hf_home", type=str, default=DEFAULT_HF_HOME)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write predictions/results (default: <adapter_path>/eval).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the eval across N independent jobs; merge with merge_eval_shards.py.")
    parser.add_argument("--shard_index", type=int, default=0, help="Which shard this job runs, 0-based.")
    # Must match the training-time video config.
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
