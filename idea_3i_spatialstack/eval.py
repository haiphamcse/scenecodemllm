"""Standalone VSI-Bench evaluation for idea_3i_spatialstack (layered VGGT fusion).

Fork of idea_3i/eval.py without the placeholder blocks: geometry is summed into
Qwen3-VL's own ``deepstack_visual_embeds``, so the prompt is plain video + text
and ``raw_videos`` is handed to ``generate`` to drive the fusion.
Videos in the pre-extracted frame cache are read from it; the rest (the
scannet/arkitscenes rows of the ``full`` split) decode live at
``--video_fps``/``--video_max_frames``, which must match the training config.

``--geometry_encoder_layers``/``--geometry_fusion_layers`` must match the trained
checkpoint, otherwise the loaded vggt_projector shapes will not line up.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i_spatialstack/eval.py \
      --adapter_path results/idea_3i_spatialstack/checkpoint-2000 --split full
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
from transformers import AutoProcessor

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# _THIS_DIR first so the local (idea_3i) collator/frame_cache shadow the common ones.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import DEFAULT_DATA_ROOT, DEFAULT_HF_HOME  # noqa: E402
from collator import (  # noqa: E402
    real_video_metadata,
    resolve_video_path,
    user_only_messages,
)
# data.py inserts lmms-eval onto sys.path and re-exports the VSI-Bench helpers.
from data import (  # noqa: E402
    LMMS_DEFAULT_VSIBENCH_KW,
    vsibench_doc_to_text,
    vsibench_doc_to_visual,
)
from frame_cache import cache_paths_for, load_cached_frames  # noqa: E402
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration  # noqa: E402
from vsibench_metrics import AGGREGATORS, vsibench_process_results  # noqa: E402

from qwen_vl_utils import process_vision_info  # noqa: E402

_DEFAULT_VGGT_CHECKPOINT = (
    "/home/ducpham/scratch/Working/cache/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)

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
    """Build a VSI-Bench eval split as rows consumable by ``user_only_messages``."""
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


def _live_messages(
    example: Dict[str, Any],
    video_fps: float,
    video_max_frames: int,
) -> List[Dict[str, Any]]:
    """Live-decode message for videos absent from the cache (same layout as
    collator._user_content, but pointing fetch_video at the video path)."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": example["video"],  # absolute path -> fetch_video decodes
                    "fps": video_fps,
                    "max_frames": video_max_frames,
                },
                {"type": "text", "text": example["conversations"][0]["value"]},
            ],
        }
    ]


def load_model(args: argparse.Namespace, processor: AutoProcessor, dtype: torch.dtype):
    """Base Qwen3-VL + VGGT/merger modules + the trained LoRA/projector adapter.

    ``initialize_vggt`` must run *before* PeftModel wrapping so the adapter's
    ``modules_to_save=['vggt_projector']`` weights have a module to land in.
    """
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn_implementation
    ).to(args.device)
    model.initialize_vggt(
        args.vggt_checkpoint,
        vggt_embed_dim=args.vggt_embed_dim,
        geometry_encoder_layers=args.geometry_encoder_layers,
        geometry_fusion_layers=args.geometry_fusion_layers,
        merger_hidden_dim=args.merger_hidden_dim,
    )
    model = PeftModel.from_pretrained(model, args.adapter_path).to(args.device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    device = args.device
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    data_root = Path(".")  # VSI-Bench video paths are absolute
    cache_root = Path(args.cache_root)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = load_model(args, processor, dtype)

    eval_dataset = build_eval_dataset(
        split=args.split, hf_home=args.hf_home, max_samples=args.max_samples
    )
    print(f"VSI-Bench split={args.split}: {len(eval_dataset)} eval rows | cache: {cache_root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"predictions_{args.split}.jsonl"
    results_path = out_dir / f"results_{args.split}.json"

    processed_docs: List[Dict[str, Any]] = []
    decoded_live = 0
    # VSI-Bench asks ~18 questions per video but does not group them, so walking
    # the rows in video order lets a one-entry memo skip all but one decode per
    # video (~25% of wall time on the live-decode split).
    order = sorted(range(len(eval_dataset)), key=lambda i: eval_dataset[i]["video"])
    last_vision: Dict[str, Any] = {}
    with preds_path.open("w", encoding="utf-8") as preds_file:
        for idx in tqdm(order, desc=f"VSI-Bench ({args.split})", unit="sample"):
            example = eval_dataset[idx]

            npy_path, json_path = cache_paths_for(
                resolve_video_path(data_root, example["video"]), cache_root
            )
            cached = npy_path.exists() and json_path.exists()
            if cached:
                frames, meta = load_cached_frames(npy_path, json_path)
                messages = user_only_messages(frames, meta, example)
            else:
                decoded_live += 1
                messages = _live_messages(example, args.video_fps, args.video_max_frames)

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if last_vision.get("key") != example["video"]:
                _, vision, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True,
                    image_patch_size=args.image_patch_size,
                    return_video_metadata=True,
                )
                last_vision.clear()
                last_vision.update(key=example["video"], val=(vision, video_kwargs))
            vision, processed_video_kwargs = last_vision["val"]
            if not vision:
                continue

            video_inputs, video_metadata_list = map(list, zip(*vision))
            if cached:
                # Replace the list-branch's fabricated metadata with the real
                # decode metadata so timestamps match the live-decode path.
                video_metadata_list = [real_video_metadata(meta)] * len(video_inputs)
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
            inputs["raw_videos"] = [v.to(device) for v in video_inputs]

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
        "adapter_path": args.adapter_path,
        "model_path": args.model_path,
        "geometry_encoder_layers": args.geometry_encoder_layers,
        "geometry_fusion_layers": args.geometry_fusion_layers,
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
    parser = argparse.ArgumentParser(
        description="VSI-Bench eval for idea_3i_spatialstack (layered VGGT fusion)."
    )
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="PEFT checkpoint dir (LoRA + vggt_projector).")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--split", type=str, default="full", choices=list(SPLIT_CONFIG),
                        help="debiased = debiased config, ScanNet++ only; full = full config, all sources.")
    parser.add_argument("--cache_root", type=str,
                        default=str(Path(DEFAULT_DATA_ROOT).parent / "vsi_590k_frame_cache"),
                        help="Frame cache dir (uncached videos decode live).")
    parser.add_argument("--hf_home", type=str, default=DEFAULT_HF_HOME)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write predictions/results (default: <adapter_path>/eval).")
    parser.add_argument("--max_samples", type=int, default=None)
    # Must match the trained checkpoint / training-time video config.
    parser.add_argument("--vggt_checkpoint", type=str, default=_DEFAULT_VGGT_CHECKPOINT)
    parser.add_argument("--vggt_embed_dim", type=int, default=2048)
    parser.add_argument("--geometry_encoder_layers", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--geometry_fusion_layers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--merger_hidden_dim", type=int, default=4096)
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--video_max_frames", type=int, default=32)
    parser.add_argument("--image_patch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.adapter_path) / "eval")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
