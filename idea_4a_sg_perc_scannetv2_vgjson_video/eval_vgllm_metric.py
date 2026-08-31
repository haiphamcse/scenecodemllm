"""Score a trained idea_4a VIDEO checkpoint with VG-LLM's OWN detection metric.

Video variant of the latent-only script: the prompt feeds the RGB frames after the 3D
tokens, exactly as training does, so this must NOT be swapped for the sibling folder's
copy (whose collator emits no video block).

The in-training callback reports mean per-sample micro-F1 (graph_vgllm.compare). VG-LLM
reports something different: tp/fp/fn pooled per category over the whole eval set, then
macro-averaged over fixed cate8 / cate20 / cate31 name lists. The two are not comparable,
so this runs their aggregation on our predictions to get a number that is.

Their code is loaded verbatim from the VG-LLM checkout (compute_ap, which matches greedily
per category at IoU 0.25 using pytorch3d, and threedod_aggregate_results, which prints the
tables). Only the parsing is ours -- our target format carries a leading {"n": N} that their
line-oriented eval() parser would choke on.

Both aggregations are reported, so the number stays tied to the existing f1 trajectory.

    conda activate worldmirror   # needs transformers>=5 AND pytorch3d
    python idea_4a_sg_perc_scannetv2_vgjson/eval_vgllm_metric.py --run results/..._lora_ep6

Wrapper with the env and paths already set: scripts/.../eval_vgllm_metric.sh
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from collator import (  # noqa: E402
    extract_raw_video_tensors,
    graph_only_messages,
    load_example_video,
)
from graph_vgllm import box_iou_3d, compare, parse  # noqa: E402
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration  # noqa: E402
from train import scannet_samples  # noqa: E402

_VGLLM = Path("/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/VG-LLM")
_DEFAULT_ANN = _VGLLM / "data/evaluation/threedod_1perscene/scannet/scannet_det_val_4frames.json"
_DEFAULT_IMAGE_ROOT = Path("/home/ducpham/scratch/Working/dataset")
_DEFAULT_BASE = "Qwen/Qwen3-VL-2B-Instruct"


def load_threedod_utils():
    """VG-LLM's scoring module, by file path.

    Importing lmms_eval.tasks.threedod.utils would drag in lmms_eval.api.task, which pulls
    openai/tenacity and is written against transformers 4.x. This module needs none of that
    -- only torch, numpy, pytorch3d and terminaltables -- so load the file directly.
    """
    path = _VGLLM / "src/lmms_eval/tasks/threedod/utils.py"
    spec = importlib.util.spec_from_file_location("vgllm_threedod_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(base: str, checkpoint: Path, args):
    """Base model + VGGT/Perceiver + the LoRA adapter, in eval mode."""
    from peft import PeftModel

    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    # Must precede the adapter load: this is what creates vggt_projector, and the adapter
    # carries it in modules_to_save. Geometry must match training or the latents are garbage.
    processor = AutoProcessor.from_pretrained(base)
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
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval().to("cuda")
    return model, processor


def latest_checkpoint(run: Path) -> Path:
    ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoint-* under {run}")
    return ckpts[-1]


def generate(model, processor, example, image_root, args):
    """One greedy decode of the box list. Mirrors callbacks.py._prep_inputs.

    graph_only_messages here comes from the VIDEO collator, so the prompt already carries a
    video block after the quad placeholders. That block only becomes real tokens if videos=
    is passed to the processor -- without it the model is scored on a prompt shape it never
    trained on, and the score would be silently wrong rather than crash.
    """
    frames, meta = load_example_video(example, image_root, image_root)
    messages = graph_only_messages(
        frames, meta,
        args.frame_placeholder_token * args.frame_num_latents,
        args.camera_placeholder_token * args.camera_num_latents,
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video_inputs = extract_raw_video_tensors(frames, meta, args.image_patch_size)
    if not video_inputs:
        return None
    _, llm_videos, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=args.image_patch_size,
        return_video_metadata=True,
    )
    if not llm_videos:
        return None
    llm_tensors, llm_metadata = zip(*llm_videos)
    inputs = processor(
        text=prompt,
        videos=list(llm_tensors),
        video_metadata=list(llm_metadata),
        return_tensors="pt",
        padding=True,
        **video_kwargs,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["raw_videos"] = [v.to(device) for v in video_inputs]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    return processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def check_iou_backends(pred_boxes, gt_boxes, threedod):
    """Our scipy halfspace IoU vs their pytorch3d IoU on real boxes.

    The two aggregations are only comparable if the underlying IoU agrees; a silent
    disagreement here would look like a metric difference rather than a backend difference.
    """
    EulerBox = threedod.EulerDepthInstance3DBoxes
    worst = 0.0
    for pred in pred_boxes[:5]:
        for gt in gt_boxes[:5]:
            ours = box_iou_3d(pred["bbox_3d"], gt["bbox_3d"])
            theirs = float(EulerBox.overlaps(
                EulerBox(torch.tensor([list(map(float, pred["bbox_3d"]))]), convention="ZXY"),
                EulerBox(torch.tensor([list(map(float, gt["bbox_3d"]))]), convention="ZXY"),
            ))
            worst = max(worst, abs(ours - theirs))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="results dir holding checkpoint-*")
    p.add_argument("--checkpoint", default=None, help="explicit checkpoint dir (default: latest)")
    p.add_argument("--ann", default=str(_DEFAULT_ANN), help="VG-LLM-format val json to score")
    p.add_argument("--image-root", default=str(_DEFAULT_IMAGE_ROOT))
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--out", default=None, help="per-sample jsonl (default: <run>/vgllm_metric.jsonl)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    # Geometry: must match the training script or the adapter loads onto a different model.
    p.add_argument("--vggt-checkpoint", default=None)
    p.add_argument("--vggt-embed-dim", type=int, default=2048)
    p.add_argument("--frame-num-latents", type=int, default=256)
    p.add_argument("--camera-num-latents", type=int, default=32)
    p.add_argument("--frame-widening-factor", type=int, default=2)
    p.add_argument("--frame-placeholder-token", default="<|quad_start|>")
    p.add_argument("--camera-placeholder-token", default="<|quad_end|>")
    p.add_argument("--image-patch-size", type=int, default=16)
    args = p.parse_args()

    if args.vggt_checkpoint is None:
        from train import _DEFAULT_VGGT_CHECKPOINT
        args.vggt_checkpoint = _DEFAULT_VGGT_CHECKPOINT

    run = Path(args.run)
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run)
    out_path = Path(args.out) if args.out else run / "vgllm_metric.jsonl"
    threedod = load_threedod_utils()

    image_root = Path(args.image_root)
    # min_boxes=1 / no token cap: the training filters would drop rows VG-LLM scored, and the
    # point of this run is to score the SAME rows it did.
    rows = scannet_samples(Path(args.ann), image_root, min_boxes=1, max_target_tokens=None)
    # scannet_samples absolutises the image paths; the annotation stores them relative to
    # image_root. Key both on the relative form so a row maps to the GT it was built from.
    raw = {tuple(r["images"]): r for r in json.load(open(args.ann))}

    def ann_key(row):
        return tuple(os.path.relpath(p, image_root) for p in row["images"])
    if args.limit:
        rows = rows[: args.limit]
    print(f"checkpoint: {ckpt}\nrows: {len(rows)} (ann has {len(raw)})", flush=True)

    model, processor = load_model(args.base, ckpt, args)

    results, micro, n_micro, iou_gap = [], defaultdict(float), 0, 0.0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in tqdm(rows, desc="eval", unit="sample"):
            pred_text = generate(model, processor, row, image_root, args)
            if pred_text is None:
                continue
            pred_boxes = parse(pred_text)
            # GT straight from the annotation, exactly the boxes VG-LLM scored -- not our
            # canonicalised target text.
            gt_boxes = raw[ann_key(row)]["boxes"]

            pred_dict, gt_dict = defaultdict(list), defaultdict(list)
            for b in pred_boxes:
                pred_dict[b["label"]].append([float(x) for x in b["bbox_3d"]])
            for b in gt_boxes:
                gt_dict[b["label"]].append([float(x) for x in b["bbox_3d"]])

            results.append({
                "result": threedod.compute_ap(gt_dict, pred_dict),
                "gt_labels": list(gt_dict.keys()),
                "images": row["images"],
            })

            # Our own per-sample micro score, so this ties back to the training curve.
            m = compare(row["graph"], pred_text)
            for k in ("precision", "recall", "f1"):
                micro[k] += float(m.get(k, 0.0))
            n_micro += 1
            if n_micro == 1 and pred_boxes and gt_boxes:
                iou_gap = check_iou_backends(pred_boxes, gt_boxes, threedod)

            fh.write(json.dumps({
                "scene": row["scene"], "images": row["images"],
                "pred": pred_text, "n_pred": len(pred_boxes), "n_gt": len(gt_boxes),
                "micro": {k: m.get(k) for k in ("precision", "recall", "f1")},
            }) + "\n")

    print(f"\nIoU backend max |ours - pytorch3d| on first sample: {iou_gap:.2e}")
    if iou_gap > 1e-3:
        print("WARNING: IoU backends disagree; the two aggregations below are not "
              "measuring the same matching.")

    print(f"\n=== VG-LLM aggregation (macro-F1 over fixed category lists) ===", flush=True)
    score = threedod.threedod_aggregate_results(results)
    print(f"threedod_score (cate31 macro-F1): {score:.4f}")

    if n_micro:
        print("\n=== our aggregation (mean per-sample micro, the training-curve number) ===")
        print(" ".join(f"{k}={micro[k] / n_micro:.4f}" for k in ("precision", "recall", "f1")))
    print(f"\nper-sample dump: {out_path}")


if __name__ == "__main__":
    main()
