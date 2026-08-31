"""Run a trained checkpoint over the 32-frame val corpus and score it.

Port of idea_4a_sg_perc_scannetv2_vgjson_video/eval_vgllm_metric.py with the VG-LLM
category aggregation removed: that half needs pytorch3d (and so a different conda env),
and everything it adds is a second aggregation of the same predictions. What is left is
the part this folder actually needs -- greedy decode per row, VG-LLM's own per-sample
micro P/R/F1 via graph_vgllm.compare (the number the training callback reports), and the
per-sample jsonl visualize_metric_pred.py reads.

The prompt is built by the same collator helpers training uses, so the model is scored on
the prompt shape it saw: quad placeholders, then the RGB video block. Passing videos= to
the processor is not optional -- without it the video block never becomes tokens and the
score is silently wrong rather than an error.

--vggt_variant must match the run being scored. It sets the encoder, the collator's
balanced resize and enable_alignment together; a mismatch feeds the Perceiver latents from
a distribution it never trained on and quietly halves the score.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_4a_scannetv2_video_clean/infer.py --run results_jeanzay/..._32f_full --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from collator import (  # noqa: E402
    extract_raw_video_tensors,
    graph_only_messages,
    load_example_video,
)
from graph_vgllm import compare, parse  # noqa: E402
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration  # noqa: E402
from train import _DEFAULT_VAL_JSON, VGGT_VARIANTS, one_per_scene, scannet_samples  # noqa: E402

_DEFAULT_IMAGE_ROOT = Path("/home/ducpham/scratch/Working/dataset")
_DEFAULT_BASE = "Qwen/Qwen3-VL-2B-Instruct"


def latest_checkpoint(run: Path) -> Path:
    ckpts = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"no checkpoint-* under {run}")
    return ckpts[-1]


def perceiver_geometry(checkpoint: Path) -> dict:
    """Read frame/camera latent counts and the widening factor off the adapter itself.

    initialize_vggt has to build a vggt_projector of exactly the trained shape before the
    adapter loads into it (modules_to_save carries the Perceiver weights). Passing those
    numbers by hand is the easiest way to score a checkpoint against a projector that was
    randomly initialised at a different size, so they are read from the file instead.
    """
    from safetensors import safe_open

    with safe_open(str(checkpoint / "adapter_model.safetensors"), "pt") as f:
        shapes = {k: f.get_slice(k).get_shape() for k in f.keys() if "vggt_projector" in k}
    if not shapes:
        raise SystemExit(f"{checkpoint} carries no vggt_projector weights.")

    def latents(encoder):
        key = next(k for k in shapes if encoder in k and k.endswith("latent_provider._query"))
        return shapes[key][0]

    # The frame encoder's MLP is the only non-square weight in it; hidden/dim is the
    # widening factor it was built with (1 = baseline, so no widened weight exists).
    widened = [
        s for k, s in shapes.items()
        if "frame_encoder" in k and len(s) == 2 and s[0] > s[1] and s[0] % s[1] == 0
    ]
    return {
        "frame_num_latents": latents("frame_encoder"),
        "camera_num_latents": latents("camera_encoder"),
        "frame_widening_factor": max((s[0] // s[1] for s in widened), default=1),
    }


def load_model(base: str, checkpoint: Path, variant: dict, geom: dict, args):
    """Base model + VGGT/Perceiver at the trained geometry + the LoRA adapter, in eval mode."""
    from peft import PeftModel

    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(base)
    # Must precede the adapter load: this creates vggt_projector, which the adapter fills.
    model.initialize_vggt(
        variant["checkpoint"],
        tokenizer=processor.tokenizer,
        vggt_embed_dim=args.vggt_embed_dim,
        frame_num_latents=geom["frame_num_latents"],
        camera_num_latents=geom["camera_num_latents"],
        frame_placeholder_token=args.frame_placeholder_token,
        camera_placeholder_token=args.camera_placeholder_token,
        frame_widening_factor=geom["frame_widening_factor"],
        enable_alignment=variant["enable_alignment"],
    )
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval().to("cuda")
    return model, processor


def generate(model, processor, example, image_root, variant, geom, args) -> str | None:
    """One greedy decode of the box list. Mirrors callbacks.py._prep_inputs."""
    frames = load_example_video(example, image_root, image_root)
    messages = graph_only_messages(
        frames,
        args.frame_placeholder_token * geom["frame_num_latents"],
        args.camera_placeholder_token * geom["camera_num_latents"],
    )
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    video_inputs = extract_raw_video_tensors(frames, variant["image_resolution"])
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="results dir holding checkpoint-*")
    p.add_argument("--checkpoint", default=None, help="explicit checkpoint dir (default: latest)")
    p.add_argument("--ann", default=_DEFAULT_VAL_JSON, help="VG-LLM-format val json to score")
    p.add_argument("--image_root", default=str(_DEFAULT_IMAGE_ROOT))
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--out", default=None, help="per-sample jsonl (default: <run>/infer_preds.jsonl)")
    p.add_argument("--limit", type=int, default=50, help="rows to score (0 = all)")
    p.add_argument("--one_per_scene", type=lambda s: s.lower() != "false", default=True,
                   help="take the first row of each scene before --limit, as train.py's eval does")
    p.add_argument("--min_boxes", type=int, default=1,
                   help="1 scores every row the annotation holds; training used 5")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--vggt_variant", default="512", choices=sorted(VGGT_VARIANTS))
    p.add_argument("--vggt_embed_dim", type=int, default=2048)
    p.add_argument("--frame_placeholder_token", default="<|quad_start|>")
    p.add_argument("--camera_placeholder_token", default="<|quad_end|>")
    p.add_argument("--image_patch_size", type=int, default=16)
    args = p.parse_args()

    run = Path(args.run)
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(run)
    out_path = Path(args.out) if args.out else run / "infer_preds.jsonl"
    variant = VGGT_VARIANTS[args.vggt_variant]
    geom = perceiver_geometry(ckpt)

    image_root = Path(args.image_root)
    rows = scannet_samples(Path(args.ann), image_root, min_boxes=args.min_boxes)
    if args.one_per_scene:
        rows = one_per_scene(rows)
    if args.limit:
        rows = rows[: args.limit]

    # GT boxes straight from the annotation rather than the canonicalised target text.
    raw = {tuple(r["images"]): r for r in json.load(open(args.ann))}

    def ann_key(row):
        return tuple(os.path.relpath(p, image_root) for p in row["images"])

    print(f"checkpoint: {ckpt}\nvariant: {args.vggt_variant} "
          f"(resize {variant['image_resolution']}, alignment {variant['enable_alignment']})\n"
          f"perceiver: {geom}\nrows: {len(rows)}", flush=True)

    model, processor = load_model(args.base, ckpt, variant, geom, args)

    micro, n = defaultdict(float), 0
    with out_path.open("w", encoding="utf-8") as fh:
        for row in tqdm(rows, desc="infer", unit="sample"):
            pred_text = generate(model, processor, row, image_root, variant, geom, args)
            if pred_text is None:
                continue
            pred_boxes = parse(pred_text)
            gt_boxes = raw[ann_key(row)]["boxes"]
            m = compare(row["graph"], pred_text)
            for k in ("precision", "recall", "f1"):
                micro[k] += float(m.get(k, 0.0))
            n += 1
            fh.write(json.dumps({
                "scene": row["scene"], "images": row["images"],
                "pred": pred_text, "n_pred": len(pred_boxes), "n_gt": len(gt_boxes),
                "micro": {k: m.get(k) for k in ("precision", "recall", "f1")},
            }) + "\n")
            fh.flush()

    if n:
        print("\n=== mean per-sample micro (the training-curve number) ===")
        print(" ".join(f"{k}={micro[k] / n:.4f}" for k in ("precision", "recall", "f1")))
    print(f"\nper-sample dump: {out_path}")


if __name__ == "__main__":
    main()
