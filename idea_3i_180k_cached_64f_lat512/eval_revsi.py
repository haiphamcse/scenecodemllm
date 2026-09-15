"""ReVSI (3dlg-hcvc/ReVSI) evaluation for the 64f forks, live decoding.

Same model/prompt flow as eval.py (VGGT-Omega on the clip, Perceiver placeholders, fork
collator budget); only the rows and the scorer differ:
- rows come from <revsi_dir>/<subset>/test-*.parquet, the clip is
  <revsi_dir>/<subset>/<scene_id>.mp4, a pre-sampled video holding exactly the subset's
  frame count (variable fps), so EVERY frame is decoded -- no fps sampling.
- prompt = ReVSI's official wording (ms_swift_register/revsi_register.py):
  "These are frames of a video." + question [+ Options] + numeric/MCQ post-prompt.
- scorer = torchmetrics_ext ReVSIMetric, re-implemented below (MCQ exact letter, numeric
  MRA over 0.50..0.95, sub-types averaged into 9 groups, overall = mean of groups).

  python <fork>/eval_revsi.py --adapter_path <ckpt> --revsi_dir <dir> --subset 64_frame \
      --output_dir <out> --num_shards 16 --shard_index i
  python <fork>/eval_revsi.py --merge --output_dir <out> --revsi_dir <dir> --subset 64_frame
  python <fork>/eval_revsi.py --selftest
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fork_eval", _THIS_DIR / "eval.py")
fork_eval = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fork_eval

MCQ_TYPES = [
    "object_rel_direction_forward_easy", "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard", "object_rel_direction_backward_hard",
    "object_rel_distance_closest", "object_rel_distance_farthest", "route_planning",
]
NUMERIC_TYPES = [
    "object_counting_single", "object_counting_multiple", "object_abs_distance",
    "object_size_estimation", "room_size_estimation_single", "room_size_estimation_multiple",
]
GROUPS = {  # ReVSIMetric.compute: sub-types averaged (unweighted) into one score
    "object_rel_direction": [t for t in MCQ_TYPES if t.startswith("object_rel_direction")],
    "object_counting": ["object_counting_single", "object_counting_multiple"],
    "object_rel_distance": ["object_rel_distance_closest", "object_rel_distance_farthest"],
    "room_size_estimation": ["room_size_estimation_single", "room_size_estimation_multiple"],
}
PRE = "These are frames of a video."
POST_NUM = "Answer the question using a single integer or decimal number."
POST_MCQ = "Answer with the option's letter from the given choices directly."


def build_rows(revsi_dir: Path, subset: str) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_parquet(next((revsi_dir / subset).glob("test-*.parquet")))
    rows = []
    for r in df.itertuples():
        if r.question_type in NUMERIC_TYPES:
            text = f"{PRE}\n{r.question}\n{POST_NUM}"
        else:
            text = f"{PRE}\n{r.question}\nOptions:\n" + "\n".join(r.options) + f"\n{POST_MCQ}"
        video = revsi_dir / subset / f"{r.scene_id}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        rows.append({
            "id": int(r.id), "scene_id": r.scene_id, "video": str(video), "task": "vqa",
            "question_type": r.question_type, "ground_truth": str(r.ground_truth),
            "num_frames": int(r.num_frames), "conversations": [{"value": text}, {"value": str(r.ground_truth)}],
        })
    return rows


def decode_all_frames(path: str, expected: int):
    import decord
    from PIL import Image
    reader = decord.VideoReader(path, num_threads=3)
    if len(reader) != expected:
        raise ValueError(f"{path}: {len(reader)} frames, subset says {expected}")
    return [Image.fromarray(f) for f in reader.get_batch(list(range(len(reader)))).asnumpy()]


def score_one(question_type: str, pred: str, gt: str) -> float:
    pred = str(pred).strip().split(" ")[0].rstrip(".").strip()
    if question_type in MCQ_TYPES:
        return 1.0 if pred.lower() == gt.lower() else 0.0
    try:
        p, t = float(pred), float(gt)
    except ValueError:
        return 0.0
    thresholds = [0.5 + 0.05 * i for i in range(10)]   # torch.linspace(0.5, 0.95, 10)
    return sum(abs(p - t) / t <= 1 - c for c in thresholds) / len(thresholds)


def revsi_scores(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    acc, tot = defaultdict(float), defaultdict(int)
    for r in rows:
        acc[r["question_type"]] += score_one(r["question_type"], r["prediction"], r["ground_truth"])
        tot[r["question_type"]] += 1
    per_type = {t: 100 * acc[t] / tot[t] if tot[t] else float("nan") for t in MCQ_TYPES + NUMERIC_TYPES}
    out = {}
    grouped = {s for subs in GROUPS.values() for s in subs}
    for t, v in per_type.items():
        if t not in grouped:
            out[f"{t}_acc"] = v
    for g, subs in GROUPS.items():   # sub-types absent from this (partial) run are skipped
        have = [per_type[s] for s in subs if not math.isnan(per_type[s])]
        out[f"{g}_acc"] = sum(have) / len(have) if have else float("nan")
    vals = [v for v in out.values() if not math.isnan(v)]
    out["overall_acc"] = sum(vals) / len(vals) if vals else float("nan")
    out["n"] = len(rows)
    return out


def selftest():
    assert score_one("route_planning", "A", "A") == 1.0 and score_one("route_planning", "B.", "b") == 1.0
    assert score_one("route_planning", "A", "B") == 0.0
    assert score_one("object_abs_distance", "3.2", "3.2") == 1.0
    assert abs(score_one("object_abs_distance", "3.52", "3.2") - 0.9) < 1e-9   # 10% off passes 9 of 10 thresholds
    assert score_one("object_abs_distance", "6.4 meters", "3.2") == 0.0 and score_one("object_counting_single", "two", "2") == 0.0
    rows = [{"question_type": t, "prediction": "A" if t in MCQ_TYPES else "1", "ground_truth": "A" if t in MCQ_TYPES else "1"}
            for t in MCQ_TYPES + NUMERIC_TYPES]
    s = revsi_scores(rows)
    assert s["overall_acc"] == 100.0 and len(s) == 7 + 2, s   # 3 single types + 4 groups, overall, n
    rows[0]["prediction"] = "B"   # one forward_easy miss -> rel_direction group 75, overall (6*100+75)/7
    assert abs(revsi_scores(rows)["overall_acc"] - (600 + 75) / 7) < 1e-9
    print("selftest ok")


def merge(args):
    shards = sorted(Path(args.output_dir).glob("predictions_shard*.jsonl"))
    rows, seen = [], set()
    for f in shards:
        for line in f.open():
            r = json.loads(line)
            if r["id"] not in seen:
                seen.add(r["id"]); rows.append(r)
    expected = len(build_rows(Path(args.revsi_dir), args.subset)) if args.revsi_dir else None
    print(f"merged {len(rows)} rows from {len(shards)} shards" + (f" (expected {expected})" if expected else ""))
    if expected and len(rows) != expected:
        raise SystemExit(f"NOT MERGING: {len(rows)} != {expected} rows")
    scores = revsi_scores(rows)
    out = Path(args.output_dir) / "results_merged.json"
    out.write_text(json.dumps({"subset": args.subset, "num_shards": len(shards), "scores": scores}, indent=2))
    for k, v in scores.items():
        print(f"  {k:36s} {v:.2f}" if isinstance(v, float) else f"  {k:36s} {v}")
    print(f"-> {out}")


def evaluate(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoProcessor
    _spec.loader.exec_module(fork_eval)   # heavy imports (torch, model class) only here
    from collator import task_messages  # noqa: E402  (fork collator, on sys.path via fork_eval)
    from export_vggt_features import extract_raw_video_tensors  # noqa: E402
    from qwen_vl_utils import process_vision_info  # noqa: E402

    if args.hf_home:
        import os
        os.environ["HF_HOME"] = args.hf_home
    device = args.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    frame_ph = args.frame_placeholder_token * args.frame_num_latents
    camera_ph = args.camera_placeholder_token * args.camera_num_latents
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = fork_eval.load_model(args, processor, dtype)

    rows = build_rows(Path(args.revsi_dir), args.subset)
    if args.max_samples:
        rows = rows[:args.max_samples]
    order = sorted(range(len(rows)), key=lambda i: rows[i]["video"])   # one decode per clip
    if args.num_shards > 1:
        n = len(order)
        lo, hi = n * args.shard_index // args.num_shards, n * (args.shard_index + 1) // args.num_shards
        order = order[lo:hi]
        print(f"shard {args.shard_index}/{args.num_shards}: rows [{lo}:{hi}] of {n}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    preds_path = out_dir / f"predictions{sfx}.jsonl"
    done, last = [], {}
    with preds_path.open("w") as pf, torch.no_grad():
        for i in tqdm(order, desc=f"ReVSI {args.subset}", unit="q"):
            ex = rows[i]
            if last.get("key") != ex["video"]:
                last = {"key": ex["video"], "frames": decode_all_frames(ex["video"], ex["num_frames"])}
            frames = last["frames"]
            messages = task_messages(ex, frames, frame_ph, camera_ph, target_text="", prompt_only=True)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            _, vision, vkw = process_vision_info(messages, return_video_kwargs=True,
                                                 image_patch_size=args.image_patch_size, return_video_metadata=True)
            videos, meta = map(list, zip(*vision))
            inputs = processor(text=prompt, videos=videos, video_metadata=meta, return_tensors="pt", padding=True, **vkw)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            inputs["raw_videos"] = [v.to(device) for v in extract_raw_video_tensors(frames, args.vggt_image_resolution)]
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            pred = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rec = {"id": ex["id"], "scene_id": ex["scene_id"], "question_type": ex["question_type"],
                   "prediction": pred, "ground_truth": ex["ground_truth"]}
            done.append(rec); pf.write(json.dumps(rec) + "\n")
    scores = revsi_scores(done)
    (out_dir / f"results{sfx}.json").write_text(json.dumps(
        {"subset": args.subset, "adapter_path": args.adapter_path, "model_path": args.model_path, "scores": scores}, indent=2))
    print(f"\nReVSI {args.subset} shard n={len(done)}: overall {scores['overall_acc']:.2f}\nPredictions: {preds_path}")


def main():
    p = argparse.ArgumentParser(description="ReVSI eval for the 64f forks (live VGGT).")
    p.add_argument("--adapter_path", type=str)
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--hf_home", type=str, default=None)
    p.add_argument("--revsi_dir", type=str, default=None)
    p.add_argument("--subset", type=str, default="64_frame")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--vggt_image_resolution", type=int, default=256)
    p.add_argument("--vggt_checkpoint", type=str, default=None)
    p.add_argument("--vggt_embed_dim", type=int, default=2048)
    p.add_argument("--frame_num_latents", type=int, default=256)
    p.add_argument("--camera_num_latents", type=int, default=32)
    p.add_argument("--frame_widening_factor", type=int, default=2)
    p.add_argument("--frame_placeholder_token", type=str, default="<|quad_start|>")
    p.add_argument("--camera_placeholder_token", type=str, default="<|quad_end|>")
    p.add_argument("--image_patch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--attn_implementation", type=str, default="sdpa")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()
    if args.merge:
        return merge(args)
    if args.vggt_checkpoint is None:
        _spec.loader.exec_module(fork_eval)
        args.vggt_checkpoint = fork_eval._DEFAULT_VGGT_CHECKPOINT
    evaluate(args)


if __name__ == "__main__":
    main()
