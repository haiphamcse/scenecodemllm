"""VSI-Bench (full split, 5,130 rows) eval of a trained checkpoint. VGGT runs live on each clip.

Clips are decoded from the VSI-Bench mp4s (1 fps, capped at --video_max_frames, uniform) and fed
to Qwen at the training budget (128 tok/frame); the VGGT backbone encodes the same frames exactly
as export.py encodes det_es frames for training (encoders.py), once per video.

    python -m sr_vggt.eval <configs/*.env flags> --adapter_path <run>/checkpoint-N \
        --output_dir <dir> --num_shards 8 --shard_index 0

Rows already in the shard's predictions file are skipped (resume after a timeout); every row is
flushed as it is written. --num_parts/--part_index split one shard's unfinished rows further.
Merge shards with ``python -m sr_vggt.merge_eval --eval_dir <dir>``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser

from . import config as cfg
from . import encoders, metrics
from .data import task_messages

SPLIT = "full"


@dataclass
class EvalArguments:
    adapter_path: str = field(default="", metadata={"help": "PEFT checkpoint dir (LoRA + vggt_projector)."})
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = ""
    vsibench_videos: str = field(default="", metadata={"help": "Dir with arkitscenes/ scannet/ scannetpp/ mp4s."})
    num_shards: int = 1
    shard_index: int = 0
    num_parts: int = 1
    part_index: int = 0
    max_samples: int = 0
    max_new_tokens: int = 16
    attn_implementation: str = "sdpa"
    dtype: str = field(default="bf16", metadata={"help": "bf16 (A100/H100) | fp32 (V100: fp16 overflows)."})
    device: str = "cuda"


def build_rows(videos_dir: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """VSI-Bench full test split from HF_HOME (download_weights.sh fetched it)."""
    from datasets import load_dataset

    ds = load_dataset("nyu-visionx/VSI-Bench", SPLIT, cache_dir=os.environ.get("HF_HOME"))["test"]
    rows = []
    for ex in ds:
        rows.append({
            "video": metrics.doc_to_visual(ex, videos_dir),
            "conversations": [{"value": metrics.doc_to_text(ex)}, {"value": str(ex["ground_truth"])}],
            "question_type": ex["question_type"],
            "ground_truth": ex["ground_truth"],
        })
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def decode_video_frames(path: str, video_fps: float, video_max_frames: int) -> List[Image.Image]:
    """Video -> PIL frames at native resolution: ``video_fps`` sampling, uniform over the clip,
    capped at ``video_max_frames`` (qwen_vl_utils.fetch_video's rule)."""
    import decord

    reader = decord.VideoReader(path, num_threads=3)
    total = len(reader)
    native_fps = reader.get_avg_fps() or 30.0
    n_frames = max(1, min(int(round(total / native_fps * video_fps)), video_max_frames, total))
    idx = np.linspace(0, total - 1, n_frames).round().astype(int)
    return [Image.fromarray(f) for f in reader.get_batch(idx).asnumpy()]


def vggt_kwargs(args: cfg.VggtArguments, aggregator, frames, video_tensor, device, dtype) -> Dict[str, Any]:
    """Live VGGT tokens of one clip, in the form the model takes (the model casts them)."""
    if args.vggt_frames == "native":
        # SpatialStack: VGGT sees Qwen's own pixels so its patch grid is Qwen's video grid.
        video = video_tensor.float()
        if video.max() > 1.0:
            video = video / 255.0
        t = encoders.encode(aggregator, video, device, dtype, layers=args.vggt_layers)
        return {"vggt_patch_tokens": [[t[f"patch_{i}"] for i in range(len(t))]]}
    video = encoders.frames_to_video(frames, "balanced", args.vggt_image_resolution, args.vggt_backbone)
    t = encoders.encode(aggregator, video, device, dtype)
    if args.arch == "vlm3r":
        return {"vggt_tokens": [torch.cat([t["camera"][:, :1], t["patch"]], dim=1)]}
    return {"vggt_patch_tokens": [t["patch"]], "vggt_camera_tokens": [t["camera"]]}


def read_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial last line of a killed job
            rows[row["index"]] = row
    return rows


def score(rows) -> Dict[str, float]:
    docs = [metrics.vsibench_process_results(
        {"question_type": r["question_type"], "ground_truth": r["ground_truth"]}, [r["prediction"]]
    )["vsibench_overall"] for r in rows]
    return {tag: fn(docs) for tag, fn in metrics.AGGREGATORS} if docs else {}


def load_model(ea: EvalArguments, va: cfg.VggtArguments, processor, dtype):
    from peft import PeftModel

    model = cfg.model_class(va).from_pretrained(
        ea.model_name_or_path, dtype=dtype, attn_implementation=ea.attn_implementation).to(ea.device)
    cfg.initialize_projector(model, va, processor.tokenizer)  # before PEFT, so the adapter lands
    model = PeftModel.from_pretrained(model, ea.adapter_path).to(ea.device)
    return model.eval()


@torch.no_grad()
def main() -> None:
    ea, va = HfArgumentParser((EvalArguments, cfg.VggtArguments)).parse_args_into_dataclasses()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[ea.dtype]
    reasoning = va.prompt == "reason"
    if reasoning:
        from .extensions.reasoning import parse_answer
    frame_ph, camera_ph = va.placeholder_texts()

    rows = build_rows(ea.vsibench_videos, ea.max_samples)
    # Video-sorted, contiguous shards: a one-entry memo decodes + encodes each video once.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["video"])
    n = len(order)
    order = order[n * ea.shard_index // ea.num_shards: n * (ea.shard_index + 1) // ea.num_shards]
    out_dir = Path(ea.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if ea.num_shards == 1 else f"_shard{ea.shard_index}of{ea.num_shards}"
    part = "" if ea.num_parts == 1 else f".part{ea.part_index}of{ea.num_parts}"
    preds_path = out_dir / f"predictions_{SPLIT}{sfx}{part}.jsonl"
    results_path = out_dir / f"results_{SPLIT}{sfx}{part}.json"
    if part:
        # A part is a slice of the rows the whole-shard job has not written; stable only while
        # that job never runs again.
        shard_done = read_rows(out_dir / f"predictions_{SPLIT}{sfx}.jsonl")
        todo = [i for i in order if i not in shard_done]
        order = todo[len(todo) * ea.part_index // ea.num_parts: len(todo) * (ea.part_index + 1) // ea.num_parts]
    done = read_rows(preds_path)
    print(f"VSI-Bench {SPLIT}: {n} rows, this job {len(order)}, already done {len(done)}", flush=True)

    processor = AutoProcessor.from_pretrained(ea.model_name_or_path, trust_remote_code=True)
    model = load_model(ea, va, processor, dtype)
    aggregator = encoders.load_aggregator(va.vggt_backbone, va.checkpoint(), ea.device, dtype)

    from qwen_vl_utils import process_vision_info

    memo: Dict[str, Any] = {}
    with preds_path.open("a", encoding="utf-8") as preds_file:
        for idx in tqdm(order, desc=f"VSI-Bench {SPLIT}{sfx}{part}", unit="row"):
            if idx in done:
                continue
            ex = rows[idx]
            if memo.get("key") != ex["video"]:
                memo = {"key": ex["video"],
                        "frames": decode_video_frames(ex["video"], va.video_fps, va.video_max_frames)}
            messages = task_messages(ex, memo["frames"], va.prompt, frame_ph, camera_ph,
                                     target_text="", prompt_only=True)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            _, vision, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True, image_patch_size=va.image_patch_size,
                return_video_metadata=True)
            videos, video_metadata = map(list, zip(*vision))
            inputs = processor(text=prompt, videos=videos, video_metadata=video_metadata,
                               return_tensors="pt", padding=True, **video_kwargs)
            inputs = {k: v.to(ea.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "vggt" not in memo:
                memo["vggt"] = vggt_kwargs(va, aggregator, memo["frames"], videos[0], ea.device, dtype)
            inputs.update(memo["vggt"])

            out = model.generate(**inputs, max_new_tokens=ea.max_new_tokens, do_sample=False)
            gen_ids = out[0, inputs["input_ids"].shape[1]:]
            row = {"index": idx, "question_type": ex["question_type"], "ground_truth": str(ex["ground_truth"])}
            if reasoning:
                # Keep <think>/</think> (special ids) in the raw text; drop the end-of-turn tokens.
                raw = processor.tokenizer.decode(gen_ids, skip_special_tokens=False)
                raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
                row.update(prediction=parse_answer(raw), raw=raw, gen_tokens=int(gen_ids.shape[0]))
            else:
                row["prediction"] = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            preds_file.write(json.dumps(row) + "\n")
            preds_file.flush()
            done[idx] = row

    mine = [done[i] for i in order if i in done]
    summary = {"split": SPLIT, "adapter_path": ea.adapter_path, "dtype": ea.dtype,
               "num_samples": len(mine), "scores": score(mine)}
    if reasoning:
        summary["num_no_answer_line"] = sum(
            not any(l.strip().startswith("Answer:") for l in r.get("raw", "").splitlines()) for r in mine)
        summary["num_hit_cap"] = sum(r.get("gen_tokens", 0) >= ea.max_new_tokens for r in mine)
    results_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["scores"], indent=2))


if __name__ == "__main__":
    main()
