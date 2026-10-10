"""VSI-Bench (full split, 5,130 rows) eval of a trained checkpoint. VGGT runs live on each clip.

Clips are decoded and fed exactly as in training (data.clip_inputs, encoders.decode_video_frames:
1 fps, at most --video_max_frames, Qwen's 128 tok/frame budget); the model runs its frozen VGGT on
the clip's ``raw_videos`` entry.

    python -m sr_vggt.eval <configs/*.env flags> --adapter_path <run>/checkpoint-N \
        --output_dir <dir> --num_shards 8 --shard_index 0

Rows already in the shard's predictions file are skipped (resume after a timeout); every row is
flushed as it is written. --num_parts/--part_index split one shard's unfinished rows further.
Merge shards with ``python -m sr_vggt.merge_eval --eval_dir <dir>``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser

from . import config as cfg
from . import encoders, metrics
from .data import clip_inputs, task_messages

SPLIT = "full"


@dataclass
class EvalArguments:
    adapter_path: str = field(default="", metadata={"help": "PEFT checkpoint dir (LoRA + vggt_projector)."})
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    output_dir: str = ""
    vsibench_dir: str = field(default="", metadata={"help": "Local nyu-visionx/VSI-Bench dataset repo with "
                              "its videos unzipped into arkitscenes/ scannet/ scannetpp/ (README.md)."})
    num_shards: int = 1
    shard_index: int = 0
    num_parts: int = 1
    part_index: int = 0
    max_samples: int = 0
    max_new_tokens: int = 16
    attn_implementation: str = "sdpa"
    dtype: str = field(default="bf16", metadata={"help": "bf16 (A100/H100) | fp32 (V100: fp16 overflows)."})
    device: str = "cuda"


def build_rows(vsibench_dir: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    """VSI-Bench full test split (5,130 rows) from the local dataset repo, in the hub's row order."""
    from datasets import load_dataset

    ds = load_dataset(vsibench_dir, SPLIT)["test"]
    rows = []
    for ex in ds:
        rows.append({
            "video": metrics.doc_to_visual(ex, vsibench_dir),
            "conversations": [{"value": metrics.doc_to_text(ex)}, {"value": str(ex["ground_truth"])}],
            "question_type": ex["question_type"],
            "ground_truth": ex["ground_truth"],
        })
        if max_samples and len(rows) >= max_samples:
            break
    return rows


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
    cfg.initialize_vggt_modules(model, va, processor.tokenizer)  # before PEFT, so the adapter lands
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

    rows = build_rows(ea.vsibench_dir, ea.max_samples)
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

    memo: Dict[str, Any] = {}
    with preds_path.open("a", encoding="utf-8") as preds_file:
        for idx in tqdm(order, desc=f"VSI-Bench {SPLIT}{sfx}{part}", unit="row"):
            if idx in done:
                continue
            ex = rows[idx]
            if memo.get("key") != ex["video"]:
                memo = {"key": ex["video"],
                        "frames": encoders.decode_video_frames(ex["video"], va.video_fps, va.video_max_frames)}
            messages = task_messages(ex, memo["frames"], va.prompt, frame_ph, camera_ph,
                                     target_text="", prompt_only=True)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            videos, video_metadata, video_kwargs, raw = clip_inputs(va, messages, memo["frames"])
            inputs = processor(text=prompt, videos=videos, video_metadata=video_metadata,
                               return_tensors="pt", padding=True, **video_kwargs)
            inputs = {k: v.to(ea.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            inputs["raw_videos"] = [raw]

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
