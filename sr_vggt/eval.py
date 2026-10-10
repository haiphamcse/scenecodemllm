"""VSI-Bench (full split, 5,130 rows) eval of a trained checkpoint. VGGT runs live on each clip.

Clips are decoded and fed exactly as in training (data.clip_inputs, data.decode_video_frames:
1 fps, at most --video_max_frames, Qwen's 128 tok/frame budget); the model runs its frozen VGGT on
the clip's ``raw_videos`` entry.

    python -m sr_vggt.eval <configs/*.env flags> --adapter_path <run>/checkpoint-N \
        --output_dir <dir> --num_shards 8 --shard_index 0

Rows already in the shard's predictions file are skipped (resume after a timeout); every row is
flushed as it is written. --num_parts/--part_index split one shard's unfinished rows further.
Merge shards with ``python -m sr_vggt.eval merge <dir>`` (merge() below). The VSI-Bench scoring and
prompt helpers at the bottom are copied from lmms-eval (no lmms-eval install needed).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial  # noqa: F401  (vsibench_process_results eval()s "partial(...)")
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, HfArgumentParser

from .data import clip_inputs, decode_video_frames, task_messages
from .model import VggtArguments, initialize_vggt_modules, model_class

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
            "video": doc_to_visual(ex, vsibench_dir),
            "conversations": [{"value": doc_to_text(ex)}, {"value": str(ex["ground_truth"])}],
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
    docs = [vsibench_process_results(
        {"question_type": r["question_type"], "ground_truth": r["ground_truth"]}, [r["prediction"]]
    )["vsibench_overall"] for r in rows]
    return {tag: fn(docs) for tag, fn in AGGREGATORS} if docs else {}


def load_model(ea: EvalArguments, va: VggtArguments, processor, dtype):
    from peft import PeftModel

    model = model_class(va).from_pretrained(
        ea.model_name_or_path, dtype=dtype, attn_implementation=ea.attn_implementation).to(ea.device)
    initialize_vggt_modules(model, va, processor.tokenizer)  # before PEFT, so the adapter lands
    model = PeftModel.from_pretrained(model, ea.adapter_path).to(ea.device)
    return model.eval()


@torch.no_grad()
def main() -> None:
    ea, va = HfArgumentParser((EvalArguments, VggtArguments)).parse_args_into_dataclasses()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[ea.dtype]
    reasoning = va.prompt == "reason"
    if reasoning:
        from .reasoning import parse_answer
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
                        "frames": decode_video_frames(ex["video"], va.video_fps, va.video_max_frames)}
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


FULL_ROWS = 5130


def merge(eval_dir, expected_rows: int = FULL_ROWS) -> int:
    """Score the union of a run's shard (and part) predictions, recomputed from the raw rows.
    Exits non-zero unless every one of the 5,130 rows is present."""
    eval_dir = Path(eval_dir)

    files = sorted(eval_dir.glob(f"predictions_{SPLIT}_shard*.jsonl"))
    rows = {}
    for f in files:
        got = read_rows(f)
        rows.update(got)  # index-keyed: a row written twice (shard + part) counts once
        print(f"  {f.name}: {len(got)} rows")
    if len(rows) != expected_rows:
        print(f"INCOMPLETE: {len(rows)}/{expected_rows} rows in {eval_dir}; not scoring.")
        return 1
    scores = score(rows.values())
    out = eval_dir / f"results_{SPLIT}_merged.json"
    out.write_text(json.dumps({"split": SPLIT, "num_samples": len(rows), "num_files": len(files),
                               "scores": scores}, indent=2))
    print(f"merged {len(rows)} rows from {len(files)} files -> {out}")
    for tag, v in scores.items():
        print(f"  {tag:44s} {v:.4f}")
    return 0


# ======================= VSI-Bench scoring (lmms-eval @ b3d9dff6, vsibench/utils.py) ======================= #

logger = logging.getLogger(__name__)

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


# lmms-eval's default VSI-Bench prompt kwargs. pre_prompt "" falls back to "These are frames of a video."
LMMS_DEFAULT_VSIBENCH_KW: dict[str, str] = {
    "pre_prompt": "",
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "na_post_prompt": "Please answer the question using a single word or phrase.",
}


def doc_to_text(doc, lmms_eval_specific_kwargs=LMMS_DEFAULT_VSIBENCH_KW) -> str:
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    if doc["question_type"] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def doc_to_visual(doc, videos_dir: str) -> str:
    """<videos_dir>/<dataset>/<scene_name>.mp4 (lmms-eval joins its yaml cache_dir the same way)."""
    path = os.path.join(videos_dir, doc["dataset"], doc["scene_name"] + ".mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(f"VSI-Bench video {path} does not exist.")
    return path


def fuzzy_matching(pred: str) -> str:
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / target


def mean_relative_accuracy(
    pred: float, target: float, start: float, end: float, interval: float
) -> float:
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


WORST_CASE_FOR_METRICS = {
    "accuracy": 0.0,
    "MRA:.5:.95:.05": 0.0,
}


def to_float(pred: Any) -> float | None:
    try:
        pred = float(pred)
    except BaseException:
        pred = None
    return pred


def vsibench_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    doc = dict(doc)
    doc["prediction"] = results[0]
    if doc["question_type"] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc["prediction"]), str(doc["ground_truth"]))
    elif doc["question_type"] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(
                    to_float(fuzzy_matching(doc["prediction"])),
                    to_float(doc["ground_truth"]),
                )
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {
        "vsibench_overall": doc,
        "obj_appearance_order_accuracy": doc,
        "object_abs_distance_mra": doc,
        "object_counting_mra": doc,
        "object_rel_distance_accuracy": doc,
        "object_size_estimation_mra": doc,
        "room_size_estimation_mra": doc,
        "route_planning_accuracy": doc,
        "object_rel_direction_accuracy": doc,
    }


def _compute_all_subscores(results: list[dict[str, Any]]) -> dict[str, float]:
    df = pd.DataFrame(results)
    output: dict[str, float] = {}

    for question_type, question_type_indexes in df.groupby("question_type").groups.items():
        per_question_type = df.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        else:
            raise ValueError(f"Unknown question type: {question_type}")

    dir_keys = (
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    )
    present_dirs = [k for k in dir_keys if k in output]
    if present_dirs:
        vals = [output.pop(k) for k in present_dirs]
        output["object_rel_direction_accuracy"] = sum(vals) / len(vals)

    output["overall"] = sum(output.values()) / len(output)
    return output


def vsibench_aggregate_overall(results: list[dict[str, Any]]) -> float:
    output = _compute_all_subscores(results)
    logger.info("VSI-Bench subscores: %s", output)
    return round(output["overall"], 6)


def vsibench_aggregate_obj_appearance_order_accuracy(results: list[dict[str, Any]]) -> float:
    return round(_compute_all_subscores(results).get("obj_appearance_order_accuracy", 0.0), 6)


def vsibench_aggregate_object_abs_distance_mra(results: list[dict[str, Any]]) -> float:
    return round(_compute_all_subscores(results).get("object_abs_distance_MRA:.5:.95:.05", 0.0), 6)


def vsibench_aggregate_object_counting_mra(results: list[dict[str, Any]]) -> float:
    return round(_compute_all_subscores(results).get("object_counting_MRA:.5:.95:.05", 0.0), 6)


def vsibench_aggregate_object_rel_distance_accuracy(results: list[dict[str, Any]]) -> float:
    return round(_compute_all_subscores(results).get("object_rel_distance_accuracy", 0.0), 6)


def vsibench_aggregate_object_size_estimation_mra(results: list[dict[str, Any]]) -> float:
    return round(
        _compute_all_subscores(results).get("object_size_estimation_MRA:.5:.95:.05", 0.0), 6
    )


def vsibench_aggregate_room_size_estimation_mra(results: list[dict[str, Any]]) -> float:
    return round(
        _compute_all_subscores(results).get("room_size_estimation_MRA:.5:.95:.05", 0.0), 6
    )


def vsibench_aggregate_route_planning_accuracy(results: list[dict[str, Any]]) -> float:
    return round(_compute_all_subscores(results).get("route_planning_accuracy", 0.0), 6)


def vsibench_aggregate_object_rel_direction_accuracy(results: list[dict[str, Any]]) -> float:
    return round(_compute_all_subscores(results).get("object_rel_direction_accuracy", 0.0), 6)


AGGREGATORS = (
    ("vsibench/overall", vsibench_aggregate_overall),
    ("vsibench/obj_appearance_order_accuracy", vsibench_aggregate_obj_appearance_order_accuracy),
    ("vsibench/object_abs_distance_mra", vsibench_aggregate_object_abs_distance_mra),
    ("vsibench/object_counting_mra", vsibench_aggregate_object_counting_mra),
    ("vsibench/object_rel_distance_accuracy", vsibench_aggregate_object_rel_distance_accuracy),
    ("vsibench/object_size_estimation_mra", vsibench_aggregate_object_size_estimation_mra),
    ("vsibench/room_size_estimation_mra", vsibench_aggregate_room_size_estimation_mra),
    ("vsibench/route_planning_accuracy", vsibench_aggregate_route_planning_accuracy),
    ("vsibench/object_rel_direction_accuracy", vsibench_aggregate_object_rel_direction_accuracy),
)


if __name__ == "__main__":
    if sys.argv[1:2] == ["merge"]:
        sys.exit(merge(sys.argv[2]))
    main()
