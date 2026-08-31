"""Training and VSI-Bench eval dataset construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from datasets import Dataset, load_dataset

from argument import DEFAULT_HF_HOME

_LMMS_EVAL_ROOT = Path(__file__).resolve().parents[2] / "lmms-eval"
if _LMMS_EVAL_ROOT.is_dir() and str(_LMMS_EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_LMMS_EVAL_ROOT))

from lmms_eval.tasks.vsibench.utils import vsibench_doc_to_text, vsibench_doc_to_visual  # noqa: E402

LMMS_DEFAULT_VSIBENCH_KW: Dict[str, str] = {
    "pre_prompt": "",
    "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
    "na_post_prompt": "Please answer the question using a single word or phrase.",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def is_scannet_record(record: Dict[str, Any]) -> bool:
    """Keep VSI-590K rows whose paths mention scannetppv2 (same as idea_2a_baseline)."""
    text_fields: List[str] = []
    for key in ("image", "video", "source", "dataset", "scene_id", "question_type"):
        value = record.get(key)
        if isinstance(value, str):
            text_fields.append(value)
    for turn in record.get("conversations") or []:
        if isinstance(turn, dict) and isinstance(turn.get("value"), str):
            text_fields.append(turn["value"])
    return "scannetppv2" in " ".join(text_fields).lower()


def record_source(record: Dict[str, Any]) -> str:
    """VSI-590K source dir of a row ("scannet", "scannetppv2", "arkitscenes", ...)."""
    media = record.get("video") or record.get("image") or ""
    return media.split("/")[0] if "/" in media else ""


def filter_train_records(
    jsonl_path: Path,
    sources: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Rows for training. ``sources=None`` keeps the legacy scannetppv2-only filter."""
    records = load_jsonl(jsonl_path)
    if sources is None:
        return [r for r in records if is_scannet_record(r)]
    wanted = set(sources)
    return [r for r in records if record_source(r) in wanted]


def build_train_dataset(jsonl_path: str, sources: Sequence[str] | None = None) -> Dataset:
    rows = filter_train_records(Path(jsonl_path), sources)
    return Dataset.from_list(rows)


def build_vsibench_eval_dataset(
    hf_home: str | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Debiased VSI-Bench test split, ScanNet++ only (path contains scannetpp)."""
    load_kwargs: Dict[str, Any] = {}
    if hf_home:
        load_kwargs["cache_dir"] = hf_home
    elif Path(DEFAULT_HF_HOME).exists():
        load_kwargs["cache_dir"] = DEFAULT_HF_HOME

    vsi_bench = load_dataset("nyu-visionx/VSI-Bench", "debiased", **load_kwargs)["test"]
    eval_rows: List[Dict[str, Any]] = []

    for ex in vsi_bench:
        visual_paths = vsibench_doc_to_visual(ex)
        if not visual_paths:
            continue
        if "scannetpp" not in str(visual_paths[0]).lower():
            continue
        user_text = vsibench_doc_to_text(ex, lmms_eval_specific_kwargs=LMMS_DEFAULT_VSIBENCH_KW)
        gt = ex["ground_truth"]
        eval_rows.append(
            {
                "video": visual_paths[0],
                "conversations": [
                    {"value": user_text},
                    {"value": str(gt)},
                ],
                "question_type": ex.get("question_type"),
                "ground_truth": gt,
                "source": "VSI-Bench",
            }
        )
        if max_samples is not None and len(eval_rows) >= max_samples:
            break

    return Dataset.from_list(eval_rows)
