"""
Copied from lmms-eval lmms_eval/tasks/vsibench/utils.py (no runtime lmms-eval dependency).
See spatial_reasoning/lmms-eval for the upstream version.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from functools import partial


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
