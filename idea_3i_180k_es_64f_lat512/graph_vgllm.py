"""VG-LLM 9-DoF JSON scene-graph parse + detection metric (CA-1M, camera-0 frame).

Replaces graph_toon.py for the ca1m_vgllm corpus, whose target is the fenced list
process_threedod.py:38-40 emits:

    ```json
    [
    	{"label": "chair", "bbox_3d": [x, y, z, dx, dy, dz, yaw, roll, pitch]},
    	...
    ]```

There are no object ids, so the TOON id-based metrics do not apply. This is VG-LLM's
own score (src/lmms_eval/tasks/threedod/utils.py:compute_ap): greedy per-category
matching at IoU 0.25, counted into tp/fp/fn -> precision / recall / f1.

3D IoU without pytorch3d: an oriented box is 6 halfspaces, the intersection of two
boxes is the 12-halfspace polytope, and scipy gives its volume (Chebyshev centre via
linprog -> HalfspaceIntersection -> ConvexHull). Euler is decoded intrinsic "ZXY", the
convention ca1m_vgllm/export_vgllm.py:to_camera encoded with, so pred and GT agree.

  python graph_vgllm.py
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation

IOU_THRESHOLD = 0.25


def parse(text: str) -> list:
    """VG-LLM JSON text -> [{'label': str, 'bbox_3d': (9,) float array}].

    Line-by-line rather than one json.loads over the block: a generating model emits
    prose, truncated lists and trailing junk, and one bad row must not void the rest.
    """
    objs = []
    for ln in text.splitlines():
        s = ln.strip().rstrip(",")
        if "bbox_3d" not in s or "label" not in s:
            continue
        try:
            item = json.loads(s)
            box = np.asarray(item["bbox_3d"], dtype=float)
        except Exception:  # malformed row -> skip, same as VG-LLM's own reader
            continue
        if box.shape != (9,) or not np.isfinite(box).all():
            continue
        objs.append({"label": str(item["label"]), "bbox_3d": box})
    return objs


def render(objs) -> str:
    """[{'label', 'bbox_3d'}] -> the fenced target, count first, in the given order."""
    items = [json.dumps({"n": len(objs)})]
    items += [
        json.dumps({"label": o["label"],
                    "bbox_3d": [round(float(v), 2) for v in o["bbox_3d"]]})
        for o in objs
    ]
    return "```json\n[\n\t" + ",\n\t".join(items) + "\n]```"


def canonicalize(text: str) -> str:
    """Rewrite a graph into the training convention: count first, largest object first.

    export_vgllm.py emits objects in instances.json order, which is arbitrary (measured
    Spearman |rho| ~0.13 against volume, ~0.17 against distance -- the noise floor). At
    position k the model then faces a near-uniform choice among every remaining object,
    so per-token loss on a long scene cannot fall however well it sees the room. Sorting
    by descending volume makes the sequence predictable, and the leading {"n": N} turns
    "when do I stop" from an implicit decision into a supervised one.

    Idempotent: parse() ignores the count entry, so re-canonicalising is a no-op.
    """
    objs = sorted(parse(text), key=lambda o: -float(np.prod(o["bbox_3d"][3:6])))
    return render(objs)


def _halfspaces(box) -> np.ndarray:
    """9-DoF box -> (6, 4) halfspaces [n, d] with n.x + d <= 0 inside."""
    centre, extent = np.asarray(box[:3], float), np.asarray(box[3:6], float)
    rot = Rotation.from_euler("ZXY", np.asarray(box[6:9], float)).as_matrix()
    rows = []
    for k in range(3):
        n = rot[:, k]
        half = extent[k] / 2.0
        rows.append([*n, -(n @ centre + half)])
        rows.append([*(-n), (n @ centre - half)])
    return np.array(rows)


def _polytope_volume(hs: np.ndarray) -> float:
    """Volume of {x : A x + b <= 0}, 0.0 if it has empty interior."""
    A, b = hs[:, :-1], hs[:, -1]
    # Chebyshev centre: maximise the inscribed radius r s.t. A x + |a_i| r <= -b.
    res = linprog(
        np.array([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.hstack([A, np.linalg.norm(A, axis=1, keepdims=True)]),
        b_ub=-b,
        bounds=[(None, None)] * 4,
    )
    if not res.success or res.x[3] <= 1e-9:
        return 0.0
    try:
        return float(ConvexHull(HalfspaceIntersection(hs, res.x[:3]).intersections).volume)
    except Exception:  # degenerate (coplanar) intersection -> zero volume
        return 0.0


def box_iou_3d(a, b) -> float:
    """3D IoU of two 9-DoF oriented boxes."""
    va, vb = float(np.prod(a[3:6])), float(np.prod(b[3:6]))
    if va <= 0 or vb <= 0:
        return 0.0
    inter = _polytope_volume(np.vstack([_halfspaces(a), _halfspaces(b)]))
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def compare(ref_text: str, new_text: str, iou_threshold: float = IOU_THRESHOLD) -> dict:
    """Score a generated graph against the reference. VG-LLM compute_ap, one scene.

    A prediction is a true positive if it shares its label with an unclaimed GT box at
    IoU > threshold; labels never match across categories, so a perfect box under the
    wrong name is a false positive AND leaves a false negative behind.
    """
    ref, new = parse(ref_text), parse(new_text)
    gt_by_label = defaultdict(list)
    for o in ref:
        gt_by_label[o["label"]].append(o["bbox_3d"])

    tp = fp = 0
    used = defaultdict(set)
    for pred in new:
        boxes = gt_by_label.get(pred["label"], [])
        best_iou, best_i = 0.0, -1
        for i, gt_box in enumerate(boxes):
            if i in used[pred["label"]]:
                continue
            iou = box_iou_3d(pred["bbox_3d"], gt_box)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_iou > iou_threshold:
            used[pred["label"]].add(best_i)
            tp += 1
        else:
            fp += 1

    fn = len(ref) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_ref_objs": len(ref),
        "n_new_objs": len(new),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _demo() -> None:
    unit = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    assert abs(box_iou_3d(unit, unit) - 1.0) < 1e-9
    # Half-overlap along x: intersection 0.5, union 1.5.
    assert abs(box_iou_3d(unit, [0.5, 0, 0, 1, 1, 1, 0, 0, 0]) - 1 / 3) < 1e-6
    # 45 deg about z: intersection area 2(sqrt2 - 1) -> IoU = 1/sqrt2.
    assert abs(box_iou_3d(unit, [0, 0, 0, 1, 1, 1, np.pi / 4, 0, 0]) - 2**-0.5) < 1e-6
    assert box_iou_3d(unit, [5, 0, 0, 1, 1, 1, 0, 0, 0]) == 0.0

    # table volume 1.92 > chair volume 1.0, so canonical order is table then chair.
    a = ('```json\n[\n\t{"label": "chair", "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]},\n'
         '\t{"label": "table", "bbox_3d": [3.0, 0.0, 0.0, 1.6, 0.8, 1.5, 0.1, 0.0, 0.0]}\n]```')
    assert compare(a, a)["f1"] == 1.0

    # Right box, wrong label: a false positive that also strands a false negative.
    wrong_name = a.replace('"chair"', '"sofa"')
    m = compare(a, wrong_name)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1), m

    # Same label, box moved well clear: below threshold.
    moved = a.replace("[0.0, 0.0, 0.0, 1.0", "[9.0, 0.0, 0.0, 1.0")
    assert compare(a, moved)["tp"] == 1, compare(a, moved)

    assert compare(a, "")["f1"] == 0.0
    assert compare(a, "I cannot determine the boxes.")["n_new_objs"] == 0
    # A truncated generation keeps the rows it did finish.
    assert len(parse(a[: a.index("\n\t{\"label\": \"table\"")])) == 1

    # canonicalize: count first, largest object first, box set preserved.
    c = canonicalize(a)
    assert c.splitlines()[2].strip().rstrip(",") == '{"n": 2}', c
    assert [o["label"] for o in parse(c)] == ["table", "chair"], "not sorted by volume"
    assert compare(a, c)["f1"] == 1.0, "reordering must not change the box set"
    assert canonicalize(c) == c, "canonicalize must be idempotent"
    # The count entry is invisible to the metric: it has no bbox_3d.
    assert compare(c, c)["n_ref_objs"] == 2
    print("ok: iou analytic, self-f1=1.0, wrong-label=fp+fn, empty=0.0, "
          "truncation survives, canonicalize sorts+counts+idempotent")


if __name__ == "__main__":
    _demo()
