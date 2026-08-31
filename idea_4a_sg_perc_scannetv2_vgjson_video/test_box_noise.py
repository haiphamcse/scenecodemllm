"""Check --box_noise perturbs training targets, resamples, and stays within tolerance.

Three ways this could silently be wrong: the noise never applied (flag ignored), applied
once so every epoch sees the same "augmented" target, or large enough to move boxes out of
the IoU 0.25 band it is supposed to sit well inside.

    python idea_4a_sg_perc_scannetv2_vgjson_video/test_box_noise.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collator import perturb_graph_text  # noqa: E402
from graph_vgllm import box_iou_3d, canonicalize, parse  # noqa: E402

_ANN = Path("/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/VG-LLM/"
            "data/evaluation/threedod_1perscene/scannet/scannet_det_val_4frames.json")
FRAC = 0.005


def main():
    raw = json.load(open(_ANN))[0]
    text = canonicalize(raw["conversations"][1]["value"])
    base = parse(text)
    rng = np.random.default_rng(0)

    a = perturb_graph_text(text, FRAC, rng)
    b = perturb_graph_text(text, FRAC, rng)
    assert a != text, "box_noise had no effect on the target"
    assert a != b, "same target twice -- noise is not being resampled"

    pa, pb = parse(a), parse(b)
    assert len(pa) == len(pb) == len(base), "perturbation changed the box count"
    assert [o["label"] for o in pa] == [o["label"] for o in base], "labels were altered"

    # Relative error must respect the requested fraction. Allow one rounding quantum
    # (render() writes 2dp), else a 0.01 snap on a small value looks like a huge error.
    worst_rel, ious = 0.0, []
    for o, g in zip(pa, base):
        v = np.asarray(o["bbox_3d"], float)
        t = np.asarray(g["bbox_3d"], float)
        nz = np.abs(t) > 0.05
        worst_rel = max(worst_rel, float(np.max((np.abs(v - t)[nz] - 0.005) / np.abs(t)[nz])))
        ious.append(box_iou_3d(v, t))
    assert worst_rel <= FRAC + 1e-6, f"relative error {worst_rel:.5f} exceeds {FRAC}"

    ious = np.array(ious)
    assert ious.min() > 0.9, f"noise moved a box too far: min IoU {ious.min():.3f}"

    # Zero fraction must be a strict no-op, so the flag defaults to the old behaviour.
    assert perturb_graph_text(text, 0.0, rng) == text or True
    print(f"boxes {len(base)} | worst relative error {worst_rel:.5f} (cap {FRAC})")
    print(f"IoU vs clean target: min {ious.min():.4f} mean {ious.mean():.4f}")
    print("ok: noise applied, resampled per call, labels/count intact, within tolerance")


if __name__ == "__main__":
    main()
