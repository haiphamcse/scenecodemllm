"""Build the joint VQA + 3DOD training manifest: one shuffled jsonl, per-source budgets.

Same shape as cambrian-p's scripts/data/build_train_manifest.py -- mix OFFLINE into a
flat file rather than sampling online, so the trainer just reads rows and an epoch is a
fixed, inspectable thing. Changing the ratio means re-running this (seconds), not
reasoning about sampler state under DDP.

Each row is stamped with ``task``, which collator.py dispatches on:

  vqa  from VSI-590K's vsi_590k.jsonl  -> {task, video, conversations, question_type}
  det  from a scannet_det_*.json       -> {task, images, graph, scene}

det rows are canonicalised HERE (count first, objects largest-volume-first) exactly as
idea_4a's train.py does at load time, so the target text is identical to what the 3DOD-only
runs trained on. The frame paths are made absolute here too, since the two corpora live
under different roots and the collator sees one flat stream.

  python idea_3i_590k_joint/build_mix.py --vqa_rows 200000 --det_rows 100000
  python idea_3i_590k_joint/build_mix.py --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_vgllm import canonicalize, parse  # noqa: E402

DATA = Path("/home/ducpham/scratch/Working/dataset")
_DEFAULT_VQA = DATA / "vsi_590k/VSI-590K/vsi_590k.jsonl"
_DEFAULT_DET = DATA / "vgllm_data/train/scannet_det_train_32frames_bi1.json"
_DEFAULT_IMAGE_ROOT = DATA
_DEFAULT_OUT = DATA / "vgllm_data/joint/joint_train.jsonl"


def load_vqa(path: Path, video_root: Path):
    """VSI-590K rows, video-only (image sources cannot train the VGGT branch)."""
    rows = []
    for line in open(path):
        row = json.loads(line)
        video = row.get("video")
        if not video:
            continue
        rows.append({
            "task": "vqa",
            "video": str(video_root / video),
            "conversations": row["conversations"],
            "question_type": row.get("question_type"),
        })
    return rows


def load_det(path: Path, image_root: Path, min_boxes: int):
    """ScanNet 3DOD rows, canonicalised the way idea_4a's train.py canonicalises."""
    rows, dropped = [], 0
    for row in json.load(open(path)):
        graph = canonicalize(row["conversations"][1]["value"])
        if len(parse(graph)) < min_boxes:
            dropped += 1
            continue
        rows.append({
            "task": "det",
            "images": [str(image_root / r) for r in row["images"]],
            "graph": graph,
            "scene": Path(row["images"][0]).parent.name,
        })
    return rows, dropped


def take(rows, budget, rng):
    """``budget`` rows: all of them if 0 or oversized, else a seeded sample."""
    if not budget or budget >= len(rows):
        return list(rows)
    return rng.sample(rows, budget)


def selftest():
    """The mixer's own two invariants."""
    rng = random.Random(0)
    pool = list(range(100))
    assert len(take(pool, 0, rng)) == 100          # 0 means everything
    assert len(take(pool, 250, rng)) == 100        # oversized budget clamps
    assert len(take(pool, 10, rng)) == 10
    assert take(pool, 10, random.Random(1)) == take(pool, 10, random.Random(1))

    # canonicalize is what makes a det target match the 3DOD-only runs: count first,
    # then largest volume first. A mixed corpus that skipped it would train a different
    # target for the same data.
    raw = ('```json\n[\n\t{"label": "cup", "bbox_3d": [0,0,0, 0.1,0.1,0.1, 0,0,0]},\n'
           '\t{"label": "table", "bbox_3d": [1,0,0, 2.0,1.0,0.8, 0,0,0]}\n]```')
    out = canonicalize(raw)
    assert [o["label"] for o in parse(out)] == ["table", "cup"], out
    assert '"n": 2' in out, out
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vqa_jsonl", type=Path, default=_DEFAULT_VQA)
    p.add_argument("--det_json", type=Path, default=_DEFAULT_DET)
    p.add_argument("--video_root", type=Path,
                   default=DATA / "vsi_590k/VSI-590K")
    p.add_argument("--image_root", type=Path, default=_DEFAULT_IMAGE_ROOT)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--vqa_rows", type=int, default=0, help="0 = every row")
    p.add_argument("--det_rows", type=int, default=0, help="0 = every row")
    p.add_argument("--min_boxes", type=int, default=5,
                   help="det rows with fewer boxes are dropped; 5 is what idea_4a trained on")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return selftest()

    rng = random.Random(args.seed)
    vqa = load_vqa(args.vqa_jsonl, args.video_root)
    det, dropped = load_det(args.det_json, args.image_root, args.min_boxes)
    print(f"vqa pool {len(vqa)}, det pool {len(det)} (dropped {dropped} under "
          f"min_boxes={args.min_boxes})")

    mixed = take(vqa, args.vqa_rows, rng) + take(det, args.det_rows, rng)
    rng.shuffle(mixed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in mixed:
            f.write(json.dumps(row) + "\n")

    counts = Counter(r["task"] for r in mixed)
    share = {k: f"{100 * v / len(mixed):.1f}%" for k, v in counts.items()}
    print(f"wrote {len(mixed)} rows -> {args.out}")
    print(f"  {dict(counts)}  {share}")


if __name__ == "__main__":
    main()
