"""VQA-only ~180k mix: VSI-590K (scannet / scannetppv2 / arkitscenes) + VLM-3R vsibench_train.

Per (source, question_type) budgets below, each drawn breadth-first over videos with
build_mix.take_per_video (every video keeps >= 1 row where possible). A budget above the
pool size takes the whole pool and reports the shortfall. VLM-3R rows are remapped to our
VSI-590K clips (_SRC_DIR) and stamped with a VSI-590K question_type from the question
template (_TYPE_RULES); rows without a clip are dropped. VLM-3R rows duplicating a chosen
VSI-590K row (same video, same normalized question) are dropped before sampling.
ReVSI clips are filtered; VSI-Bench clips are asserted absent. Output rows are the
130k_cached_64f schema (task, video, conversations, question_type) plus `source`.

  python idea_3i_180k_cached_64f_lat512/build_mix_180k.py --vsi_jsonl ... --vlm3r_dir ... \
      --data_root ... --revsi_dir ... --vsib_dir ... --out .../joint180k_vlm3r/vqa_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mix import take_per_video  # noqa: E402

VSI_BUDGETS = {
    "appearance_order": 13728, "absolute_count": 9726, "absolute_size_room": 4956,
    "absolute_size_object": 16534, "absolute_distance_object": 20000,
    "relative_distance_object": 20000, "relative_direction_object": 18000,
    "relative_size_object": 5000, "absolute_direction_object": 3000,
}  # = 110,944
VLM3R_BUDGETS = {
    "route_planning": 3663, "absolute_count": 9000, "absolute_size_room": 2000,
    "absolute_size_object": 8000, "absolute_distance_object": 16000,
    "relative_distance_object": 16000, "relative_direction_object": 14000,
}  # = 68,663
VSI_SOURCES = ("scannet", "scannetppv2", "arkitscenes")

# build_mix_vlm3r.py's rules, renamed to VSI-590K question_type values.
_TYPE_RULES = [
    ("absolute_count", r"How many"), ("absolute_size_room", r"size of this room"),
    ("absolute_size_object", r"longest dimension"), ("absolute_distance_object", r"direct distance between"),
    ("relative_distance_object", r"closest to the"),
    ("relative_direction_object", r"(positioned|standing).*(facing|looking at)"),
    ("appearance_order", r"appearance order"), ("route_planning", r"navigate to"),
]
_SRC_DIR = {"scannet": "scannet", "scannetpp": "scannetppv2", "arkitscenes": "arkitscenes"}


def question_type(text: str) -> str:
    return next((t for t, pat in _TYPE_RULES if re.search(pat, text, re.S)), "other")


def dedupe_key(row):
    q = row["conversations"][0]["value"].replace("<image>", "")
    return row["video"], " ".join(q.lower().split())


def load_vsi(path: Path, data_root: Path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        video = r.get("video")
        if not video or video.split("/")[0] not in VSI_SOURCES:
            continue
        rows.append({"task": "vqa", "video": str(data_root / video), "conversations": r["conversations"],
                     "question_type": r.get("question_type"), "source": "vsi590k"})
    return rows


def load_vlm3r(vlm3r_dir: Path, data_root: Path):
    rows, missing = [], Counter()
    for f in sorted(vlm3r_dir.glob("*.json")):
        for r in json.load(open(f)):
            video = data_root / _SRC_DIR[r["data_source"]] / f"{r['scene_name']}.mp4"
            if not video.exists():
                missing[r["data_source"]] += 1
                continue
            rows.append({"task": "vqa", "video": str(video), "conversations": r["conversations"],
                         "question_type": question_type(r["conversations"][0]["value"]), "source": "vlm3r"})
    return rows, missing


def draw(pool, budgets, rng, seen):
    """Per question_type budget over ``pool``; rows whose dedupe key is in ``seen`` are skipped."""
    chosen, stats = [], {}
    by_type = {}
    for r in pool:
        if r["question_type"] in budgets and dedupe_key(r) not in seen:
            seen.add(dedupe_key(r))
            by_type.setdefault(r["question_type"], []).append(r)
    for t, budget in budgets.items():
        p = by_type.get(t, [])
        got = take_per_video(p, budget, rng)
        stats[t] = {"taken": len(got), "pool": len(p), "budget": budget,
                    "videos": len({r["video"] for r in got}), "pool_videos": len({r["video"] for r in p})}
        chosen.extend(got)
    return chosen, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vsi_jsonl", type=Path, required=True)
    p.add_argument("--vlm3r_dir", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True, help="VSI-590K/ (video dirs under it)")
    p.add_argument("--revsi_dir", type=Path, required=True, help="ReVSI clips (*.mp4); their ids are dropped")
    p.add_argument("--vsib_dir", type=Path, required=True, help="VSI-Bench videos (recursive *.mp4); asserted absent")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)

    revsi = {f.stem for f in args.revsi_dir.glob("*.mp4")}
    vsib = {f.stem for f in args.vsib_dir.rglob("*.mp4")}
    assert revsi and vsib, (len(revsi), len(vsib))

    vsi = load_vsi(args.vsi_jsonl, args.data_root)
    vlm, missing = load_vlm3r(args.vlm3r_dir, args.data_root)
    print(f"pools: vsi590k {len(vsi)} rows, vlm3r {len(vlm)} rows (no clip: {dict(missing)}, "
          f"type=other: {sum(r['question_type'] == 'other' for r in vlm)})")
    drops = {"vlm3r_no_clip": sum(missing.values())}
    for name, pool in (("vsi590k", vsi), ("vlm3r", vlm)):
        assert not any(Path(r["video"]).stem in vsib for r in pool), f"{name} overlaps VSI-Bench"
        n = len(pool)
        pool[:] = [r for r in pool if Path(r["video"]).stem not in revsi]
        drops[f"{name}_revsi"] = n - len(pool)

    seen = set()
    vsi_rows, vsi_stats = draw(vsi, VSI_BUDGETS, rng, seen)
    drops["vsi590k_dup"] = sum(r["question_type"] in VSI_BUDGETS for r in vsi) - sum(s["pool"] for s in vsi_stats.values())
    vlm_rows, vlm_stats = draw(vlm, VLM3R_BUDGETS, rng, seen)
    drops["vlm3r_dup"] = sum(r["question_type"] in VLM3R_BUDGETS for r in vlm) - sum(s["pool"] for s in vlm_stats.values())

    mixed = vsi_rows + vlm_rows
    rng.shuffle(mixed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in mixed:
            f.write(json.dumps(r) + "\n")

    stats = {"total": len(mixed), "drops": drops,
             "vsi590k": vsi_stats, "vlm3r": vlm_stats,
             "shortfall": {f"{s}/{t}": v["budget"] - v["taken"] for s, st in (("vsi590k", vsi_stats), ("vlm3r", vlm_stats))
                           for t, v in st.items() if v["taken"] < v["budget"]}}
    json.dump(stats, open(args.out.parent / "mix_stats.json", "w"), indent=1)
    for s, st in (("vsi590k", vsi_stats), ("vlm3r", vlm_stats)):
        print(f"{s}: {sum(v['taken'] for v in st.values())} rows")
        for t, v in st.items():
            print(f"  {t:26s} {v['taken']:6d}/{v['budget']:6d} (pool {v['pool']}, videos {v['videos']}/{v['pool_videos']})")
    print(f"drops {drops}\nshortfall {stats['shortfall']}\nwrote {len(mixed)} rows -> {args.out}")

    # Self-check on the written file.
    rows = [json.loads(l) for l in open(args.out)]
    assert len(rows) == len(mixed)
    assert len({dedupe_key(r) for r in rows}) == len(rows), "duplicates"
    assert not any(Path(r["video"]).stem in revsi | vsib for r in rows), "ReVSI/VSI-Bench clip in mix"
    assert all(Path(r["video"]).exists() for r in rows), "missing mp4"
    c = Counter((r["source"], r["question_type"]) for r in rows)
    for s, budgets in (("vsi590k", VSI_BUDGETS), ("vlm3r", VLM3R_BUDGETS)):
        assert all(c[(s, t)] <= b for t, b in budgets.items()), (s, c)
        assert {t for (s_, t) in c if s_ == s} <= set(budgets), (s, c)
    assert set(rows[0]) == {"task", "video", "conversations", "question_type", "source"}
    print("self-check ok")


if __name__ == "__main__":
    main()
