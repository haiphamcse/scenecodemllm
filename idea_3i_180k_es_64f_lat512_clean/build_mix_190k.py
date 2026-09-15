"""vqa_train_180k_es_clean.jsonl + new rows -> vqa_train_190k_es_clean.jsonl (190,000 rows).

Keeps every clean-mix row. New rows come from allowed scenes only: det_es/det.json present, not
in dropped_scenes.txt, not ReVSI (bench_ids.json) or VSI-Bench. Pools are deduped VSI-590K first,
then VLM-3R (a VLM-3R row duplicating a VSI-590K row is dropped), as in build_mix_180k.py.
Budget: relative_direction_object goes to 35,000 at the clean mix's VSI-590K : VLM-3R ratio; the
rest is split over the (source, type) cells with spare pool, proportional to their clean-mix
count, each capped at its spare pool (overflow goes round again). Each cell is drawn
breadth-first over videos with build_mix.take_per_video.

  python idea_3i_180k_es_64f_lat512_clean/build_mix_190k.py --vsi_jsonl ... --vlm3r_dir ... \
      --data_root ... --vsib_dir ... --bench_ids .../bench_ids.json --mix_dir .../idea_3i_dataset_jsons
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mix import take_per_video  # noqa: E402
from build_mix_180k import VLM3R_BUDGETS, VSI_BUDGETS, dedupe_key, load_vlm3r, load_vsi  # noqa: E402

TOTAL = 190_000
REL_DIR, REL_DIR_TOTAL = "relative_direction_object", 35_000
TYPES = {"vsi590k": VSI_BUDGETS, "vlm3r": VLM3R_BUDGETS}


def scene(r):
    v = Path(r["video"])
    return f"{v.parent.name}/{v.stem}"


def has_det_es(video):
    return (Path(video).with_suffix("") / "det_es" / "det.json").is_file()


def allocate(n, weight, cap):
    """``n`` rows over cells in proportion to ``weight``, each capped at ``cap``; overflow goes round again."""
    add = dict.fromkeys(weight, 0)
    while n:
        live = [c for c in weight if weight[c] and add[c] < cap[c]]
        assert live, f"spare pool too small, {n} rows left"
        tot = sum(weight[c] for c in live)
        share = {c: n * weight[c] / tot for c in live}
        got = {c: min(int(share[c]), cap[c] - add[c]) for c in live}
        if not any(got.values()):  # last few rows: one each, largest share first
            got.update({c: 1 for c in sorted(live, key=lambda c: -share[c])[:n]})
        for c in live:
            add[c] += got[c]
        n -= sum(got.values())
    return add


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vsi_jsonl", type=Path, required=True)
    p.add_argument("--vlm3r_dir", type=Path, required=True)
    p.add_argument("--data_root", type=Path, required=True, help="VSI-590K/ (video dirs under it)")
    p.add_argument("--vsib_dir", type=Path, required=True, help="VSI-Bench videos (recursive *.mp4)")
    p.add_argument("--bench_ids", type=Path, required=True, help="json with 'revsi': {source: [ids]}")
    p.add_argument("--mix_dir", type=Path, required=True,
                   help="reads vqa_train_180k_es[_clean].jsonl + dropped_scenes.txt, writes the 190k mix here")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = random.Random(args.seed)
    out = args.mix_dir / "vqa_train_190k_es_clean.jsonl"

    revsi = {i for ids in json.load(open(args.bench_ids))["revsi"].values() for i in ids}
    vsib = {f.stem for f in args.vsib_dir.rglob("*.mp4")}
    dropped = {l.split("\t")[0] for l in open(args.mix_dir / "dropped_scenes.txt") if l.strip()}
    assert revsi and vsib and len(dropped) == 18, (len(revsi), len(vsib), len(dropped))
    bench = revsi | vsib
    clean = [json.loads(l) for l in open(args.mix_dir / "vqa_train_180k_es_clean.jsonl")]
    es_videos = {json.loads(l)["video"] for l in open(args.mix_dir / "vqa_train_180k_es.jsonl")}
    clean_keys = {dedupe_key(r) for r in clean}

    vsi = load_vsi(args.vsi_jsonl, args.data_root)
    vlm, _ = load_vlm3r(args.vlm3r_dir, args.data_root)
    det = {v: has_det_es(v) for v in {r["video"] for r in vsi + vlm}}

    # Pools on allowed scenes, VSI-590K first so VLM-3R duplicates of it are dropped.
    seen, pools = set(), {}
    for r in vsi + vlm:
        if (r["question_type"] in TYPES[r["source"]] and det[r["video"]] and scene(r) not in dropped
                and Path(r["video"]).stem not in bench and dedupe_key(r) not in seen):
            seen.add(dedupe_key(r))
            pools.setdefault((r["source"], r["question_type"]), []).append(r)
    assert clean_keys <= seen, f"{len(clean_keys - seen)} clean-mix rows outside the pools"

    cells = sorted(pools)
    before = Counter((r["source"], r["question_type"]) for r in clean)
    spare = {c: [r for r in pools[c] if dedupe_key(r) not in clean_keys] for c in cells}
    cap = {c: len(spare[c]) for c in cells}

    rd = [("vsi590k", REL_DIR), ("vlm3r", REL_DIR)]
    rd_vsi = round(REL_DIR_TOTAL * before[rd[0]] / (before[rd[0]] + before[rd[1]]))
    added = {rd[0]: rd_vsi - before[rd[0]], rd[1]: REL_DIR_TOTAL - rd_vsi - before[rd[1]]}
    assert all(0 <= added[c] <= cap[c] for c in rd), (added, cap)
    rest = [c for c in cells if c not in rd]
    added.update(allocate(TOTAL - len(clean) - sum(added.values()),
                          {c: before[c] for c in rest}, {c: cap[c] for c in rest}))

    new = []
    for c in cells:
        if added[c]:  # take_per_video treats a 0 budget as "everything"
            got = take_per_video(spare[c], added[c], rng)
            assert len(got) == added[c], (c, len(got))
            new.extend(got)
    mixed = clean + new
    rng.shuffle(mixed)
    with open(out, "w") as f:
        for r in mixed:
            f.write(json.dumps(r) + "\n")

    stats = {"total": len(mixed), "seed": args.seed,
             "cells": {f"{s}/{t}": {"before": before[(s, t)], "added": added[(s, t)],
                                    "after": before[(s, t)] + added[(s, t)], "pool": len(pools[(s, t)])}
                       for s, t in cells}}
    json.dump(stats, open(args.mix_dir / "mix_stats_190k.json", "w"), indent=1)
    print(f"{'source/type':42s} {'before':>7s} {'added':>6s} {'after':>7s} {'pool':>7s}")
    for k, v in stats["cells"].items():
        print(f"{k:42s} {v['before']:7d} {v['added']:6d} {v['after']:7d} {v['pool']:7d}")
    print(f"{'total':42s} {len(clean):7d} {len(new):6d} {len(mixed):7d}\nwrote {len(mixed)} rows -> {out}")

    # Self-check on the written file.
    rows = [json.loads(l) for l in open(out)]
    assert len(rows) == TOTAL, len(rows)
    assert sum(r["question_type"] == REL_DIR for r in rows) == REL_DIR_TOTAL
    as_str = lambda rs: Counter(json.dumps(r, sort_keys=True) for r in rs)  # noqa: E731
    assert not as_str(clean) - as_str(rows), "clean-mix row missing or changed"
    assert len({dedupe_key(r) for r in rows}) == len(rows), "duplicates"
    assert not any(scene(r) in dropped or Path(r["video"]).stem in bench for r in rows), \
        "dropped/ReVSI/VSI-Bench scene in mix"
    videos = {r["video"] for r in rows}
    assert all(has_det_es(v) for v in videos), "video without det_es/det.json"
    assert videos <= es_videos, f"{len(videos - es_videos)} videos not in vqa_train_180k_es.jsonl: " \
                                f"{sorted(videos - es_videos)[:5]}"
    assert all(set(r) == {"task", "video", "conversations", "question_type", "source"} for r in rows)
    print("self-check ok")


if __name__ == "__main__":
    main()
