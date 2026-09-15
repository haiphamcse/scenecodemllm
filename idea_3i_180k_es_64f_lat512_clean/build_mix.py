"""Build a VSI-590K VQA training manifest: one shuffled jsonl, per-source budgets.

Mixed OFFLINE into a flat file, so an epoch is a fixed, inspectable thing. Rows:
{task: "vqa", video, conversations, question_type}. build_mix_180k.py reuses take_per_video.

  python idea_3i_180k_es_64f_lat512_clean/build_mix.py --vqa_budgets scannet=52000,...
  python idea_3i_180k_es_64f_lat512_clean/build_mix.py --selftest
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATA = Path("/home/ducpham/scratch/Working/dataset")
_DEFAULT_VQA = DATA / "vsi_590k/VSI-590K/vsi_590k.jsonl"
_DEFAULT_OUT = DATA / "vgllm_data/joint130k/joint_train.jsonl"


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


def source_of(row) -> str:
    return Path(row["video"]).parent.name


def take_per_video(rows, budget, rng):
    """``budget`` rows spread over EVERY video, question types balanced within a video.

    Per video: rows are ordered round-robin over question types (1st of each type, 2nd of
    each, ...; seeded shuffle inside a type). Then rows are taken breadth-first across
    videos -- round r takes each video's r-th row -- until the budget is met, the last
    round being a seeded subset. So every video contributes >= 1 row (round 0), heavy
    videos are capped by the round the budget runs out on, and a video never spends its
    quota on one question type while it has others.
    """
    by_video = {}
    for r in rows:
        by_video.setdefault(r["video"], []).append(r)
    ordered = {}
    for v, vrows in by_video.items():
        by_type = {}
        for r in vrows:
            by_type.setdefault(r.get("question_type"), []).append(r)
        queues = [list(q) for q in by_type.values()]
        for q in queues:
            rng.shuffle(q)
        rr, i = [], 0
        while any(queues):
            q = queues[i % len(queues)]
            if q:
                rr.append(q.pop())
            i += 1
        ordered[v] = rr
    if not budget or budget >= len(rows):
        return list(rows)
    out, videos = [], sorted(ordered)
    for rnd in range(max(len(x) for x in ordered.values())):
        layer = [ordered[v][rnd] for v in videos if rnd < len(ordered[v])]
        if len(out) + len(layer) > budget:
            out.extend(rng.sample(layer, budget - len(out)))
            break
        out.extend(layer)
        if len(out) == budget:
            break
    return out


def parse_budgets(spec: str) -> dict:
    """'scannet=52000,scannetppv2=52000,arkitscenes=26000' -> {source: rows}."""
    return {k: int(v) for k, v in (item.split("=") for item in spec.split(",") if item)}


def take(rows, budget, rng):
    """``budget`` rows: all of them if 0 or oversized, else a seeded sample."""
    if not budget or budget >= len(rows):
        return list(rows)
    return rng.sample(rows, budget)


def selftest():
    # take_per_video: every video kept, exact budget, per-video type balance, deterministic.
    rng = random.Random(0)
    pool = [{"video": f"v{v}", "question_type": f"t{t}", "i": t * 100 + k}
            for v in range(5) for t in range(3) for k in range(v + 1)]   # video v has 3*(v+1) rows
    got = take_per_video(pool, 12, random.Random(0))
    assert len(got) == 12
    assert {r["video"] for r in got} == {f"v{v}" for v in range(5)}       # floor 1 for all
    for v in range(5):                                                    # round-robin over types
        types = [r["question_type"] for r in got if r["video"] == f"v{v}"]
        assert max(types.count(t) for t in set(types)) - min(types.count(t) for t in set(types)) <= 1, types
    assert [r["i"] for r in got] == [r["i"] for r in take_per_video(pool, 12, random.Random(0))]
    assert len(take_per_video(pool, 0, rng)) == len(pool)                 # 0 = everything
    single = [{"video": f"s{v}", "i": v * 10 + k} for v in range(3) for k in range(v * 3 + 1)]  # 1, 4, 7 rows
    got = take_per_video(single, 6, random.Random(0))
    assert sorted(Counter(r["video"] for r in got).values()) == [1, 2, 3], got     # breadth-first cap
    assert parse_budgets("a=1,b=20") == {"a": 1, "b": 20}

    pool = list(range(100))
    assert len(take(pool, 0, rng)) == 100          # 0 means everything
    assert len(take(pool, 250, rng)) == 100        # oversized budget clamps
    assert len(take(pool, 10, rng)) == 10
    assert take(pool, 10, random.Random(1)) == take(pool, 10, random.Random(1))
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vqa_jsonl", type=Path, default=_DEFAULT_VQA)
    p.add_argument("--video_root", type=Path,
                   default=DATA / "vsi_590k/VSI-590K")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument("--vqa_rows", type=int, default=0, help="0 = every row (ignored when --vqa_budgets is set)")
    p.add_argument("--vqa_sources", default="arkitscenes,scannet,scannetppv2",
                   help="VSI-590K video sources to keep (dir name under VSI-590K/)")
    p.add_argument("--vqa_budgets", default="scannet=52000,scannetppv2=52000,arkitscenes=26000",
                   help="rows per source, spread over every video of that source (take_per_video)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return selftest()

    rng = random.Random(args.seed)
    keep = set(args.vqa_sources.split(","))
    vqa = [r for r in load_vqa(args.vqa_jsonl, args.video_root) if source_of(r) in keep]
    if args.vqa_budgets:
        budgets = parse_budgets(args.vqa_budgets)
        assert set(budgets) <= keep, (set(budgets), keep)
        mixed = []
        for src, budget in budgets.items():
            pool = [r for r in vqa if source_of(r) == src]
            got = take_per_video(pool, budget, rng)
            vids = {r["video"] for r in pool}
            print(f"  {src}: {len(got)} of {len(pool)} rows, videos covered "
                  f"{len({r['video'] for r in got})}/{len(vids)}")
            mixed.extend(got)
    else:
        mixed = take(vqa, args.vqa_rows, rng)
    rng.shuffle(mixed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in mixed:
            f.write(json.dumps(row) + "\n")

    print(f"wrote {len(mixed)} rows -> {args.out}")
    qt = Counter((source_of(r), r.get("question_type")) for r in mixed)
    for src in sorted({k[0] for k in qt}):
        print(f"  {src} types: " + ", ".join(f"{t}={n}" for (s_, t), n in qt.most_common() if s_ == src))


if __name__ == "__main__":
    main()
