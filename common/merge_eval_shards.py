"""Merge sharded eval.py runs into one results file.

eval.py --num_shards N writes predictions_<split>_shard<i>ofN.jsonl per job. Each line
carries question_type / prediction / ground_truth, which is everything the scorer needs,
so the merged score is recomputed from the raw predictions rather than averaged from the
per-shard summaries (averaging subscores across unequal shards would be wrong).

  python idea_3i_590k_joint/merge_eval_shards.py --eval_dir <dir> --split full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

# vsibench_metrics is lmms-eval's scorer, vendored; eval.py scores with the same one.
from vsibench_metrics import AGGREGATORS, vsibench_process_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--split", default="full")
    args = ap.parse_args()

    shards = sorted(args.eval_dir.glob(f"predictions_{args.split}_shard*.jsonl"))
    if not shards:
        raise SystemExit(f"no shard files in {args.eval_dir} for split={args.split}")

    docs, seen = [], set()
    for f in shards:
        n = 0
        for line in f.open():
            row = json.loads(line)
            # Shards are contiguous slices, but a duplicate would silently skew the
            # score, so index-dedupe rather than trusting the split.
            if row["index"] in seen:
                continue
            seen.add(row["index"])
            docs.append(
                vsibench_process_results(
                    {"question_type": row["question_type"], "ground_truth": row["ground_truth"]},
                    [row["prediction"]],
                )["vsibench_overall"]
            )
            n += 1
        print(f"  {f.name}: {n} rows")

    scores = {tag: agg(docs) for tag, agg in AGGREGATORS}
    out = args.eval_dir / f"results_{args.split}_merged.json"
    out.write_text(json.dumps({"split": args.split, "num_samples": len(docs),
                               "num_shards": len(shards), "scores": scores}, indent=2))
    print(f"\nmerged {len(docs)} rows from {len(shards)} shards -> {out}")
    for tag, v in scores.items():
        print(f"  {tag:44s} {v:.4f}")


if __name__ == "__main__":
    main()
