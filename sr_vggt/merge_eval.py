"""Merge eval.py shard (and part) predictions into one score.

The score is recomputed from the raw per-row predictions (averaging per-shard subscores over
unequal shards would be wrong). Exits non-zero unless every one of the 5,130 rows is present.

    python -m sr_vggt.merge_eval --eval_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .eval import SPLIT, read_rows, score

FULL_ROWS = 5130


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--expected_rows", type=int, default=FULL_ROWS)
    args = ap.parse_args()

    files = sorted(args.eval_dir.glob(f"predictions_{SPLIT}_shard*.jsonl"))
    rows = {}
    for f in files:
        got = read_rows(f)
        rows.update(got)  # index-keyed: a row written twice (shard + part) counts once
        print(f"  {f.name}: {len(got)} rows")
    if len(rows) != args.expected_rows:
        print(f"INCOMPLETE: {len(rows)}/{args.expected_rows} rows in {args.eval_dir}; not scoring.")
        return 1
    scores = score(rows.values())
    out = args.eval_dir / f"results_{SPLIT}_merged.json"
    out.write_text(json.dumps({"split": SPLIT, "num_samples": len(rows), "num_files": len(files),
                               "scores": scores}, indent=2))
    print(f"merged {len(rows)} rows from {len(files)} files -> {out}")
    for tag, v in scores.items():
        print(f"  {tag:44s} {v:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
