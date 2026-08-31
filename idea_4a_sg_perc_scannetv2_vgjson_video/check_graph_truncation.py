"""Does --max_graph_tokens actually cut any training target?

Two different mechanisms use that number, and they do opposite things:

- train.py's scannet_samples() DROPS a row whose target exceeds it (max_target_tokens).
- collator.py's truncate_graph_text() CUTS the target of a row that gets through.

So a row is either discarded or silently shortened, and a shortened one teaches the model
to stop mid-box. This counts both against the real tokenizer.

    python idea_4a_sg_perc_scannetv2_vgjson_video/check_graph_truncation.py
    python .../check_graph_truncation.py --limit 20000 --max-graph-tokens 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_TRAIN_JSON = "/home/ducpham/scratch/Working/dataset/vgllm_data/train/scannet_det_train_4frames.json"
_BASE = "Qwen/Qwen3-VL-2B-Instruct"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=_TRAIN_JSON)
    p.add_argument("--max-graph-tokens", type=int, default=4096)
    p.add_argument("--min-boxes", type=int, default=5, help="train.py's other filter")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from graph_vgllm import canonicalize

    tok = AutoTokenizer.from_pretrained(_BASE)
    rows = json.load(open(args.json))
    if args.limit:
        rows = rows[: args.limit]

    # Score exactly what training sees: the canonicalised assistant turn, after the
    # min_boxes filter that runs first.
    kept = [r for r in rows if len(r["boxes"]) >= args.min_boxes]
    texts = [canonicalize(r["conversations"][1]["value"]) for r in kept]

    lens = []
    for i in range(0, len(texts), args.batch):
        enc = tok(texts[i : i + args.batch], add_special_tokens=False)["input_ids"]
        lens.extend(len(x) for x in enc)
        if i % (args.batch * 40) == 0:
            print(f"  {i}/{len(texts)}", end="\r", flush=True)

    lens.sort()
    n = len(lens)
    over = sum(1 for x in lens if x > args.max_graph_tokens)

    def pct(q):
        return lens[min(n - 1, int(q * n))]

    print(f"\njson rows           : {len(rows)}")
    print(f"after min_boxes>={args.min_boxes}: {n}")
    print(f"target tokens       : min {lens[0]}  median {pct(0.5)}  p90 {pct(0.90)}  "
          f"p99 {pct(0.99)}  p99.9 {pct(0.999)}  max {lens[-1]}")
    print(f"\nover {args.max_graph_tokens} tokens : {over}  ({100.0 * over / n:.4f}%)")
    if over:
        worst = [x for x in lens if x > args.max_graph_tokens]
        print(f"  those rows run {worst[0]}..{worst[-1]} tokens "
              f"(up to {worst[-1] - args.max_graph_tokens} tokens past the cap)")
        print("  train.py DROPS these; they never reach the collator, so nothing is truncated"
              " -- but the corpus loses them.")
    else:
        print("  no row exceeds the cap: nothing dropped, nothing truncated.")

    head = max(1, int(0.001 * n))
    print(f"\nheadroom: the largest target uses {100.0 * lens[-1] / args.max_graph_tokens:.1f}% "
          f"of the budget; top 0.1% start at {lens[-head]} tokens.")


if __name__ == "__main__":
    main()
