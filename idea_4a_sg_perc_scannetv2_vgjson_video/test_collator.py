"""Print the shape of every tensor make_collator() emits, for one batch.

Loads rows the same way train.py does (scannet_samples: frame-on-disk filter, min_boxes,
token budget), builds the real processor + collator, and dumps key/dtype/shape. Also
decodes the supervised span and checks it carries the row's graph text, which is the one
piece of collator logic that can silently go wrong (the delta offset at collator.py:289).

  conda activate vsibench_eval_full
  python test_collator.py                       # 4-frame json, batch of 1
  python test_collator.py --json /path/to/scannet_det_train_32frames_bi1.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
for _p in (_THIS_DIR.parent / "common", _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import DEFAULT_HF_HOME

os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)

from transformers import AutoProcessor  # noqa: E402

from collator import IGNORE_INDEX, make_collator  # noqa: E402
from train import _DEFAULT_TRAIN_JSON, scannet_samples  # noqa: E402

_DATA = Path("/home/ducpham/scratch/Working/dataset")


def describe(name, value, indent=""):
    if torch.is_tensor(value):
        print(f"{indent}{name:24s} {str(tuple(value.shape)):24s} {value.dtype}")
    elif isinstance(value, (list, tuple)):
        print(f"{indent}{name:24s} {type(value).__name__}[{len(value)}]")
        for i, v in enumerate(value):
            describe(f"[{i}]", v, indent + "  ")
    else:
        print(f"{indent}{name:24s} {type(value).__name__}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", default=_DEFAULT_TRAIN_JSON)
    p.add_argument("--image_root", default=str(_DATA))
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--min_boxes", type=int, default=5)
    p.add_argument("--max_graph_tokens", type=int, default=4096)
    p.add_argument("--frame_num_latents", type=int, default=128)
    p.add_argument("--camera_num_latents", type=int, default=32)
    p.add_argument("--image_patch_size", type=int, default=16)
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    samples = scannet_samples(
        Path(args.json),
        Path(args.image_root),
        tokenizer=processor.tokenizer,
        min_boxes=args.min_boxes,
        max_target_tokens=args.max_graph_tokens,
    )
    batch_rows = samples[: args.batch_size]
    print(f"\n{len(batch_rows)} row(s): "
          + ", ".join(f"{r['scene']} ({len(r['images'])} frames, {r['n_boxes']} boxes)"
                      for r in batch_rows))

    collate = make_collator(
        processor,
        data_root=Path(args.image_root),
        cache_root=Path(args.image_root),
        frame_placeholder_text="<|quad_start|>" * args.frame_num_latents,
        camera_placeholder_text="<|quad_end|>" * args.camera_num_latents,
        max_graph_tokens=args.max_graph_tokens,
        image_patch_size=args.image_patch_size,
    )
    batch = collate(batch_rows)

    print("\nbatch:")
    for key, value in batch.items():
        describe(key, value, "  ")

    print("\nlabel span:")
    labels = batch["labels"]
    for i, row in enumerate(batch_rows):
        keep = labels[i] != IGNORE_INDEX
        span = processor.tokenizer.decode(labels[i][keep])
        print(f"  row {i}: {int(keep.sum())} supervised tokens of {labels.shape[1]}")
        assert row["graph"] in span, (
            f"row {i}: supervised span does not contain the graph text.\n"
            f"span[:200]={span[:200]!r}\ngraph[:200]={row['graph'][:200]!r}"
        )
    print("  OK: every supervised span contains its row's graph text.")


if __name__ == "__main__":
    main()
