"""Run the cached+FSDP trainer with CUDA memory history on; on OOM, dump who owns what.

Two runs OOM'd at the same point with no Python frame past _engine_run_backward, and
guessing the tensor from its byte count picked wrong twice. This attributes every live
block to the stack that allocated it, at the moment of failure.

  SR_ENTRY=profiling/memsnap_train.py bash scripts/idea_3i_590k_joint_cached_fsdp/jz_overfit_cached_fsdp.sh --max_steps 1
"""
import collections
import os
import pickle
import runpy
import sys
from pathlib import Path

import torch

rank = int(os.environ.get("LOCAL_RANK", "0"))
out = Path(os.environ.get("WORK", ".")) / "logs" / f"memsnap_rank{rank}.pickle"
torch.cuda.memory._record_memory_history(max_entries=200000)


def report():
    snap = torch.cuda.memory._snapshot()
    with open(out, "wb") as f:
        pickle.dump(snap, f)
    by_frame = collections.Counter()
    blocks = []
    for seg in snap["segments"]:
        for b in seg["blocks"]:
            if b["state"] != "active_allocated":
                continue
            frames = [f for f in b.get("frames", []) if "site-packages/torch/" not in f["filename"]]
            key = " <- ".join(f"{Path(f['filename']).name}:{f['line']} {f['name']}" for f in frames[:3]) or "?"
            by_frame[key] += b["size"]
            blocks.append((b["size"], key))
    print(f"\n=== rank {rank}: {sum(by_frame.values())/2**30:.2f} GiB live at OOM, by allocating stack ===", flush=True)
    for key, size in by_frame.most_common(14):
        print(f"{size/2**30:6.2f} GiB  {key}", flush=True)
    print(f"=== rank {rank}: largest single blocks ===", flush=True)
    for size, key in sorted(blocks, reverse=True)[:8]:
        print(f"{size/2**30:6.2f} GiB  {key}", flush=True)
    print(f"snapshot: {out}", flush=True)


try:
    sys.argv[0] = "idea_3i_590k_joint_cached_fsdp/train.py"
    runpy.run_path(sys.argv[0], run_name="__main__")
except torch.OutOfMemoryError:
    report()
    raise
