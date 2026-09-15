"""Check the det_es VGGT cache against the mix. Full scan: every vqa video in the mix has det_es
frames, a fresh cache entry (params incl. frames=det_es) and T == det_es frame count; sampled
entries are compared in dtype / patch count / dim with the same clip's entry in the old
(decoded-video) cache.

  python idea_3i_180k_es_64f_lat512_clean/verify_cache.py --mix_jsonl ... --vggt_cache_root ... [--old_cache_root ...]
"""
import argparse
import json
import random
import sys
from pathlib import Path

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vggt_cache  # noqa: E402
from collator import det_es_frames  # noqa: E402
from export_vggt_features import DEFAULT_VGGT_CHECKPOINT  # noqa: E402


def shapes(root, key):
    with safe_open(vggt_cache.cache_path(root, key), "pt") as f:
        p, c = f.get_slice("patch"), f.get_slice("camera")
        return tuple(p.get_shape()), tuple(c.get_shape()), p.get_dtype()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix_jsonl", required=True)
    p.add_argument("--vggt_cache_root", required=True)
    p.add_argument("--old_cache_root", default=None, help="vggt_cache_256_64f, for dtype/shape parity")
    p.add_argument("--video_max_frames", type=int, default=64)
    p.add_argument("--n_videos", type=int, default=6)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.mix_jsonl)]
    vqa = sorted({r["video"] for r in rows})
    params = vggt_cache.params_from(256, 1.0, args.video_max_frames, DEFAULT_VGGT_CHECKPOINT)
    print(f"mix: {len(rows)} rows, {len(vqa)} vqa videos; params {params}")
    bad = []
    for v in vqa:
        n = len(list((Path(v).with_suffix("") / "det_es" / "frames").glob("frame*.jpg")))
        if not n:
            bad.append((v, "no det_es frames")); continue
        if not vggt_cache.is_fresh(args.vggt_cache_root, v, params):
            bad.append((v, "vggt cache missing/stale")); continue
        (tp, _, _), (tc, _, _), _ = shapes(args.vggt_cache_root, v)
        if tp != n or tc != n:
            bad.append((v, f"T mismatch det_es {n} vggt {tp}/{tc}"))
    print(f"full scan: {len(bad)} problems")
    for v, why in bad[:20]:
        print("  ", why, v)

    random.seed(0)
    for v in random.sample(vqa, args.n_videos):
        fr = det_es_frames(v)
        line = f"{Path(v).parent.name}/{Path(v).name}: det_es {len(fr)} frames {fr[0].size} | new {shapes(args.vggt_cache_root, v)}"
        if args.old_cache_root:
            line += f" | old {shapes(args.old_cache_root, v)}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
