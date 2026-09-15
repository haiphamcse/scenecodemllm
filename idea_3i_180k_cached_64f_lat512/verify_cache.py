"""Check the 64-frame caches against the live path the trainer would otherwise run.

1. Every clip in the mix: frame cache and VGGT cache exist, carry the run's params, and
   agree on the frame count (vqa: meta n_frames == patch T; det: T == len(images)).
2. Sampled videos: live decode -> Qwen tensor vs cached JPEGs -> Qwen tensor, same shape,
   pixel error is JPEG noise only. Live VGGT input T matches the cache.
3. --gpu: VGGT-Omega on the live frames vs the cached tokens (fp32, same as the export).

  python idea_3i_130k_joint_cached_64f/verify_cache.py --mix_jsonl ... --vggt_cache_root ... \
      --frame_cache_root ... --data_root ... [--gpu]
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vggt_cache  # noqa: E402
from collator import (  # noqa: E402
    _LLM_MAX_PIXELS, _LLM_MIN_PIXELS, _video_content_llm, frame_cache_llm, load_frames, task_of,
)
from export_vggt_features import (  # noqa: E402
    DEFAULT_VGGT_CHECKPOINT, encode, extract_raw_video_tensors, load_vggt,
)
from qwen_vl_utils import process_vision_info  # noqa: E402


def llm_tensor(frames):
    _, videos, _ = process_vision_info(
        [{"role": "user", "content": [_video_content_llm(frames)]}],
        return_video_kwargs=True, image_patch_size=16,
    )
    return videos[0]


def frame_dir(root, video):
    key = hashlib.sha1(video.encode()).hexdigest()
    return Path(root) / key[:2] / key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix_jsonl", required=True)
    p.add_argument("--vggt_cache_root", required=True)
    p.add_argument("--frame_cache_root", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--hf_home", default=None)
    p.add_argument("--video_fps", type=float, default=1.0)
    p.add_argument("--video_max_frames", type=int, default=64)
    p.add_argument("--n_videos", type=int, default=6)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--skip_full_scan", action="store_true")
    args = p.parse_args()
    if args.hf_home:
        os.environ.setdefault("HF_HOME", args.hf_home)

    rows = [json.loads(l) for l in open(args.mix_jsonl)]
    vqa = {r["video"]: r for r in rows if task_of(r) != "det"}
    det = {vggt_cache.clip_key(r, args.data_root): r for r in rows if task_of(r) == "det"}
    vparams = vggt_cache.params_from(256, args.video_fps, args.video_max_frames, DEFAULT_VGGT_CHECKPOINT)
    fparams = {"video_fps": args.video_fps, "video_max_frames": args.video_max_frames,
               "vggt_image_resolution": 256, "min_pixels": _LLM_MIN_PIXELS, "max_pixels": _LLM_MAX_PIXELS}
    print(f"mix: {len(vqa)} vqa videos, {len(det)} det scenes; params {fparams}")

    def vggt_T(key):
        with safe_open(vggt_cache.cache_path(args.vggt_cache_root, key), "pt") as f:
            return f.get_slice("patch").get_shape()[0], f.get_slice("camera").get_shape()[0]

    if not args.skip_full_scan:
        bad = []
        for v in vqa:
            d = frame_dir(args.frame_cache_root, v)
            try:
                meta = json.loads((d / "meta.json").read_text())
            except OSError:
                bad.append((v, "frame cache missing")); continue
            stale = {k for k in fparams if meta.get(k) != fparams[k]}
            if stale:
                bad.append((v, f"frame params {stale}")); continue
            if len(list((d / "llm").glob("*.jpg"))) != meta["n_frames"]:
                bad.append((v, "jpg count != n_frames")); continue
            if not vggt_cache.is_fresh(args.vggt_cache_root, v, vparams):
                bad.append((v, "vggt cache missing/stale")); continue
            tp, tc = vggt_T(v)
            if tp != meta["n_frames"] or tc != meta["n_frames"]:
                bad.append((v, f"T mismatch frames {meta['n_frames']} vggt {tp}/{tc}"))
        for k, r in det.items():
            if not vggt_cache.is_fresh(args.vggt_cache_root, k, vparams):
                bad.append((k, "det vggt cache missing/stale")); continue
            tp, _ = vggt_T(k)
            if tp != len(r["images"]):
                bad.append((k, f"det T {tp} != {len(r['images'])} images"))
        print(f"full scan: {len(bad)} problems")
        for v, why in bad[:20]:
            print("  ", why, v)

    random.seed(0)
    sample = random.sample(sorted(vqa), args.n_videos)
    agg = load_vggt(DEFAULT_VGGT_CHECKPOINT, "cuda", torch.float32) if args.gpu else None
    for v in sample:
        ex = vqa[v]
        live = load_frames(ex, args.data_root, args.video_fps, args.video_max_frames)
        cached = frame_cache_llm(Path(args.frame_cache_root), v, fparams)
        tl, tc = llm_tensor(live), llm_tensor(cached)
        vt = extract_raw_video_tensors(live, 256)[0]
        tp, _ = vggt_T(v)
        line = (f"{Path(v).parent.name}/{Path(v).name}: frames live {len(live)} cached {len(cached)} "
                f"vggt T {tp} (vggt input {tuple(vt.shape)}) | qwen live {tuple(tl.shape)} cached {tuple(tc.shape)}")
        if tl.shape == tc.shape:
            diff = (tl.float() - tc.float()).abs()
            line += f" | px abs diff mean {diff.mean():.2f} max {diff.max():.0f} (0-255)"
        else:
            line += " | SHAPE MISMATCH"
        if agg is not None:
            patch, camera = encode(agg, live, 256, "cuda", torch.float32)
            cp, cc = vggt_cache.load(args.vggt_cache_root, v, vparams)
            cos = torch.nn.functional.cosine_similarity(patch.flatten(0, 1), cp.flatten(0, 1), dim=-1)
            rel = (patch - cp).norm() / cp.norm()
            line += (f" | vggt live vs cache: shape {tuple(patch.shape)} vs {tuple(cp.shape)} "
                     f"rel err {rel:.2e} cos min {cos.min():.4f} camera rel err {((camera - cc).norm() / cc.norm()):.2e}")
        print(line, flush=True)
    if agg is not None:
        k, r = next(iter(det.items()))
        live = load_frames(r, args.data_root, args.video_fps, args.video_max_frames)
        patch, camera = encode(agg, live, 256, "cuda", torch.float32)
        cp, cc = vggt_cache.load(args.vggt_cache_root, k, vparams)
        rel = (patch - cp).norm() / cp.norm()
        print(f"det {Path(k).name}: {len(live)} images, vggt shape {tuple(patch.shape)} vs {tuple(cp.shape)} rel err {rel:.2e}")


if __name__ == "__main__":
    main()
