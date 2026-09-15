"""Run every row of a mix through the training collator (built as train.py builds it) and check it.

Per row: det_es frames load (count recorded, < 64 flagged), collate succeeds, <|video_pad|> count
matches video_grid_thw, no frame resize, 512 / 32 VGGT placeholders, VGGT tensors [T, Np, d] /
[T, 17, d] with T = frame count, labels only on the answer span at the sequence end (decodes to
answer + <|im_end|>\\n), prompt contains the question. Failures go to failures_<shard>.jsonl,
counts to summary_<shard>.json; --summarize merges the summaries.

--stub_vggt replaces vggt_cache.load with zeros of the exported shapes (for machines without the
cache). Without it the real cache under --vggt_cache_root is read.

  HF_HUB_OFFLINE=1 python idea_3i_180k_es_64f_lat512_clean/check_mix_collator.py --mix <jsonl> \
      --out_dir <dir> --stub_vggt [--num_shards N --shard_index i] [--workers W] [--limit n]
  python idea_3i_180k_es_64f_lat512_clean/check_mix_collator.py --summarize <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR in sys.path:
    sys.path.remove(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)

VGGT_DIM = 2048
G = {}  # per-worker processor / collator / token ids


def stub_load(cache_root, key, params):
    """Zeros shaped like an exported entry: patch [T, Np, d], camera [T, 17, d]."""
    import torch
    from PIL import Image
    from export_vggt_features import extract_raw_video_tensors

    files = sorted((Path(key).with_suffix("") / "det_es" / "frames").glob("frame*.jpg"))
    h, w = extract_raw_video_tensors([Image.open(files[0]).convert("RGB")],
                                     params["vggt_image_resolution"])[0].shape[-2:]
    zero = torch.zeros(1, 1, VGGT_DIM)  # expand: right shape, no memory
    return zero.expand(len(files), (h // 16) * (w // 16), VGGT_DIM), zero.expand(len(files), 17, VGGT_DIM)


def init(args):
    os.environ["HF_HOME"] = args.hf_home
    import torch
    torch.set_num_threads(1)
    import vggt_cache
    from collator import make_collator
    from export_vggt_features import DEFAULT_VGGT_CHECKPOINT
    from transformers import AutoProcessor

    if args.stub_vggt:
        vggt_cache.load = stub_load
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = processor.tokenizer
    G.update(
        processor=processor,
        tok=tok,
        collate=make_collator(
            processor,
            data_root=Path(args.data_root),
            vggt_cache_root=Path(args.vggt_cache_root),
            vggt_checkpoint=args.vggt_checkpoint or DEFAULT_VGGT_CHECKPOINT,
            frame_placeholder_text="<|quad_start|>" * args.frame_num_latents,
            camera_placeholder_text="<|quad_end|>" * args.camera_num_latents,
            video_fps=1.0,
            video_max_frames=64,
            image_patch_size=16,
            vggt_image_resolution=256,
        ),
        ids={t: tok.convert_tokens_to_ids(t) for t in ("<|video_pad|>", "<|quad_start|>", "<|quad_end|>")},
        args=args,
    )


def check(item):
    i, row = item
    from PIL import Image

    args, tok, ids_of = G["args"], G["tok"], G["ids"]
    files = sorted((Path(row["video"]).with_suffix("") / "det_es" / "frames").glob("frame*.jpg"))
    rec = {"i": i, "video": row["video"], "question_type": row["question_type"], "source": row["source"],
           "frames": len(files), "seq_len": None, "errors": [], "flags": []}
    err = rec["errors"].append
    if len(files) < 64:
        rec["flags"].append(f"short_frames: {len(files)}")
    try:
        batch = G["collate"]([row])
    except Exception as exc:
        err(f"collate: {type(exc).__name__}: {str(exc)[:300]}")
        return rec

    ids, labels = batch["input_ids"][0], batch["labels"][0]
    rec["seq_len"] = len(ids)
    if not batch["attention_mask"].all():
        err("padding: attention_mask has zeros at batch size 1")

    thw = batch["video_grid_thw"]
    merge = G["processor"].video_processor.merge_size
    n_video = int((ids == ids_of["<|video_pad|>"]).sum())
    if n_video != int(thw.prod(-1).sum()) // merge ** 2:
        err(f"video_tokens: {n_video} <|video_pad|> vs grid {thw.tolist()}")
    t, h, w = thw[0].tolist()
    width, height = Image.open(files[0]).size if files else (0, 0)
    if (h * 16, w * 16) != (height, width):
        err(f"resized: grid {h}x{w} * 16 vs frame {height}x{width}")
    if thw.shape[0] != 1 or t != (len(files) + 1) // 2:
        err(f"grid_t: grid {thw.tolist()} vs {len(files)} frames")

    for token, n in (("<|quad_start|>", args.frame_num_latents), ("<|quad_end|>", args.camera_num_latents)):
        got = int((ids == ids_of[token]).sum())
        if got != n:
            err(f"vggt_placeholders: {got} {token}, expected {n}")
    patch, camera = batch["vggt_patch_tokens"][0], batch["vggt_camera_tokens"][0]
    if not (patch.ndim == camera.ndim == 3 and patch.shape[0] == camera.shape[0] == len(files)
            and camera.shape[1] == 17 and patch.shape[2] == camera.shape[2] == VGGT_DIM):
        err(f"vggt_shape: patch {tuple(patch.shape)} camera {tuple(camera.shape)} frames {len(files)}")

    pos = (labels != -100).nonzero().flatten().tolist()
    answer = row["conversations"][1]["value"]
    if not pos:
        err("labels: no supervised token")
        return rec
    if pos != list(range(pos[0], len(ids))):
        err(f"labels: span {pos[0]}..{pos[-1]} not contiguous to the end ({len(ids)})")
    if not (labels[pos] == ids[pos]).all():
        err("labels: label ids differ from input ids")
    supervised = tok.decode(ids[pos])
    if supervised != answer + "<|im_end|>\n":
        err(f"label_text: {supervised!r} vs answer {answer!r}")
    if row["conversations"][0]["value"] not in tok.decode(ids[:pos[0]]):
        err("prompt: question not in prompt")
    return rec


def summarize(out_dir: Path):
    tot = {"rows": 0, "passed": 0, "failed": 0, "flagged": 0, "seconds": 0.0, "max_seq_len": 0}
    cats, frames, examples = Counter(), Counter(), {}
    for f in sorted(out_dir.glob("summary_*.json")):
        s = json.load(open(f))
        for k in ("rows", "passed", "failed", "flagged", "seconds"):
            tot[k] += s[k]
        if s["max_seq_len"] > tot["max_seq_len"]:
            tot["max_seq_len"], tot["max_seq_len_row"] = s["max_seq_len"], s["max_seq_len_row"]
        cats.update(s["categories"])
        frames.update(s["frames"])
        for c, ex in s["examples"].items():
            examples.setdefault(c, []).extend(ex)
    tot.update(categories=dict(cats), examples={c: sorted(e)[:5] for c, e in examples.items()},
               frames=dict(sorted(frames.items(), key=lambda kv: int(kv[0]))))
    print(json.dumps(tot, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mix", type=Path)
    p.add_argument("--out_dir", type=Path)
    p.add_argument("--summarize", type=Path, help="merge summary_*.json in this dir and exit")
    p.add_argument("--stub_vggt", action="store_true", help="zeros instead of the VGGT cache")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="first n rows only (0 = all)")
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--hf_home", default=os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache"))
    p.add_argument("--data_root", default="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K")
    p.add_argument("--vggt_cache_root",
                   default=f"{os.environ.get('SCRATCH', '/home/ducpham/scratch/Working')}/dataset/vggt_cache_256_64f_es")
    p.add_argument("--vggt_checkpoint", default="", help="cache key; default export_vggt_features.DEFAULT_VGGT_CHECKPOINT")
    p.add_argument("--frame_num_latents", type=int, default=512)
    p.add_argument("--camera_num_latents", type=int, default=32)
    args = p.parse_args()
    if args.summarize:
        return summarize(args.summarize)

    rows = [json.loads(l) for l in open(args.mix)]
    items = list(enumerate(rows))[:args.limit or None][args.shard_index::args.num_shards]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start, n = time.time(), 0
    s = {"rows": len(items), "passed": 0, "failed": 0, "flagged": 0, "max_seq_len": 0, "max_seq_len_row": None}
    cats, frames, examples = Counter(), Counter(), {}
    with Pool(args.workers, initializer=init, initargs=(args,)) as pool, \
            open(args.out_dir / f"failures_{args.shard_index}.jsonl", "w") as fail:
        for rec in pool.imap_unordered(check, items, chunksize=4):
            n += 1
            frames[rec["frames"]] += 1
            s["failed" if rec["errors"] else "passed"] += 1
            s["flagged"] += bool(rec["flags"])
            if (rec["seq_len"] or 0) > s["max_seq_len"]:
                s["max_seq_len"], s["max_seq_len_row"] = rec["seq_len"], rec["i"]
            for e in rec["errors"] + rec["flags"]:
                c = e.split(":")[0]
                cats[c] += 1
                if len(examples.setdefault(c, [])) < 5:
                    examples[c].append(rec["i"])
            if rec["errors"] or rec["flags"]:
                fail.write(json.dumps(rec) + "\n")
                fail.flush()
            if n % 1000 == 0:
                print(f"{n}/{len(items)} rows, {s['failed']} failed, {time.time() - start:.0f}s", flush=True)
    s.update(seconds=time.time() - start, categories=dict(cats), frames=dict(frames), examples=examples)
    json.dump(s, open(args.out_dir / f"summary_{args.shard_index}.json", "w"), indent=1)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
