"""Step-by-step (CoT) variant of det_vsibench_eval.py: detection list + the posed jpgs of meta.json (no mp4 decode),
gemma is told to reason with the box coordinates and end with a 'Final answer: X' line. No thinking mode.
Same output files as det_vsibench_eval.py, so scoring is `det_vsibench_eval.py --out_dir <dir> --merge --subset_ids ...`.

  python det_vsibench_eval_cot.py --out_dir results/det_vsibench/<run> --subset_ids <ids.json>
  python det_vsibench_eval_cot.py --subset_ids <ids.json> --compare <dir1> <dir2> ...   # per-category table
"""
import argparse, functools, json, re, time
from pathlib import Path
import pandas as pd

from det_vsibench_eval import W, AGGREGATORS, ask, build_question, det_to_nodes, extract_answer, load_llm, read_jsonl, score

COT_INSTRUCTION = (
    "Reason step by step before answering. Use the box coordinates (size and base) for every count, distance, size, "
    "direction and route: name the objects involved, write down their coordinates, and compute the answer from them "
    "(object centres are base + size/2). Use the video frames for what the list does not give you, e.g. the order in "
    "which objects first appear, or the room extent. Then end with exactly one line of the form 'Final answer: X' "
    "where X is the option letter for multiple choice or a single number for numeric questions.")


@functools.lru_cache(maxsize=1)  # questions are iterated scene by scene
def load_posed_frames(meta_path: str, img_root: str):
    """The posed jpgs listed in meta.json, in meta order (frame 0 = box origin), resized to 640x480."""
    from PIL import Image
    m = json.load(open(meta_path))
    return [Image.open(Path(img_root) / f["img_path"]).convert("RGB").resize((640, 480)) for f in m["frames"]]


def extract_final(text: str, qtype: str) -> str:
    """Read the last 'Final answer:' line; fall back to the det_vsibench_eval rules on the whole text."""
    lines = [l for l in text.splitlines() if re.match(r"\s*\**\s*final answer\b", l, re.IGNORECASE)]
    return extract_answer(lines[-1] if lines else text, qtype)


def compare(dirs, ids):
    """Per-category (n, score) table for results dirs answered on the same ids."""
    cols = {}
    for d in dirs:
        rows = {r["id"]: r for p in sorted(Path(d).glob("predictions_shard*.jsonl")) for r in read_jsonl(p)}
        missing = [i for i in ids if i not in rows]
        assert not missing, f"{d}: {len(missing)} ids missing"
        cols[d] = score([rows[i] for i in ids])
    n = pd.Series([re.sub(r"_(easy|medium|hard)$", "", rows[i]["question_type"]) for i in ids]).value_counts()
    print(f"{'category':32s} {'n':>4s} " + " ".join(f"{Path(d).parent.name[:22]:>22s}" for d in dirs))
    for tag, _ in AGGREGATORS:
        cat = tag.split("/")[1].rsplit("_", 1)[0]
        print(f"{cat:32s} {n.get(cat, len(ids)):4d} " + " ".join(f"{cols[d][tag]:22.3f}" for d in dirs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_parquet", default=f"{W}/dataset/vsib/VSI-Bench-extended/test_debiased_scannet.parquet")
    ap.add_argument("--det_root", default=f"{W}/dataset/vsib_det64f/scannet")
    ap.add_argument("--img_root", default=f"{W}/dataset", help="meta.json img_path is relative to this")
    ap.add_argument("--llm_model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out_dir")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--image_tokens", type=int, default=140, help="gemma-4 max_soft_tokens per frame")
    ap.add_argument("--subset_ids", default=None, help="json list of question ids; outputs go to <out_dir>/subset/")
    ap.add_argument("--compare", nargs="+", default=None, help="results dirs to tabulate on --subset_ids")
    a = ap.parse_args()
    df = pd.read_parquet(a.split_parquet)
    df = df[df["dataset"] == "scannet"].sort_values("id").reset_index(drop=True)
    ids = json.load(open(a.subset_ids)) if a.subset_ids else sorted(int(i) for i in df["id"])
    if a.compare:
        compare(a.compare, ids); return
    df = df[df["id"].isin(ids)].reset_index(drop=True)
    out = Path(a.out_dir) / ("subset" if a.subset_ids else "")
    out.mkdir(parents=True, exist_ok=True)

    preds_path = out / "predictions_shard0.jsonl"
    done = {r["id"] for r in read_jsonl(preds_path)}  # resume
    print(f"{len(df)} questions ({len(done)} done); loading {a.llm_model}", flush=True)
    proc, model = load_llm(a.llm_model, True, a.image_tokens)
    import torch
    nodes, t0, n_new = {}, time.time(), 0
    with preds_path.open("a") as f:
        for n, (_, row) in enumerate(df.sort_values(["scene_name", "id"]).iterrows(), 1):
            if int(row["id"]) in done:
                continue
            sc = row["scene_name"]
            if sc not in nodes:
                nodes[sc] = det_to_nodes(json.load(open(Path(a.det_root) / sc / "det.json")))
            frames = load_posed_frames(str(Path(a.det_root) / sc / "meta.json"), a.img_root)
            question = build_question(row).rsplit("\n", 1)[0] + "\n" + COT_INSTRUCTION  # drop 'answer directly' line
            ask._logged = False  # make ask() print input_len for every question
            raw, ans = ask(proc, model, nodes[sc], question, a.max_new_tokens, False, frames)
            f.write(json.dumps({"id": int(row["id"]), "scene_name": sc, "question_type": row["question_type"],
                                "prediction": extract_final(ans, row["question_type"]), "raw_output": raw,
                                "ground_truth": str(row["ground_truth"]), "model": a.llm_model}) + "\n"); f.flush()
            n_new += 1
            print(f"{n}/{len(df)} {(time.time() - t0) / n_new:.0f}s/q "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    rows = read_jsonl(preds_path)
    res = {"model": a.llm_model, "shard": 0, "num_shards": 1, "num_answered": len(rows), "thinking": False,
           "cot": True, "max_new_tokens": a.max_new_tokens, "video": "posed_jpg", "num_frames": 64,
           "image_tokens": a.image_tokens, "scores": score(rows),
           "sec_per_q": round((time.time() - t0) / max(n_new, 1), 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    (out / "results_shard0.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
