"""Trace-guided variant of det_vsibench_eval.py: detection list + the 64 posed jpgs + question + the answer-guided
reasoning trace exported by det_vsibench_reasoning_export.py (cleaned 'reasoning' field, ground truth withheld)
inserted as a "step-by-step guide". gemma answers directly (no CoT, no thinking). Traces flagged leaked are skipped.
Resumable per question; at the end results_merged.json (same scoring) and compare.txt (baselines on the same ids).

  python det_vsibench_eval_traceguided.py --trace_jsonl <reasoning.jsonl> --out_dir results/det_vsibench/<run> \
      --subset_ids <ids.json> [--baselines <dir1> <dir2>]
"""
import argparse, json, re, time
from pathlib import Path
import pandas as pd

from det_vsibench_eval import (W, AGGREGATORS, MCA_QUESTION_TYPES, SYSTEM_PROMPT, VIDEO_SUFFIX, build_question,
                               det_to_nodes, extract_answer, load_llm, read_jsonl, score)
from det_vsibench_eval_cot import load_posed_frames

GUIDE_HEADER = "Here is a step-by-step guide for answering this question:\n"
FINAL_MCA = "Follow the guide and answer with only the option letter."
FINAL_NA = "Follow the guide and answer with only a single number."


def build_user(nodes: str, row, trace: str) -> str:
    """User text: box list, question (+options), the guide block, the answer-format line."""
    question = build_question(row).rsplit("\n", 1)[0]  # drop the default 'answer directly' line
    final = FINAL_MCA if row["question_type"] in MCA_QUESTION_TYPES else FINAL_NA
    user = f"3D detections:\n{nodes}\n\nQuestion:\n{question}\n\n{GUIDE_HEADER}{trace}\n\n{final}"
    gt = str(row["ground_truth"])
    assert "Ground-truth" not in user, "export answer line leaked into the prompt"
    assert not re.search(rf"answer\s*(?:is|:)\s*\(?\**{re.escape(gt)}\b", user, re.IGNORECASE), f"answer {gt} in prompt"
    return user


def ask(proc, model, user: str, frames, max_new_tokens: int):
    """-> (raw_output, cleaned_text, n_in, n_out). Same message layout as det_vsibench_eval.ask (video mode)."""
    import torch
    note = (f"The {len(frames)} images above are video frames in temporal order; image 1 is the first video frame "
            "(the coordinate origin).\n\n")
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT + VIDEO_SUFFIX}]},
            {"role": "user", "content": [{"type": "image", "image": im} for im in frames]
             + [{"type": "text", "text": note + user}]}]
    inputs = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, add_generation_prompt=True,
                                      return_tensors="pt", enable_thinking=False)
    inputs = inputs.to(device=model.device, dtype=model.dtype)
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[0, n_in:]
    raw = proc.decode(gen, skip_special_tokens=False).strip()
    clean = proc.decode(gen, skip_special_tokens=True).strip()
    if hasattr(proc, "parse_response"):
        try:
            clean = (proc.parse_response(raw).get("content") or "").strip() or clean
        except Exception:
            pass
    return raw, clean, int(n_in), int(len(gen))


def load_preds(d: Path) -> dict:
    """id -> row from a results dir (shards, else predictions_merged.jsonl)."""
    files = sorted(d.glob("predictions_shard*.jsonl")) or [d / "predictions_merged.jsonl"]
    return {r["id"]: r for p in files for r in read_jsonl(p)}


def compare(named: dict, ids: list, qtype: dict) -> str:
    """Per-category (n, score) table for {name: id->row} on ids. Lines with an empty score are dirs lacking ids."""
    n = pd.Series([re.sub(r"_(easy|medium|hard)$", "", qtype[i]) for i in ids]).value_counts()
    cols = {k: score([rows[i] for i in ids]) if all(i in rows for i in ids) else {} for k, rows in named.items()}
    lines = [f"{'category':30s} {'n':>4s} " + " ".join(f"{k:>16s}" for k in named)]
    for tag, _ in AGGREGATORS:
        cat = tag.split("/")[1].rsplit("_", 1)[0]
        lines.append(f"{cat:30s} {n.get(cat, len(ids)):4d} "
                     + " ".join(f"{cols[k][tag]:16.3f}" if cols[k] else f"{'-':>16s}" for k in named))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_parquet", default=f"{W}/dataset/vsib/VSI-Bench-extended/test_debiased_scannet.parquet")
    ap.add_argument("--det_root", default=f"{W}/dataset/vsib_det64f/scannet")
    ap.add_argument("--img_root", default=f"{W}/dataset", help="meta.json img_path is relative to this")
    ap.add_argument("--llm_model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--trace_jsonl", required=True, help="reasoning.jsonl from det_vsibench_reasoning_export.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subset_ids", default=None, help="json list of question ids; outputs go to <out_dir>/subset/")
    ap.add_argument("--max_questions", type=int, default=None, help="smoke: stop after N new questions")
    ap.add_argument("--skip_leaked", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--image_tokens", type=int, default=140, help="gemma-4 max_soft_tokens per frame")
    ap.add_argument("--baselines", nargs="*", default=[], help="results dirs for compare.txt (name=dir or dir)")
    a = ap.parse_args()
    df = pd.read_parquet(a.split_parquet)
    df = df[df["dataset"] == "scannet"].sort_values("id").reset_index(drop=True)
    ids = json.load(open(a.subset_ids)) if a.subset_ids else sorted(int(i) for i in df["id"])
    df = df[df["id"].isin(ids)].reset_index(drop=True)
    out = Path(a.out_dir) / ("subset" if a.subset_ids else "")
    out.mkdir(parents=True, exist_ok=True)
    if a.subset_ids:
        (out / "subset_ids.json").write_text(json.dumps(ids))

    traces = {r["id"]: r for r in read_jsonl(Path(a.trace_jsonl))}
    missing = [i for i in ids if i not in traces]
    assert not missing, f"{len(missing)} ids without a trace, e.g. {missing[:10]}"
    skipped = df[df["id"].map(lambda i: a.skip_leaked and traces[i]["leaked"])]
    print("skipped (leaked) per type:", skipped["question_type"].value_counts().to_dict(), flush=True)
    df = df[~df["id"].isin(skipped["id"])]
    run_ids = sorted(int(i) for i in df["id"])
    (out / "run_ids.json").write_text(json.dumps(run_ids))

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
            if a.max_questions and n_new >= a.max_questions:
                break
            sc, tr = row["scene_name"], traces[int(row["id"])]
            if sc not in nodes:
                nodes[sc] = det_to_nodes(json.load(open(Path(a.det_root) / sc / "det.json")))
            frames = load_posed_frames(str(Path(a.det_root) / sc / "meta.json"), a.img_root)
            user = build_user(nodes[sc], row, tr["reasoning"])
            if n_new == 0:  # one full prompt for the smoke log
                print(f"----- prompt (id={row['id']}, {len(frames)} frames as [image]) -----\n{SYSTEM_PROMPT + VIDEO_SUFFIX}\n"
                      f"\n[image] x{len(frames)}\n{user}\n-----", flush=True)
            t1 = time.time()
            raw, ans, n_in, n_out = ask(proc, model, user, frames, a.max_new_tokens)
            f.write(json.dumps({"id": int(row["id"]), "scene_name": sc, "question_type": row["question_type"],
                                "prediction": extract_answer(ans, row["question_type"]), "raw_output": raw,
                                "ground_truth": str(row["ground_truth"]), "model": a.llm_model,
                                "trace_id": tr["id"], "leaked": tr["leaked"], "n_input_tokens": n_in,
                                "n_output_tokens": n_out, "seconds": round(time.time() - t1, 1)}) + "\n"); f.flush()
            n_new += 1
            print(f"{n}/{len(df)} id={row['id']} {row['question_type']} in={n_in} out={n_out} "
                  f"{time.time() - t1:.0f}s pred={extract_answer(ans, row['question_type'])!r} gt={row['ground_truth']} "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    rows = read_jsonl(preds_path)
    res = {"model": a.llm_model, "shard": 0, "num_shards": 1, "num_answered": len(rows), "thinking": False,
           "trace_guided": True, "max_new_tokens": a.max_new_tokens, "video": "posed_jpg", "num_frames": 64,
           "image_tokens": a.image_tokens, "skipped_leaked": int(len(skipped)), "scores": score(rows),
           "sec_per_q": round((time.time() - t0) / max(n_new, 1), 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
           "input_tokens": {"mean": int(sum(r["n_input_tokens"] for r in rows) / len(rows)),
                            "max": max(r["n_input_tokens"] for r in rows)},
           "output_tokens": {"mean": round(sum(r["n_output_tokens"] for r in rows) / len(rows), 1),
                             "max": max(r["n_output_tokens"] for r in rows)}}
    (out / "results_shard0.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if len(rows) < len(df):
        return
    (out / "results_merged.json").write_text(json.dumps({"model": a.llm_model, "num_answered": len(rows),
                                                          "scores": res["scores"]}, indent=2))
    named = {(b.split("=", 1)[0] if "=" in b else Path(b).parent.name[:16]): load_preds(Path(b.split("=", 1)[-1]))
             for b in a.baselines}
    named["trace_guided"] = {r["id"]: r for r in rows}
    qtype = dict(zip(df["id"].astype(int), df["question_type"]))
    qtype.update(zip(skipped["id"].astype(int), skipped["question_type"]))
    txt = (f"non-leaked ids (n={len(run_ids)})\n{compare(named, run_ids, qtype)}\n\n"
           f"full subset (n={len(ids)}), baselines only\n{compare(named, ids, qtype)}\n")
    (out / "compare.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
