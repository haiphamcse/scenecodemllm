"""Export gemma-4 reasoning traces for VSI-Bench-Debiased (ScanNet): box list + 64 posed frames + question + the
GROUND-TRUTH answer -> 'explain how one would work this out' with the answer withheld. No thinking, greedy.
Resumable per question (reasoning.jsonl); one text file per question in per_question/; summary.json at the end.

  python det_vsibench_reasoning_export.py --out_dir results/det_vsibench_reasoning/<run> --subset_ids <ids.json>
  python det_vsibench_reasoning_export.py --out_dir <same dir>            # all 935, resumes what the subset did
"""
import argparse, functools, json, re, time
from pathlib import Path
import pandas as pd

from det_vsibench_eval import W, MCA_QUESTION_TYPES, build_question, det_to_nodes, load_llm, read_jsonl

SYSTEM_PROMPT = (
    "You are a spatial-reasoning tutor. You are given a list of 3D object detections from a video of an indoor "
    "scene, one per line as '<id> <label> size=(dx,dy,dz) base=(x,y,z)'. size is the object's axis-aligned extent in "
    "meters along (x,y,z); base is its minimum corner (x_min,y_min,z_min) in meters. Coordinates are measured from "
    "the camera position at the FIRST video frame: x grows to the right of that view, y grows downward (toward the "
    "floor), z grows forward (away from the camera). So dy is an object's height, dx and dz its footprint, and the "
    "floor is near the largest y values. Object centres are base + size/2. Distances are ordinary 3D Euclidean "
    "distances in meters. The list covers the whole scene, one entry per object; it does not say in which frame an "
    "object appears. You are also given frames of the video, in temporal order; frame 1 is the first video frame, "
    "i.e. the camera whose position defines the coordinates.")
INSTRUCTION = (
    "Explain step by step how someone could work out this answer from the object list and the video frames: which "
    "objects matter, what to look at or compute, and why. Do NOT state the final answer, the option letter, or the "
    "final number anywhere in your explanation.")
# stronger wording, tried only if the default leaks on >20% (--strict)
INSTRUCTION_STRICT = INSTRUCTION + (
    " This is a strict rule: the reader must be able to finish the last step themselves. Never write the option "
    "letter of the answer, never quote the answer's option text as a conclusion, and never write the resulting "
    "number, not even rounded or as an intermediate result. Stop right before the conclusion.")


@functools.lru_cache(maxsize=1)  # questions are iterated scene by scene
def load_posed_frames(meta_path: str, img_root: str):
    """The posed jpgs listed in meta.json, in meta order (frame 0 = box origin), resized to 640x480."""
    from PIL import Image
    m = json.load(open(meta_path))
    return [Image.open(Path(img_root) / f["img_path"]).convert("RGB").resize((640, 480)) for f in m["frames"]]


def answer_text(row) -> str:
    """'B (sofa, window, table, closet)' for MCA, the number as given otherwise."""
    gt = str(row["ground_truth"])
    if row["question_type"] in MCA_QUESTION_TYPES:
        opt = next(o for o in row["options"] if o.startswith(gt + "."))
        return f"{gt} ({opt[2:].strip()})"
    return gt


def leak(text: str, row) -> str:
    """Snippet around the first place the reasoning gives the answer away, '' if none. Letter: answer-like patterns
    ('answer is B', 'option B', '(B)', '**B**'); option text anywhere if it is 3+ words (appearance-order lists),
    else only inside an answer phrase (bare 'left'/'bed' would flag every honest explanation); numeric: the number
    or its 1-decimal rounding, not as an object id."""
    gt = str(row["ground_truth"])
    if row["question_type"] in MCA_QUESTION_TYPES:
        opt = next(o for o in row["options"] if o.startswith(gt + "."))[2:].strip()
        pats = [rf"(?:answer|option|choice|correct)\w*\s*(?:is|:|=|would be|must be)?\s*\(?\**{gt}\**\)?\b",
                rf"\({gt}\)", rf"\*\*{gt}\*\*", rf"^\s*{gt}[.)]",
                (r"" if len(opt.split()) >= 3 else r"(?:answer|correct option|correct choice|conclusion)\w*\s*(?:is|:|=|would be|must be)[^.\n]{0,20}")
                + rf"\b{re.escape(opt)}\b"]
    else:
        nums = {gt, f"{float(gt):.1f}", f"{float(gt):g}"}
        pats = [rf"(?<![\d._^-])(?:{'|'.join(re.escape(n) for n in nums)})(?![\d]|\.\d|D\b)"]  # not '3D', not the 2 of 2.15
    for p in pats:
        for m in re.finditer(p, text, re.IGNORECASE | re.MULTILINE):
            before, after = text[max(0, m.start() - 8):m.start()], text[m.end():m.end() + 2]
            if re.search(r"(?:object|box|node|id|#)\s*$", before, re.IGNORECASE):
                continue  # an object id
            if re.search(r"(?:^|\n)\s*$", before) and after.startswith(". "):
                continue  # markdown list numbering
            if re.search(r"[,(]\s*$", before) and after[:1] in ",)":
                continue  # inside an id list '(3, 7, 23)'
            return text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
    return ""


def ask(proc, model, nodes, question, answer, frames, instruction, max_new_tokens):
    """-> (raw_output, cleaned_text, n_in, n_out)."""
    import torch
    note = (f"The {len(frames)} images above are video frames in temporal order; image 1 is the first video frame "
            "(the coordinate origin).\n\n")
    user = f"3D detections:\n{nodes}\n\nQuestion:\n{question}\n\nGround-truth answer: {answer}\n\n{instruction}"
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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


def write_txt(out: Path, rec: dict, row):
    (out / "per_question" / f"{rec['id']}.txt").write_text(
        f"[{rec['scene_name']}] {rec['question_type']} id={rec['id']}\n\n{build_question(row).rsplit(chr(10), 1)[0]}"
        f"\n\nGround-truth answer: {answer_text(row)}\n\n--- reasoning (leaked={rec['leaked']}) ---\n{rec['reasoning']}\n")


def summarize(rows, max_new_tokens, peak_gb):
    df = pd.DataFrame(rows)
    per_type = {t: {"n": int(len(g)), "leaked": int(g["leaked"].sum()),
                    "truncated": int((g["n_output_tokens"] >= max_new_tokens).sum())}
                for t, g in df.groupby("question_type")}
    return {"n": len(df), "per_type": per_type, "leaked": int(df["leaked"].sum()),
            "truncated": int((df["n_output_tokens"] >= max_new_tokens).sum()),
            "mean_seconds": round(float(df["seconds"].mean()), 1),
            "input_tokens": {"mean": int(df["n_input_tokens"].mean()), "max": int(df["n_input_tokens"].max())},
            "output_tokens": {"mean": int(df["n_output_tokens"].mean()), "max": int(df["n_output_tokens"].max())},
            "peak_mem_gb_this_run": peak_gb, "max_new_tokens": max_new_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_parquet", default=f"{W}/dataset/vsib/VSI-Bench-extended/test_debiased_scannet.parquet")
    ap.add_argument("--det_root", default=f"{W}/dataset/vsib_det64f/scannet")
    ap.add_argument("--img_root", default=f"{W}/dataset", help="meta.json img_path is relative to this")
    ap.add_argument("--llm_model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subset_ids", default=None, help="json list of question ids; default: all questions")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--image_tokens", type=int, default=140, help="gemma-4 max_soft_tokens per frame")
    ap.add_argument("--strict", action="store_true", help="use INSTRUCTION_STRICT")
    ap.add_argument("--recheck", action="store_true", help="no GPU: recompute leaked/leak_match + summary.json")
    a = ap.parse_args()
    df = pd.read_parquet(a.split_parquet)
    df = df[df["dataset"] == "scannet"].sort_values("id").reset_index(drop=True)
    if a.subset_ids:
        df = df[df["id"].isin(json.load(open(a.subset_ids)))].reset_index(drop=True)
    out = Path(a.out_dir); (out / "per_question").mkdir(parents=True, exist_ok=True)
    path = out / "reasoning.jsonl"
    if a.recheck:
        rows, byid = read_jsonl(path), df.set_index("id")
        for r in rows:
            r["leak_match"] = leak(r["reasoning"], byid.loc[r["id"]]); r["leaked"] = bool(r["leak_match"])
            write_txt(out, r, byid.loc[r["id"]])
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        summ = summarize(rows, a.max_new_tokens, None)
        (out / "summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=2)); return
    done = {r["id"] for r in read_jsonl(path)}  # resume
    print(f"{len(df)} questions ({len(done)} done); loading {a.llm_model}", flush=True)
    proc, model = load_llm(a.llm_model, True, a.image_tokens)
    import torch
    instruction = INSTRUCTION_STRICT if a.strict else INSTRUCTION
    nodes, t0, n_new = {}, time.time(), 0
    with path.open("a") as f:
        for n, (_, row) in enumerate(df.sort_values(["scene_name", "id"]).iterrows(), 1):
            if int(row["id"]) in done:
                continue
            sc = row["scene_name"]
            if sc not in nodes:
                nodes[sc] = det_to_nodes(json.load(open(Path(a.det_root) / sc / "det.json")))
            frames = load_posed_frames(str(Path(a.det_root) / sc / "meta.json"), a.img_root)
            question = build_question(row).rsplit("\n", 1)[0]  # drop the 'answer directly' line
            answer = answer_text(row)
            t1 = time.time()
            raw, clean, n_in, n_out = ask(proc, model, nodes[sc], question, answer, frames, instruction, a.max_new_tokens)
            hit = leak(clean, row)
            rec = {"id": int(row["id"]), "scene_name": sc, "question_type": row["question_type"],
                   "question": row["question"], "options": None if row["options"] is None else list(row["options"]),
                   "ground_truth": str(row["ground_truth"]), "reasoning": clean, "raw_output": raw,
                   "leaked": bool(hit), "leak_match": hit, "n_input_tokens": n_in, "n_output_tokens": n_out,
                   "seconds": round(time.time() - t1, 1)}
            f.write(json.dumps(rec) + "\n"); f.flush()
            write_txt(out, rec, row)
            n_new += 1
            print(f"{n}/{len(df)} id={rec['id']} {row['question_type']} in={n_in} out={n_out} "
                  f"{rec['seconds']:.0f}s leaked={rec['leaked']} peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB",
                  flush=True)
    summ = summarize(read_jsonl(path), a.max_new_tokens, round(torch.cuda.max_memory_allocated() / 2**30, 1))
    summ["total_seconds_this_run"] = round(time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
