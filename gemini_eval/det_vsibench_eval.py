"""VSI-Bench-Debiased (ScanNet) answered by a text LLM that sees ONLY a 3D detection list.

Input per scene: <det_root>/<scene>/det.json (posed-64f format: boxes = 9-DoF ZXY in the camera of frame 0, VG-LLM
convention). Each rotated box becomes one v2 node line from its AABB: '<id> <label> size=(dx,dy,dz) base=(x,y,z)'.
LLM load / generation / answer extraction / scoring follow ScanEdit/scanedit_vsibench_eval_gemma{,_v2}.py.

  python det_vsibench_eval.py --llm_model google/gemma-4-26B-A4B-it --out_dir results_llm/det_gemma \
      --num_shards 4 --shard_index 0 --no-thinking          # -> predictions_shard0.jsonl, results_shard0.json
  python det_vsibench_eval.py --out_dir results_llm/det_gemma --merge   # union of all shards, refuses if incomplete
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from vsibench_metrics import AGGREGATORS, MCA_QUESTION_TYPES, NA_QUESTION_TYPES, vsibench_process_results  # noqa: E402

W = "/home/ducpham/scratch/Working"
SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant. You are given a list of 3D object detections from a video of an indoor "
    "scene, one per line as '<id> <label> size=(dx,dy,dz) base=(x,y,z)'. size is the object's axis-aligned extent in "
    "meters along (x,y,z); base is its minimum corner (x_min,y_min,z_min) in meters. All coordinates are in the camera "
    "frame of the video's FIRST frame (OpenCV convention: origin at that camera, x to the right, y down, z forward "
    "along the viewing direction; the floor is at large positive y and gravity points roughly along +y). Use only "
    "this list to answer the question.")
NOTHINK_MAX_NEW_TOKENS, THINK_MAX_NEW_TOKENS = 4096, 16384
CORNER = np.array([[sx, sy, sz] for sx in (-.5, .5) for sy in (-.5, .5) for sz in (-.5, .5)])


def det_to_nodes(det: dict) -> str:
    """boxes -> v2 node block; AABB of each rotated 9-DoF box (centre, size, ZXY euler), 2-decimal metres."""
    lines = ["nodes:"]
    for i, b in enumerate(det["boxes"]):
        c, s, r = (np.array(b["bbox_3d"][k:k + 3], float) for k in (0, 3, 6))
        pts = c + (Rotation.from_euler("ZXY", r).as_matrix() @ (CORNER * s).T).T
        lo, hi = pts.min(0), pts.max(0)
        lines.append(f"{i} {b['label']} size=({hi[0]-lo[0]:.2f},{hi[1]-lo[1]:.2f},{hi[2]-lo[2]:.2f}) "
                     f"base=({lo[0]:.2f},{lo[1]:.2f},{lo[2]:.2f})")
    return "\n".join(lines)


def build_question(row) -> str:
    """lmms-eval VSI-Bench question text (video pre-prompt omitted)."""
    if row["question_type"] in NA_QUESTION_TYPES:
        return row["question"] + "\nPlease answer the question using a single word or phrase."
    return "\n".join([row["question"], "Options:\n" + "\n".join(list(row["options"])),
                      "Answer with the option's letter from the given choices directly."])


# ---- answer extraction: the scorer reads only the first whitespace token, so pull the final answer out of any CoT
_NUM = r"[-+]?\d*\.?\d+"


def extract_answer(text: str, qtype: str) -> str:
    if qtype in MCA_QUESTION_TYPES:
        for pat in (r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\s*\(?\**([A-H])\**\)?\b",
                    r"\\boxed\{\s*([A-H])\s*\}", r"\*\*\s*([A-H])\s*\*\*"):
            m = re.findall(pat, text, re.IGNORECASE)
            if m:
                return m[-1].upper()
        m = re.findall(r"\b([A-H])\b", text)
        return m[-1] if m else text.split(" ")[0]
    for pat in (r"\\boxed\{\s*(" + _NUM + r")", r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\D{0,15}?(" + _NUM + r")"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1]
    nums = re.findall(_NUM, text)
    return nums[-1] if nums else text.split(" ")[0]


# ---- LLM
def load_llm(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    try:
        proc = AutoProcessor.from_pretrained(name)
    except Exception:
        proc = AutoTokenizer.from_pretrained(name)
    kw = {}
    if torch.cuda.get_device_capability()[0] < 9:  # MoE default torch._grouped_mm is Hopper-only (gemma-4 on A100)
        kw["experts_implementation"] = "eager"  # batched_mm gathers per-token expert weights: 39 GiB OOM at prefill
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    return proc, model


def ask(proc, model, nodes: str, question: str, max_new_tokens: int, thinking: bool):
    """-> (raw_output, answer_text). Thinking traces (gemma <|channel>..., Qwen <think>...) are split off."""
    import torch
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"3D detections:\n{nodes}\n\nQuestion:\n{question}"}]
    try:
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    except Exception:  # template without a system role
        msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{msgs[1]['content']}"}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = proc(text=text, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[0, n_in:]
    raw = proc.decode(gen, skip_special_tokens=False).strip()
    if hasattr(proc, "parse_response"):
        try:
            content = (proc.parse_response(raw).get("content") or "").strip()
            if content:
                return raw, content
        except Exception:
            pass
    if "<channel|>" in raw:
        return raw, re.sub(r"<[^>]*>", " ", raw.rsplit("<channel|>", 1)[1]).strip()
    clean = proc.decode(gen, skip_special_tokens=True).strip()
    return raw, re.sub(r"<think>.*?</think>", "", clean, flags=re.S).strip() or clean


# ---- scoring
def score(rows):
    docs = [vsibench_process_results({"question_type": r["question_type"], "ground_truth": r["ground_truth"]},
                                     [r["prediction"]])["vsibench_overall"] for r in rows]
    return {tag: fn(docs) for tag, fn in AGGREGATORS} if docs else {}


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_parquet", default=f"{W}/dataset/vsib/VSI-Bench-extended/test_debiased_scannet.parquet")
    ap.add_argument("--det_root", default=f"{W}/dataset/vsib_det64f/scannet")
    ap.add_argument("--llm_model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=None, help="cap BEFORE sharding (smoke)")
    ap.add_argument("--thinking", dest="thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--merge", action="store_true", help="score the union of all shards in out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.split_parquet)
    df = df[df["dataset"] == "scannet"].sort_values("id").reset_index(drop=True)
    if a.max_questions:
        df = df.head(a.max_questions)

    if a.merge:
        rows = {r["id"]: r for p in sorted(out.glob("predictions_shard*.jsonl")) for r in read_jsonl(p)}
        missing = sorted(set(df["id"]) - set(rows))
        assert not missing, f"{len(missing)} questions missing, e.g. {missing[:10]}"
        rows = [rows[i] for i in df["id"]]
        res = {"model": rows[0]["model"], "num_answered": len(rows), "scores": score(rows)}
        (out / "results_merged.json").write_text(json.dumps(res, indent=2))
        (out / "predictions_merged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(res, indent=2)); return

    df = df.iloc[a.shard_index::a.num_shards]
    max_new = a.max_new_tokens or (THINK_MAX_NEW_TOKENS if a.thinking else NOTHINK_MAX_NEW_TOKENS)
    preds_path = out / f"predictions_shard{a.shard_index}.jsonl"
    done = {r["id"] for r in read_jsonl(preds_path)}     # resume after requeue
    print(f"shard {a.shard_index}/{a.num_shards}: {len(df)} questions ({len(done)} done); loading {a.llm_model}")
    proc, model = load_llm(a.llm_model)
    import torch
    nodes = {}
    with preds_path.open("a") as f:
        for n, (_, row) in enumerate(df.iterrows(), 1):
            if int(row["id"]) in done:
                continue
            sc = row["scene_name"]
            if sc not in nodes:
                nodes[sc] = det_to_nodes(json.load(open(Path(a.det_root) / sc / "det.json")))
            raw, ans = ask(proc, model, nodes[sc], build_question(row), max_new, a.thinking)
            f.write(json.dumps({"id": int(row["id"]), "scene_name": sc, "question_type": row["question_type"],
                                "prediction": extract_answer(ans, row["question_type"]), "raw_output": raw,
                                "ground_truth": str(row["ground_truth"]), "model": a.llm_model}) + "\n"); f.flush()
            if n % 10 == 0:
                print(f"[{a.shard_index}] {n}/{len(df)}", flush=True)
    rows = read_jsonl(preds_path)
    res = {"model": a.llm_model, "shard": a.shard_index, "num_shards": a.num_shards, "num_answered": len(rows),
           "thinking": a.thinking, "max_new_tokens": max_new, "scores": score(rows),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    (out / f"results_shard{a.shard_index}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
