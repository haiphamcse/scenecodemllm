"""VSI-Bench-Debiased (ScanNet) answered by gemma-4 from a 3D detection list, optionally plus the video frames.

Input per scene: <det_root>/<scene>/det.json (posed-64f format: boxes = 9-DoF ZXY in the camera of frame 0, VG-LLM
convention). Each rotated box becomes one v2 node line from its AABB: '<id> <label> size=(dx,dy,dz) base=(x,y,z)'.
--video adds the posed frames listed in meta.json (decoded from the VSI-Bench mp4 at meta 'mp4_idx') as images.
LLM load / generation / answer extraction / scoring follow ScanEdit/scanedit_vsibench_eval_gemma{,_v2}.py.

  python det_vsibench_eval.py --llm_model google/gemma-4-26B-A4B-it --out_dir results_llm/det_gemma \
      --num_shards 4 --shard_index 0 --no-thinking          # -> predictions_shard0.jsonl, results_shard0.json
  python det_vsibench_eval.py --out_dir results_llm/det_gemma --merge   # union of all shards, refuses if incomplete
  python det_vsibench_eval.py ... --video --image_tokens 140 --subset_n 20   # det+video on a stratified subset
  python det_vsibench_eval.py --out_dir <old text run> --merge --subset_ids <dir>/subset/subset_ids.json  # same subset
"""
import argparse, functools, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from vsibench_metrics import AGGREGATORS, MCA_QUESTION_TYPES, NA_QUESTION_TYPES, vsibench_process_results  # noqa: E402

W = "/home/ducpham/scratch/Working"
SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant. You are given a list of 3D object detections from a video of an indoor "
    "scene, one per line as '<id> <label> size=(dx,dy,dz) base=(x,y,z)'. size is the object's axis-aligned extent in "
    "meters along (x,y,z); base is its minimum corner (x_min,y_min,z_min) in meters. Coordinates are measured from "
    "the camera position at the FIRST video frame: x grows to the right of that view, y grows downward (toward the "
    "floor), z grows forward (away from the camera). So dy is an object's height, dx and dz its footprint, and the "
    "floor is near the largest y values. Distances are ordinary 3D Euclidean distances in meters. "
    "For numeric questions (distance, size, count, room area) always answer with a single best-estimate number, even "
    "if the list is incomplete or imprecise; never refuse or say the information is insufficient. Estimate room area "
    "from the spread of the objects.")
TEXT_SUFFIX = " Use only this list to answer the question."
VIDEO_SUFFIX = (" You are also given frames of the video, in temporal order; frame 1 is the first video frame, i.e. "
                "the camera whose position defines the coordinates. Use both the list and the frames.")
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
    text = re.sub(r"\b[23]D\b", "", text)  # '3D detections' is not a number
    for pat in (r"\\boxed\{\s*(" + _NUM + r")", r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\D{0,15}?(" + _NUM + r")"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1]
    nums = re.findall(_NUM, text)
    return nums[-1] if nums else text.split(" ")[0]


# ---- video frames
@functools.lru_cache(maxsize=1)  # questions are iterated scene by scene, so one slot is enough
def load_frames(meta_path: str, num_frames: int):
    """The posed frames listed in meta.json, decoded from the mp4 at 'mp4_idx'. Frame 0 is the box origin.
    num_frames < len(list): uniform subsample that always keeps frame 0."""
    import cv2
    from PIL import Image
    m = json.load(open(meta_path))
    want = [f["mp4_idx"] for f in m["frames"]]
    if num_frames < len(want):
        want = [want[i] for i in np.linspace(0, len(want) - 1, num_frames).round().astype(int)]
    cap, got = cv2.VideoCapture(m["mp4_local"]), {}
    for i in range(max(want) + 1):
        ok, im = cap.read()
        assert ok, f"{m['mp4_local']}: cannot read frame {i}"
        if i in want:
            got[i] = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cap.release()
    return [got[i] for i in want]


# ---- LLM
def load_llm(name, video: bool, image_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    try:
        proc = AutoProcessor.from_pretrained(name)
    except Exception:
        proc = AutoTokenizer.from_pretrained(name)
    kw = {}
    if torch.cuda.get_device_capability()[0] < 9:  # MoE default torch._grouped_mm is Hopper-only (gemma-4 on A100)
        kw["experts_implementation"] = "eager"  # batched_mm gathers per-token expert weights: 39 GiB OOM at prefill
    if video:  # AutoModelForCausalLM drops the vision tower
        proc.image_processor.max_soft_tokens = image_tokens  # gemma-4 image-size knob: 70/140/280/560/1120 tokens
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda", **kw).eval()
    return proc, model


def ask(proc, model, nodes: str, question: str, max_new_tokens: int, thinking: bool, frames=None):
    """-> (raw_output, answer_text). Thinking traces (gemma <|channel>..., Qwen <think>...) are split off.
    frames: list of PIL images (det+video mode); on CUDA OOM retried with half the frames, tagged [frames_used=N]."""
    import torch
    system = SYSTEM_PROMPT + (VIDEO_SUFFIX if frames else TEXT_SUFFIX)
    user = f"3D detections:\n{nodes}\n\nQuestion:\n{question}"
    tag = ""
    if frames is None:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        except Exception:  # template without a system role
            msgs = [{"role": "user", "content": f"{system}\n\n{user}"}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = proc(text=text, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[0, n_in:]
    else:
        n = len(frames)
        while True:
            sub = [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]
            note = (f"The {n} images above are video frames in temporal order; image 1 is the first video frame "
                    "(the coordinate origin).\n\n")
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "image", "image": im} for im in sub]
                     + [{"type": "text", "text": note + user}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, add_generation_prompt=True,
                                              return_tensors="pt", enable_thinking=thinking)
            inputs = inputs.to(device=model.device, dtype=model.dtype)
            n_in = inputs["input_ids"].shape[-1]
            if not getattr(ask, "_logged", False):  # proof frames reach the model
                ask._logged = True
                n_img = int((inputs["input_ids"] == proc.tokenizer.image_token_id).sum())
                print(f"[video] {n} frames, input_len={n_in}, image_tokens={n_img} ({n_img / n:.0f}/frame)", flush=True)
            try:
                with torch.no_grad():
                    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[0, n_in:]
                break
            except torch.OutOfMemoryError:
                if n <= 4:
                    raise
                del inputs; torch.cuda.empty_cache()
                n //= 2
                print(f"[oom] retrying with {n} frames", flush=True)
        if n != len(frames):
            tag = f"\n[frames_used={n}]"
    raw = proc.decode(gen, skip_special_tokens=False).strip()
    if hasattr(proc, "parse_response"):
        try:
            content = (proc.parse_response(raw).get("content") or "").strip()
            if content:
                return raw + tag, content
        except Exception:
            pass
    if "<channel|>" in raw:
        return raw + tag, re.sub(r"<[^>]*>", " ", raw.rsplit("<channel|>", 1)[1]).strip()
    clean = proc.decode(gen, skip_special_tokens=True).strip()
    return raw + tag, re.sub(r"<think>.*?</think>", "", clean, flags=re.S).strip() or clean


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
    ap.add_argument("--video", action=argparse.BooleanOptionalAction, default=False, help="add meta.json frames")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--image_tokens", type=int, default=280, help="gemma-4 max_soft_tokens per frame (image size)")
    ap.add_argument("--subset_ids", default=None, help="json list of question ids; outputs go to <out_dir>/subset/")
    ap.add_argument("--subset_n", type=int, default=None, help="draw N questions per type (--seed) into <out_dir>/subset/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="score the union of all shards in out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.split_parquet)
    df = df[df["dataset"] == "scannet"].sort_values("id").reset_index(drop=True)
    if a.max_questions:
        df = df.head(a.max_questions)
    if a.subset_ids or a.subset_n:
        out = out / "subset"; out.mkdir(exist_ok=True)
        if a.subset_ids:
            ids = json.load(open(a.subset_ids))
        else:
            ids = sorted(int(i) for i in df.groupby("question_type", group_keys=False)
                         .apply(lambda g: g.sample(min(a.subset_n, len(g)), random_state=a.seed))["id"])
            (out / "subset_ids.json").write_text(json.dumps(ids))
        df = df[df["id"].isin(ids)].reset_index(drop=True)

    if a.merge:
        shards = sorted(out.glob("predictions_shard*.jsonl"))
        if not shards and out.name == "subset":  # score a full run on the subset
            shards = sorted(out.parent.glob("predictions_shard*.jsonl"))
        rows = {r["id"]: r for p in shards for r in read_jsonl(p)}
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
    proc, model = load_llm(a.llm_model, a.video, a.image_tokens)
    import torch
    nodes, t0, n_new = {}, time.time(), 0
    with preds_path.open("a") as f:
        for n, (_, row) in enumerate(df.sort_values(["scene_name", "id"]).iterrows(), 1):
            if int(row["id"]) in done:
                continue
            sc = row["scene_name"]
            if sc not in nodes:
                nodes[sc] = det_to_nodes(json.load(open(Path(a.det_root) / sc / "det.json")))
            frames = load_frames(str(Path(a.det_root) / sc / "meta.json"), a.num_frames) if a.video else None
            raw, ans = ask(proc, model, nodes[sc], build_question(row), max_new, a.thinking, frames)
            f.write(json.dumps({"id": int(row["id"]), "scene_name": sc, "question_type": row["question_type"],
                                "prediction": extract_answer(ans, row["question_type"]), "raw_output": raw,
                                "ground_truth": str(row["ground_truth"]), "model": a.llm_model}) + "\n"); f.flush()
            n_new += 1
            if n_new == 1 or n % 10 == 0:
                print(f"[{a.shard_index}] {n}/{len(df)} {(time.time() - t0) / n_new:.0f}s/q "
                      f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    rows = read_jsonl(preds_path)
    res = {"model": a.llm_model, "shard": a.shard_index, "num_shards": a.num_shards, "num_answered": len(rows),
           "thinking": a.thinking, "max_new_tokens": max_new, "video": a.video, "num_frames": a.num_frames,
           "image_tokens": a.image_tokens, "scores": score(rows),
           "sec_per_q": round((time.time() - t0) / max(n_new, 1), 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    (out / f"results_shard{a.shard_index}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
