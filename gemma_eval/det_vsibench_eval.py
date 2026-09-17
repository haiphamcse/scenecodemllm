"""VSI-Bench-Debiased (ScanNet) answered by gemma-4 from the det_es cache: det list, cached frames, or both.

Input per scene: <video_root>/<scene>/det_es/{det.json, meta.json, frames/frameNN.jpg}. Frames are the posed-64f
frames already decoded (384x256); frame 0 is the box origin. No mp4 decode. Prompts, answer extraction and scoring
are imported from gemini_eval/det_vsibench_eval.py so det_video runs compare like for like with the old det+video run.

  python det_vsibench_eval.py --mode det_video --out_dir results/det_vsibench/x --subset_ids ids.json
  python det_vsibench_eval.py --mode video     --out_dir results/det_vsibench/y --subset_ids ids.json   # frames only
  python det_vsibench_eval.py --mode det       --out_dir results/det_vsibench/z --subset_ids ids.json   # text only
  python det_vsibench_eval.py --mode det_video --out_dir results/det_vsibench/x --subset_ids ids.json --merge
"""
import argparse, functools, importlib.util, json, re, time
from pathlib import Path
import numpy as np, pandas as pd

_spec = importlib.util.spec_from_file_location(
    "gemini_det_eval", Path(__file__).resolve().parent.parent / "gemini_eval" / "det_vsibench_eval.py")
G = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(G)

W = "/home/ducpham/scratch/Working"
MODEL = "google/gemma-4-26B-A4B-it"
SPLIT_PARQUET = f"{W}/dataset/vsib/VSI-Bench-extended/test_debiased_scannet.parquet"
VIDEO_SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant. You are given frames of a video of an indoor scene, in temporal order; "
    "frame 1 is the first video frame. Distances are ordinary 3D Euclidean distances in meters. "
    "For numeric questions (distance, size, count, room area) always answer with a single best-estimate number, even "
    "if the frames are incomplete or imprecise; never refuse or say the information is insufficient. "
    "Use only the frames to answer the question.")


@functools.lru_cache(maxsize=1)  # questions are iterated scene by scene, so one slot is enough
def load_frames(frames_dir: str, num_frames: int):
    """det_es/frames/frameNN.jpg in order. num_frames < available: uniform subsample that always keeps frame 0."""
    from PIL import Image
    paths = sorted(Path(frames_dir).glob("frame*.jpg"))
    if num_frames < len(paths):
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, num_frames).round().astype(int)]
    return [Image.open(p).convert("RGB") for p in paths]


def load_llm(with_frames: bool, image_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL)
    if with_frames:  # AutoModelForCausalLM drops the vision tower
        proc.image_processor.max_soft_tokens = image_tokens  # gemma-4 image-size knob: 70/140/280/560/1120 tokens
        model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    return proc, model


def ask(proc, model, mode: str, nodes: str, question: str, max_new_tokens: int, thinking: bool, frames):
    """-> (raw_output, answer_text). det_video prompt == gemini_eval det+video prompt.
    On CUDA OOM retried with half the frames, tagged [frames_used=N]."""
    import torch
    tag = ""
    if mode == "det":
        msgs = [{"role": "system", "content": G.SYSTEM_PROMPT + G.TEXT_SUFFIX},
                {"role": "user", "content": f"3D detections:\n{nodes}\n\nQuestion:\n{question}"}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        inputs = proc(text=text, return_tensors="pt").to(model.device)
        n_in = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)[0, n_in:]
    else:
        if mode == "det_video":
            system, user = G.SYSTEM_PROMPT + G.VIDEO_SUFFIX, f"3D detections:\n{nodes}\n\nQuestion:\n{question}"
            origin = " (the coordinate origin)"
        else:
            system, user, origin = VIDEO_SYSTEM_PROMPT, f"Question:\n{question}", ""
        n = len(frames)
        while True:
            sub = [frames[i] for i in np.linspace(0, len(frames) - 1, n).round().astype(int)]
            note = f"The {n} images above are video frames in temporal order; image 1 is the first video frame{origin}.\n\n"
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "image", "image": im} for im in sub]
                     + [{"type": "text", "text": note + user}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, return_dict=True, add_generation_prompt=True,
                                              return_tensors="pt", enable_thinking=thinking)
            inputs = inputs.to(device=model.device, dtype=model.dtype)
            n_in = inputs["input_ids"].shape[-1]
            if not getattr(ask, "_logged", False):  # proof frames reach the model + one full prompt
                ask._logged = True
                n_img = int((inputs["input_ids"] == proc.tokenizer.image_token_id).sum())
                print(f"[video] {n} frames, input_len={n_in}, image_tokens={n_img} ({n_img / n:.0f}/frame)", flush=True)
                print(f"[prompt] system:\n{system}\n[prompt] user:\n[{n} frames]\n{note}{user}\n[prompt] end", flush=True)
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
    try:
        content = (proc.parse_response(raw).get("content") or "").strip()
        if content:
            return raw + tag, content
    except Exception:
        pass
    if "<channel|>" in raw:
        return raw + tag, re.sub(r"<[^>]*>", " ", raw.rsplit("<channel|>", 1)[1]).strip()
    return raw + tag, proc.decode(gen, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_root", default=f"{W}/dataset/vsib/vsibench/scannet", help="<root>/<scene>/det_es/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", choices=["det_video", "video", "det"], default="det_video")
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--image_tokens", type=int, default=140, help="gemma-4 max_soft_tokens per frame (image size)")
    ap.add_argument("--thinking", dest="thinking", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--max_questions", type=int, default=None, help="cap BEFORE sharding (smoke)")
    ap.add_argument("--subset_ids", default=None, help="json list of question ids; outputs go to <out_dir>/subset/")
    ap.add_argument("--subset_n", type=int, default=None, help="draw N questions per type (--seed) into <out_dir>/subset/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true", help="score the union of all shards in out_dir")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SPLIT_PARQUET)
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
        rows = {r["id"]: r for p in shards for r in G.read_jsonl(p)}
        missing = sorted(set(df["id"]) - set(rows))
        assert not missing, f"{len(missing)} questions missing, e.g. {missing[:10]}"
        rows = [rows[i] for i in df["id"]]
        res = {"model": MODEL, "mode": a.mode, "num_answered": len(rows), "scores": G.score(rows)}
        (out / "results_merged.json").write_text(json.dumps(res, indent=2))
        (out / "predictions_merged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(res, indent=2)); return

    df = df.iloc[a.shard_index::a.num_shards]
    max_new = a.max_new_tokens or (G.THINK_MAX_NEW_TOKENS if a.thinking else G.NOTHINK_MAX_NEW_TOKENS)
    preds_path = out / f"predictions_shard{a.shard_index}.jsonl"
    done = {r["id"] for r in G.read_jsonl(preds_path)}     # resume after requeue
    print(f"shard {a.shard_index}/{a.num_shards}: {len(df)} questions ({len(done)} done); mode={a.mode}; loading {MODEL}")
    proc, model = load_llm(a.mode != "det", a.image_tokens)
    import torch
    nodes, t0, n_new = {}, time.time(), 0
    with preds_path.open("a") as f:
        for n, (_, row) in enumerate(df.sort_values(["scene_name", "id"]).iterrows(), 1):
            if int(row["id"]) in done:
                continue
            sc = row["scene_name"]
            es = Path(a.video_root) / sc / "det_es"
            if sc not in nodes:
                nodes[sc] = G.det_to_nodes(json.load(open(es / "det.json"))) if a.mode != "video" else ""
            frames = load_frames(str(es / "frames"), a.num_frames) if a.mode != "det" else None
            raw, ans = ask(proc, model, a.mode, nodes[sc], G.build_question(row), max_new, a.thinking, frames)
            f.write(json.dumps({"id": int(row["id"]), "scene_name": sc, "question_type": row["question_type"],
                                "prediction": G.extract_answer(ans, row["question_type"]), "raw_output": raw,
                                "ground_truth": str(row["ground_truth"]), "model": MODEL}) + "\n"); f.flush()
            n_new += 1
            if n_new == 1 or n % 10 == 0:
                print(f"[{a.shard_index}] {n}/{len(df)} {(time.time() - t0) / n_new:.0f}s/q "
                      f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    rows = G.read_jsonl(preds_path)
    res = {"model": MODEL, "mode": a.mode, "shard": a.shard_index, "num_shards": a.num_shards,
           "num_answered": len(rows), "thinking": a.thinking, "max_new_tokens": max_new,
           "num_frames": a.num_frames, "image_tokens": a.image_tokens, "scores": G.score(rows),
           "sec_per_q": round((time.time() - t0) / max(n_new, 1), 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    (out / f"results_shard{a.shard_index}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
