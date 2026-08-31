"""Gemini eval on a balanced subsample of VSI-Bench debiased ScanNet++.

Two steps, one file:

  1) --sample: draw ~10 questions per question_type (all 10 types; types with
     fewer rows contribute everything they have), spread over as many distinct
     scenes as the split allows, and save them as
     <out>.jsonl (what step 2 reads) + <out>.txt (readable, for eyeballing).
  2) --eval:  feed each question's v2 NODES-ONLY scene graph + question to
     Gemini through the google-genai Interactions API, then score with the
     upstream VSI-Bench metric (accuracy for MCA types, MRA for numeric ones).

  conda activate base                      # google-genai 2.17.0 lives here
  export GEMINI_API_KEY=...                # https://aistudio.google.com/apikey
  python spatial_reasoning/finetuning/gemini_eval/gemini_vsibench_subsample_eval.py \
      --sample --out gemini_subsample
  python spatial_reasoning/finetuning/gemini_eval/gemini_vsibench_subsample_eval.py \
      --eval gemini_subsample.jsonl --model models/gemini-3-flash-preview

--eval appends to <raw_out> and skips ids already there, so an interrupted run
resumes; delete that file to force a redo. Rerun scoring alone with --score.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))
sys.path.insert(0, str(_HERE))
from vsibench_metrics import (  # noqa: E402
    AGGREGATORS, MCA_QUESTION_TYPES, NA_QUESTION_TYPES, vsibench_process_results,
)
from gemini_route_planning_quiz import to_v2_nodes  # noqa: E402

DEFAULT_JSONL = Path("/scratch/ducpham/Working/dataset/vsib/vsibench_debiased_scannetpp.jsonl")
DEFAULT_SG_DIR = Path("/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ScanEdit/scanedit_eval_out")

# Verbatim from scanedit_vsibench_eval_gemma_v2.SYSTEM_PROMPT_NODES.
SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant. You are given a 3D scene graph: a "
    "list of objects, one per line as '<id> <name> size=(dx,dy,dz) "
    "base=(x,y,z_min)', where size is the object's extent in meters and base is "
    "its minimum-corner position (x,y,z_min) in meters. Use only this scene "
    "graph to answer the question.")

_VIDEO_PREFIX = "These are frames of a video."
ALL_TYPES = MCA_QUESTION_TYPES + NA_QUESTION_TYPES


# --------------------------------------------------------------------------- #
# 1. subsample
# --------------------------------------------------------------------------- #
def load_split(jsonl_path: Path) -> list[dict]:
    rows = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append({
            "id": r["id"],
            "scene": Path(r["video"]).stem,
            "question_type": r["question_type"],
            "question": (r["conversations"][0]["value"]
                         .replace("<video>", "").replace(_VIDEO_PREFIX, "").strip()),
            "gt": r["conversations"][1]["value"],
        })
    return rows


def subsample(rows: list[dict], n_per_type: int, seed: int) -> list[dict]:
    """n_per_type per question_type (or all of them if the type has fewer),
    picking the least-used scene first so the sample spans as many distinct
    scenes as possible, both within a type and across the whole sample."""
    rng = random.Random(seed)
    used = Counter()
    picked = []
    for qt in ALL_TYPES:
        pool = [r for r in rows if r["question_type"] == qt]
        rng.shuffle(pool)
        for _ in range(min(n_per_type, len(pool))):
            r = min(pool, key=lambda x: used[x["scene"]])  # rng.shuffle breaks ties
            pool.remove(r)
            used[r["scene"]] += 1
            picked.append(r)
    picked.sort(key=lambda r: (ALL_TYPES.index(r["question_type"]), r["id"]))
    return picked


def write_sample(rows: list[dict], out: Path, n_per_type: int, seed: int) -> None:
    jsonl_path, txt_path = out.with_suffix(".jsonl"), out.with_suffix(".txt")
    jsonl_path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    counts = Counter(r["question_type"] for r in rows)
    lines = [f"VSI-Bench debiased ScanNet++ subsample -- {len(rows)} questions "
             f"over {len({r['scene'] for r in rows})} scenes "
             f"(n_per_type={n_per_type}, seed={seed})", ""]
    lines += [f"  {qt:<32s}{counts[qt]}" for qt in ALL_TYPES]
    for i, r in enumerate(rows, 1):
        lines += ["", "=" * 78,
                  f"[{i}/{len(rows)}] id={r['id']}  type={r['question_type']}  scene={r['scene']}",
                  "-" * 78, r["question"], "-" * 78,
                  f"GROUND TRUTH: {r['gt']}"]
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} questions to {jsonl_path} and {txt_path}")
    for qt in ALL_TYPES:
        print(f"  {qt:<32s}{counts[qt]}")


# --------------------------------------------------------------------------- #
# 2. Gemini (google-genai Interactions API)
# --------------------------------------------------------------------------- #
_RETRY_CODES = {429, 500, 502, 503}


def make_client():
    import os
    from google import genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (https://aistudio.google.com/apikey)")
    return genai.Client(api_key=key)


def ask_gemini(client, model: str, prompt: str, thinking_level: str,
               tries: int = 6) -> str:
    """One Interactions call. No tools: the scene graph must be the only evidence.

    503/429 mean overloaded or rate-limited -> exponential backoff; other API
    errors are fatal.
    """
    from google.genai import errors
    for attempt in range(tries):
        try:
            inter = client.interactions.create(
                model=model,
                input=prompt,
                system_instruction=SYSTEM_PROMPT,
                generation_config={"thinking_level": thinking_level},
            )
            return inter.output_text or ""
        except errors.APIError as e:
            if getattr(e, "code", None) not in _RETRY_CODES or attempt == tries - 1:
                raise
            delay = 2 ** attempt * 5
            print(f"    HTTP {e.code}, retry in {delay}s ({attempt + 1}/{tries - 1})")
            time.sleep(delay)


# --------------------------------------------------------------------------- #
# answer extraction (mirrors scanedit_vsibench_eval_gemma._extract_mca/_extract_na;
# copied rather than imported so this script stays free of the torch import chain)
# --------------------------------------------------------------------------- #
_NUM = r"[-+]?\d*\.?\d+"


def _extract_mca(text: str) -> str:
    for pat in (r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\s*\(?\**([A-H])\**\)?\b",
                r"\\boxed\{\s*([A-H])\s*\}",
                r"\*\*\s*([A-H])\s*\*\*"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].upper()
    m = re.findall(r"\b([A-H])\b", text)
    return m[-1] if m else text.split(" ")[0]


def _extract_na(text: str) -> str:
    for pat in (r"\\boxed\{\s*(" + _NUM + r")",
                r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\D{0,15}?(" + _NUM + r")"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1]
    nums = re.findall(_NUM, text)
    return nums[-1] if nums else text.split(" ")[0]


def extract_answer(text: str, question_type: str) -> str:
    if question_type in MCA_QUESTION_TYPES:
        return _extract_mca(text)
    if question_type in NA_QUESTION_TYPES:
        return _extract_na(text)
    return text.split(" ")[0]


def build_prompt(row: dict, sg_dir: Path) -> str:
    graph = to_v2_nodes((sg_dir / row["scene"] / "scene_graph.txt").read_text())
    return f"3D scene graph:\n{graph}\n\nQuestion:\n{row['question']}"


def run_gemini(rows: list[dict], model: str, sg_dir: Path, raw_path: Path,
               thinking_level: str) -> None:
    done = set()
    if raw_path.exists():
        done = {json.loads(l)["id"] for l in raw_path.read_text().splitlines() if l.strip()}
        print(f"Resuming: {len(done)} answers already in {raw_path}")
    client = make_client()
    with raw_path.open("a") as f:
        for i, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            text = ask_gemini(client, model, build_prompt(r, sg_dir), thinking_level)
            pred = extract_answer(text, r["question_type"])
            f.write(json.dumps({**r, "prediction": pred, "raw_output": text}) + "\n")
            f.flush()
            print(f"  [{i}/{len(rows)}] {r['question_type']:<28s} pred={pred!r} gt={r['gt']!r}")
    print(f"Raw replies: {raw_path}")


# --------------------------------------------------------------------------- #
# 3. score
# --------------------------------------------------------------------------- #
def score(raw_path: Path) -> dict:
    preds = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    docs = [vsibench_process_results(
        {"question_type": p["question_type"], "ground_truth": p["gt"]}, [p["prediction"]],
    )["vsibench_overall"] for p in preds]
    scores = {tag: round(fn(docs), 4) for tag, fn in AGGREGATORS}
    print(f"\nScored {len(docs)} answers from {raw_path}")
    for tag, value in scores.items():
        print(f"  {tag:<45s}{value}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--sg_dir", type=Path, default=DEFAULT_SG_DIR)
    ap.add_argument("--sample", action="store_true", help="draw a subsample and save it")
    ap.add_argument("--out", type=Path, default=_HERE / "gemini_subsample",
                    help="--sample writes <out>.jsonl and <out>.txt")
    ap.add_argument("--n_per_type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", type=Path, metavar="SUBSAMPLE_JSONL",
                    help="ask Gemini about the questions in this file, then score")
    ap.add_argument("--model", default="models/gemini-3-flash-preview")
    ap.add_argument("--thinking_level", default="high")
    ap.add_argument("--raw_out", type=Path, default=None,
                    help="raw replies jsonl (default: <eval stem>_raw.jsonl)")
    ap.add_argument("--score", type=Path, metavar="RAW_JSONL",
                    help="re-score an existing raw replies file and exit")
    args = ap.parse_args()

    if args.score:
        score(args.score)
        return
    if args.sample:
        write_sample(subsample(load_split(args.jsonl), args.n_per_type, args.seed),
                     args.out, args.n_per_type, args.seed)
    if args.eval:
        rows = [json.loads(l) for l in args.eval.read_text().splitlines() if l.strip()]
        raw_path = args.raw_out or args.eval.with_name(args.eval.stem + "_raw.jsonl")
        run_gemini(rows, args.model, args.sg_dir, raw_path, args.thinking_level)
        score(raw_path)
    if not (args.sample or args.eval):
        ap.error("nothing to do: pass --sample, --eval or --score")


if __name__ == "__main__":
    main()
