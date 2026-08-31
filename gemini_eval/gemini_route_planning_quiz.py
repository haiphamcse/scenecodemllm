"""Gemini (manual copy-paste) quiz on VSI-Bench debiased ScanNet++ route_planning.

Feeds the SAME input as the local scene-graph LLM evals (v2 representation,
NODES ONLY -- relations block stripped) but for a closed-source model you drive
by hand: this script writes the 22 prompts to one file, you paste each block
into Gemini, write the option letters back into an answers file, and it scores
them with the upstream VSI-Bench metric (common/vsibench_metrics.py).

Baselines to beat (results_llm/*/results.json, same 22 questions):
  qwen35_35b_v2_nothink_nodes  0.545  |  gemma_26b_v2_nothink_full   0.455
  gemma_26b_v2_nothink_nodes   0.409  |  qwen2.5-7b raw graph        0.318

Usage:
  # 1) write prompts + a blank answers template
  python spatial_reasoning/finetuning/gemini_eval/gemini_route_planning_quiz.py \
      --prompts gemini_prompts.txt --template gemini_answers.txt
  # 2) paste each ===== block into Gemini, put its letter on the matching
  #    ANSWER: line of gemini_answers.txt, then score:
  python spatial_reasoning/finetuning/gemini_eval/gemini_route_planning_quiz.py \
      --answers gemini_answers.txt

  --selfcheck scores the ground-truth letters (must print 1.0).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "common"))
from vsibench_metrics import (  # noqa: E402
    vsibench_aggregate_route_planning_accuracy,
    vsibench_process_results,
)

DEFAULT_JSONL = Path("/scratch/ducpham/Working/dataset/vsib/vsibench_debiased_scannetpp.jsonl")
DEFAULT_SG_DIR = Path("/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ScanEdit/scanedit_eval_out")

# Verbatim from scanedit_vsibench_eval_gemma_v2.SYSTEM_PROMPT_NODES.
SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant. You are given a 3D scene graph: a "
    "list of objects, one per line as '<id> <name> size=(dx,dy,dz) "
    "base=(x,y,z_min)', where size is the object's extent in meters and base is "
    "its minimum-corner position (x,y,z_min) in meters. Use only this scene "
    "graph to answer the question.")

# lmms-eval's video pre-prompt, meaningless for a text-only model (stripped by
# the local evals too).
_VIDEO_PREFIX = "These are frames of a video."

# raw scene_graph.txt line:
#   id 0: book, size(dx,dy,dz)=(0.27,0.32,0.02), base(x,y,z_min)=(3.83,3.03,0.8), relations: ...
_NODE_RE = re.compile(
    r"^id (\d+): (.+?), size\(dx,dy,dz\)=\(([^)]*)\), base\(x,y,z_min\)=\(([^)]*)\)")


def to_v2_nodes(raw_text: str) -> str:
    """raw scene_graph.txt -> v2 node block ('<id> <name> size=(..) base=(..)').

    Reimplements repr_v2.convert's node half (that module's path in
    scanedit_vsibench_eval_gemma_v2.py no longer exists); relations are dropped,
    which is what --graph_part nodes fed gemma.
    """
    lines = ["nodes:"]
    for line in raw_text.splitlines():
        m = _NODE_RE.match(line.strip())
        if m:
            oid, name, size, base = m.groups()
            lines.append(f"{oid} {name} size=({size}) base=({base})")
    if len(lines) == 1:
        raise ValueError("no node lines parsed from scene graph")
    return "\n".join(lines)


def load_questions(jsonl_path: Path, sg_dir: Path) -> list[dict]:
    rows = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["question_type"] != "route_planning":
            continue
        scene = Path(r["video"]).stem
        question = (r["conversations"][0]["value"]
                    .replace("<video>", "").replace(_VIDEO_PREFIX, "").strip())
        rows.append({
            "id": r["id"],
            "scene": scene,
            "question": question,
            "gt": r["conversations"][1]["value"],
            "graph": to_v2_nodes((sg_dir / scene / "scene_graph.txt").read_text()),
        })
    rows.sort(key=lambda r: r["id"])
    return rows


def write_prompts(rows: list[dict], path: Path) -> None:
    blocks = []
    for i, r in enumerate(rows, 1):
        blocks.append(
            f"===== [{i}/{len(rows)}] id={r['id']} scene={r['scene']} =====\n"
            f"{SYSTEM_PROMPT}\n\n"
            f"3D scene graph:\n{r['graph']}\n\n"
            f"Question:\n{r['question']}")
    path.write_text("\n\n".join(blocks) + "\n")
    print(f"Wrote {len(rows)} prompts to {path}")


def write_template(rows: list[dict], path: Path) -> None:
    lines = ["# Gemini route_planning answers -- one option letter after each 'ANSWER:'.",
             "# Order matches the prompts file (sorted by question id)."]
    for i, r in enumerate(rows, 1):
        lines.append(f"# [{i}/{len(rows)}] id={r['id']} scene={r['scene']}")
        lines.append("ANSWER: ")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote answers template to {path}")


_GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "{model}:generateContent")


_RETRY_CODES = {429, 500, 502, 503}


def ask_gemini(prompt: str, model: str, api_key: str, tries: int = 6) -> str:
    """One generateContent call; returns the answer text (thought parts dropped).

    503 UNAVAILABLE / 429 mean the model is overloaded or rate-limited, so retry
    with exponential backoff; other HTTP errors are re-raised with the body.
    """
    req = urllib.request.Request(
        _GEMINI_URL.format(model=model),
        data=json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode(),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_CODES or attempt == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}") from e
            delay = 2 ** attempt * 5
            print(f"    HTTP {e.code}, retry in {delay}s ({attempt + 1}/{tries - 1})")
            time.sleep(delay)
    parts = body["candidates"][0]["content"].get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def extract_letter(text: str) -> str:
    """Option letter out of a possibly-reasoning reply (mirrors the gemma eval's
    _extract_mca; copied rather than imported to keep this script torch-free)."""
    for pat in (r"(?:final answer|the answer|answer)\s*(?:is|:|=)?\s*\(?\**([A-H])\**\)?\b",
                r"\\boxed\{\s*([A-H])\s*\}",
                r"\*\*\s*([A-H])\s*\*\*"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1].upper()
    m = re.findall(r"\b([A-H])\b", text)
    return m[-1] if m else text.split(" ")[0]


def run_gemini(rows: list[dict], model: str, api_key: str, raw_path: Path) -> list[str]:
    """Answer every row, appending to raw_path; already-answered ids are reused so
    a crashed run resumes instead of re-asking (delete raw_path to force a redo)."""
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["id"]] = d["prediction"]
        print(f"Resuming: {len(done)} answers already in {raw_path}")

    with raw_path.open("a") as f:
        for i, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            prompt = (f"{SYSTEM_PROMPT}\n\n3D scene graph:\n{r['graph']}\n\n"
                      f"Question:\n{r['question']}")
            text = ask_gemini(prompt, model, api_key)
            letter = extract_letter(text)
            done[r["id"]] = letter
            f.write(json.dumps({"id": r["id"], "scene": r["scene"], "gt": r["gt"],
                                "prediction": letter, "raw_output": text}) + "\n")
            f.flush()
            print(f"  [{i}/{len(rows)}] id={r['id']} -> {letter}")
    print(f"Raw replies: {raw_path}")
    return [done[r["id"]] for r in rows]


def parse_answers(path: Path) -> list[str]:
    return [l.split(":", 1)[1].strip() for l in path.read_text().splitlines()
            if l.strip().lower().startswith("answer:")]


def score(rows: list[dict], answers: list[str]) -> None:
    if len(answers) != len(rows):
        raise SystemExit(f"{len(answers)} ANSWER: lines but {len(rows)} questions.")
    docs = []
    for r, pred in zip(rows, answers):
        doc = vsibench_process_results(
            {"question_type": "route_planning", "ground_truth": r["gt"]}, [pred],
        )["vsibench_overall"]
        docs.append(doc)
        print(f"  id={r['id']:<6} scene={r['scene']}  pred={pred!r:<5} gt={r['gt']!r:<5} "
              f"correct={bool(doc['accuracy'])}")
    acc = round(vsibench_aggregate_route_planning_accuracy(docs), 4)
    print(f"\nroute_planning accuracy: {acc}  ({sum(bool(d['accuracy']) for d in docs)}/{len(docs)})")
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    ap.add_argument("--sg_dir", type=Path, default=DEFAULT_SG_DIR)
    ap.add_argument("--prompts", type=Path, help="write the Gemini prompts here")
    ap.add_argument("--template", type=Path, help="write a blank answers file here")
    ap.add_argument("--answers", type=Path, help="score the letters in this file")
    ap.add_argument("--gemini", metavar="MODEL",
                    help="call the Gemini API with this model (e.g. gemini-2.5-pro) "
                         "instead of pasting by hand; needs GEMINI_API_KEY")
    ap.add_argument("--raw_out", type=Path, default=_HERE / "gemini_rp_raw.jsonl",
                    help="where --gemini writes the raw replies")
    ap.add_argument("--scenes", action="store_true", help="print the scene list and exit")
    ap.add_argument("--selfcheck", action="store_true",
                    help="score the ground truth (must be 1.0)")
    args = ap.parse_args()

    rows = load_questions(args.jsonl, args.sg_dir)
    print(f"{len(rows)} route_planning questions over {len({r['scene'] for r in rows})} scenes.")

    if args.scenes:
        for r in rows:
            print(f"{r['scene']}  id={r['id']}")
        return
    if args.prompts:
        write_prompts(rows, args.prompts)
    if args.template:
        write_template(rows, args.template)
    if args.selfcheck:
        assert score(rows, [r["gt"] for r in rows]) == 1.0, "scorer broken"
        print("selfcheck OK")
    if args.gemini:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("set GEMINI_API_KEY (https://aistudio.google.com/apikey)")
        score(rows, run_gemini(rows, args.gemini, key, args.raw_out))
    if args.answers:
        score(rows, parse_answers(args.answers))
    if not any((args.prompts, args.template, args.answers, args.selfcheck, args.gemini)):
        ap.error("nothing to do: pass --prompts/--template/--answers/--scenes/--selfcheck")


if __name__ == "__main__":
    main()
