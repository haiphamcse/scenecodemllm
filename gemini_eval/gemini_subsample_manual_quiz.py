"""Manual (copy-paste) version of gemini_vsibench_subsample_eval.

Same subsample, same v2 NODES-ONLY prompts, same scorer -- but you drive the
model yourself in a web UI instead of calling the API:

  # 1) prompts + a blank answers sheet
  python gemini_subsample_manual_quiz.py \
      --subsample gemini_subsample.jsonl \
      --prompts gemini_manual_prompts.txt --template gemini_manual_answers.txt
  # 2) paste each ===== block into the model, write its answer on the matching
  #    ANSWER: line (option letter for multiple choice, a number for the rest)
  python spatial_reasoning/finetuning/gemini_eval/gemini_subsample_manual_quiz.py \
      --subsample gemini_subsample.jsonl --answers gemini_manual_answers.txt

Make the subsample first with:
  python spatial_reasoning/finetuning/gemini_eval/gemini_vsibench_subsample_eval.py --sample

--selfcheck scores the ground truth (must print 1.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from gemini_vsibench_subsample_eval import (  # noqa: E402
    AGGREGATORS, DEFAULT_SG_DIR, MCA_QUESTION_TYPES, SYSTEM_PROMPT,
    build_prompt, vsibench_process_results,
)


def load_subsample(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def answer_hint(row: dict) -> str:
    return ("option letter" if row["question_type"] in MCA_QUESTION_TYPES
            else "number")


def write_prompts(rows: list[dict], path: Path, sg_dir: Path) -> None:
    blocks = [
        f"===== [{i}/{len(rows)}] id={r['id']} type={r['question_type']} "
        f"scene={r['scene']} =====\n{SYSTEM_PROMPT}\n\n{build_prompt(r, sg_dir)}"
        for i, r in enumerate(rows, 1)
    ]
    path.write_text("\n\n".join(blocks) + "\n")
    print(f"Wrote {len(rows)} prompts to {path}")


def write_template(rows: list[dict], path: Path) -> None:
    lines = ["# VSI-Bench subsample answers -- one answer after each 'ANSWER:'.",
             "# Order matches the prompts file."]
    for i, r in enumerate(rows, 1):
        lines.append(f"# [{i}/{len(rows)}] id={r['id']} type={r['question_type']} "
                     f"scene={r['scene']}  ({answer_hint(r)})")
        lines.append("ANSWER: ")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote answers template to {path}")


def parse_answers(path: Path) -> list[str]:
    return [l.split(":", 1)[1].strip() for l in path.read_text().splitlines()
            if l.strip().lower().startswith("answer:")]


def score(rows: list[dict], answers: list[str]) -> dict:
    if len(answers) != len(rows):
        raise SystemExit(f"{len(answers)} ANSWER: lines but {len(rows)} questions.")
    docs = []
    for r, pred in zip(rows, answers):
        doc = vsibench_process_results(
            {"question_type": r["question_type"], "ground_truth": r["gt"]}, [pred],
        )["vsibench_overall"]
        docs.append(doc)
        # MCA docs carry 'accuracy', numeric-answer docs an 'MRA:...' key instead.
        val = doc.get("accuracy", doc.get("MRA:.5:.95:.05"))
        print(f"  id={r['id']:<6} {r['question_type']:<28s} pred={pred!r:<8} "
              f"gt={r['gt']!r:<8} score={round(val, 4)}")
    scores = {tag: round(fn(docs), 4) for tag, fn in AGGREGATORS}
    print()
    for tag, value in scores.items():
        print(f"  {tag:<45s}{value}")
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsample", type=Path, default=_HERE / "gemini_subsample.jsonl")
    ap.add_argument("--sg_dir", type=Path, default=DEFAULT_SG_DIR)
    ap.add_argument("--prompts", type=Path, help="write the prompts here")
    ap.add_argument("--template", type=Path, help="write a blank answers sheet here")
    ap.add_argument("--answers", type=Path, help="score the answers in this file")
    ap.add_argument("--selfcheck", action="store_true",
                    help="score the ground truth (must be 1.0)")
    args = ap.parse_args()

    rows = load_subsample(args.subsample)
    print(f"{len(rows)} questions over {len({r['scene'] for r in rows})} scenes "
          f"from {args.subsample}")

    if args.prompts:
        write_prompts(rows, args.prompts, args.sg_dir)
    if args.template:
        write_template(rows, args.template)
    if args.selfcheck:
        assert score(rows, [r["gt"] for r in rows])["vsibench/overall"] == 1.0
        print("selfcheck OK")
    if args.answers:
        score(rows, parse_answers(args.answers))
    if not any((args.prompts, args.template, args.answers, args.selfcheck)):
        ap.error("nothing to do: pass --prompts/--template/--answers/--selfcheck")


if __name__ == "__main__":
    main()
