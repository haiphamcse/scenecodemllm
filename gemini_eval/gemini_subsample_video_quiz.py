"""Manual (copy-paste) quiz on the VSI-Bench subsample with VIDEO + scene graph.

Same subsample and same scorer as gemini_subsample_manual_quiz, but each prompt
also names the scene's mp4, which you upload in the web UI yourself. Prompts are
grouped by scene (one block per video, all of that scene's questions in it), so
you upload 49 videos instead of 100:

  # 1) prompts + a blank answers sheet
  python gemini_subsample_video_quiz.py \
      --prompts gemini_video_prompts.txt --template gemini_video_answers.txt
  # 2) for each ===== block: upload the video, paste the text, write each Qn
  #    answer on the matching ANSWER: line (letter for MCA, number for the rest)
  python gemini_subsample_video_quiz.py \
      --answers gemini_video_answers.txt

--no_graph writes video-only prompts (no scene graph, graph clause dropped from
the system prompt) -- the ablation twin of the run above:
  python gemini_subsample_video_quiz.py --no_graph \
      --prompts gemini_videoonly_prompts.txt --template gemini_videoonly_answers.txt

Make the subsample first with:
  python spatial_reasoning/finetuning/gemini_eval/gemini_vsibench_subsample_eval.py --sample

--selfcheck scores the ground truth (must print 1.0).
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from gemini_vsibench_subsample_eval import DEFAULT_SG_DIR, to_v2_nodes  # noqa: E402
from gemini_subsample_manual_quiz import (  # noqa: E402
    answer_hint, load_subsample, parse_answers, score,
)

DEFAULT_VIDEO_DIR = Path("/scratch/ducpham/Working/dataset/vsib/vsibench/scannetpp")

# gemini_vsibench_subsample_eval.SYSTEM_PROMPT with the video added; the
# scene-graph format sentence is kept verbatim so the graph reads the same.
SYSTEM_PROMPT = (
    "You are a spatial-reasoning assistant. You are given a video walkthrough of "
    "a room and a 3D scene graph of that same room: a list of objects, one per "
    "line as '<id> <name> size=(dx,dy,dz) base=(x,y,z_min)', where size is the "
    "object's extent in meters and base is its minimum-corner position "
    "(x,y,z_min) in meters. Use both the video and the scene graph to answer the "
    "questions.")

# --no_graph: same prompt with the scene-graph clause removed, so the graph is
# the only variable between the two runs.
SYSTEM_PROMPT_VIDEO_ONLY = (
    "You are a spatial-reasoning assistant. You are given a video walkthrough of "
    "a room. Use the video to answer the questions.")


def group_by_scene(rows: list[dict]) -> "OrderedDict[str, list[dict]]":
    """Scene -> its questions, first-appearance order (so the sheet is stable)."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for r in rows:
        groups.setdefault(r["scene"], []).append(r)
    return groups


def flatten(groups) -> list[dict]:
    return [r for qs in groups.values() for r in qs]


def video_path(scene: str, video_dir: Path) -> Path:
    p = video_dir / f"{scene}.mp4"
    if not p.exists():
        raise SystemExit(f"missing video: {p}")
    return p


def write_prompts(groups, path: Path, sg_dir: Path, video_dir: Path,
                  use_graph: bool = True) -> None:
    blocks = []
    for i, (scene, qs) in enumerate(groups.items(), 1):
        if use_graph:
            graph = to_v2_nodes((sg_dir / scene / "scene_graph.txt").read_text())
            body = f"{SYSTEM_PROMPT}\n\n3D scene graph:\n{graph}"
        else:
            body = SYSTEM_PROMPT_VIDEO_ONLY
        questions = "\n\n".join(
            f"Q{j} [id={r['id']}] ({answer_hint(r)}):\n{r['question']}"
            for j, r in enumerate(qs, 1))
        blocks.append(
            f"===== [{i}/{len(groups)}] scene={scene}  ({len(qs)} questions) =====\n"
            f"UPLOAD VIDEO: {video_path(scene, video_dir)}\n\n"
            f"{body}\n\n{questions}")
    path.write_text("\n\n".join(blocks) + "\n")
    print(f"Wrote {len(groups)} scene blocks "
          f"({sum(len(q) for q in groups.values())} questions) to {path}")


def write_template(groups, path: Path) -> None:
    lines = ["# VSI-Bench video answers -- one answer after each 'ANSWER:'.",
             "# Order matches the prompts file (grouped by scene)."]
    for i, (scene, qs) in enumerate(groups.items(), 1):
        lines.append(f"# --- [{i}/{len(groups)}] scene={scene}")
        for j, r in enumerate(qs, 1):
            lines.append(f"#   Q{j} id={r['id']} {r['question_type']} ({answer_hint(r)})")
            lines.append("ANSWER: ")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote answers template to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsample", type=Path, default=_HERE / "gemini_subsample.jsonl")
    ap.add_argument("--sg_dir", type=Path, default=DEFAULT_SG_DIR)
    ap.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--prompts", type=Path, help="write the prompts here")
    ap.add_argument("--template", type=Path, help="write a blank answers sheet here")
    ap.add_argument("--answers", type=Path, help="score the answers in this file")
    ap.add_argument("--no_graph", action="store_true",
                    help="video-only prompts: drop the scene graph")
    ap.add_argument("--selfcheck", action="store_true",
                    help="score the ground truth (must be 1.0)")
    args = ap.parse_args()

    groups = group_by_scene(load_subsample(args.subsample))
    rows = flatten(groups)
    print(f"{len(rows)} questions over {len(groups)} scenes from {args.subsample}")

    if args.prompts:
        write_prompts(groups, args.prompts, args.sg_dir, args.video_dir,
                      use_graph=not args.no_graph)
    if args.template:
        write_template(groups, args.template)
    if args.selfcheck:
        assert score(rows, [r["gt"] for r in rows])["vsibench/overall"] == 1.0
        print("selfcheck OK")
    if args.answers:
        score(rows, parse_answers(args.answers))
    if not any((args.prompts, args.template, args.answers, args.selfcheck)):
        ap.error("nothing to do: pass --prompts/--template/--answers/--selfcheck")


if __name__ == "__main__":
    main()
