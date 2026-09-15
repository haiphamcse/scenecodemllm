"""Completeness audit of VSI-Bench (and ReVSI) eval result dirs. Read-only.

Walks <results_root>/*/eval_* and reports, per dir: shard result files, prediction rows,
unique row ids, missing / duplicate / mismatched rows vs the split eval.py used, whether
results_full_merged.json exists and covers every row, the merged score, and (--rescore)
an independent recomputation of the VSI-Bench score from the merged predictions.

  python scripts/audit/check_eval_results.py --results_root $SCRATCH/results --hf_home $WORK/cache --rescore

VSI-Bench rows are keyed by eval.py's `index` = position in the `full` split loaded via
datasets (video-sorted later, so the position is the dataset row). ReVSI rows carry the
parquet `id`.
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

MCA = {"object_rel_direction_easy", "object_rel_direction_medium", "object_rel_direction_hard",
       "object_rel_distance", "route_planning", "obj_appearance_order"}
NA = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}
THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]  # 0.50 .. 0.95


def load_full_split(hf_home):
    """Same call eval.py makes; falls back to the hub snapshot's test.jsonl."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset
        ds = load_dataset("nyu-visionx/VSI-Bench", "full", cache_dir=hf_home)["test"]
        return [{"id": r["id"], "question_type": r["question_type"], "ground_truth": str(r["ground_truth"])}
                for r in ds], "datasets"
    except Exception as e:  # noqa: BLE001
        hits = sorted(glob.glob(f"{hf_home}/hub/datasets--nyu-visionx--VSI-Bench/snapshots/*/test.jsonl"))
        if not hits:
            raise SystemExit(f"cannot load VSI-Bench full split: {e}")
        rows = [json.loads(l) for l in open(hits[-1])]
        return [{"id": r["id"], "question_type": r["question_type"], "ground_truth": str(r["ground_truth"])}
                for r in rows], hits[-1]


def score_vsibench(rows):
    """Independent VSI-Bench scorer: rows = [{question_type, prediction, ground_truth}]."""
    per = {}
    for r in rows:
        qt, pred, gt = r["question_type"], r["prediction"], str(r["ground_truth"])
        tok = pred.split(" ")[0].rstrip(".").strip()
        if qt in MCA:
            s = float(tok.lower() == gt.lower())
        elif qt in NA:
            try:
                p, t = float(tok), float(gt)
                # +1e-9: a rel error sitting exactly on a threshold (1.1 vs 1.0) counts as
                # within it; lmms-eval's np.linspace thresholds get this by float luck.
                s = sum(abs(p - t) / t <= 1 - c + 1e-9 for c in THRESHOLDS) / len(THRESHOLDS)
            except (ValueError, ZeroDivisionError):
                s = 0.0
        else:
            raise ValueError(qt)
        per.setdefault(qt, []).append(s)
    cat = {qt: sum(v) / len(v) for qt, v in per.items()}
    dirs = [cat.pop(k) for k in list(cat) if k.startswith("object_rel_direction_")]
    if dirs:
        cat["object_rel_direction"] = sum(dirs) / len(dirs)
    cat["overall"] = sum(cat.values()) / len(cat)
    return cat


def read_jsonl(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def check_vsib(d, split_rows, rescore, tol):
    exp_n = len(split_rows)
    shard_json = sorted(d.glob("results_full_shard*.json"))
    pred_files = sorted(d.glob("predictions_full_shard*.jsonl")) or sorted(d.glob("predictions_full.jsonl"))
    rows = list(read_jsonl(pred_files))
    idx = Counter(r["index"] for r in rows)
    seen = set(idx)
    dups = sorted(i for i, c in idx.items() if c > 1)
    missing = sorted(set(range(exp_n)) - seen)
    out_of_range = sorted(i for i in seen if i >= exp_n)
    mism = [r["index"] for r in rows if r["index"] < exp_n and (
        r["question_type"] != split_rows[r["index"]]["question_type"]
        or str(r["ground_truth"]) != split_rows[r["index"]]["ground_truth"])]
    nshards = {int(m.group(1)) for f in pred_files for m in [re.search(r"of(\d+)\.jsonl$", f.name)] if m}
    merged = d / "results_full_merged.json"
    mj = json.loads(merged.read_text()) if merged.exists() else None
    rec = dict(dir=f"{d.parent.name}/{d.name}", kind="vsib", shard_results=len(shard_json),
               pred_files=len(pred_files), shard_counts=sorted(nshards), rows=len(rows), unique=len(seen),
               missing=len(missing), missing_ids=[split_rows[i]["id"] for i in missing[:10]],
               dups=len(dups), dup_ids=[split_rows[i]["id"] for i in dups[:10]],
               mismatch=len(mism), out_of_range=len(out_of_range), merged=merged.exists(),
               merged_n=mj["num_samples"] if mj else None,
               merged_overall=mj["scores"].get("vsibench/overall") if mj else None)
    if rescore and rows:
        first = {}
        for r in rows:  # merge_eval_shards keeps the first occurrence of an index
            first.setdefault(r["index"], r)
        my = score_vsibench(list(first.values()))
        rec["rescore_overall"] = round(my["overall"], 6)
        if mj:
            key = lambda k: k.split("/")[1].replace("_accuracy", "").replace("_mra", "")
            rec["rescore_diff"] = {k: (round(my[key(k)], 6), v) for k, v in mj["scores"].items()
                                   if abs(my[key(k)] - v) > tol}
    complete = (len(seen) == exp_n and not dups and not missing and not mism and not out_of_range)
    if not pred_files:
        rec["verdict"] = "NO_PREDICTIONS"
    elif mj and not complete:
        rec["verdict"] = "INVALID"
    elif mj and mj["num_samples"] != exp_n:
        rec["verdict"] = "INVALID"
    elif "vsibench/overall" in rec.get("rescore_diff", {}):
        rec["verdict"] = "SCORE_MISMATCH"
    elif not mj:
        rec["verdict"] = "COMPLETE_UNMERGED" if complete else "INCOMPLETE"
    elif len(nshards) > 1 or len(shard_json) != len(pred_files):
        rec["verdict"] = "OK_BUT_MIXED_SHARDS"
    else:
        rec["verdict"] = "OK"
    return rec


def check_revsi(d, revsi_dir):
    merged = d / "results_merged.json"
    mj = json.loads(merged.read_text()) if merged.exists() else None
    subset = (mj or {}).get("subset") or re.search(r"revsi(\d+)", d.name).group(1) + "_frame"
    pred_files = sorted(d.glob("predictions_shard*.jsonl")) or sorted(d.glob("predictions.jsonl"))
    rows = list(read_jsonl(pred_files))
    idc = Counter(r["id"] for r in rows)
    rec = dict(dir=f"{d.parent.name}/{d.name}", kind=f"revsi:{subset}", shard_results=len(list(d.glob("results_shard*.json"))),
               pred_files=len(pred_files), rows=len(rows), unique=len(idc),
               dups=sum(1 for c in idc.values() if c > 1), merged=merged.exists(),
               merged_n=(mj or {}).get("scores", {}).get("n"), merged_overall=(mj or {}).get("scores", {}).get("overall_acc"))
    pq = sorted(glob.glob(f"{revsi_dir}/{subset}/test-*.parquet"))
    if pq:
        import pandas as pd
        exp = set(int(i) for i in pd.read_parquet(pq[0], columns=["id"])["id"])
        rec["expected"] = len(exp); rec["missing"] = len(exp - set(idc)); rec["extra"] = len(set(idc) - exp)
        complete = not rec["missing"] and not rec["extra"] and not rec["dups"]
        rec["verdict"] = ("INVALID" if mj and not complete else "OK" if mj else
                          "COMPLETE_UNMERGED" if complete else "INCOMPLETE")
    else:
        rec["verdict"] = "NOT_CHECKED (no parquet)"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=os.environ.get("SR_OUTPUT_ROOT", "results"))
    ap.add_argument("--hf_home", default=os.environ.get("SR_HF_HOME", os.environ.get("HF_HOME")))
    ap.add_argument("--revsi_dir", default=os.path.join(os.environ.get("SCRATCH", ""), "dataset/revsi"))
    ap.add_argument("--rescore", action="store_true", help="recompute VSI-Bench score independently")
    ap.add_argument("--tol", type=float, default=2e-3,
                    help="rescore vs merged tolerance; the vendored lmms-eval scorer differs from this one by up to "
                         "~1.3e-3 on exact-boundary numeric rows (float rounding of 1-threshold), never more")
    ap.add_argument("--json", type=Path, default=None, help="also dump records here")
    a = ap.parse_args()

    split_rows, src = load_full_split(a.hf_home)
    print(f"full split: {len(split_rows)} rows from {src}")
    recs = []
    for d in sorted(Path(a.results_root).glob("*/eval_*")):
        if not d.is_dir():
            continue
        recs.append(check_revsi(d, a.revsi_dir) if d.name.startswith("eval_revsi") else
                    check_vsib(d, split_rows, a.rescore, a.tol))
    hdr = ("dir", "kind", "shard_results", "rows", "unique", "missing", "dups", "mismatch", "merged", "merged_n",
           "merged_overall", "rescore_overall", "verdict")
    print("\t".join(hdr))
    for r in recs:
        print("\t".join(str(r.get(k, "")) for k in hdr))
    for r in recs:
        if r.get("rescore_diff") or r.get("missing_ids") or r.get("dup_ids") or r.get("shard_counts", [None]) not in ([], [8], [16], [None]):
            print("DETAIL", json.dumps({k: v for k, v in r.items() if k in ("dir", "shard_counts", "missing_ids", "dup_ids", "out_of_range", "rescore_diff")}))
    if a.json:
        a.json.write_text(json.dumps(recs, indent=1))


if __name__ == "__main__":
    main()
