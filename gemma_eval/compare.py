"""Per-category n + score for several predictions_merged.jsonl on their common ids, and identical-prediction count
between the last two runs.  python compare.py name=path.jsonl name2=path2.jsonl ... > compare.txt"""
import importlib.util, json, sys
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "gemini_det_eval", Path(__file__).resolve().parent.parent / "gemini_eval" / "det_vsibench_eval.py")
G = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(G)

runs = {}
for arg in sys.argv[1:]:
    name, p = arg.split("=", 1)
    runs[name] = {r["id"]: r for r in G.read_jsonl(Path(p))}
ids = sorted(set.intersection(*(set(r) for r in runs.values())))
first = runs[next(iter(runs))]
n_type = Counter(first[i]["question_type"].replace("_easy", "").replace("_medium", "").replace("_hard", "") for i in ids)
scores = {k: G.score([rs[i] for i in ids]) for k, rs in runs.items()}
print(f"common ids: {len(ids)}\n")
print(f"{'category':32s} {'n':>4s} " + " ".join(f"{k:>18s}" for k in runs))
for tag in scores[next(iter(runs))]:
    cat = tag.split("/")[1]
    n = len(ids) if cat == "overall" else next(v for k, v in n_type.items() if cat.startswith(k))
    print(f"{cat:32s} {n:4d} " + " ".join(f"{scores[k][tag]:18.3f}" for k in runs))
a, b = list(runs)[-2:]
same = Counter(first[i]["question_type"] for i in ids if runs[a][i]["prediction"] == runs[b][i]["prediction"])
print(f"\nidentical predictions {a} vs {b}: {sum(same.values())}/{len(ids)}")
for t in sorted(set(first[i]["question_type"] for i in ids)):
    print(f"  {t:32s} {same[t]:3d}/{sum(1 for i in ids if first[i]['question_type'] == t)}")
