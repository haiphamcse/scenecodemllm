"""vqa_train.jsonl -> vqa_train_es.jsonl: drop rows whose video has no det_es frames (60 ARKit videos).

  python idea_3i_180k_es_64f_lat512/filter_mix_es.py <in.jsonl> <out.jsonl>
"""
import json
import sys
from collections import Counter
from pathlib import Path

src, dst = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(src)]
has = {v: (Path(v).with_suffix("") / "det_es" / "det.json").is_file() for v in {r["video"] for r in rows}}
keep = [r for r in rows if has[r["video"]]]
with open(dst, "w") as f:
    for r in keep:
        f.write(json.dumps(r) + "\n")
dropped = [r for r in rows if not has[r["video"]]]
print(f"rows {len(rows)} -> {len(keep)} (dropped {len(dropped)}); videos {len(has)} -> {sum(has.values())} "
      f"(dropped {sum(not v for v in has.values())}: {dict(Counter(Path(v).parent.name for v, ok in has.items() if not ok))})")
print("kept per source:", dict(Counter(Path(r['video']).parent.name for r in keep)))
