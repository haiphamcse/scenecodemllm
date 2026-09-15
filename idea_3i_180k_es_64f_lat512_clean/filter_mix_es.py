"""vqa_train_180k.jsonl -> <out_dir>/vqa_train_180k_es.jsonl and vqa_train_180k_es_clean.jsonl.

es:       drop rows whose video has no det_es frames (60 ARKit videos).
es_clean: also drop the scenes in DROP (det_es_orientation.py audit, 2026-09-15).

  python idea_3i_180k_es_64f_lat512_clean/filter_mix_es.py <vqa_train_180k.jsonl> <out_dir>
"""
import json
import sys
from collections import Counter
from pathlib import Path

# (source dir, scene) -> reason.
DROP = {
    ("scannetppv2", "09d6e808b4"): "upside down",
    ("scannetppv2", "0f69aefe3d"): "upside down",
    ("scannetppv2", "6ef2ac745a"): "upside down",
    ("scannetppv2", "1b379f1114"): "sideways",
    ("scannetppv2", "1cbb105c6a"): "sideways",
    ("scannetppv2", "2c7c10379b"): "sideways",
    ("scannetppv2", "46638cfd0f"): "sideways",
    ("scannetppv2", "898a7dfd0c"): "sideways",
    ("scannetppv2", "aa852f7871"): "sideways",
    ("scannetppv2", "d27235711b"): "sideways",
    ("scannetppv2", "eea4ad9c04"): "sideways",
    ("arkitscenes", "41254810"): "sideways (41/64 frames)",
    ("scannetppv2", "7dab70c8c8"): "broken pose",
    ("scannetppv2", "928c9da20c"): "broken pose (180 deg flip)",
    ("scannetppv2", "120acffd90"): "broken pose (286 m jump)",
    ("scannetppv2", "46001f434d"): "broken pose (71 m jump)",
    ("scannetppv2", "ab4f373966"): "broken pose (49 m jump)",
    ("scannetppv2", "cc0aa81452"): "broken pose (41 m jump)",
}


def scene(r):
    v = Path(r["video"])
    return v.parent.name, v.stem


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {path}; per source {dict(Counter(scene(r)[0] for r in rows))}")


src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in open(src)]
has = {v: (Path(v).with_suffix("") / "det_es" / "det.json").is_file() for v in {r["video"] for r in rows}}
es = [r for r in rows if has[r["video"]]]
print(f"{src.name}: {len(rows)} rows; no det_es: {sum(not v for v in has.values())} videos, {len(rows) - len(es)} rows")
write(out / "vqa_train_180k_es.jsonl", es)

clean = [r for r in es if scene(r) not in DROP]
hit = Counter(scene(r) for r in es if scene(r) in DROP)
assert set(hit) == set(DROP), f"drop scenes not in es mix: {set(DROP) - set(hit)}"
write(out / "vqa_train_180k_es_clean.jsonl", clean)
with open(out / "dropped_scenes.txt", "w") as f:
    for k, why in DROP.items():
        f.write(f"{k[0]}/{k[1]}\t{hit[k]} rows\t{why}\n")
print(f"dropped {len(es) - len(clean)} rows from {len(DROP)} scenes -> {out / 'dropped_scenes.txt'}")
