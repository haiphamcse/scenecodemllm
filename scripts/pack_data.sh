#!/bin/bash
# Build the data zip a new user receives (run by the data owner on Jean Zay; NOT by new users).
#   scripts/pack_data.sh <out.zip>
# Contents (paths inside the zip, relative to the user's SR_DATA):
#   mixes/vqa_train_190k_es_clean.jsonl   190k mix, `video` rewritten relative to VSI-590K/
#   mixes/simplesr_train_traces.jsonl     SimpleSR reasoning traces, same rewrite, `meta` kept
#   VSI-590K/<source>/<scene>/det_es/frames/frameNN.jpg   every clip either mix uses
#   vsibench/{arkitscenes,scannet,scannetpp}/<scene>.mp4   VSI-Bench eval videos
# VGGT caches and weights are NOT included (export_vggt.sh / download_weights.sh rebuild them).
#
# LICENSING: the frames and videos derive from ScanNet, ScanNet++ and ARKitScenes, whose terms
# forbid redistribution. Share the zip privately, only with people who have signed those
# datasets' agreements themselves. Never upload it anywhere public.
#
# Sources (defaults = the original Jean Zay layout):
#   SRC_MIX_190K SRC_MIX_TRACES SRC_VSI590K SRC_VSIBENCH
set -euo pipefail
OUT_ZIP=$(realpath -m "${1:?usage: scripts/pack_data.sh <out.zip>}")
SRC_MIX_190K=${SRC_MIX_190K:-${SCRATCH}/dataset/idea_3i_dataset_jsons/vqa_train_190k_es_clean.jsonl}
SRC_MIX_TRACES=${SRC_MIX_TRACES:-${SCRATCH}/dataset/idea_3i_dataset_jsons/simplesr_traces_20260920/train_traces.jsonl}
SRC_VSI590K=${SRC_VSI590K:-${SCRATCH}/dataset/vsi_590k/VSI-590K}
SRC_VSIBENCH=${SRC_VSIBENCH:-${SCRATCH}/dataset/vsib/vsibench}

STAGE=$(mktemp -d "${SCRATCH}/sr_vggt_pack.XXXXXX")
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/mixes"
ln -s "$(realpath "${SRC_VSI590K}")" "${STAGE}/VSI-590K"
ln -s "$(realpath "${SRC_VSIBENCH}")" "${STAGE}/vsibench"

# Relative video paths + the file list (frames of every used clip, every VSI-Bench mp4).
python3 - "${STAGE}" "${SRC_VSI590K}" "${SRC_MIX_190K}" "${SRC_MIX_TRACES}" <<'PY'
import json, os, sys
from pathlib import Path
stage, root = Path(sys.argv[1]), os.path.realpath(sys.argv[2])
names = {sys.argv[3]: "vqa_train_190k_es_clean.jsonl", sys.argv[4]: "simplesr_train_traces.jsonl"}
clips = set()
for src, name in names.items():
    n = 0
    with open(src) as f, open(stage / "mixes" / name, "w") as out:
        for line in f:
            row = json.loads(line)
            v = os.path.realpath(row["video"])
            assert v.startswith(root + "/"), f"{src}: {row['video']} is outside {root}"
            row["video"] = os.path.relpath(v, root)
            clips.add(row["video"])
            out.write(json.dumps(row) + "\n")
            n += 1
    print(f"{name}: {n} rows")
files = []
for rel in sorted(clips):
    frames = sorted((Path(root) / rel).with_suffix("").joinpath("det_es", "frames").glob("frame*.jpg"))
    assert frames, f"no det_es frames for {rel}"
    files += [f"VSI-590K/{f.relative_to(root)}" for f in frames]
vsib = sorted(p.relative_to(stage / "vsibench") for p in (stage / "vsibench").glob("*/*.mp4"))
files += [f"vsibench/{p}" for p in vsib]
files += [f"mixes/{n}" for n in names.values()]
(stage / "files.txt").write_text("\n".join(files) + "\n")
print(f"{len(clips)} clips, {len(files) - len(vsib) - 2} frames, {len(vsib)} VSI-Bench videos")
PY

# -0: JPEG/MP4 are already compressed. zip follows the two symlinks; Zip64 handles >65k entries.
cd "${STAGE}" && zip -q -0 "${OUT_ZIP}" -@ < files.txt
ls -lh "${OUT_ZIP}"
