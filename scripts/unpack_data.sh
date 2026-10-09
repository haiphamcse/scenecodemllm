#!/bin/bash
# Unpack the data zip into SR_DATA (site.env) and count what landed. The jsonl `video` paths are
# relative to SR_VSI590K, so nothing is rewritten.
#   scripts/unpack_data.sh <data.zip>
# ~300k small files: Lustre is metadata-bound, expect tens of minutes. Use a detached session
# (screen/tmux, or nohup) so a dropped ssh does not leave a half-extracted tree.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
ZIP=$(realpath "${1:?usage: scripts/unpack_data.sh <data.zip>}")
mkdir -p "${SR_DATA}"
unzip -q -n "${ZIP}" -d "${SR_DATA}"
echo "mix 190k:  $(wc -l < "${SR_MIX_190K}") rows (expect 190000)"
echo "traces:    $(wc -l < "${SR_MIX_TRACES}") rows (expect 145647)"
echo "det_es:    $(find "${SR_VSI590K}" -path '*/det_es/frames/*.jpg' | wc -l) frames"
echo "VSI-Bench: $(find "${SR_VSIBENCH}" -name '*.mp4' | wc -l) videos"
echo "next: scripts/verify_data.sh <experiment> (after export_vggt.sh) checks every clip"
