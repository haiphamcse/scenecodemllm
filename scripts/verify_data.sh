#!/bin/bash
# Full-scan data check for one experiment (login node, CPU): mix rows, det_es frames, every cache
# entry fresh with one row per frame, VSI-Bench full = 5,130 rows with all videos, weights offline.
#   scripts/verify_data.sh <experiment>      exit 0 = ready to train and eval
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config "${1:?usage: scripts/verify_data.sh <experiment>}"
activate_env v100
offline
python -m sr_vggt.verify "${MODEL_ARGS[@]}" "${CACHE_ARGS[@]}" \
  --jsonl_path "${MIX_JSONL}" --data_root "${SR_VSI590K}" --vggt_cache_root "${CACHE_ROOT}" \
  --vsibench_videos "${SR_VSIBENCH}"
