#!/bin/bash
# Data check for one experiment (login node, CPU): mix rows and every video under SR_VSI590K,
# VSI-Bench full = 5,130 rows with all videos, weights offline. Exit 0 = ready to train and eval.
#   scripts/verify_data.sh <experiment>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
load_config "${1:?usage: scripts/verify_data.sh <experiment>}"
activate_env h100
offline
python -m sr_vggt.data "${MODEL_ARGS[@]}" "${VGGT_ARGS[@]}" \
  --jsonl_path "${MIX_JSONL}" --data_root "${SR_VSI590K}" --vsibench_dir "${SR_VSIBENCH}"
