#!/bin/bash
# One VGGT export shard on 1 V100-32GB, fp32 compute (V100 has no bf16), bf16 on disk
# (submitted by scripts/export_vggt.sh). args: <cache> <num shards>; shard = array task id.
set -euo pipefail
source "${SR_ROOT}/scripts/lib.sh"
load_cache "$1"
activate_env v100
offline
echo "=== ${SLURM_JOB_ID} | cache $1 | shard ${SLURM_ARRAY_TASK_ID}/$2 | $(hostname) ==="
mixes=("${SR_MIX_190K}")
[[ -f "${SR_MIX_TRACES}" ]] && mixes+=("${SR_MIX_TRACES}")
python -m sr_vggt.export "${CACHE_ARGS[@]}" \
  --cache_root "${CACHE_ROOT}" --mix_jsonl "${mixes[@]}" --data_root "${SR_VSI590K}" \
  --num_shards "$2" --shard_index "${SLURM_ARRAY_TASK_ID}" --dtype fp32 --store_dtype bf16
