#!/bin/bash
# One VSI-Bench eval shard on 1 A100, bf16 (submitted by scripts/eval.sh).
#   args: <experiment> <checkpoint> <eval dir> <num shards> [<num parts> <part index>]
# The shard index is the array task id. Resumes from rows already on disk.
set -euo pipefail
source "${SR_ROOT}/scripts/lib.sh"
load_config "$1"
CKPT=$2 EVAL_DIR=$3 NUM_SHARDS=$4
activate_env a100
offline
echo "=== ${SLURM_JOB_ID} | ${EXP} | shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} ${5:+part $6/$5} | ${CKPT} ==="
args=(
  "${MODEL_ARGS[@]}" "${VGGT_ARGS[@]}" "${EVAL_ARGS[@]}"
  --adapter_path "${CKPT}"
  --output_dir "${EVAL_DIR}"
  --vsibench_dir "${SR_VSIBENCH}"
  --num_shards "${NUM_SHARDS}" --shard_index "${SLURM_ARRAY_TASK_ID}"
  --dtype bf16
)
[[ -n "${5:-}" ]] && args+=(--num_parts "$5" --part_index "$6")
[[ -n "${EVAL_MAX_SAMPLES:-}" ]] && args+=(--max_samples "${EVAL_MAX_SAMPLES}")   # smoke only
python -m sr_vggt.eval "${args[@]}"
