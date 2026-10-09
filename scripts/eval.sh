#!/bin/bash
# VSI-Bench full-split eval of a checkpoint: 8 shards on A100 dev (bf16, 2 h each), resubmitting
# unfinished shards (eval.py resumes from the rows on disk), then merge + score. Runs as a
# login-node watcher; the score lands in <eval dir>/results_full_merged.json and the log.
#
#   scripts/eval.sh <experiment> <checkpoint dir>       e.g. main $SR_OUTPUT_ROOT/sr_vggt_main/checkpoint-743
# Env overrides: EVAL_DIR (<ckpt>/../eval_<ckpt name>_full), NUM_SHARDS (8), QOS (qos_gpu_a100-dev),
# TIME (02:00:00), MAX_ROUNDS (6). Configs with EVAL_PARTS > 1 (reason16k) split a timed-out
# shard's leftover rows over that many extra jobs instead of rerunning the shard alone.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WATCH=0
[[ "${1:-}" == --watch ]] && { WATCH=1; shift; }
load_config "${1:?usage: scripts/eval.sh <experiment> <checkpoint dir>}"
CKPT=$(realpath "${2:?checkpoint dir}")
EVAL_DIR=${EVAL_DIR:-$(dirname "${CKPT}")/eval_$(basename "${CKPT}")_full}
NUM_SHARDS=${NUM_SHARDS:-8}
QOS=${QOS:-qos_gpu_a100-dev}
TIME=${TIME:-02:00:00}
MAX_ROUNDS=${MAX_ROUNDS:-6}
JOB_NAME=eval_${RUN_NAME}_$(basename "${CKPT}")

if (( ! WATCH )); then
  [[ -f "${CKPT}/adapter_model.safetensors" ]] || { echo "ERROR: ${CKPT} has no adapter_model.safetensors" >&2; exit 1; }
  mkdir -p "${EVAL_DIR}"
  WLOG="${SR_LOG_DIR}/watch_eval_${RUN_NAME}_$(basename "${CKPT}").log"
  EVAL_DIR="${EVAL_DIR}" nohup bash -l "$0" --watch "${EXP}" "${CKPT}" >> "${WLOG}" 2>&1 &
  echo "eval watcher pid $! on $(hostname) -> ${WLOG}; eval dir ${EVAL_DIR}"
  exit 0
fi

submit() {  # <array spec> [parts index] -> job id
  sbatch --parsable --job-name="${JOB_NAME}" --account="${SR_PROJECT}@a100" -C a100 --qos="${QOS}" \
    --array="$1" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread \
    --time="${TIME}" --output="${SR_LOG_DIR}/%x-%A_%a.out" --export=ALL,SR_ROOT="${SR_ROOT}" \
    "${SR_ROOT}/scripts/jobs/eval.slurm" "${EXP}" "${CKPT}" "${EVAL_DIR}" "${NUM_SHARDS}" "${@:2}"
}
shard_done() {  # a shard is done when its own results file, or all its parts' files, exist
  local f="${EVAL_DIR}/results_full_shard$1of${NUM_SHARDS}"
  [[ -f "${f}.json" ]] && return 0
  (( EVAL_PARTS > 1 )) || return 1
  local k; for ((k = 0; k < EVAL_PARTS; k++)); do [[ -f "${f}.part${k}of${EVAL_PARTS}.json" ]] || return 1; done
}

set +e
log "eval ${EXP} ${CKPT} -> ${EVAL_DIR} (${NUM_SHARDS} shards, ${QOS})"
for ((round = 1; round <= MAX_ROUNDS; round++)); do
  todo=(); for ((i = 0; i < NUM_SHARDS; i++)); do shard_done "${i}" || todo+=("${i}"); done
  (( ${#todo[@]} == 0 )) && break
  ids=()
  for i in "${todo[@]}"; do
    if (( round > 1 && EVAL_PARTS > 1 )); then
      # Leftover rows of a timed-out shard -> EVAL_PARTS jobs; later rounds resume unfinished parts.
      for ((k = 0; k < EVAL_PARTS; k++)); do
        [[ -f "${EVAL_DIR}/results_full_shard${i}of${NUM_SHARDS}.part${k}of${EVAL_PARTS}.json" ]] && continue
        j=$(submit "${i}" "${EVAL_PARTS}" "${k}") && ids+=("${j}")
      done
    else
      j=$(submit "${i}") && ids+=("${j}")
    fi
  done
  log "round ${round}: shards ${todo[*]} -> jobs ${ids[*]:-none}"
  (( ${#ids[@]} )) || { sleep 300; continue; }
  while :; do
    sleep 300
    left=$(my_jobs "${JOB_NAME}") || { log "squeue failed, skip"; continue; }
    [[ -z "${left}" ]] && break
  done
done

activate_env v100   # CPU-side merge on the login node
python -m sr_vggt.merge_eval --eval_dir "${EVAL_DIR}" 2>&1 | tail -15
