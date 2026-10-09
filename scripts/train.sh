#!/bin/bash
# Train an experiment: chained 4 h windows on 2 x H100 (qos_gpu_h100-t3), kept queued by a
# login-node watcher until the final adapter is saved (~4-6 windows for the 2B runs).
#
#   scripts/train.sh <experiment> [extra sr_vggt.train flags]
#   SMOKE=1 scripts/train.sh <experiment> --max_steps 20     one 2 h dev window, no watcher,
#                                                            into ${SR_OUTPUT_ROOT}/_smoke_sr_vggt_<exp>
# Env overrides: NGPU (2), QOS, TIME, MAX_WINDOWS (12), OUT (run dir), INIT_CKPT (reason16k).
# The watcher logs to ${SR_LOG_DIR}/watch_<run>.log; stop it with `kill <pid>` (printed below),
# never `pkill -f` on a shared login node.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

WATCH=0
[[ "${1:-}" == --watch ]] && { WATCH=1; shift; }
load_config "${1:?usage: scripts/train.sh <experiment> [train flags]}"
shift
EXTRA=("$@")
SMOKE=${SMOKE:-0}
NGPU=${NGPU:-2}
if (( SMOKE )); then
  OUT=${OUT:-${SR_OUTPUT_ROOT}/_smoke_${RUN_NAME}}; QOS=${QOS:-qos_gpu_h100-dev}; TIME=${TIME:-02:00:00}
  JOB_NAME=_smoke_${RUN_NAME}
else
  OUT=${OUT:-${SR_OUTPUT_ROOT}/${RUN_NAME}}; QOS=${QOS:-qos_gpu_h100-t3}; TIME=${TIME:-04:00:00}
  JOB_NAME=${RUN_NAME}
fi
MAX_WINDOWS=${MAX_WINDOWS:-12}

submit() {  # [sbatch flags...] -> job id
  sbatch --parsable --job-name="${JOB_NAME}" --account="${SR_PROJECT}@h100" -C h100 --qos="${QOS}" \
    --nodes=1 --ntasks=1 --gres="gpu:${NGPU}" --cpus-per-task=$((NGPU * 24)) --hint=nomultithread \
    --time="${TIME}" --output="${SR_LOG_DIR}/%x-%j.out" --export=ALL,SR_ROOT="${SR_ROOT}" "$@" \
    "${SR_ROOT}/scripts/jobs/train.slurm" "${EXP}" "${OUT}" "${EXTRA[@]}"
}

if (( ! WATCH )); then
  [[ -f "${MIX_JSONL}" ]] || { echo "ERROR: mix ${MIX_JSONL} missing (scripts/unpack_data.sh)" >&2; exit 1; }
  [[ -d "${CACHE_ROOT}" ]] || { echo "ERROR: cache ${CACHE_ROOT} missing (scripts/export_vggt.sh ${CACHE})" >&2; exit 1; }
  [[ -f "${OUT}/adapter_config.json" ]] && { echo "${OUT} already holds a final adapter; nothing to do."; exit 0; }
  mkdir -p "${OUT}"
  if (( SMOKE )); then
    echo "smoke job $(submit) -> ${OUT}, log ${SR_LOG_DIR}/${JOB_NAME}-<id>.out"
    exit 0
  fi
  WLOG="${SR_LOG_DIR}/watch_${RUN_NAME}.log"
  # bash -l: a watcher started from a non-login shell has no Slurm/$WORK environment.
  nohup bash -l "$0" --watch "${EXP}" "${EXTRA[@]}" >> "${WLOG}" 2>&1 &
  echo "watcher pid $! on $(hostname) -> ${WLOG}; run dir ${OUT}"
  exit 0
fi

# ---- watcher: keep 2 windows queued (the next afterany the current) until the final save ----
set +e
windows=0
log "watching ${RUN_NAME}: ${OUT}, max ${MAX_WINDOWS} windows of ${TIME} on ${NGPU}x H100 (${QOS})"
while :; do
  if [[ -f "${OUT}/adapter_config.json" ]]; then
    ids=$(my_jobs "${JOB_NAME}") && [[ -n "${ids}" ]] && scancel ${ids}
    log "final adapter saved (last $(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sed 's#.*-##' | sort -n | tail -1)); cancelled: ${ids:-none}. done."
    exit 0
  fi
  ids=$(my_jobs "${JOB_NAME}") || { log "squeue failed, skip"; sleep 300; continue; }
  queued=$(wc -w <<<"${ids}")
  while (( queued < 2 )); do
    if (( windows >= MAX_WINDOWS )); then
      (( queued == 0 )) && { log "cap of ${MAX_WINDOWS} windows reached without a final save; stopping."; exit 1; }
      break
    fi
    last=$(tr ' ' '\n' <<<"${ids}" | sort -n | tail -1)
    if j=$(submit ${last:+--dependency=afterany:${last}}); then
      windows=$((windows + 1)); queued=$((queued + 1)); ids="${ids} ${j}"
      log "window ${windows}: job ${j} (after ${last:-none}), last ckpt $(ls -d "${OUT}"/checkpoint-* 2>/dev/null | sed 's#.*-##' | sort -n | tail -1)"
    else
      log "sbatch failed, retry in 5 min"; break
    fi
  done
  sleep 300
done
