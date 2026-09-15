#!/bin/bash
# Keeps N dev windows queued per run (afterany-chained) until the run's final
# save (adapter_config.json at the top of its output dir) exists. Login-node watcher:
#   nohup bash scripts/jz/chain_topup.sh > $WORK/logs/chain_topup.log 2>&1 &
# RUNS entries: jobname|train script|results dir|windows to keep queued|max windows to add|submit env (GPUKIND NGPU QOS SR_*)
set -u
cd "${WORK}/spatial_reasoning/finetuning"
set -a; source site.env; set +a
RUNS=(
  "jz_train_180k_lat512_ms|scripts/idea_3i_180k_cached_64f_lat512_multiscale/jz_train_180k_lat512_ms.sh|idea_3i_180k_cached_64f_lat512_multiscale|3|14|GPUKIND=h100 NGPU=2 QOS=qos_gpu_h100-dev"
  "jz_train_180k_es|scripts/idea_3i_180k_es_64f_lat512/jz_train_180k_es.sh|idea_3i_180k_es_64f_lat512|3|10|GPUKIND=h100 NGPU=2 QOS=qos_gpu_h100-dev"
)
declare -A added=() done=()
log() { echo "$(date '+%F %T') $*"; }
log "watching: ${RUNS[*]}"
for _ in $(seq 1 864); do   # 72 h at 5 min
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name script dir keep max envs <<<"${spec}"
    [[ -n "${done[$name]:-}" ]] && continue
    if [[ -f "${SR_OUTPUT_ROOT}/${dir}/adapter_config.json" ]]; then
      done[$name]=1; log "${name}: final save present, done (last ckpt $(ls -d ${SR_OUTPUT_ROOT}/${dir}/checkpoint-* | sed 's#.*-##' | sort -n | tail -1))"; continue
    fi
    queued=$(squeue -h -u "${USER}" -n "${name}" -o %i | wc -l)
    while (( queued < keep && ${added[$name]:-0} < max )); do
      last=$(squeue -h -u "${USER}" -n "${name}" -o %i | sort -n | tail -1)
      j=$(env ${envs} TIME=02:00:00 scripts/jz/submit.sh "${script}" 2>&1 | grep -oE "[0-9]+$")
      [[ -n "${j}" ]] || { log "${name}: submit failed"; break; }
      if [[ -n "${last}" ]]; then scontrol hold "${j}"; scontrol update jobid="${j}" Dependency="afterany:${last}"; scontrol release "${j}"; fi
      added[$name]=$(( ${added[$name]:-0} + 1 )); queued=$(( queued + 1 ))
      log "${name}: added ${j} (afterany:${last:-none}), queued now ${queued}, ckpt $(ls -d ${SR_OUTPUT_ROOT}/${dir}/checkpoint-* 2>/dev/null | sed 's#.*-##' | sort -n | tail -1)"
    done
  done
  (( ${#done[@]} == ${#RUNS[@]} )) && { log "all runs done"; exit 0; }
  sleep 300
done
log "gave up after 72 h"
