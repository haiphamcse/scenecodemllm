#!/bin/bash
# Waits for checkpoint-300 of the two 130k runs, submits the 16-shard V100 eval for each
# as soon as it lands, merges the shards when the array finishes, logs the score.
# Runs detached on a login node:  nohup bash scripts/jz/eval_watch.sh &
set -u
cd "${WORK}/spatial_reasoning/finetuning"
source "${WORK}/miniconda3/etc/profile.d/conda.sh"; conda activate vsibench_eval_full
set -a; source site.env; set +a
STEP="${STEP:-702}"
RUNS="${RUNS:-idea_3i_180k_cached_64f_lat512}"
EVAL_SLURM="${EVAL_SLURM:-}"   # e.g. the 64f fork's eval_full_v100_64f.slurm
declare -A job=()   # run -> array job id once submitted
declare -A done=()  # run -> 1 once merged
log() { echo "$(date '+%F %T') $*"; }
log "watching ${RUNS} for checkpoint-${STEP}"
for _ in $(seq 1 576); do   # 48 h at 5 min
  for r in ${RUNS}; do
    [[ -n "${done[$r]:-}" ]] && continue
    ck="${SR_OUTPUT_ROOT}/${r}/checkpoint-${STEP}"
    out="${SR_OUTPUT_ROOT}/${r}/eval_ckpt${STEP}_full"
    if [[ -z "${job[$r]:-}" ]]; then
      # trainer_state.json is written last; the mtime guard skips a save still in flight.
      if [[ -f "${ck}/trainer_state.json" ]] && (( $(date +%s) - $(stat -c %Y "${ck}/trainer_state.json") > 60 )); then
        mkdir -p "${out}"
        id=$(cd "${WORK}/logs" && sbatch --parsable --export=ALL,CKPT="${ck}",EVAL_OUT="${out}" \
             "${EVAL_SLURM:-${WORK}/spatial_reasoning/finetuning/scripts/idea_3i_180k_cached_64f_lat512/eval_full_v100.slurm}")
        job[$r]="${id%%;*}"; log "${r}: checkpoint-${STEP} found, submitted eval array ${job[$r]} -> ${out}"
      fi
    elif ! squeue -h -j "${job[$r]}" -o %i 2>/dev/null | grep -q .; then
      # results_*.json only exists for a shard that finished; a failed shard still leaves a
      # partial predictions file, which is how a 3542-row merge once scored as 0.5499.
      n=$(ls "${out}"/results_full_shard*.json 2>/dev/null | wc -l)
      rows=$(cat "${out}"/predictions_full_shard*.jsonl 2>/dev/null | wc -l)
      log "${r}: array ${job[$r]} finished, ${n}/16 shards complete, ${rows}/5130 rows"
      if (( n == 16 && rows == 5130 )); then
        python common/merge_eval_shards.py --eval_dir "${out}" --split full 2>&1 | tail -20
      else
        log "${r}: shards missing, not merging: $(sacct -j "${job[$r]}" -X -n -o JobID,State | tr -s ' ' | tr '\n' ';')"
      fi
      done[$r]=1
    fi
  done
  [[ ${#done[@]} -eq $(wc -w <<<"${RUNS}") ]] && { log "all done"; exit 0; }
  sleep 300
done
log "gave up after 48 h"
