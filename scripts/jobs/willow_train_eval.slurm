#!/bin/bash
#SBATCH --job-name=sr_vggt_willow
#SBATCH --partition=willow
#SBATCH --account=willow
#SBATCH --qos=proprietary
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
# Train <experiment> on 2 GPUs in one allocation, then eval the final adapter on VSI-Bench full
# (one shard per GPU), then merge + score. For non-JZ sites: site.env sets SR_PYTHON_BIN.
#   sbatch [--time=..] scripts/jobs/willow_train_eval.slurm <experiment> <output dir> [extra train flags...]
# Smoke: pass e.g. `--max_steps 6 --save_steps 3` and a _smoke_ output dir; eval then runs on
# that adapter with EVAL_MAX_SAMPLES rows per shard if set.
# Rerun the same command to resume: training picks up the newest checkpoint, eval the rows on disk.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export SR_ROOT="${SLURM_SUBMIT_DIR}"
EXP=$1 OUT=$2
shift 2
NGPU=${SLURM_GPUS_ON_NODE:-$(tr , '\n' <<< "${CUDA_VISIBLE_DEVICES}" | wc -l)}
export SLURM_GPUS_ON_NODE=${NGPU}
echo "=== ${SLURM_JOB_ID} | ${EXP} | GPUs ${CUDA_VISIBLE_DEVICES} (${NGPU}) | $(hostname) | out ${OUT} ==="

if [[ ! -f "${OUT}/adapter_config.json" ]]; then
  bash scripts/jobs/train.slurm "${EXP}" "${OUT}" "$@"
fi
[[ -f "${OUT}/adapter_config.json" ]] || { echo "ERROR: no final adapter in ${OUT}" >&2; exit 1; }

EVAL_DIR="${OUT}/eval_full"
mkdir -p "${EVAL_DIR}"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
pids=()
for i in "${!GPUS[@]}"; do
  SLURM_ARRAY_TASK_ID=$i CUDA_VISIBLE_DEVICES=${GPUS[$i]} \
    bash scripts/jobs/eval.slurm "${EXP}" "${OUT}" "${EVAL_DIR}" "${#GPUS[@]}" > "${EVAL_DIR}.shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
tail -n 3 "${EVAL_DIR}".shard*.log
[[ -n "${EVAL_MAX_SAMPLES:-}" ]] && { echo "smoke eval done (no merge)"; exit 0; }
source scripts/lib.sh && activate_env h100
python -m sr_vggt.eval merge "${EVAL_DIR}"
