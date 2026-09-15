#!/bin/bash
#SBATCH --job-name=prof3ie2e
#SBATCH --account=btf@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

cd "${WORK}/spatial_reasoning/finetuning"
source "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate vsibench_eval_full
set -a; source "${WORK}/spatial_reasoning/finetuning/site.env"; set +a

export HF_HOME="${SR_HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUTF8=1
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export FORCE_QWENVL_VIDEO_READER=decord
export DECORD_EOF_RETRY_MAX=40960
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name --format=csv,noheader
echo "=== job ${SLURM_JOB_ID} | cpus=${SLURM_CPUS_PER_TASK} ==="

for V in joint cold; do
  echo; echo "###################### ${V} e2e ######################"
  python profiling/profile_micro_step.py --variant "${V}" --phase e2e \
    --n 400 --steps 60 --workers 8 || echo "${V} FAILED"
done
