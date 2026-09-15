#!/bin/bash
#SBATCH --job-name=benchattn
#SBATCH --account=btf@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
# Kernel-level A/B for the cached H100 continuation. Same model, same synthetic
# 6800-token step, same GPU: torch 2.6 conda vs torch 2.8 / 2.13 venvs, sdpa vs flash.
#
# Modules on H100 nodes: the job inherits the LOGIN node's MODULEPATH (--export=ALL), under
# which pytorch-gpu resolves to a build whose openmpi is absent here (libmpi.so.40).
# Re-init modules from this node's profile and load arch/h100 to get the H100 tree.
# /etc/profile touches unset vars (PS1 ...) and non-zero intermediates: source it
# BEFORE the strict flags, or set -eu kills the job silently at line 1.
unset MODULEPATH; source /etc/profile >/dev/null 2>&1 || true
set -euo pipefail
module purge
cd "${WORK}/spatial_reasoning/finetuning"
set -a; source site.env; set +a
export HF_HOME="${SR_HF_HOME}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== A: conda vsibench_eval_full, torch 2.6, sdpa ==="
source "${WORK}/miniconda3/etc/profile.d/conda.sh"; conda activate vsibench_eval_full
python profiling/bench_attn_env.py --attn sdpa
conda deactivate

for V in 2.8.0 2.13.0; do
  N=$(echo "$V" | sed 's/\.//g' | cut -c1-3)
  module purge; module load arch/h100; module load "pytorch-gpu/py3/${V}"
  source "${WORK}/envs/h100_t${N}/bin/activate"
  echo "=== venv h100_t${N}: $(python -c 'import torch;print("torch",torch.__version__,"cuda",torch.version.cuda)') ==="
  echo "=== B(${V}): sdpa ===";              python profiling/bench_attn_env.py --attn sdpa
  echo "=== C(${V}): flash_attention_2 ==="; python profiling/bench_attn_env.py --attn flash_attention_2
  deactivate
done
