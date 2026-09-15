#!/bin/bash
#SBATCH --job-name=det_vsib
#SBATCH --account=btf@a100
#SBATCH --constraint=a100
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fswork/projects/rech/btf/uff48lg/logs/%x-%A_%a.out
# Text-only VSI-Bench-Debiased ScanNet eval from det jsons, one shard per job (1 GPU each).
#   A100 array:  sbatch --array=0-3 scripts/gemini_eval/det_vsibench.slurm
#   H100 single: ARCH=h100 MODULE=2.13.0 SHARD=i sbatch --account=btf@h100 --constraint=h100 --qos=qos_gpu_h100-dev \
#                  --time=02:00:00 --cpus-per-task=24 scripts/gemini_eval/det_vsibench.slurm
#   smoke: NSHARDS=47 OUT=... sbatch --qos=qos_gpu_a100-dev --time=00:40:00 --array=0 ...   (20 questions strided by id)
#   LLM=Qwen/Qwen3-14B VENV=a100_t280 ... ; then: python gemini_eval/det_vsibench_eval.py --out_dir <OUT> --merge
set -eu
# venv on the module stack (see scripts/idea_3a_180k_cached_64f/jz_train_3a_180k.sh); /etc/profile trips set -u.
set +eu; unset MODULEPATH; source /etc/profile >/dev/null 2>&1; set -eu
ARCH="${ARCH:-a100}"; MODULE="${MODULE:-2.8.0}"   # h100: MODULE=2.13.0 (arch/h100 tree; grouped_mm MoE path needs sm_90)
module purge; module load "arch/$ARCH"; module load "pytorch-gpu/py3/$MODULE"
source "$WORK/envs/${VENV:-${ARCH}_gemma4}/bin/activate"   # {a100,h100}_gemma4: transformers 5.5.3 (gemma4); a100_t280: 5.2.0 (Qwen3 only)
export HF_HOME="$WORK/cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUTF8=1

LLM="${LLM:-google/gemma-4-26B-A4B-it}"
NSHARDS="${NSHARDS:-4}"
OUT="${OUT:-$SCRATCH/results/det_vsibench/$(basename "$LLM")_nothink}"
cd "$WORK/spatial_reasoning/finetuning"
python gemini_eval/det_vsibench_eval.py \
  --split_parquet "$SCRATCH/dataset/vsib/VSI-Bench-extended/test_debiased_scannet.parquet" \
  --det_root "$SCRATCH/dataset/vsib_det64f/scannet" \
  --llm_model "$LLM" --out_dir "$OUT" --num_shards "$NSHARDS" --shard_index "${SHARD:-${SLURM_ARRAY_TASK_ID:-0}}" --no-thinking "$@"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
