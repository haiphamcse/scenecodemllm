#!/bin/bash
#SBATCH --job-name=evl180kes
#SBATCH --account=btf@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x-%A_%a.out
#
# VSI-Bench eval of an idea_3i_130k_cached_64f checkpoint with the
# 64f fork eval.py: LIVE decoding (VGGT-Omega on the clip, 64 frames, 128 tok/frame cap),
# and the code path that scored ckpt-1200 of the cached continuation (0.5783; the 130k forks
# are copies of that cached fork, same adapter/perceiver layout). Clean frames, no masking.
# CKPT and EVAL_OUT are required (set by eval300_watch.sh or on the sbatch --export line).
# 16 shards took ~29 min each for ckpt-1200; 2 h limit for backfill.
#
# An array of 16, not one job: each task is an independent single-GPU job, so they slot
# into partly-used nodes instead of waiting for whole ones.
#
# -C v100-32g, not --partition: routes to gpu_p13 (270 nodes x 4 GPUs, mostly partly
# free) AND pins 32 GB, excluding gpu_p13's 126 v100-16g nodes. gpu_p2's nodes carry no
# size feature at all. 10 cores = a gpu_p13 node's 40 physical over 4 GPUs.
#
# fp32: V100 is sm_70, so no bf16, and fp16 returned "!!!!!!" (NaN logits) in job 1892043.
# ~321 rows per shard at 16 shards. The eval writes per-shard output.
set -euo pipefail

cd "${WORK}/spatial_reasoning/finetuning"
source "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate vsibench_eval_full
set -a; source "${WORK}/spatial_reasoning/finetuning/site.env"; set +a

export HF_HOME="${SR_HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUTF8=1
export LANG=${LANG:-en_US.UTF-8}
export LC_ALL=${LC_ALL:-en_US.UTF-8}
export FORCE_QWENVL_VIDEO_READER=decord
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT="${CKPT:?checkpoint dir}"
EVAL_OUT="${EVAL_OUT:?eval output dir}"
SPLIT="${SPLIT:-full}"
NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-8}}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== job ${SLURM_JOB_ID} | shard ${SHARD_INDEX}/${NUM_SHARDS} | ckpt ${CKPT} | res 256, 64 frames (live VGGT) ==="

args=(
  --adapter_path "${CKPT}"
  --split "${SPLIT}"
  --output_dir "${EVAL_OUT}"
  --hf_home "${SR_HF_HOME}"
  --num_shards "${NUM_SHARDS}"
  --shard_index "${SHARD_INDEX}"
  --dtype "${DTYPE:-fp32}"
  # Must match jz_train_180k_es.sh; live VGGT, no cache.
  --frame_num_latents 512
  --camera_num_latents 32
  --frame_widening_factor 2
  --vggt_image_resolution 256
  --video_fps 1.0
  --video_max_frames 64
)
[[ -n "${MAX_SAMPLES:-}" ]] && args+=(--max_samples "${MAX_SAMPLES}")

python idea_3i_180k_es_64f_lat512/eval.py "${args[@]}"
