#!/bin/bash
#SBATCH --job-name=vggtexp64fms
#SBATCH --account=btf@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --output=%x-%A_%a.out
#
# Export the frozen VGGT-Omega features once, as 16 independent single-GPU jobs.
#
# MULTISCALE variant (idea_3i_180k_cached_64f_lat512_multiscale): the 4,956 videos of the
# 180k mix ONLY (no VSI-Bench: eval is live), resolution 256, fps 1.0 capped at 64 frames,
# storing the patch tokens of the 4 cached aggregator layers (4/11/17/23) + the deepest
# layer's camera tokens per clip, into vggt_cache_256_64f_ms (~4x the single-scale file).
#
#   sbatch --output=$WORK/logs/%x-%A_%a.out \
#     scripts/idea_3i_180k_cached_64f_lat512_multiscale/export_vggt_64f_ms_v100.slurm
#   Smoke: sbatch --array=0 --qos=qos_gpu-dev --time=00:30:00 --export=ALL,NUM_SHARDS=1,MAX_CLIPS=3 ...
#
# 64-FRAME variant (idea_3i_130k_joint_cached_64f): the 130k mix's clips + VSI-Bench at
# resolution 256, fps 1.0 capped at 64 frames. Global attention over 2x the frames, so
# 10 h wall instead of 2; resumable like the original.
#
# RESOLUTION 256 variant: the same clips as vggt_cache_512, at the resolution the
# idea_3i_590k_joint run was trained at, so a cached continuation of that run keeps its
# recipe. ~266 patches/frame instead of 1036 -> ~4x cheaper (2 h wall) and ~0.5 TB.
#
#   sbatch --output=$WORK/logs/%x-%A_%a.out \
#     scripts/idea_3i_130k_joint_cached_64f/export_vggt_64f_v100.slurm
#
# An array, not one multi-node job: each task is a separate single-GPU job that the
# scheduler places on its own, so they slot into partly-used nodes instead of waiting
# for whole ones. The work is embarrassingly parallel -- no communication, one output
# file per clip -- so there is nothing a shared allocation would have bought.
# A 2-node/16-task version of this queued behind an estimated 11 h wait, because taking
# all 8 GPUs of a gpu_p2 node makes IDRIS set exclusive mode.
#
# -C v100-32g, not --partition: it routes to gpu_p13 (270 nodes x 4 GPUs) AND pins 32 GB.
# gpu_p13 also holds 126 v100-16g nodes, which are too small here; gpu_p2's nodes carry
# no v100-32g feature at all, so this constraint picks exactly the right set.
# 10 cores: a gpu_p13 node is 80 logical / 40 physical over 4 GPUs.
#
# qos_gpu-t3, not -dev: dev caps MaxSubmitJobsPU at 10, so 16 tasks do not fit.
#
# Resumable: export_vggt_features.py skips entries already fresh, so a task that hits
# the wall clock can be resubmitted and picks up its remainder. Shards are disjoint
# slices of a sorted clip list, so tasks never contend for the same file.
#
# --dtype fp32: V100 is sm_70, so bf16 (the script's default) does not exist here.
set -euo pipefail

cd "${WORK}/spatial_reasoning/finetuning"
source "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate vsibench_eval_full

set -a; source "${WORK}/spatial_reasoning/finetuning/site.env"; set +a
export HF_HOME="${SR_HF_HOME}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUTF8=1 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export FORCE_QWENVL_VIDEO_READER=decord
export DECORD_EOF_RETRY_MAX=40960

CACHE_ROOT="${CACHE_ROOT:-${SCRATCH}/dataset/vggt_cache_256_64f_ms}"
STORE_DTYPE="${STORE_DTYPE:-fp32}"
mkdir -p "${CACHE_ROOT}"

NUM_SHARDS="${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-16}}"
echo "=== shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} | job ${SLURM_JOB_ID} | $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python idea_3i_180k_cached_64f_lat512_multiscale/export_vggt_features.py \
  --cache_root "${CACHE_ROOT}" \
  --mix_jsonl "${SR_VGLLM_ROOT}/joint180k_vlm3r/vqa_train.jsonl" \
  --store_dtype "${STORE_DTYPE}" \
  ${MAX_CLIPS:+--max_clips "${MAX_CLIPS}"} \
  --data_root "${SR_DATA_ROOT}" \
  --hf_home "${SR_HF_HOME}" \
  --vggt_image_resolution 256 \
  --video_max_frames 64 \
  --dtype fp32 \
  --num_shards "${NUM_SHARDS}" --shard_index "${SLURM_ARRAY_TASK_ID}"
