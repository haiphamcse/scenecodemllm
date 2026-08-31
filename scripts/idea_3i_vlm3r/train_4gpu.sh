#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# multi_gpu.yaml sets `gpu_ids: all`; pin explicitly anyway so a co-tenant's GPU
# is never grabbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

OUTPUT_DIR="results/idea_3i_vlm3r"
JSONL_PATH="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl"
DATA_ROOT="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
HF_HOME="/home/ducpham/scratch/Working/cache"

# Resume point. "latest" picks the newest checkpoint-* in OUTPUT_DIR; pass a
# path to pin one, or "" for a fresh run.
RESUME="${RESUME:-latest}"

GLOBAL_BATCH=64
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))


# 4-GPU sibling of train_2gpu.sh. Same hyperparameters, same global batch 64
# (4 x 1 x 16 instead of 2 x 1 x 32), so checkpoints stay step-comparable with
# results/idea_3i_tuned (Perceiver) and results/idea_3i_spatialstack.
#
# TOPOLOGY WARNING for this H100 node: the four cards are two NVLink islands,
# not one fabric.
#     GPU0 <-NV12-> GPU1   (NUMA 0, CPUs 0-31,64-95)
#     GPU2 <-NV12-> GPU3   (NUMA 1, CPUs 32-63,96-127)
#     0/1 <-> 2/3 = SYS    (cross-socket)
# Every 4-way allreduce therefore crosses the socket boundary. P2P is left ON
# because the SYS hop still works over PCIe here and disabling it would also
# kill NVLink inside each pair. If the run hangs at startup or the first
# allreduce, relaunch with NCCL_P2P_DISABLE=1, or fall back to one island:
#     CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 bash scripts/idea_3i_vlm3r/train_4gpu.sh
# See [[slurm-gpu-allocation-gotchas]] for the cross-socket hang this guards.
#
# ddp_timeout: REQUIRED. The VSI-Bench callback is rank-0 only and ends in
# wait_for_everyone(), an NCCL allreduce with a 1800s default. A 752-row eval
# takes ~19min here, but the margin is not worth the risk. 14400s = 4h.
args=(
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct"
  --attn_implementation sdpa
  --jsonl_path "${JSONL_PATH}"
  --data_root "${DATA_ROOT}"
  --hf_home "${HF_HOME}"
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --num_train_epochs 1
  --logging_steps 5
  --save_steps 100
  --save_total_limit 5
  --ddp_find_unused_parameters False
  --ddp_timeout 14400
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --eval_strategy no
  --report_to tensorboard
  --overfit_on_eval False
  --video_fps 1.0
  --video_max_frames 32
  --vsibench_eval_enable
  --vsibench_eval_steps 200
  --geometry_encoder_layer 23
  --fusion_num_heads 16
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  --dataloader_num_workers 3
)

if [[ -n "${RESUME}" ]]; then
  args+=(--resume_from_checkpoint "${RESUME}")
fi

# main_process_port 0 auto-picks a free port: the default 29500 collides when
# another accelerate job is co-scheduled on the same node.
accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_3i_vlm3r/train.py "${args[@]}" "$@"
