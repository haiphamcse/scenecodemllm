#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# multi_gpu.yaml sets `gpu_ids: all`; pin the pair explicitly so a co-tenant's
# GPU is never grabbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"

OUTPUT_DIR="results/idea_3i_vlm3r"
JSONL_PATH="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl"
DATA_ROOT="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=64
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))


# Full 1-epoch run over the 138,701 ScanNet++ rows of VSI-590K (856 videos, all
# frame-cached). Hyperparameters mirror the idea_3i BASELINE
# (scripts/legacy/idea_3i/train.sh) so the fusion module is the only changed
# variable vs the Perceiver and SpatialStack runs: same global batch 64, lr 1e-4,
# cosine, 1 epoch. Directly comparable to results/idea_3i_tuned (Perceiver) and
# results/idea_3i_spatialstack (layered deepstack add) at matched steps.
#
# vs scripts/legacy/idea_3i/train_tuned_2gpu.sh (the other 2-GPU script):
#   NCCL_P2P_DISABLE      set -> unset. That was a parq-gpu001 workaround where
#                         GPU0<->GPU1 are `SYS` (cross-socket, P2P hangs). On
#                         gpu013 every pair is `NV4` (NVLink), GPU1<->GPU3
#                         included, so leave P2P on.
#   gradient_checkpointing on -> off. Also a parq-gpu001 workaround for 40GB
#                         cards; gpu013 has 80GB A100s, and the 1-GPU overfit run
#                         fit without it. Saves ~25-30% compute.
#   perceiver_lr 3e-4     -> dropped. Single lr 1e-4 for the fusion block and LoRA
#                         alike (--fusion_lr exists if that ever needs splitting).
#   save_steps 50 /       -> 100 / 5. /scratch is at 349GB free against a nearly
#   save_total_limit 100     exhausted 30TiB project quota; checkpoints are 1.5GB
#                            each, so the baseline's ~43 retained checkpoints
#                            (~65GB) is not affordable. 5 rolling = ~7.5GB.
#
# vsibench_eval_steps 200: eval is rank-0 only and ends in wait_for_everyone(),
# so GPU1 idles through each ~752-row eval pass. 200 keeps that tax to ~10 evals.
#
# ddp_timeout: REQUIRED for the eval callback. That wait_for_everyone() is an
# NCCL allreduce whose default timeout is 1800s, but a full 752-row eval takes
# ~34min at ~2.7s/sample -- so rank 1's watchdog kills the run mid-eval
# ("Watchdog caught collective operation timeout ... ran for 1800091 ms").
# It killed the first attempt at step 400. 14400s = 4h of headroom.
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


# main_process_port 0 auto-picks a free port: the default 29500 collides when
# another accelerate job is co-scheduled on the same node.
accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_3i_vlm3r/train.py "${args[@]}" "$@"
