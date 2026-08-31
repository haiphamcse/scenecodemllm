#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# COLD run of the video variant: base Qwen3-VL-2B, fresh LoRA, fresh Perceiver. Every
# earlier video run descended from the latent-only checkpoint via --init_lora_from, so
# none of them isolate what the architecture learns on its own. This one does.
#
# NOTHING is inherited from a checkpoint -- no --init_lora_from, no --init_from. The
# Perceiver starts random, which is the point.
#
# Data is the LATEST config (matches ckpt-9000/10000): --min_boxes 0 keeps all 144164 rows
# including 71 with empty targets, and --box_noise 0.005 perturbs all 9 box values per
# epoch. So vs ckpt-9000 this run changes init AND batch size, not one variable.
#
# BATCH 128, per_device 1, grad_accum = 128 / (1 x NPROC_PER_NODE). Measured 121 s/it on a
# single H100; two GPUs roughly halves that. NPROC_PER_NODE is env-overridable:
#   NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=<uuid0>,<uuid1> bash <this script>
# PIN BY UUID, not index. Index 0 is allocation-relative: launched from a shell inside SLURM
# job 5193367 it resolved to PHYSICAL GPU1, not GPU0. On gpu017 only {0,1} and {2,3} are
# NVLinked (NV12); 0<->2/3 is SYS, the cross-socket path that hangs NCCL. 3381 steps at
# ~60 s/it is roughly 56 h. The
# allocation has less than that, hence --save_steps 100 (~3.2 h apart) and
# --resume_from_checkpoint latest, which idea_4a's train.py resolves via
# get_last_checkpoint -- a path when one exists, None on the first run. Relaunch this same
# script on the next allocation and it picks up where it stopped.
#
# lr stays 1e-4 despite the 4x batch: every prior number here used it, so batch size and
# init are the only changes. Expect a gentler effective schedule per sample.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="results/idea_4a_video_scratch_b128_lora"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=128
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))



args=(
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct"
  --attn_implementation sdpa
  --hf_home "${HF_HOME}"
  --max_graph_tokens 4096
  --min_boxes 0
  --box_noise 0.005
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 3
  --logging_steps 5
  # Fresh cosine, not the constant lr the ep6 run used. That run sat flat at 1e-4 for 4000
  # steps (loss 0.237 -> 0.241, held-out f1 flat-to-down), so its schedule is not worth
  # inheriting; a cosine at least ends in a phase that consolidates.
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --save_steps 100
  --save_total_limit 5
  # --resume_from_checkpoint latest
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 10
  --vsibench_eval_enable
  # 200, not 1000. At ~114 s per optimizer step, step 1000 is ~31 h in -- most of an
  # allocation before the first held-out number. 200 puts it at ~6 h.
  --vsibench_eval_steps 200
  --vsibench_max_eval_samples 10
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_scannetv2_vgjson_video/train.py "${args[@]}" "$@"
