#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Continuation of the video run, resumed from one of its own checkpoints on another pair
# of cards. Both knobs are env-overridable, so moving the run again is:
#   RESUME_FROM=.../checkpoint-3000 CUDA_VISIBLE_DEVICES=0,1 bash <this script>
# Everything about the run is unchanged -- same corpus, same cosine schedule position
# (scheduler.pt is restored), same 3-epoch/12420-step horizon. Only the cards differ.
#
# WHY --resume_from_checkpoint AND NOT --init_lora_from HERE. This is the same run
# continuing, so the Adam state and the step counter SHOULD carry over; init_lora_from
# deliberately discards both and would restart the cosine schedule from step 0.
#
# --ignore_data_skip True: without it Trainer replays 2500 steps x 32 samples through the
# collator (~80k image loads off lustre) before the first optimizer step, to reproduce a
# data ORDER that does not affect the result. Epoch 1's ordering restarts instead.
#
# Moved off GPUs 0+1 at step ~2500 (loss 0.221, tok_acc 0.913, held-out f1 0.146 @2000 --
# already past the latent-only run's best). See train_lora_2gpu.sh for the design notes.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

RESUME_FROM="${RESUME_FROM:-results/idea_4a_sg_perc_scannetv2_vgjson_video_lora/checkpoint-2500}"
OUTPUT_DIR="results/idea_4a_sg_perc_scannetv2_vgjson_video_lora"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=32
BATCH_SIZE=1
NPROC_PER_NODE=2

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))

if [[ ! -f "${RESUME_FROM}/adapter_model.safetensors" ]]; then
  echo "ERROR: no adapter at ${RESUME_FROM}" >&2
  exit 1
fi


args=(
  --output_dir "${OUTPUT_DIR}"
  --resume_from_checkpoint "${RESUME_FROM}"
  --ignore_data_skip True
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct"
  --attn_implementation sdpa
  --hf_home "${HF_HOME}"
  --max_graph_tokens 4096
  --min_boxes 5
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
  --save_steps 500
  --save_total_limit 5
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 10
  --vsibench_eval_enable
  --vsibench_eval_steps 1000
  --vsibench_max_eval_samples 10
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_scannetv2_vgjson_video/train.py "${args[@]}" "$@"
