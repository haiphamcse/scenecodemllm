#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# ScanNet-v2 twin of scripts/idea_4a_sg_perc_ca1m_vgjson/overfit.sh. Same model, same
# hyperparameters, same target format -- only the data changes:
#
#   - VG-LLM's own scannet_det_train_4frames.json (144k 4-frame clips) instead of the
#     CA-1M per-scene corpus. 4 frames per sample, not 64.
#   - The ScanNet download is partial, so rows whose jpgs are absent are dropped; the
#     overfit set is drawn from whatever is on disk.
#   - One clip per scene, then stratified over box count, so 20 samples == 20 rooms
#     spanning the length range (the length-collapse probe the CA-1M run needed).
#
# max_graph_tokens 4096: ScanNet tops out near 53 boxes (~68 tokens/box).
#
# Watch: `distinct` in analyze_eval.py and the predicted-count spread.
OUTPUT_DIR="results/overfit/idea_4a_sg_perc_scannetv2_vgjson_overfit"
HF_HOME="/home/ducpham/scratch/Working/cache"

# Global batch == the overfit set, so one optimizer step is one full pass over it.
GLOBAL_BATCH=20
BATCH_SIZE=1
NPROC_PER_NODE=1

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
  --min_boxes 5
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # Full-FT: whole LLM (no LoRA) at 1e-5, Perceiver at its own 1e-4.
  --full_finetune
  --learning_rate 1e-5
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 100
  --logging_steps 1
  --save_strategy no
  --ddp_find_unused_parameters False
  --lr_scheduler_type constant_with_warmup
  --warmup_ratio 0.03
  --weight_decay 0.0
  --eval_strategy no
  --report_to tensorboard
  --overfit True
  --overfit_num_samples 20
  --vsibench_eval_enable
  --vsibench_eval_steps 20
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_scannetv2_vgjson/train.py "${args[@]}" "$@"
