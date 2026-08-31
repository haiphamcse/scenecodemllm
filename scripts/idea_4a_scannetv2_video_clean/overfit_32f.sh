#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 32-frame overfit probe for idea_4a_scannetv2_video_clean. Same shape as
# scripts/idea_4a_sg_perc_scannetv2_vgjson/overfit.sh, with three differences:
#
#   - 32 frames per row (scannet_det_train_32frames_bi1.json) instead of 4. Measured
#     ~14.5k input_ids per sample, p99 16.7k, so --gradient_checkpointing is on; without
#     it an 80GB A100 is not a safe bet at this length.
#   - LoRA, not --full_finetune. The clean fork dropped the full-FT and perceiver-only
#     paths, so LoRA (r=64, vggt_projector in modules_to_save) is the only mode left.
#     A LoRA overfit climbs slower than full-FT did; judge it on the f1 trend, not on
#     hitting 1.0 by a fixed step.
#   - Nothing truncates any more: the collator's truncate_graph_text and train.py's
#     max_target_tokens row filter are both gone. TRL's own max_length default (1024)
#     never applies because train.py passes its own data_collator, so SFTTrainer never
#     builds the collator that would enforce it.
#
# max_graph_tokens 4096 is now ONLY the eval generation budget. Targets measured at ~800
# tokens mean / ~1900 p99, so no eval generation should hit the cap.
#
# A healthy run drives train loss toward 0 and precision/recall/f1 toward 1.0 on the same
# 10 clips it trains on. Watch scannet_eval.txt.
OUTPUT_DIR="results/overfit/idea_4a_scannetv2_video_clean_32f"
HF_HOME="/home/ducpham/scratch/Working/cache"

# Global batch == the overfit set, so one optimizer step is one full pass over it.
GLOBAL_BATCH=10
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
  --vggt_variant 512
  --max_graph_tokens 4096
  --min_boxes 5
  --bf16 True
  --gradient_checkpointing True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  # 10 rows at grad-accum 10 == 1 optimizer step per epoch, so this is 200 steps.
  --num_train_epochs 200
  --logging_steps 1
  --save_strategy no
  --lr_scheduler_type constant_with_warmup
  --warmup_ratio 0.03
  --weight_decay 0.0
  --ddp_find_unused_parameters False
  --eval_strategy no
  --report_to tensorboard
  --overfit True
  --overfit_num_samples 10
  --vsibench_eval_enable
  --vsibench_eval_steps 50
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_scannetv2_video_clean/train.py "${args[@]}" "$@"
