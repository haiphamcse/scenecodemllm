#!/bin/bash

set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Full (non-overfit) 32-frame ScanNet detection run. Derived from train_32f.sh, which
# was an unrunnable fragment carrying overfit leftovers (200 epochs, save_strategy no).
# --overfit False is required: the dataclass default is True.
# 118184 usable rows (min_boxes 5) / batch 128 = 923 steps/epoch; 3 epochs = 2769.
# Val json is 4-frame while training is 32-frame -- a known mismatch in this code.

# Not exported: FORCE_QWENVL_VIDEO_READER, CUDA_VISIBLE_DEVICES, NCCL_* — submit.sh owns
# those. PYTORCH_CUDA_ALLOC_CONF it does not, so it stays here.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/idea_4a_scannetv2_video_clean_32f_full"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"
DATA="${SR_SCANNET_ROOT:-/home/ducpham/scratch/Working/dataset}"

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
  # Explicit so the run does not depend on train.py's defaults.
  --train_json "${DATA}/vgllm_data/train/scannet_det_train_32frames_bi1.json"
  --val_json   "${DATA}/vgllm_data/evaluation/threedod/scannet/scannet_det_val_4frames.json"
  --image_root "${DATA}"
  # Train on every usable row instead of the 20-scene sanity check.
  --overfit False
  --val_size 50
  --vggt_variant 512
  --max_graph_tokens 8192
  --min_boxes 5
  --bf16 True
  # 32 frames at ~14.5k tokens/sample; this is what keeps it inside 80 GB.
  --gradient_checkpointing True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 3
  --logging_steps 5
  # Was save_strategy no, which makes chaining impossible: each window would restart
  # from step 0. At ~50 steps between saves a lost window costs a couple of hours.
  --save_steps 50
  --save_total_limit 10
  --resume_from_checkpoint latest
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.0
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --vsibench_eval_enable
  --vsibench_eval_steps 200
  --dataloader_num_workers 8
  --lora_r 256
  --lora_alpha 512
  --lora_dropout 0.05
)

accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  idea_4a_scannetv2_video_clean/train.py "${args[@]}" "$@"
