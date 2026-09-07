#!/bin/bash

set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# idea_3i_590k on all 6 VSI-590K video sources, live decode, no cache.
# 374148 video rows / batch 64 = 5846 steps/epoch. One 20h window covers a fraction

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/idea_3i_590k_256"
JSONL_PATH="${SR_JSONL:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"

GLOBAL_BATCH=256
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
  --jsonl_path "${JSONL_PATH}"
  --data_root "${DATA_ROOT}"
  --hf_home "${HF_HOME}"
  # Omitted on purpose: train.py falls back to VIDEO_SOURCES, all 6 video sources.
  --overfit_on_eval False
  --bf16 True
  # OFF deliberately. Measured on joint (job 1846554 vs 1819661, both from ckpt-500):
  # gc on = 41.7 GB peak / 50.2 reserved / 228 s/it; gc off = 67.1 / 75.3 / 190 s/it.
  # 17% faster for +25 GB, leaving ~9 GiB margin -- the same margin the 2026-09-04
  # OOM cascade died on. Flip back to True if any run OOMs.
  --gradient_checkpointing False
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 1
  --logging_steps 10
  # 2 GPUs makes a step ~6-7 min, so 100 steps is >10 h between saves. 50 halves
  # what a wall-time TIMEOUT throws away.
  --save_steps 50
  --save_total_limit 100
  # Slurm requeues this job on an OOM kill; without this every requeue restarts
  # at step 0. get_last_checkpoint makes it a no-op when there is no checkpoint.
  --resume_from_checkpoint latest
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --video_fps 1.0
  --video_max_frames 32
  --vsibench_eval_enable
  # 200 for every experiment: puts all four on one eval grid so scores compare at
  # matched steps. An eval is 25 min (VQA) or 44 min (joint/multiscale -- their
  # collator preprocesses each sample twice, once per branch), so 100 was costing
  # joint ~12% of a window against cold's 8%.
  --vsibench_eval_steps 200
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  # 4 (idea_3i's cached default) starves a live-decode run; a gpu_p6 node has 24 cores/GPU.
  # 12 OOM-killed a worker 4 h in (job 1632322): 48 workers x prefetched 32-frame
  # clips overran the node's 480 G. 8 keeps the decode fed with headroom.
  --dataloader_num_workers 8
)

# main_process_port 0 auto-picks a free port; the default 29500 collides with a co-scheduled job.
accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  idea_3i_590k/train.py "${args[@]}" "$@"
