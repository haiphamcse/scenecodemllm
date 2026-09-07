#!/bin/bash

set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# idea_3i_590k on all 6 VSI-590K video sources, initialised from idea_4a detection weights
# instead of stock Qwen3-VL. Otherwise identical to jz_train_590k.sh.
#
# The init comes in two halves, because from_pretrained() runs before initialize_vggt():
#   MERGED_MODEL          -> the LLM with idea_4a's LoRA merged in (merge_lora.py --out)
#   vggt_projector.safetensors -> the trained Perceiver, loaded AFTER initialize_vggt
#                                 replaces it with random weights (--init_projector)
# Pointing --model_name_or_path at the merged dir alone would silently drop the projector.
#
# Source: results_jeanzay/idea_4a_scannetv2_video_clean_32f_full/checkpoint-850, merged by
#   python idea_3i_from_idea4a/merge_lora.py \
#     --adapter results_jeanzay/idea_4a_scannetv2_video_clean_32f_full/checkpoint-850 \
#     --out results/idea_4a_clean_merged850
#
# frame_num_latents/frame_widening_factor must stay 256/2 and camera_num_latents 32 --
# they are the shape of the projector being loaded, not free hyperparameters here.
#
# NOTE ON FRAME PREPROCESSING: idea_4a fed VGGT frames resized by VGGT's own "balanced"
# mode at 512 and normalised to [0,1] in the collator. idea_3i_590k feeds fetch_video's
# output at Qwen's resolution and divides by 255 in the model. The Perceiver is
# length-agnostic so nothing breaks, but it sees a different frame size than it trained on.

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/idea_3i_590k_from_idea4a850"
MERGED_MODEL="${MERGED_MODEL:-${SR_OUTPUT_ROOT:-results}/idea_4a_clean_merged850}"
JSONL_PATH="${SR_JSONL:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"

if [[ ! -f "${MERGED_MODEL}/vggt_projector.safetensors" ]]; then
  echo "ERROR: ${MERGED_MODEL}/vggt_projector.safetensors missing. Run merge_lora.py first." >&2
  exit 1
fi

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
  --model_name_or_path "${MERGED_MODEL}"
  --init_projector "${MERGED_MODEL}/vggt_projector.safetensors"
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
  --camera_num_latents 32
  --frame_widening_factor 2
  --num_train_epochs 1
  --logging_steps 10
  # 2 GPUs makes a step ~6-7 min, so 100 steps is >10 h between saves. 50 halves
  # what a wall-time TIMEOUT throws away.
  --save_steps 50
  --save_total_limit 100
  # Resumes from the newest checkpoint; get_last_checkpoint makes it a no-op if none.
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
  --dataloader_num_workers 8
)

# main_process_port 0 auto-picks a free port; the default 29500 collides with a co-scheduled job.
accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  idea_3i_590k/train.py "${args[@]}" "$@"
