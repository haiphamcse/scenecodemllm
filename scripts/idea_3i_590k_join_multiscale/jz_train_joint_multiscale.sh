#!/bin/bash

set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Joint VQA + 3DOD: VSI-590K questions and ScanNet 32-frame detection in one mix.
#
# MULTI-SCALE variant of idea_3i_590k_joint. The frozen VGGT aggregator already caches four
# depths (blocks 4/11/17/23); the single-scale parent kept only the last. Here ONE shared
# frame_encoder compresses each depth to --frame_num_latents latents and the results are
# concatenated, so frame placeholders = frame_num_latents * frame_num_scales = 256*4 = 1024.
# Camera/register tokens stay single-scale.
#
# The corpus is built OFFLINE first -- this script trains on whatever build_mix.py wrote,
# so the task ratio is a property of the manifest, not of this file:
#
#   python idea_3i_590k_join_multiscale/build_mix.py --vqa_rows 300000 --det_rows 100000
#
# Both tasks share ONE frame pipeline (VGGT balanced-512, [0,1] in the collator), which is
# idea_4a's convention rather than idea_3i_590k's. A single frozen VGGT and a single shared
# Perceiver see both streams, so they cannot be fed two different resolutions. Consequence:
# the VQA numbers from this run are NOT comparable to earlier idea_3i_590k runs.
#
# To start from the idea_4a detection weights instead of stock Qwen, set both halves --
# the merged LLM and its projector (see scripts/idea_3i_590k/jz_train_590k_from_idea4a.sh):
#   MERGED_MODEL=results/idea_4a_clean_merged850 \
#   INIT_PROJECTOR=results/idea_4a_clean_merged850/vggt_projector.safetensors \
#   bash scripts/idea_3i_590k_join_multiscale/jz_train_joint.sh

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Job 1666171 died at step 77 on scannetppv2/0eba3981c9.mp4 with DECORDError:
# "Unable to handle EOF ... DECORD_EOF_RETRY_MAX=10240". The file is fine (decodes
# in 7.8 s on an idle node) -- 48 decode workers made Lustre slow enough to burn the
# retry budget. More budget, and fewer workers below.
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"

OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/idea_3i_590k_join_multiscale"
MIX_JSONL="${SR_MIX:-/home/ducpham/scratch/Working/dataset/vgllm_data/joint/joint_train.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"
MERGED_MODEL="${MERGED_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
INIT_PROJECTOR="${INIT_PROJECTOR:-}"

if [[ ! -f "${MIX_JSONL}" ]]; then
  echo "ERROR: ${MIX_JSONL} missing. Run idea_3i_590k_joint/build_mix.py first." >&2
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
  --attn_implementation sdpa
  --jsonl_path "${MIX_JSONL}"
  --data_root "${DATA_ROOT}"
  --hf_home "${HF_HOME}"
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
  --frame_num_scales 4
  --camera_num_latents 32
  --frame_widening_factor 2
  # Must match the VGGT variant below; the collator resizes to this and the encoder
  # was trained at it.
  --vggt_image_resolution 256
  # det rows only. The 3DOD-only runs that produced ckpt-850 left this at 0.0;
  # 0.005 is what the older idea_4a noise runs used (see train_lora_2gpu_noise.sh).
  --box_noise 0.005
  # Probability of jittering a det clip; one draw per clip, DUSt3R strength. 0.0 disables.
  --color_jitter 0.5
  --num_train_epochs 1
  --logging_steps 10
  # 2 GPUs makes a step ~6-7 min, so 100 steps is >10 h between saves. 50 halves
  # what a wall-time TIMEOUT throws away.
  --save_steps 50
  --save_total_limit 100
  # get_last_checkpoint guard in train.py makes this a no-op on the first window.
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
  # 12 (48 across 4 ranks) starved Lustre and tripped decord's EOF retry limit.
  --dataloader_num_workers 8
)

if [[ -n "${INIT_PROJECTOR}" ]]; then
  args+=(--init_projector "${INIT_PROJECTOR}")
fi

accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  idea_3i_590k_join_multiscale/train.py "${args[@]}" "$@"
