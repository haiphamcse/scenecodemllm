#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"


# idea_3i (VSI-590K spatial reasoning) started from idea_4a's DETECTION weights instead of
# stock Qwen3-VL. Port of scripts/legacy/idea_3i/train_tuned_2gpu.sh; every hyperparameter
# is inherited unchanged, only the initialisation and the row filter differ.
#
# WHAT THE INIT ACTUALLY IS. merge_lora.py bakes idea_4a checkpoint-10000's LoRA into the
# base weights (verified by diffing a q_proj tensor -- merge_and_unload() is a silent no-op
# if nothing matches) and writes the trained vggt_projector out SEPARATELY. The projector
# cannot ride along in the model dir: train.py does from_pretrained() and only then
# initialize_vggt(), which builds a fresh random projector, so those keys would be dropped
# as unexpected and then overwritten. --init_projector loads them after that call.
#
# ROW FILTER. The frame cache covers 856 of 5963 videos = ~138.7k of 374k rows. An uncached
# row is not skipped, it raises FileNotFoundError inside load_cached_frames, so train.py
# filters to cached rows at load time. Run idea_3i/pre_extract_videos.py to grow it.
#
# GPUs 2,3 on gpu012 (A100-80GB). Indices are allocation-relative -- override rather than
# editing this file if SLURM grants a different set.
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# multi_gpu.yaml sets `gpu_ids: all`; pin the pair explicitly so a co-tenant's
# GPU (e.g. GPU2 on parq-gpu001) is never grabbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# P2P left ON here, unlike the legacy script. That one ran on parq-gpu001, where GPU0<->GPU1
# are `SYS` (different NUMA nodes, no NVLink) and cross-socket P2P hung DDP's initial
# broadcast silently. gpu012 reports NV4 between EVERY pair, so disabling P2P would push a
# ~745MB per-step allreduce through host memory for no reason. Set NCCL_P2P_DISABLE=1 if a
# hang ever shows up at startup.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
# So a future hang says something instead of spinning silently.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MERGED_MODEL="${MERGED_MODEL:-results/idea_4a_video_noise_merged10k}"
OUTPUT_DIR="results/idea_3i_from_idea4a"
JSONL_PATH="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl"
DATA_ROOT="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=64
BATCH_SIZE=1
NPROC_PER_NODE=2

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))


# Inherited verbatim from train_tuned_2gpu.sh except where noted: global batch 64
# (1 x 2 GPUs x 32 accum), 1 epoch, cosine, LoRA r64/alpha128, 32 frames at 1 fps.
# frame_num_latents/frame_widening_factor must stay 256/2 -- they define the Perceiver
# latent array, and --init_projector's tensors are shaped for exactly that.
args=(
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MERGED_MODEL}"
  --init_projector "${MERGED_MODEL}/vggt_projector.safetensors"
  --attn_implementation sdpa
  --jsonl_path "${JSONL_PATH}"
  --data_root "${DATA_ROOT}"
  --hf_home "${HF_HOME}"
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # Kept on: 32 frames through VGGT + the ViT is what drives peak memory here, and the
  # legacy run peaked at 75.6GB on an 80GB A100 without it. Override with a trailing
  # `--gradient_checkpointing False` to reclaim the ~25-30% compute if headroom allows.
  --gradient_checkpointing True
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --learning_rate 1e-4
  # 5e-5, NOT the legacy 3e-4. That value was for a perceiver starting from random init,
  # where large early steps are what get it off the ground. Here --init_projector loads a
  # projector with ~10k steps of training behind it, and 3e-4 against 1e-4 on the LLM would
  # move it 3x faster than the weights it has to stay matched to -- i.e. spend the transfer
  # before the LLM can use it. Lower than the LLM lr on purpose.
  --perceiver_lr 5e-5
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 1
  --logging_steps 5
  --save_steps 50
  --save_total_limit 100
  --ddp_find_unused_parameters False
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
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  --dataloader_num_workers 3
  --resume_from_checkpoint latest

)


# main_process_port 0 auto-picks a free port: the default 29500 collides when
# another accelerate job is co-scheduled on the same node (e.g. idea_3j).
accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_3i_from_idea4a/train.py "${args[@]}" "$@"
