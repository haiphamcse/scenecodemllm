#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"


# 4-GPU continuation of results/idea_3i_from_idea4a, resumed from its own checkpoint-250.
# Same experiment as train_2gpu.sh -- idea_3i started from idea_4a's merged detection
# weights -- only the world size changes. See that script for the init/merge rationale.
#
# WORLD SIZE 2 -> 4. GLOBAL_BATCH stays 64, so grad_accum drops 32 -> 16 and the step count,
# LR schedule and horizon are unchanged. Per-rank data sharding DOES change, so steps after
# 250 are not bitwise comparable to the first 250; the loss curve is still continuous.
#
# NCCL_P2P_DISABLE=1, unlike the 2-GPU script. On gpu017 `nvidia-smi topo -m` gives NV12
# within {0,1} and within {2,3}, but SYS between those pairs -- different NUMA nodes, no
# NVLink. A 4-way group spans that boundary, and cross-socket P2P is what hung DDP's initial
# broadcast silently on parq-gpu001. Staging through host memory costs bandwidth inside each
# NVLinked pair but is the safe choice for a 4-way ring; drop it to 0 to try full P2P.
#
# CO-TENANCY: GPU2 (and sometimes GPU0) carries another user's process. DDP runs at the
# slowest rank, so expect the step time to track whatever they are doing.
#
# --resume_from_checkpoint latest resolves through get_last_checkpoint, so this picks up
# checkpoint-250 automatically and starts fresh if the dir is ever cleared.
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# multi_gpu.yaml sets `gpu_ids: all`; pin the pair explicitly so a co-tenant's
# GPU (e.g. GPU2 on parq-gpu001) is never grabbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# See the header: P2P is OFF because a 4-way group on gpu017 spans the SYS boundary between
# the two NVLinked pairs. (The 2-GPU script keeps it ON -- that run stayed inside {2,3}.)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
# So a future hang says something instead of spinning silently.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MERGED_MODEL="${MERGED_MODEL:-results/idea_4a_video_noise_merged10k}"
OUTPUT_DIR="results/idea_3i_from_idea4a"
JSONL_PATH="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl"
DATA_ROOT="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=64
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

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
