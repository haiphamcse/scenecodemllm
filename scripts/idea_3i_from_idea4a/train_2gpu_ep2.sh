#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"


# SECOND EPOCH of results/idea_3i_from_idea4a_scratch1600, on gpu012 (2 GPUs).
#
# Epoch 1 finished at checkpoint-2168 (loss ~0.394, tok_acc 0.803) with its cosine schedule
# fully spent -- the last logged lr was 8.9e-10. This is a real HF Trainer resume, not a
# re-init: --resume_from_checkpoint restores the LoRA weights, optimizer moments and RNG,
# and --num_train_epochs 2 extends max_steps to 4336 so training continues 2168 -> 4336 in
# the SAME output dir. No merge, no second adapter.
#
# --ignore_data_skip True matters. Without it HF replays the dataloader from step 0 to reach
# the resume point, which for a whole epoch means grinding through 2168 steps of video
# decoding before any compute happens. Skipping it means epoch 2 sees a fresh shuffle, which
# is what a second epoch should see anyway.
#
# LR NOTE: the scheduler is rebuilt for 4336 steps and then fast-forwarded to step 2168, so
# the effective starting lr is the cosine value at the midpoint of a 3e-5 curve (~1.5e-5),
# NOT 3e-5. It decays to 0 at 4336. Verify against the first logged learning_rate.
#
# --- original header, for the lineage this run came from ---
# 4-GPU idea_3i run started from a DIFFERENT idea_4a lineage than train_4gpu.sh.
#
# train_4gpu.sh   -> idea_4a_video_noise_lora/checkpoint-10000 (perceiver warm-started from
#                    the latent-only run via --init_lora_from, then 10k more steps)
# train_4gpu_scratch1400.sh
#                 -> the same lineage at checkpoint-1400 (loss 0.341, tok_acc 0.867)
# this script     -> idea_4a_video_scratch_b128_lora_done/checkpoint-1600, 200 steps further
#                    along that run: loss 0.331, tok_acc 0.870, epoch 1.42 of 3381 steps,
#                    ScanNet detection f1 0.245 (checkpoint-400 was 0.208).
#
# Point of this run is 1400 vs 1600 as the merge source, so everything except MERGED_MODEL,
# OUTPUT_DIR and --perceiver_lr is held identical to train_4gpu_scratch1400.sh.
#
# Merge is the same two-part flow -- LoRA into the base weights, projector alongside as
# vggt_projector.safetensors:
#   python idea_3i_from_idea4a/merge_lora.py \
#       --adapter results/idea_4a_video_scratch_b128_lora_done/checkpoint-1600 \
#       --out     results/idea_4a_video_scratch_merged1600
#
# Separate OUTPUT_DIR on purpose. Sharing one with train_4gpu.sh would let
# --resume_from_checkpoint latest pick up a checkpoint trained from the other init.
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
# --resume_from_checkpoint latest resolves through get_last_checkpoint, so the first launch
# starts fresh (empty dir) and any relaunch picks up the newest checkpoint automatically.
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# multi_gpu.yaml sets `gpu_ids: all`; pin the pair explicitly so a co-tenant's
# GPU (e.g. GPU2 on parq-gpu001) is never grabbed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

# P2P ON. gpu012 is not gpu017: `nvidia-smi topo -m` there gives NV4 between EVERY pair
# (A100-SXM4-80GB, full mesh), so {2,3} has no SYS hop to work around.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
# So a future hang says something instead of spinning silently.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MERGED_MODEL="${MERGED_MODEL:-results/idea_4a_video_scratch_merged1600}"
OUTPUT_DIR="results/idea_3i_from_idea4a_scratch1600"
JSONL_PATH="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl"
DATA_ROOT="/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=64
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

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
  # 3e-5, down from epoch 1's 1e-4: the adapter is already converged (loss flat ~0.38 for
  # the last 800 steps), so this epoch refines rather than re-explores.
  --learning_rate 3e-5
  # 1e-5, held at the same ABSOLUTE value as epoch 1 (GAP-MLLM's --mm_projector_lr). Note
  # the ratio changed: it was 0.1x the LoRA lr when that was 1e-4, and is 0.33x now that the
  # LoRA lr is 3e-5. So the projector moves relatively more this epoch than last.
  --perceiver_lr 1e-5
  --frame_num_latents 256
  --frame_widening_factor 2
  # 2, not 1. Read together with --resume_from_checkpoint below: max_steps becomes 4336
  # and the resumed run covers the second half.
  --num_train_epochs 2
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
  # 400, halved from epoch 1. Two GPUs instead of four makes each 752-sample eval cost
  # more wall clock, and epoch 1's curve swung +-0.03 between adjacent 200-step evals,
  # so the extra resolution was mostly noise.
  --vsibench_eval_steps 400
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  --dataloader_num_workers 3
  # Explicit path, not `latest`: train.py passes a non-'latest' value straight through to
  # trainer.train(). Pinning it means a relaunch cannot silently pick up an epoch-2
  # checkpoint written by a run whose args differed.
  #
  # This is a PATCHED COPY of checkpoint-2168, not the original, and it has to be.
  # HF restores base_lrs from scheduler.pt and initial_lr from optimizer.pt, which
  # silently overrides --learning_rate on resume: the first attempt logged 5.241e-5
  # (= epoch 1's 1e-4 peak x the cosine factor) instead of anything derived from 3e-5.
  # The copy has base_lrs[0]/initial_lr[0] rewritten 1e-4 -> 3e-5 with last_epoch left
  # at 2168, so the curve is the second half of a 3e-5 cosine: 1.573e-5 -> 0. Adam
  # moments are untouched. It lives OUTSIDE OUTPUT_DIR so save_total_limit's
  # checkpoint scan cannot see or rotate it.
  --resume_from_checkpoint results/idea_3i_from_idea4a_scratch1600_ep2_init/checkpoint-2168
  --ignore_data_skip True

)


# main_process_port 0 auto-picks a free port: the default 29500 collides when
# another accelerate job is co-scheduled on the same node (e.g. idea_3j).
accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_3i_from_idea4a/train.py "${args[@]}" "$@"
