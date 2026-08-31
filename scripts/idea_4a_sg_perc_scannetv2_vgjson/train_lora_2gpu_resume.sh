#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Epochs 4-6 on the ScanNet-v2 json, continuing train_lora_2gpu.sh's checkpoint-12420 at a
# CONSTANT lr. Everything not listed below is inherited from that script unchanged: LoRA
# (see its header for why not --full_finetune), effective batch 32, 256 latents, widening 2,
# same corpus, same held-out eval.
#
# WHY --resume_from_checkpoint AND NOT --init_from. init_from globs *.safetensors and does
# model.load_state_dict(..., strict=False) on the BASE model, then raises on any unexpected
# key (train.py:415-435). A LoRA checkpoint's keys are all base_model.model.*.lora_A/lora_B,
# so every one is unexpected -- it fails immediately. That path was built for the full-FT
# overfit runs, which save a plain state dict. resume_from_checkpoint goes through peft and
# restores the optimizer too, which is the better behaviour here anyway: the overfit run
# showed a fresh optimizer costs ~30 steps (f1 0.332 -> 0.045 -> recovered past the peak).
#
# WHY --num_train_epochs 6 RATHER THAN 3. Trainer recomputes max_steps from this arg and
# compares it against the checkpoint's global_step. checkpoint-12420 holds global_step
# 12420 / max_steps 12420 / should_training_stop True, so 3 would resume and halt at once.
# 6 gives max_steps 24840 and trains 12420 fresh steps (4140 per epoch).
#
# WHY --ignore_data_skip True. Without it Trainer fast-forwards the dataloader to the
# resume point -- 12420 steps x 32 samples pushed through the collator, i.e. ~400k image
# loads off lustre before a single optimizer step. The cost buys byte-identical data order,
# which is worthless here because the schedule changed anyway.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RESUME_FROM="results/idea_4a_sg_perc_scannetv2_vgjson_lora/checkpoint-12420"
# A NEW dir, not the source run's. trainer.save_model() at the end writes the final adapter
# to output_dir; pointing it at the old run would overwrite the 3-epoch adapter with the
# 6-epoch one. Checkpoints, tensorboard runs/ and scannet_eval.txt stay separated too.
OUTPUT_DIR="results/idea_4a_sg_perc_scannetv2_vgjson_lora_ep6"
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
  # Constant at the cosine run's PEAK, so the model restarts at the largest lr it ever saw
  # rather than the 1.7e-12 it ended on. Loss is already at 0.223; expect it to jump before
  # resettling, and read the first few hundred steps as a restart, not a regression.
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 6
  --logging_steps 5
  # get_constant_schedule ignores warmup entirely, so warmup_ratio is set to 0 to keep the
  # log honest rather than left at 0.03 where it would read as if 372 warmup steps existed.
  # Note the interaction with the restored state: optimizer.pt carries lr 1.7e-12 in its
  # param groups, and the scheduler only overwrites that on its first step() -- so optimizer
  # step 1 lands at ~0 and every step after it at 1e-4. One no-op step, nothing to fix.
  --lr_scheduler_type constant
  --warmup_ratio 0.0
  --weight_decay 0.01
  # save_steps is NOT settable here: DefaultFlowCallback reads state.save_steps out of the
  # resumed trainer_state.json (500) and transformers only warns that the args disagree.
  # save_total_limit is args-driven and does apply, so that is the disk knob: 5 x ~2.5 GB
  # instead of the 12420/500 = 24 checkpoints (~60 GB) this run would otherwise leave. To
  # actually change the cadence, edit trainer_state.json in ${RESUME_FROM}.
  --save_steps 500
  --save_total_limit 5
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 10
  --vsibench_eval_enable
  # Unchanged from the source run so the f1 trajectory extends the existing one rather than
  # starting a new series. n=10 cannot separate 0.106 from 0.131 -- score the final adapter
  # on a larger held-out set separately rather than trusting a single point here.
  --vsibench_eval_steps 1000
  --vsibench_max_eval_samples 10
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_scannetv2_vgjson/train.py "${args[@]}" "$@"
