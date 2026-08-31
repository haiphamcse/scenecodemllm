#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# idea_4a + the RGB video, fed AFTER the 3D tokens (idea_3i's ordering). Started from the
# latent-only run's checkpoint-16500, so the only new thing the model must learn is how to
# use pixels alongside latents it already knows how to read.
#
# WHAT CHANGED vs scripts/idea_4a_sg_perc_scannetv2_vgjson/train_lora_2gpu.sh:
#   - _graph_user_content puts a video block after the quad placeholders, and the prompt
#     says so ("These are the RGB frames of the same scene ... Use both").
#   - The collator passes videos= to the processor, so pixels reach Qwen's ViT. VGGT still
#     gets its own NATIVE tensor: two resizes of one clip, deliberately. idea_3i shares a
#     single resize between both encoders, but that would move VGGT's input at the same
#     moment the video appears and make the two effects inseparable.
#   - --init_lora_from instead of a cold start. --init_from cannot read a LoRA checkpoint
#     (it loads *.safetensors into the BASE model and every adapter key is unexpected), and
#     --resume_from_checkpoint would restore the step counter and Adam state of a run whose
#     prompt shape no longer matches.
#
# COST: ~1536 video tokens per row (qwen_vl_utils' own budget, 768x1024 grid) on top of the
# 288 latents, so a sample goes ~1800 -> ~2500 tokens. Expect ~20-23 s/it against the
# latent-only run's 14.4. Left WITHOUT --gradient_checkpointing; turn it on if this OOMs.
#
# Vision tower stays frozen: LoRA targets the LLM only, vggt_projector in modules_to_save.
# Same choice VG-LLM makes, and it keeps every earlier number comparable.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

INIT_LORA_FROM="results/idea_4a_sg_perc_scannetv2_vgjson_lora_ep6/checkpoint-16500"
OUTPUT_DIR="results/idea_4a_sg_perc_scannetv2_vgjson_video_lora"
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

if [[ ! -f "${INIT_LORA_FROM}/adapter_model.safetensors" ]]; then
  echo "ERROR: no adapter at ${INIT_LORA_FROM}" >&2
  exit 1
fi


args=(
  --output_dir "${OUTPUT_DIR}"
  --init_lora_from "${INIT_LORA_FROM}"
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct"
  --attn_implementation sdpa
  --hf_home "${HF_HOME}"
  --max_graph_tokens 4096
  --min_boxes 5
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 3
  --logging_steps 5
  # Fresh cosine, not the constant lr the ep6 run used. That run sat flat at 1e-4 for 4000
  # steps (loss 0.237 -> 0.241, held-out f1 flat-to-down), so its schedule is not worth
  # inheriting; a cosine at least ends in a phase that consolidates.
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --save_steps 500
  --save_total_limit 5
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 10
  --vsibench_eval_enable
  --vsibench_eval_steps 1000
  --vsibench_max_eval_samples 10
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_scannetv2_vgjson_video/train.py "${args[@]}" "$@"
