#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Video run continued from checkpoint-6000 with TWO data changes, on GPUs 2+3.
#
# 1. --box_noise 0.005: every one of the 9 box values is scaled by 1 +/- U(0.005),
#    resampled in the collator on every __getitem__, so a clip gets a different
#    perturbation each epoch rather than one permanently-wrong label. Eval GT is never
#    perturbed. NOTE render() rounds to 2dp, so at this magnitude the noise lands as
#    ~26% of values shifting by exactly +/-0.01 rather than smooth jitter (measured over
#    256k boxes; only 5.4% of boxes come out unchanged). See test_box_noise.py.
#
# 2. --min_boxes 0: no box-count filter at all. 144164 rows instead of 132455 (+8.8%),
#    including 71 rows whose target is an empty list, {"n": 0}. Those teach "emit nothing"
#    for a valid clip; they are 0.05% of the corpus, kept deliberately per instruction.
#    Steps per epoch go 4139 -> 4505, so 3 epochs is 13515 steps, not 12420.
#
# HARDWARE: this is gpu012 with A100-SXM4-80GB, not gpu017's H100 NVL. Expect step time
# well above the 14-20 s/it the H100 run saw. CUDA_VISIBLE_DEVICES is set to the two free
# cards; if SLURM regrants a different set, override it rather than editing this file --
# device indices are allocation-relative and "GPU2" means nothing outside this grant.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

RESUME_FROM="${RESUME_FROM:-results/idea_4a_sg_perc_scannetv2_vgjson_video_lora/checkpoint-6000}"
OUTPUT_DIR="results/idea_4a_sg_perc_scannetv2_vgjson_video_noise_lora"
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
  --min_boxes 0
  --box_noise 0.005
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
