#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# overfit.sh rerun at 200 epochs instead of 100, as the control for
# idea_4a_sg_perc_ca1m_vgjson_top100: that experiment reached f1 0.966 on 10 scenes, but
# only after epoch 140 -- its first 100 epochs looked like a failure at f1 0.147. So the
# open question is how much of the win is the filtered corpus + class-listing system
# prompt, and how much is simply running twice as long. This run holds everything but
# the epoch count fixed.
#
# Unchanged from overfit.sh on purpose: unfiltered ca1m_vgllm corpus (no top-100 label
# mapping, no 50-box cap), no system prompt, --max_graph_tokens 8192. The corpus is
# median 63 / max 311 boxes, so the stratified 10 spans ~5-120 boxes -- a harder sample
# than top100's 5-50, which is part of what is being compared.
#
# One continuous 200-epoch run, not 100 + a restarted 100: Adam state stays live and
# warmup runs once, so the optimizer restart is not a confound here.
#
# New output dir: the old run's ca1m_eval.txt lives in ..._overfit and the callback
# appends, which would interleave two runs' step blocks in one file.
OUTPUT_DIR="results/overfit/idea_4a_sg_perc_ca1m_vgjson_overfit_ep200"
CORPUS_ROOT="/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ml-cubifyanything/ca1m_vgllm/train"
HF_HOME="/home/ducpham/scratch/Working/cache"

# Global batch == the overfit set, so one optimizer step is one full pass over it.
GLOBAL_BATCH=10
BATCH_SIZE=1
NPROC_PER_NODE=1

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
  --corpus_root "${CORPUS_ROOT}"
  --hf_home "${HF_HOME}"
  --max_graph_tokens 8192
  --min_boxes 5
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # Full-FT: whole LLM (no LoRA) at 1e-5, Perceiver at its own 1e-4.
  --full_finetune
  --learning_rate 1e-5
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 200
  --logging_steps 1
  --save_strategy no
  --ddp_find_unused_parameters False
  --lr_scheduler_type constant_with_warmup
  --warmup_ratio 0.03
  --weight_decay 0.0
  --eval_strategy no
  --report_to tensorboard
  --overfit True
  --overfit_num_samples 10
  --vsibench_eval_enable
  # Every 40, not 20: generation at an 8192-token budget over all 10 scenes is the
  # expensive part of this run, and 5 evals still bracket the 140-200 region where the
  # top100 run made its jump.
  --vsibench_eval_steps 40
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_ca1m_vgjson/train.py "${args[@]}" "$@"
