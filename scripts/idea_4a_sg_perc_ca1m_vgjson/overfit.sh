#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# Overfit sanity, second attempt. The first collapsed to a one-box output for every
# scene (f1 0.150 == precision == recall, because 3 of its 20 scenes genuinely had 1
# box) -- the model found the shortest valid string and sat there. Four length fixes,
# model and hyperparameters otherwise untouched:
#
#   1. --min_boxes 5           degenerate 1-box exports removed, so "emit one box"
#                              scores zero instead of 0.15.
#   2. token-budget filter     scenes whose target exceeds --max_graph_tokens are
#                              dropped rather than clipped mid-object; a truncated
#                              target has no closing bracket and no EOS, which trains
#                              the model never to terminate. ~68 tokens/box measured,
#                              so 8192 holds ~120 boxes.
#   3. canonical object order  largest-first by volume (graph_vgllm.canonicalize).
#                              Export order was arbitrary -- Spearman |rho| 0.13 vs
#                              volume -- so long scenes were unlearnable by position.
#   4. leading {"n": N}        sequence length becomes a supervised token instead of
#                              an implicit decision.
#
# The sample is now stratified across the box-count range instead of scenes[:N], so the
# run actually probes whether the model can produce DIFFERENT lengths -- the thing that
# failed. 10 scenes, not 20.
#
# Watch: `distinct` (was 1/20) and the predicted-count spread (was [1]).
OUTPUT_DIR="results/overfit/idea_4a_sg_perc_ca1m_vgjson_overfit"
CORPUS_ROOT="/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ml-cubifyanything/ca1m_vgllm/train"
HF_HOME="/home/ducpham/scratch/Working/cache"

# Global batch == the overfit set, so one optimizer step is one full pass over it.
# 100 epochs -> 100 steps, matching the previous run's step budget.
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
  # Full-FT: whole LLM (no LoRA) at 1e-5, Perceiver at its own 1e-4. Unchanged, so the
  # length fixes are the only moving part vs the previous overfit.
  --full_finetune
  --learning_rate 1e-5
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 100
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
  --vsibench_eval_steps 20
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_ca1m_vgjson/train.py "${args[@]}" "$@"
