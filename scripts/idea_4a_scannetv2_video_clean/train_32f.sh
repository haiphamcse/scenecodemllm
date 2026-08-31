set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"     # yes

GLOBAL_BATCH=128
BATCH_SIZE=1

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
  --hf_home "${HF_HOME}"
  --vggt_variant 512
  --max_graph_tokens 8192
  --min_boxes 5
  --bf16 True
  --gradient_checkpointing True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  # 10 rows at grad-accum 10 == 1 optimizer step per epoch, so this is 200 steps.
  --num_train_epochs 200
  --logging_steps 1
  --save_strategy no
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.0
  --ddp_find_unused_parameters False
  --eval_strategy no
  --report_to tensorboard
  --vsibench_eval_enable
  --vsibench_eval_steps 200
  --dataloader_num_workers 8
  --lora_r 256
  --lora_alpha 512 
  --lora_dropout 0.05
)

