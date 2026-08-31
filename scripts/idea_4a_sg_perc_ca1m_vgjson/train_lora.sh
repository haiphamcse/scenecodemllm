#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Full train on the CA-1M VG-LLM-JSON corpus, LoRA instead of full fine-tune. This is the
# rerun of train.sh after that run stalled: loss went 0.85 -> 0.66 (ep1) -> 0.556 (ep10)
# -> 0.541 (ep16) and stopped, grad norm 0.046 -> 0.004. The postmortem measured, over
# checkpoint-275 -> checkpoint-330 (55 optimizer steps), 98.7% of LLM weights BITWISE
# unchanged and a median relative change of exactly 0.0. Three compounding causes, all
# three addressed below.
#
#   1. NO fp32 MASTER WEIGHTS. train.py loads the model in bf16 and trains the params in
#      place, so Adam's update is rounded back into an 8-mantissa-bit weight. A step below
#      half the local spacing lands on the same value and the weight never moves.
#
#      LoRA fixes most of this for free: peft's get_peft_model defaults to
#      autocast_adapter_dtype=True, so lora_A/lora_B come back fp32 regardless of the base
#      dtype. It does NOT cast the `modules_to_save` copies, and build_lora_config puts
#      vggt_projector there -- so the Perceiver still trains in bf16. Measured on a toy of
#      the same shape: 74% of that module bitwise unchanged after 5 AdamW steps at lr 1e-4,
#      0% if cast to fp32. Deliberate call to keep it bf16 for now; if the Perceiver looks
#      frozen again, casting the trainable params to fp32 after the trainer is built is the
#      fix, and it is affordable here (LoRA already dropped the LLM's grads + Adam state,
#      ~12 GB, off a run that peaked at 94.3 GB of the H100's 95.8).
#
#   2. GLOBAL_BATCH 128 -> 32. Averaging 128 diverse scenes' gradients drove
#      per-parameter SNR down, so Adam's m/sqrt(v) stayed far below lr. Smaller batches
#      are noisier per step but move further, and give 4x the optimizer steps per epoch
#      (22 -> 87). Peak memory is unchanged: per_device batch stays 1, this is grad accum.
#
#   3. --learning_rate 1e-5 -> 1e-4. ASSUMPTION worth knowing: the earlier plan said
#      5e-5, but that number was chosen to outrun bf16 rounding on a full fine-tune. Under
#      LoRA this group trains only the adapters, where 1e-4 is the usual operating point
#      (r=64 / alpha=128 => scaling 2.0). Drop it back to 5e-5 if you want the more
#      conservative version -- it is one number and nothing else depends on it.
#
# Unchanged on purpose so the comparison holds: corpus, --min_boxes 5, canonicalize(),
# --max_graph_tokens 16384, --gradient_checkpointing (still required at 16384),
# --perceiver_lr 1e-4, frame_num_latents 256 / widening 2.
#
# New output dir: the callback APPENDS to ca1m_eval.txt, so writing into the old dir would
# interleave two runs' step blocks, and the stalled run's checkpoints are the evidence the
# postmortem rests on.
OUTPUT_DIR="results/idea_4a_sg_perc_ca1m_vgjson_lora"
CORPUS_ROOT="/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ml-cubifyanything/ca1m_vgllm/train"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=32
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
  --max_graph_tokens 16384
  --min_boxes 5
  --bf16 True
  --gradient_checkpointing True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # No --full_finetune: that flag is what selects the LoRA-disabled path in train.py.
  # Defaults from common/argument.py apply -- r=64, alpha=128, dropout=0.1 -- targeting
  # q/k/v/o + gate/up/down, with vggt_projector in modules_to_save so the Perceiver is
  # still trained in full rather than through a low-rank factorisation.
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  # 2788 usable - 5 val = 2783 train / 32 -> 87 optimizer steps/epoch, 26100 total.
  # At the stalled run's measured throughput that is roughly 24 days end to end.
  --num_train_epochs 300
  --logging_steps 5
  # Cosine over the full 26100-step horizon, warmup 0.03 = 783 steps (~9 epochs). Note the
  # coupling: cosine only reaches its low-lr phase if the run actually finishes. Killed at
  # epoch 60 it will have spent its whole life near peak lr, and a restart re-plans the
  # schedule from step 0 rather than resuming the curve.
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  # Every 435 steps = every 5 epochs, ~9.1 h at the measured 75.4 s/step. Halved from 870
  # after the first allocation expired mid-run: only checkpoint-870 existed, so steps
  # 871-1577 (~15 h) were lost. Eval stays at 870 -- it is a slow 16384-token greedy decode
  # and is deliberately decoupled from checkpoint cadence.
  # limit 20, not 10, so halving save_steps does not halve the recoverable window: 20 x 435
  # = 8700 steps = 100 epochs, ~50 GB of LoRA checkpoints (adapters + vggt_projector +
  # optimizer state, 2.5 G each -- not a 13 GB full model).
  --save_steps 435
  --save_total_limit 20
  --ddp_find_unused_parameters False
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 5
  --vsibench_eval_enable
  --vsibench_eval_steps 870
  --vsibench_max_eval_samples 5
  # 8, not 4: each step reads scenes x 64 PNGs off lustre. At 4 workers the GPU sat idle
  # ~37% of wall-clock with the workers at ~7% CPU apiece -- blocked on I/O, not compute.
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_ca1m_vgjson/train.py "${args[@]}" "$@"
