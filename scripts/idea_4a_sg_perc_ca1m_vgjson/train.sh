#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"

# Measured on the first attempt: 94300 MiB steady of the H100's 95830 (98.4%), with
# only ~1.3 GB of headroom over a 1100-step run whose longest targets have not all
# been drawn yet. expandable_segments reuses a single growable virtual segment instead
# of fixed size-class blocks, so a late 16k-token sample cannot OOM on fragmentation
# alone.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Full train on the CA-1M VG-LLM-JSON corpus (9-DoF boxes in camera-0's frame), 64
# frames, latent-only MLLM input. Full fine-tune of the LLM (no LoRA) alongside the
# Perceiver; VGGT + native vision tower frozen.
#
# Carried over from the v2 overfit, which fixed the one-box length collapse (predicted
# counts went from [1] to 5..121 by step 40):
#   --min_boxes 5           degenerate exports dropped.
#   canonicalize()          largest-first order + leading {"n": N}, applied at load
#                           time in collator.load_scene_graph_text, so training and
#                           the eval callback's GT agree.
#
# Changed for full training:
#   --max_graph_tokens 16384  the overfit's 8192 (~120 boxes) silently dropped the top
#                             14% of the corpus -- and non-randomly, every large room.
#                             Corpus is median 63 / p90 133 / max 311 boxes at ~68
#                             tok/box, so 16384 (~240 boxes) keeps 99%.
#   --gradient_checkpointing  REQUIRED at 16384, not optional. Full-FT static cost is
#                             ~37 GB (params+grads+Adam+Perceiver) of the H100's 95 GB;
#                             the overfit already sat at 67.8 GB with 8192-token
#                             targets, so doubling the sequence OOMs without it.
#                             Costs ~30-40% throughput.
OUTPUT_DIR="results/idea_4a_sg_perc_ca1m_vgjson"
CORPUS_ROOT="/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ml-cubifyanything/ca1m_vgllm/train"
HF_HOME="/home/ducpham/scratch/Working/cache"

GLOBAL_BATCH=128
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
  # Full-FT: whole LLM (no LoRA) at 1e-5, Perceiver at its own 1e-4. Same as the
  # overfit, so the corpus is the only moving part.
  --full_finetune
  --learning_rate 1e-5
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 50
  --logging_steps 5
  # 2738 train scenes (2788 usable - 50 val) / 128 -> 22 optimizer steps/epoch,
  # 1100 total, measured at 370 s/step on the first attempt.
  # Eval every 63 = every ~3 epochs (generation at a 16k budget is slow; per-epoch
  # would spend hours of the run inside the callback). Save every 55 = every 2.5
  # epochs, ~5.6 h, so little is lost to a crash or a kill. 5 checkpoints at ~25 GB
  # each; the /scratch project quota has only ~3.3 TB of headroom, so
  # save_total_limit is a real constraint here rather than the usual "keep
  # everything". Note the two interact: 55 x 5 means the retained checkpoints only
  # ever span the last 275 steps, so an early-epoch model cannot be recovered late
  # in the run. No load_best_model_at_end: the recon metric is a callback, not a
  # Trainer metric, so the best epoch cannot be auto-selected -- pick the
  # checkpoint by reading ca1m_eval.txt.
  --save_steps 55
  --save_total_limit 20
  --ddp_find_unused_parameters False
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 5
  --vsibench_eval_enable
  --vsibench_eval_steps 63
  --vsibench_max_eval_samples 5
  # 8, not 4: each step reads 128 scenes x 64 PNGs off lustre. At 4 workers the GPU
  # sat idle for ~37% of wall-clock while the workers ran at ~7% CPU apiece -- they
  # were blocked on I/O, not compute, and the node has 12 cores.
  --dataloader_num_workers 8
)


# main_process_port 0 auto-picks a free port: the default 29500 collides when
# another accelerate job is co-scheduled on the same node.
accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_ca1m_vgjson/train.py "${args[@]}" "$@"
