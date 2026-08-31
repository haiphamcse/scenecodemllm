#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Full train on VG-LLM's ScanNet-v2 json, LoRA, 2 GPUs. Port of
# scripts/idea_4a_sg_perc_ca1m_vgjson/train_lora_2gpu.sh; the optimisation settings are
# copied deliberately, only the data and the horizon differ.
#
# WHY LoRA AND NOT --full_finetune (which scripts/.../overfit.sh uses). train.py loads the
# model in bf16 and trains the params in place, with no fp32 master weights. On CA-1M that
# stalled a full fine-tune outright: over checkpoint-275 -> 330 (55 optimizer steps) 98.7%
# of LLM weights were BITWISE unchanged, because an Adam step below half the local spacing
# of an 8-mantissa-bit weight rounds back onto the old value. peft 0.19.1's get_peft_model
# defaults to autocast_adapter_dtype=True, so lora_A/lora_B come back fp32 regardless of
# base dtype and sidestep it. It does NOT cast modules_to_save, where build_lora_config
# puts vggt_projector -- the Perceiver still trains in bf16, at lr 1e-4 where the steps are
# large enough to land. Same deliberate call as the CA-1M run. The 20-scene overfit here
# reached f1 0.830 under full-FT, which is not counter-evidence: memorising 20 samples for
# 100 epochs needs far larger updates than fitting 132k rows does.
#
# GPUs 2 and 3 on gpu013. NOT the gpu017 situation the CA-1M script documents: `nvidia-smi
# topo -m` here reports NV4 between EVERY pair, so there is no cross-socket P2P aperture to
# fault on and NCCL_P2P_DISABLE is left OFF (override the env var if an Xid 31 MMU fault
# shows up in dmesg). GPU1 is another user's (~9 GB); GPU0 is free if you want to move.
#
# CARRY-OVER WARNING from the CA-1M 2-GPU run, unresolved: at matched epoch/lr/effective
# batch, grad_norm ran 0.033-0.050 on 2 GPUs against 0.010-0.021 on 1, every point above
# the single-GPU max over n=50. Ruled out: batch size, lr, gradient clipping, NCCL P2P
# corruption (tested with P2P off, no change), num_items_in_batch gathering. Leading
# UNVERIFIED candidate: loss is token-normalized against a globally gathered count while
# DDP averages rank gradients instead of token-weighting them, and box counts run 5-57 per
# clip so the ranks carry very different token counts. Treat any 1-GPU curve as a different
# run, not a comparable one.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"

OUTPUT_DIR="results/idea_4a_sg_perc_scannetv2_vgjson_lora"
HF_HOME="/home/ducpham/scratch/Working/cache"

# 32, not 64. The effective batch is what the lr and the step horizon are defined against,
# and a LARGER batch is what stalled the original CA-1M full-FT run (128 -> per-parameter
# SNR collapse). The second GPU takes half the grad-accum work: 1 x 2 x 16 = 32.
GLOBAL_BATCH=32
BATCH_SIZE=1
NPROC_PER_NODE=2

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
  # No --train_json / --val_json / --image_root: the ScanNetArguments defaults already
  # point at dataset/vgllm_data/{train,evaluation/threedod/scannet} and dataset/ as the
  # root the json's "scannet/posed_images/..." paths hang off.
  # 4096, not CA-1M's 16384: ScanNet tops out at 57 boxes (~68 tokens each). Measured over
  # the full json, 0 rows are dropped for exceeding this budget.
  --max_graph_tokens 4096
  --min_boxes 5
  --bf16 True
  # Left OFF, unlike the CA-1M script. That run needed it for a 16384-token target over 64
  # frames; here it is 4096 tokens over 4 frames, and the 20-scene overfit ran a heavier
  # config (full-FT, so LLM grads + Adam state resident) without it on the same 80 GB card.
  # Turn it on with --gradient_checkpointing True if this OOMs.
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # No --full_finetune: that flag is what selects the LoRA-disabled path in train.py.
  # Defaults from common/argument.py apply -- r=64, alpha=128, dropout=0.1 -- over
  # q,k,v,o + gate,up,down, with vggt_projector in modules_to_save so the Perceiver trains
  # in full rather than through a low-rank factorisation.
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  # Full corpus: 132455 usable rows / 951 scenes (0 dropped for missing frames -- the
  # posed_images download that was partial in July has finished; 11709 rows dropped for
  # <5 boxes). 132455 / 32 = 4139 optimizer steps per epoch, 12417 over 3 epochs.
  # NOTE the redundancy this buys: ~139 overlapping 4-frame clips per room, so an "epoch"
  # here is far less diverse than the row count suggests.
  --num_train_epochs 3
  --logging_steps 5
  # Cosine over the whole 12417-step horizon, warmup 0.03 = 372 steps. The schedule only
  # reaches its low-lr phase if the run finishes; a restart re-plans from step 0.
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  # 20 x 500 = a 10000-step recoverable window, ~50 GB of LoRA checkpoints (adapters +
  # vggt_projector + optimizer state, not a 13 GB full model). /scratch is at 96% with
  # ~1.3 TiB to the project soft quota, so this fits with room.
  # TRAP: --save_steps is SILENTLY IGNORED on resume. DefaultFlowCallback reads
  # state.save_steps out of the checkpoint's trainer_state.json and transformers only
  # WARNS that the args disagree. To change it mid-run, edit trainer_state.json in the
  # checkpoint you resume from. Same applies to eval_steps and logging_steps.
  --save_steps 500
  --save_total_limit 20
  --ddp_find_unused_parameters False
  # 7200 s, up from the 1800 s default. REQUIRED under DDP, not tuning: the eval callback
  # generates on rank 0 only (callbacks.py `if self.state.is_main_process`), so rank 1 runs
  # ahead and blocks inside the gradient allreduce for the whole eval. On CA-1M that was
  # 21.5 min against a 30 min default -- a timeout here aborts the job via NCCL watchdog.
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  # HELD-OUT eval: --overfit False makes train.py read the val json (2188 usable rows over
  # 240 scenes) and take one clip per scene, so eval scenes are not train scenes.
  --overfit False
  --val_size 10
  --vsibench_eval_enable
  # A 4096-token greedy decode x 10 clips, and CA-1M's held-out f1 sat at 0.0 with a
  # repetition collapse for its whole run -- low information per unit of wall clock, so it
  # is deliberately decoupled from the save cadence.
  --vsibench_eval_steps 1000
  --vsibench_max_eval_samples 10
  # Per process, so 16 workers. Each step reads clips x 4 jpgs off lustre (lighter than
  # CA-1M's 64 PNGs); the node has 128 cores.
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_scannetv2_vgjson/train.py "${args[@]}" "$@"
