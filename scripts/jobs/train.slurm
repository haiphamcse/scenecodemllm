#!/bin/bash
# One training window on 2 H100s (submitted by scripts/train.sh; sbatch options are given there).
#   args: <experiment> <output dir> [extra sr_vggt.train flags...]
# Resumes from the newest checkpoint in <output dir>; saves every 25 steps and 5 min before the
# window ends. Global batch 256 = per-device 1 x grad accum (256 / #GPUs).
set -euo pipefail
source "${SR_ROOT}/scripts/lib.sh"
load_config "$1"
OUT=$2
shift 2
activate_env h100
offline
NGPU=$(nvidia-smi -L | wc -l)
echo "=== ${SLURM_JOB_ID} | ${EXP} | ${NGPU} GPUs | $(hostname) | out ${OUT} ==="
nvidia-smi topo -m || true   # SYS between the GPUs hangs DDP: resubmit with NCCL_P2P_DISABLE=1

args=(
  "${MODEL_ARGS[@]}" "${CACHE_ARGS[@]}" "${TRAIN_ARGS[@]}"
  --output_dir "${OUT}"
  --jsonl_path "${MIX_JSONL}"
  --data_root "${SR_VSI590K}"
  --vggt_cache_root "${CACHE_ROOT}"
  --attn_implementation sdpa          # torch 2.13 SDPA beats flash-attn by 11% on H100
  --bf16 True
  --gradient_checkpointing False      # 17% faster; flip to True on OOM
  --per_device_train_batch_size 1
  --gradient_accumulation_steps $((256 / NGPU))
  --use_liger_kernel True             # fused CE (model.py); TRL needs the flag too
  --num_train_epochs 1
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --logging_steps 10
  --save_steps 25
  --save_total_limit 100
  --resume_from_checkpoint latest
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --report_to tensorboard
)
if [[ -n "${INIT_FROM}" ]]; then
  INIT_CKPT="${INIT_CKPT:-${SR_OUTPUT_ROOT}/${INIT_FROM}}"
  [[ -f "${INIT_CKPT}/adapter_model.safetensors" ]] || { echo "ERROR: ${INIT_CKPT} has no adapter" >&2; exit 1; }
  args+=(--init_from_checkpoint "${INIT_CKPT}")
fi
accelerate launch --config_file "${SR_ROOT}/scripts/jobs/accelerate_ddp.yaml" \
  --num_processes "${NGPU}" --main_process_port 0 -m sr_vggt.train "${args[@]}" "$@"
