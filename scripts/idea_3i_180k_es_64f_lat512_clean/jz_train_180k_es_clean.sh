#!/bin/bash

set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# idea_3i_180k_es_64f_lat512_clean: VQA-only 190k mix on det_es frames. Qwen frames = each clip's
# <scene>/det_es/frames/frameNN.jpg (64 posed frames, stored at the 128-tok size), VGGT features =
# vggt_cache_256_64f_es (exported from the same JPEGs, stamped frames=det_es),
# mix = vqa_train_190k_es_clean.jsonl (build_mix_190k.py: all 176,812 rows of vqa_train_180k_es_clean.jsonl
# + 13,188 new rows from the same det_es videos, relative_direction_object at 35,000; 190,000 rows).

# Runs in a venv layered on the pytorch-gpu module, not vsibench_eval_full (kept untouched).
# The job inherits the login node's MODULEPATH, under which pytorch-gpu resolves to a build
# whose openmpi is absent on H100 nodes (libmpi.so.40 at import): re-init modules and load
# arch/h100. SR_VENV="" (set but empty) stays in the conda env submit.sh activated (A100 path).
# ${VAR-default} (no colon) keeps an explicit empty value.
SR_ARCH="${SR_ARCH:-h100}"
SR_MODULE="${SR_MODULE:-2.13.0}"
SR_VENV="${SR_VENV-${WORK}/envs/h100_t213}"
if [[ -z "${SR_VENV}" ]]; then
  echo "=== env: conda $(python -c 'import torch,sys;print("py",sys.version.split()[0],"torch",torch.__version__)') (SR_VENV empty) ==="
elif [[ -d "${SR_VENV}" ]]; then
  # /etc/profile trips set -u (PS1 ...); relax the flags around it.
  set +eu; unset MODULEPATH; source /etc/profile >/dev/null 2>&1; set -eu
  module purge
  module load "arch/${SR_ARCH}"
  module load "pytorch-gpu/py3/${SR_MODULE}"
  # shellcheck disable=SC1091
  source "${SR_VENV}/bin/activate"
  echo "=== env: $(python -c 'import torch,sys;print("py",sys.version.split()[0],"torch",torch.__version__)') via ${SR_VENV} ==="
else
  echo "ERROR: ${SR_VENV} missing -- build it with ARCH=${SR_ARCH} MODULE_VERSION=${SR_MODULE} VENV=${SR_VENV} bash jeanzay/setup_fsdp_env.sh" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Many decode workers on Lustre burned decord's EOF retry budget (job 1666171); more budget.
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"

OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/idea_3i_190k_es_64f_lat512_clean"
CACHE_ROOT="${SR_VGGT_CACHE:-${SCRATCH:-/home/ducpham/scratch/Working}/dataset/vggt_cache_256_64f_es}"
MIX_JSONL="${SR_MIX_180K:-${SCRATCH:-/home/ducpham/scratch/Working}/dataset/idea_3i_dataset_jsons/vqa_train_190k_es_clean.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"
MERGED_MODEL="${MERGED_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"

if [[ ! -f "${MIX_JSONL}" ]]; then
  echo "ERROR: ${MIX_JSONL} missing. Run idea_3i_180k_es_64f_lat512_clean/build_mix_190k.py first." >&2
  exit 1
fi
[[ -d "${CACHE_ROOT}" ]] || { echo "ERROR: ${CACHE_ROOT} missing. Run scripts/idea_3i_180k_es_64f_lat512_clean/export_vggt_es_v100.slurm first." >&2; exit 1; }

GLOBAL_BATCH=256
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))

args=(
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MERGED_MODEL}"
  # torch 2.13 SDPA beat the flash-attn 2.8.3 package by 11% on H100 (243 vs 273 ms/step).
  # SR_ATTN=flash_attention_2 to use the package.
  --attn_implementation "${SR_ATTN:-sdpa}"
  --jsonl_path "${MIX_JSONL}"
  --data_root "${DATA_ROOT}"
  --hf_home "${HF_HOME}"
  --bf16 True
  # Off deliberately: 17% faster for +25 GB peak (~9 GiB margin left). Flip to True on OOM.
  --gradient_checkpointing False
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --frame_num_latents 512
  --camera_num_latents 32
  --frame_widening_factor 2
  # Cache key: must match the export of vggt_cache_256_64f_es.
  --vggt_image_resolution 256
  --vggt_cache_root "${CACHE_ROOT}"
  # Liger fused CE + kernels. The model forward carries the fused-loss port; TRL needs the
  # flag too, or it reads outputs.logits, which is None under Liger.
  --use_liger_kernel True
  --num_train_epochs 1
  --logging_steps 10
  --save_steps 25
  --save_total_limit 100
  # get_last_checkpoint guard in train.py makes this a no-op on the first window.
  --resume_from_checkpoint latest
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --ddp_find_unused_parameters False
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --video_fps 1.0
  --video_max_frames 64
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  # 12 per rank starved Lustre and tripped decord's EOF retry limit.
  --dataloader_num_workers 8
)

accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  idea_3i_180k_es_64f_lat512_clean/train.py "${args[@]}" "$@"
