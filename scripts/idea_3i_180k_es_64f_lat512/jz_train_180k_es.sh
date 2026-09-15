#!/bin/bash

set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# idea_3i_180k_es_64f_lat512: the lat512 launcher on det_es FRAMES. Qwen frames = each clip's
# <scene>/det_es/frames/frameNN.jpg (64 EmbodiedScan/ARKit-posed frames, already at the 128-tok
# size; no frame cache), VGGT features = vggt_cache_256_64f_es (exported from those same JPEGs,
# stamped frames=det_es), mix = vqa_train_es.jsonl (vqa_train minus the 60 ARKit videos without
# det_es). Everything else identical to jz_train_180k_lat512.sh.
#
# idea_3i_180k_cached_64f_lat512: VQA-only 180k mix = VSI-590K (scannet/scannetppv2/
# arkitscenes, 110,944 rows) + VLM-3R vsibench_train (68,548 rows incl. all route rows), built
# by idea_3i_180k_cached_64f_lat512/build_mix_180k.py (joint180k_vlm3r/vqa_train.jsonl,
# ~702 steps/epoch at GLOBAL_BATCH 256). Otherwise the lat512 launcher of idea_3i_130k_cached_64f:
# frame_num_latents 512, lr 1e-4 shared, LoRA r64, 64 frames at 128 tok.
# 64-FRAME variant of jz_train_130k.sh: same 130k mix, fps 1.0 capped at 64 frames instead
# of 32, Qwen budget 128 tok/frame (collator constants), VGGT features from vggt_cache_256_64f
# and Qwen frames from joint130k/frame_cache_64f (both built at 64 frames). Det rows keep
# their 32 pre-extracted frames.
# Joint VQA + 3DOD on the 130k mix, from scratch: 130k VSI-590K rows (scannet 52k,
# scannetppv2 52k, arkitscenes 26k, every video of those sources) + 40k ScanNet det
# windows (every scene). Built offline by idea_3i_130k_joint_cached/build_mix.py.
# Same recipe as the cached H100 continuation of the 590k joint run (torch 2.13 venv, sdpa,
# Liger, VGGT-256 features from vggt_cache_256, pre-decoded Qwen frames from the frame
# cache, no in-training eval), only the manifest and the output dir differ.

# Runs in a venv layered on the pytorch-gpu module, NOT vsibench_eval_full: flash-attn 2
# lives in the module stack (no nvcc on the login node to build it into conda) and the
# conda env is kept untouched for every other experiment. submit.sh has already activated
# conda by the time this runs; sourcing the venv puts its python first on PATH.
# The job inherits the LOGIN node's MODULEPATH (sbatch --export=ALL); under it pytorch-gpu
# resolves to a build whose openmpi is absent on H100 nodes (libmpi.so.40 at import).
# Re-init modules from this node's profile and load arch/h100 for the H100 tree, which
# is also the tree these venvs were built on. SR_MODULE/SR_VENV pick the torch stack.
# SR_VENV="" (set but empty) means: stay in the conda env submit.sh activated. That is the
# A100 path -- arch/a100 has no torch 2.13 build and the h100_t213 venv is welded to the
# H100 build -- so A100 windows run torch 2.6 + sdpa + Liger from vsibench_eval_full.
# ${VAR-default} (no colon) keeps an explicit empty value; ${VAR:-default} would not.
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
# Job 1666171 died at step 77 on scannetppv2/0eba3981c9.mp4 with DECORDError:
# "Unable to handle EOF ... DECORD_EOF_RETRY_MAX=10240". The file is fine (decodes
# in 7.8 s on an idle node) -- 48 decode workers made Lustre slow enough to burn the
# retry budget. More budget, and fewer workers below.
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"

OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/idea_3i_180k_es_64f_lat512"
CACHE_ROOT="${SR_VGGT_CACHE:-${SCRATCH:-/home/ducpham/scratch/Working}/dataset/vggt_cache_256_64f_es}"
MIX_JSONL="${SR_MIX_180K:-${SCRATCH:-/home/ducpham/scratch/Working}/dataset/vgllm_data/joint180k_vlm3r/vqa_train_es.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"
MERGED_MODEL="${MERGED_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
INIT_PROJECTOR="${INIT_PROJECTOR:-}"

if [[ ! -f "${MIX_JSONL}" ]]; then
  echo "ERROR: ${MIX_JSONL} missing. Run idea_3i_180k_es_64f_lat512/filter_mix_es.py first." >&2
  exit 1
fi
[[ -d "${CACHE_ROOT}" ]] || { echo "ERROR: ${CACHE_ROOT} missing. Run scripts/idea_3i_180k_es_64f_lat512/export_vggt_es_v100.slurm first." >&2; exit 1; }

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
  # Measured 2026-09-10 on one H100, same 6800-token bf16 step (profiling/bench_attn_h100.slurm):
  #   torch 2.6 sdpa 282 ms | 2.8 sdpa 275 | 2.8 flash 273 | 2.13 flash 273 | 2.13 SDPA 243 ms.
  # On torch 2.13 PyTorch's own SDPA (fused flash/cuDNN kernels) beats the flash-attn 2.8.3
  # package by 11%, so sdpa is the default here. SR_ATTN=flash_attention_2 to use the package.
  --attn_implementation "${SR_ATTN:-sdpa}"
  --jsonl_path "${MIX_JSONL}"
  --data_root "${DATA_ROOT}"
  --hf_home "${HF_HOME}"
  --overfit_on_eval False
  --bf16 True
  # OFF deliberately. Measured on joint (job 1846554 vs 1819661, both from ckpt-500):
  # gc on = 41.7 GB peak / 50.2 reserved / 228 s/it; gc off = 67.1 / 75.3 / 190 s/it.
  # 17% faster for +25 GB, leaving ~9 GiB margin -- the same margin the 2026-09-04
  # OOM cascade died on. Flip back to True if any run OOMs.
  --gradient_checkpointing False
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate 1e-4
  --frame_num_latents 512
  --camera_num_latents 32
  --frame_widening_factor 2
  # Must match the VGGT variant below; the collator resizes to this and the encoder
  # was trained at it.
  --vggt_image_resolution 256
  --vggt_cache_root "${CACHE_ROOT}"
  # det rows only. The 3DOD-only runs that produced ckpt-850 left this at 0.0;
  # 0.005 is what the older idea_4a noise runs used (see train_lora_2gpu_noise.sh).
  --box_noise 0.005
  # Probability of jittering a det clip; one draw per clip, DUSt3R strength. 0.0 disables.
  --color_jitter 0.0
  # Fused linear cross-entropy (never materialises the [seq, 151936] logits) plus
  # Liger's RMSNorm/RoPE/SwiGLU kernels. The fork's forward carries the fused-loss port;
  # TRL needs the flag too, or it reads outputs.logits, which is None under Liger.
  --use_liger_kernel True
  --num_train_epochs 1
  --logging_steps 10
  # 2 GPUs makes a step ~6-7 min, so 100 steps is >10 h between saves. 50 halves
  # what a wall-time TIMEOUT throws away.
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
  # EXPLICIT False: the dataclass default is True with eval_steps 20, so merely omitting
  # --vsibench_eval_enable fired the callback at step 1180 and killed job 1967514 (its
  # generate() path also passes vggt_*_tokens kwargs the model rejects -- unfixed, unused).
  --vsibench_eval_enable False
  --report_to tensorboard
  --video_fps 1.0
  --video_max_frames 64
  # 200 for every experiment: puts all four on one eval grid so scores compare at
  # matched steps. An eval is 25 min (VQA) or 44 min (joint/multiscale -- their
  # collator preprocesses each sample twice, once per branch), so 100 was costing
  # joint ~12% of a window against cold's 8%.
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  # 12 (48 across 4 ranks) starved Lustre and tripped decord's EOF retry limit.
  --dataloader_num_workers 8
)

if [[ -n "${INIT_PROJECTOR}" ]]; then
  args+=(--init_projector "${INIT_PROJECTOR}")
fi

accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  idea_3i_180k_es_64f_lat512/train.py "${args[@]}" "$@"
