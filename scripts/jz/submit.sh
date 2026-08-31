#!/bin/bash
#
# Generic Jean Zay submitter. Wraps any train script in this repo in a Slurm job,
# and chains further jobs until the run reaches its step target.
#
#   scripts/jz/submit.sh scripts/legacy/idea_3i/train_tuned.sh
#   NGPU=8 GPUKIND=a100 scripts/jz/submit.sh scripts/.../foo.sh
#   SMOKE=1 scripts/jz/submit.sh scripts/.../foo.sh --max_steps 2 --save_steps 1
#
# The train script is a plain bash script; it only has to honour NPROC_PER_NODE
# for its --num_processes and read paths from the SR_* vars. Chaining is opt-in
# from the CALLER, not the script:
#   JZ_OUTPUT_DIR=$SR_OUTPUT_ROOT/foo JZ_MAX_STEPS=4336 scripts/jz/submit.sh <script>
# Set neither and the job runs exactly one window and stops.
#
# Jean Zay picks the GPU from -C, NOT --partition. Passing --partition=gpu_p6
# trips the IDRIS submit filter: "Account btf@h100 ----- Job type v100".
# Memory is not requestable (--mem is rejected); it scales with the cores asked.
set -euo pipefail

TRAIN_SCRIPT="${1:?usage: submit.sh <train script> [extra args...]}"
shift || true

GPUKIND=${GPUKIND:-h100}
NGPU=${NGPU:-4}
SMOKE=${SMOKE:-0}

case "$GPUKIND" in
  # gpu_p6 node = 4x H100 + 96 physical cores => 24 per GPU.
  h100) ACCOUNT=${ACCOUNT:-btf@h100}; CPUS=${CPUS:-$(( NGPU * 24 ))} ;;
  # gpu_p5 node = 8x A100-80 + 64 physical cores => 8 per GPU.
  a100) ACCOUNT=${ACCOUNT:-btf@a100}; CPUS=${CPUS:-$(( NGPU * 8 ))} ;;
  *) echo "GPUKIND must be h100 or a100, got '$GPUKIND'" >&2; exit 1 ;;
esac

if (( SMOKE )); then
  TIME=${TIME:-02:00:00}; QOS=${QOS:-qos_gpu_${GPUKIND}-dev}
else
  TIME=${TIME:-20:00:00}; QOS=${QOS:-qos_gpu_${GPUKIND}-t3}
fi

# ---------------------------------------------------------------- submit side
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "${WORK}/logs"
  exec sbatch \
    --job-name="$(basename "${TRAIN_SCRIPT}" .sh)" \
    --account="${ACCOUNT}" --constraint="${GPUKIND}" --qos="${QOS}" \
    --nodes=1 --ntasks=1 --gres="gpu:${NGPU}" \
    --cpus-per-task="${CPUS}" --hint=nomultithread --time="${TIME}" \
    --output="${WORK}/logs/%x-%j.out" \
    --export=ALL,GPUKIND="${GPUKIND}",NGPU="${NGPU}",TIME="${TIME}",QOS="${QOS}",ACCOUNT="${ACCOUNT}",CPUS="${CPUS}",SMOKE="${SMOKE}",JZ_OUTPUT_DIR="${JZ_OUTPUT_DIR:-}",JZ_MAX_STEPS="${JZ_MAX_STEPS:-0}" \
    "$0" "${TRAIN_SCRIPT}" "$@"
fi

# ----------------------------------------------------------------- job side
cd "${WORK}/spatial_reasoning/finetuning"
source "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate vsibench_eval_full

# Site config: where the data, models and outputs actually are on this machine.
# Written by jz_provision.sh stage 7.
set -a; source "${WORK}/spatial_reasoning/finetuning/site.env"; set +a

# Compute nodes have no route out. Anything not already in HF_HOME must have been
# prefetched on a login node; failing loudly beats dying at step 0 of a 20 h job.
export HF_HOME="${SR_HF_HOME}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Compute nodes do not reliably inherit the login shell's locale, and Python then
# falls back to ASCII for encoding-less reads. TRL's model-card template contains
# UTF-8 bytes, so every checkpoint save dies with
#   UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3
# AFTER the training steps already succeeded. Force UTF-8 regardless of the node.
export PYTHONUTF8=1
export LANG=${LANG:-en_US.UTF-8}
export LC_ALL=${LC_ALL:-en_US.UTF-8}

export NPROC_PER_NODE="${NGPU}"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU-1)))
export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export NCCL_DEBUG=WARN
# Printed, not assumed: a group spanning a SYS hop hangs DDP's first broadcast in
# silence. If the matrix shows SYS between allocated GPUs, resubmit with
# NCCL_P2P_DISABLE=1. A full gpu_p6 node is 4x H100 behind NVLink (NV6 all pairs).
echo "=== nvidia-smi topo -m ==="; nvidia-smi topo -m || true
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

echo "=== job ${SLURM_JOB_ID} | ${NGPU}x${GPUKIND} | ${CPUS} cores | ${TIME} | ${QOS} ==="
echo "=== data=${SR_DATA_ROOT} cache=${SR_CACHE_ROOT} out=${SR_OUTPUT_ROOT} ==="

set +e
bash "${TRAIN_SCRIPT}" "$@"
rc=$?
set -e

(( SMOKE )) && exit $rc
(( rc != 0 )) && { echo "train script exited ${rc}; not chaining"; exit $rc; }

# Chain only if the caller said what to watch and it is short of target.
OUT="${JZ_OUTPUT_DIR:-}"; MAX="${JZ_MAX_STEPS:-0}"
[[ -z "$OUT" || "$MAX" == 0 ]] && { echo "no JZ_OUTPUT_DIR/JZ_MAX_STEPS; not chaining"; exit 0; }
NOW=0
compgen -G "${OUT}/checkpoint-*" >/dev/null && \
  NOW=$(ls -1d "${OUT}"/checkpoint-* | sed 's#.*/checkpoint-##' | sort -n | tail -1)
if (( NOW < MAX )); then
  echo "at ${NOW}/${MAX}, chaining another job"
  exec "$0" "${TRAIN_SCRIPT}" "$@"
fi
echo "reached ${NOW}/${MAX}, done"
