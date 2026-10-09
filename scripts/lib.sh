# Sourced by every script and job: site config, env vars, experiment config, Python env.
# shellcheck shell=bash

SR_ROOT="${SR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SR_ROOT
if [[ ! -f "${SR_ROOT}/site.env" ]]; then
  echo "ERROR: ${SR_ROOT}/site.env missing: cp site.env.example site.env and edit it." >&2
  exit 1
fi
set -a; source "${SR_ROOT}/site.env"; set +a

export HF_HOME="${SR_HF_HOME}"
export PYTHONPATH="${SR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Compute nodes lose the UTF-8 locale; TRL's model-card write then kills every checkpoint save.
export PYTHONUTF8=1 LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
export FORCE_QWENVL_VIDEO_READER=decord DECORD_EOF_RETRY_MAX=40960
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${SR_LOG_DIR}"

# pytorch-gpu module each venv is layered on (setup_env.sh builds them the same way).
declare -A SR_MODULE=([h100]=2.13.0 [a100]=2.8.0 [v100]=2.6.0)
declare -A SR_ARCH=([h100]=arch/h100 [a100]=arch/a100 [v100]="")

# activate_env <h100|a100|v100>: re-init modules for THIS node, load the GPU's module tree, venv.
# A job inherits the login node's MODULEPATH, under which pytorch-gpu resolves to a build whose
# OpenMPI is absent on H100 nodes (libmpi.so.40 at import torch). /etc/profile trips `set -u`.
activate_env() {
  local kind=$1
  set +eu; unset MODULEPATH; source /etc/profile >/dev/null 2>&1
  module purge >/dev/null 2>&1
  [[ -n "${SR_ARCH[$kind]}" ]] && module load "${SR_ARCH[$kind]}"
  module load "pytorch-gpu/py3/${SR_MODULE[$kind]}"
  set -eu
  local venv="${SR_ENV_ROOT}/${kind}"
  [[ -d "${venv}" ]] || { echo "ERROR: ${venv} missing: scripts/setup_env.sh ${kind}" >&2; exit 1; }
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
}

offline() { export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; }

# load_config <experiment>: configs/<exp>.env + its cache; sets RUN_NAME, MIX_JSONL, CACHE_ROOT.
load_config() {
  EXP=$1
  local f="${SR_ROOT}/configs/${EXP}.env"
  [[ -f "${f}" ]] || { echo "ERROR: no config ${f} (have: $(ls "${SR_ROOT}/configs" | grep env$ | sed 's/.env$//' | tr '\n' ' '))" >&2; exit 1; }
  EVAL_PARTS=1; INIT_FROM=""
  source "${f}"
  load_cache "${CACHE}"
  RUN_NAME="sr_vggt_${EXP}"
  case "${MIX}" in
    190k) MIX_JSONL="${SR_MIX_190K}" ;;
    traces) MIX_JSONL="${SR_MIX_TRACES}" ;;
    *) echo "ERROR: unknown MIX=${MIX}" >&2; exit 1 ;;
  esac
}

# load_cache <cache>: configs/caches/<cache>.env; sets CACHE_ARGS, CACHE_ROOT.
load_cache() {
  local f="${SR_ROOT}/configs/caches/$1.env"
  [[ -f "${f}" ]] || { echo "ERROR: no cache config ${f}" >&2; exit 1; }
  source "${f}"
  CACHE_ROOT="${SR_CACHE_ROOT}/${CACHE_DIR}"
}

log() { echo "$(date '+%F %T') $*"; }

# Jobs I own with this name (pending or running). Empty output on squeue failure is NOT
# "no jobs": callers must check the exit status (a transient squeue error double-submitted once).
my_jobs() { squeue -h -u "${USER}" -n "$1" -o %i; }
