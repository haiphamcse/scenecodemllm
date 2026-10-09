#!/bin/bash
# Parity of sr_vggt vs the original forks on one A100 (submitted from a login node):
#   sbatch --account=<proj>@a100 -C a100 --qos=qos_gpu_a100-dev --gres=gpu:1 --cpus-per-task=8 \
#     --hint=nomultithread --time=02:00:00 --export=ALL,SR_ROOT=$PWD,FT=<finetuning dir>,... \
#     tests/parity.slurm <out root> <exp> [<exp> ...]
# Env (the original site): FT, FT_RESULTS, FT_CACHES, FT_DATA_ROOT, FT_MIX_190K, FT_MIX_TRACES,
# SR_HF_HOME (via site.env). Outputs <out root>/<exp>/{old,new}.pt, parity.json.
set -euo pipefail
source "${SR_ROOT}/scripts/lib.sh"
activate_env a100
offline
export PYTHONDONTWRITEBYTECODE=1   # never write __pycache__ into the original tree
OUT_ROOT=$1; shift
for exp in "$@"; do
  load_config "${exp}"
  export NEW_CACHE_ROOT="${CACHE_ROOT}"
  out="${OUT_ROOT}/${exp}"; mkdir -p "${out}"
  echo "=== ${exp} $(date +%T) ==="
  python "${SR_ROOT}/tests/parity.py" old "${exp}" "${out}" > "${out}/old.log" 2>&1 || { echo "old side failed"; tail -30 "${out}/old.log"; continue; }
  python "${SR_ROOT}/tests/parity.py" new "${exp}" "${out}" "${MODEL_ARGS[@]}" "${CACHE_ARGS[@]}" \
    > "${out}/new.log" 2>&1 || { echo "new side failed"; tail -30 "${out}/new.log"; continue; }
  python "${SR_ROOT}/tests/parity.py" compare "${exp}" "${out}" || true
done
