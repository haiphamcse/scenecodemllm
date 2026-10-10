#!/bin/bash
# Build the Python venv for one GPU kind, layered on the IDRIS pytorch-gpu module (login node,
# needs internet through the IDRIS proxy). Run once per kind:
#   scripts/setup_env.sh h100     training + login-node tools   (arch/h100 + pytorch-gpu 2.13.0)
#   scripts/setup_env.sh a100     eval                          (arch/a100 + pytorch-gpu 2.8.0)
#
# Why a venv: --system-site-packages inherits the module's torch/CUDA/NCCL for ~2k inodes, where a
# conda env costs ~150k of the 500k $WORK inode quota. A venv is welded to the interpreter it was
# built on, so each GPU kind gets its own, built on that kind's module tree (arch/h100 and arch/a100
# switch MODULEPATH to the partition's own software tree; H100 nodes lack the default tree's OpenMPI).
# perceiver-io, vggt-omega and VGGT-1B are vendored in third_party/: only PyPI packages here.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
KIND=${1:?usage: scripts/setup_env.sh <h100|a100>}
VENV="${SR_ENV_ROOT}/${KIND}"

set +eu; unset MODULEPATH; source /etc/profile >/dev/null 2>&1; module purge >/dev/null 2>&1
module load "${SR_ARCH[$KIND]}"
module load "pytorch-gpu/py3/${SR_MODULE[$KIND]}"
set -eu
echo "=== building ${VENV} on $(command -v python) ==="
[[ -d "${VENV}" ]] || python -m venv --system-site-packages "${VENV}"
source "${VENV}/bin/activate"

# The 2.7+ modules ship a transformer_engine whose import raises TypeError; peft probes it with an
# unguarded import + hasattr(mod, "pytorch"). An EMPTY module earlier on sys.path hides it.
SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
if python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('transformer_engine') else 1)" 2>/dev/null; then
  printf '# Shadows the module copy, whose import raises TypeError. Intentionally empty.\n' > "${SP}/transformer_engine.py"
fi

python -m pip install --no-cache-dir --upgrade pip
pkgs=(
  "transformers==5.2.0" "trl==1.2.0" "peft==0.19.1" "accelerate==1.7.0" "datasets==4.8.4"
  "huggingface_hub==1.28.0" "decord==0.6.0" "qwen-vl-utils==0.0.14" "tensorboard==2.20.0"
  "fairscale==0.4.13"            # vendored perceiver-io imports checkpoint_wrapper from it
  "mistral_common>=1.8.8"        # the modules' copy is too old for transformers 5.2
  "liger-kernel==0.8.2"          # fused CE (needs the module's Triton)
  "einops" "safetensors" "pandas"
)
python -m pip install --no-cache-dir "${pkgs[@]}"

PYTHONPATH="${SR_ROOT}" python - <<'PY'
import torch, transformers, trl, peft, accelerate, decord
import sr_vggt.model, sr_vggt.ablations.spatialstack, sr_vggt.ablations.vlm3r  # vendored deps import
print("ok: torch", torch.__version__, "cuda", torch.version.cuda, "| transformers", transformers.__version__,
      "trl", trl.__version__, "peft", peft.__version__, "accelerate", accelerate.__version__)
PY
