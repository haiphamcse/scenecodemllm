#!/bin/bash
# Fetch the model weights the offline compute nodes need into HF_HOME (${SR_HF_HOME}). Login node.
#   scripts/download_weights.sh
# Gated repos need a token whose account accepted their terms on huggingface.co: either
# `export HF_TOKEN=...` in this shell or `hf auth login` (with HF_HOME=${SR_HF_HOME} so the token
# lands there). The token is never printed or copied anywhere by this script.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
activate_env h100
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download

for repo in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"):
    print(repo, "->", snapshot_download(repo), flush=True)
# Revisions pinned to the snapshots the reference runs used (also in configs/*.env).
omega = "05654241adc2f218dfb089c373a011f8a7040576"
for f in ("vggt_omega_1b_512.pt", "vggt_omega_1b_256_text.pt"):
    print(f, "->", hf_hub_download("facebook/VGGT-Omega", f, revision=omega), flush=True)
print("VGGT-1B ->", hf_hub_download("facebook/VGGT-1B", "model.safetensors",
                                    revision="860abec7937da0a4c03c41d3c269c366e82abdf9"), flush=True)
PY
