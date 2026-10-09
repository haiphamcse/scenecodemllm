#!/bin/bash
# Fetch everything the offline compute nodes need into HF_HOME (${SR_HF_HOME}). Login node only.
#   scripts/download_weights.sh
# Needs a Hugging Face token whose account accepted the terms of any gated repo below. Either
# `export HF_TOKEN=...` in this shell or `hf auth login` (with HF_HOME=${SR_HF_HOME} so the token
# lands there). The token is never printed or copied anywhere by this script.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
activate_env v100
python - <<'PY'
import os
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

for repo in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"):
    print(repo, "->", snapshot_download(repo), flush=True)
# Revisions pinned to the snapshots the reference runs used (also in configs/caches/*.env).
omega = "05654241adc2f218dfb089c373a011f8a7040576"
for f in ("vggt_omega_1b_256_text.pt", "vggt_omega_1b_512.pt"):
    print(f, "->", hf_hub_download("facebook/VGGT-Omega", f, revision=omega), flush=True)
print("VGGT-1B ->", hf_hub_download("facebook/VGGT-1B", "model.safetensors",
                                    revision="860abec7937da0a4c03c41d3c269c366e82abdf9"), flush=True)
# VSI-Bench question table (the videos come with the data zip). cache_dir=HF_HOME, as eval.py reads it.
ds = load_dataset("nyu-visionx/VSI-Bench", "full", cache_dir=os.environ["HF_HOME"])["test"]
print("VSI-Bench full:", len(ds), "rows")
PY
