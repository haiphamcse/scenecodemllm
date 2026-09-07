#!/usr/bin/env python
"""Upload scannet_posed/ and vgllm_data/ to the HF repo, preserving both as top-level folders.

Auth: `hf auth login` or HF_TOKEN in the environment.
Resumable: rerun after an interruption; upload_large_folder skips what already landed.
"""

import os

from huggingface_hub import HfApi

SRC = os.environ.get("HF_SRC", "/home/ducpham/scratch/Working/dataset")
REPO = os.environ.get("HF_REPO", "haiphamcse/scenecodemlllm")

HfApi().upload_large_folder(
    repo_id=REPO,
    folder_path=SRC,
    repo_type="model",
    allow_patterns=["scannet_posed/**", "vgllm_data/**"],
    ignore_patterns=["**/.cache/**"],
    num_workers=int(os.environ.get("HF_WORKERS", 8)),
)
