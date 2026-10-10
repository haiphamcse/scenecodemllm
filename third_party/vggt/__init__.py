# Vendored from https://github.com/facebookresearch/vggt @ a288dd0f14786c93483e45524328726ab7b1b4ce
# (2026-05-18), unmodified. Kept: models/aggregator.py and layers/ (all of it; the DINOv2
# patch embed pulls in every layer module). Dropped: models/vggt.py (the VGGT wrapper and its
# huggingface_hub mixin), heads/, dependency/, utils/, training/, demos. Only the aggregator
# is built; export_vggt_features.load_vggt fills it from the checkpoint's aggregator.* keys.
# Licensed under LICENSE.txt (VGGT License). This __init__.py and models/__init__.py are ours:
# they make vggt a regular package so this copy wins over any installed `vggt`.
