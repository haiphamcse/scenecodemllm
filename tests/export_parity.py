"""Re-export the first N clips of a mix with sr_vggt.encoders and compare with an existing cache
(the original runs' export). Run where that cache was made (V100, fp32 compute):

    python tests/export_parity.py <existing cache dir> <mix> <data root> <n> <cache config flags...>
"""

import json
import sys

import torch
from transformers import HfArgumentParser

from sr_vggt import config as cfg
from sr_vggt import encoders, vggt_cache
from sr_vggt.data import det_es_frames, video_path

existing, mix, data_root, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
(va,) = HfArgumentParser((cfg.VggtArguments,)).parse_args_into_dataclasses(sys.argv[5:])
params = va.cache_params()
agg = encoders.load_aggregator(va.vggt_backbone, va.checkpoint(), "cuda", torch.float32)
seen, ok = set(), True
for line in open(mix):
    row = json.loads(line)
    key = vggt_cache.clip_key(row, data_root)
    if key in seen:
        continue
    seen.add(key)
    old = vggt_cache.load(existing, key, params)  # also checks the stamped parameters match
    frames = det_es_frames(video_path(row, data_root))
    video = encoders.frames_to_video(frames, va.vggt_frames, va.vggt_image_resolution, va.vggt_backbone)
    new = encoders.encode(agg, video, "cuda", torch.float32, layers=va.vggt_layers)
    for k in old:
        d = (old[k].float() - new[k].to(old[k].dtype).float()).abs().max().item()
        ok &= d == 0
        print(f"{key} {k} {tuple(old[k].shape)} {old[k].dtype} max|diff| {d:.3g}")
    if len(seen) == n:
        break
print("EXPORT PARITY", "PASS" if ok else "FAIL")
