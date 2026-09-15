"""CPU check of the det_es collator on 2 local VSI-590K rows (scannet + arkitscenes det_es frames exist locally):
frames come from det_es, T == det_es frame count == cached VGGT T, the processor resize is a no-op (collator assert),
and the supervised span is the answer. VGGT entries are random tensors written with the run's params (frames=det_es).

  python idea_3i_180k_es_64f_lat512/test_collator_es.py
"""
import glob
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vggt_cache  # noqa: E402
from collator import IGNORE_INDEX, det_es_frames, make_collator  # noqa: E402

ROOT = "/scratch/ducpham/Working/dataset/vsi_590k/VSI-590K"
rows = []
for src in ("scannet", "arkitscenes"):
    v = sorted(glob.glob(f"{ROOT}/{src}/*/det_es/det.json"))[0].split("/det_es/")[0] + ".mp4"
    rows.append({"task": "vqa", "video": v, "source": "vsi590k", "question_type": "object_counting",
                 "conversations": [{"from": "human", "value": "<image>\nHow many chairs are there?"}, {"from": "gpt", "value": "3"}]})

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
with tempfile.TemporaryDirectory() as d:
    params = vggt_cache.params_from(256, 1.0, 64, "ckpt")
    assert params["frames"] == "det_es"
    for r in rows:
        T = len(det_es_frames(r["video"]))
        vggt_cache.save(d, vggt_cache.clip_key(r), torch.randn(T, 252, 8), torch.randn(T, 17, 8), params)
    collate = make_collator(processor, data_root=Path(ROOT), vggt_cache_root=Path(d), vggt_checkpoint="ckpt",
                            frame_placeholder_text="<|quad_start|>" * 512, camera_placeholder_text="<|quad_end|>" * 32,
                            vggt_image_resolution=256, video_fps=1.0, video_max_frames=64)
    for r in rows:
        frames = det_es_frames(r["video"])
        batch = collate([r])
        patch, camera = batch["vggt_patch_tokens"][0], batch["vggt_camera_tokens"][0]
        assert patch.shape[0] == camera.shape[0] == len(frames), (patch.shape, len(frames))
        thw = batch["video_grid_thw"][0].tolist()
        assert thw[1] * 16 == frames[0].size[1] and thw[2] * 16 == frames[0].size[0], (thw, frames[0].size)
        sup = processor.tokenizer.decode(batch["labels"][0][batch["labels"][0] != IGNORE_INDEX])
        assert "3" in sup, sup
        print(f"{Path(r['video']).parent.name}: {len(frames)} det_es frames {frames[0].size}, grid_thw {thw}, "
              f"input_ids {tuple(batch['input_ids'].shape)}, patch {tuple(patch.shape)}, supervised {sup!r}")
    # old-cache entry (no frames tag) must be refused
    vggt_cache.save(d, "/x/y.mp4", torch.randn(2, 4, 8), torch.randn(2, 17, 8), dict(params, frames="video"))
    try:
        vggt_cache.load(d, "/x/y.mp4", params); raise AssertionError("frames mismatch must raise")
    except ValueError:
        pass
print("collator det_es test ok")
