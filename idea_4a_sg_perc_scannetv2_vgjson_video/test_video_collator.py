"""Check the collator still supervises exactly the graph span once video tokens exist.

The label bounds are computed from PLAIN-tokenizer lengths and shifted by
``delta = expanded_len - unexpanded_len``. Adding the video makes that delta jump by
~600-2400 tokens, so if the expansion were not entirely inside the user turn the
supervised span would slide off the target silently -- loss would still look sane.

    python idea_4a_sg_perc_scannetv2_vgjson_video/test_video_collator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collator import IGNORE_INDEX, make_collator  # noqa: E402

_ANN = Path("/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/VG-LLM/"
            "data/evaluation/threedod_1perscene/scannet/scannet_det_val_4frames.json")
_DATA = Path("/home/ducpham/scratch/Working/dataset")
_BASE = "Qwen/Qwen3-VL-2B-Instruct"


def main():
    from graph_vgllm import canonicalize

    processor = AutoProcessor.from_pretrained(_BASE)
    tok = processor.tokenizer

    rows = []
    for raw in json.load(open(_ANN))[:2]:
        rows.append({
            "images": [str(_DATA / p) for p in raw["images"]],
            "graph": canonicalize(raw["conversations"][1]["value"]),
            "scene": raw["images"][0].split("/")[-2],
        })

    collate = make_collator(
        processor=processor,
        data_root=_DATA,
        cache_root=_DATA,
        frame_placeholder_text="<|quad_start|>" * 256,
        camera_placeholder_text="<|quad_end|>" * 32,
        max_graph_tokens=4096,
        image_patch_size=16,
    )
    batch = collate(rows)

    assert "pixel_values_videos" in batch, "video never reached the processor"
    assert "raw_videos" in batch and len(batch["raw_videos"]) == len(rows)
    print(f"seq len {batch['input_ids'].shape[1]} | video grid {batch['video_grid_thw'].tolist()}")

    # VGGT keeps native frames; the MLLM copy must be a DIFFERENT (smaller) resize, else
    # the budgeted branch silently inherited native resolution.
    native = batch["raw_videos"][0].shape
    n_video_tok = int(batch["video_grid_thw"].prod(dim=1).sum()) // 4  # 2x2 spatial merge
    print(f"VGGT tensor {tuple(native)} | MLLM video tokens ~{n_video_tok}")
    assert native[-1] >= 1280, f"VGGT input was resized away from native: {native}"

    for i, row in enumerate(rows):
        sup = batch["labels"][i][batch["labels"][i] != IGNORE_INDEX]
        assert sup.numel() > 0, f"row {i}: nothing supervised"
        decoded = tok.decode(sup, skip_special_tokens=True)
        target = row["graph"].strip()
        # The supervised span must BE the target, not merely overlap it.
        assert decoded.strip().startswith(target[:40]), (
            f"row {i}: supervised span starts wrong\n  got: {decoded[:120]!r}\n"
            f"  want: {target[:120]!r}"
        )
        assert target[-30:] in decoded, f"row {i}: supervised span truncated before target end"
        print(f"row {i}: {sup.numel()} supervised tokens, span matches target")

    # Nothing in the prompt may be supervised: video pads land there.
    vid_id = tok.convert_tokens_to_ids("<|video_pad|>")
    if vid_id is not None and vid_id >= 0:
        overlap = ((batch["input_ids"] == vid_id) & (batch["labels"] != IGNORE_INDEX)).sum()
        assert overlap == 0, f"{overlap} video tokens are supervised"
        print("no video tokens are supervised")

    print("\nok: video present, VGGT native, graph span exactly supervised")


if __name__ == "__main__":
    main()
