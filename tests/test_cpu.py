"""CPU checks (no GPU, no weights beyond the Qwen processor/tokenizer in HF_HOME):

1. train/eval input identity: the training collator and eval build the same Qwen pixels, grid and
   VGGT input for the same video (synthetic mp4, odd frame count) under every frame mode.
2. live VGGT inside the model == the same tokens passed in directly, for all three fusion archs,
   with a tiny random VGGT-Omega aggregator and a tiny random Qwen3-VL.
3. the spatialstack merger / reasoning parser self-checks.

    HF_HOME=<hf home> HF_HUB_OFFLINE=1 python tests/test_cpu.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sr_vggt import config as cfg  # noqa: E402  (also puts third_party on sys.path)
from sr_vggt import encoders  # noqa: E402
from sr_vggt.data import clip_inputs, make_collator, task_messages  # noqa: E402

QWEN = "Qwen/Qwen3-VL-2B-Instruct"


def synthetic_video(path, n=45, fps=15, size=(320, 240)):
    import cv2

    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for _ in range(n):  # 3 s at 15 fps -> 3 frames at 1 fps (odd T)
        w.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    w.release()


def check_train_eval_inputs(processor, video):
    row = {"video": os.path.basename(video), "task": "vqa",
           "conversations": [{"value": "How many chairs?"}, {"value": "3"}]}
    for frames_mode, arch, prompt, backbone in [("balanced", "perceiver", "clean", "omega"),
                                                ("balanced", "perceiver", "text256", "vggt1b"),
                                                ("balanced", "vlm3r", "plain", "omega"),
                                                ("native", "spatialstack", "plain", "omega")]:
        va = cfg.VggtArguments(arch=arch, prompt=prompt, vggt_frames=frames_mode, vggt_backbone=backbone,
                               vggt_layers=[0, 1], frame_num_latents=8, camera_num_latents=4)
        train = make_collator(processor, os.path.dirname(video), va)([row])
        # eval.py's path for the same row
        frames = encoders.decode_video_frames(video, va.video_fps, va.video_max_frames)
        fp, cp = va.placeholder_texts()
        msgs = task_messages(row, frames, va.prompt, fp, cp, target_text="", prompt_only=True)
        videos, meta, kw, raw = clip_inputs(va, msgs, frames)
        ev = processor(text=processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                       videos=videos, video_metadata=meta, return_tensors="pt", padding=True, **kw)
        assert torch.equal(train["pixel_values_videos"], ev["pixel_values_videos"])
        assert torch.equal(train["video_grid_thw"], ev["video_grid_thw"])
        assert torch.equal(train["raw_videos"][0], raw)
        n_prompt = ev["input_ids"].shape[1]
        assert torch.equal(train["input_ids"][0, :n_prompt], ev["input_ids"][0]), "prompt tokens differ"
        print(f"  {arch}/{prompt}/{backbone}/{frames_mode}: T={len(frames)} raw {tuple(raw.shape)} "
              f"grid {train['video_grid_thw'].tolist()} prompt {n_prompt} tok: identical")


def check_live_vggt(tok):
    from transformers import Qwen3VLConfig
    from vggt_omega.models.aggregator import Aggregator

    qcfg = Qwen3VLConfig(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, vocab_size=len(tok), head_dim=16,
                         rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}),
        vision_config=dict(hidden_size=32, intermediate_size=64, depth=2, num_heads=2, out_hidden_size=64,
                           deepstack_visual_indexes=[0, 1], patch_size=16, temporal_patch_size=2,
                           spatial_merge_size=2))
    ids_of = tok.convert_tokens_to_ids
    T, grid = 5, torch.tensor([[3, 4, 4]])          # 5 frames, padded to 3 temporal groups
    video = torch.rand(T, 3, 64, 64)                # native: 64x64 -> 4x4 patches = Qwen's grid
    for arch, extra in [("perceiver", {}), ("vlm3r", {}),
                        ("spatialstack", dict(vggt_layers=[0, 1], geometry_encoder_layers=[0, 1],
                                              geometry_fusion_layers=[0, 2], merger_hidden_dim=32))]:
        va = cfg.VggtArguments(arch=arch, vggt_embed_dim=128, frame_num_latents=8, camera_num_latents=4, **extra)
        torch.manual_seed(0)
        agg = Aggregator(patch_size=16, embed_dim=64, depth=2, num_heads=4, register_attention_block_indices=[1],
                         cached_layer_indices=(0, 1))
        with torch.no_grad():  # the real weights come from the checkpoint; some tensors start uninitialised
            for p in agg.parameters():
                p.normal_(0, 0.02)
            for b in agg.buffers():
                b.nan_to_num_(1.0)  # qkv bias masks
        model = cfg.model_class(va)(qcfg)
        cfg.initialize_vggt_modules(model, va, tok, aggregator=agg)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "vggt_projector" in n:
                    p.copy_(torch.randn_like(p) * 0.05)  # zero-init branches must matter here
        model.eval()
        ids = [151644, 872, 198] + ([ids_of("<|quad_start|>")] * 8 + [ids_of("<|quad_end|>")] * 4
                                    if arch == "perceiver" else [])
        ids += [ids_of("<|vision_start|>")] + [ids_of("<|video_pad|>")] * 12 + [ids_of("<|vision_end|>"), 1000, 151645]
        ids = torch.tensor([ids])
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(), video_grid_thw=grid,
                     pixel_values_videos=torch.randn(3 * 4 * 4, 3 * 2 * 16 * 16))
        with torch.no_grad():
            live = model(**batch, raw_videos=[video])
            direct = model(**batch, **model.model.vggt_features([video]))
            other = model(**batch, raw_videos=[torch.rand_like(video)])  # different geometry input
        assert not live.logits.isnan().any()
        assert torch.equal(live.logits, direct.logits), arch
        assert not torch.equal(live.logits, other.logits), f"{arch}: VGGT input had no effect"
        print(f"  {arch}: live VGGT == direct tokens (loss {live.loss.item():.6f}); VGGT changes the output")


if __name__ == "__main__":
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(QWEN)
    print("1. train/eval inputs")
    with tempfile.TemporaryDirectory() as d:
        synthetic_video(os.path.join(d, "clip.mp4"))
        check_train_eval_inputs(processor, os.path.join(d, "clip.mp4"))
    print("2. live VGGT in the model")
    check_live_vggt(processor.tokenizer)
    print("3. self-checks")
    import runpy
    runpy.run_module("sr_vggt.ablations.spatialstack", run_name="__main__")
    runpy.run_module("sr_vggt.extensions.reasoning", run_name="__main__")
    print("all CPU checks passed")
