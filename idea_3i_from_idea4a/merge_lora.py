"""Merge an idea_4a LoRA checkpoint into the base Qwen3-VL weights.

idea_3i loads the LLM with from_pretrained() and only THEN calls initialize_vggt(), which
builds a fresh random vggt_projector. So the two halves of an idea_4a checkpoint have to
travel separately:

  * the LoRA adapters      -> merged into the base weights, written to --out
  * the trained projector  -> written to --out/vggt_projector.safetensors, loaded by
                              train.py AFTER initialize_vggt (see --init_projector)

Writing the projector into the merged model dir instead would silently do nothing: those
keys do not exist at from_pretrained() time and are dropped as unexpected.

    python idea_3i_from_idea4a/merge_lora.py \
        --adapter results/..._video_noise_lora/checkpoint-10000 \
        --out results/idea_4a_video_noise_merged10k
"""

from __future__ import annotations

import os

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

from safetensors.torch import save_file  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration  # noqa: E402

_VGGT = (os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
         "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True, help="idea_4a LoRA checkpoint dir")
    p.add_argument("--out", required=True)
    p.add_argument("--base", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--vggt-checkpoint", default=_VGGT)
    # Must match the run that produced the adapter, or the projector shapes will not line up.
    p.add_argument("--frame-num-latents", type=int, default=256)
    p.add_argument("--camera-num-latents", type=int, default=32)
    p.add_argument("--frame-widening-factor", type=int, default=2)
    p.add_argument("--vggt-embed-dim", type=int, default=2048)
    args = p.parse_args()

    from peft import PeftModel

    out = Path(args.out)
    processor = AutoProcessor.from_pretrained(args.base)
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.initialize_vggt(
        args.vggt_checkpoint,
        tokenizer=processor.tokenizer,
        vggt_embed_dim=args.vggt_embed_dim,
        frame_num_latents=args.frame_num_latents,
        camera_num_latents=args.camera_num_latents,
        frame_placeholder_token="<|quad_start|>",
        camera_placeholder_token="<|quad_end|>",
        frame_widening_factor=args.frame_widening_factor,
    )

    before = {k: v.detach().clone() for k, v in model.state_dict().items()
              if "language_model.layers.0.self_attn.q_proj" in k}

    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()

    # merge_and_unload() is a no-op if nothing matched; comparing a targeted weight is the
    # only way to tell a real merge from a silently empty one.
    after = {k: v for k, v in merged.state_dict().items() if k in before}
    changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
    if not changed:
        raise RuntimeError("merge changed no q_proj weight -- adapter did not apply")
    print(f"merge verified: {changed}/{len(before)} probed q_proj tensors changed")

    projector = {k: v.detach().cpu().contiguous()
                 for k, v in merged.state_dict().items() if "vggt_projector" in k}
    if not projector:
        raise RuntimeError("no vggt_projector tensors found after merge")

    out.mkdir(parents=True, exist_ok=True)
    # Strip the frozen VGGT encoder from what gets written: it is reloaded from
    # --vggt_checkpoint at train time and would otherwise add ~5 GB of dead weight.
    # Filter the state dict rather than clearing the attribute -- vggt_model is a
    # read-only property on this class.
    keep = {k: v for k, v in merged.state_dict().items()
            if k.split(".")[0] != "vggt_model" and "vggt_model." not in k}
    dropped = len(merged.state_dict()) - len(keep)
    print(f"dropping {dropped} frozen vggt_model tensors from the saved model")
    merged.save_pretrained(str(out), safe_serialization=True, state_dict=keep)
    processor.save_pretrained(str(out))
    save_file(projector, str(out / "vggt_projector.safetensors"))
    print(f"wrote merged model -> {out}")
    print(f"wrote {len(projector)} projector tensors -> {out / 'vggt_projector.safetensors'}")


if __name__ == "__main__":
    main()
