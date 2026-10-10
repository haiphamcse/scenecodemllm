"""All checks of sr_vggt, in one file.

CPU (no GPU; only the Qwen processor/tokenizer in HF_HOME):
  1. train/eval input identity: the training collator and eval build the same Qwen pixels, grid,
     prompt tokens and VGGT input for the same video (synthetic mp4, odd frame count), every mode.
  2. live VGGT inside the model == the same tokens passed in directly, for all three fusion archs,
     with a tiny random VGGT-Omega aggregator and a tiny random Qwen3-VL.
  3. the ablations / reasoning self-checks.
GPU parity (Jean Zay, tests/parity.slurm): sr_vggt eval vs the original experiment forks' eval
(finetuning/<fork>/eval.py, which already decoded VSI-Bench video and ran VGGT live), on a trained
checkpoint and the first 2 VSI-Bench full rows: every generate() input, the prefill logits of the
last prompt position and the generated ids. Training cannot be compared this way: the original
runs trained on det_es frames + cached VGGT features, sr_vggt decodes the videos.

    HF_HOME=<hf home> HF_HUB_OFFLINE=1 python tests/checks.py                 (CPU checks)
    python tests/checks.py parity old  <exp> <out dir>              (env: FT, FT_RESULTS, HF_HOME)
    python tests/checks.py parity new  <exp> <out dir> <config flags...>   (env: SR_VSIBENCH)
    python tests/checks.py parity compare <exp> <out dir>
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sr_vggt import model as cfg  # noqa: E402  (importing sr_vggt puts third_party on sys.path)
from sr_vggt.data import clip_inputs, decode_video_frames, make_collator, task_messages  # noqa: E402

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
        frames = decode_video_frames(video, va.video_fps, va.video_max_frames)
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



# ================================ GPU parity ================================ #

FT = Path(os.environ.get("FT", ""))
RESULTS = Path(os.environ.get("FT_RESULTS", ""))
NEW_TOKENS = 32  # reason16k generates 8192 in a real eval; 32 suffice to compare

# The original forks as their eval slurms ran them (independent of configs/*.env).
_EV_P = ["--frame_num_latents", "512", "--camera_num_latents", "32", "--frame_widening_factor", "2",
         "--vggt_image_resolution", "256", "--video_fps", "1.0", "--video_max_frames", "64"]
OLD = {
    "main": dict(fork="idea_3i_190k_es_64f_lat512_clean", ev=_EV_P),
    "lora128": dict(fork="idea_3i_190k_es_64f_lat512_lora128", ev=_EV_P),
    "text256": dict(fork="idea_3i_190k_es_64f_lat512_text256", ev=_EV_P),
    "4b": dict(fork="idea_3i_190k_es_64f_lat512_text256_4b", ev=_EV_P),
    "lat3072": dict(fork="idea_3i_190k_es_64f_lat3072_text256", ev=["--frame_num_latents", "3072"] + _EV_P[2:]),
    "vggt1b": dict(fork="idea_3i_190k_es_64f_lat512_vggt1b", ev=_EV_P),
    "reason16k": dict(fork="idea_3i_190k_es_64f_lat512_text256_reason_cont", ev=_EV_P,
                      run="idea_3i_190k_es_64f_lat512_text256_reason_cont_16k", step=569),
    "spatialstack": dict(fork="idea_3i_190k_es_64f_spatialstack",
                         ev=["--geometry_encoder_layers", "11", "17", "23", "--geometry_fusion_layers", "0", "1",
                             "2", "--merger_hidden_dim", "4096", "--video_fps", "1.0", "--video_max_frames", "64"]),
    "vlm3r": dict(fork="idea_3i_190k_es_64f_vlm3r",
                  ev=["--fusion_num_heads", "16", "--vggt_image_resolution", "256", "--video_fps", "1.0",
                      "--video_max_frames", "64"]),
}


def ckpt_of(exp: str) -> str:
    o = OLD[exp]
    return str(RESULTS / o.get("run", o["fork"]) / f"checkpoint-{o.get('step', 743)}")


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return [to_cpu(v) for v in x]
    return x


def record_generate(store: list):
    """Patch PeftModel.generate: keep every input, the prefill logits and the output ids."""
    from peft import PeftModel

    orig = PeftModel.generate

    def generate(self, **kw):
        out = orig(self, **kw)
        logits = self(**{k: v for k, v in kw.items() if k not in ("max_new_tokens", "do_sample")}).logits
        store.append({"inputs": {k: to_cpu(v) for k, v in kw.items()},
                      "prefill_last": logits[0, -1].float().cpu(), "out": out.cpu()})
        return out

    PeftModel.generate = generate


def run(side: str, exp: str, out: Path, flags: list) -> None:
    store: list = []
    record_generate(store)
    common = ["--adapter_path", ckpt_of(exp), "--max_samples", "2", "--max_new_tokens", str(NEW_TOKENS),
              "--dtype", "bf16", "--output_dir", str(out / f"{side}_eval")]
    if side == "old":
        spec = importlib.util.spec_from_file_location("old_eval", FT / OLD[exp]["fork"] / "eval.py")
        ev = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ev
        spec.loader.exec_module(ev)
        sys.argv = ["eval.py", "--split", "full", "--hf_home", os.environ["HF_HOME"]] + common + OLD[exp]["ev"]
        ev.evaluate(ev.parse_args())
    else:
        from sr_vggt import eval as sr_eval
        sys.argv = ["eval.py"] + flags + common + ["--vsibench_dir", os.environ["SR_VSIBENCH"]]
        sr_eval.main()
    torch.save(store, out / f"{side}.pt")


def diff(a, b, path, report, tol=0.0):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.shape != b.shape:
            report.append((path, "SHAPE", f"{getattr(a, 'shape', a)} vs {getattr(b, 'shape', b)}"))
            return
        d = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        report.append((path, "ok" if d == 0 else ("close" if d <= tol else "DIFF"), f"{d:.3g}"))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            report.append((path, "LEN", f"{len(a)} vs {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", report, tol)
    elif a != b:
        report.append((path, "VALUE", f"{a!r} vs {b!r}"))


def compare(exp: str, out: Path) -> int:
    old, new = torch.load(out / "old.pt", weights_only=False), torch.load(out / "new.pt", weights_only=False)
    rep = [] if len(old) == len(new) else [("rows", "LEN", f"{len(old)} vs {len(new)}")]
    for i, (o, n) in enumerate(zip(old, new)):
        for k in ("input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"):
            diff(o["inputs"][k], n["inputs"][k], f"row{i}.{k}", rep)
        if "raw_videos" in o["inputs"]:  # Perceiver forks: the same VGGT input tensor
            diff(o["inputs"]["raw_videos"], n["inputs"]["raw_videos"], f"row{i}.raw_videos", rep)
        diff(o["prefill_last"], n["prefill_last"], f"row{i}.prefill_logits", rep, tol=0.5)
        diff(o["out"], n["out"], f"row{i}.generated", rep)
    bad = [r for r in rep if r[1] not in ("ok", "close")]
    summary = {"exp": exp, "pass": not bad, "checks": len(rep),
               "max_logit_diff": max((float(r[2]) for r in rep if "logits" in r[0]), default=0.0),
               "problems": bad[:20]}
    (out / "parity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if not bad else 1


def cpu_checks():
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
    runpy.run_module("sr_vggt.ablations", run_name="__main__")
    runpy.run_module("sr_vggt.reasoning", run_name="__main__")
    print("all CPU checks passed")


if __name__ == "__main__":
    if sys.argv[1:2] == ["parity"]:
        mode, exp, out = sys.argv[2], sys.argv[3], Path(sys.argv[4])
        out.mkdir(parents=True, exist_ok=True)
        if mode == "compare":
            sys.exit(compare(exp, out))
        run(mode, exp, out, sys.argv[5:])
    else:
        cpu_checks()
