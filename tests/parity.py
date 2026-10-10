"""GPU: sr_vggt eval vs the original experiment forks' eval (finetuning/<fork>/eval.py, which
already decoded VSI-Bench video and ran VGGT live), on a trained checkpoint and the first 2
VSI-Bench full rows. For both, every generate() input, the prefill logits of the last prompt
position and the generated ids are recorded and compared. Training cannot be compared this way:
the original runs trained on det_es frames + cached VGGT features, sr_vggt decodes the videos.

    python tests/parity.py old <exp> <out dir>              (env: FT, FT_RESULTS, HF_HOME)
    python tests/parity.py new <exp> <out dir> <config flags...> (env: SR_VSIBENCH)
    python tests/parity.py compare <exp> <out dir>
tests/parity.slurm drives all three per experiment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

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


if __name__ == "__main__":
    mode, exp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "compare":
        sys.exit(compare(exp, out))
    run(mode, exp, out, sys.argv[4:])
