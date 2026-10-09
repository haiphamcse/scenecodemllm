"""Parity of sr_vggt against the original experiment forks (finetuning/<fork>), on real data and a
real trained checkpoint. Not needed to use sr_vggt; kept as the record of how it was validated.

Each side runs in its own process (the forks' module names collide) and saves what it computed:
  train path: the collated batch of the first 2 mix rows (input_ids, labels, pixels, VGGT tensors),
              the fused Liger loss, and the logits at every supervised position, with the
              checkpoint's adapter loaded (eval mode, so dropout is off);
  eval path:  for the first 2 VSI-Bench full rows, every generate() input, the prefill logits of
              the last prompt position and the generated ids (VGGT run live, as in eval).

    python tests/parity.py old  <exp> <out dir>              (env: FT, SR_* from the original site)
    python tests/parity.py new  <exp> <out dir> <config flags...>
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
CACHES = Path(os.environ.get("FT_CACHES", ""))
DATA_ROOT = os.environ.get("FT_DATA_ROOT", "")
MIXES = {"190k": os.environ.get("FT_MIX_190K", ""), "traces": os.environ.get("FT_MIX_TRACES", "")}
EVAL_NEW_TOKENS = 32  # reason16k generates 8192 in a real eval; the first 32 tokens suffice here

# The original forks as their launchers / eval slurms ran them (independent of configs/*.env).
_P512 = dict(frame_num_latents=512, camera_num_latents=32, frame_widening_factor=2)
_EV_P = ["--frame_num_latents", "512", "--camera_num_latents", "32", "--frame_widening_factor", "2",
         "--vggt_image_resolution", "256", "--video_fps", "1.0", "--video_max_frames", "64"]
OLD = {
    "main": dict(fork="idea_3i_190k_es_64f_lat512_text256", kind="perceiver", proj=_P512,
                 cache="vggt_cache_256_64f_es_text", mix="190k", ev=_EV_P),
    "4b": dict(fork="idea_3i_190k_es_64f_lat512_text256_4b", kind="perceiver", proj=_P512,
               cache="vggt_cache_256_64f_es_text", mix="190k", ev=_EV_P, model="Qwen/Qwen3-VL-4B-Instruct"),
    "lat3072": dict(fork="idea_3i_190k_es_64f_lat3072_text256", kind="perceiver",
                    proj={**_P512, "frame_num_latents": 3072}, cache="vggt_cache_256_64f_es_text", mix="190k",
                    ev=["--frame_num_latents", "3072"] + _EV_P[2:]),
    "lora128": dict(fork="idea_3i_190k_es_64f_lat512_lora128", kind="perceiver", proj=_P512,
                    cache="vggt_cache_256_64f_es", mix="190k", ev=_EV_P),
    "vggt1b": dict(fork="idea_3i_190k_es_64f_lat512_vggt1b", kind="perceiver", proj=_P512,
                   cache="vggt_cache_256_64f_es_vggt1b", mix="190k", ev=_EV_P),
    "reason16k": dict(fork="idea_3i_190k_es_64f_lat512_text256_reason_cont", kind="perceiver", proj=_P512,
                      cache="vggt_cache_256_64f_es_text", mix="traces", ev=_EV_P,
                      run="idea_3i_190k_es_64f_lat512_text256_reason_cont_16k", step=569),
    "spatialstack": dict(fork="idea_3i_190k_es_64f_spatialstack", kind="spatialstack",
                         proj=dict(geometry_encoder_layers=[11, 17, 23], geometry_fusion_layers=[0, 1, 2],
                                   merger_hidden_dim=4096),
                         cache="vggt_cache_256_64f_es_ss", mix="190k",
                         ev=["--geometry_encoder_layers", "11", "17", "23", "--geometry_fusion_layers", "0", "1",
                             "2", "--merger_hidden_dim", "4096", "--video_fps", "1.0", "--video_max_frames", "64"]),
    "vlm3r": dict(fork="idea_3i_190k_es_64f_vlm3r", kind="vlm3r", proj=dict(fusion_num_heads=16, fusion_dropout=0.1),
                  cache="vggt_cache_256_64f_es", mix="190k",
                  ev=["--fusion_num_heads", "16", "--vggt_image_resolution", "256", "--video_fps", "1.0",
                      "--video_max_frames", "64"]),
}


def ckpt_of(exp: str) -> str:
    o = OLD[exp]
    return str(RESULTS / o.get("run", o["fork"]) / f"checkpoint-{o.get('step', 743)}")


def first_rows(mix: str, n: int = 2):
    rows = []
    with open(mix) as f:
        for line in f:
            row = json.loads(line)
            row.pop("meta", None)
            rows.append(row)
            if len(rows) == n:
                return rows


def import_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, (list, tuple)):
        return [to_cpu(v) for v in x]
    return x


@torch.no_grad()
def train_outputs(model, batch):
    """Fused (training) loss + logits at the supervised positions, eval mode."""
    dev = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model(**dev, skip_logits=True).loss.float().cpu()
    nolab = {k: v for k, v in dev.items() if k != "labels"}
    logits = model(**nolab).logits
    sel = (dev["labels"][:, 1:] != -100)
    return {"loss": loss, "sup_logits": logits[:, :-1][sel].float().cpu(),
            "argmax": logits.argmax(-1).cpu()}


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


def run_old(exp: str, out: Path) -> None:
    from peft import PeftModel
    from transformers import AutoProcessor

    o = OLD[exp]
    fork = FT / o["fork"]
    model_path = o.get("model", "Qwen/Qwen3-VL-2B-Instruct")
    # ---- train path: the fork's own train.py imports (collator, model, default VGGT ckpt) ----
    train = import_path("old_train", fork / "train.py")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    kw = dict(data_root=Path(DATA_ROOT), vggt_cache_root=CACHES / o["cache"],
              vggt_checkpoint=train._DEFAULT_VGGT_CHECKPOINT, video_fps=1.0, video_max_frames=64,
              image_patch_size=16)
    if o["kind"] != "spatialstack":
        kw["vggt_image_resolution"] = 256
    if o["kind"] == "perceiver":
        kw["frame_placeholder_text"] = "<|quad_start|>" * o["proj"]["frame_num_latents"]
        kw["camera_placeholder_text"] = "<|quad_end|>" * o["proj"]["camera_num_latents"]
    batch = train.make_collator(processor, **kw)(first_rows(MIXES[o["mix"]]))
    model = train.Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", dtype=torch.bfloat16)
    torch.manual_seed(0)
    init = dict(o["proj"], vggt_embed_dim=2048)
    if o["kind"] == "perceiver":
        init.update(tokenizer=processor.tokenizer, frame_placeholder_token="<|quad_start|>",
                    camera_placeholder_token="<|quad_end|>")
    model.initialize_projector(**init)
    model = PeftModel.from_pretrained(model, ckpt_of(exp)).cuda().eval()
    res = {"batch": {k: to_cpu(v) for k, v in batch.items()}, **train_outputs(model, batch)}
    del model
    torch.cuda.empty_cache()
    # ---- eval path: the fork's eval.py as its eval slurm ran it, first 2 rows ----
    for m in ("collator", "model_with_vggt", "vggt_cache", "export_vggt_features", "train"):
        sys.modules.pop(m, None)
    ev = import_path("old_eval", fork / "eval.py")
    store: list = []
    record_generate(store)
    sys.argv = ["eval.py", "--adapter_path", ckpt_of(exp), "--split", "full", "--output_dir", str(out / "old_eval"),
                "--hf_home", os.environ["HF_HOME"], "--dtype", "bf16", "--max_samples", "2",
                "--max_new_tokens", str(EVAL_NEW_TOKENS)] + o["ev"]
    ev.evaluate(ev.parse_args())
    res["eval"] = store
    torch.save(res, out / "old.pt")


def run_new(exp: str, out: Path, flags: list) -> None:
    from peft import PeftModel
    from transformers import AutoProcessor, HfArgumentParser

    from sr_vggt import config as cfg
    from sr_vggt import eval as sr_eval
    from sr_vggt.data import make_collator
    from sr_vggt.train import ModelArguments

    o = OLD[exp]
    ma, va, _ = HfArgumentParser((ModelArguments, cfg.VggtArguments)).parse_args_into_dataclasses(
        flags, return_remaining_strings=True)
    processor = AutoProcessor.from_pretrained(ma.model_name_or_path, trust_remote_code=True)
    fp, cp = va.placeholder_texts()
    collate = make_collator(processor, data_root=DATA_ROOT, prompt=va.prompt, frame_placeholder_text=fp,
                            camera_placeholder_text=cp, vggt_cache_root=os.environ["NEW_CACHE_ROOT"],
                            cache_params=va.cache_params(), batch_vggt=cfg.batch_vggt_fn(va))
    batch = collate(first_rows(MIXES[o["mix"]]))
    model = cfg.model_class(va).from_pretrained(ma.model_name_or_path, attn_implementation="sdpa",
                                                dtype=torch.bfloat16)
    torch.manual_seed(0)
    cfg.initialize_projector(model, va, processor.tokenizer)
    model = PeftModel.from_pretrained(model, ckpt_of(exp)).cuda().eval()
    res = {"batch": {k: to_cpu(v) for k, v in batch.items()}, **train_outputs(model, batch)}
    del model
    torch.cuda.empty_cache()
    store: list = []
    record_generate(store)
    sys.argv = ["eval.py"] + flags + ["--adapter_path", ckpt_of(exp), "--output_dir", str(out / "new_eval"),
                                      "--vsibench_videos", os.environ["SR_VSIBENCH"], "--max_samples", "2",
                                      "--max_new_tokens", str(EVAL_NEW_TOKENS), "--dtype", "bf16"]
    sr_eval.main()
    res["eval"] = store
    torch.save(res, out / "new.pt")


def diff(a, b, path, report, tol=0.0):
    """Walk two saved structures; record max |a-b| per tensor and any structural mismatch.
    tol > 0 (logits, loss): a few bf16 ulps of difference count as "close", not a failure."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.shape != b.shape:
            report.append((path, "SHAPE", f"{getattr(a, 'shape', a)} vs {getattr(b, 'shape', b)}"))
            return
        d = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        report.append((path, "ok" if d == 0 else ("close" if d <= tol else "DIFF"), f"{d:.3g}"))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                report.append((f"{path}.{k}", "MISSING", "old" if k not in a else "new"))
            else:
                diff(a[k], b[k], f"{path}.{k}", report, tol)
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
    rep: list = []
    diff(old["batch"], new["batch"], "batch", rep)
    diff(old["loss"], new["loss"], "loss", rep, tol=1e-3)
    diff(old["sup_logits"], new["sup_logits"], "sup_logits", rep, tol=0.5)
    diff(old["argmax"], new["argmax"], "argmax", rep)
    if len(old["eval"]) != len(new["eval"]):
        rep.append(("eval", "LEN", f"{len(old['eval'])} vs {len(new['eval'])}"))
    for i, (eo, en) in enumerate(zip(old["eval"], new["eval"])):
        io, inn = dict(eo["inputs"]), dict(en["inputs"])
        io.pop("raw_videos", None)  # old perceiver eval ran VGGT inside the model; new passes tokens
        if "vggt_tokens" in io:     # old VLM-3R eval handed bf16 GPU tokens, new fp32 (model casts)
            io["vggt_tokens"] = [t.bfloat16() for t in io["vggt_tokens"]]
            inn["vggt_tokens"] = [t.bfloat16() for t in inn["vggt_tokens"]]
        if "vggt_patch_tokens" not in io:  # (old SpatialStack eval passes the same lists as new)
            inn.pop("vggt_patch_tokens", None)
            inn.pop("vggt_camera_tokens", None)
        diff(io, inn, f"eval[{i}].inputs", rep)
        diff(eo["prefill_last"], en["prefill_last"], f"eval[{i}].prefill_logits", rep, tol=0.5)
        diff(eo["out"], en["out"], f"eval[{i}].generated", rep)
    bad = [r for r in rep if r[1] not in ("ok", "close")]
    worst = max((float(r[2]) for r in rep if r[1] in ("ok", "close", "DIFF")), default=0.0)
    summary = {"exp": exp, "pass": not bad, "checks": len(rep), "max_abs_diff": worst,
               "loss_old": old["loss"].item(), "loss_new": new["loss"].item(), "problems": bad[:20]}
    (out / "parity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    mode, exp, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "old":
        run_old(exp, out)
    elif mode == "new":
        run_new(exp, out, sys.argv[4:])
    else:
        sys.exit(compare(exp, out))
