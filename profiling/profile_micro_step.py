"""Wall-clock attribution for ONE micro-step of the idea_3i_590k joint / cold runs.

Read-only: builds the same model + collator the train scripts build, feeds it real rows,
and times the pieces. Writes nothing into any results/idea_3i_590k* dir.

  python profiling/profile_micro_step.py --variant joint --n 8
  python profiling/profile_micro_step.py --variant cold  --n 8 --phase cpu

phases
  cpu  per-sample collate cost, split decode / VGGT-resize / LLM-resize / tokenise+processor
  gpu  per-micro-step forward+backward, split VGGT / Perceiver / Qwen-vision / Qwen-LLM / backward
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sys_path(variant: str) -> None:
    for p in (ROOT / "common", ROOT / ("idea_3i_590k_joint" if variant == "joint" else "idea_3i_590k")):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


class T:
    """Accumulating named timer. cuda=True syncs, so the numbers are attributable."""

    def __init__(self, cuda: bool = False):
        self.acc: dict[str, float] = {}
        self.cuda = cuda

    def sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    def add(self, name: str, dt: float):
        self.acc[name] = self.acc.get(name, 0.0) + dt

    def wrap(self, name: str, fn):
        def inner(*a, **kw):
            self.sync()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            self.sync()
            self.add(name, time.perf_counter() - t0)
            return out
        return inner

    def report(self, n: int, total_key: str | None = None):
        rows = sorted(self.acc.items(), key=lambda kv: -kv[1])
        tot = self.acc.get(total_key, sum(v for k, v in rows if k != total_key))
        for k, v in rows:
            print(f"  {k:28} {v / n:8.3f} s/sample  {100 * v / tot:6.1f}%")
        print(f"  {'(reference total)':28} {tot / n:8.3f} s/sample")


def load_rows(variant: str, n: int, seed: int) -> list[dict]:
    if variant == "joint":
        path = os.environ["SR_MIX"]
        rows = [json.loads(l) for l in open(path)]
    else:
        path = os.environ["SR_JSONL"]
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("video")]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def build_collator(variant: str, processor, data_root: Path):
    import collator as C
    kw = dict(
        frame_placeholder_text="<|quad_start|>" * 256,
        camera_placeholder_text="<|quad_end|>" * 32,
        video_fps=1.0, video_max_frames=32, image_patch_size=16,
    )
    if variant == "joint":
        return C.make_collator(processor, data_root=data_root, cache_root=None,
                               box_noise=0.005, color_jitter=0.5,
                               vggt_image_resolution=256, **kw), C
    return C.make_collator(processor, data_root=data_root, **kw), C


def phase_cpu(variant: str, rows: list[dict], processor, data_root: Path):
    collate, C = build_collator(variant, processor, data_root)
    t = T(cuda=False)

    # Instrument the collator's module-level helpers in place.
    C.process_vision_info = t.wrap("llm_resize(process_vision_info)", C.process_vision_info)
    if variant == "joint":
        C.decode_video_frames = t.wrap("decord_decode", C.decode_video_frames)
        C.extract_raw_video_tensors = t.wrap("vggt_resize", C.extract_raw_video_tensors)
        C.load_example_video = t.wrap("det_jpeg_open", C.load_example_video)
    orig_proc = processor.__call__

    per_row, tasks = [], []
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        collate([row])
        dt = time.perf_counter() - t0
        per_row.append(dt)
        tasks.append(row.get("task", "vqa"))
        t.add("TOTAL_collate", dt)
        print(f"    row {i:3} task={tasks[-1]:4} {dt:7.2f} s", flush=True)

    print(f"\n[cpu] {variant}: {len(rows)} rows, single-threaded")
    t.report(len(rows), total_key="TOTAL_collate")
    print(f"  per-row min/med/max: {min(per_row):.2f} / "
          f"{sorted(per_row)[len(per_row) // 2]:.2f} / {max(per_row):.2f} s")
    if variant == "cold":
        # cold's decode and resize are fused inside process_vision_info; time the decode
        # alone on the same videos to get its share.
        import decord, numpy as np
        acc = 0.0
        for row in rows:
            p = row["video"]
            p = p if os.path.isabs(p) else str(data_root / p)
            t0 = time.perf_counter()
            r = decord.VideoReader(p, num_threads=1)
            idx = np.linspace(0, len(r) - 1,
                              max(1, min(int(round(len(r) / (r.get_avg_fps() or 30.0))), 32,
                                         len(r)))).round().astype(int)
            r.get_batch(idx).asnumpy()
            acc += time.perf_counter() - t0
        print(f"  decord_decode (measured separately) {acc / len(rows):8.3f} s/sample")
    return per_row, tasks


def phase_gpu(variant: str, rows: list[dict], processor, data_root: Path, model_path: str):
    import torch
    from peft import LoraConfig, get_peft_model
    from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

    collate, _ = build_collator(variant, processor, data_root)
    print("[gpu] pre-collating batches on CPU (excluded from GPU timing)...", flush=True)
    batches = [collate([r]) for r in rows]

    vggt_ckpt = (os.environ["SR_HF_HOME"] + "/hub/models--facebook--VGGT-Omega/snapshots/"
                 "05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt")
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", dtype=torch.bfloat16)
    model.initialize_vggt(vggt_ckpt, tokenizer=processor.tokenizer, vggt_embed_dim=2048,
                          frame_num_latents=256, camera_num_latents=32,
                          frame_placeholder_token="<|quad_start|>",
                          camera_placeholder_token="<|quad_end|>", frame_widening_factor=2)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=64, lora_alpha=128, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["vggt_projector"]))
    model.cuda().train()
    # the frozen encoder stays in eval, as the real run keeps it
    inner = model.base_model.model.model
    inner.vggt_model.eval()

    t = T(cuda=True)
    inner._extract_vggt_tokens = t.wrap("vggt_encoder", inner._extract_vggt_tokens)
    inner.vggt_projector.forward = t.wrap("perceiver", inner.vggt_projector.forward)
    inner.get_video_features = t.wrap("qwen_vision_tower", inner.get_video_features)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def one(batch, record: bool):
        raw = [v.cuda() for v in batch["raw_videos"]]
        kw = {k: (v.cuda() if hasattr(v, "cuda") else v)
              for k, v in batch.items() if k != "raw_videos"}
        kw["raw_videos"] = raw
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = model(**kw)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        if record:
            t.add("FORWARD_total", t1 - t0)
            t.add("backward", t2 - t1)
            t.add("MICROSTEP_total", t2 - t0)
        model.zero_grad(set_to_none=True)
        return batch["input_ids"].shape[1], raw[0].shape

    print("[gpu] warmup x2", flush=True)
    for b in batches[:2]:
        one(b, record=False)
    t.acc.clear()

    print(f"[gpu] timing {len(batches)} micro-steps", flush=True)
    for i, b in enumerate(batches):
        seq, shp = one(b, record=True)
        print(f"    micro {i:3} task={rows[i].get('task','vqa'):4} seq={seq:6} "
              f"raw_video={tuple(shp)}", flush=True)

    n = len(batches)
    print(f"\n[gpu] {variant}: per micro-step (batch 1), bf16, gradient_checkpointing off")
    t.report(n, total_key="MICROSTEP_total")
    fwd = t.acc["FORWARD_total"]
    known = t.acc["vggt_encoder"] + t.acc["perceiver"] + t.acc["qwen_vision_tower"]
    print(f"  {'qwen_llm_fwd (by subtraction)':28} {(fwd - known) / n:8.3f} s/sample")

    # optimizer step: paid once per 64 micro-steps
    for b in batches[:1]:
        one(b, record=False)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    opt.step()
    torch.cuda.synchronize()
    print(f"  optimizer_step (once per 64)  {time.perf_counter() - t0:8.3f} s")
    print(f"  peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def phase_e2e(variant, n_rows, seed, processor, data_root, model_path, workers, steps):
    """The real DataLoader feeding the real model on one GPU. Answers: loader or GPU?"""
    import torch
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model
    from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

    rows = load_rows(variant, n_rows, seed)
    collate, _ = build_collator(variant, processor, data_root)

    def mk_loader():
        return DataLoader(rows, batch_size=1, shuffle=False, num_workers=workers,
                          collate_fn=collate, prefetch_factor=2, persistent_workers=False)

    # ---- loader alone: how fast can `workers` workers actually deliver? ----
    it = iter(mk_loader())
    for _ in range(min(steps // 4, 15)):
        next(it)                       # fill the pipeline / warm the page cache
    t0 = time.perf_counter()
    for _ in range(steps):
        next(it)
    dt = time.perf_counter() - t0
    print(f"\n[e2e] loader-only, {workers} workers: {dt / steps:.3f} s/sample "
          f"({steps / dt:.2f} samples/s)", flush=True)
    del it

    # ---- loader + model ----
    vggt_ckpt = (os.environ["SR_HF_HOME"] + "/hub/models--facebook--VGGT-Omega/snapshots/"
                 "05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt")
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_path, attn_implementation="sdpa", dtype=torch.bfloat16)
    model.initialize_vggt(vggt_ckpt, tokenizer=processor.tokenizer, vggt_embed_dim=2048,
                          frame_num_latents=256, camera_num_latents=32,
                          frame_placeholder_token="<|quad_start|>",
                          camera_placeholder_token="<|quad_end|>", frame_widening_factor=2)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=64, lora_alpha=128, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["vggt_projector"]))
    model.cuda().train()
    model.base_model.model.model.vggt_model.eval()

    def run(batch):
        kw = {k: (v.cuda() if hasattr(v, "cuda") else v)
              for k, v in batch.items() if k != "raw_videos"}
        kw["raw_videos"] = [v.cuda() for v in batch["raw_videos"]]
        out = model(**kw)
        out.loss.backward()          # grads accumulate, as the real run does

    it = iter(mk_loader())
    for _ in range(4):
        run(next(it))
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)

    gpu_busy = 0.0
    wait = 0.0
    t0 = time.perf_counter()
    for i in range(steps):
        ta = time.perf_counter()
        batch = next(it)
        tb = time.perf_counter()
        run(batch)
        torch.cuda.synchronize()
        tc = time.perf_counter()
        wait += tb - ta
        gpu_busy += tc - tb
        if i % 10 == 0:
            print(f"    step {i:3} wait={tb - ta:6.3f} gpu={tc - tb:6.3f}", flush=True)
    total = time.perf_counter() - t0
    print(f"\n[e2e] {variant}: {steps} micro-steps, 1 GPU, {workers} loader workers")
    print(f"  wall            {total / steps:7.3f} s/micro-step")
    print(f"  waiting on data {wait / steps:7.3f} s ({100 * wait / total:5.1f}%)")
    print(f"  gpu fwd+bwd     {gpu_busy / steps:7.3f} s ({100 * gpu_busy / total:5.1f}%)")
    print(f"  peak_mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["joint", "cold"], required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", choices=["cpu", "gpu", "e2e", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.environ["SR_HF_HOME"])
    _sys_path(args.variant)
    from transformers import AutoProcessor

    data_root = Path(os.environ["SR_DATA_ROOT"])
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    rows = load_rows(args.variant, args.n, args.seed)
    print(f"=== {args.variant} | {len(rows)} rows | data_root={data_root} ===", flush=True)

    if args.phase in ("cpu", "all"):
        phase_cpu(args.variant, rows, processor, data_root)
    if args.phase in ("gpu", "all"):
        phase_gpu(args.variant, rows, processor, data_root, args.model)
    if args.phase == "e2e":
        phase_e2e(args.variant, args.n, args.seed, processor, data_root, args.model,
                  args.workers, args.steps)


if __name__ == "__main__":
    main()
