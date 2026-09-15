"""One Qwen3-VL-2B text forward+backward at a training-like length, in whatever env runs it.

Answers "does bumping torch / switching to flash-attn help on this GPU?" with a number,
for the kernel-bound part of a step. The joint run is ~68% dataloader time, so a kernel
gain of X% moves wall-clock by at most ~0.32 X.

  python profiling/bench_attn_env.py --attn sdpa
  python profiling/bench_attn_env.py --attn flash_attention_2
"""
import argparse, os, time, torch
from transformers import AutoModelForCausalLM, AutoConfig

ap = argparse.ArgumentParser()
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--seq", type=int, default=6800)   # census mean for the joint mix
ap.add_argument("--iters", type=int, default=8)
a = ap.parse_args()

name = "Qwen/Qwen3-VL-2B-Instruct"
cfg = AutoConfig.from_pretrained(name)
# Text decoder only: the attention/MLP stack that dominates the LLM's share of a step.
from transformers import Qwen3VLForConditionalGeneration
m = Qwen3VLForConditionalGeneration.from_pretrained(
    name, dtype=torch.bfloat16,
    attn_implementation=a.attn).cuda()
m.gradient_checkpointing_enable()
for p in m.parameters(): p.requires_grad_(False)
for n, p in m.named_parameters():
    if "q_proj" in n or "v_proj" in n: p.requires_grad_(True)   # LoRA-sized trainable set

ids = torch.randint(1000, 100000, (1, a.seq), device="cuda")
def step():
    out = m(input_ids=ids, labels=ids); out.loss.backward()
for _ in range(2): step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
for _ in range(a.iters): step()
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / a.iters
print(f"torch {torch.__version__} | attn={a.attn} | seq={a.seq} | {dt*1000:.0f} ms/step | peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
