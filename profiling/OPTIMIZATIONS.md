# Optimizations applied 2026-09-09/10, in order, and what each bought

Companion to `RESULTS.md`. "Measured" = a number read off a job; *est.* = arithmetic.
Baseline for the joint run: 2x H100, 193 s/it, `GLOBAL_BATCH` 256 (accum 128).

| # | change | kind | measured effect |
|---|---|---|---|
| 1 | 2 -> 4 H100 (`jz_train_joint_4gpu.sh`, accum 128 -> 64) | throughput | joint 193 -> ~103 s/it, cold 173 -> ~89, multiscale 234 -> ~122. ~1.8x for 2x GPUs. Per micro-step slightly *worse* (1.51 -> 1.69 s): not compute-bound, just more ranks. Same 224k samples/step either way. |
| 2 | `save_steps` 100 -> 25 | wasted work | no s/it change; caps work lost at a 2 h wall from <=99 steps to <=24. Cost ~40 min/window at 2 GPUs (steps 1200-1213 redone). |
| 3 | frame cache extraction moved A100 -> 40-core CPU job | GPU-hours | 5963 videos, 13 GB, 11 min, 0 GPU. **Still has no consumer** (`collator.py:137` calls decord; `train.py` passes `cache_root=None`). 0 s/it so far. |
| 4 | profiling (`RESULTS.md`) | diagnosis | joint step = 32 s GPU + 70 s other (68% non-GPU); `get_batch_samples` fetch burst. Proposed `--dataloader_prefetch_factor 4-8` (*est.* -24..27%) and dropping `num_threads=1` -- **not applied**. |
| 5 | `vsibench_eval_steps` 200 -> 400, then eval **off** for cached windows | window budget | one eval = 44 min at 4 GPUs (12% of a window at 200). At 2 GPUs a window cannot hold 25 steps + eval, so "off" was the difference between passing step 1200 and never. |
| 6 | FSDP fork for V100 (`common/fsdp.yaml`, FULL_SHARD) | fit | enables 32 GB cards. 452 s/it at 4x V100 GB 64 (28 s/micro-step); 1029 s/it at 8x V100 GB 256, 16 frames, VGGT live. Speed is a floor, not a gain: V100 fp32 is ~4x slower per sample than H100 bf16. |
| 7 | VGGT/Perceiver activation checkpointing via FSDP `NO_REENTRANT` (perceiver-io's fairscale one crashes under FSDP: resharded params) | memory | -1 GiB at the OOM point. Recompute cost not isolated. |
| 8 | Liger fused linear cross-entropy (forward port + `--use_liger_kernel`) | memory | removes the fp32 logits `[seq, 151936]` and their grad: 4.7 GiB each at seq 8300. Not the binding term on V100. Speed not isolated. |
| 9 | CPU-resident batches (`get_train_dataloader` with `device_placement=False`) | memory | -12.46 GiB at 2x V100 accum 32 (memory snapshot); ~32 GB at 2x H100 accum 128 (77.5 GiB OOM -> runs). s/it 183 vs ~190: **no slowdown**. |
| 10 | `use_gqa_in_sdpa=False` on sm<80 (forces `repeat_kv`; mem-efficient SDPA refuses GQA on torch 2.6 *and* 2.8) | memory | the fix that made V100 train: removes 4 x 4.09 GiB of math-backend attention scratch at seq 8286. Before it no 32-frame step ever completed on V100. |
| 11 | `ddp_timeout` 7200 -> 900 + NCCL flight recorder | failure latency | a hang costs 15 min, not 2 h (job 1962056 sat 65 min silent). |
| 12 | host RAM: workers 4 -> 2, `pin_memory` off, prefetch 1, `fsdp_cpu_ram_efficient_loading` | fit (host) | 2-node pilot died at 230 GB RSS vs 156 GB cgroup. Effect on speed unmeasured (2-node run still queued). |
| 13 | `HYBRID_SHARD` for multi-node | comm | pilot 2x4 V100: 39.5 s/micro-step vs ~31 single-node = **+26% cross-node** even with hybrid; accelerate's `no_sync` is DDP-only, so grads sync every micro-step under FSDP. |
| 14 | torch 2.13 + sdpa on H100 (`$WORK/envs/h100_t213`, arch/h100 tree) | kernels | 6800-token bf16 step: 2.6 sdpa 282 ms, 2.8 sdpa 275, 2.8 flash 273, 2.13 flash 273, **2.13 sdpa 243 (-14%)**. flash-attn 2 package is *slower* than 2.13 sdpa. Wall-clock effect *est.* <=5%: the step is dataloader-bound. |
| 15 | VGGT feature cache, 512 then 256 (7171 clips, 1.8 TB / 456 GB) | fit / compute | H100: removes ~0.11 s/sample GPU (VGGT = 7% of step) but the Qwen branch still decodes video (6.11 s/sample measured), so s/it 183-197 cached vs 193-300 live = **wash**. V100: what makes 512 fit at all. |

## What actually moved wall-clock

1. More GPUs (#1): ~1.8x. Everything else combined is within window-to-window noise.
2. Not killing the run: #5, #9, #10 turned "never completes" into "completes". No s/it to compare against.
3. Kernel work (#8, #14): real at the GPU level (-14%), nearly invisible at the step level.

## The lever still on the table

Video decode is 6.11 s of a ~7.4 s per-sample collate and the step is ~60-70% collate.
The 13 GB frame cache (#3) removes that term entirely and is already built; the
`dataloader_prefetch_factor` change (#4) hides the rest of the burst. Neither is wired in.
*Est.* together: ~2x on the joint step at 2-4 GPUs. That is the only remaining change
of the same order as #1.

## Changes that apply only to the H100 jobs (2x H100, DDP)

Live joint chain (`scripts/idea_3i_590k_joint/jz_train_joint*.sh`, conda env, VGGT live):
- `--vsibench_eval_steps` 200 -> 400 (both joint scripts). Only recipe change to the live run.

Cached continuation (`scripts/idea_3i_590k_joint_cached/jz_train_joint_cached_resume.sh`,
output `results/idea_3i_590k_joint_cached_from1175`, seeded from the newest live checkpoint):
- Env: `$WORK/envs/h100_t213` = venv on `arch/h100` + `pytorch-gpu/py3/2.13.0`
  (torch 2.13, CUDA 13.2, cuDNN 9.20, flash-attn 2.8.3, Triton 3.7.1, Liger 0.8.2,
  transformers 5.2.0 / trl 1.2.0 / peft 0.19.1). Job re-inits modules from the node
  (`unset MODULEPATH; source /etc/profile` outside `set -eu`) then loads `arch/h100`.
  Measured: -14% per kernel step vs torch 2.6. `vsibench_eval_full` untouched.
- Attention: `sdpa` (measured faster than the flash-attn package on 2.13); `SR_ATTN` knob.
- Parallelism: DDP (`common/multi_gpu.yaml`), NOT FSDP -- the DDP `optimizer.pt` resumes
  only under DDP, and 80 GB needs no sharding.
- Data: VGGT-256 features from `vggt_cache_256` (H100 only; V100 uses 512); `--color_jitter 0`.
- `--use_liger_kernel True` + fused-loss `forward` port (`idea_3i_590k_joint_cached/model_with_vggt.py`).
- CPU-resident batches: `get_train_dataloader` override in `idea_3i_590k_joint_cached/train.py`
  (after the 77.5 GiB OOM at accum 128).
- `--vsibench_eval_enable False` explicitly (default is True, eval_steps 20).
- Windows: 2 h, head on `qos_gpu_h100-dev`, chain on `-t3`, `afterany`, `save_steps` 25.

NOT applied to H100 (V100/FSDP only): GQA->repeat_kv patch (sm<80 gated), FSDP configs and
activation checkpointing, `ddp_timeout 900` + NCCL flight recorder, host-RAM knobs
(workers 2 / pin_memory off / prefetch 1 / RAM-efficient loading). The H100 cached script
still runs `--ddp_timeout 7200`, 8 workers/rank, default pinning.

## Addendum 2026-09-10 (later)

- **Cached ckpt-1200 (H100 continuation) scored 0.5783 on the full split** -- identical to
  live ckpt-1000; categories reshuffled +-2 pts (appearance-order +5.5). Cached pipeline
  validated end to end; the joint run is flat at ~0.578 from 1000 to 1200.
- **4x H100 cached window: ~100 s/it** (accum 64), same as the live 4-GPU run.
- **2x4 V100 res-512 line parked.** Two windows died on host memory (`/dev/shm` Bus error)
  after 1-2 steps: the accumulation window (32 x ~450 MB/rank) stays shm-resident with
  device placement off, ~58 GB/node, against a 156 GB cgroup with model + workers. A
  clone-on-receipt override did not release it (originals are held until step end) and
  doubled memory -- added and removed the same day. Levers if revived: `num_workers=0`
  or 4 nodes (accum 16). Multi-node FSDP itself works (1108 s/it at accum 32).
