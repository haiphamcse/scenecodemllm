# Iteration time breakdown — idea_3i_590k joint / cold

Measured 2026-09-08 on 1x H100 (bf16) plus the live 4x H100 runs.
Scripts: `profile_micro_step.py`, `profile_h100.slurm`, `profile_e2e_h100.slurm`.

One **iteration** = one optimizer step = **64 micro-steps per rank** at 4 GPUs
(`GRAD_ACCUM = 256 / world_size`, `per_device_train_batch_size 1`).
Divide by 64 before comparing anything per-sample.

## Headline

Neither run is GPU-bound. Most of the wall-clock is not compute.

| | joint | cold |
|---|---|---|
| wall / iteration | 103 s | 90.6 s |
| GPU compute | 32.8 s (32%) | 55.3 s (61%) |
| everything else | **70.2 s (68%)** | **35.3 s (39%)** |

## Why: the accumulation burst

`transformers` `trainer.py:2144` `get_batch_samples` pulls all 64 micro-batches
from the DataLoader *before* running any of them. The prefetch queue is only
8 workers x `prefetch_factor` 2 = 16, so 48 of every 64 are produced with the
GPU idle, and the workers then idle through the whole compute phase.

```
fetch phase    ~22-25 s   GPU idle,     workers saturated
compute phase  ~65-70 s   workers idle, GPU saturated
```

Confirmed in production without extra instrumentation: `qwen_vl_utils` logs one
timestamped line per decode. Bucketing 10 min of the live cold log into 5 s bins
gives bursts strictly periodic at the 90 s step time, with 66% of bins empty.
Each burst is ~256 decodes = one optimizer step across 4 ranks.

## GPU components, per iteration (x64)

| | joint | cold |
|---|---|---|
| VGGT-Omega (frozen) | 7.0 s | **31.3 s** |
| Qwen LLM forward | 8.7 s | 7.8 s |
| backward (LoRA r64 + Perceiver) | 13.2 s | 12.4 s |
| Qwen vision tower | 3.3 s | 3.2 s |
| Perceiver | 0.6 s | 0.6 s |
| optimizer step (once, not x64) | ~0 | ~0 |
| **total** | **32.8 s** | **55.3 s** |

Cold's VGGT costs 4.5x joint's: cold has no separate VGGT resize, so the encoder
gets the full 416x576 Qwen tensor (~936 patches/frame) where joint feeds it
224x288 at `vggt_image_resolution 256` (256 patches/frame).
DDP all-reduce is negligible — once per 64 micro-steps.

## Collate cost, per sample (single-threaded)

| | joint | cold |
|---|---|---|
| **total** | **4.673 s** | **0.403 s** |
| decord decode | 3.786 s (81%) | fused into `process_vision_info` |
| tokenize + processor | ~0.54 s | ~0.03 s |
| VGGT resize | 0.228 s | none |
| LLM resize | 0.058 s | 0.371 s (incl. decode) |
| det JPEG open | 0.061 s | — |

Under 32-worker Lustre contention the effective rate is ~2.75 s/sample, which is
why cold's fetch burst is 22-25 s rather than ~3 s.

Joint's extra text cost is `apply_chat_template` + `_unexpanded_len` running
**twice per row** to find the supervised span. That is the real "twice" — the
vision path runs once (`collator.py:429`).

Live decode by source (n=15428, mean 1.292 s, p50 0.867, p90 3.89):
`adt 4.033 · scannetppv2 1.334 · s3dis 0.484 · arkitscenes 0.314 · scannet 0.294 · procthor 0.259`

## Joint is slower than cold despite 22 s less GPU work

Overlapped (real DataLoader, 8 workers, 24 cores, 1 GPU):

| | delivered | needed | GPU busy |
|---|---|---|---|
| cold | 3.59 samples/s | 1.06 | **98.5%** |
| joint | 1.12 samples/s | 1.83 | 46.6% |

Cold's loader has 3.4x headroom — its decode is already hideable. Joint's cannot
feed its own GPU even with overlap, so it stalls during the compute phase too.

## Fixes, cheapest first

1. **`--dataloader_prefetch_factor 8`** (queue 16 -> 64, burst pre-produced during
   the previous compute phase). *Estimate*: ~24% cold, ~27% joint. Costs ~40 GB
   extra host RAM (*estimate*) — this run has OOM-killed workers before, so try 4 first.
2. **Drop `num_threads=1`** in `idea_3i_590k_joint/collator.py:137`. One token,
   joint only. `qwen_vl_utils` uses multi-threaded decode by default.
3. **Wire the frame cache** (13 GB, 5963 videos, already built). Joint only, and
   only worth it *after* fix 1 — with the burst intact it just shortens the burst.
   For cold it is worth ~0: cold is already 98.5% GPU-busy when overlapped.
4. Cold's VGGT resolution — matching joint's 256 would cut cold's GPU time from
   55.3 to ~31 s/iteration, but it changes what the model sees. A modelling
   decision, not a free win.

## Caveats

- Everything in the tables is measured; items marked *estimate* are not.
- The 2->4 GPU speedup (1.8x) does **not** prove compute-bound: `GLOBAL_BATCH` is
  pinned, so 2->4 halves per-rank micro-steps *and* doubles cores/workers. Per-rank
  resources are identical, and per micro-step it got slightly worse
  (joint 1.51 -> 1.69 s).
- Joint's 4.673 s/sample collate includes two cold-Lustre adt outliers (20.5 s,
  18.4 s). The 1.292 s live-log mean is the better steady-state number; the table
  is for the *relative* split.
