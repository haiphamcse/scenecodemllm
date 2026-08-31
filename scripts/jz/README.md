# Writing a train script that works with `jz/submit.sh`

Reference for developing on this machine and running on Jean Zay. The cluster facts
behind these rules are in `../../../jeanzay/CLAUDE.md`; the workflow is in
`../../../jeanzay/PIPELINE.md`.

The whole contract is five variables. Everything else `submit.sh` handles.

---

## The five rules

### 1. Take `NPROC_PER_NODE` from the environment

```bash
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"     # yes
NPROC_PER_NODE=1                          # no — overrides the allocation
```

`submit.sh` exports this from the Slurm allocation. A hardcoded value silently wins,
so a 4-GPU job runs 1-way and takes four times as long. **65 of the existing scripts
still hardcode it** — they were written before this wrapper and have not been converted.

Keep `GLOBAL_BATCH` fixed and derive accumulation from it, so the optimisation is
identical regardless of GPU count:

```bash
GRAD_ACCUM=$((GLOBAL_BATCH / (BATCH_SIZE * NPROC_PER_NODE)))
```

### 2. Take paths from `SR_*`, with the local literal as fallback

```bash
JSONL_PATH="${SR_JSONL:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"
```

Unset means the literal, so running the script here behaves exactly as before. The
fallback is what makes one script work on both machines without a rewrite step.

Available (from `jeanzay/sites/*.env`, sourced by `submit.sh`):

| var | holds |
|---|---|
| `SR_JSONL` | `vsi_590k.jsonl` |
| `SR_DATA_ROOT` | VSI-590K root |
| `SR_CACHE_ROOT` | frame cache |
| `SR_HF_HOME` | HF cache (models) |
| `SR_MODEL_ROOT` | **input** weights — merged models, init checkpoints |
| `SR_OUTPUT_ROOT` | **output** checkpoints |
| `SR_VSIB` | VSI-Bench videos |
| `SR_VGLLM_ROOT`, `SR_SCANNET_ROOT` | vgllm corpora, ScanNet images |

### 3. `OUTPUT_DIR` must be absolute

```bash
OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/my_run"     # yes
OUTPUT_DIR="results/my_run"                        # no — relative
```

A relative path resolves inside the code directory. On Jean Zay that is `$WORK`, which
is inode-limited and not where results belong; outputs go to `$SCRATCH`.

### 4. Inputs come from `SR_MODEL_ROOT`, not `results/`

```bash
MERGED_MODEL="${SR_MODEL_ROOT:-results}/idea_4a_video_scratch_merged1600"
```

On this machine `results/` holds inputs and outputs together. On Jean Zay they are split:
inputs on `$WORK` (persistent — a purge would break every run), outputs on `$SCRATCH`
(purges, re-shippable). Using `SR_OUTPUT_ROOT` for a model lookup sends it to the wrong
filesystem.

### 5. Don't set what `submit.sh` already sets

It exports these; setting them in your script fights the wrapper:

```
NPROC_PER_NODE  CUDA_VISIBLE_DEVICES  HF_HOME  HF_HUB_OFFLINE  HF_DATASETS_OFFLINE
PYTHONUTF8  LANG  LC_ALL  NCCL_DEBUG  NCCL_P2P_DISABLE  FORCE_QWENVL_VIDEO_READER
```

It also owns every sbatch flag — `--account --constraint --qos --gres --cpus-per-task
--time --nodes --ntasks --hint --output --job-name --export`. Your script never calls
sbatch and never contains `#SBATCH` lines.

`NCCL_P2P_DISABLE` defaults to 0 and is overridable at submit time; see the topology
trap below.

---

## Template

Copy this for a new experiment. It is complete and runnable.

```bash
#!/bin/bash
set -euo pipefail
cd "${WORK:-/home/ducpham/scratch/Working}/spatial_reasoning/finetuning" 2>/dev/null \
  || cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# --- what this run tests, and what differs from its sibling. Write it here, not in
# --- a commit message: the next person to read this is you in three months.

MODEL="${SR_MODEL_ROOT:-results}/idea_4a_video_scratch_merged1600"
OUTPUT_DIR="${SR_OUTPUT_ROOT:-results}/my_experiment"
JSONL_PATH="${SR_JSONL:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K/vsi_590k.jsonl}"
DATA_ROOT="${SR_DATA_ROOT:-/home/ducpham/scratch/Working/dataset/vsi_590k/VSI-590K}"
HF_HOME="${SR_HF_HOME:-/home/ducpham/scratch/Working/cache}"

# Global batch fixed; accumulation absorbs the GPU count, so 1x and 4x optimise
# identically and results stay comparable across allocations.
GLOBAL_BATCH=64
BATCH_SIZE=1
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} not divisible by BATCH_SIZE*NPROC (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))

args=(
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "${MODEL}"
  --attn_implementation sdpa
  --jsonl_path "${JSONL_PATH}"
  --data_root  "${DATA_ROOT}"
  --hf_home    "${HF_HOME}"
  --bf16 True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # 32 frames through VGGT + the ViT drives peak memory; a legacy run peaked at
  # 75.6 GB on an 80 GB card without this.
  --gradient_checkpointing True
  --gradient_checkpointing_kwargs '{"use_reentrant": false}'
  --learning_rate 1e-4
  --perceiver_lr 3e-4
  # 256/2 define the Perceiver latent array; --init_projector tensors are shaped
  # for exactly these. Changing them breaks the projector load.
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 1
  --logging_steps 5
  --save_steps 50
  --save_total_limit 100
  --ddp_find_unused_parameters False
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  --eval_strategy no
  --report_to tensorboard
  --video_fps 1.0
  --video_max_frames 32
  --vsibench_eval_enable
  --vsibench_eval_steps 400
  --lora_enable
  --lora_r 64
  --lora_alpha 128
  --lora_dropout 0.1
  --dataloader_num_workers 3
)

# main_process_port 0 auto-picks a free port; the default 29500 collides when two
# accelerate jobs land on one node.
accelerate launch --config_file common/multi_gpu.yaml \
  --num_processes "${NPROC_PER_NODE}" --main_process_port 0 \
  my_experiment/train.py "${args[@]}" "$@"
```

Trailing `"$@"` matters: `submit.sh` forwards extra args through to your script, and
HF takes the last occurrence of a flag. That is how `--vsibench_eval_steps 400` and
`--max_steps 2` get injected at submit time without editing the file.

---

## Running it

```bash
# 1. sync code up (code only; excludes results/, logs/, legacy/, trash/)
bash jeanzay/jz_provision.sh 7

# 2. smoke first — 2 steps, dev QoS, throwaway dir
source jeanzay/jz.sh
jz <<'SH'
cd "$WORK/spatial_reasoning/finetuning"; set -a; source site.env; set +a
SMOKE=1 GPUKIND=h100 NGPU=4 bash scripts/jz/submit.sh scripts/my_exp/train.sh \
  --output_dir "$SR_OUTPUT_ROOT/_smoke" --max_steps 2 --save_steps 1 --vsibench_eval_steps 100000
SH

# 3. real run, with chaining past the 20 h wall
jz <<'SH'
cd "$WORK/spatial_reasoning/finetuning"; set -a; source site.env; set +a
JZ_OUTPUT_DIR="$SR_OUTPUT_ROOT/my_experiment" JZ_MAX_STEPS=2168 \
GPUKIND=h100 NGPU=4 TIME=20:00:00 \
  bash scripts/jz/submit.sh scripts/my_exp/train.sh
SH
```

Chaining is opt-in **from the caller**, never from the script: set `JZ_OUTPUT_DIR` and
`JZ_MAX_STEPS` and the job resubmits itself from its last checkpoint until it reaches
the target. Set neither and it runs exactly one window.

`GPUKIND=h100|a100` · `NGPU` · `TIME` · `SMOKE=1` · `CPUS` · `NCCL_P2P_DISABLE`.

---

## Traps that ruin a run silently

These do not raise. Each one has cost a real run.

### Resume overrides your learning rate

`--learning_rate` is **inert when resuming an HF checkpoint**. `scheduler.pt` stores
`base_lrs`, `optimizer.pt` stores `initial_lr`, and both are restored *after* the
optimizer is built. Worse, `train.py` logs the optimizer LR groups at construction
time — before the restore — so the log reads like confirmation while being wrong.

Check the **first logged `learning_rate`**, not the LR-groups line. If it does not sit
on the curve you asked for, the run is on the old schedule.

Safe by construction: an output dir that starts empty and only ever holds checkpoints
from one schedule. Dangerous: reusing a dir that already holds a previous run's
checkpoints — which is also why `jz_provision.sh` never syncs `results/`.

A whole-epoch resume additionally needs `--ignore_data_skip True`, or HF replays the
dataloader from step 0 to reach the resume point, grinding through an entire epoch of
video decoding before any compute.

### Reusing an output dir interleaves runs

A script with no `--resume_from_checkpoint` starts fresh but still *writes* into the
same directory, mixing new checkpoints with old ones at identical step numbers. Use a
new name, or move the old dir first.

### A wrong cache path yields an empty dataset, not an error

The frame cache is keyed by each video's resolved absolute path. Wrong prefix means
every lookup misses, and `keep_cached_rows` drops missing rows *by design, silently*.
Training then runs on less data and reports nothing wrong.

Expected for the scannetppv2 filter: **138701 rows / 856 videos / 0 uncached**.

`idea_3i/train.py` is worse — it predates `keep_cached_rows` and does not skip an
uncached row at all; it raises on a bare `np.load` mid-epoch, potentially hours in.

### Smoke-test with `--save_steps 1`

Several failures land only on **checkpoint save**, after the progress bar already reads
100%, which makes a save failure look like a training failure. The UTF-8 locale bug did
exactly this. Two steps with `--save_steps 1` exercises that path twice for about four
minutes of dev-QoS time.

A good smoke log contains:

```
Frame cache: 138701/138701 rows usable (856/856 distinct videos cached).
init_projector .../vggt_projector.safetensors: loaded 170/170 tensors.
Optimizer LR groups: [(0.0001, ...), (0.0003, ...)] (base_lr=0.0001, perceiver_lr=0.0003)
```

`loaded 170/170` is the one that matters. The ~170 `model.vggt_projector.<...> |
UNEXPECTED |` warnings above it are normal — `from_pretrained()` runs before
`initialize_vggt()` builds that submodule. If it ever reads `0/170` while those warnings
still appear, the projector is random.

### NCCL topology

`submit.sh` prints `nvidia-smi topo -m` into every log. If it shows `SYS` between
allocated GPUs, a group spanning that hop hangs DDP's first broadcast **in silence** —
resubmit with `NCCL_P2P_DISABLE=1`. A full `gpu_p6` node is 4×H100 behind NVLink
(`NV6` between all pairs), so P2P stays on by default.

### Eval cadence costs wall clock

`--vsibench_eval_steps 100` runs a 752-sample generation eval ~21 times an epoch, inside
a 20 h window. 400 is usually the better trade. Historical curves swung ±0.03 between
adjacent 200-step evals, so the extra resolution is largely noise — and do not judge a
run before ~1200 steps; step-200 rankings have been actively misleading.

---

## Status of existing scripts

Only `legacy/idea_3i/train_tuned.sh` has been converted. Of the other 126:

| | count |
|---|---|
| hardcode `NPROC_PER_NODE` | 65 |
| literal `HF_HOME` | 101 |
| relative `OUTPUT_DIR="results/..."` | 82 |
| literal `DATA_ROOT` | 72 |
| literal `JSONL_PATH` | 59 |

They still run correctly **on this machine**. They will not run correctly on Jean Zay
until converted — the rewrite is mechanical and backwards-compatible.
