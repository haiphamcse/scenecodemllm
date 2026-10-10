# sr_vggt

Spatial-reasoning finetuning of Qwen3-VL with geometry features from a frozen VGGT model, trained
and evaluated on the IDRIS Jean Zay cluster. The main model reaches 0.600 on VSI-Bench (full split).
The best ablations reach 0.621 (SpatialStack fusion, 2B) and 0.678 (4B). [EXPERIMENTS.md](EXPERIMENTS.md) lists every configuration and the scores
to reproduce. [CLUSTER_NOTES.md](CLUSTER_NOTES.md) collects the Jean Zay pitfalls.

**Main model.** Qwen3-VL-2B sees 64 RGB frames per clip (128 tokens per frame). VGGT-Omega
(1B, 512 checkpoint) encodes the same frames. Two Perceiver IO encoders compress its patch tokens into 512 latents and its
camera tokens into 32 latents. Those latents replace placeholder tokens in the prompt
("... This is the 3D scene context."). Training
uses LoRA (r64) on the LLM plus the Perceivers, for one epoch over a 190k-question mix built
from VSI-590K and VLM-3R data.

## Quickstart (new user, Jean Zay)

You need a Jean Zay project with V100, A100 and H100 hours, a Hugging Face account, and the data
zip.

**Licensing:** the zip holds frames and videos derived from ScanNet, ScanNet++ and ARKitScenes. You
may receive it only if you signed those datasets' terms yourself. Do not redistribute it or put it
anywhere public.

All steps run from a Jean Zay **login node**, in the code directory (`$WORK/sr_vggt`).

```bash
# 0. Config: paths, project. Everything below reads it.
cp site.env.example site.env && $EDITOR site.env

# 1. Python envs (once; ~15 min each; login node has internet through the IDRIS proxy).
scripts/setup_env.sh h100      # training
scripts/setup_env.sh a100      # eval
scripts/setup_env.sh v100      # VGGT export + login-node tools

# 2. Weights + VSI-Bench question table into HF_HOME (compute nodes are offline).
#    Gated repos need a token whose account accepted their terms on huggingface.co:
#    `export HF_TOKEN=...` in this shell, or `HF_HOME=<SR_HF_HOME> hf auth login`. Never commit it.
scripts/download_weights.sh

# 3. Data: ~300k files, tens of minutes on Lustre (run inside screen/tmux).
scripts/unpack_data.sh /path/to/sr_vggt_data.zip

# 4. VGGT feature caches (V100 dev, 10 shards x ~45 min). Rerun after a timeout: it resumes.
scripts/export_vggt.sh es          # main, lora128, vlm3r
#   es_text (text256, 4b, lat3072, reason16k), es_ss (spatialstack), es_vggt1b (vggt1b)

# 5. Full data check for an experiment (exact counts; exit 0 = ready).
scripts/verify_data.sh main

# 6. Smoke test: one 2 h dev window, 20 steps, into $SR_OUTPUT_ROOT/_smoke_sr_vggt_main.
SMOKE=1 scripts/train.sh main --max_steps 20
#    then the real run: 4 h windows on 2xH100, chained by a login-node watcher until done.
scripts/train.sh main

# 7. VSI-Bench full eval: 8 A100 shards + resubmit + merge; score in the watcher log and in
#    <run>/eval_checkpoint-743_full/results_full_merged.json.
scripts/eval.sh main $SCRATCH/sr_vggt_results/sr_vggt_main/checkpoint-743
```

`<experiment>` is any `configs/<name>.env`: `main lora128 text256 4b lat3072 vggt1b spatialstack
vlm3r reason16k`. `reason16k` extends the text256 ablation and starts from
`sr_vggt_text256/checkpoint-743`, so train `text256` first. To start from another checkpoint,
set `INIT_CKPT=<dir>`.

## Layout

```
configs/<exp>.env          one per experiment: cache, mix, model/fusion flags, train + eval flags
configs/caches/<c>.env     one per VGGT cache: backbone, weights, frame mode, resolution / layers
sr_vggt/                   the Python package
  model.py                 Qwen3-VL + Perceiver placeholder scatter (core); fusion hooks
  data.py                  mix rows, det_es frames, prompts, collator
  encoders.py              VGGT backbones, preprocessing, token extraction (export + eval)
  vggt_cache.py            the per-clip safetensors cache and its parameter guard
  train.py eval.py export.py verify.py merge_eval.py metrics.py config.py
  ablations/               spatialstack.py, vlm3r.py (own fusion modules), vggt1b/ (vendored VGGT-1B)
  extensions/reasoning.py  reasoning-trace system prompt, length filter, init-from-checkpoint, answer parsing
scripts/                   setup_env, download_weights, unpack_data, export_vggt, verify_data,
                           train, eval, pack_data (data owner only); jobs/*.slurm bodies; lib.sh
tests/parity.py            how this code was checked against the original experiment code
```

How data flows: every training row names a VSI-590K clip. Qwen gets that clip's 64 det_es
JPEGs. These are posed frames, stored already at the 128 tok/frame size. The VGGT branch reads
the same clip's tokens from the cache, which `export_vggt.sh` built from those same JPEGs. At eval
time, VSI-Bench clips are decoded from mp4 at 1 fps (at most 64 frames), and VGGT runs live on
them with the same preprocessing.

Paths come only from `site.env` and the configs. Cache entries are keyed by the clip's resolved
absolute path and stamped with the VGGT weight path. If you move `SR_DATA` or `SR_HF_HOME` after
exporting, every entry goes stale. The loader then raises an error; it never silently recomputes.
