# sr_vggt

Spatial-reasoning finetuning of Qwen3-VL with geometry features from a frozen VGGT model, trained
and evaluated on the IDRIS Jean Zay cluster. [EXPERIMENTS.md](EXPERIMENTS.md) lists every
configuration and the scores of the original runs. [CLUSTER_NOTES.md](CLUSTER_NOTES.md) collects
the Jean Zay pitfalls.

**Main model.** Each clip is decoded at 1 fps (at most 64 frames). Qwen3-VL-2B sees the frames at
128 tokens per frame. A frozen VGGT-Omega (1B, 512 checkpoint) runs on the same frames inside the
model, in bf16 with no gradients. Two Perceiver IO encoders compress its patch tokens into 512
latents and its camera tokens into 32 latents. Those latents replace placeholder tokens in the
prompt. Training uses LoRA (r64) on the LLM plus the Perceivers, for one epoch over a
190k-question mix of VSI-590K and VLM-3R questions on 4,878 VSI-590K videos.

The repo holds everything except the videos and the model weights: the two training mixes
(`data/*.jsonl.gz`), and vendored copies of perceiver-io, VGGT-Omega and VGGT-1B
(`third_party/`, with their licenses).

## Quickstart (Jean Zay)

You need a Jean Zay project with A100 and H100 hours, a Hugging Face account, and the VSI-590K
videos. All steps run from a Jean Zay **login node**, in the code directory (e.g.
`$WORK/sr_vggt`).

```bash
# 0. Videos. Training reads <SR_VSI590K>/{arkitscenes,scannet,scannetppv2}/<scene>.mp4 (4,878
#    clips, ~132 GB) from the nyu-visionx/VSI-590K dataset. Eval data: see "VSI-Bench" below.

# 1. Config: project, video dirs, HF_HOME, envs, outputs. Every script reads it.
cp site.env.example site.env && $EDITOR site.env

# 2. Python envs (once, ~15 min each; the login node has internet through the IDRIS proxy).
scripts/setup_env.sh h100      # training + login-node tools
scripts/setup_env.sh a100      # eval

# 3. Weights into HF_HOME (compute nodes are offline). Gated repos need a token whose account
#    accepted their terms on huggingface.co: `export HF_TOKEN=...` in this shell, or
#    `HF_HOME=<SR_HF_HOME> hf auth login`. Never commit it.
scripts/download_weights.sh

# 4. Data check for an experiment (counts every row and video; exit 0 = ready).
scripts/verify_data.sh main

# 5. Smoke test: one 2 h dev window, 20 steps, into $SR_OUTPUT_ROOT/_smoke_sr_vggt_main.
SMOKE=1 scripts/train.sh main --max_steps 20
#    then the real run: 4 h windows on 2xH100, chained by a login-node watcher until done.
scripts/train.sh main

# 6. VSI-Bench full eval: 8 A100 shards + resubmit + merge; score in the watcher log and in
#    <run>/eval_checkpoint-743_full/results_full_merged.json.
scripts/eval.sh main $SR_OUTPUT_ROOT/sr_vggt_main/checkpoint-743
```

`<experiment>` is any `configs/<name>.env`: `main lora128 text256 4b lat3072 vggt1b spatialstack
vlm3r reason16k`. `reason16k` extends the text256 ablation and starts from
`sr_vggt_text256/checkpoint-743`, so train `text256` first. To start from another checkpoint,
set `INIT_CKPT=<dir>`.

## VSI-Bench

The eval needs a local copy of the `nyu-visionx/VSI-Bench` dataset repository, with its three
video zips unzipped in place. `SR_VSIBENCH` in `site.env` points to it. The reference runs used
revision `d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d`. On a login node:

```bash
hf download nyu-visionx/VSI-Bench --repo-type dataset \
  --revision d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d --local-dir "$SR_VSIBENCH"
cd "$SR_VSIBENCH" && for z in arkitscenes scannet scannetpp; do unzip -q $z.zip; done
```

Expected layout: `$SR_VSIBENCH/README.md` (it defines the `full` config), `test*.parquet`, and
`{arkitscenes,scannet,scannetpp}/<scene>.mp4` (512 videos). The eval loads
`load_dataset($SR_VSIBENCH, "full")`, which gives 5,130 questions in the hub's order.
`verify_data.sh` checks all of this.

## Layout

```
configs/<exp>.env          one per experiment: VGGT backbone/input, mix, model/fusion, train + eval flags
data/*.jsonl.gz            the two training mixes (video paths relative to SR_VSI590K)
sr_vggt/                   the Python package (7 files)
  model.py                 Qwen3-VL + live frozen VGGT + Perceiver placeholder scatter (core), fusion
                           hooks, VGGT weight loading, the experiment flags (VggtArguments)
  ablations.py             SpatialStack and VLM-3R fusion modules (subclasses of the core)
  data.py                  frame sampling, VGGT input preprocessing, mixes, prompts, collator, data check
  reasoning.py             reasoning-trace prompt, length filter, init-from-checkpoint, answer parsing
  train.py                 LoRA SFT entry point
  eval.py                  VSI-Bench eval, shard merge, vendored lmms-eval scoring
third_party/               vendored perceiver-io @46c6325, vggt-omega @39a0cb8, VGGT-1B (licenses inside)
scripts/                   setup_env, download_weights, verify_data, train, eval; jobs/*.slurm; lib.sh
tests/checks.py            CPU checks; GPU eval parity vs the original code (tests/parity.slurm)
```

Train and eval build inputs with the same functions (`data.decode_video_frames`,
`data.clip_inputs`), so a clip looks the same to the model in both. Paths come only from
`site.env`.
