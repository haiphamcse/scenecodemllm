# Experiments

Every experiment trains 1 epoch at global batch 256: 743 steps on the 190k mix, or 569 steps
for `reason16k`. All runs use LoRA on every LLM linear layer (dropout 0.1), cosine schedule, 3%
warmup, weight decay 0.01, bf16, and 2 x H100. Clips are decoded at 1 fps, at most 64 frames,
128 tok/frame, with frozen VGGT run live in bf16.

**The scores below are not reproduction targets in the strict sense.** They come from the
original runs (VSI-Bench full split, 5,130 rows, final checkpoint), which trained differently:
- Training frames were each clip's precomputed "det_es" frames: 64 posed frames, stored at the
  128-token size.
- VGGT features came from a cache exported once in fp32 on V100.

sr_vggt decodes the training videos with the eval's sampling rule and runs VGGT live in bf16, so
the training inputs differ. Treat the scores as the level to expect, not exact targets. Eval
itself is unchanged: the original evals already decoded VSI-Bench video and ran VGGT live, and
`tests/parity.py` checks that the eval paths match. `scripts/eval.sh` runs A100 bf16; several
original scores came from V100 fp32 evals.

| config | original run (finetuning/) | base | what it changes vs its base | VGGT weights / input | original VSI-Bench full |
|---|---|---|---|---|---|
| **`main`** | idea_3i_190k_es_64f_lat512_clean | (main) | Qwen3-VL-2B, VGGT-Omega 512 checkpoint, Perceiver 512 + 32 latents, short "clean" prompt, LoRA r64/a128, lr 1e-4 | Omega 512, balanced 256 | **0.6000** @743 |
| *ablations of main* | | | | | |
| `lora128` | idea_3i_190k_es_64f_lat512_lora128 | main | LoRA r128/a256 | Omega 512, balanced 256 | 0.6059 @743 |
| `text256` | idea_3i_190k_es_64f_lat512_text256 | main | text-aligned VGGT-Omega checkpoint (`vggt_omega_1b_256_text.pt`) + descriptive prompt (2 changes) | Omega text256, balanced 256 | 0.6060 @743 |
| `spatialstack` | idea_3i_190k_es_64f_spatialstack | main | SpatialStack layered fusion instead of the Perceiver; layers 11/17/23 on Qwen's own pixels; plain prompt (no placeholders) | Omega 512, Qwen pixels, layers 11/17/23 | **0.6208** @743 (best 2B) |
| `vlm3r` | idea_3i_190k_es_64f_vlm3r | main | VLM-3R cross-attention of video tokens to VGGT tokens instead of the Perceiver; plain prompt (no placeholders) | Omega 512, balanced 256 | 0.6013 @743 |
| `vggt1b` | idea_3i_190k_es_64f_lat512_vggt1b | main | original VGGT-1B backbone (patch 14, 5 camera tokens) + descriptive text256 prompt (2 changes; the fork was copied from text256) | VGGT-1B, balanced 256 | not evaluated (trained to 743) |
| *ablations of text256* | | | | | |
| `4b` | idea_3i_190k_es_64f_lat512_text256_4b | **text256** | Qwen3-VL-4B-Instruct | Omega text256, balanced 256 | **0.6784** @743 |
| `lat3072` | idea_3i_190k_es_64f_lat3072_text256 | **text256** | 3072 frame latents | Omega text256, balanced 256 | 0.6117 @743 |
| *extension of text256* | | | | | |
| `reason16k` | idea_3i_190k_es_64f_lat512_text256_reason_cont_16k | **text256** ckpt-743 | continues on 145,647 SimpleSR reasoning traces; lr 5e-5; 16,384-token cap; eval generates up to 8,192 tokens and scores the line after the last "Answer:" | Omega text256, balanced 256 | debiased 0.3381 @569 (text256 on the same questions: 0.5404); no full-split score |

`4b`, `lat3072` and `reason16k` were built on text256, not on main. Comparing them with main
mixes in text256's two changes. Compare them with `text256` (0.6060) instead.

## What each config changes

- **main.** Two Perceiver IO encoders read VGGT-Omega (512 checkpoint) features from the last
  aggregator layer. The frame encoder has 4 cross-attention layers and 4x4 self-attention layers,
  with widening 2, and outputs 512 latents. The camera encoder (2 + 2x2 layers) reads the camera
  and register tokens and outputs 32 latents. The latents fill `<|quad_start|>` x512 and
  `<|quad_end|>` x32, followed by "This is the 3D scene context.", the video and the question.
  The system prompt is the short one (data.py `_SHORT_SYSTEM`).
- **lora128.** Its fork's code is byte-identical to main's, and its launcher differs from main's
  in `--lora_r/--lora_alpha` only.
- **text256.** Uses the text-aligned VGGT-Omega checkpoint (aggregator tokens only; its
  text-alignment head is loaded and dropped). It also uses the descriptive prompt: a longer
  system prompt naming the three inputs, and `Per-frame geometry features: ... Camera and scene
  features: ...` around the placeholders. The model code is identical to main's.
- **4b, lat3072.** Single-flag changes of text256. Diffing their fork code against text256 shows
  only docstrings and the eval-side resume/flush, which sr_vggt has for every config.
- **vggt1b.** The model code is unchanged. The backbone is VGGT-1B: patch 14, so a 384x256 frame
  gives 210x308 pixels and 330 patches, and each frame has 1 camera + 4 register tokens. It keeps
  text256's descriptive prompt, because the original fork was copied from text256. Relative to
  main it is therefore two changes (backbone + prompt); relative to text256 it is one change
  (VGGT-1B replacing the text-aligned VGGT-Omega).
- **spatialstack.** No Perceiver and no placeholders: the prompt is main's minus the placeholders.
  Each of VGGT-Omega 512 layers 11, 17 and 23 feeds a zero-initialised merger (RMSNorm, 2x2 patch
  merge, MLP). The merger outputs are added to Qwen's deepstack features at decoder layers 0, 1
  and 2. VGGT runs on Qwen's own processed pixels, so its patch grid equals Qwen's. Frames are
  averaged in pairs to match Qwen's temporal groups.
- **vlm3r.** No Perceiver and no placeholders, and the same VGGT-Omega 512 input as main.
  Each Qwen temporal group of video embeddings cross-attends to the VGGT camera and patch tokens
  of its 2 frames. The residual is zero-initialised; 16 heads.
- **reason16k.** Only data and settings differ; the model code is text256's. The system prompt
  is text256's plus instructions to produce `<scene>` boxes, then `<think>`, then `Answer:`. The
  first window loads text256's ckpt-743 adapter with a fresh optimizer. The 16k length cap keeps
  all 145,647 rows (the original 6,144 cap dropped 14,439). The original eval used
  `--max_new_tokens 8192`. Going from 4,096 to 8,192 rescued no rows: about 10% of generations
  loop until the cap. The original eval resumed timed-out shards with `--num_parts/--part_index`
  (commit 1266068), and `scripts/eval.sh` does the same via `EVAL_PARTS=4`.

## Reproduction notes

- Original training speed on 2 x H100, with det_es frames and cached VGGT: about 57-60 s/step
  for the 2B Perceiver runs, about 85 s/step for lat3072 and spatialstack.
- Live VGGT adds its forward pass for every sample. Back-of-envelope: one 64-frame clip at
  balanced 256 is ~17.7k VGGT tokens, ~60 TFLOP (half of it the global attention). That is
  ≥0.15 s/clip on an H100, so 128 clips per GPU per step add ≥20 s/step. spatialstack's
  native-resolution input (~25k tokens) costs about 1.5x that. Earlier 32-frame runs measured the
  VGGT share at ~7% of GPU time. Decoding 64 frames per clip moves to the dataloader workers
  (8 per GPU). The smoke runs will measure the real s/step.
- Data order is the jsonl order shuffled by the HF Trainer sampler (seed 42). The Perceiver
  initial weights are not seeded, as in the original runs.
- Eval rows are sorted by video and split into 8 contiguous shards.
