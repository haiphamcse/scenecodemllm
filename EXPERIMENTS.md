# Experiments

Every experiment trains 1 epoch at global batch 256: 743 steps on the 190k mix, or 569 steps
for `reason16k`. All runs use LoRA on every LLM linear layer (dropout 0.1), cosine schedule, 3%
warmup, weight decay 0.01, bf16, 64 det_es frames per clip at 128 tok/frame, and 2 x H100.

The scores are VSI-Bench full split (5,130 rows) at the final checkpoint, from the original runs.
They are the reproduction targets. `scripts/eval.sh` runs A100 bf16. Several reference scores came
from V100 fp32 evals. Expect differences of a few thousandths from precision alone.

| config | original run (finetuning/) | what it changes vs main | cache | VSI-Bench full |
|---|---|---|---|---|
| `main` | idea_3i_190k_es_64f_lat512_text256 | (Qwen3-VL-2B, VGGT-Omega text256, Perceiver 512 + 32 latents, LoRA r64/a128, lr 1e-4, text256 prompt) | es_text | **0.6060** @743 |
| `4b` | idea_3i_190k_es_64f_lat512_text256_4b | Qwen3-VL-4B-Instruct | es_text | **0.6784** @743 |
| `lat3072` | idea_3i_190k_es_64f_lat3072_text256 | 3072 frame latents | es_text | 0.6117 @743 |
| `lora128` | idea_3i_190k_es_64f_lat512_lora128 | LoRA r128/a256; **also** VGGT-Omega 512 (not text-aligned) and the short "clean" prompt (see below) | es | 0.6059 @743 |
| `vggt1b` | idea_3i_190k_es_64f_lat512_vggt1b | original VGGT-1B backbone (patch 14, 5 camera tokens) | es_vggt1b | not evaluated (trained to 743) |
| `spatialstack` | idea_3i_190k_es_64f_spatialstack | SpatialStack layered fusion instead of the Perceiver; VGGT-Omega 512 layers 11/17/23 on Qwen's own pixels; no placeholders | es_ss | **0.6208** @743 (best 2B) |
| `vlm3r` | idea_3i_190k_es_64f_vlm3r | VLM-3R cross-attention of video tokens to VGGT tokens; VGGT-Omega 512; no placeholders | es | 0.6013 @743 |
| `reason16k` | idea_3i_190k_es_64f_lat512_text256_reason_cont_16k | continues `main` ckpt-743 on 145,647 SimpleSR reasoning traces; lr 5e-5; 16,384-token cap; eval generates up to 8,192 tokens and scores the line after the last "Answer:" | es_text | debiased 0.3381 @569 (text256 on the same questions: 0.5404); no full-split score |

## What each config changes

- **main.** Two Perceiver IO encoders read VGGT-Omega text-aligned features from the last
  aggregator layer. The frame encoder has 4 cross-attention layers and 4x4 self-attention layers,
  with widening 2, and outputs 512 latents. The camera encoder (2 + 2x2 layers) reads the camera
  and register tokens and outputs 32 latents. The latents fill `<|quad_start|>` x512 and
  `<|quad_end|>` x32 in the prompt `Per-frame geometry features: ... Camera and scene features:
  ...`, which is followed by the video and the question.
- **4b, lat3072.** Single-flag changes of main. Diffing their fork code against main shows
  only docstrings and the eval-side resume/flush, which sr_vggt has for every config.
- **lora128.** This is not a single-flag change. Its fork is a byte-identical copy of the
  earlier "clean" run (idea_3i_190k_es_64f_lat512_clean). That run predates text256, so lora128
  also uses the non-text VGGT-Omega 512 checkpoint (cache `es`) and the short prompt (`--prompt
  clean`). Compared with main, it changes three things, not one.
- **vggt1b.** The model code is unchanged. The backbone is VGGT-1B: patch 14, so a 384x256 frame
  gives 210x308 pixels and 330 patches, and each frame has 1 camera + 4 register tokens. The cache
  is stamped with `vggt_patch_size=14`, so an Omega cache can never be read by mistake.
- **spatialstack.** No Perceiver and no placeholders. Each of VGGT layers 11, 17 and 23 feeds a
  zero-initialised merger (RMSNorm, 2x2 patch merge, MLP). The merger outputs are added to Qwen's
  deepstack features at decoder layers 0, 1 and 2. VGGT runs on the det_es pixels unresized, so its
  patch grid equals Qwen's. Frames are averaged in pairs to match Qwen's temporal groups.
- **vlm3r.** No Perceiver and no placeholders. Each Qwen temporal group of video embeddings
  cross-attends to the VGGT camera and patch tokens of its 2 frames. The residual is
  zero-initialised; 16 heads.
- **reason16k.** Only data and settings differ; the model code is main's. The system prompt
  asks for `<scene>` boxes, then `<think>`, then `Answer:`. The first window loads main's adapter
  with a fresh optimizer. The 16k length cap keeps all 145,647 rows (the original 6,144 cap
  dropped 14,439). The original eval used `--max_new_tokens 8192`. Going from 4,096 to 8,192
  rescued no rows: about 10% of generations loop until the cap. The original eval resumed
  timed-out shards with `--num_parts/--part_index` (commit 1266068), and `scripts/eval.sh` does
  the same via `EVAL_PARTS=4`.

## Reproduction notes

- Original training speed on 2 x H100, measured with the original code: about 60 s/step for the
  2B Perceiver runs, about 85 s/step for lat3072 and spatialstack (spatialstack is IO-bound on
  its 300 MB/clip cache). The sr_vggt smoke runs (see the hand-back report) measure the same.
- Data order is the jsonl order shuffled by the HF Trainer sampler (seed 42). The Perceiver
  initial weights are not seeded, as in the original runs.
- Eval rows are sorted by video and split into 8 contiguous shards. VGGT runs once per video.
