#!/bin/bash

set -euo pipefail

# Score an idea_4a checkpoint with VG-LLM's OWN detection aggregation (macro-F1 over the
# cate8 / cate20 / cate31 name lists), on the same 243 one-per-scene rows the released
# vgllm-3d-vggt-4b was scored on. Prints our per-sample micro-F1 alongside so the number
# stays tied to the training curve.
#
#   scripts/idea_4a_sg_perc_scannetv2_vgjson/eval_vgllm_metric.sh                  # latest ep6 ckpt
#   scripts/idea_4a_sg_perc_scannetv2_vgjson/eval_vgllm_metric.sh --limit 5        # smoke
#   RUN=results/idea_4a_sg_perc_scannetv2_vgjson_lora \
#     scripts/idea_4a_sg_perc_scannetv2_vgjson/eval_vgllm_metric.sh \
#     --checkpoint results/idea_4a_sg_perc_scannetv2_vgjson_lora/checkpoint-12420
#
# WHY worldmirror AND NOT the training env. The metric needs pytorch3d (VG-LLM's IoU) and
# the model needs transformers>=5 (Qwen3-VL). vsibench_eval_full has no pytorch3d, so
# worldmirror was upgraded to transformers 5.2 + peft + trl to hold both. That upgrade
# replaced its transformers 4.46; roll back with
#   pip install "transformers==4.46.1" "tokenizers==0.20.3"
# if something else in worldmirror needs the old one.
#
# vggt_omega and perceiver are source trees, not installed packages, so they go on
# PYTHONPATH rather than being pip-installed into whichever env runs this.

FINETUNING_ROOT="/home/ducpham/scratch/Working/spatial_reasoning/finetuning"
PYTHON="/scratch/ducpham/conda/envs/worldmirror/bin/python"
RUN="${RUN:-results/idea_4a_sg_perc_scannetv2_vgjson_lora_ep6}"

cd "${FINETUNING_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HF_HOME="${HF_HOME:-/home/ducpham/scratch/Working/cache}"
export PYTHONPATH="/scratch/ducpham/Working/spatial_reasoning/reconstruction_models/vggt-omega:/scratch/ducpham/Working/spatial_reasoning/perceiver-io:${FINETUNING_ROOT}/common:${PYTHONPATH:-}"

exec "${PYTHON}" idea_4a_sg_perc_scannetv2_vgjson/eval_vgllm_metric.py \
  --run "${RUN}" \
  "$@"
