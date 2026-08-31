#!/bin/bash

set -euo pipefail

# Viser view of PREDICTED vs GT boxes over the back-projected clip cloud, for one sample
# of the LoRA run's scannet_eval.txt. Wraps visualize_pred.py with the env and the paths.
#
#   scripts/idea_4a_sg_perc_scannetv2_vgjson/visualize.sh            # sample 0
#   scripts/idea_4a_sg_perc_scannetv2_vgjson/visualize.sh 3          # sample 3
#   scripts/idea_4a_sg_perc_scannetv2_vgjson/visualize.sh 3 --check  # no server, assert
#                                                                    # the cloud lands in
#                                                                    # the GT boxes
#
# Indices run 0..9 (the run's --vsibench_max_eval_samples). The eval log holds several
# step blocks; the viewer always shows the LAST one.
#
# This run has no overfit_scenes.json (it is --overfit False), so visualize_pred.py
# rebuilds the eval rows from the val json and verifies each against the GT the log
# recorded. --min-boxes must match the run's --min_boxes (5) or that check trips.

FINETUNING_ROOT="/home/ducpham/scratch/Working/spatial_reasoning/finetuning"
PYTHON="/scratch/ducpham/conda/envs/vsibench_eval_full/bin/python"
RUN="results/idea_4a_sg_perc_scannetv2_vgjson_lora"
PORT="${PORT:-8081}"

SG_INDEX="${1:-0}"
shift || true

cd "${FINETUNING_ROOT}"

echo "Port-forward from your machine, then open http://localhost:${PORT} :"
echo "  ssh -N -L ${PORT}:$(hostname):${PORT} <login-host>"
echo

exec "${PYTHON}" idea_4a_sg_perc_scannetv2_vgjson/visualize_pred.py \
  --run "${RUN}" \
  --sg-index "${SG_INDEX}" \
  --min-boxes 5 \
  --port "${PORT}" \
  "$@"
