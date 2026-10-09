#!/bin/bash
# Build one VGGT feature cache: N single-V100 shards (default 10, ~45 min each for the 4,878 clips
# of the 190k mix) on the V100 dev QoS. Resumable: rerun the same command after a timeout and only
# missing entries are exported; then check with scripts/verify_data.sh.
#
#   scripts/export_vggt.sh <cache>          caches: es_text es es_ss es_vggt1b (configs/caches/)
# Env: NUM_SHARDS (10; the dev QoS takes at most 10 jobs per user), QOS (qos_gpu-dev), TIME (02:00:00).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CACHE=${1:?usage: scripts/export_vggt.sh <cache> ($(ls "${SR_ROOT}/configs/caches" | sed 's/.env$//' | tr '\n' ' '))}
load_cache "${CACHE}"
NUM_SHARDS=${NUM_SHARDS:-10}
mkdir -p "${CACHE_ROOT}"
# -C v100-32g, not --partition: also excludes the 16 GB V100 nodes.
j=$(sbatch --parsable --job-name="export_${CACHE}" --account="${SR_PROJECT}@v100" -C v100-32g \
  --qos="${QOS:-qos_gpu-dev}" --array="0-$((NUM_SHARDS - 1))" --nodes=1 --ntasks=1 --gres=gpu:1 \
  --cpus-per-task=10 --hint=nomultithread --time="${TIME:-02:00:00}" \
  --output="${SR_LOG_DIR}/%x-%A_%a.out" --export=ALL,SR_ROOT="${SR_ROOT}" \
  "${SR_ROOT}/scripts/jobs/export.slurm" "${CACHE}" "${NUM_SHARDS}")
echo "export array ${j} (${NUM_SHARDS} shards) -> ${CACHE_ROOT}; logs ${SR_LOG_DIR}/export_${CACHE}-${j}_*.out"
