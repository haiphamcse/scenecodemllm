#!/bin/bash
#SBATCH --job-name=boxer64f
#SBATCH --account=btf@v100
#SBATCH --constraint=v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=%x-%A_%a.out
#
# Boxer (--fuse, 288 embodiedscan labels) over the det64f ScanNet++ scene tars, one shard per array task.
# Inputs $SCRATCH/dataset/vgllm_data/det64f/scannetpp/_boxer_in/tars/<scene>.tar (scannet/<scene>/frames/...),
# outputs _boxer_out/<scene>/boxer_3dbbs{,_fused,_fused_md2}.csv + .done. Resumable: scenes with .done are skipped.
# Smoke: sbatch --array=0 --qos=qos_gpu-dev --time=00:30:00 --export=ALL,SCENES=<scene> boxer_v100.slurm
set -euo pipefail
source "${WORK}/miniconda3/etc/profile.d/conda.sh"
conda activate vsibench_eval_full
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

B=$WORK/spatial_reasoning/boxer
IN=$SCRATCH/dataset/vgllm_data/det64f/scannetpp/_boxer_in
OUT=$SCRATCH/dataset/vgllm_data/det64f/scannetpp/_boxer_out
N=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}; K=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
SCENES=${SCENES:-$(ls $IN/tars | sed 's/\.tar$//' | awk -v k=$K -v n=$N 'NR%n==k')}
W=$IN/work_$K
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== job ${SLURM_JOB_ID} shard $K/$N: $(echo $SCENES | wc -w) scenes ==="
cd $B
for S in $SCENES; do
  [ -f $OUT/$S/.done ] && continue
  rm -rf $W; mkdir -p $W $OUT/$S; tar -xf $IN/tars/$S.tar -C $W
  t0=$(date +%s)
  python run_boxer.py --input $W/scannet/$S --labels=embodiedscan --skip_viz --fuse --output_dir $OUT --write_name boxer
  python -c "from utils.fuse_3d_boxes import fuse_obbs_from_csv; fuse_obbs_from_csv('$OUT/$S/boxer_3dbbs.csv', '$OUT/$S/boxer_3dbbs_fused_md2.csv', min_detections=2)"
  echo "$S $(( $(date +%s) - t0 )) s" | tee $OUT/$S/.done
  rm -rf $W
done
echo "=== shard $K finished $(date) ==="
