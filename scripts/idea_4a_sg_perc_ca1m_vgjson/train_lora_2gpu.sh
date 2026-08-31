#!/bin/bash

set -euo pipefail

FINETUNING_ROOT="./"
cd "${FINETUNING_ROOT}"

export FORCE_QWENVL_VIDEO_READER="${FORCE_QWENVL_VIDEO_READER:-decord}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Two-GPU version of train_lora.sh. Everything about the optimisation is deliberately
# IDENTICAL -- this buys wall-clock only, so that the loss curve stays a continuation of
# the single-GPU one and checkpoint-1740 can be resumed into it.
#
# GPUs 0 and 3 because they are the only free pair on gpu017: GPU1 is another user's
# (~70 GB) and GPU2 is another user's (~21 GB).
#
# TOPOLOGY WARNING, accepted knowingly. nvidia-smi topo -m on this node:
#     GPU0/GPU1 -> NV12 (NVLink), NUMA 0, CPUs 0-31,64-95
#     GPU2/GPU3 -> NV12 (NVLink), NUMA 1, CPUs 32-63,96-127
#     GPU0/GPU3 -> SYS  (PCIe + the UPI link between sockets)
# So 0+3 is the WORST pair on the node: cross-socket, no NVLink. It is still the right
# call here because LoRA makes the allreduce tiny -- 372.5M trainable params in bf16 is
# ~745 MB per step against a ~75 s step, i.e. well under a second even over UPI. If the
# NVLink pair frees up, 0+1 or 2+3 is strictly better and costs nothing to switch to.
# NCCL_P2P_DISABLE=1 is now MANDATORY here, not a fallback. The first 2-GPU attempt died
# at step 1939 with `CUDA error: an illegal memory access was encountered` on rank 1, and
# the driver log names the cause exactly:
#     [Mon Aug  3 02:04:34 2026] NVRM: Xid (PCI:0000:e3:00): 31, MMU Fault:
#     ENGINE GRAPHICS GPC1 GPCCLIENT_T1_8 faulted @ 0x7fed_ea201000.
#     Fault is of type FAULT_UNSUPPORTED_APERTURE ACCESS_TYPE_VIRT_WRITE
# PCI e3:00 is GPU3 = rank 1, and the timestamp is 11 s before the Python traceback.
# UNSUPPORTED_APERTURE on a VIRT_WRITE is GPU3 writing into GPU0's memory through a P2P
# aperture that is not usable across the socket boundary. `nvidia-smi topo -p2p r` reports
# OK for 0<->3, so the driver advertises P2P that does not work -- disabling it routes the
# allreduce through host memory, which costs ~nothing at LoRA's 745 MB message size.
# ECC is clean (0 uncorrected on both), so this is addressing, not failing silicon. Note
# the node has form: same GPU3 and GPU1 logged Xid 31 MMU faults on Jul 14/15 plus Xid 109
# CTX SWITCH TIMEOUT.
#
# SEPARATE, STILL-UNEXPLAINED ANOMALY -- do NOT attribute it to the P2P fault above, that
# was tested and ruled out. grad_norm runs 0.033-0.050 at epochs 20.06-20.17 against
# 0.010-0.021 over n=50 for the single-GPU run at the SAME epochs and lr, every 2-GPU point
# above the single-GPU max. Disabling P2P changed nothing: 0.04587 with P2P on, 0.04696
# with it off, and loss reproduced to four decimals across the two launches (0.3681 /
# 0.3682). That reproducibility means it is deterministic and structural, not corruption.
# Ruled out: effective batch (max_steps 26100 both), lr, `num_items_in_batch` gathering
# (transformers/trainer.py:2187 gathers across ranks), gradient clipping, P2P corruption.
# Leading remaining candidate, UNVERIFIED: the loss is token-normalized against a globally
# gathered count, but DDP averages rank gradients rather than token-weighting them, and
# this corpus runs 5-311 boxes per scene, so the two ranks routinely carry very different
# token counts and the step gets systematically reweighted.
# Consequence: treat the 2-GPU run as a NEW run, not a continuation of the single-GPU
# curve. Use train_lora.sh (1 GPU) if exact comparability matters more than the 2x speed.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,3}"

OUTPUT_DIR="results/idea_4a_sg_perc_ca1m_vgjson_lora"
CORPUS_ROOT="/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/ml-cubifyanything/ca1m_vgllm/train"
HF_HOME="/home/ducpham/scratch/Working/cache"

# GLOBAL_BATCH stays 32, NOT 64. The effective batch is the thing the LR schedule, the
# 26100-step horizon and checkpoint-1740's optimizer state are all defined against; the
# second GPU takes half the grad-accum work instead of doubling the batch. 1 x 2 x 16 = 32,
# same as the single-GPU 1 x 1 x 32. Also worth remembering that a LARGER batch is what
# stalled the original full-FT run (128 -> per-parameter SNR collapse).
GLOBAL_BATCH=32
BATCH_SIZE=1
NPROC_PER_NODE=2

_per_step=$((BATCH_SIZE * NPROC_PER_NODE))
if (( GLOBAL_BATCH % _per_step != 0 )); then
  echo "ERROR: GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by BATCH_SIZE * NPROC_PER_NODE (${_per_step})." >&2
  exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / _per_step))


args=(
  --output_dir "${OUTPUT_DIR}"
  --model_name_or_path "Qwen/Qwen3-VL-2B-Instruct"
  --attn_implementation sdpa
  --corpus_root "${CORPUS_ROOT}"
  --hf_home "${HF_HOME}"
  --max_graph_tokens 16384
  --min_boxes 5
  --bf16 True
  --gradient_checkpointing True
  --per_device_train_batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  # No --full_finetune: LoRA path, r=64 / alpha=128 / dropout=0.1 over q,k,v,o + gate,up,
  # down, with vggt_projector in modules_to_save.
  --learning_rate 1e-4
  --perceiver_lr 1e-4
  --frame_num_latents 256
  --frame_widening_factor 2
  --num_train_epochs 300
  --logging_steps 5
  --lr_scheduler_type cosine
  --warmup_ratio 0.03
  --weight_decay 0.01
  # 100: ~1 h between saves at 37 s/step. Three kills so far -- SLURM expiry at step 1577
  # and 1992, then the Xid 31 MMU fault at 1939. That last one landed 18 steps before the
  # 1957 save and cost all 199 steps back to checkpoint-1740, which is what 217 was
  # supposed to prevent. 20 x 2.5 G = 50 GB, retained window 2000 steps (~23 epochs).
  --save_steps 100
  --save_total_limit 20
  --ddp_find_unused_parameters False
  # 7200 s, up from the 1800 s default. REQUIRED under DDP, not tuning: the eval callback
  # generates on rank 0 only (callbacks.py:111 `if self.state.is_main_process`), so rank 1
  # runs ahead and blocks inside the gradient allreduce for the whole eval. Measured on the
  # single-GPU run, step 1739 -> 1740 took 18:17:36 -> 18:39:09 = 21.5 min of eval + save.
  # That is inside 30 min today but with no room, and the eval only gets slower as
  # predictions lengthen -- a timeout here kills the job with a NCCL watchdog abort.
  --ddp_timeout 7200
  --eval_strategy no
  --report_to tensorboard
  --overfit False
  --val_size 5
  --vsibench_eval_enable
  # Left at 870 on purpose, NOT halved with save_steps. It is a 16384-token greedy decode
  # x 5 samples, it is the expensive thing in the loop, and it currently reports f1 0.0 with
  # a 250-box repetition collapse -- low information per unit of wall clock.
  --vsibench_eval_steps 870
  --vsibench_max_eval_samples 5
  # Per process, so 16 workers total. The node has 128 cores and load average ~3.7.
  --dataloader_num_workers 8
)


accelerate launch --config_file common/multi_gpu.yaml --num_processes "${NPROC_PER_NODE}" --main_process_port 0 idea_4a_sg_perc_ca1m_vgjson/train.py "${args[@]}" "$@"
