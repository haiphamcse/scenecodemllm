# Jean Zay notes

Short list of what has bitten before. The scripts already handle all of it.

- **Pick the GPU with `-C h100|a100`, never `--partition`.** A partition flag trips the
  IDRIS submit filter (`Account xyz@h100 ----- Job type v100`). The account must match too:
  `--account=<project>@h100`.
- **No `--mem`.** It is rejected. RAM scales with `--cpus-per-task`, which is 24 per H100 and 8 per
  A100. Training decodes video in 8 dataloader workers per GPU, so keep the 24 cores per H100.
- **QoS wall times.** `-dev` QoS: 2 h, and at most ~10 jobs per user (larger arrays are
  rejected). `-t3`: 20 h. `-t4`: 100 h, lower priority. Training uses 4 h windows on
  `qos_gpu_h100-t3`.
- **H100 jobs must re-init modules.** A job inherits the login node's `MODULEPATH`. Under it,
  `pytorch-gpu` resolves to a build whose OpenMPI is missing on H100 nodes (`libmpi.so.40` at
  `import torch`). The fix: `unset MODULEPATH; source /etc/profile; module purge; module load
  arch/h100; module load pytorch-gpu/py3/<v>`. Source `/etc/profile` outside `set -u`. A venv is
  tied to the module tree it was built on, so build one per GPU kind (`setup_env.sh`).
- **`transformer_engine` in the 2.7+ modules** raises at import, and peft probes it. `setup_env.sh`
  shadows it with an empty module.
- **Compute nodes are offline.** Fetch everything on a login node first (`download_weights.sh`).
  Jobs run with `HF_HUB_OFFLINE=1` so that a missing file fails at once instead of hanging.
- **Locale.** Compute nodes drop UTF-8, and TRL's model card then kills every checkpoint save
  (`UnicodeDecodeError 0xc3`), after the step itself succeeded. `lib.sh` sets `PYTHONUTF8=1`.
- **`$SCRATCH` is purged** after about 30 days without access. Videos and results live there,
  so copy finished checkpoints you care about to `$WORK` or `$STORE`. Weights and venvs live on
  `$WORK`.
- **Inode quotas.** `$WORK` allows 500k inodes per project (a conda env costs about 150k; a module
  venv about 2k). Check with `idr_quota_user` / `idr_quota_project`.
- **Lustre is metadata-bound.** Many small files take hours to create or rsync, and large video
  reads are bandwidth-bound. If the dataloader starves (GPU idle between steps), lower
  `--dataloader_num_workers` before raising it: 12 workers per rank once exhausted decord's EOF
  retries (`DECORD_EOF_RETRY_MAX` is raised in `lib.sh`).
- **HF cache files are symlinks into `blobs/`.** When copying one snapshot file, use `rsync -aL`.
- **Login-node watchers** (`train.sh`, `eval.sh`) run under `nohup bash -l`, because a non-login
  shell lacks the Slurm and `$WORK` environment. There are several login nodes (jean-zay1..5).
  `ps` and `kill` only see the node you are on, so note the hostname the script prints.
- **Stop a watcher with `kill <pid>`** on its own login node. Never use `pkill -f`: on a shared
  login node it matches other users' and your other sessions' processes.
- **`squeue` can fail transiently** ("Invalid user"). The watchers treat a failed `squeue` as
  "unknown", not as "no jobs", which once caused a double submission.
- **NCCL.** If `nvidia-smi topo -m` (printed in every training log) shows `SYS` between the GPUs,
  resubmit with `NCCL_P2P_DISABLE=1`.
- **Resume LR trap.** `--learning_rate` has no effect when resuming an HF checkpoint, because the
  saved scheduler and optimizer win. Never reuse a run dir for a different schedule.
