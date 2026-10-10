"""Qwen3-VL LoRA SFT with a frozen VGGT encoder run live on the decoded VSI-590K clips.

    accelerate launch -m sr_vggt.train <configs/*.env flags> --jsonl_path <mix> \
        --data_root <VSI-590K root> --output_dir <run dir> ...

scripts/train.sh builds this command from an experiment config.
Trainable: the VGGT fusion module (``vggt_projector``) + LoRA on the LLM; VGGT itself is frozen.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoProcessor, HfArgumentParser, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from .data import load_mix, make_collator
from .model import VggtArguments, initialize_vggt_modules, model_class


@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    attn_implementation: str = "sdpa"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1


@dataclass
class DataArguments:
    jsonl_path: str = field(default="", metadata={"help": "Training mix (.jsonl or .jsonl.gz, VQA rows)."})
    data_root: str = field(default="", metadata={"help": "VSI-590K root (the rows' `video` paths are relative to it)."})
    max_seq_len: int = field(default=0, metadata={"help": "Drop rows longer than this (0 = keep all)."})
    init_from_checkpoint: str = field(default="", metadata={"help": "First window only: start from "
                                      "this checkpoint's adapter (LoRA + vggt_projector), fresh optimizer."})


class SaveBeforeTimeout(TrainerCallback):
    """Save once, MARGIN_S before the Slurm window ends (SLURM_JOB_END_TIME), so a chained window
    loses at most one step. Rank 0 decides and broadcasts so every rank saves on the same step."""

    MARGIN_S = 300

    def __init__(self, end_time=None):
        self.end_time = end_time if end_time is not None else int(os.environ.get("SLURM_JOB_END_TIME", 0))
        self.done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self.end_time or self.done:
            return control
        near = time.time() > self.end_time - self.MARGIN_S
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(near)], device=args.device)
            dist.broadcast(flag, src=0)
            near = bool(flag.item())
        if near:
            control.should_save = True
            self.done = True
            print(f"[SaveBeforeTimeout] saving at step {state.global_step}", flush=True)
        return control


class CpuBatchSFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        """Keep micro-batches on the CPU until their own training_step.

        Trainer.get_batch_samples pulls the whole accumulation window before running any of it;
        a device-placed loader parks all of it (with ~50-75 MB of raw VGGT input per sample)
        on the GPU and OOMs. Without device placement, _prepare_inputs moves one batch
        at a time.
        """
        placement = self.accelerator.device_placement
        self.accelerator.device_placement = False
        try:
            return super().get_train_dataloader()
        finally:
            self.accelerator.device_placement = placement


def build_lora_config(model_args: ModelArguments) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # Train and checkpoint the VGGT module with the adapter (else PEFT drops it).
        modules_to_save=["vggt_projector"],
    )


def to_dataset(rows) -> Dataset:
    keys = {k for r in rows for k in r}
    return Dataset.from_dict({k: [r.get(k) for r in rows] for k in keys})


def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
                        handlers=[logging.StreamHandler(sys.stdout)])
    parser = HfArgumentParser((ModelArguments, DataArguments, VggtArguments, SFTConfig))
    model_args, data_args, vggt_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_args.data_root) if data_args.data_root else None

    rows = load_mix(data_args.jsonl_path)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    if data_args.max_seq_len:
        from .reasoning import kept_rows
        kept = kept_rows(rows, data_args.max_seq_len, output_dir, processor, data_root, vggt_args)
        rows = [rows[i] for i in kept]
    train_dataset = to_dataset(rows)
    logging.info("Train rows: %d", len(rows))
    logging.info("Train sources: %s", dict(Counter(r.get("source") for r in rows)))
    logging.info("Train question types: %s", dict(Counter(r.get("question_type") for r in rows)))

    model = model_class(vggt_args).from_pretrained(
        model_args.model_name_or_path,
        attn_implementation=model_args.attn_implementation,
        dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    initialize_vggt_modules(model, vggt_args, processor.tokenizer)

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    collator = make_collator(processor, data_root, vggt_args)

    # "latest": resume from the newest checkpoint if one exists, else start fresh.
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt in (True, "latest", "True"):
        resume_ckpt = get_last_checkpoint(str(output_dir))

    peft_config = build_lora_config(model_args)
    if data_args.init_from_checkpoint and resume_ckpt is None:
        # First window only; later windows restore their own adapter via the resume.
        from .reasoning import load_init_adapter
        model = load_init_adapter(model, data_args.init_from_checkpoint)
        peft_config = None

    trainer = CpuBatchSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=[SaveBeforeTimeout()],
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logging.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))
    if torch.cuda.is_available():
        logging.info("peak GPU memory allocated: %.1f GiB", torch.cuda.max_memory_allocated() / 2 ** 30)


if __name__ == "__main__":
    main()
