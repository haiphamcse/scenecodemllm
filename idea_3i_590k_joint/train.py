"""Qwen3-VL SFT with VGGT-Perceiver placeholder scatter (idea_3i_590k).

Fork of idea_3i that decodes video live instead of reading a pre-extracted frame
cache, so it can train on all 374148 VSI-590K video rows rather than the 856
scannetppv2 videos that happened to be cached. Nothing needs pre-building.

VGGTOmega (frozen) features are compressed by two Perceiver IO encoders and
scattered into ``<|quad_start|>``/``<|quad_end|>`` placeholder tokens
(see model_with_vggt.py). Trainable: the two Perceivers (``vggt_projector``)
+ LoRA adapters on the LLM.

VIDEO ROWS ONLY. VSI-590K also holds 216519 image rows (hypersim, ytb_roomtour,
robotics) keyed ``image`` rather than ``video``; they are dropped here, with the
count logged rather than silently discarded.

By default this runs an *overfitting sanity check*: it trains on the first
``--overfit_num_samples`` VSI-Bench (ScanNet++) eval rows and evaluates on those
same rows, so a healthy setup should drive train loss toward zero and VSI-Bench
accuracy toward 1.0.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i_590k/train.py --output_dir ./results/idea_3i_590k
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import LoraConfig
from safetensors.torch import load_file
from datasets import Dataset
from transformers import AutoProcessor, HfArgumentParser, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# Put idea_3i-local modules (collator, callbacks, model_with_vggt) ahead of the
# common package so they shadow the common versions. Remove any existing entry
# first and re-insert: running as a script auto-adds _THIS_DIR, so a plain
# "not in sys.path" guard would skip it and let _COMMON land in front instead.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    VsibenchEvalArguments,
    DEFAULT_HF_HOME,
)
from callbacks import VsibenchMetricsCallback
from collator import make_collator
from data import build_vsibench_eval_dataset

# VSI-590K's video-bearing sources. The other three (hypersim, ytb_roomtour,
# robotics) are single images and are not trainable by this architecture here.
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

logging.getLogger("decord").disabled = True

_DEFAULT_VGGT_CHECKPOINT = (
    os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)


@dataclass
class JointArguments:
    """Knobs that exist only because this run mixes two objectives."""

    box_noise: float = field(
        default=0.005,
        metadata={"help": "Scale each det box value by 1 +/- U(box_noise). det rows only; "
                          "0.0 disables, which is what the 3DOD-only runs used."},
    )

    color_jitter: float = field(
        default=0.0,
        metadata={"help": "Probability of applying SeqColorJitter to a det row, at "
                          "cambrian-p's strength (brightness/contrast/saturation 0.5, hue "
                          "0.1), one draw shared by the whole clip. det rows only."},
    )


@dataclass
class OverfitArguments:
    """Controls the overfitting sanity run (train == a few eval rows)."""

    overfit_on_eval: bool = field(
        default=True,
        metadata={"help": "Train on the first N VSI-Bench eval rows (memorization test)."},
    )
    overfit_num_samples: int = field(
        default=20,
        metadata={"help": "Number of eval rows to overfit on when overfit_on_eval is set."},
    )


@dataclass
class VggtPerceiverArguments:
    """VGGT encoder + Perceiver latent / placeholder settings."""

    vggt_checkpoint: str = field(default=_DEFAULT_VGGT_CHECKPOINT)
    vggt_embed_dim: int = field(default=2048, metadata={"help": "VGGTOmega token dim."})
    frame_num_latents: int = field(default=128, metadata={"help": "frame_encoder latents (= placeholders)."})
    camera_num_latents: int = field(default=32, metadata={"help": "camera_encoder latents (= placeholders)."})
    frame_placeholder_token: str = field(default="<|quad_start|>")
    camera_placeholder_token: str = field(default="<|quad_end|>")
    frame_widening_factor: int = field(default=1, metadata={"help": "frame_encoder cross+self MLP widening (default 1 = baseline)."})
    vggt_image_resolution: int = field(
        default=256,
        metadata={"help": "Balanced-resize target for the VGGT branch. Forced to 256 for "
                          "every VGGT-Omega checkpoint, not matched to the variant."},
    )
    init_projector: str = field(
        default="",
        metadata={"help": "vggt_projector.safetensors from merge_lora.py. Loaded AFTER "
                          "initialize_vggt, which builds a fresh random projector."},
    )
    perceiver_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Separate LR for the Perceivers (vggt_projector). None = use --learning_rate for all."},
    )


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def configure_hf_home(training_args: TrainingArguments) -> None:
    if training_args.hf_home:
        os.environ["HF_HOME"] = training_args.hf_home
    else:
        os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)


def load_mix(path: Path) -> Dataset:
    """The build_mix.py manifest, as-is.

    A plain jsonl read rather than data.build_train_dataset: that helper filters VSI-590K
    by source and knows nothing about det rows, and the mix is already exactly the corpus
    to train on. Kept as a datasets.Dataset so the Trainer's sampler/resume behave as they
    do everywhere else in this repo.
    """
    rows = [json.loads(line) for line in open(path)]
    if not rows:
        raise ValueError(f"{path} is empty. Run build_mix.py first.")
    missing = {r.get("task") for r in rows} - {"vqa", "det"}
    if missing:
        raise ValueError(f"{path} has rows with unknown task {missing}; rebuild the mix.")
    # Column-per-key, so absent fields (a vqa row has no `images`) become None instead of
    # a schema clash.
    keys = {k for r in rows for k in r}
    return Dataset.from_dict({k: [r.get(k) for r in rows] for k in keys})


def build_lora_config(model_args: ModelArguments) -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # Train and checkpoint the two Perceivers alongside the LoRA adapter
        # (otherwise PEFT saves only LoRA keys and the projector is dropped).
        modules_to_save=["vggt_projector"],
    )


def state_dict_without_vggt_encoder(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop frozen VGGTOmega encoder weights; keep Qwen VL + LoRA + vggt_projector."""
    return {k: v for k, v in state_dict.items() if "vggt_model" not in k.split(".")}


class SFTTrainerSaveLlMAndProjectorOnly(SFTTrainer):
    """Omit frozen VGGTOmega weights from checkpoints; optional separate Perceiver LR."""

    def __init__(self, *args, perceiver_lr: Optional[float] = None, **kwargs):
        self.perceiver_lr = perceiver_lr
        super().__init__(*args, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict: Dict[str, Any] | None = None) -> None:
        if state_dict is None:
            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            state_dict = unwrapped.state_dict()
        state_dict = state_dict_without_vggt_encoder(state_dict)
        return super()._save(output_dir, state_dict=state_dict)

    def create_optimizer(self):
        """Same as HF Trainer, but put vggt_projector params in their own LR group."""
        if self.perceiver_lr is None or self.optimizer is not None:
            return super().create_optimizer()

        opt_model = self.model
        decay = set(self.get_decay_parameter_names(opt_model))
        is_perc = lambda n: "vggt_projector" in n
        named = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
        groups = [
            {"params": [p for n, p in named if n in decay and not is_perc(n)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in named if n not in decay and not is_perc(n)],
             "weight_decay": 0.0},
            {"params": [p for n, p in named if n in decay and is_perc(n)],
             "weight_decay": self.args.weight_decay, "lr": self.perceiver_lr},
            {"params": [p for n, p in named if n not in decay and is_perc(n)],
             "weight_decay": 0.0, "lr": self.perceiver_lr},
        ]
        groups = [g for g in groups if g["params"]]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        base_lr = optimizer_kwargs.get("lr", self.args.learning_rate)
        logging.info(
            "Optimizer LR groups: %s (base_lr=%s, perceiver_lr=%s)",
            [(g.get("lr", base_lr), sum(p.numel() for p in g["params"])) for g in groups],
            base_lr, self.perceiver_lr,
        )
        return self.optimizer


def main() -> None:
    setup_logging()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            VsibenchEvalArguments,
            JointArguments,
            OverfitArguments,
            VggtPerceiverArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        vsibench_args,
        joint_args,
        overfit_args,
        vggt_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    configure_hf_home(training_args)

    output_dir = Path(training_args.output_dir)
    data_root = Path(data_args.data_root)
    frame_placeholder_text = vggt_args.frame_placeholder_token * vggt_args.frame_num_latents
    camera_placeholder_text = vggt_args.camera_placeholder_token * vggt_args.camera_num_latents
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset = build_vsibench_eval_dataset(
        hf_home=training_args.hf_home or os.environ.get("HF_HOME"),
        max_samples=(
            overfit_args.overfit_num_samples
            if overfit_args.overfit_on_eval
            else vsibench_args.vsibench_max_eval_samples
        ),
        chosen_dataset="scannet"
    )
    logging.info(
        "VSI-Bench eval rows: %s | live decode (%s fps, max %s frames)",
        len(eval_dataset), data_args.video_fps, data_args.video_max_frames,
    )

    if overfit_args.overfit_on_eval:
        train_dataset = eval_dataset
        logging.info("OVERFIT MODE: training on %d VSI-Bench eval rows (train == eval).", len(train_dataset))
    else:
        # build_mix.py already did the source filtering, the budgeting and the shuffle,
        # so there is nothing to select here -- the mix IS the corpus. Rows arrive
        # pre-stamped with `task`, which the collator dispatches on.
        train_dataset = load_mix(Path(data_args.jsonl_path))
        counts = Counter(r["task"] for r in train_dataset)
        logging.info(
            "Train rows: %d | mix=%s | shares=%s",
            len(train_dataset), dict(counts),
            {k: f"{100 * v / len(train_dataset):.1f}%" for k, v in counts.items()},
        )
        if len(counts) < 2:
            logging.warning("Mix carries only %s -- this is not a joint run.", set(counts))

    dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model.initialize_vggt(
        vggt_args.vggt_checkpoint,
        tokenizer=processor.tokenizer,
        vggt_embed_dim=vggt_args.vggt_embed_dim,
        frame_num_latents=vggt_args.frame_num_latents,
        camera_num_latents=vggt_args.camera_num_latents,
        frame_placeholder_token=vggt_args.frame_placeholder_token,
        camera_placeholder_token=vggt_args.camera_placeholder_token,
        frame_widening_factor=vggt_args.frame_widening_factor,
    )

    if vggt_args.init_projector:
        # initialize_vggt() has just replaced the projector with fresh random weights, so
        # this has to come after it. Putting these tensors in the merged model dir instead
        # would be a silent no-op: they do not exist at from_pretrained() time.
        proj = load_file(vggt_args.init_projector)
        _, unexpected = model.load_state_dict(proj, strict=False)
        landed = [k for k in proj if k not in unexpected]
        if not landed:
            raise RuntimeError(f"init_projector matched nothing in the model: {list(proj)[:3]}")
        logging.info("init_projector %s: loaded %d/%d tensors.",
                     vggt_args.init_projector, len(landed), len(proj))

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    # idea_3i trains the Perceivers (modules_to_save) + LoRA adapters on the LLM.
    peft_config = build_lora_config(model_args)

    collator = make_collator(
        processor,
        data_root=data_root,
        cache_root=None,
        frame_placeholder_text=frame_placeholder_text,
        camera_placeholder_text=camera_placeholder_text,
        video_fps=data_args.video_fps,
        video_max_frames=data_args.video_max_frames,
        image_patch_size=data_args.image_patch_size,
        box_noise=joint_args.box_noise,
        color_jitter=joint_args.color_jitter,
        vggt_image_resolution=vggt_args.vggt_image_resolution,
    )

    callbacks = []
    if vsibench_args.vsibench_eval_enable:
        eval_log = (
            Path(vsibench_args.vsibench_eval_log)
            if vsibench_args.vsibench_eval_log
            else output_dir / "vsibench_eval.txt"
        )
        callbacks.append(
            VsibenchMetricsCallback(
                model=model,
                processor=processor,
                eval_dataset=eval_dataset,
                data_root=data_root,
                frame_placeholder_text=frame_placeholder_text,
                camera_placeholder_text=camera_placeholder_text,
                log_path=eval_log,
                eval_steps=vsibench_args.vsibench_eval_steps,
                image_patch_size=data_args.image_patch_size,
                max_new_tokens=vsibench_args.vsibench_max_new_tokens,
                max_eval_samples=vsibench_args.vsibench_max_eval_samples,
                video_fps=data_args.video_fps,
                video_max_frames=data_args.video_max_frames,
                vggt_image_resolution=vggt_args.vggt_image_resolution,
            )
        )

    trainer = SFTTrainerSaveLlMAndProjectorOnly(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        peft_config=peft_config,
        processing_class=processor,
        callbacks=callbacks,
        perceiver_lr=vggt_args.perceiver_lr,
    )

    for cb in callbacks:
        if isinstance(cb, VsibenchMetricsCallback):
            cb.model = trainer.model

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logging.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    # Liger/memory probe. HF applies liger inside Trainer.__init__, so this has to run
    # after construction. Three OOM postmortems were ambiguous because a flag was
    # accepted but silently not applied; these two lines make that visible.
    # PreTrainedModel.base_model is `getattr(self, base_model_prefix, self)` -- it
    # returns SELF when the prefix attr is missing, and hasattr is always true since
    # it is a property. A `while hasattr(...)` unwrap spins forever (job 1745257).
    _inner = trainer.model
    for _ in range(4):
        _next = getattr(_inner, "base_model", None)
        if _next is None or _next is _inner:
            break
        _inner = _next
    _lf = getattr(_inner, "loss_function", None)
    logging.info("liger probe: use_liger_kernel=%s loss_function=%s forward_owner=%s",
                 getattr(training_args, "use_liger_kernel", None),
                 getattr(_lf, "__qualname__", type(_lf).__name__ if _lf else None),
                 type(_inner).forward.__qualname__)

    class PeakMemCallback(TrainerCallback):
        """Peak allocated bytes per step. The runs died with <2 GB spare, so the
        margin is the number that matters, not pass/fail."""

        def on_step_end(self, args, state, control, **kw):
            if torch.cuda.is_available():
                logging.info("step %d peak_alloc=%.1f GB reserved=%.1f GB",
                             state.global_step,
                             torch.cuda.max_memory_allocated() / 1e9,
                             torch.cuda.max_memory_reserved() / 1e9)
                torch.cuda.reset_peak_memory_stats()

    trainer.add_callback(PeakMemCallback())

    # "latest"/True: resume from the newest checkpoint if one exists, else start
    # fresh. Plain True makes HF raise on the first run (empty output_dir).
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt in (True, "latest", "True"):
        resume_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
