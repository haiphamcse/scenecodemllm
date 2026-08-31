"""Qwen3-VL SFT, ScanNet-v2 3D detection in VG-LLM's 9-DoF JSON format (latent-only).

Fork of idea_4a_sg_perc_ca1m_vgjson/train.py with the DATASET swapped and the model left
alone. Same architecture: VGGTOmega (frozen) -> two Perceiver IO encoders -> latents
scattered into ``<|quad_start|>``/``<|quad_end|>`` placeholders (model_with_vggt.py),
MLLM sees no pixels. Each example is a 2-turn conversation: the LLM emits the fenced
```json box list (turn 2, supervised) from the 3D latents.

Differences from idea_4a_sg_perc_ca1m_vgjson:
- Source is VG-LLM's own ScanNet json (``scannet_det_train_4frames.json``, 144k rows),
  not a per-scene corpus dir. One row = one 4-frame clip, target already in VG-LLM
  format, boxes in the camera frame of image[0].
- 4 frames per sample, not 64.
- Images live under ``<image_root>/scannet/posed_images/<scene>/<frame>.jpg``. The
  ScanNet download is partial, so rows whose frames are not on disk are dropped.
- Eval (non-overfit) is VG-LLM's held-out val json, not a tail split of train.

Scored the same way: VG-LLM's detection metric (graph_vgllm.compare, IoU 0.25 P/R/F1).

By default this runs an OVERFIT sanity check: one clip from each of
``--overfit_num_samples`` DISTINCT scenes, train == eval, so a healthy setup drives
train loss toward zero and precision/recall/f1 toward 1.0. Pass ``--overfit False`` to
train on every usable row.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_4a_sg_perc_scannetv2_vgjson/train.py --output_dir ./results/scannetv2
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoProcessor, HfArgumentParser, Trainer
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer

_FINETUNING_ROOT = Path(__file__).resolve().parent.parent
_COMMON = _FINETUNING_ROOT / "common"
_THIS_DIR = Path(__file__).resolve().parent
# Local modules (collator, callbacks, model_with_vggt, graph_vgllm) must shadow the
# common package: remove any existing entry then re-insert so _THIS_DIR wins.
for _p in (_COMMON, _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)

from argument import (  # common
    DataArguments,
    ModelArguments,
    TrainingArguments,
    VsibenchEvalArguments,
    DEFAULT_HF_HOME,
)
from callbacks import SceneGraphReconCallback
from collator import make_collator
from graph_vgllm import canonicalize, parse
from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

logging.getLogger("decord").disabled = True

_DEFAULT_VGGT_CHECKPOINT = (
    "/home/ducpham/scratch/Working/cache/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)
_DATA = Path("/home/ducpham/scratch/Working/dataset")
_DEFAULT_TRAIN_JSON = str(_DATA / "vgllm_data/train/scannet_det_train_4frames.json")
_DEFAULT_VAL_JSON = str(
    _DATA / "vgllm_data/evaluation/threedod/scannet/scannet_det_val_4frames.json"
)
# The rows an overfit run trained on, dumped so a continuation replays them exactly.
_SCENE_LIST = "overfit_scenes.json"


@dataclass
class ScanNetArguments:
    """VG-LLM ScanNet json + the posed_images tree its paths are relative to."""

    train_json: str = field(default=_DEFAULT_TRAIN_JSON)
    val_json: str = field(default=_DEFAULT_VAL_JSON)
    image_root: str = field(
        default=str(_DATA),
        metadata={"help": "Root the json's 'scannet/posed_images/...' paths hang off."},
    )
    overfit: bool = field(
        default=True,
        metadata={"help": "Train on N clips from N distinct scenes and eval on the same."},
    )
    overfit_num_samples: int = field(default=20)
    val_size: int = field(
        default=50,
        metadata={"help": "Val-json rows given to the eval callback when overfit is off."},
    )
    max_graph_tokens: int = field(default=4096)
    min_boxes: int = field(
        default=5,
        metadata={"help": "Drop rows with fewer boxes (degenerate targets)."},
    )
    init_from: str = field(
        default="",
        metadata={"help": "Run dir whose saved weights initialise this run (fresh optimizer)."},
    )
    box_noise: float = field(
        default=0.0,
        metadata={
            "help": "Uniform +/-fraction perturbation applied to every one of the 9 box "
                    "values, resampled per epoch in the collator (0.005 = +/-0.5%). "
                    "0 disables. Training targets only -- eval GT is never perturbed."
        },
    )
    init_lora_from: str = field(
        default="",
        metadata={
            "help": "LoRA checkpoint dir whose adapter + vggt_projector initialise this run "
                    "(fresh optimizer). Use instead of --init_from for LoRA checkpoints."
        },
    )


@dataclass
class VggtPerceiverArguments:
    """VGGT encoder + Perceiver latent / placeholder settings (same as idea_3i)."""

    vggt_checkpoint: str = field(default=_DEFAULT_VGGT_CHECKPOINT)
    vggt_embed_dim: int = field(default=2048)
    frame_num_latents: int = field(default=128)
    camera_num_latents: int = field(default=32)
    frame_widening_factor: int = field(default=1, metadata={"help": "frame_encoder cross+self MLP widening (default 1 = baseline)."})
    frame_placeholder_token: str = field(default="<|quad_start|>")
    camera_placeholder_token: str = field(default="<|quad_end|>")
    train_perceiver_only: bool = field(default=False)
    full_finetune: bool = field(default=False)
    perceiver_lr: Optional[float] = field(default=None)


def _frames_on_disk(image_root: Path) -> Dict[str, set]:
    """{scene id: {frame filename}} for every non-empty posed_images dir.

    The ScanNet download here is partial (~228 of 1513 scene dirs have jpgs at the time
    of writing), so most json rows point at files that do not exist. One listdir per
    scene builds the whole existence table; filtering 144k rows is then set lookups
    rather than 576k stat() calls on lustre.
    """
    base = Path(image_root) / "scannet" / "posed_images"
    on_disk = {}
    for d in sorted(base.iterdir()):
        if d.is_dir():
            names = set(os.listdir(d))
            if names:
                on_disk[d.name] = names
    return on_disk


def scannet_samples(
    json_path: Path,
    image_root: Path,
    tokenizer=None,
    min_boxes: int = 1,
    max_target_tokens: int | None = None,
) -> List[Dict[str, Any]]:
    """VG-LLM ScanNet json -> usable rows [{images, graph, scene, n_boxes}].

    The target is the assistant turn read verbatim, then canonicalised (leading
    ``{"n": N}``, objects largest-first) -- the same two conventions the CA-1M variant
    trains, which exist to make sequence length learnable.

    Three filters:

    - missing frames: rows whose 4 jpgs are not all downloaded.
    - ``min_boxes``: degenerate short targets, where "emit one box" scores non-zero and
      greedy decoding finds it.
    - ``max_target_tokens``: rows whose target does not fit the generation budget.
      truncate_graph_text would clip them mid-object, leaving a target with no closing
      bracket and no EOS -- i.e. training the model never to terminate.
    """
    on_disk = _frames_on_disk(Path(image_root))
    rows = json.loads(Path(json_path).read_text())
    samples: List[Dict[str, Any]] = []
    dropped = {"missing": 0, "small": 0, "long": 0}
    for row in rows:
        rels = row["images"]
        scene = Path(rels[0]).parent.name
        names = on_disk.get(scene)
        if names is None or not all(Path(r).name in names for r in rels):
            dropped["missing"] += 1
            continue
        text = canonicalize(row["conversations"][1]["value"])
        n = len(parse(text))
        if n < min_boxes:
            dropped["small"] += 1
            continue
        if max_target_tokens is not None and tokenizer is not None:
            if len(tokenizer.encode(text, add_special_tokens=False)) > max_target_tokens:
                dropped["long"] += 1
                continue
        samples.append({
            "images": [str(Path(image_root) / r) for r in rels],
            "graph": text,
            "scene": scene,
            "n_boxes": n,
        })
    if not samples:
        raise RuntimeError(
            f"No usable rows in {json_path} (dropped {dropped}); are the ScanNet "
            f"posed_images under {image_root} downloaded?"
        )
    logging.info(
        "%s: %d usable rows over %d scenes (dropped %d missing frames, %d with <%d "
        "boxes, %d over the token budget).",
        Path(json_path).name, len(samples), len({s["scene"] for s in samples}),
        dropped["missing"], dropped["small"], min_boxes, dropped["long"],
    )
    return samples


def one_per_scene(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """First clip of each scene -- an overfit set of N scenes, not N clips of one scene."""
    seen = set()
    out = []
    for s in samples:
        if s["scene"] not in seen:
            seen.add(s["scene"])
            out.append(s)
    return out


def stratified_sample(scenes: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """k samples spanning the box-count range, so the sample probes every length.

    Taking scenes[:k] instead gives whatever the json order yields; a CA-1M overfit drew
    [1, 1, 1, 24, ..., 123] that way -- three degenerate scenes plus two over the token
    budget, which is most of why it collapsed to a one-box output.
    """
    if k >= len(scenes):
        return scenes
    by_len = sorted(scenes, key=lambda s: s["n_boxes"])
    idx = sorted({round(i * (len(by_len) - 1) / (k - 1)) for i in range(k)})
    return [by_len[i] for i in idx]


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
        modules_to_save=["vggt_projector"],
    )


def state_dict_without_vggt_encoder(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop frozen VGGTOmega encoder weights; keep Qwen VL + LoRA + vggt_projector."""
    return {k: v for k, v in state_dict.items() if "vggt_model" not in k.split(".")}


def is_perceiver_param(name: str) -> bool:
    return "vggt_projector" in name.split(".")


def perceiver_param_groups(named, decay_names, perceiver_lr, base_lr, weight_decay):
    """Split trainable params into (perceiver | rest) x (decay | no-decay) groups."""
    groups = []
    for perceiver, lr in ((True, perceiver_lr), (False, base_lr)):
        for in_decay, wd in ((True, weight_decay), (False, 0.0)):
            params = [
                p for n, p in named
                if is_perceiver_param(n) is perceiver and (n in decay_names) is in_decay
            ]
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": wd})
    return groups


class SFTTrainerSaveLlMAndProjectorOnly(SFTTrainer):
    """Omit frozen VGGTOmega weights from checkpoints (see idea_3i for rationale)."""

    projector_only: bool = False
    perceiver_lr: Optional[float] = None

    def create_optimizer(self):
        if self.optimizer is None and self.perceiver_lr is not None:
            decay = set(self.get_decay_parameter_names(self.model))
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            groups = perceiver_param_groups(
                named, decay, self.perceiver_lr, self.args.learning_rate, self.args.weight_decay,
            )
            cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
            kwargs.pop("lr", None)
            self.optimizer = cls(groups, lr=self.args.learning_rate, **kwargs)
            n_perc = sum(p.numel() for n, p in named if "vggt_projector" in n.split("."))
            n_rest = sum(p.numel() for n, p in named if "vggt_projector" not in n.split("."))
            logging.info(
                "Optimizer: Perceiver lr=%g (%d params) | rest lr=%g (%d params)",
                self.perceiver_lr, n_perc, self.args.learning_rate, n_rest,
            )
        return super().create_optimizer()

    def _save(self, output_dir: str | None = None, state_dict: Dict[str, Any] | None = None) -> None:
        if state_dict is None:
            unwrapped = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            state_dict = unwrapped.state_dict()
        if self.projector_only:
            state_dict = {k: v for k, v in state_dict.items() if "vggt_projector" in k.split(".")}
        else:
            state_dict = state_dict_without_vggt_encoder(state_dict)
        return super()._save(output_dir, state_dict=state_dict)


def main() -> None:
    setup_logging()

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            VsibenchEvalArguments,
            ScanNetArguments,
            VggtPerceiverArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        vsibench_args,
        sn_args,
        vggt_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    configure_hf_home(training_args)

    output_dir = Path(training_args.output_dir)
    image_root = Path(sn_args.image_root)
    frame_placeholder_text = vggt_args.frame_placeholder_token * vggt_args.frame_num_latents
    camera_placeholder_text = vggt_args.camera_placeholder_token * vggt_args.camera_num_latents
    output_dir.mkdir(parents=True, exist_ok=True)

    # Processor first: scannet_samples needs its tokenizer to drop rows whose target
    # would be truncated by the generation budget.
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    def load(json_path):
        return scannet_samples(
            Path(json_path),
            image_root,
            tokenizer=processor.tokenizer,
            min_boxes=sn_args.min_boxes,
            max_target_tokens=sn_args.max_graph_tokens,
        )

    if sn_args.overfit:
        # One clip per scene first, so N samples means N different rooms.
        # The ScanNet download is still filling in, so the usable set grows between runs
        # and a fresh stratified_sample would pick DIFFERENT scenes -- which turns
        # "continue overfitting" into "fine-tune on partly unseen data". A run started
        # from another run's weights replays that run's rows verbatim (image paths and
        # target included), so a partially-downloaded scene cannot swap clips either.
        pinned = Path(sn_args.init_from) / _SCENE_LIST if sn_args.init_from else None
        if pinned is not None and pinned.exists():
            train_rows = json.loads(pinned.read_text())
            logging.info("Overfit set replayed from %s (%d rows).", pinned, len(train_rows))
        else:
            train_rows = stratified_sample(
                one_per_scene(load(sn_args.train_json)), sn_args.overfit_num_samples
            )
        eval_rows = train_rows
        (output_dir / _SCENE_LIST).write_text(json.dumps(train_rows))
        logging.info(
            "OVERFIT MODE: train == eval on %d ScanNet scenes, box counts %s.",
            len(train_rows), [s["n_boxes"] for s in train_rows],
        )
    else:
        train_rows = load(sn_args.train_json)
        eval_rows = one_per_scene(load(sn_args.val_json))[: max(1, sn_args.val_size)]
        logging.info("Train: %d rows | eval: %d val-json rows.", len(train_rows), len(eval_rows))

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    dtype = torch.bfloat16 if training_args.bf16 else None
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
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

    if sn_args.init_from:
        # Continue from a finished run's weights with a fresh optimizer/schedule. Not
        # --resume_from_checkpoint: that restores the trainer state too, so a finished
        # run resumes at its last step and stops immediately. Must come after
        # initialize_vggt, which is what creates vggt_projector.
        state = {}
        for shard in sorted(Path(sn_args.init_from).glob("*.safetensors")):
            state.update(load_file(shard))
        if not state:
            raise FileNotFoundError(f"No *.safetensors under {sn_args.init_from}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.tie_weights()
        # Two kinds of missing key are expected: the frozen VGGT encoder, stripped on
        # save, and tied weights (lm_head.weight aliases the input embedding, so
        # safetensors stores it once). Anything else -- a projector key above all --
        # would mean a randomly initialised module.
        tied = set(getattr(model, "_tied_weights_keys", None) or [])
        stale = [k for k in missing if "vggt_model" not in k.split(".") and k not in tied]
        if stale or unexpected:
            raise RuntimeError(f"init_from mismatch: missing {stale[:5]}, unexpected {list(unexpected)[:5]}")
        logging.info("init_from %s: loaded %d tensors.", sn_args.init_from, len(state))

    training_args.eval_strategy = "no"
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.remove_unused_columns = False
    if training_args.logging_dir is None:
        training_args.logging_dir = str(output_dir / "tensorboard")

    if vggt_args.train_perceiver_only and vggt_args.full_finetune:
        raise ValueError("train_perceiver_only and full_finetune are mutually exclusive.")
    if vggt_args.train_perceiver_only:
        peft_config = None
        for name, param in model.named_parameters():
            param.requires_grad = "vggt_projector" in name.split(".")
        logging.info("PERCEIVER-ONLY: LoRA disabled; LLM frozen, training vggt_projector only.")
    elif vggt_args.full_finetune:
        peft_config = None
        for name, param in model.named_parameters():
            param.requires_grad = not ({"vggt_model", "visual"} & set(name.split(".")))
        logging.info("FULL-FT: LoRA disabled; training LLM + vggt_projector; VGGT + visual frozen.")
    else:
        peft_config = build_lora_config(model_args)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        model.enable_input_require_grads()

    # Frames and target both travel with the row; the roots below are inert (kept so the
    # collator/callback signatures match the CA-1M variant).
    collator = make_collator(
        processor,
        data_root=image_root,
        cache_root=image_root,
        frame_placeholder_text=frame_placeholder_text,
        camera_placeholder_text=camera_placeholder_text,
        scene_graph_root=image_root,
        max_graph_tokens=sn_args.max_graph_tokens,
        image_patch_size=data_args.image_patch_size,
        box_noise=sn_args.box_noise,
    )

    callbacks = []
    if vsibench_args.vsibench_eval_enable:
        eval_log = (
            Path(vsibench_args.vsibench_eval_log)
            if vsibench_args.vsibench_eval_log
            else output_dir / "scannet_eval.txt"
        )
        callbacks.append(
            SceneGraphReconCallback(
                model=model,
                processor=processor,
                eval_dataset=eval_dataset,
                data_root=image_root,
                cache_root=image_root,
                frame_placeholder_text=frame_placeholder_text,
                camera_placeholder_text=camera_placeholder_text,
                log_path=eval_log,
                eval_steps=vsibench_args.vsibench_eval_steps,
                scene_graph_root=image_root,
                image_patch_size=data_args.image_patch_size,
                max_graph_tokens=sn_args.max_graph_tokens,
                max_eval_samples=vsibench_args.vsibench_max_eval_samples,
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
    )
    trainer.projector_only = vggt_args.train_perceiver_only
    trainer.perceiver_lr = vggt_args.perceiver_lr

    for cb in callbacks:
        if isinstance(cb, SceneGraphReconCallback):
            cb.model = trainer.model

    if sn_args.init_lora_from:
        # Continue from a LoRA run's weights with a FRESH optimizer and schedule.
        # Not --init_from: that loads *.safetensors into the base model and raises on any
        # unexpected key, and every key in a LoRA checkpoint is base_model.model.*.lora_*.
        # Not --resume_from_checkpoint either: that restores the step counter and Adam state
        # of a run whose prompt shape (no video) no longer matches this one.
        # Must come after the trainer wraps the model with peft_config -- trainer.model is
        # the PeftModel these keys belong to.
        from peft import set_peft_model_state_dict

        adapter = Path(sn_args.init_lora_from) / "adapter_model.safetensors"
        if not adapter.exists():
            raise FileNotFoundError(f"No adapter_model.safetensors under {sn_args.init_lora_from}")
        lora_state = load_file(str(adapter))
        outcome = set_peft_model_state_dict(trainer.model, lora_state)
        unexpected = list(getattr(outcome, "unexpected_keys", []) or [])
        if unexpected:
            raise RuntimeError(f"init_lora_from: unexpected keys {unexpected[:5]}")
        # A silent no-op here would look exactly like a successful cold start, so require
        # that the projector (modules_to_save) and the adapters both actually arrived.
        n_lora = sum(1 for k in lora_state if "lora_" in k)
        n_proj = sum(1 for k in lora_state if "vggt_projector" in k)
        if not n_lora or not n_proj:
            raise RuntimeError(
                f"init_lora_from matched {n_lora} lora and {n_proj} projector tensors; "
                f"expected both to be non-zero."
            )
        logging.info(
            "init_lora_from %s: loaded %d tensors (%d lora, %d vggt_projector).",
            sn_args.init_lora_from, len(lora_state), n_lora, n_proj,
        )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    logging.info("Trainable params: %d / %d (%.2f%%)", trainable, total, 100.0 * trainable / total)

    # "latest"/True: resume from the newest checkpoint if one exists, else start fresh.
    # Plain True would make HF raise on the first run (empty output_dir).
    resume_ckpt = training_args.resume_from_checkpoint
    if resume_ckpt in (True, "latest", "True"):
        resume_ckpt = get_last_checkpoint(str(output_dir)) if output_dir.exists() else None

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))


if __name__ == "__main__":
    main()
