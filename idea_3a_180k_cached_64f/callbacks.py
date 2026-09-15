"""In-training VSI-Bench generation metrics (plain Qwen3-VL, frames decoded live)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import PartialState
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoProcessor, TrainerCallback

from collator import load_frames, task_messages
from vsibench_metrics import AGGREGATORS, vsibench_process_results

from qwen_vl_utils import process_vision_info


def _unwrap_for_generate(model):
    """Peel DDP wrappers so `.generate()` hits the underlying module."""
    while hasattr(model, "module") and not hasattr(model, "generate"):
        model = model.module
    return model


class VsibenchMetricsCallback(TrainerCallback):
    """Generation-based VSI-Bench metrics on a fixed step interval."""

    def __init__(
        self,
        model,
        processor: AutoProcessor,
        eval_dataset: Dataset,
        data_root: Path,
        log_path: Path,
        eval_steps: int,
        image_patch_size: int = 16,
        max_new_tokens: int = 16,
        max_eval_samples: Optional[int] = None,
        video_fps: float = 1.0,
        video_max_frames: int = 32,
    ):
        self.model = model
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.data_root = data_root
        self.log_path = Path(log_path)
        self.eval_steps = max(1, int(eval_steps))
        self.image_patch_size = image_patch_size
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.max_new_tokens = max_new_tokens
        self.max_eval_samples = max_eval_samples
        self.state = PartialState()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _eval_indices(self) -> range:
        n = len(self.eval_dataset)
        if self.max_eval_samples is not None:
            n = min(n, int(self.max_eval_samples))
        return range(n)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step % self.eval_steps != 0:
            return control
        return self._run_vsibench_eval(step, control)

    def _run_vsibench_eval(self, step: int, control):
        if self.state.is_main_process:
            model = _unwrap_for_generate(self.model)
            was_training = model.training
            model.eval()
            device = next(model.parameters()).device

            processed_docs: List[Dict[str, Any]] = []
            out_lines: List[str] = [f"step {step}"]

            eval_indices = list(self._eval_indices())
            with torch.no_grad():
                for idx in tqdm(
                    eval_indices,
                    desc=f"VSI-Bench eval (step {step})",
                    unit="sample",
                ):
                    # VSI-Bench is a VQA benchmark, so the eval prompt is always the
                    # vqa turn regardless of what the training mix holds.
                    example = dict(self.eval_dataset[idx], task="vqa")
                    frames = load_frames(example, self.data_root,
                                         self.video_fps, self.video_max_frames)
                    messages = task_messages(
                        example, frames, target_text="", prompt_only=True,
                    )
                    prompt = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    _, video_inputs, processed_video_kwargs = process_vision_info(
                        messages,
                        return_video_kwargs=True,
                        image_patch_size=self.image_patch_size,
                        return_video_metadata=True,
                    )
                    if not video_inputs:
                        out_lines.append(f"{idx}\tskip")
                        continue

                    # Decode metadata is real; nothing to correct.
                    video_inputs, video_metadata_list = map(list, zip(*video_inputs))
                    inputs = self.processor(
                        text=prompt,
                        videos=video_inputs,
                        video_metadata=video_metadata_list,
                        return_tensors="pt",
                        padding=True,
                        **processed_video_kwargs,
                    )
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )

                    input_len = inputs["input_ids"].shape[1]
                    gen_ids = outputs[0, input_len:]
                    prediction = self.processor.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    ).strip()
                    gt = example["ground_truth"]
                    processed_docs.append(
                        vsibench_process_results(
                            {
                                "question_type": example["question_type"],
                                "ground_truth": gt,
                            },
                            [prediction],
                        )["vsibench_overall"]
                    )
                    out_lines.append(f"{idx}\t{prediction}\t{gt}")

            if processed_docs:
                for tag, agg_fn in AGGREGATORS:
                    out_lines.append(f"{tag}\t{agg_fn(processed_docs)}")

            with self.log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n\n")

            if was_training:
                model.train()

        self.state.wait_for_everyone()
        return control
