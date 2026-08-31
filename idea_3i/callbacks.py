"""In-training VSI-Bench generation metrics (idea_3i: cache + Perceiver scatter).

Ports idea_3d's callback to the idea_3i collator: video frames come from the
pre-extracted cache, the user turn carries the VGGT placeholder blocks, and
``raw_videos`` is passed at generation time so the VGGT-Perceiver scatter is
active at eval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import PartialState
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoProcessor, TrainerCallback

from collator import load_example_video, real_video_metadata, user_only_messages
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
        cache_root: Path,
        frame_placeholder_text: str,
        camera_placeholder_text: str,
        log_path: Path,
        eval_steps: int,
        image_patch_size: int = 16,
        max_new_tokens: int = 16,
        max_eval_samples: Optional[int] = None,
    ):
        self.model = model
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.data_root = data_root
        self.cache_root = cache_root
        self.frame_placeholder_text = frame_placeholder_text
        self.camera_placeholder_text = camera_placeholder_text
        self.log_path = Path(log_path)
        self.eval_steps = max(1, int(eval_steps))
        self.image_patch_size = image_patch_size
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
                    example = self.eval_dataset[idx]
                    frames, meta = load_example_video(example, self.data_root, self.cache_root)
                    messages = user_only_messages(
                        frames, meta, example,
                        self.frame_placeholder_text, self.camera_placeholder_text,
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

                    video_inputs, _fabricated_metadata = map(list, zip(*video_inputs))
                    video_metadata_list = [real_video_metadata(meta)] * len(video_inputs)
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
                    inputs["raw_videos"] = [v.to(device) for v in video_inputs]
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
