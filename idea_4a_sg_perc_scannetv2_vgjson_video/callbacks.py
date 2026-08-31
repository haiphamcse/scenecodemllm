"""In-training 3D detection metrics for the CA-1M VG-LLM-format corpus.

Each eval sample: generate the 9-DoF JSON box list from the VGGT latents (single
generation), score it against the reference with VG-LLM's own detection metric
(greedy per-category matching at IoU 0.25 -> precision / recall / f1, graph_vgllm.py),
and dump pred vs GT. No question answering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import PartialState
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoProcessor, TrainerCallback

from qwen_vl_utils import process_vision_info

from collator import (
    extract_raw_video_tensors,
    graph_only_messages,
    load_example_video,
    load_scene_graph_text,
)

# VG-LLM detection score: per-category greedy matching at IoU 0.25.
from graph_vgllm import compare

_COMPARE_KEYS = ["precision", "recall", "f1"]


def _unwrap_for_generate(model):
    """Peel DDP wrappers so `.generate()` hits the underlying module."""
    while hasattr(model, "module") and not hasattr(model, "generate"):
        model = model.module
    return model


class SceneGraphReconCallback(TrainerCallback):
    """Generate + score scene-graph reconstructions on a fixed step interval."""

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
        scene_graph_root: Path,
        image_patch_size: int = 16,
        max_graph_tokens: int = 2048,
        max_eval_samples: Optional[int] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
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
        self.scene_graph_root = Path(scene_graph_root)
        self.image_patch_size = image_patch_size
        self.max_graph_tokens = max_graph_tokens
        self.max_eval_samples = max_eval_samples
        # Default is unchanged greedy decoding; infer_sampled.py overrides it.
        self.gen_kwargs = gen_kwargs or {"do_sample": False}
        self.state = PartialState()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _eval_indices(self) -> range:
        n = len(self.eval_dataset)
        if self.max_eval_samples is not None:
            n = min(n, int(self.max_eval_samples))
        return range(n)

    def _prep_inputs(self, messages, frames, meta, device):
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Native frames for VGGT, budgeted copy for the MLLM -- the same two resizes the
        # collator builds. Eval must feed the video too, or it would score the model on a
        # prompt shape it never trained on.
        video_inputs = extract_raw_video_tensors(frames, meta, self.image_patch_size)
        if not video_inputs:
            return None
        _, llm_videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            image_patch_size=self.image_patch_size,
            return_video_metadata=True,
        )
        if not llm_videos:
            return None
        llm_tensors, llm_metadata = zip(*llm_videos)
        inputs = self.processor(
            text=prompt,
            videos=list(llm_tensors),
            video_metadata=list(llm_metadata),
            return_tensors="pt",
            padding=True,
            **video_kwargs,
        )
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        inputs["raw_videos"] = [v.to(device) for v in video_inputs]
        return inputs

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step <= 0 or step % self.eval_steps != 0:
            return control
        return self._run_eval(step, control)

    def _run_eval(self, step: int, control):
        if self.state.is_main_process:
            model = _unwrap_for_generate(self.model)
            was_training = model.training
            model.eval()
            device = next(model.parameters()).device

            out_lines: List[str] = [f"step {step}"]
            score_sums: Dict[str, float] = {k: 0.0 for k in _COMPARE_KEYS}
            n_scored = 0

            with torch.no_grad():
                for idx in tqdm(
                    list(self._eval_indices()),
                    desc=f"SG recon eval (step {step})", unit="sample",
                ):
                    example = self.eval_dataset[idx]
                    frames, meta = load_example_video(example, self.data_root, self.cache_root)
                    messages = graph_only_messages(
                        frames, meta,
                        self.frame_placeholder_text, self.camera_placeholder_text,
                    )
                    inputs = self._prep_inputs(messages, frames, meta, device)
                    if inputs is None:
                        out_lines.append(f"{idx}\tskip")
                        continue
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.gen_kwargs.get("max_new_tokens", self.max_graph_tokens),
                        **{k: v for k, v in self.gen_kwargs.items() if k != "max_new_tokens"},
                    )
                    input_len = inputs["input_ids"].shape[1]
                    pred_graph = self.processor.tokenizer.decode(
                        outputs[0, input_len:], skip_special_tokens=True
                    ).strip()

                    gt_graph = load_scene_graph_text(
                        example, self.scene_graph_root
                    ) or ""
                    metrics = compare(gt_graph, pred_graph)
                    for k in _COMPARE_KEYS:
                        score_sums[k] += float(metrics.get(k, 0.0))
                    n_scored += 1

                    score_str = " ".join(f"{k}={metrics.get(k)}" for k in _COMPARE_KEYS)
                    out_lines.append(f"{idx}\t{score_str}")
                    out_lines.append(f"--- SG {idx} pred ---\n{pred_graph}")
                    out_lines.append(f"--- SG {idx} gt ---\n{gt_graph or '<missing>'}")

            if n_scored:
                means = {k: round(score_sums[k] / n_scored, 4) for k in _COMPARE_KEYS}
                out_lines.append("MEAN\t" + " ".join(f"{k}={means[k]}" for k in _COMPARE_KEYS))

            with self.log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(out_lines) + "\n\n")

            if was_training:
                model.train()

        self.state.wait_for_everyone()
        return control
