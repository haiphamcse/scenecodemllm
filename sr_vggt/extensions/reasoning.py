"""Reasoning extension (16k): continue the main run on SimpleSR reasoning traces.

Training rows have an assistant turn ``opener\\n<scene>[boxes]</scene>\\n<think>...</think>\\nAnswer: X``
(<think>/</think> are Qwen's own special tokens; <scene> is plain text). What differs from main:
- the system prompt (SYSTEM_PROMPT; --prompt reason),
- rows longer than --max_seq_len tokens are dropped before training (16384 keeps every row),
- the first window starts from the main run's adapter (--init_from_checkpoint, LoRA +
  vggt_projector loaded trainable, fresh optimizer); later windows resume their own checkpoints,
- eval generates long (--max_new_tokens 8192) and scores the text after the last "Answer:" line.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = ("You are a spatial reasoning assistant. You are given three inputs about one indoor "
                 "scene: (1) per-frame geometry features, compressed from a 3D reconstruction model's "
                 "patch tokens of every video frame; (2) camera and scene features, compressed from the "
                 "same model's per-frame camera tokens, which encode where each frame was taken; (3) the "
                 "RGB video frames. Answer the question about the scene. First list the objects the "
                 "question needs inside <scene>...</scene> as a JSON list of {\"label\", \"bbox_3d\": "
                 "[x, y, z, w, d, h]} in metres, origin at the first frame's camera, z up. Then reason "
                 "step by step inside <think>...</think>. Finish with 'Answer: <answer>'.")


def parse_answer(text: str) -> str:
    """Text after the last line starting with 'Answer:'; the whole text if there is none."""
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("Answer:")]
    return lines[-1][len("Answer:"):].strip() if lines else text


def seq_lens(rows, processor, data_root, va) -> List[int]:
    """Full tokenised length of each row (video expansion included), without decoding videos:
    plain-tokenizer length of the chat text + the processor's video expansion, measured once per
    distinct (frame count, frame size) on black frames -- the collator's own label-span arithmetic."""
    from ..data import task_messages, video_path
    from ..encoders import video_meta

    prompt, image_patch_size = va.prompt, va.image_patch_size
    frame_ph, camera_ph = va.placeholder_texts()

    tokenizer = processor.tokenizer
    deltas: Dict[tuple, int] = {}
    metas: Dict[str, tuple] = {}

    def delta_for(n_frames, size):
        if (n_frames, size) not in deltas:
            frames = [Image.new("RGB", size)] * n_frames
            msgs = task_messages({"conversations": [{"value": "q"}, {"value": "a"}]}, frames,
                                 prompt, frame_ph, camera_ph, "a")
            text = processor.apply_chat_template(msgs, tokenize=False)
            _, vids, kw = process_vision_info(msgs, return_video_kwargs=True,
                                              image_patch_size=image_patch_size, return_video_metadata=True)
            t, m = zip(*vids)
            n = processor(text=[text], videos=list(t), video_metadata=list(m), **kw)["input_ids"]
            deltas[(n_frames, size)] = len(n[0]) - len(tokenizer(text)["input_ids"])
        return deltas[(n_frames, size)]

    out = []
    for row in rows:
        path = video_path(row, data_root)
        if path not in metas:
            metas[path] = video_meta(path, va.video_fps, va.video_max_frames)
        msgs = task_messages(row, [], prompt, frame_ph, camera_ph, row["conversations"][1]["value"])
        text = processor.apply_chat_template(msgs, tokenize=False)
        out.append(len(tokenizer(text)["input_ids"]) + delta_for(*metas[path]))
    logging.info("seq_lens: %d rows, video deltas %s", len(rows), deltas)
    return out


def kept_rows(rows, max_seq_len: int, output_dir: Path, *len_args) -> List[int]:
    """Indices of rows that fit in max_seq_len. The ~11 min scan is cached in the output dir
    for the chained windows (same jsonl, same max_seq_len)."""
    import json

    kept_file = output_dir / f"kept_rows_le{max_seq_len}.json"
    if kept_file.exists():
        kept = json.load(open(kept_file))
        logging.info("kept rows from %s: %d/%d", kept_file, len(kept), len(rows))
        return kept
    lens = seq_lens(rows, *len_args)
    kept = [i for i, n in enumerate(lens) if n <= max_seq_len]
    logging.info("dropped %d/%d rows over %d tokens (max seen %d)",
                 len(rows) - len(kept), len(rows), max_seq_len, max(lens))
    tmp = kept_file.with_suffix(f".{os.getpid()}.tmp")  # every rank computes the same list
    json.dump(kept, open(tmp, "w"))
    os.replace(tmp, kept_file)
    return kept


def load_init_adapter(model, ckpt_dir: str):
    """eval.py's load path, trainable: the saved LoRA adapter (+ vggt_projector via
    modules_to_save) into its own LoRA config, no merge. Every saved tensor is checked."""
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    peft_model = PeftModel.from_pretrained(model, ckpt_dir, is_trainable=True)
    saved = load_file(str(Path(ckpt_dir) / "adapter_model.safetensors"))
    live = get_peft_model_state_dict(peft_model)
    if saved.keys() != live.keys():
        raise RuntimeError(f"init_from_checkpoint {ckpt_dir}: adapter keys differ "
                           f"(missing {len(saved.keys() - live.keys())}, extra {len(live.keys() - saved.keys())})")
    bad = [k for k in saved if not torch.equal(live[k].detach().cpu().to(saved[k].dtype), saved[k])]
    if bad:
        raise RuntimeError(f"init_from_checkpoint {ckpt_dir}: {len(bad)} tensors not loaded, e.g. {bad[:3]}")
    proj = [k for k in saved if "vggt_projector" in k]
    proj_abs = torch.stack([saved[k].float().abs().mean() for k in proj]).mean().item() if proj else 0.0
    if proj_abs == 0:
        raise RuntimeError(f"init_from_checkpoint {ckpt_dir}: vggt_projector weights missing or all zero")
    logging.info("init_from_checkpoint %s: %d LoRA + %d vggt_projector tensors loaded (projector "
                 "mean|w| %.4g); fresh optimizer/schedule.", ckpt_dir, len(saved) - len(proj), len(proj), proj_abs)
    return peft_model


if __name__ == "__main__":
    assert parse_answer("<think>x</think>\nAnswer: B") == "B"
    assert parse_answer("Answer: 1\nmore\nAnswer: 2.5 ") == "2.5"
    assert parse_answer("no answer line") == "no answer line"
    print("reasoning selftest ok")
