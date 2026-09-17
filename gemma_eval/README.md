# gemma_eval

gemma-4 VSI-Bench-Debiased (ScanNet) eval reading frames from the det_es cache
(`<video_root>/<scene>/det_es/{det.json,meta.json,frames/frameNN.jpg}`, 384x256, no mp4 decode).
Prompts / extraction / scoring are imported from `../gemini_eval/det_vsibench_eval.py`; the `det_video`
prompt is identical to the gemini_eval det+video prompt.

```
# det list + frames (same prompt as gemini_eval --video)
python det_vsibench_eval.py --mode det_video --out_dir results/det_vsibench/x --subset_ids ids.json \
    --image_tokens 140 --num_frames 64 --no-thinking --max_new_tokens 4096
# frames only
python det_vsibench_eval.py --mode video --out_dir results/det_vsibench/y --subset_ids ids.json
# det list only (text path)
python det_vsibench_eval.py --mode det --out_dir results/det_vsibench/z --subset_ids ids.json
```

Resume per question (append-only `predictions_shard<i>.jsonl`); `--num_shards/--shard_index` to split,
`--merge` to score the union; `--subset_ids`/`--subset_n --seed` write to `<out_dir>/subset/`.
Runner: `sbatching/dummy/det_vsibench_gemma_cache_local.sh <out_dir> [args]` inside an srun.
