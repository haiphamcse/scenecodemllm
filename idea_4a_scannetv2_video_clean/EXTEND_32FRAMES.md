# Generating a 32-frame `scannet_det` corpus

How to produce the 32-frame twin of `scannet_det_train_4frames.json`. Format is
unchanged — same 6 keys, same 9-DoF ZXY boxes in frame 0's camera frame (see
[DATA_FORMAT.md](DATA_FORMAT.md)). Only `nframe` and `base_interval` move.

## What exists

`vgllm_data/train/scannet_det_train_32frames_bi1_5scenes.json` — 283 KB, 13 rows,
5 scenes, 32 images at stride 10. A pilot, not a corpus. Produced by:

```bash
cd VG-LLM/scripts/preprocess
/scratch/ducpham/conda/envs/vsibench_eval_full/bin/python make_det_subset.py \
  --nframe 32 --base_interval 1 --scenes 5 --per-scene 3
```

`vsibench_eval_full` is the env — `utils.py` needs `open3d`, `shapely`, `scipy`, and the
default `python3` has none of them.

## The two generators

| script | what it does | use for |
|---|---|---|
| `make_det_subset.py` | imports `process_data_item` from `process_threedod`, limits to `--scenes` scans and `--per-scene` windows, spread across each scan | eyeballing a config in seconds |
| `process_threedod.py` | every window of every scan | the real corpus |

Same windowing and same box transform — the subset script reimplements neither.

## Windowing

Both walk `sample["images"]`, split into runs of consecutive frames (index step 10), then
slide:

```python
sel = frames[i : i + nframe*base_interval : base_interval]   # for i in range(len(frames)+1)
```

So a window covers `(nframe-1) * base_interval * 10` frame indices, and **every start
offset produces a window** — consecutive windows overlap by all but one frame.

Enumerated over all 958 ScanNet train scans (mean 162 frames/scan, range 12–826):

| nframe | base_interval | windows | scans covered | span (frame idx) | est. json |
|---|---|---|---|---|---|
| 4 | 3 | 144,164 | 958 / 958 | 90 | 635 MB (shipped) |
| 16 | 1 | 137,140 | 956 / 958 | 150 | — |
| 16 | 3 | 105,189 | 913 / 958 | 450 | — |
| 32 | 1 | 119,496 | 941 / 958 | 310 | ~2.5 GB |
| 32 | 2 | 89,181 | 852 / 958 | 620 | ~1.9 GB |
| 32 | 3 | 64,126 | 697 / 958 | 930 | ~1.3 GB |

The `4 / 3` row reproduces the shipped file's 144,164 rows exactly, so the counts above
are the real generator's, not an approximation.

### Choosing `base_interval`

The axes trade against each other and there is no free option:

- **`bi=1`** (the pilot). Stride 10, ~10 s window. Keeps almost every scan (941/958) and
  yields the most rows. But consecutive frames are 1/3 the spacing the model trained on,
  and windows overlap by 31/32 frames — a highly redundant corpus.
- **`bi=3`**. Stride 30, identical to the 4-frame corpus, so window *length* is the only
  variable that changed. Costs 261 scans outright (697/958 covered): a 32×3 window needs
  a 94-frame consecutive run, and a third of ScanNet scans never have one.
- **`bi=2`** splits the difference — 852 scans, stride 20, matching neither.

`bi=3` is the clean experiment; `bi=1` is the one that keeps the data. Pick per what you
are testing.

### Window redundancy

At any `base_interval`, `range(len(frames)+1)` emits a window per start offset. The
4-frame corpus lives with this (144k rows over 958 scans). At 32 frames the neighbouring
windows share 31 of 32 frames, so the redundancy is far worse. Two options if it matters:
stride the start offsets in `windows()`, or subsample after generation the way the val
split already does.

## Full-scale command

```bash
cd VG-LLM/scripts/preprocess
/scratch/ducpham/conda/envs/vsibench_eval_full/bin/python process_threedod.py \
  --embodiedscan /scratch/ducpham/Working/dataset/embodiedscan/embodiedscan \
  --split train --nframe 32 --base_interval 3 \
  --include_cam_params \
  --output_dir /scratch/ducpham/Working/dataset/vgllm_data/train
```

Then the same with `--split val` against
`vgllm_data/evaluation/threedod/scannet/`.

## Gotchas

**Output filename ignores `base_interval`.** `process_threedod.py` writes
`scannet_det_{split}_{nframe}frames.json` — so a `bi=1` and a `bi=3` run at 32 frames
**overwrite each other**. Rename immediately, or patch the f-string. `make_det_subset.py`
already encodes `bi` in its default name; the real generator does not.

**Camera params are val-only by default.** `process_threedod.py:104` emits `cam2img` /
`cam2global` / `axis_align_matrix` only `if args.split == "val" or args.include_cam_params`.
The shipped 4-frame train file lacks them; the pilot has them because
`make_det_subset.py` hardcodes `include_cam_params=True`. Pass the flag if you want parity.

**Val is randomly subsampled.** For `--split val`, scans with more than 10 windows keep a
`random.sample(items, 10)` — seeded 42 at the top of `main()`, so it reproduces, but the
val corpus is not exhaustive at any `nframe`.

**No `in_reference`.** This repo's preprocess emits only `bbox_3d` and `label`
(`process_threedod.py:97-103`); `grep -rn in_reference` across VG-LLM returns nothing. The
shipped 4-frame file has the field from some other pipeline. A regenerated corpus will not
have it — harmless, since nothing reads it.

**Boxes per row more than doubles.** The box set is the union of `visible_instance_ids`
over every frame in the window, so a longer window sees more objects. Measured on the full
train export: mean 7.94 boxes/row at 4 frames, 19.42 at 32 (max 73). The pilot's 5 scenes
gave 14.0 — its `spread()` sampling under-represents dense scans, so trust the full number.
Longer targets, and the `--min_boxes` filter bites differently.

**Runtime.** `process_threedod.py` walks every window of every scan and builds an Open3D
OBB per box. Measured: the full 32-frame train export (958 scans, 119,496 windows, ~2.4x
the boxes per row) took **10m55s** single-process, plus a few minutes to load the pkl.
Cheaper than the folklore ~1-hour figure — the OBB transform is not the bottleneck.

## Verify before the full run

```bash
# a couple of scenes, seconds
python make_det_subset.py --nframe 32 --base_interval 3 --scenes 2 --per-scene 2 \
  --out /tmp/smoke32.json

# then eyeball the geometry
python ../../visualize_scannet_det_gt.py --ann /tmp/smoke32.json --sample 0 --stride 4 --check
```

The subset script prints `frames, windows, kept, boxes` per scan — enough to catch a
config that silently drops scans (`windows 0`) before committing to the full pass.
