# `scannet_det_*_4frames.json`

VG-LLM's ScanNet-v2 3D-detection corpus. One row = one 4-frame clip + oriented 3D boxes
of every annotated object, in the camera frame of **frame 0**. What `train.py` here trains on.

## Files

| | train | val |
|---|---|---|
| path | `vgllm_data/train/scannet_det_train_4frames.json` | `vgllm_data/evaluation/threedod/scannet/scannet_det_val_4frames.json` |
| size / rows / scenes | 635 MB / 144,164 / 958 | 20 MB / 2,424 / 243 |
| keys per row | **3**: `conversations`, `images`, `boxes` | **6**: + `axis_align_matrix`, `cam2global`, `cam2img` |
| boxes/row | 0–57, mean 11.76 | 1–53, mean 11.44 |
| label vocab | 204 | 164 (5 unseen in train) |

Camera metadata is val-only. Nothing here reads it, but future use must recover it from
`posed_images/*.txt` for train. Sibling `scannet_det_val_6frames.json` (25 MB): same
schema, 6 frames. Unused.

## Keys

### `images` — list[str], 4 paths

```json
["scannet/posed_images/scene0415_00/00810.jpg", …00840, …00870, …00900]
```

Relative to `--image_root` (default `/home/ducpham/scratch/Working/dataset`). 1296×968 jpgs.

- Frame index gaps always `(30, 30, 30)` — 4 frames at stride 30, ~1 s apart.
- **Order is load-bearing.** `images[0]` is the reference camera; all boxes are in its
  frame. Shuffling silently invalidates the target (`collator.py:16-17`).
- Scene id = `Path(images[0]).parent.name`. No separate field.
- `posed_images/<scene>/` also holds `<frame>.png` (depth), `<frame>.txt` (4×4 pose).
  Only the jpg is read.

### `conversations` — 2 turns

Turn 0 (`human`): one fixed string across all 146,588 rows, `<image>`×4 + the detect
instruction. **Unused** — `collator.py:44-53` writes its own `SCENE_GRAPH_QUESTION` adding
our two conventions (leading `{"n": N}`, largest-first).

Turn 1 (`gpt`): **the training target**, a fenced JSON list:

````
```json
[
	{"label": "bag", "bbox_3d": [0.61, 0.02, 0.91, 0.26, 0.26, 0.15, -0.68, -1.05, 2.51]},
	…
]```
````

Fence included, as VG-LLM trains it. Zero-box rows emit `` ```json\n[\n\t\n]``` `` (71 in
train, 0 in val). Values are `boxes` at `round(v, 2)`.

### `boxes` — list[dict]

| field | meaning |
|---|---|
| `bbox_3d` | 9 floats `[x, y, z, dx, dy, dz, yaw, roll, pitch]` — centre, full extents, intrinsic **ZXY** Euler. Metres/radians in frame 0's camera frame (x right, y down, z forward, OpenCV). |
| `label` | open-vocabulary category, not a fixed id set |
| `in_reference` | **unused, and absent from VG-LLM's own preprocess.** val 17,549/10,171 True/False; train 1,105,720/589,973 |

**`in_reference` is not a filter.** All 2424 val rows have `len(gpt) == len(boxes)`, never
`len(in_reference == True)`. `val[0]` has 8 boxes, 6 flagged, all 8 in the answer.

### `axis_align_matrix` / `cam2global` / `cam2img` — val only

Per-scene world→axis-aligned-world transform (pure yaw + translation); one 4×4 cam→world
pose per frame in `images` order; pinhole intrinsics padded to 4×4. Not needed to read
`bbox_3d`, which is already in camera-0 frame.

## How the coordinates were built

Source is **EmbodiedScan**, not ScanNet directly (`process_threedod.py:58-77`):

```python
extrinsic  = axis_align_matrix @ reference_frame["cam2global"]   # cam -> aligned world
global2cam = np.linalg.inv(extrinsic)
R_in_cam      = global2cam[:3,:3] @ geo.R
center_in_cam = global2cam @ [*geo.center, 1]
bbox_3d_in_cam = o3d_geo_to_9dof(OBB(center_in_cam, R_in_cam, geo.extent), "ZXY")
```

- Extent is **carried over untransformed** — `dx,dy,dz` are along the *world* box's axes.
- Rotation is re-decoded via `R.from_matrix(...).as_euler("ZXY")`, a branch choice. That
  is why angles look arbitrary rather than small; gimbal lock near X = ±90° is live.
- **roll/pitch are camera orientation, not object shape.** In world frame 95.6% of val
  instances have roll = pitch = 0 and 85.1% also have yaw at a multiple of 90°. One shared
  `global2cam` smears those few discrete yaws into arbitrary triples, so objects sharing a
  world yaw share a triple exactly — ~4.8 distinct triples per ~11.4 boxes. Verified:
  world yaw π/2 → `[-0.68 -1.05 2.51]`, the literal `val[0]` value for 6 of its 8 objects.

## Worked example — `val[0]`, scene0415_00 (bathroom)

```
images   scene0415_00 frames 00810 / 00840 / 00870 / 00900
boxes    8: bag, doorframe, door, vanity, towel, sink, mirror, stopcock (6 in_reference)
cam2img  fx=fy=1170.188  cx=647.75  cy=483.75
```

```
boxes[0]  [0.6112697681571052, 0.018131137368381567, 0.9051805802692294,
           0.26441568355512635, 0.26486724557876595, 0.14745813608169556,
           -0.6812369803373142, -1.045151011159886, 2.5103809032876874]
gpt       {"label": "bag", "bbox_3d": [0.61, 0.02, 0.91, 0.26, 0.26, 0.15, -0.68, -1.05, 2.51]}
```

A 26×26×15 cm bag, 91 cm in front of camera 0 and 61 cm right. Sanity check on y-down:
mirror sits at y=0.01, sink below it at y=0.30.

Loader output (`train.py:190-210`):

```python
{"images": ["/…/scene0415_00/00810.jpg", …4],
 "graph":  '```json\n[\n\t{"n": 8},\n\t{"label": "doorframe", …}, …]```',
 "scene":  "scene0415_00", "n_boxes": 8}
```

Target is **re-ordered**: `canonicalize()` sorts largest-volume-first, so `doorframe`
(0.41 m³) leads and `bag` falls back, and `{"n": 8}` is prepended — both to make sequence
length learnable (`graph_vgllm.py:69-81`).

## What training reads

**Two keys:**

| key | where | what happens |
|---|---|---|
| `images` | `train.py:192` | absolutised, existence-checked, loaded as PIL (`collator.py:66`) |
| `conversations[1]["value"]` | `train.py:198` | `canonicalize()` then the supervised span |

`boxes`, `axis_align_matrix`, `cam2global`, `cam2img`, `in_reference`, and the `human`
turn have **zero references** in all 10 `.py` files here — except eval, below.

### Load-time filters (`train.py:186-217`)

| filter | drops | today |
|---|---|---|
| missing frames | rows whose 4 jpgs aren't on disk | **0 rows** — download complete, 1513/1513 scene dirs. The `train.py:150` docstring's "~228 of 1513" is stale. |
| `--min_boxes` (runs use 5) | degenerate targets where one box scores non-zero | val 2424→2188, train 144,164→132,455 |
| `--max_target_tokens` | targets over the generation budget | prevents a target with no closing bracket and no EOS |

### Row to batch (`collator.py`)

```
turn 1  user       [VGGT quad placeholders] "This is the 3D scene context."
                   [video block] "These are the RGB frames…"  SCENE_GRAPH_QUESTION
turn 2  assistant  canonicalised graph text        <- the only supervised span
```

- Clip resized **twice**: native for VGGT, `qwen_vl_utils`-budgeted for the MLLM
  (`collator.py:112-115`).
- `--box_noise` re-perturbs the 9 values per `__getitem__` — fresh jitter each epoch
  (`collator.py:180-199`).
- Labels `IGNORE_INDEX` except `[graph_start+delta, graph_end+delta)`. `delta` absorbs
  placeholder/video-token expansion, correct **only because all of it sits in the user
  turn** (`collator.py:283-288`).

## What eval reads — the exception

`eval_vgllm_metric.py:206-231` re-opens the json and scores against raw `boxes`, not our
2dp target:

```python
raw = {tuple(r["images"]): r for r in json.load(open(args.ann))}
gt_boxes = raw[ann_key(row)]["boxes"]
```

`images` doubles as the join key. Reported VG-LLM f1 is against full-precision GT in
original order — the boxes VG-LLM itself scored. A second micro score against
`row["graph"]` ties back to the training curve. The two differ by design.

## Summary

- 9-DoF boxes, ZXY Euler, metres, **frame 0's camera frame**. Frame order matters.
- roll/pitch encode camera orientation, shared across objects — not per-object shape.
- Target = the `gpt` turn, re-sorted largest-first with a `{"n": N}` header.
- Training uses `images` + `conversations[1]`. Eval uses `boxes` as GT.
- `in_reference`, `cam2img`, `cam2global`, `axis_align_matrix` unused here; the last three
  are val-only.
