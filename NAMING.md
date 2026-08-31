# Experiment naming: `idea_4a_sg_*`

The `idea_3i_*` names carried no information, so every folder needed a lookup in
`notes.md` to say what it was. The `idea_4a_sg_*` scheme puts the three things that
actually vary into the name, so the note stops being necessary.

## Grammar

```
idea_4a_sg_<cond>_<corpus>_<fmt>[_<variant>]
```

| part      | meaning                                                     |
|-----------|-------------------------------------------------------------|
| `idea_4a` | the family: scene-graph generation from video                |
| `sg`      | scene graph (the task)                                       |
| `cond`    | how 3D geometry reaches the LLM                              |
| `corpus`  | which dataset                                                |
| `fmt`     | what the LLM is trained to emit                              |
| `variant` | optional, only when one experiment has more than one setup   |

`idea_4a` is a **family prefix, not a serial number** — it does not increment per
experiment. Every scene-graph-from-video run is an `idea_4a_sg_*`; the descriptive part
tells them apart. A new experiment names itself, with no letter to assign by hand.

## Vocabulary

### `cond` — geometry conditioning

| value   | what it does |
|---------|--------------|
| `perc`  | Frozen VGGTOmega → two Perceiver IO encoders → latents `masked_scatter`-ed into `<|quad_start|>` / `<|quad_end|>` placeholders. **Latent-only: the MLLM sees no pixels**, the latents are its entire visual input. |
| `vgadd` | VGGT patch features → zero-init merger → **added** onto Qwen's video-token embeddings at prefill (VG-LLM / SpatialStack "add" fusion). The MLLM still sees the native video, so geometry augments a working visual stream instead of replacing it. |
| `head`  | Small LM decoder head regressing the graph directly. Older line, not currently used. |

`latentonly` is not in the name because `perc` already implies it, and `vgadd` implies
the opposite. Only mark it with a `_video` variant if you build a `perc` run that also
feeds the video.

### `corpus`

| value  | what it is |
|--------|------------|
| `ca1m` | CA-1M, one directory per scene: `scene_graph.txt` + `frames/*.png`. |
| `scannetpp` | ScanNet++ v2. Videos from VSI-590K (`scannettppv2/<scene>.mp4`), targets rebuilt from the ScanNet++ mesh + instance GT by ScanEdit. |
| `vsi`  | VSI-590K / ScanEdit, unspecified sub-corpus. Superseded by `scannetpp`, which names the actual scenes; kept only so the reserved value is not silently reused. |

### `fmt` — the supervised target

| value    | shape | frame |
|----------|-------|-------|
| `toon9`  | `obj[N]{id,name,dx,dy,dz,cx,cy,cz,yaw}:` header, one indented row per object. Metres. | Origin = first camera, **+z up, gravity-aligned**. Only `yaw` is free; roll and pitch are identically 0. |
| `toon8`  | v3 TOON. `base` (centre-xy + min-z) instead of a true centre, no yaw. | As above. |
| `vgjson` | Fenced ` ```json ` list of `{"label": ..., "bbox_3d": [x,y,z,dx,dy,dz,yaw,roll,pitch]}`. VG-LLM's own format. | **Camera 0's own frame** (x right, y down, z forward), so all three Euler angles carry signal. |
| `json6`  | The same fenced list with the three rotation terms **dropped**: `bbox_3d` is `[x,y,z,dx,dy,dz]`. For corpora whose boxes are axis-aligned, where the angles would be three constant zeros in every row. | As `toon8` — gravity-aligned world frame. |

The format also picks the metric, which is why it belongs in the name:

- `toon8` / `toon9` → `graph_toon.compare` — id Jaccard, name / numeric match
  fractions, exact-line fraction.
- `vgjson` → `graph_vgllm.compare` — VG-LLM's detection score: greedy per-category
  matching at IoU 0.25 → precision / recall / f1.
- `json6` → `graph_json6.compare` — the same score, but axis-aligned IoU computed
  analytically instead of via scipy's halfspace polytope. Same keys as `vgjson`,
  so the two read alike; the values are not comparable (different corpus and frame).

Numbers are comparable **within** a `fmt`, never across one.

## Current experiments

| folder | cond | fmt | what it tests |
|--------|------|-----|---------------|
| `idea_4a_sg_perc_ca1m_toon9`  | `perc`  | `toon9`  | The original latent-only Perceiver baseline. |
| `idea_4a_sg_vgadd_ca1m_toon9` | `vgadd` | `toon9`  | Same target, fusion swapped — does conditioning cause the collapse? |
| `idea_4a_sg_perc_ca1m_vgjson` | `perc`  | `vgjson` | Same fusion, target swapped — does the format cause it? |
| `idea_4a_sg_perc_scannetpp_toon8` | `perc` | `toon8` | Same fusion and near-same target, **corpus** swapped to ScanNet++ — the older line, migrated in. |
| `idea_4a_sg_perc_scannetpp_json6` | `perc` | `json6` | Same 839/49 scenes and same filter policy as `..._scannetpp_toon8`, serialization swapped — does the target format cause the collapse? Built from the raw ScanEdit graphs at 2dp rather than v3's 1dp, so it carries ~1.7% more objects; 1dp flattens thin objects to zero extent, which an IoU metric cannot score. |

`ls idea_4a_sg_*_ca1m_*` is therefore a contingency table, and reading it shows the
empty cell:

```
              toon9                          vgjson
perc     idea_4a_sg_perc_ca1m_toon9     idea_4a_sg_perc_ca1m_vgjson
vgadd    idea_4a_sg_vgadd_ca1m_toon9    idea_4a_sg_vgadd_ca1m_vgjson   <- not run
```

That last cell is the one where both changes compound. Surfacing gaps like this is the
point of the scheme; the old names hid it.

## Naming a new experiment

Pick the value on each axis and concatenate. If the new run differs from an existing
one along **none** of the three axes, it is not a new experiment — it is a variant of
an existing one, so add a `_variant` suffix (`_video`, `_rmsnorm`) or, better, a run
tag under that experiment's results rather than a new folder.

## Legacy names

| old | new |
|-----|-----|
| `ca1m_metric_toon_latentonly`      | `idea_4a_sg_perc_ca1m_toon9`  |
| `idea_3i_toon_ca1m_metric_vg`      | `idea_4a_sg_vgadd_ca1m_toon9` |
| `idea_3i_vg_ca1m_metric_perceiver` | `idea_4a_sg_perc_ca1m_vgjson` |
| `idea_3i_scene_graph_v3toon_latentonly` | `idea_4a_sg_perc_scannetpp_toon8` |

The old directories still exist and each holds a `DEPRECATED.md` pointing at its
replacement. **Edit the `idea_4a_*` tree** — the old one is kept only so in-flight runs
and existing logs resolve. Delete an old folder once nothing is running from it.

Everything else (`idea_3i`, `idea_3i_scene_graph`, `idea_3i_scene_graph_v3toon_video`, and
the `idea_1*` / `idea_2*` / `idea_3a`-`3h` lines) keeps its original name and is not
covered by this scheme.

## Known wart: results directories

`OUTPUT_DIR` in `scripts/idea_4a_sg_*/` still points at the **old** result paths:

```
results/ca1m_metric_toon_latentonly
results/idea_3i_toon_ca1m_metric_vg
results/overfit/idea_3i_vg_ca1m_metric_perceiver_overfit
results/idea_3i_scene_graph_v3toon_latentonly
```

That is deliberate — renaming them would strand existing checkpoints and break
`--resume_from_checkpoint latest`. So folder names and result names disagree until
nothing depends on the old paths.

When that day comes, the better layout is one directory per experiment with run tags
inside it:

```
results/idea_4a_sg_perc_ca1m_toon9/{baseline,rmsnorm,killed_step88}/
```

rather than today's suffixed siblings (`..._killed_step88`, `..._rmsnorm_step176`),
which scatter one experiment's history alphabetically across `results/`.
