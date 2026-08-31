"""Viser view of one row of an eval_vgllm_metric.py jsonl: predicted vs GT boxes on the cloud.

The jsonl stores ``scene``, ``images``, ``pred`` and the per-sample micro scores but NOT the
target, so the GT is re-read from the annotation the run scored (``--ann``, the same
threedod_1perscene val json eval_vgllm_metric defaults to) and keyed by scene id. The frame
lists are compared before anything is drawn: without that check a mismatched annotation would
put boxes over the wrong room silently.

Self-contained by the same convention as the rest of this folder -- graph_vgllm.py,
callbacks.py and model_with_vggt.py are copies rather than imports, and the geometry helpers
here are copied from ``idea_4a_sg_perc_scannetv2_vgjson/visualize_pred.py``.

The cloud is ScanNet's own sensor depth back-projected into the camera frame of image[0] --
the frame the boxes are expressed in -- so a box that floats off the geometry is a real error,
not a viewer artefact.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_4a_sg_perc_scannetv2_vgjson_video/visualize_metric_pred.py --index 0
  python idea_4a_sg_perc_scannetv2_vgjson_video/visualize_metric_pred.py --index 0 --check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import viser
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_vgllm import canonicalize, parse  # noqa: E402

_VGLLM = Path("/scratch/ducpham/Working/spatial_reasoning/scene_graph_idea/VG-LLM")
_DEFAULT_ANN = _VGLLM / "data/evaluation/threedod_1perscene/scannet/scannet_det_val_4frames.json"
_DEFAULT_IMAGE_ROOT = Path("/home/ducpham/scratch/Working/dataset")
_DEFAULT_JSONL = Path(
    "/home/ducpham/scratch/Working/spatial_reasoning/finetuning/results/"
    "idea_4a_sg_perc_scannetv2_vgjson_video_lora/vgllm_metric_ckpt6000.jsonl"
)

PRED_COLOR = (240, 70, 70)
GT_COLOR = (70, 210, 110)


def _pose(jpg):
    """Camera-to-world for a frame. ScanNet writes -inf for frames it failed to track."""
    pose = np.loadtxt(jpg[:-4] + ".txt")
    if not np.isfinite(pose).all():
        raise ValueError(f"non-finite pose for {jpg}; frame was not tracked")
    return pose


def backproject(image_paths, i, stride):
    """Frame i's depth -> (points, rgb) in the camera frame of frame 0.

    The depth png is 640x480 and the jpg 1296x968, so colour is looked up by projecting each
    point through the colour intrinsics rather than assuming the grids line up.
    """
    jpg = image_paths[i]
    scene_dir = os.path.dirname(jpg)
    depth = np.asarray(Image.open(jpg[:-4] + ".png"), dtype=np.float32) / 1000.0  # ScanNet mm
    color = np.asarray(Image.open(jpg), dtype=np.float32) / 255.0

    kd = np.loadtxt(os.path.join(scene_dir, "depth_intrinsic.txt"))
    kc = np.loadtxt(os.path.join(scene_dir, "intrinsic.txt"))

    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    d = depth[::stride, ::stride]
    keep = d > 0
    u, v, d = u[keep], v[keep], d[keep]
    pts = np.stack([(u - kd[0, 2]) / kd[0, 0] * d, (v - kd[1, 2]) / kd[1, 1] * d, d], axis=1)

    uc = np.round(kc[0, 0] * pts[:, 0] / pts[:, 2] + kc[0, 2]).astype(int)
    vc = np.round(kc[1, 1] * pts[:, 1] / pts[:, 2] + kc[1, 2]).astype(int)
    inside = (uc >= 0) & (uc < color.shape[1]) & (vc >= 0) & (vc < color.shape[0])
    pts, rgb = pts[inside], color[vc[inside], uc[inside]]

    to_ref = np.linalg.inv(_pose(image_paths[0])) @ _pose(jpg)
    return pts @ to_ref[:3, :3].T + to_ref[:3, 3], rgb


def box_pose(bbox_3d):
    """9-DoF -> (center, wxyz quaternion, extent, rotation). Intrinsic ZXY, as exported."""
    center = np.asarray(bbox_3d[:3], dtype=float)
    extent = np.asarray(bbox_3d[3:6], dtype=float)
    rot = Rotation.from_euler("ZXY", np.asarray(bbox_3d[6:9], dtype=float))
    x, y, z, w = rot.as_quat()
    return center, np.array([w, x, y, z]), extent, rot


def occupancy(points, boxes):
    """Points falling inside each box. A frame or Euler-convention slip drives these to zero."""
    counts = []
    for box in boxes:
        center, _, extent, rot = box_pose(box["bbox_3d"])
        local = (points - center) @ rot.as_matrix()  # world->box is R^T, i.e. right-multiply by R
        counts.append(int(np.all(np.abs(local) <= extent / 2, axis=1).sum()))
    return counts


def add_boxes(server, prefix, boxes, color):
    """One wireframe box + label per object; returns the handles so the GUI can toggle them."""
    handles, labels = [], []
    for i, box in enumerate(boxes):
        center, wxyz, extent, _ = box_pose(box["bbox_3d"])
        handles.append(server.scene.add_box(
            f"/{prefix}_{i}", color=color, dimensions=tuple(extent),
            wxyz=wxyz, position=center, wireframe=True))
        labels.append(server.scene.add_label(
            f"/{prefix}_label_{i}", box["label"], position=center))
    return handles, labels


def load_gt(ann_path, image_root):
    """{scene: (absolute frame paths, canonicalised target)} for the scored annotation.

    eval_vgllm_metric ran with min_boxes=1 and no token cap, so every annotation row is a
    scored row -- reading the json directly gives the same set without importing train.py
    (and torch/trl/VGGT with it).
    """
    out = {}
    for row in json.loads(Path(ann_path).read_text()):
        rels = row["images"]
        scene = Path(rels[0]).parent.name
        out.setdefault(scene, ([str(Path(image_root) / r) for r in rels],
                               canonicalize(row["conversations"][1]["value"])))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default=str(_DEFAULT_JSONL), help="eval_vgllm_metric per-sample jsonl")
    p.add_argument("--index", type=int, default=0, help="line of the jsonl")
    p.add_argument("--ann", default=str(_DEFAULT_ANN), help="annotation the run scored")
    p.add_argument("--image-root", default=str(_DEFAULT_IMAGE_ROOT))
    p.add_argument("--port", type=int, default=8082)  # 8080/8081 are the other two viewers
    p.add_argument("--stride", type=int, default=2, help="depth pixel stride; 1 is 307k points/frame")
    p.add_argument("--check", action="store_true", help="headless: assert the cloud lands in the GT")
    args = p.parse_args()

    records = [json.loads(ln) for ln in Path(args.jsonl).read_text().splitlines() if ln.strip()]
    if not 0 <= args.index < len(records):
        raise SystemExit(f"--index {args.index} out of range ({len(records)} rows in {args.jsonl})")
    rec = records[args.index]

    gt_rows = load_gt(args.ann, args.image_root)
    if rec["scene"] not in gt_rows:
        raise SystemExit(f"{rec['scene']} is not in {args.ann}; wrong annotation for this jsonl")
    images, gt_text = gt_rows[rec["scene"]]
    if [os.path.realpath(p) for p in images] != [os.path.realpath(p) for p in rec["images"]]:
        raise SystemExit(
            f"{rec['scene']}: the annotation's frames differ from the ones scored -- the jsonl "
            f"was produced against a different --ann, and the boxes would be drawn on the "
            f"wrong clip.")

    pred_boxes, gt_boxes = parse(rec["pred"]), parse(gt_text)
    micro = rec.get("micro", {})
    print(f"row {args.index} = {rec['scene']}: {len(pred_boxes)} predicted vs {len(gt_boxes)} GT "
          f"boxes (jsonl n_pred={rec.get('n_pred')}, n_gt={rec.get('n_gt')}, "
          f"f1={micro.get('f1')} P={micro.get('precision')} R={micro.get('recall')})")
    print("prediction:")
    print(rec["pred"])
    print("ground truth (canonicalized: count first, then volume descending):")
    print(gt_text)

    clouds = []
    for i, path in enumerate(images):
        clouds.append(backproject(images, i, args.stride))
        print(f"  frame {i}: {path}  ({len(clouds[-1][0])} points)")
    points = np.concatenate([c[0] for c in clouds])
    rgb = np.concatenate([c[1] for c in clouds])
    print(f"cloud: {len(points)} points from {len(clouds)} frames, in frame 0's camera frame")

    if args.check:
        counts = occupancy(points, gt_boxes)
        for box, n in zip(gt_boxes, counts):
            print(f"  {box['label']:<20} {n:>7} points inside")
        hit = sum(n > 10 for n in counts)
        assert hit >= max(1, len(gt_boxes) // 2), f"only {hit}/{len(gt_boxes)} GT boxes contain points"
        print(f"OK: {hit}/{len(gt_boxes)} GT boxes contain points")
        return

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")  # camera frame has +y down; navigation only
    cloud = server.scene.add_point_cloud("/cloud", points=points, colors=rgb, point_size=0.01)
    server.scene.add_frame("/camera0", axes_length=0.4, axes_radius=0.01)

    @server.on_client_connect
    def _(client):
        client.camera.position = np.array([0.0, -1.5, -4.0])
        client.camera.look_at = points.mean(axis=0)

    pred_handles, pred_labels = add_boxes(server, "pred", pred_boxes, PRED_COLOR)
    gt_handles, gt_labels = add_boxes(server, "gt", gt_boxes, GT_COLOR)

    with server.gui.add_folder("Cloud"):
        show_cloud = server.gui.add_checkbox("Show cloud", True)
        point_size = server.gui.add_slider("Point size", 0.001, 0.05, 0.001, 0.01)
    with server.gui.add_folder("Boxes"):
        show_pred = server.gui.add_checkbox(f"Predicted ({len(pred_boxes)})", True)
        show_gt = server.gui.add_checkbox(f"GT ({len(gt_boxes)})", True)
        show_labels = server.gui.add_checkbox("Show labels", False)

    def redraw_cloud(_=None):
        # viser 1.0 has no live point_size, so resizing means re-adding under the same name.
        nonlocal cloud
        cloud = server.scene.add_point_cloud(
            "/cloud", points=points, colors=rgb, point_size=point_size.value)
        cloud.visible = show_cloud.value

    def restyle(_=None):
        for handles, labels, shown in ((pred_handles, pred_labels, show_pred.value),
                                       (gt_handles, gt_labels, show_gt.value)):
            for handle, label in zip(handles, labels):
                handle.visible = shown
                label.visible = shown and show_labels.value

    show_cloud.on_update(redraw_cloud)
    point_size.on_update(redraw_cloud)
    for control in (show_pred, show_gt, show_labels):
        control.on_update(restyle)
    restyle()

    print(f"serving on http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
