"""Helpers for viz_det_es_trace.py, copied from viz_det_es/utils.py and trimmed to the eval (VSI-Bench) ScanNet path so this
folder is self-contained. Everything geometric is in the camera-0 frame the det rows store."""
import json, os, pickle
import cv2, numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation

W = "/scratch/ducpham/Working"
ROOT = f"{W}/dataset/vsib/vsibench/scannet"
SCANNET_POSED = f"{W}/dataset/scannet/posed_images"
SCANNET_NATIVE_HW = (968, 1296)                           # posed jpg size = what the EmbodiedScan cam2img is for
ES_PKL = W + "/dataset/embodiedscan/embodiedscan/embodiedscan_infos_{}.pkl"
MAX_PTS = 400_000
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)   # corners() spams PaintUniformColor warnings


def scene_dir(scene):
    return f"{ROOT}/{scene}/det_es"


def wxyz(R):
    q = Rotation.from_matrix(R).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


def embodiedscan_bbox_to_o3d_geo(box):
    """EmbodiedScan 9-DoF [x, y, z, dx, dy, dz, rz, rx, ry] (Euler ZXY) -> o3d OrientedBoundingBox. From VG-LLM scripts/preprocess/utils.py."""
    box = np.asarray(box)
    rot = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(box[6:].reshape(3, 1))
    return o3d.geometry.OrientedBoundingBox(box[:3].reshape(3, 1), rot, box[3:6].reshape(3, 1))


def corners(b):
    """8 corners + 12 edges of a box dict (center, R, size), o3d convention."""
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox(b["center"], b["R"], b["size"]))
    return np.asarray(ls.points), np.asarray(ls.lines)


def draw(img, boxes, T_cam0_to_i, K, color):
    """Project cam0-frame boxes into frame i (edges clipped at z=eps) and draw them in place."""
    eps = 0.05
    for b in boxes:
        P, L = corners(b); P = P @ T_cam0_to_i[:3, :3].T + T_cam0_to_i[:3, 3]
        uv = lambda p: tuple(((K @ p)[:2] / p[2]).astype(int))
        for i, k in L:
            a, c = P[i], P[k]
            if a[2] <= eps and c[2] <= eps:
                continue
            if a[2] <= eps: a = c + (a - c) * (c[2] - eps) / (c[2] - a[2])
            if c[2] <= eps: c = a + (c - a) * (a[2] - eps) / (a[2] - c[2])
            cv2.line(img, uv(a), uv(c), color, 1)
        vis = P[P[:, 2] > eps]
        if len(vis):
            p = vis[np.argmin((K @ vis.T)[1] / vis[:, 2])]                      # top-most visible corner
            cv2.putText(img, b["label"], uv(p), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def unproject(depth, K, T, step=1):
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]; z = depth[v, u] / 1000.0; m = z > 0
    pc = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
    return pc @ T[:3, :3].T + T[:3, 3]


def load_es_sample(scene):
    """EmbodiedScan ScanNet sample (train / val / test pkls): cam2img for the posed jpgs, depth_cam2img for the depth pngs."""
    for split in ("train", "val", "test"):
        for s in pickle.load(open(ES_PKL.format(split), "rb"))["data_list"]:
            if s["sample_idx"] == f"scannet/{scene}":
                return s
    raise SystemExit(f"scannet/{scene} not in the EmbodiedScan pkls")


def load_scene(d, es):
    """meta, cam0-frame box dicts, RGB frames, cam0 -> cam_i poses, per-frame K in saved-frame pixels.
    es = EmbodiedScan sample: the eval (vsib) ScanNet meta has no cam2img / native_hw, so K comes from the pkl."""
    meta, row = json.load(open(f"{d}/meta.json")), json.load(open(f"{d}/det.json"))
    fr = meta["frames"]
    frames = [cv2.cvtColor(cv2.imread(f"{d}/frames/frame{i:02d}.jpg"), cv2.COLOR_BGR2RGB) for i in range(meta["n_frames"])]
    if "cam2img" not in fr[0]:
        meta["native_hw"] = SCANNET_NATIVE_HW
        for f in fr:
            f["cam2img"] = np.array(es["cam2img"])[:3, :3].tolist()
    (H, Wd), (h, w) = meta["native_hw"], frames[0].shape[:2]
    Ks = [np.diag([w / Wd, h / H, 1]) @ np.array(f["cam2img"]) for f in fr]                 # native -> saved pixels
    cam0 = np.array(fr[0]["cam2global"]); poses = [np.linalg.inv(np.array(f["cam2global"])) @ cam0 for f in fr]  # cam0 -> cam_i
    boxes = [dict(label=b["label"], center=(g := embodiedscan_bbox_to_o3d_geo(b["bbox_3d"])).center, R=g.R, size=g.extent)
             for b in row["boxes"]]
    return meta, boxes, frames, poses, Ks


def cloud(scene, meta, frames, es):
    """(P, C) in the meta global frame from the ScanNet posed depth pngs (depth_cam2img from the pkl), coloured from the det_es
    frames on a step-2 pixel grid; None when the depth is not on disk."""
    fr, k = meta["frames"], meta.get("rot90_k") or 0
    if not os.path.isdir(f"{SCANNET_POSED}/{scene}"):
        return None
    Kd = np.array(es["depth_cam2img"])[:3, :3]
    P, C = [], []
    for f, img in zip(fr, frames):
        if not np.isfinite(f["cam2global"]).all():
            print(f"warning: frame{f['frame']:02d} ({f['posed_id']}) has a non-finite cam2global, skipped from the cloud", flush=True); continue
        dep = np.rot90(cv2.imread(f"{SCANNET_POSED}/{scene}/{f['posed_id']}.png", -1), k)
        v, u = np.mgrid[0:dep.shape[0]:2, 0:dep.shape[1]:2]; m = dep[v, u] > 0
        P.append(unproject(dep, Kd, np.array(f["cam2global"]), step=2)); C.append(cv2.resize(img, dep.shape[::-1])[v[m], u[m]])
    P, C = np.concatenate(P), np.concatenate(C)
    if len(P) > MAX_PTS:
        i = np.random.default_rng(0).choice(len(P), MAX_PTS, replace=False); P, C = P[i], C[i]
    return P, C
