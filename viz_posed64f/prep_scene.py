"""ScanNet++ / ScanNet / ARKitScenes scene -> Boxer ScanNetLoader layout + GT json.

  python prep_scene.py --scene 39f36da05b                      # ScanNet++ (default --dataset scannetpp)
  python prep_scene.py --dataset scannet --scene scene0347_02  # EmbodiedScan v1 poses/GT, local posed_images
  python prep_scene.py --dataset arkitscenes --scene 42897815  # EmbodiedScan v2 poses/GT, Apple 3DOD lowres frames

EmbodiedScan datasets (see main_embodiedscan): frames = the 64-uniform training rule over the VSI-590K mp4, each
snapped by TIME to the nearest EmbodiedScan posed frame. ScanNet: mp4 frame i == raw 30 fps frame i (verified by
pixel match; the mp4 header says 24 fps, which only changes how many frames the rule picks). ARKit: mp4 = 30 fps
wide stream that starts a bit before the first lowres_wide png; the offset (and the 90/180 deg rotation VSI-590K
applied) is found per scene by pixel matching. World = axis_align_matrix @ cam2global (z-up), GT bbox_3d as-is.
ARKit lowres_wide frames are stored sideways/upside-down; image+depth+K+pose are rotated about the camera z axis
to upright so the 2D detector sees what the VLM sees. Depth = the real depth pngs (mm).

Frames: training rule (collator.decode_video_frames: n=round(len/fps*1.0) capped 64, linspace over rgb.mkv),
each sampled index snapped to the nearest COLMAP-registered frame (stride 10 -> <=5 frames / 0.08 s off).
Poses: iphone/colmap cam2world -- the only iPhone poses in the mesh/GT frame (ARKit aligned_pose is not);
the mesh frame is metric and z-up, which is what Boxer's gravity assumption needs. Colour undistorted to
pinhole (COLMAP OPENCV k1,k2,p1,p2). Depth: iphone/depth.bin (ARKit 256x192, frame i == video frame i)
if present, else rendered from scans/mesh_aligned_0.05.ply (320x240) -- mesh depth is cleaner than ARKit.
Output: data/scannet/<scene>/frames/{color,pose,intrinsic,depth} (the '/scannet/' in the path is what
run_boxer.py keys on) + gt.json + meta.json."""
import argparse, json, os, zlib
import cv2, decord, numpy as np
from scipy.spatial.transform import Rotation
from labelmap import to_vocab

DATA = "/scratch/ducpham/Working/dataset/scannetpp/data"
HERE = "/scratch/ducpham/Working/spatial_reasoning/boxer/scannetpp_probe"                     # probe outputs (data/) stay in the probe folder
DEPTH_HW = (192, 256)


def sample_indices(n_total, fps, video_fps=1.0, max_frames=64):
    n = int(round(n_total / fps * video_fps))
    n = max(1, min(n, max_frames, n_total))
    return np.linspace(0, n_total - 1, n).round().astype(int)


def read_colmap(d):
    line = next(l for l in open(f"{d}/cameras.txt") if not l.startswith("#"))
    _, model, w, h, fx, fy, cx, cy, *dist = line.split()
    assert model == "OPENCV", model
    K = np.array([[float(fx), 0, float(cx)], [0, float(fy), float(cy)], [0, 0, 1]])
    poses = {}
    for l in open(f"{d}/images.txt"):
        p = l.split()
        if l.startswith("#") or len(p) != 10:                       # pose lines only (points2D lines can be long)
            continue
        qw, qx, qy, qz = map(float, p[1:5]); t = np.array(p[5:8], float)
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()          # world->cam
        T = np.eye(4); T[:3, :3], T[:3, 3] = R.T, -R.T @ t              # cam->world
        poses[int(p[9].split("_")[1].split(".")[0])] = T
    return K, np.array(dist[:4], float), poses, int(w), int(h)


def read_depth_bin(path, wanted):
    """ARKit depth.bin -> {frame_idx: (192,256) uint16 mm}. Mirrors scannetpp/iphone/prepare_iphone_data.py."""
    H, W = DEPTH_HW
    raw = open(path, "rb").read()
    try:                                                   # whole-file zlib float32 metres
        arr = np.frombuffer(zlib.decompress(raw, wbits=-zlib.MAX_WBITS), np.float32).reshape(-1, H, W)
        return {i: (arr[i] * 1000).astype(np.uint16) for i in wanted if i < len(arr)}
    except zlib.error:
        pass
    import lz4.block                                       # per-frame lz4 uint16 mm
    out, pos, i = {}, 0, 0
    while pos + 4 <= len(raw):
        n = int.from_bytes(raw[pos:pos + 4], "little"); pos += 4
        if i in wanted:
            try:
                out[i] = np.frombuffer(lz4.block.decompress(raw[pos:pos + n], uncompressed_size=H * W * 2), np.uint16).reshape(H, W)
            except Exception:
                out[i] = (np.frombuffer(zlib.decompress(raw[pos:pos + n], wbits=-zlib.MAX_WBITS), np.float32).reshape(H, W) * 1000).astype(np.uint16)
        pos += n; i += 1
    return out


class MeshDepth:
    def __init__(self, ply, K, W, H, dw=320, dh=240):
        import open3d as o3d
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(ply)))
        self.K = K * np.array([[dw / W], [dh / H], [1]]); self.dw, self.dh = dw, dh

    def __call__(self, T_wc):
        o3d = self.o3d
        rays = self.scene.create_rays_pinhole(o3d.core.Tensor(self.K), o3d.core.Tensor(np.linalg.inv(T_wc)), self.dw, self.dh)
        t = self.scene.cast_rays(rays)["t_hit"].numpy()
        r = rays.numpy(); hit = r[..., :3] + t[..., None] * r[..., 3:]        # world hit points
        z = (hit - T_wc[:3, 3]) @ T_wc[:3, 2]                                  # z-depth in camera
        z[~np.isfinite(t)] = 0
        return np.clip(z * 1000, 0, 65535).astype(np.uint16)


def load_gt(scene_dir):
    groups = json.load(open(f"{scene_dir}/scans/segments_anno.json"))["segGroups"]
    gt = []
    for g in groups:                                    # OBB parse as export_scannetpp_3dod.load_boxes
        obb = g["obb"]
        R = np.array(obb["normalizedAxes"], float).reshape(3, 3).T
        gt.append({"id": g["id"], "label_raw": g["label"], "label": to_vocab(g["label"]),
                   "center": obb["centroid"], "size": obb["axesLengths"], "rot": R.tolist(),
                   "yaw": float(np.arctan2(R[1, 0], R[0, 0]))})
    return gt


def main_scannetpp(a):
    src = f"{DATA}/{a.scene}"; out = f"{HERE}/data/scannet/{a.scene}"; fr = f"{out}/frames"
    for d in ("color", "pose", "intrinsic", "depth"):
        os.makedirs(f"{fr}/{d}", exist_ok=True)

    vr = decord.VideoReader(f"{src}/iphone/rgb.mkv", num_threads=4)
    idx = sample_indices(len(vr), vr.get_avg_fps() or 30.0)
    K, dist, poses, W, H = read_colmap(f"{src}/iphone/colmap")
    reg = np.array(sorted(poses))
    snapped = reg[np.abs(reg[None, :] - idx[:, None]).argmin(1)]
    frames = sorted(set(snapped.tolist()))
    print(f"{a.scene}: video {len(vr)} frames @{vr.get_avg_fps():.0f}fps -> {len(idx)} sampled, "
          f"{len(frames)} unique after COLMAP snap (max snap |d|={np.abs(snapped - idx).max()} frames)")

    depth_bin = f"{src}/iphone/depth.bin"
    if os.path.exists(depth_bin) and not a.mesh_depth:
        depth_src = "arkit_depth.bin"; depths = read_depth_bin(depth_bin, set(frames))
    else:
        depth_src = "mesh_raycast"; render = MeshDepth(f"{src}/scans/mesh_aligned_0.05.ply", K, W, H)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0)
    imgs = vr.get_batch(frames).asnumpy()
    for f, im in zip(frames, imgs):
        im = cv2.undistort(im, K, dist, None, newK)
        cv2.imwrite(f"{fr}/color/{f}.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savetxt(f"{fr}/pose/{f}.txt", poses[f])
        d = depths.get(f) if depth_src == "arkit_depth.bin" else render(poses[f])
        if d is not None:
            cv2.imwrite(f"{fr}/depth/{f}.png", d)
    K4 = np.eye(4); K4[:3, :3] = newK
    np.savetxt(f"{fr}/intrinsic/intrinsic_color.txt", K4)

    gt = load_gt(src)
    json.dump(gt, open(f"{out}/gt.json", "w"))
    meta = {"scene": a.scene, "video_frames": len(vr), "fps": vr.get_avg_fps(), "sampled_idx": idx.tolist(),
            "frames": frames, "world_offset": poses[frames[0]][:3, 3].tolist(), "depth_src": depth_src,
            "K": newK.tolist(), "W": W, "H": H, "n_gt": len(gt), "n_gt_mapped": sum(g["label"] is not None for g in gt)}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"depth={depth_src}; GT {meta['n_gt']} boxes, {meta['n_gt_mapped']} mapped to 288-vocab; "
          f"unmapped: {sorted(set(g['label_raw'] for g in gt if g['label'] is None))}")

    if depth_src == "mesh_raycast":                    # self-check: rendered depth vs projected mesh vertices
        import open3d as o3d
        V = np.asarray(o3d.io.read_triangle_mesh(f"{src}/scans/mesh_aligned_0.05.ply").vertices)
        f = frames[len(frames) // 2]; T = poses[f]; d = cv2.imread(f"{fr}/depth/{f}.png", -1) / 1000.0
        c = (V - T[:3, 3]) @ T[:3, :3]; m = c[:, 2] > 0.1
        uv = (c[m] @ render.K.T); uv = uv[:, :2] / uv[:, 2:]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < render.dw - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < render.dh - 1)
        dr = d[uv[ok, 1].astype(int), uv[ok, 0].astype(int)]
        err = dr - c[m][ok, 2]; vis = np.abs(err) < 0.05
        print(f"depth self-check frame {f}: {vis.mean()*100:.0f}% of projected verts within 5cm of render "
              f"(rest = occluded, err<0 expected: median err {np.median(err[~vis]):+.2f} m)")
        assert vis.mean() > 0.4



# ---------------------------------------------------------------- EmbodiedScan-annotated datasets (ScanNet, ARKitScenes)
VSI = "/scratch/ducpham/Working/dataset/vsi_590k/VSI-590K"
ES_PKL = {"scannet": "/scratch/ducpham/Working/dataset/embodiedscan/embodiedscan/embodiedscan_infos_{}.pkl",
          "arkitscenes": "/scratch/ducpham/Working/dataset/embodiedscan_arkitscenes/embodiedscan-v2/embodiedscan_infos_{}.pkl"}
SCANNET_POSED = "/scratch/ducpham/Working/dataset/scannet/posed_images"
ARKIT = "/scratch/ducpham/Working/dataset/embodiedscan_arkitscenes/arkitscenes/Training"
ES_EXCLUDE = {"wall", "ceiling", "floor", "object"}          # process_threedod.py's drop set
SCANNET_FPS = 30.0                                            # true colour-stream rate; mp4 header says 24


def load_es_sample(dataset, scene):
    import pickle
    key = {"scannet": f"scannet/{scene}", "arkitscenes": f"arkitscenes/Training/{scene}"}[dataset]
    for split in ("train", "val"):
        d = pickle.load(open(ES_PKL[dataset].format(split), "rb"))
        for s in d["data_list"]:
            if s["sample_idx"] == key:
                return s, {v: k for k, v in d["metainfo"]["categories"].items()}, split
    raise SystemExit(f"{key} not in EmbodiedScan {dataset} pkls")


def rot90_cam(img, depth, K, T, k):
    """Rotate an image (np.rot90, k times CCW) and keep K / cam2world consistent: a rot90 is a rotation about the
    camera z axis (new x = old y, new y = -old x), so pose gets R_old_new on the right and K swaps axes."""
    for _ in range(k % 4):
        W = img.shape[1]
        img, depth = np.rot90(img), np.rot90(depth)
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], W - 1 - K[0, 2]], [0, 0, 1]])
        R = np.eye(4); R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        T = T @ R
    return np.ascontiguousarray(img), np.ascontiguousarray(depth), K, T


def unproject(depth, K, T, step=1):
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]; z = depth[v, u] / 1000.0; m = z > 0
    pc = np.stack([(u[m] - K[0, 2]) / K[0, 0] * z[m], (v[m] - K[1, 2]) / K[1, 1] * z[m], z[m]], 1)
    return pc @ T[:3, :3].T + T[:3, 3]


def corr(a, b):
    a = cv2.resize(a, (64, 48)).astype(float).ravel(); b = cv2.resize(b, (64, 48)).astype(float).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def arkit_mp4_offset(vr, pngs, ts):
    """VSI-590K mp4 frame i shows lowres_wide png at ts[0] + i/fps + off, rotated by k (np.rot90). Grid-search
    off in [-4, 4] s and k on 4 probe frames; returns (off, k, mean corr, per-probe corr with 0.1 s neighbours)."""
    fps = vr.get_avg_fps(); probe = [int(len(vr) * q) for q in (0.2, 0.4, 0.6, 0.8)]
    frs = {i: vr[i].asnumpy() for i in probe}; png = lambda j: cv2.cvtColor(cv2.imread(pngs[j]), cv2.COLOR_BGR2RGB)
    best = None
    for k in range(4):
        if np.rot90(frs[probe[0]], k).shape[0] > np.rot90(frs[probe[0]], k).shape[1]:
            continue                                                            # pngs are landscape
        for off in np.arange(-4, 4.01, 0.1):
            c = np.mean([corr(np.rot90(frs[i], k), png(int(np.abs(ts - (ts[0] + i / fps + off)).argmin()))) for i in probe])
            if best is None or c > best[0]:
                best = (c, k, off)
    c, k, off = best
    nb = []
    for i in probe:
        j = int(np.abs(ts - (ts[0] + i / fps + off)).argmin())
        nb.append([round(corr(np.rot90(frs[i], k), png(jj)), 3) for jj in range(max(0, j - 2), min(len(pngs), j + 3))])
    return float(off), int(k), float(c), nb


def main_embodiedscan(a):
    s, cats, split = load_es_sample(a.dataset, a.scene)
    A = np.array(s["axis_align_matrix"]); ims = s["images"]
    out = f"{HERE}/data/scannet/{a.scene}"; fr = f"{out}/frames"
    for d in ("color", "pose", "intrinsic", "depth"):
        os.makedirs(f"{fr}/{d}", exist_ok=True)
    vr = decord.VideoReader(f"{VSI}/{a.dataset}/{a.scene}.mp4", num_threads=4)
    idx = sample_indices(len(vr), vr.get_avg_fps())
    mapping = {"mp4_frames": len(vr), "mp4_header_fps": float(vr.get_avg_fps()), "mp4_hw": list(vr[0].shape[:2])}

    if a.dataset == "scannet":                       # mp4 frame i == raw frame i; posed ids are raw frame ids
        posed = np.array([int(im["img_path"].split("/")[-1][:-4]) for im in ims])
        t_mp4 = idx / SCANNET_FPS; t_posed = posed / SCANNET_FPS
        mapping.update(raw_fps=SCANNET_FPS, raw_duration_s=float(posed[-1] / SCANNET_FPS),
                       mp4_duration_s_at_raw_fps=len(vr) / SCANNET_FPS)
        j = min(len(ims) // 2, len(ims) - 1)         # alignment evidence: mp4[posed j] vs posed jpg j
        jpg = cv2.cvtColor(cv2.imread(f"{SCANNET_POSED}/{a.scene}/{posed[j]:05d}.jpg"), cv2.COLOR_BGR2RGB)
        mapping["check_corr_mp4_i_vs_posed_i"] = corr(vr[int(posed[j])].asnumpy(), jpg)
        mapping["check_corr_mp4_i24over30_vs_posed_i"] = corr(vr[int(round(posed[j] * 24 / 30))].asnumpy(), jpg)
        K = np.array(s["cam2img"])[:3, :3]; rot_k = 0
    else:                                             # ARKit: mp4 = 30 fps wide stream, starts before 1st lowres png
        fdir = f"{ARKIT}/{a.scene}/{a.scene}_frames"
        pngs = sorted(os.listdir(f"{fdir}/lowres_wide")); ts_all = np.array([float(p[len(a.scene) + 1:-4]) for p in pngs])
        off, k_mp4, c, nb = arkit_mp4_offset(vr, [f"{fdir}/lowres_wide/{p}" for p in pngs], ts_all)
        t_mp4 = ts_all[0] + idx / vr.get_avg_fps() + off
        t_posed = np.array([float(im["img_path"].split("_")[-1][:-4]) for im in ims])
        mapping.update(lowres_pngs=len(pngs), lowres_fps=float(1 / np.median(np.diff(ts_all))),
                       lowres_span_s=float(ts_all[-1] - ts_all[0]), mp4_duration_s=len(vr) / vr.get_avg_fps(),
                       posed_stride_s=float(np.median(np.diff(t_posed))), mp4_offset_s=off, mp4_rot90_k=k_mp4,
                       match_corr=c, match_corr_neighbours=nb)
        Ks = np.array([im["cam2img"][:3, :3] for im in ims]); K = np.median(Ks, 0)
        mapping["K_max_dev_px"] = float(np.abs(Ks - K).max())
        Rw = np.array([(A @ im["cam2global"])[:3, :3] for im in ims]); R1 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        # after rot90^k the new image-up (-y') is -R1^k[:,1] in old cam coords; pick k whose world-z is largest
        upz = [float((Rw @ -np.linalg.matrix_power(R1, k)[:, 1])[:, 2].mean()) for k in range(4)]
        rot_k = int(np.argmax(upz)); mapping.update(pose_rot90_k=rot_k, image_up_z_before=float(upz[0]), image_up_z_after=float(upz[rot_k]))
        if (4 - k_mp4) % 4 != rot_k:
            print(f"WARNING: pixel-derived rotation {(4 - k_mp4) % 4} != pose-derived {rot_k}")

    snap = np.abs(t_posed[None, :] - t_mp4[:, None]).argmin(1); gap = np.abs(t_posed[snap] - t_mp4)
    frames = sorted(set(snap.tolist()))
    print(f"{a.scene} [{a.dataset} {split}]: mp4 {len(vr)} frames @{vr.get_avg_fps():.0f}fps(header) -> {len(idx)} sampled, "
          f"{len(frames)} unique after time snap to {len(ims)} posed frames; gap median {np.median(gap):.3f}s max {gap.max():.3f}s")

    names, vis = [], set()
    for j in frames:
        im = ims[j]; T = A @ np.array(im["cam2global"]); vis |= set(im["visible_instance_ids"])
        if a.dataset == "scannet":
            name = int(im["img_path"].split("/")[-1][:-4])
            rgb = cv2.imread(f"{SCANNET_POSED}/{a.scene}/{name:05d}.jpg"); dep = cv2.imread(f"{SCANNET_POSED}/{a.scene}/{name:05d}.png", -1)
            Kf = K
        else:
            name = int(round(float(im["img_path"].split("_")[-1][:-4]) * 1000))
            rgb = cv2.imread(f"{ARKIT}/{im['img_path'].split('Training/')[1]}"); dep = cv2.imread(f"{ARKIT}/{im['depth_path'].split('Training/')[1]}", -1)
            if j == frames[0]:
                p0 = unproject(dep, K, T)
            rgb, dep, Kf, T = rot90_cam(rgb, dep, K, T, rot_k)
            if j == frames[0]:                        # rotation self-check: world cloud must be unchanged
                p1 = unproject(dep, Kf, T); dm = np.abs(p0.mean(0) - p1.mean(0)).max(); dc = np.abs(np.cov(p0.T) - np.cov(p1.T)).max()
                assert dm < 1e-4 and dc < 1e-4, (dm, dc)
        cv2.imwrite(f"{fr}/color/{name}.jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(f"{fr}/depth/{name}.png", dep)
        np.savetxt(f"{fr}/pose/{name}.txt", T); names.append(name)
    K4 = np.eye(4); K4[:3, :3] = Kf; np.savetxt(f"{fr}/intrinsic/intrinsic_color.txt", K4)
    H, W = rgb.shape[:2]

    gt = []
    for i, ins in enumerate(s["instances"]):
        b = ins["bbox_3d"]; R = Rotation.from_euler("ZXY", b[6:9]).as_matrix(); lab = cats[ins["bbox_label_3d"]]
        gt.append({"id": i, "label_raw": lab, "label": to_vocab(lab) if lab not in ES_EXCLUDE else None,
                   "center": b[:3], "size": b[3:6], "rot": R.tolist(), "yaw": float(np.arctan2(R[1, 0], R[0, 0]))})
    json.dump(gt, open(f"{out}/gt.json", "w"))
    meta = {"scene": a.scene, "dataset": a.dataset, "split": split, "video_frames": len(vr), "fps": vr.get_avg_fps(),
            "sampled_idx": idx.tolist(), "frames": names, "posed_idx": frames, "world_offset": np.loadtxt(f"{fr}/pose/{names[0]}.txt")[:3, 3].tolist(),
            "depth_src": "posed_images png (mm)" if a.dataset == "scannet" else "lowres_depth png (mm)",
            "K": Kf.tolist(), "W": W, "H": H, "n_gt": len(gt), "n_gt_mapped": sum(g["label"] is not None for g in gt),
            "visible_ids": sorted(vis), "n_posed": len(ims), "snap_gap_s": {"median": float(np.median(gap)), "max": float(gap.max()), "mean": float(gap.mean())},
            "mapping": mapping}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    unm = sorted({g["label_raw"] for g in gt if g["label"] is None and g["label_raw"] not in ES_EXCLUDE})
    print(f"GT {len(gt)} boxes, {meta['n_gt_mapped']} in 288-vocab (non-structure unmapped: {unm}); visible in sampled frames: {len(vis)}; "
          f"image {W}x{H}; mapping {json.dumps({k: v for k, v in mapping.items() if k != 'match_corr_neighbours'})}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True)
    ap.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "scannet", "arkitscenes"])
    ap.add_argument("--mesh_depth", action="store_true", help="scannetpp: render mesh depth even if iphone/depth.bin exists (lz4-framed; no lz4 in env)")
    a = ap.parse_args()
    main_scannetpp(a) if a.dataset == "scannetpp" else main_embodiedscan(a)


if __name__ == "__main__":
    main()
