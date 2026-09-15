"""viser view of one det64f row (scene-level 64-frame det) in the cam0 frame the rows store.

  python viz_det64f.py --dataset scannetpp|scannet --scene <id> [--port 8080] [--seconds N]
then open http://localhost:<port> (ssh -L <port>:localhost:<port> if remote). --seconds N: run N s and exit (headless test).

Everything is in the camera-0 frame (x right, y down, z forward; axes gizmo at the origin). Red = row boxes
(scannetpp: dropdown det_train / det_train_md2), green = GT (scannetpp segments_anno mapped to the 288 vocab,
scannet EmbodiedScan instances), blue = frustums of the 64 sampled frames (current frame yellow). The frame slider
shows the decoded mp4 frame with both box sets projected. A scene missing from det_train still shows GT + frames
and the drop reason. Also writes the frame-0 overlay to viz_out/<dataset>_<scene>.png.

Poses: scannetpp = iphone/colmap images.txt (mesh frame) at the snapped COLMAP ids; scannet = EmbodiedScan pkl
axis_align @ cam2global at the nearest posed frame. Cloud: scannetpp = mesh_aligned_0.05.ply vertices;
scannet = posed_images depth pngs unprojected with depth_cam2img, coloured from the posed jpgs."""
import argparse, json, os, sys, time
import cv2, decord, numpy as np, open3d as o3d, viser   # cv2 before decord (symbol clash)

W = "/scratch/ducpham/Working"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{W}/spatial_reasoning/scene_graph_idea/VG-LLM/scripts/preprocess")
sys.path.insert(0, HERE)
from utils import embodiedscan_bbox_to_o3d_geo                         # VG-LLM (import before the probe: name clash with boxer utils/)
from prep_scene import read_colmap, load_gt, load_es_sample, sample_indices, ES_EXCLUDE, SCANNET_FPS, SCANNET_POSED
from labelmap import EXCLUDE
from viz_viser import wxyz

DET = f"{W}/dataset/vgllm_data/det64f"
VSI = f"{W}/dataset/vsi_590k/VSI-590K"
MAX_PTS = 400_000


def load_rows(dataset, scene):
    """{variant: [box dicts]} for this scene from det_train*.jsonl (line scan, files are big)."""
    out = {}
    for fn in sorted(os.listdir(f"{DET}/{dataset}")):
        if fn.startswith("det_train") and fn.endswith(".jsonl"):
            for l in open(f"{DET}/{dataset}/{fn}"):
                if f'"scene": "{scene}"' in l:
                    out[fn[:-6]] = json.loads(l)["boxes"]; break
    return out


def obb_in_cam0(center, R, size, world2cam0):
    return dict(center=(world2cam0 @ [*center, 1])[:3], R=world2cam0[:3, :3] @ np.asarray(R), size=np.asarray(size, float))


def corners(b):
    """8 corners + 12 edges of a box dict (center, R, size), o3d convention."""
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox(b["center"], b["R"], b["size"]))
    return np.asarray(ls.points), np.asarray(ls.lines)


def load_scannetpp(scene):
    pm = json.load(open(f"{DET}/scannetpp/_boxer_in/meta/{scene}.json"))       # exists for all 856 videos
    if "drop" in pm:
        raise SystemExit(f"{scene}: prep failed: {pm['drop']}")
    cam0 = np.array(pm["cam2world0"]); gap = np.array(pm["gap_frames"]) / pm["fps"]
    _, _, poses, _, _ = read_colmap(f"{W}/dataset/scannetpp/data/{scene}/iphone/colmap")
    drop = None if os.path.exists(f"{DET}/scannetpp/per_video/{scene}.json") else \
        (f"cam0_gap {gap[0]:.2f}s > 0.25s" if gap[0] > 0.25 else "boxer_zero_boxes")
    gt = [dict(label=f"{g['label_raw']}->{g['label']}", center=g["center"], R=g["rot"], size=g["size"])
          for g in load_gt(f"{W}/dataset/scannetpp/data/{scene}") if g["label_raw"].lower() not in EXCLUDE]
    mesh = o3d.io.read_triangle_mesh(f"{W}/dataset/scannetpp/data/{scene}/scans/mesh_aligned_0.05.ply")
    P, C = np.asarray(mesh.vertices), (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    world = {}                                                                    # per variant: per_video world boxes for the check
    for v, suf in (("det_train", ""), ("det_train_md2", "_md2")):
        p = f"{DET}/scannetpp/per_video/{scene}{suf}.json"
        if os.path.exists(p):
            world[v] = [dict(center=x["bbox_world"]["center"], R=x["bbox_world"]["rot"], size=x["bbox_world"]["size"])
                        for x in json.load(open(p))["instances"]]
    return dict(video=f"{VSI}/scannetppv2/{scene}.mp4", frame_idx=pm["sampled_idx"], gap=gap, cam0=cam0, K=np.array(pm["K"]),
                poses=[poses[c] for c in pm["snapped"]], gt=gt, cloud=(P, C), drop=drop, world=world)


def load_scannet(scene):
    s, cats, _ = load_es_sample("scannet", scene)
    A = np.array(s["axis_align_matrix"]); ims = s["images"]
    posed = np.array([int(im["img_path"].split("/")[-1][:-4]) for im in ims])
    vr = decord.VideoReader(f"{VSI}/scannet/{scene}.mp4", num_threads=1)
    idx = sample_indices(len(vr), float(vr.get_avg_fps()))                       # collator rule, as the builder
    j = np.abs(posed[None, :] - idx[:, None]).argmin(1); gap = np.abs(posed[j] - idx) / SCANNET_FPS
    T = [A @ np.array(im["cam2global"]) for im in ims]
    K = np.array(s["cam2img"])[:3, :3] * np.array([[640 / 1296], [480 / 968], [1]])  # posed jpg 1296x968 -> mp4 640x480
    drop = None if os.path.exists(f"{DET}/scannet/per_video/{scene}.json") else f"snap_gap max {gap.max():.2f}s > 0.50s"
    gt = []
    for ins in s["instances"]:
        lab = cats[ins["bbox_label_3d"]]
        if lab not in ES_EXCLUDE:
            g = embodiedscan_bbox_to_o3d_geo(ins["bbox_3d"]); gt.append(dict(label=lab, center=g.center, R=g.R, size=g.extent))
    Kd = np.array(s["depth_cam2img"])[:3, :3]; P, C = [], []
    for k in sorted(set(j.tolist())):
        d = f"{SCANNET_POSED}/{scene}/{posed[k]:05d}"
        dep = cv2.imread(f"{d}.png", -1); rgb = cv2.cvtColor(cv2.resize(cv2.imread(f"{d}.jpg"), dep.shape[::-1]), cv2.COLOR_BGR2RGB)
        v, u = np.mgrid[0:dep.shape[0]:3, 0:dep.shape[1]:3]; z = dep[v, u] / 1000.0; m = z > 0
        pc = np.stack([(u[m] - Kd[0, 2]) / Kd[0, 0] * z[m], (v[m] - Kd[1, 2]) / Kd[1, 1] * z[m], z[m]], 1)
        P.append(pc @ T[k][:3, :3].T + T[k][:3, 3]); C.append(rgb[v[m], u[m]])
    world = {}
    if not drop:
        world["det_train"] = [dict(center=(g := embodiedscan_bbox_to_o3d_geo(x["bbox_3d"])).center, R=g.R, size=g.extent)
                              for x in json.load(open(f"{DET}/scannet/per_video/{scene}.json"))["instances"]]
    return dict(video=f"{VSI}/scannet/{scene}.mp4", frame_idx=idx.tolist(), gap=gap, cam0=T[0], K=K,
                poses=[T[k] for k in j], gt=gt, cloud=(np.concatenate(P), np.concatenate(C)), drop=drop, world=world)


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=["scannetpp", "scannet"])
    ap.add_argument("--scene", required=True); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seconds", type=float, default=0, help="exit after N s (headless test)")
    a = ap.parse_args()
    S = (load_scannetpp if a.dataset == "scannetpp" else load_scannet)(a.scene)
    n, cam0 = len(S["frame_idx"]), S["cam0"]; w2c = np.linalg.inv(cam0)
    vr = decord.VideoReader(S["video"], num_threads=2); frames = vr.get_batch(S["frame_idx"]).asnumpy()
    H, Wd = frames.shape[1:3]; K = S["K"]
    rows = load_rows(a.dataset, a.scene)
    boxer = {v: [dict(label=b["label"], center=(g := embodiedscan_bbox_to_o3d_geo(b["bbox_3d"])).center, R=g.R, size=g.extent)
                 for b in bs] for v, bs in rows.items()}
    for v, wb in S["world"].items():                # check: row cam0 corners == per_video world boxes through inv(cam0)
        err = max(np.abs(corners(rb)[0] - corners(obb_in_cam0(x["center"], x["R"], x["size"], w2c))[0]).max() for rb, x in zip(boxer[v], wb))
        assert len(wb) == len(boxer[v]) and err < 1e-4, (v, err)
        print(f"check {v}: {len(wb)} boxes, max corner |row - inv(cam0)@world| = {err:.2e}")
    gt = [dict(b, **obb_in_cam0(b["center"], b["R"], b["size"], w2c)) for b in S["gt"]]
    poses = [w2c @ T for T in S["poses"]]          # cam_i in the cam0 frame
    P, C = S["cloud"]; P = (P - cam0[:3, 3]) @ cam0[:3, :3]
    if len(P) > MAX_PTS:
        i = np.random.default_rng(0).choice(len(P), MAX_PTS, replace=False); P, C = P[i], C[i]

    server = viser.ViserServer(port=a.port); server.scene.set_up_direction("-y")
    server.scene.add_frame("/cam0", axes_length=0.3, axes_radius=0.01)
    groups = {k: server.scene.add_frame(f"/{k}", show_axes=False) for k in ["cloud", "frustums", "gt", *boxer]}
    server.scene.add_point_cloud("/cloud/pts", P.astype(np.float32), C, point_size=0.01)
    fov = 2 * np.arctan(H / 2 / K[1, 1]); frusta = []
    for i, T in enumerate(poses):
        frusta.append(server.scene.add_camera_frustum(f"/frustums/{i}", fov=fov, aspect=Wd / H, scale=0.1, color=(60, 120, 255),
                                                      wxyz=wxyz(T[:3, :3]), position=T[:3, 3]))
        server.scene.add_label(f"/frustums/l{i}", str(i), position=T[:3, 3])
    for k, bs, col in [("gt", gt, (0, 200, 0)), *[(v, boxer[v], (230, 30, 30)) for v in boxer]]:
        for i, b in enumerate(bs):
            server.scene.add_box(f"/{k}/b{i}", dimensions=b["size"], wireframe=True, color=col, wxyz=wxyz(b["R"]), position=b["center"])
            server.scene.add_label(f"/{k}/l{i}", b["label"], position=b["center"])

    variant = next(iter(boxer), None)
    for v in list(boxer)[1:]:
        groups[v].visible = False
    def overlay(i):
        img = np.ascontiguousarray(frames[i]); T = np.linalg.inv(poses[i])
        draw(img, gt, T, K, (0, 200, 0))
        if variant:
            draw(img, boxer[variant], T, K, (255, 40, 40))
        return img
    os.makedirs(f"{HERE}/viz_out", exist_ok=True)
    cv2.imwrite(f"{HERE}/viz_out/{a.dataset}_{a.scene}.png", cv2.cvtColor(overlay(0), cv2.COLOR_RGB2BGR))

    server.gui.add_markdown(f"**{a.dataset}/{a.scene}**  \nn_frames {n}, Boxer {len(boxer.get(variant, []))}, GT {len(gt)}, "
                            f"{len(P)} pts  \nsnap gap max {S['gap'].max():.2f}s (cam0 {S['gap'][0]:.2f}s)  \n"
                            + (f"**DROPPED: {S['drop']}**" if S["drop"] else "in det_train"))
    with server.gui.add_folder("show"):
        for k in groups:
            cb = server.gui.add_checkbox(k, groups[k].visible)
            cb.on_update(lambda e, k=k: setattr(groups[k], "visible", e.target.value))
    if len(boxer) > 1:
        dd = server.gui.add_dropdown("variant", list(boxer), initial_value=variant)
        @dd.on_update
        def _(e):
            nonlocal variant
            variant = e.target.value
            for v in boxer:
                groups[v].visible = v == variant
            panel.image = overlay(sl.value)
    sl = server.gui.add_slider("frame", 0, n - 1, 1, 0)
    panel = server.gui.add_image(overlay(0), label="frame (red=row, green=GT)")
    @sl.on_update
    def _(e):
        for i, f in enumerate(frusta):
            f.color = (255, 220, 0) if i == e.target.value else (60, 120, 255)
        panel.image = overlay(e.target.value)
    frusta[0].color = (255, 220, 0)
    print(f"viser [{a.dataset}/{a.scene}]: open http://localhost:{a.port}  (Boxer {len(boxer.get(variant, []))}, GT {len(gt)}, "
          f"{n} frames, drop={S['drop']})", flush=True)
    time.sleep(a.seconds) if a.seconds else server.sleep_forever()


if __name__ == "__main__":
    main()
