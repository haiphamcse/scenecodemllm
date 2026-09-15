"""viser view of one posed-64f det_es scene (EmbodiedScan pixels) of VSI-590K ScanNet / ARKitScenes, in the cam0 frame.

  python viz_posed64f.py --source scannet|arkitscenes|scannetpp --scene <id> [--port 8080] [--seconds N]
then open http://localhost:<port> (ssh -L <port>:localhost:<port> if remote). --seconds N: run N s and exit (headless test).

Everything is in the camera-0 (= frame00) frame the det row stores (x right, y down, z forward; axes gizmo at the origin):
green = det.json boxes (EmbodiedScan GT, visible union), depth cloud unprojected from the 64 posed depth pngs and coloured
from the det_es frames. The frame slider shows the saved det_es frame with the boxes projected (K scaled native -> llm_hw).
Depth: ScanNet = posed_images/<scene>/<posed_id>.png (640x480 mm, depth_cam2img from the EmbodiedScan pkl); ARKit =
<id>_frames/lowres_depth/<posed_id>.png (256x192 mm, meta cam2img, rotated by rot90_k like the colour frame), fetched
on demand from the Apple 3DOD zip (only the 64 needed pngs are kept, in the official layout); ScanNet++ (dir scannetppv2,
build_posed64f_scannetpp.py) = open3d raycast of scans/mesh_aligned_0.05.ply at the 64 aligned_pose frames (320x240, meta K)."""
import argparse, json, os, tempfile, time, urllib.request, zipfile
import cv2, numpy as np, open3d as o3d, viser

W = "/scratch/ducpham/Working"
from viz_det64f import VSI, MAX_PTS, corners, draw, embodiedscan_bbox_to_o3d_geo, load_es_sample, SCANNET_POSED, wxyz  # noqa: F401
from prep_scene import unproject, MeshDepth

ARKIT = f"{W}/dataset/embodiedscan_arkitscenes/arkitscenes/Training"
SCANNETPP = f"{W}/dataset/scannetpp/data"
VSI_DIR = {"scannetpp": "scannetppv2"}                 # VSI-590K folder differs from the source name
ZIP = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/threedod/Training/{}.zip"


def arkit_depth_dir(scene, names):
    """Official lowres_depth dir for the scene; missing pngs are pulled from the Apple zip (downloaded to a temp file, deleted)."""
    d = f"{ARKIT}/{scene}/{scene}_frames/lowres_depth"
    todo = [n for n in names if not os.path.exists(f"{d}/{n}")]
    if todo:
        os.makedirs(d, exist_ok=True)
        print(f"fetching {ZIP.format(scene)} (~100 MB) for {len(todo)} lowres_depth pngs", flush=True)
        with tempfile.NamedTemporaryFile(dir=f"{ARKIT}/{scene}", suffix=".zip") as tmp:
            urllib.request.urlretrieve(ZIP.format(scene), tmp.name)
            with zipfile.ZipFile(tmp.name) as zf:
                for n in todo:
                    open(f"{d}/{n}", "wb").write(zf.read(f"{scene}/{scene}_frames/lowres_depth/{n}"))
    return d


def cloud(source, scene, meta, frames):
    """Depth cloud in the EmbodiedScan global frame, coloured from the det_es frames (step-2 pixel grid)."""
    fr, k = meta["frames"], meta.get("rot90_k") or 0
    if source == "scannet":
        Kd = np.array(load_es_sample("scannet", scene)[0]["depth_cam2img"])[:3, :3]
        paths = [f"{SCANNET_POSED}/{scene}/{f['posed_id']}.png" for f in fr]
    elif source == "scannetpp":                        # no depth pngs: raycast the GT mesh at the frame poses
        (H, Wd) = meta["native_hw"]
        render = MeshDepth(f"{SCANNETPP}/{scene}/scans/mesh_aligned_0.05.ply", np.array(fr[0]["cam2img"]), Wd, H, 320, 240)
        paths = [None] * len(fr)
    else:
        d = arkit_depth_dir(scene, [f["posed_id"] + ".png" for f in fr])
        paths = [f"{d}/{f['posed_id']}.png" for f in fr]
    P, C = [], []
    for f, p, img in zip(fr, paths, frames):
        if not np.isfinite(f["cam2global"]).all():
            print(f"warning: frame{f['frame']:02d} ({f['posed_id']}) has a non-finite cam2global, skipped from the cloud", flush=True); continue
        dep = np.rot90(cv2.imread(p, -1), k) if p else render(np.array(f["cam2global"]))
        K = Kd if source == "scannet" else render.K if source == "scannetpp" else np.array(f["cam2img"])
        v, u = np.mgrid[0:dep.shape[0]:2, 0:dep.shape[1]:2]; m = dep[v, u] > 0
        P.append(unproject(dep, K, np.array(f["cam2global"]), step=2)); C.append(cv2.resize(img, dep.shape[::-1])[v[m], u[m]])
    P, C = np.concatenate(P), np.concatenate(C)
    if len(P) > MAX_PTS:
        i = np.random.default_rng(0).choice(len(P), MAX_PTS, replace=False); P, C = P[i], C[i]
    return P, C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True, choices=["scannet", "arkitscenes", "scannetpp"])
    ap.add_argument("--scene", required=True); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seconds", type=float, default=0, help="exit after N s (headless test)")
    a = ap.parse_args()
    d = f"{VSI}/{VSI_DIR.get(a.source, a.source)}/{a.scene}/det_es"
    meta, row = json.load(open(f"{d}/meta.json")), json.load(open(f"{d}/det.json"))
    fr, n = meta["frames"], meta["n_frames"]
    frames = [cv2.cvtColor(cv2.imread(f"{d}/frames/frame{i:02d}.jpg"), cv2.COLOR_BGR2RGB) for i in range(n)]
    (H, Wd), (h, w) = meta["native_hw"], meta["llm_hw"]
    Ks = [np.diag([w / Wd, h / H, 1]) @ np.array(f["cam2img"]) for f in fr]                 # native -> saved llm_hw pixels
    cam0 = np.array(fr[0]["cam2global"]); poses = [np.linalg.inv(np.array(f["cam2global"])) @ cam0 for f in fr]  # cam0 -> cam_i
    boxes = [dict(label=b["label"], center=(g := embodiedscan_bbox_to_o3d_geo(b["bbox_3d"])).center, R=g.R, size=g.extent)
             for b in row["boxes"]]
    P, C = cloud(a.source, a.scene, meta, frames); P = (P - cam0[:3, 3]) @ cam0[:3, :3]

    c = np.array([b["center"] for b in boxes]); uv = (Ks[0] @ c.T).T; uv = uv[:, :2] / uv[:, 2:3]
    inside = int(((c[:, 2] > 0) & (uv >= 0).all(1) & (uv[:, 0] < w) & (uv[:, 1] < h)).sum())
    pc, inbox = o3d.utility.Vector3dVector(P), np.zeros(len(P), bool)          # cloud/box registration sanity
    for b in row["boxes"]:
        inbox[np.asarray(embodiedscan_bbox_to_o3d_geo(b["bbox_3d"]).get_point_indices_within_bounding_box(pc), dtype=int)] = True   # [] -> float
    print(f"check: {inside}/{len(boxes)} box centers inside frame00 (boxes = visible union over all frames), "
          f"cloud {len(P)} pts, {inbox.mean():.0%} inside a GT box", flush=True)

    server = viser.ViserServer(port=a.port); server.scene.set_up_direction("-y")
    server.scene.add_frame("/cam0", axes_length=0.3, axes_radius=0.01)
    groups = {k: server.scene.add_frame(f"/{k}", show_axes=False) for k in ["cloud", "boxes"]}
    server.scene.add_point_cloud("/cloud/pts", P.astype(np.float32), C, point_size=0.01)
    for i, b in enumerate(boxes):
        server.scene.add_box(f"/boxes/b{i}", dimensions=b["size"], wireframe=True, color=(0, 200, 0), wxyz=wxyz(b["R"]), position=b["center"])
        server.scene.add_label(f"/boxes/l{i}", b["label"], position=b["center"])

    def overlay(i):
        img = np.ascontiguousarray(frames[i]); draw(img, boxes, poses[i], Ks[i], (0, 200, 0)); return img
    server.gui.add_markdown(f"**{a.source}/{a.scene}**  \nn_frames {n}, boxes {len(boxes)}, {len(P)} pts, pixel_source {meta['pixel_source']}  \n"
                            f"frame00: t={fr[0]['time_s']:.2f}s, posed {fr[0]['posed_id']}")
    with server.gui.add_folder("show"):
        for k in groups:
            cb = server.gui.add_checkbox(k, True)
            cb.on_update(lambda e, k=k: setattr(groups[k], "visible", e.target.value))
    sl = server.gui.add_slider("frame", 0, n - 1, 1, 0)
    panel = server.gui.add_image(overlay(0), label="det_es frame (green=det.json boxes)")
    sl.on_update(lambda e: setattr(panel, "image", overlay(e.target.value)))
    print(f"viser [{a.source}/{a.scene}]: open http://localhost:{a.port}  ({len(boxes)} boxes, {n} frames)", flush=True)
    time.sleep(a.seconds) if a.seconds else server.sleep_forever()


if __name__ == "__main__":
    main()
