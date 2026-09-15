"""viser view of one probed scene: depth cloud + frustums of the sampled frames, GT boxes (green), Boxer fused (red).

  python viz_viser.py --dataset scannetpp|scannet|arkitscenes --scene <id> [--port 8080] [--variant fused|fused_md2] [--seconds N]
then open http://<host>:<port> (ssh -L <port>:localhost:<port> if remote). Everything is in the z-up world of gt.json
(ScanNet++ mesh frame / EmbodiedScan axis-aligned global)."""
import argparse, json, os, time
import cv2, numpy as np, viser
from scipy.spatial.transform import Rotation

HERE = "/scratch/ducpham/Working/spatial_reasoning/boxer/scannetpp_probe"                     # probe data/ and out/ stay in the probe folder


def wxyz(R):
    q = Rotation.from_matrix(R).as_quat(); return np.array([q[3], q[0], q[1], q[2]])


def cloud(d, meta, max_pts=400_000):
    K = np.array(meta["K"]); P, C = [], []
    for f in meta["frames"]:
        dep = cv2.imread(f"{d}/frames/depth/{f}.png", -1)
        if dep is None:
            continue
        h, w = dep.shape; s = np.array([w / meta["W"], h / meta["H"]])
        fx, fy, cx, cy = K[0, 0] * s[0], K[1, 1] * s[1], K[0, 2] * s[0], K[1, 2] * s[1]
        rgb = cv2.cvtColor(cv2.resize(cv2.imread(f"{d}/frames/color/{f}.jpg"), (w, h)), cv2.COLOR_BGR2RGB)
        v, u = np.nonzero(dep); z = dep[v, u] / 1000.0
        pc = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
        T = np.loadtxt(f"{d}/frames/pose/{f}.txt"); P.append(pc @ T[:3, :3].T + T[:3, 3]); C.append(rgb[v, u])
    P, C = np.concatenate(P), np.concatenate(C)
    if len(P) > max_pts:
        i = np.random.default_rng(0).choice(len(P), max_pts, replace=False); P, C = P[i], C[i]
    return P.astype(np.float32), C


def add_boxes(server, root, boxes, color, text):
    for i, b in enumerate(boxes):
        server.scene.add_box(f"{root}/b{i}", dimensions=np.array(b["size"]), wireframe=True, color=color,
                             wxyz=wxyz(np.array(b["rot"])), position=np.array(b["center"]))
        server.scene.add_label(f"{root}/l{i}", text(b), position=np.array(b["center"]) + [0, 0, b["size"][2] / 2])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--dataset", default="scannetpp", choices=["scannetpp", "scannet", "arkitscenes"])
    ap.add_argument("--variant", default="fused"); ap.add_argument("--seconds", type=float, default=0, help="exit after N s (headless test)")
    a = ap.parse_args()
    d = f"{HERE}/data/scannet/{a.scene}"; meta = json.load(open(f"{d}/meta.json"))
    assert meta.get("dataset", "scannetpp") == a.dataset, meta.get("dataset")
    boxes = json.load(open(f"{HERE}/out/{a.scene}/boxes.json"))
    server = viser.ViserServer(port=a.port); server.scene.set_up_direction("+z")
    groups = {n: server.scene.add_frame(f"/{n}", show_axes=False) for n in ("gt", "boxer", "cloud", "frustums")}
    P, C = cloud(d, meta); server.scene.add_point_cloud("/cloud/pts", P, C, point_size=0.01)
    K = np.array(meta["K"]); fov = 2 * np.arctan(meta["H"] / 2 / K[1, 1])
    for f in meta["frames"]:
        T = np.loadtxt(f"{d}/frames/pose/{f}.txt")
        server.scene.add_camera_frustum(f"/frustums/{f}", fov=fov, aspect=meta["W"] / meta["H"], scale=0.12,
                                        color=(60, 120, 255), wxyz=wxyz(T[:3, :3]), position=T[:3, 3])
    vis = set(boxes["gt_visible_ids"])
    add_boxes(server, "/gt", [g for g in boxes["gt"] if g["id"] in vis and g["label_raw"].lower() not in
              {"wall", "ceiling", "floor", "object", "split"}], (0, 200, 0), lambda b: f"{b['label_raw']}->{b['label']}")
    add_boxes(server, "/boxer", boxes[a.variant], (230, 30, 30), lambda b: f"{b['label']} {b['prob']:.2f}")
    with server.gui.add_folder("show"):
        for name in ("gt", "boxer", "cloud", "frustums"):
            cb = server.gui.add_checkbox(name, True)
            cb.on_update(lambda e, n=name: setattr(groups[n], "visible", e.target.value))
    print(f"viser [{a.dataset}/{a.scene}]: open http://localhost:{a.port}  (GT {len(vis)} visible boxes, Boxer {len(boxes[a.variant])} {a.variant}, {len(P)} pts)", flush=True)
    if a.seconds:
        time.sleep(a.seconds)
    else:
        server.sleep_forever()


if __name__ == "__main__":
    main()
