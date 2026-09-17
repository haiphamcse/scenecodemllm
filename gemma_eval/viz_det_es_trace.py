"""viser view of one VSI-Bench eval ScanNet det_es cache (cam0 frame) plus the exported gemma reasoning traces of the scene.

  python viz_det_es_trace.py --scene <id> [--reasoning <jsonl>] [--port 8080] [--seconds N] [--save out.png] [--no_cloud]
then open http://localhost:<port> (ssh -L <port>:localhost:<port> if remote). --seconds N: run N s and exit.
--save: write the frame-0 overlay; with --seconds 0 it exits without starting a server (headless test).

Scene view = viz_det_es/viz_det_es.py (green det.json boxes, blue frustums, depth cloud, frame slider with projected boxes).
Trace panel: a dropdown of the scene's questions ("<question_type> #<id>") and a markdown box with the question, options,
ground truth, leak flag, tokens / seconds and the reasoning text of the selected one."""
import argparse, re, json, os, time
import cv2, numpy as np, open3d as o3d

from utils import scene_dir, load_es_sample, load_scene, cloud, draw, wxyz

BLUE, YELLOW, GREEN = (60, 120, 255), (255, 220, 0), (0, 200, 0)
REASONING = ("/home/ducpham/scratch/Working/spatial_reasoning/finetuning/gemini_eval/results/det_vsibench_reasoning/"
             "gemma-4-26B-A4B-it_video64_posed/reasoning.jsonl")


def load_traces(path, scene):
    """{"<question_type> #<id>": row} for the scene, sorted by type then id."""
    rows = [r for r in map(json.loads, open(path)) if r["scene_name"] == scene]
    return {f"{r['question_type']} #{r['id']}": r for r in sorted(rows, key=lambda r: (r["question_type"], r["id"]))}


def mdx_safe(s):
    """viser renders markdown as MDX: bare { } < > are JS/JSX and break the parse (LaTeX like \\vec{v}). Escape them."""
    return re.sub(r"([{}<>])", r"\\\1", s)


def trace_md(r):
    opts = "  \n".join(map(mdx_safe, r["options"])) if r["options"] else "(none)"
    leak = f"**leaked** {r['leaked']}" + (f" (match `{mdx_safe(r['leak_match'])}`)" if r["leak_match"] else "")
    q = mdx_safe(r["question"]).replace("\n", "  \n")
    return (f"**question**  \n{q}\n\n**options**  \n{opts}\n\n**ground truth** {mdx_safe(str(r['ground_truth']))}  \n{leak}  \n"
            f"tokens in/out {r['n_input_tokens']}/{r['n_output_tokens']}, {r['seconds']:.1f} s\n\n---\n\n{mdx_safe(r['reasoning'])}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--reasoning", default=REASONING)
    ap.add_argument("--port", type=int, default=8080); ap.add_argument("--seconds", type=float, default=0, help="exit after N s")
    ap.add_argument("--save", help="write the frame-0 overlay png here"); ap.add_argument("--no_cloud", action="store_true")
    a = ap.parse_args()
    es = load_es_sample(a.scene)
    meta, boxes, frames, poses, Ks = load_scene(scene_dir(a.scene), es)
    n, cam0 = len(frames), np.array(meta["frames"][0]["cam2global"]); h, w = frames[0].shape[:2]
    traces = load_traces(a.reasoning, a.scene)
    print(f"{len(traces)} traces for {a.scene} in {a.reasoning}", flush=True)
    PC = None if a.no_cloud else cloud(a.scene, meta, frames, es)
    if PC:
        P, C = PC; P = (P - cam0[:3, 3]) @ cam0[:3, :3]
        pc, inbox = o3d.utility.Vector3dVector(P), np.zeros(len(P), bool)              # cloud / box registration sanity
        for b in boxes:
            g = o3d.geometry.OrientedBoundingBox(b["center"], b["R"], b["size"])
            inbox[np.asarray(g.get_point_indices_within_bounding_box(pc), dtype=int)] = True
        print(f"check: cloud {len(P)} pts, {inbox.mean():.0%} inside a det box", flush=True)
    else:
        print("no cloud" + ("" if a.no_cloud else f" (no depth on disk for {a.scene})"), flush=True)

    def overlay(i):
        img = np.ascontiguousarray(frames[i])
        if np.isfinite(poses[i]).all():
            draw(img, boxes, poses[i], Ks[i], GREEN)
        return img
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        cv2.imwrite(a.save, cv2.cvtColor(overlay(0), cv2.COLOR_RGB2BGR)); print(f"saved {a.save} ({len(boxes)} boxes)", flush=True)
        if not a.seconds:
            return

    import viser
    server = viser.ViserServer(port=a.port); server.scene.set_up_direction("-y")
    server.scene.add_frame("/cam0", axes_length=0.3, axes_radius=0.01)
    groups = {k: server.scene.add_frame(f"/{k}", show_axes=False) for k in ["cloud", "frustums", "boxes"]}
    if PC:
        server.scene.add_point_cloud("/cloud/pts", P.astype(np.float32), C, point_size=0.01)
    frusta = {}
    for i, (T, K) in enumerate(zip(poses, Ks)):
        if np.isfinite(T).all():
            T = np.linalg.inv(T)                                                          # cam_i in the cam0 frame
            frusta[i] = server.scene.add_camera_frustum(f"/frustums/{i}", fov=2 * np.arctan(h / 2 / K[1, 1]), aspect=w / h, scale=0.1,
                                                        color=BLUE, wxyz=wxyz(T[:3, :3]), position=T[:3, 3])
            server.scene.add_label(f"/frustums/l{i}", str(i), position=T[:3, 3])
    for i, b in enumerate(boxes):
        server.scene.add_box(f"/boxes/b{i}", dimensions=b["size"], wireframe=True, color=GREEN, wxyz=wxyz(b["R"]), position=b["center"])
        server.scene.add_label(f"/boxes/l{i}", b["label"], position=b["center"])

    server.gui.add_markdown(f"**eval scannet/{a.scene}**  \nn_frames {n}, boxes {len(boxes)}, {len(P) if PC else 0} pts  \n"
                            f"pixel_source {meta.get('pixel_source', '?')}, pose_source {meta.get('pose_source', '?')}")
    with server.gui.add_folder("show"):
        for k in groups:
            cb = server.gui.add_checkbox(k, True)
            cb.on_update(lambda e, k=k: setattr(groups[k], "visible", e.target.value))
    sl = server.gui.add_slider("frame", 0, n - 1, 1, 0)
    panel = server.gui.add_image(overlay(0), label="det_es frame (green = det.json boxes)")
    @sl.on_update
    def _(e):
        for i, f in frusta.items():
            f.color = YELLOW if i == e.target.value else BLUE
        panel.image = overlay(e.target.value)
    if 0 in frusta:
        frusta[0].color = YELLOW
    with server.gui.add_folder("trace"):
        if traces:
            dd = server.gui.add_dropdown("question", list(traces))
            md = server.gui.add_markdown(trace_md(traces[dd.value]))
            dd.on_update(lambda e: setattr(md, "content", trace_md(traces[e.target.value])))
        else:
            server.gui.add_markdown("no traces for this scene")
    print(f"viser [eval scannet/{a.scene}]: open http://localhost:{a.port}  ({len(boxes)} boxes, {n} frames, {len(traces)} traces)", flush=True)
    time.sleep(a.seconds) if a.seconds else server.sleep_forever()


if __name__ == "__main__":
    main()
