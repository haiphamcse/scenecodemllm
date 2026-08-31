"""3D detection inference for the idea_3i_from_idea4a checkpoint on VSI-Bench ScanNet++.

The checkpoint is a VSI-Bench QA LoRA trained on top of the MERGED idea_4a detection model
(adapter_config: base_model_name_or_path = results/idea_4a_video_scratch_merged1600), so it
still answers idea_4a's detection prompt. This runs that prompt over VSI-Bench ScanNet++
videos and shows the boxes in viser.

Two modes, one file:

  run   (default)  per scene: 32 cached frames -> box json + VGGT cloud + ICP to the GT mesh
  --show SCENE     viser: predicted boxes over the ScanNet++ mesh, in camera-0 frame

Frames: the idea_3i frame cache (32 frames @ 1 fps), the video regime this checkpoint was
trained under. Predicted boxes are METRIC, in the camera frame of frame 0.

Alignment: no iPhone poses are on disk for these scenes, so the mesh (world frame) is brought
into camera-0 frame by registering VGGT-Omega's own point cloud -- which IS in camera-0 frame
-- to the mesh: floor-normal z-up init, then a scale x yaw sweep of Open3D sim3 ICP, scored
two-way at a tight radius. Scale must be solved for, because VGGT-Omega normalises scene scale
(measured: a 6.9 m room comes back 1.25 m across), and the sweep is anchored by CAMERA HEIGHT
(the handheld camera is 0.9-2.0 m above the floor) because scoring alone has two degenerate
optima -- see register_to_mesh. The fitted scale is then divided out, so what is stored maps
world -> the METRIC camera-0 frame the boxes live in. The boxes are never touched by the fit,
so a box floating off the geometry is a real error: measured box-on-geometry rates run from
1/23 to 34/34 across scenes, and the low ones are the model, not the alignment.

Scenes: the ScanNet++ scenes that have BOTH a VSI-Bench video and a downloaded mesh (10 of
them today), in sorted order.

  conda activate vsibench_eval_full
  cd spatial_reasoning/finetuning
  python idea_3i_from_idea4a/detect_vsibench.py                       # infer 10 scenes
  python idea_3i_from_idea4a/detect_vsibench.py --show 0d2ee665be     # viewer
  python idea_3i_from_idea4a/detect_vsibench.py --show 0d2ee665be --check   # headless check
"""

from __future__ import annotations

import os

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_FINETUNING_ROOT = _THIS_DIR.parent
# _THIS_DIR first (its collator/frame_cache/model_with_vggt must win over common/ and over the
# idea_4a copies); the idea_4a dir goes LAST so only graph_vgllm.py -- which exists nowhere
# else -- is picked up from it.
for _p in (_FINETUNING_ROOT / "common", _THIS_DIR):
    _sp = str(_p)
    if _sp in sys.path:
        sys.path.remove(_sp)
    sys.path.insert(0, _sp)
sys.path.append(str(_FINETUNING_ROOT / "idea_4a_sg_perc_scannetv2_vgjson_video"))

from graph_vgllm import parse as parse_boxes  # noqa: E402

_CKPT = _FINETUNING_ROOT / "results/idea_3i_from_idea4a_scratch1600_done/checkpoint-1300"
_BASE = _FINETUNING_ROOT / "results/idea_4a_video_scratch_merged1600"
_VIDEO_DIR = Path("/home/ducpham/scratch/Working/dataset/vsib/vsibench/scannetpp")
_SCANNETPP = Path("/home/ducpham/scratch/Working/dataset/scannetpp/data")
_CACHE_ROOT = Path(os.environ.get("SR_CACHE_ROOT", "/home/ducpham/scratch/Working/dataset/vsi_590k/vsi_590k_frame_cache"))
_VGGT_CKPT = (
    os.environ.get("SR_HF_HOME", "/home/ducpham/scratch/Working/cache")
    + "/hub/models--facebook--VGGT-Omega/"
    "snapshots/05654241adc2f218dfb089c373a011f8a7040576/vggt_omega_1b_512.pt"
)

# idea_4a's detection prompt, verbatim from its collator (SCENE_GRAPH_QUESTION). Copied
# rather than imported: that module is a sibling collator.py and would shadow this folder's.
SCENE_GRAPH_QUESTION = (
    "Detect the 3D bounding boxes in the camera coordinate system of the first frame.\n"
    "Output a json list whose first entry is {\"n\": N} giving the number of objects, "
    "followed by one entry per object containing the object name in \"label\" and its "
    "3D bounding box in \"bbox_3d\".\n"
    "Order the objects from largest to smallest by volume.\n"
    "The 3D bounding box format should be [x_center, y_center, z_center, x_size, "
    "y_size, z_size, yaw, roll, pitch].")

PRED_COLOR = (240, 70, 70)


def scene_videos(num: int) -> list[Path]:
    """VSI-Bench ScanNet++ videos whose scene also has a downloaded mesh."""
    with_mesh = {
        d.name for d in _SCANNETPP.iterdir()
        if (d / "scans" / "mesh_aligned_0.05.ply").exists()
    }
    return sorted(p for p in _VIDEO_DIR.glob("*.mp4") if p.stem in with_mesh)[:num]


# --------------------------------------------------------------------------- #
# inference                                                                    #
# --------------------------------------------------------------------------- #
def load_model(args):
    """Merged idea_4a base + VGGT/Perceiver + the QA LoRA. Same order as eval.py."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor

    from model_with_vggt import Qwen3VLWithVggtForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = Qwen3VLWithVggtForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(args.device)
    # Must precede the adapter: this creates vggt_projector, which the adapter carries in
    # modules_to_save. Geometry must match training or the latents are garbage.
    model.initialize_vggt(
        args.vggt_checkpoint,
        tokenizer=processor.tokenizer,
        vggt_embed_dim=args.vggt_embed_dim,
        frame_num_latents=args.frame_num_latents,
        camera_num_latents=args.camera_num_latents,
        frame_placeholder_token=args.frame_placeholder_token,
        camera_placeholder_token=args.camera_placeholder_token,
        frame_widening_factor=args.frame_widening_factor,
    )
    model = PeftModel.from_pretrained(model, str(args.checkpoint)).to(args.device)
    model.eval()
    return model, processor


def generate_boxes(model, processor, video: Path, args):
    """One greedy decode of the box list for a video. Returns (text, raw video tensor)."""
    import torch
    from qwen_vl_utils import process_vision_info

    from collator import real_video_metadata, user_only_messages
    from frame_cache import cache_paths_for, load_cached_frames

    npy_path, json_path = cache_paths_for(str(video), args.cache_root)
    if not (npy_path.exists() and json_path.exists()):
        raise SystemExit(f"{video.stem}: no frame cache at {npy_path}; run pre_extract_videos.py")
    frames, meta = load_cached_frames(npy_path, json_path)

    frame_text = args.frame_placeholder_token * args.frame_num_latents
    camera_text = args.camera_placeholder_token * args.camera_num_latents
    if args.prompt_layout == "idea_3i":
        example = {"video": str(video), "conversations": [{"value": SCENE_GRAPH_QUESTION}]}
        messages = user_only_messages(frames, meta, example, frame_text, camera_text)
    else:
        # idea_4a's _graph_user_content, verbatim -- the layout the box format was trained
        # under, down to the period after "context". Asked with idea_3i's QA layout instead,
        # this checkpoint answers with a bare 9-number list and no label.
        messages = [{"role": "user", "content": [
            {"type": "text", "text": frame_text + camera_text},
            {"type": "text", "text": "This is the 3D scene context.\n"},
            {"type": "video", "video": frames},  # no resized_*: qwen's own video budget
            {"type": "text", "text": "These are the RGB frames of the same scene, in the same order.\n"},
            {"type": "text", "text": "Use both the 3D scene context and the frames.\n"},
            {"type": "text", "text": SCENE_GRAPH_QUESTION},
        ]}]

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, vision, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True,
        image_patch_size=args.image_patch_size, return_video_metadata=True,
    )
    video_inputs, _fabricated = map(list, zip(*vision))
    # VGGT's copy is always the pinned-resize tensor (the cache frames at their own size),
    # which is what both trainings fed the Perceiver; the LLM's copy may differ above.
    vggt_message = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "resized_height": meta["resized_height"], "resized_width": meta["resized_width"]},
    ]}]
    _, vggt_vision, _ = process_vision_info(
        vggt_message, return_video_kwargs=True,
        image_patch_size=args.image_patch_size, return_video_metadata=True,
    )
    raw_videos = [v for v, _ in vggt_vision]
    inputs = processor(
        text=prompt,
        videos=video_inputs,
        # The list branch fabricates metadata; the cache holds the real decode metadata, so
        # timestamps match what training saw.
        video_metadata=[real_video_metadata(meta)] * len(video_inputs),
        return_tensors="pt",
        padding=True,
        **video_kwargs,
    )
    inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["raw_videos"] = [v.to(args.device) for v in raw_videos]

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    text = processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return text, raw_videos[0]


def vggt_cloud(raw_video, vggt, conf_percentile: float):
    """VGGT-Omega depth+pose -> (points, colors) in the camera frame of frame 0, metric.

    Same unprojection as legacy/idea_3j/reconstruct.py, minus its z-up rotation and
    vertical rescale: those normalise the cloud into VGGT's own frame, and here the whole
    point is to stay in the metric camera-0 frame the predicted boxes live in.
    """
    import torch
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = next(vggt.parameters()).device
    dtype = next(vggt.parameters()).dtype
    images = raw_video.to(device=device, dtype=dtype)
    if images.max() > 1.0:
        images = images / 255.0

    with torch.no_grad():
        pred = vggt(images)
    extrinsic, intrinsic = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    out = {k: v.detach().float().cpu().numpy().squeeze(0)
           for k, v in (("images", pred["images"]), ("depth", pred["depth"]),
                        ("depth_conf", pred["depth_conf"]),
                        ("extrinsic", extrinsic), ("intrinsic", intrinsic))}

    depth = out["depth"][..., 0]                     # (S, H, W)
    s, h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    K = out["intrinsic"]
    cam = np.stack([(x[None] - K[:, 0, 2][:, None, None]) / K[:, 0, 0][:, None, None] * depth,
                    (y[None] - K[:, 1, 2][:, None, None]) / K[:, 1, 1][:, None, None] * depth,
                    depth], axis=-1)
    R, t = out["extrinsic"][:, :3, :3], out["extrinsic"][:, :3, 3]
    # extrinsic is world<-camera with world == camera 0, so this lands in camera-0 frame.
    points = np.einsum("sij,shwj->shwi", np.transpose(R, (0, 2, 1)),
                       cam - t[:, None, None, :]).reshape(-1, 3)
    colors = out["images"].transpose(0, 2, 3, 1).reshape(-1, 3)

    conf = out["depth_conf"].reshape(-1)
    keep = (conf >= np.percentile(conf, conf_percentile)) & (conf > 1e-5)
    # Camera centres travel with the cloud: they are what pins the scale (register_to_mesh).
    centres = np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), -t)
    return (points[keep].astype(np.float32), colors[keep].astype(np.float32),
            centres.astype(np.float32))


# --------------------------------------------------------------------------- #
# mesh registration                                                            #
# --------------------------------------------------------------------------- #
def _pcd(points):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return p


def _zup_rotation(points: np.ndarray) -> np.ndarray:
    """Rotation taking the cloud's dominant plane normal to +z, floor below the points.

    The camera-0 frame is OpenCV (+y down), the mesh frame is z-up, so an ICP yaw sweep on
    its own starts ~90 degrees out and never recovers. The floor plane gives that missing
    rotation; the sign is fixed by putting most of the cloud ABOVE the plane.
    """
    import open3d as o3d

    plane, _ = _pcd(points).segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=400)
    n = np.asarray(plane[:3], dtype=float)
    n /= np.linalg.norm(n)
    if np.mean(points @ n + plane[3]) < 0:
        n, plane = -n, [-v for v in plane]
    v = np.cross(n, [0.0, 0.0, 1.0])
    c = float(n @ [0.0, 0.0, 1.0])
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def register_to_mesh(src: np.ndarray, cams: np.ndarray, tgt: np.ndarray, args):
    """Fit the VGGT cloud to the mesh. Returns (world->metric-camera-0 4x4, scale, rmse, score).

    Coarse init and sim3 ICP from legacy/idea_3k/annotate_semantics.py (scale by z-extent
    ratio, yaw sweep, centroids in xy, floors in z), preceded by the floor-normal z-up
    rotation above.

    The fit MUST solve for scale: VGGT-Omega normalises scene scale, so its cloud comes out
    ~5x too small (measured: a 6.9 m room reconstructed 1.25 m across). A rigid fit therefore
    "succeeds" -- the shrunken cloud sits inside the room and every point finds a
    correspondence -- while placing nothing where it belongs.

    The returned transform is rigid anyway: the fitted similarity is s*R, and dividing it out
    maps world -> the METRIC camera-0 frame, which is exactly the frame the predicted boxes
    are in. The boxes are never touched.
    """
    import open3d as o3d

    rz_up = _zup_rotation(src)
    src_up = src @ rz_up.T
    # Subsample: the sweep below is O(scales * yaws) ICP runs, and 5M points is 100x more
    # than a rigid-body fit needs.
    rng = np.random.default_rng(0)
    src_ds = src_up[rng.choice(len(src_up), min(len(src_up), args.icp_points), replace=False)]
    src_pcd = _pcd(src_ds)
    tgt_pcd = _pcd(tgt).voxel_down_sample(args.icp_voxel)
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    # z-extent ratio: the right order of magnitude, but wrong whenever the walkthrough covers
    # only part of the room, so it only centres the sweep.
    # Scale band from the CAMERA HEIGHT, not from extents. VGGT's scene scale is arbitrary,
    # and a free scale sweep has two degenerate optima (see the scoring note below), so the
    # sweep is restricted to scales that put the handheld camera 0.9-2.0 m above the floor.
    # That is a physical fact about how the video was shot, independent of both the mesh and
    # the model, and it is what separates the right optimum from the pretty ones.
    floor_z = np.percentile(src_up[:, 2], 1.0)
    cam_h = float(np.median((cams @ rz_up.T)[:, 2] - floor_z))
    if cam_h <= 1e-6:
        raise SystemExit("camera height came out non-positive; the z-up rotation is upside down")

    best = None
    for scale in np.geomspace(args.min_cam_height / cam_h, args.max_cam_height / cam_h,
                              args.scale_steps):
        for yaw in np.arange(0.0, 360.0, args.yaw_step):
            a = np.deg2rad(yaw)
            m = scale * np.array([[np.cos(a), -np.sin(a), 0],
                                  [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
            init = np.eye(4)
            init[:3, :3] = m
            moved = src_ds @ m.T
            init[:2, 3] = tgt[:, :2].mean(0) - moved[:, :2].mean(0)
            init[2, 3] = tgt[:, 2].min() - moved[:, 2].min()
            reg = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, args.icp_max_corr, init, est, crit)
            # Score at a TIGHT radius, both ways. One-way scoring is degenerate in BOTH
            # directions and sim3 ICP walks straight into it: a cloud fitted too small sits
            # inside the room and matches everything within a loose radius (measured: 1.000
            # at half the true scale), and one collapsed to a blob scores 1.000 even at a
            # tight radius (measured: 1.000 at scale 0.05). Coverage -- the share of the MESH
            # explained by the cloud -- is what a collapse cannot fake, so the two are
            # combined as a harmonic mean. Coverage is never near 1: the walkthrough only
            # sees part of the room.
            fwd = o3d.pipelines.registration.evaluate_registration(
                src_pcd, tgt_pcd, args.icp_score_radius, reg.transformation)
            back = o3d.pipelines.registration.evaluate_registration(
                tgt_pcd, src_pcd, args.icp_score_radius,
                np.linalg.inv(reg.transformation))
            p, r = fwd.fitness, back.fitness
            score = 2 * p * r / (p + r) if p + r else 0.0
            key = (score, -fwd.inlier_rmse)
            if best is None or key > best[0]:
                best = (key, np.asarray(reg.transformation), fwd.inlier_rmse, score)

    to_zup = np.eye(4)
    to_zup[:3, :3] = rz_up
    sim3 = best[1] @ to_zup                       # vggt-camera-0 -> world, with scale
    scale = float(abs(np.linalg.det(sim3[:3, :3])) ** (1.0 / 3.0))
    rot = sim3[:3, :3] / scale
    world_to_cam0 = np.eye(4)                      # world -> metric camera-0
    world_to_cam0[:3, :3] = rot.T
    world_to_cam0[:3, 3] = -rot.T @ sim3[:3, 3]
    return world_to_cam0, scale, best[2], best[3]


def load_mesh(scene: str):
    """ScanNet++ mesh vertices + colors, world frame."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(_SCANNETPP / scene / "scans" / "mesh_aligned_0.05.ply"))
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.vertex_colors, dtype=np.float32))


def run(args) -> None:
    videos = scene_videos(args.num)
    print(f"{len(videos)} scenes with a video AND a mesh: {[v.stem for v in videos]}")
    args.out.mkdir(parents=True, exist_ok=True)

    from vggt_omega.models import VGGTOmega
    import torch

    model, processor = load_model(args)
    vggt = VGGTOmega()
    vggt.load_state_dict(torch.load(args.vggt_checkpoint, map_location="cpu"))
    # float32, as legacy/idea_3j/pre_extract_teacher.py runs it: the camera/depth heads
    # contain LayerNorms that reject bf16 ("expected scalar type Float but found BFloat16").
    # Only the aggregator inside the LLM runs bf16.
    vggt = vggt.to(device=args.device, dtype=torch.float32).eval()

    with (args.out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for video in videos:
            scene = video.stem
            text, raw_video = generate_boxes(model, processor, video, args)
            boxes = parse_boxes(text)

            points, colors, cams = vggt_cloud(raw_video, vggt, args.conf_percentile)
            verts, _ = load_mesh(scene)
            world_to_cam0, scale, rmse, score = register_to_mesh(points, cams, verts, args)

            np.savez_compressed(
                args.out / f"{scene}.npz",
                points=points * scale,  # metric camera-0, the frame the boxes are in
                colors=colors, world_to_cam0=world_to_cam0,
                scale=scale, rmse=rmse, score=score,
            )
            fh.write(json.dumps({"scene": scene, "video": str(video), "n_boxes": len(boxes),
                                 "reg_score": float(score), "reg_rmse": float(rmse),
                                 "vggt_scale": scale, "pred": text}) + "\n")
            fh.flush()
            print(f"{scene}: {len(boxes)} boxes | {len(points)} vggt points | "
                  f"fit score={score:.3f} rmse={rmse:.3f}m | "
                  f"vggt->metric scale={scale:.2f}")

    print(f"\nwrote {args.out}/predictions.jsonl and one npz per scene\n"
          f"view: python {Path(__file__).name} --show {videos[0].stem}")


# --------------------------------------------------------------------------- #
# viewer                                                                       #
# --------------------------------------------------------------------------- #
def box_pose(bbox_3d):
    """9-DoF -> (center, wxyz quaternion, extent, rotation). Intrinsic ZXY, as trained."""
    from scipy.spatial.transform import Rotation
    center = np.asarray(bbox_3d[:3], dtype=float)
    extent = np.asarray(bbox_3d[3:6], dtype=float)
    rot = Rotation.from_euler("ZXY", np.asarray(bbox_3d[6:9], dtype=float))
    x, y, z, w = rot.as_quat()
    return center, np.array([w, x, y, z]), extent, rot


def occupancy(points, boxes):
    """Mesh points inside each predicted box. Zero everywhere = a frame or ICP failure."""
    counts = []
    for box in boxes:
        center, _, extent, rot = box_pose(box["bbox_3d"])
        local = (points - center) @ rot.as_matrix()
        counts.append(int(np.all(np.abs(local) <= extent / 2, axis=1).sum()))
    return counts


def show(args) -> None:
    scene = args.show
    npz_path = args.out / f"{scene}.npz"
    if not npz_path.exists():
        raise SystemExit(f"no {npz_path}; run the inference mode first")
    data = np.load(npz_path)
    rec = next(json.loads(ln) for ln in (args.out / "predictions.jsonl").read_text().splitlines()
               if json.loads(ln)["scene"] == scene)
    boxes = parse_boxes(rec["pred"])

    verts, vcolors = load_mesh(scene)
    world_to_cam0 = data["world_to_cam0"]
    mesh_pts = verts @ world_to_cam0[:3, :3].T + world_to_cam0[:3, 3]

    print(f"{scene}: {len(boxes)} predicted boxes | mesh {len(verts)} verts | "
          f"fit score={float(data['score']):.3f} rmse={float(data['rmse']):.3f}m | "
          f"vggt->metric scale={float(data['scale']):.2f}")
    print(rec["pred"])

    counts = occupancy(mesh_pts, boxes)
    for box, n in zip(boxes, counts):
        print(f"  {box['label']:<24} {n:>8} mesh points inside")
    hit = sum(n > 10 for n in counts)
    print(f"{hit}/{len(boxes)} boxes contain mesh geometry")
    if args.check:
        # Gates the ALIGNMENT, not the model: how many boxes land on furniture is the result
        # being looked at, and it varies by scene (measured 1/23 to 34/34 across four scenes).
        assert float(data["score"]) > 0.15, f"registration score {float(data['score']):.3f} too low"
        assert 0.5 < float(data["scale"]) < 20.0, f"implausible scale {float(data['scale']):.2f}"
        print(f"OK: registration score {float(data['score']):.3f}, "
              f"rmse {float(data['rmse']):.3f} m, vggt->metric scale {float(data['scale']):.2f}")
        return

    import viser
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")  # camera-0 frame has +y down; navigation only
    mesh_handle = server.scene.add_point_cloud(
        "/mesh", points=mesh_pts, colors=vcolors, point_size=0.01)
    vggt_handle = server.scene.add_point_cloud(
        "/vggt", points=data["points"], colors=data["colors"], point_size=0.01)
    vggt_handle.visible = False
    server.scene.add_frame("/camera0", axes_length=0.4, axes_radius=0.01)

    handles, labels = [], []
    for i, box in enumerate(boxes):
        center, wxyz, extent, _ = box_pose(box["bbox_3d"])
        handles.append(server.scene.add_box(f"/pred_{i}", color=PRED_COLOR,
                                            dimensions=tuple(extent), wxyz=wxyz,
                                            position=center, wireframe=True))
        labels.append(server.scene.add_label(f"/pred_label_{i}", box["label"], position=center))

    show_mesh = server.gui.add_checkbox("ScanNet++ mesh", True)
    show_vggt = server.gui.add_checkbox("VGGT cloud", False)
    show_boxes = server.gui.add_checkbox(f"Predicted boxes ({len(boxes)})", True)
    show_labels = server.gui.add_checkbox("Labels", False)

    def restyle(_=None):
        mesh_handle.visible = show_mesh.value
        vggt_handle.visible = show_vggt.value
        for handle, label in zip(handles, labels):
            handle.visible = show_boxes.value
            label.visible = show_boxes.value and show_labels.value

    for control in (show_mesh, show_vggt, show_boxes, show_labels):
        control.on_update(restyle)
    restyle()

    print(f"serving on http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show", default=None, help="scene id: open the viewer instead of inferring")
    p.add_argument("--check", action="store_true", help="with --show: headless sanity assert")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--checkpoint", type=Path, default=_CKPT)
    p.add_argument("--base", type=str, default=str(_BASE))
    p.add_argument("--out", type=Path, default=None, help="default: <checkpoint>/detect_vsibench")
    p.add_argument("--cache_root", type=Path, default=_CACHE_ROOT)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--prompt_layout", default="idea_4a", choices=("idea_4a", "idea_3i"),
                   help="idea_4a = the layout the box format was trained under (default); "
                        "idea_3i = this checkpoint's QA layout")
    p.add_argument("--port", type=int, default=8080)
    # Perceiver geometry: must match the trained checkpoint.
    p.add_argument("--vggt_checkpoint", type=str, default=_VGGT_CKPT)
    p.add_argument("--vggt_embed_dim", type=int, default=2048)
    p.add_argument("--frame_num_latents", type=int, default=256)
    p.add_argument("--camera_num_latents", type=int, default=32)
    p.add_argument("--frame_widening_factor", type=int, default=2)
    p.add_argument("--frame_placeholder_token", default="<|quad_start|>")
    p.add_argument("--camera_placeholder_token", default="<|quad_end|>")
    p.add_argument("--image_patch_size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    # Registration.
    p.add_argument("--conf_percentile", type=float, default=50.0,
                   help="drop VGGT points below this depth-confidence percentile")
    p.add_argument("--icp_voxel", type=float, default=0.03)
    p.add_argument("--icp_max_corr", type=float, default=0.3,
                   help="ICP correspondence radius, metres in the mesh frame")
    p.add_argument("--icp_score_radius", type=float, default=0.05,
                   help="tight radius used to SCORE a fit (see register_to_mesh)")
    p.add_argument("--icp_points", type=int, default=30000,
                   help="VGGT points kept for the fit")
    p.add_argument("--scale_steps", type=int, default=9,
                   help="scale candidates swept across the camera-height band")
    p.add_argument("--min_cam_height", type=float, default=0.9,
                   help="lowest plausible handheld camera height above the floor, metres")
    p.add_argument("--max_cam_height", type=float, default=2.0)
    p.add_argument("--yaw_step", type=float, default=30.0)
    args = p.parse_args()
    if args.out is None:
        args.out = args.checkpoint / "detect_vsibench"
    return args


if __name__ == "__main__":
    _args = parse_args()
    show(_args) if _args.show else run(_args)
