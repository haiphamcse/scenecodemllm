"""VSI-590K ScanNet++ video -> Boxer ScanNetLoader input (det64f).

  python prep_scene_det64f.py --scenes a b c            # -> OUT/scannet/<scene>/frames/{color,pose,intrinsic,depth} + meta.json
  python prep_scene_det64f.py --all --shard 0/8 --tar   # tar to OUT/tars/<scene>.tar, meta to OUT/meta/<scene>.json, dir deleted
  python prep_scene_det64f.py --scenes bcd2436daf --video .../rgb.mkv --resize 640 480 --out_root ab/data   # A/B on an mkv

Frames: collator rule (n = round(len/fps) capped 64, linspace) on the video; each index snapped to the nearest
COLMAP-registered frame (video frame i == mkv frame i == COLMAP frame_i). Frames further than TOL from a registered
frame are not given to Boxer (the nearest one to frame 0 always is); a video whose frame 0 is unregistered is
flagged in meta (cam0_gap_s) and dropped by build_scannetpp_det64f.py. The frame Boxer sees is the video
frame at the snapped index (<= 5 frames off the sampled one) so image and pose agree; native 640x480, undistorted
with the COLMAP OPENCV model scaled from 1920x1440. Depth = open3d raycast of mesh_aligned_0.05.ply at 320x240."""
import argparse, json, os, shutil, subprocess, sys
import cv2, decord, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                        # sibling prep_scene helpers
from prep_scene import sample_indices, read_colmap, MeshDepth

DATA = "/scratch/ducpham/Working/dataset/scannetpp/data"
VSI = "/scratch/ducpham/Working/dataset/vsi_590k/VSI-590K/scannetppv2"
OUT = "/scratch/ducpham/Working/dataset/vgllm_data/det64f/scannetpp/_boxer_in"
SPLIT = "/scratch/ducpham/Working/dataset/scannetpp/splits/nvs_sem_train.txt"
TOL = 15                                                        # frames (0.25 s at 60 fps)


def prep(scene, video, out_root, resize=None):
    out = f"{out_root}/scannet/{scene}"; fr = f"{out}/frames"
    vr = decord.VideoReader(video, num_threads=4)
    fps = vr.get_avg_fps() or 60.0
    idx = sample_indices(len(vr), fps)
    K, dist, poses, W, H = read_colmap(f"{DATA}/{scene}/iphone/colmap")
    reg = np.array(sorted(poses))
    snapped = reg[np.abs(reg[None, :] - idx[:, None]).argmin(1)]
    gap = np.abs(snapped - idx); keep = gap <= TOL
    meta = {"scene": scene, "video": video, "video_frames": len(vr), "fps": fps, "n_frames": len(idx),
            "sampled_idx": idx.tolist(), "snapped": snapped.tolist(), "gap_frames": gap.tolist(), "kept": keep.tolist(),
            "snap_gap_s": {"median": float(np.median(gap) / fps), "max": float(gap.max() / fps), "mean": float(gap.mean() / fps)},
            "n_registered": len(reg), "cam0_frame": int(snapped[0]), "cam0_gap_s": float(gap[0] / fps), "depth_src": "mesh_raycast"}
    if not keep[0]:                                              # not dropped here: the row builder applies the cam0 tolerance
        print(f"{scene}: frame 0 unregistered, nearest COLMAP frame {snapped[0]} ({gap[0]} frames)")
    frames = sorted(set(snapped[keep].tolist()) | {int(snapped[0])})
    for d in ("color", "pose", "intrinsic", "depth"):
        os.makedirs(f"{fr}/{d}", exist_ok=True)
    imgs = vr.get_batch(frames).asnumpy()
    h, w = imgs.shape[1:3]
    if resize:
        w, h = resize
    Ks = K * np.array([[w / W], [h / H], [1]])                  # COLMAP camera scaled to the video resolution
    newK, _ = cv2.getOptimalNewCameraMatrix(Ks, dist, (w, h), 0)
    render = MeshDepth(f"{DATA}/{scene}/scans/mesh_aligned_0.05.ply", newK, w, h)
    for f, im in zip(frames, imgs):
        if resize:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        im = cv2.undistort(im, Ks, dist, None, newK)
        cv2.imwrite(f"{fr}/color/{f}.jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.savetxt(f"{fr}/pose/{f}.txt", poses[f])
        cv2.imwrite(f"{fr}/depth/{f}.png", render(poses[f]))
    K4 = np.eye(4); K4[:3, :3] = newK
    np.savetxt(f"{fr}/intrinsic/intrinsic_color.txt", K4)
    meta.update(frames=frames, world_offset=poses[frames[0]][:3, 3].tolist(), K=newK.tolist(), W=w, H=h,
                cam2world0=poses[snapped[0]].tolist())
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    print(f"{scene}: {len(vr)} frames @{fps:.0f} -> {len(idx)} sampled, {int(keep.sum())} within {TOL} frames of a COLMAP pose, "
          f"{len(frames)} unique; gap median {np.median(gap):.0f} max {gap.max()} frames; {w}x{h}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=[]); ap.add_argument("--all", action="store_true")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--out_root", default=OUT)
    ap.add_argument("--video", default=None, help="override video path (single scene)")
    ap.add_argument("--resize", nargs=2, type=int, default=None, help="W H: resize decoded frames (A/B on the mkv)")
    ap.add_argument("--tar", action="store_true", help="tar scannet/<scene> to tars/<scene>.tar, keep meta/<scene>.json, delete the dir")
    a = ap.parse_args()
    scenes = a.scenes or ([l.strip() for l in open(SPLIT) if l.strip()] if a.all else [])
    k, n = map(int, a.shard.split("/")); scenes = scenes[k::n]
    os.makedirs(f"{a.out_root}/meta", exist_ok=True); os.makedirs(f"{a.out_root}/tars", exist_ok=True)
    for s in scenes:
        if os.path.exists(f"{a.out_root}/meta/{s}.json"):
            continue
        try:
            meta = prep(s, a.video or f"{VSI}/{s}.mp4", a.out_root, a.resize)
        except Exception as e:                                   # keep the shard going; the meta records the failure
            print(f"{s}: ERROR {e!r}"); meta = {"scene": s, "drop": f"error: {e!r}"}
        if a.tar:
            if "drop" not in meta:
                subprocess.check_call(["tar", "-cf", f"{a.out_root}/tars/{s}.tar", "-C", a.out_root, f"scannet/{s}"])
            shutil.rmtree(f"{a.out_root}/scannet/{s}", ignore_errors=True)
            json.dump(meta, open(f"{a.out_root}/meta/{s}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
