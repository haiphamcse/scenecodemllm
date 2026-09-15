"""One mp4 per dataset for eyeballing det_es rows: each video frame = one scene as an 8x8 grid of its 64 saved frames
with all det.json boxes projected (green), header `<source> <k>/<N> <scene> n_frames=<n> boxes=<b>` (n_frames red if < 64).

  python make_grid_videos.py --source scannet|arkitscenes|scannetpp|all [--fps 2] [--limit N] [--workers 16] [--scenes id ...]
-> videos/<source>.mp4 (overwritten). H.264 via ffmpeg (PATH, else the python env's bin/), else cv2 mp4v."""
import argparse, os, shutil, subprocess, sys, time
from multiprocessing import get_context
import cv2
from grid_video_utils import VSI, VSI_DIR, grid_image

HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = shutil.which("ffmpeg") or shutil.which("ffmpeg", path=os.path.dirname(sys.executable))   # env bin/ is off PATH unless activated


def scenes_of(source):
    d = f"{VSI}/{VSI_DIR.get(source, source)}"
    return sorted(s for s in os.listdir(d) if os.path.isdir(f"{d}/{s}/det_es"))


def make(source, a):
    scenes = a.scenes or scenes_of(source)[:a.limit]
    N, out = len(scenes), f"{HERE}/videos/{source}.mp4"
    os.makedirs(f"{HERE}/videos", exist_ok=True)
    t0, writer = time.time(), None
    with get_context("spawn").Pool(a.workers) as pool:          # spawn: fork after `import open3d` can deadlock
        for k, img in enumerate(pool.imap(grid_image, [(source, s, i + 1, N) for i, s in enumerate(scenes)], chunksize=2)):
            if writer is None:
                h, w = img.shape[:2]
                if FFMPEG:
                    writer = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                                               "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", out],
                                              stdin=subprocess.PIPE)
                else:
                    print("no ffmpeg: falling back to cv2.VideoWriter mp4v (not H.264)", flush=True)
                    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
            if FFMPEG:
                writer.stdin.write(img.tobytes())
            else:
                writer.write(img)
            if (k + 1) % 200 == 0:
                print(f"{source}: {k + 1}/{N} scenes, {time.time() - t0:.0f} s", flush=True)
    if FFMPEG:
        writer.stdin.close(); assert writer.wait() == 0, "ffmpeg failed"
    else:
        writer.release()
    print(f"{source}: {N} scenes -> {out} ({w}x{h}, {os.path.getsize(out) / 1e6:.1f} MB) in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--source", required=True, choices=["scannet", "arkitscenes", "scannetpp", "all"])
    ap.add_argument("--fps", type=float, default=2); ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scenes", nargs="*", help="explicit scene ids instead of all det_es scenes (single source)")
    a = ap.parse_args()
    for s in (["scannet", "arkitscenes", "scannetpp"] if a.source == "all" else [a.source]):
        make(s, a)


if __name__ == "__main__":
    main()
