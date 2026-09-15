"""Helpers for make_grid_videos.py: one det_es scene -> one 8x8 grid image of its 64 saved frames with the det.json
camera-0 boxes projected on every frame (same K / pose / draw path as viz_posed64f.py's frame panel)."""
import json
import cv2, numpy as np, open3d as o3d
from viz_det64f import VSI, draw, embodiedscan_bbox_to_o3d_geo

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)   # corners() spams PaintUniformColor warnings
VSI_DIR = {"scannetpp": "scannetppv2"}                                # VSI-590K folder differs from the source name
HEADER = 36
TILE = (192, 128)                                                     # landscape tile (w, h); portrait is (128, 192)
CELL = {"arkitscenes": (192, 192)}                                   # ARKit mixes orientations; others (192, 128)
GREEN, WHITE, RED = (0, 200, 0), (255, 255, 255), (0, 0, 255)         # BGR


def scene_dir(source, scene):
    return f"{VSI}/{VSI_DIR.get(source, source)}/{scene}/det_es"


def load_scene(d):
    """meta, det row, BGR frames (None if missing), cam0-frame boxes (no labels), cam0 -> cam_i poses."""
    meta, row = json.load(open(f"{d}/meta.json")), json.load(open(f"{d}/det.json"))
    fr = meta["frames"]
    frames = [cv2.imread(f"{d}/frames/frame{i:02d}.jpg") for i in range(len(fr))]
    boxes = [dict(label="", center=(g := embodiedscan_bbox_to_o3d_geo(b["bbox_3d"])).center, R=g.R, size=g.extent)
             for b in row["boxes"]]
    cam0 = np.array(fr[0]["cam2global"])
    poses = [np.linalg.inv(np.array(f["cam2global"])) @ cam0 for f in fr]          # cam0 -> cam_i
    return meta, row, frames, boxes, poses


def draw_frame(img, meta, f, boxes, pose):
    """Draw all boxes on the full-res saved frame, meta K scaled native -> saved size. Non-finite pose: no boxes."""
    (H, Wd), (h, w) = meta["native_hw"], img.shape[:2]
    if np.isfinite(pose).all():
        draw(img, boxes, pose, np.diag([w / Wd, h / H, 1]) @ np.array(f["cam2img"]), GREEN)
    return img


def tile(img, cell):
    """Resize to 192x128 / 128x192 and letterbox (black) into the cell."""
    h, w = img.shape[:2]
    tw, th = TILE if w >= h else TILE[::-1]
    out = np.zeros((cell[1], cell[0], 3), np.uint8)
    y, x = (cell[1] - th) // 2, (cell[0] - tw) // 2
    out[y:y + th, x:x + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
    return out


def header(width, parts):
    """Header bar with [(text, colour), ...] written left to right."""
    bar = np.zeros((HEADER, width, 3), np.uint8); x = 8
    for text, col in parts:
        cv2.putText(bar, text, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        x += cv2.getTextSize(text + " ", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
    return bar


def grid_image(job):
    """(source, scene, k, N) -> header + 8x8 grid (BGR uint8), row-major, missing tiles black."""
    source, scene, k, N = job
    cell = CELL.get(source, TILE)
    grid = np.zeros((8 * cell[1], 8 * cell[0], 3), np.uint8)
    meta, row, frames, boxes, poses = load_scene(scene_dir(source, scene))
    for i, (img, f, pose) in enumerate(zip(frames[:64], meta["frames"], poses)):
        if img is not None:
            r, c = divmod(i, 8)
            grid[r * cell[1]:(r + 1) * cell[1], c * cell[0]:(c + 1) * cell[0]] = tile(draw_frame(img, meta, f, boxes, pose), cell)
    n = sum(img is not None for img in frames)
    bar = header(grid.shape[1], [(f"{source} {k}/{N} {scene}", WHITE), (f"n_frames={n}", WHITE if n >= 64 else RED),
                                 (f"boxes={len(boxes)}", WHITE)])
    return np.vstack([bar, grid])
