"""
data_setup.py (Fusion copy) - where the data lives, the shared split, calib / semantic helpers and a
sanity check. Shared by every script in Fusion/. Put it next to pp_common.py.

Same as Detection_LIDAR/data_setup.py, except:
  - outputs go to Fusion/runs_lidar/pp_painted_kitti_carla and Fusion/runs_fusion
  - the split is NEVER created or changed here: the fusion model must use the very same
    split_lidar.json as the LiDAR-only model, or the fusion-vs-LiDAR comparison is not fair
  - calib / semantic helpers for painting (paint_points.py) and the checks

Layout it expects (what your kitti_carla/training folder has):
    <data_root>/velodyne/<id>.bin      float32 x, y(LEFT), z, intensity   (4 values per point)
    <data_root>/labels_3d/<id>.txt     cls x y z l w h yaw [num_points ...]   (LiDAR frame)
    <data_root>/split_lidar.json       {"train": [ids], "val": [ids]}   (from the collector / Lidar_train.py)
    <data_root>/image_2/<id>.png       RGB camera (input of the segmentation model)
    <data_root>/calib/<id>.txt         KITTI calib: P2, R0_rect, Tr_velo_to_cam (velodyne -> camera)
    <data_root>/semantic/<id>.png      CARLA's semantic camera, class id in the RED channel (checks only)
    <data_root>/velodyne_paint/<id>.npy   written by paint_points.py
"""
import os
import json
import hashlib

import numpy as np

import pp_common as C

HERE = os.path.dirname(os.path.abspath(__file__))
# defaults are anchored to THIS file, so they work from any working directory
# (VS Code runs scripts from the workspace root, not from Fusion)
PYTHON_CODE = os.path.normpath(os.path.join(HERE, '..'))
DEFAULT_DATA_ROOT = os.path.join(PYTHON_CODE, 'LIDAR_only', 'kitti_carla', 'training')
DEFAULT_OUT_DIR = os.path.join(HERE, 'runs_lidar', 'pp_painted_kitti_carla')
DEFAULT_LIDAR_CKPT = os.path.join(PYTHON_CODE, 'Detection_LIDAR', 'runs_lidar', 'pp_kitti_carla', 'best.pt')
DEFAULT_SEG_MODEL = os.path.join(PYTHON_CODE, 'Detection_YOLO', 'runs', 'seg', 'best')
DEFAULT_PAINT_NAME = 'velodyne_paint'
FUSION_RUNS = os.path.join(HERE, 'runs_fusion')

# CARLA 0.9.15 semantic tags (carla.CityObjectLabel), as stored in semantic/*.png
SEM_TAGS = {0: 'unlabeled', 1: 'road', 2: 'sidewalk', 3: 'building', 4: 'wall', 5: 'fence', 6: 'pole',
            7: 'traffic_light', 8: 'traffic_sign', 9: 'vegetation', 10: 'terrain', 11: 'sky',
            12: 'pedestrian', 13: 'rider', 14: 'car', 15: 'truck', 16: 'bus', 17: 'train',
            18: 'motorcycle', 19: 'bicycle', 20: 'static', 21: 'dynamic', 22: 'other', 23: 'water',
            24: 'road_line', 25: 'ground', 26: 'bridge', 27: 'rail_track', 28: 'guard_rail'}
VEHICLE_TAGS = (14, 15, 16, 17, 18, 19)
PERSON_TAGS = (12, 13)
GROUND_TAGS = (1, 2, 10, 24, 25)


# ------------------------------------------------------- pp_common check
REQUIRED = ('CLASS_NAMES', 'NUM_CLASSES', 'LidarDataset', 'load_labels', 'load_points', 'count_pillars',
            'draw_bev', 'collate', 'PointPillars', 'compute_loss', 'build_model_from_ckpt',
            'decode_predictions', 'pair_ious', 'average_precision', 'EVAL_IOU',
            'MAX_PILLARS_TRAIN', 'MAX_PILLARS_TEST', 'paint_info')


def check_pp_common():
    """Stops with a clear message if the imported pp_common.py is not the expected version."""
    missing = [n for n in REQUIRED if not hasattr(C, n)]
    if missing:
        raise SystemExit(f"The pp_common.py that Python imported is:\n    {os.path.abspath(C.__file__)}\n"
                         f"and it lacks: {missing}\n"
                         f"It is an older version (or Detection_LIDAR's copy, which has no painting). "
                         f"Use the Fusion pp_common.py, delete __pycache__ in this folder, and run again.")


check_pp_common()


def frame_ids(root):
    """Frame ids (file stems) in <root>/velodyne, in chronological order."""
    vdir = os.path.join(root, 'velodyne')
    ids = {os.path.splitext(f)[0] for f in os.listdir(vdir) if f.endswith('.bin') or f.endswith('.npy')}
    return sorted(ids, key=lambda s: (len(s), s))


def load_ckpt(path):
    """torch.load that also works on torch >= 2.6 (weights_only now defaults to True, and our
    checkpoints hold optimizer / scheduler state) and on old torch without that argument."""
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


# ------------------------------------------------------------------ paths
def resolve_data_root(path, label_dir='labels_3d'):
    """Accepts .../kitti_carla/training or .../kitti_carla; returns the folder that holds velodyne/."""
    p = os.path.abspath(os.path.expanduser(path))
    root = None
    for cand in (p, os.path.join(p, 'training')):
        if os.path.isdir(os.path.join(cand, 'velodyne')):
            root = cand
            break
    if root is None:
        raise SystemExit(f"no 'velodyne' folder in {p} (or in {os.path.join(p, 'training')}).\n"
                         f"  Pass --data_root <folder that contains velodyne/ and {label_dir}/>")
    if label_dir == 'label_2':
        raise SystemExit("label_2 holds KITTI camera-frame boxes; the trainer needs the LiDAR-frame "
                         "boxes: use --label_dir labels_3d")
    if not os.path.isdir(os.path.join(root, label_dir)):
        raise SystemExit(f"{os.path.join(root, label_dir)} does not exist (--label_dir {label_dir})")
    if not frame_ids(root):
        raise SystemExit(f"{os.path.join(root, 'velodyne')} has no .bin / .npy files")
    return root


def need_dir(root, name, why):
    d = os.path.join(root, name)
    if not os.path.isdir(d):
        raise SystemExit(f"{d} not found ({why})")
    return d


# ------------------------------------------------------------------ split
def split_problems(split, root):
    """Empty list = the split is usable for this data folder."""
    if not (isinstance(split, dict) and isinstance(split.get('train'), list)
            and isinstance(split.get('val'), list)):
        return ["not a JSON object with 'train' and 'val' lists"]
    have = set(frame_ids(root))
    out = []
    for k in ('train', 'val'):
        if not split[k]:
            out.append(f"'{k}' is empty")
        miss = [i for i in split[k] if not (isinstance(i, str) and i in have)]
        if miss:
            out.append(f"{len(miss)} of {len(split[k])} '{k}' ids are not in velodyne/ (e.g. {miss[:3]})")
    if not out and set(split['train']) & set(split['val']):
        out.append('train and val share frames')
    return out


def read_split(root, split_path):
    """Strict: never creates or changes a split. The LiDAR-only model's split is the one to use."""
    if not os.path.exists(split_path):
        raise SystemExit(f'{split_path} not found. The fusion model must use the same split as the '
                         f'LiDAR-only model: run Detection_LIDAR/Lidar_train.py first (it checks / creates '
                         f'split_lidar.json), then train here.')
    try:
        with open(split_path) as f:
            split = json.load(f)
    except ValueError as e:
        raise SystemExit(f'{split_path}: invalid JSON ({e})')
    problems = split_problems(split, root)
    if problems:
        raise SystemExit(f'{split_path} does not fit {root}: ' + '; '.join(problems))
    return split


def split_sha1(split):
    """Fingerprint of the train/val ids, stored in the checkpoint to prove which split it used."""
    s = json.dumps({'train': list(split['train']), 'val': list(split['val'])}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ------------------------------------------------------- calib / semantic
def read_calib(path):
    """KITTI calib file -> {'P2' 3x4, 'R0' 3x3, 'Tr' 3x4 (velodyne -> camera)} (as kitti_utils.read_calib)."""
    vals = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                vals[k.strip()] = np.array([float(x) for x in v.split()])
    return {'P2': vals['P2'].reshape(3, 4), 'R0': vals['R0_rect'].reshape(3, 3),
            'Tr': vals['Tr_velo_to_cam'].reshape(3, 4)}


def frame_calib(root, fid):
    p = os.path.join(root, 'calib', fid + '.txt')
    if not os.path.exists(p):
        raise SystemExit(f'{p} not found: painting needs the calib file of every frame')
    return read_calib(p)


def velo_to_cam(xyz, calib):
    """velodyne points [N,3] (x fwd, y left, z up) -> rectified camera frame (x right, y down, z fwd)."""
    Tr = calib['Tr']
    cam = np.asarray(xyz, np.float64) @ Tr[:, :3].T + Tr[:, 3]
    return cam @ calib['R0'].T


def project(xyz, calib, w, h, near=0.5):
    """velodyne points -> pixel (u, v), depth, and a mask of points that land inside the w x h image
    and at least `near` m in front of the camera (same rule as the old fixed-FOV projection)."""
    cam = velo_to_cam(xyz, calib)
    P = calib['P2']
    z = cam[:, 2]
    ok = z > near
    u = np.full(len(cam), -1.0)
    v = np.full(len(cam), -1.0)
    if ok.any():
        hom = cam[ok] @ P[:, :3].T + P[:, 3]
        u[ok] = hom[:, 0] / hom[:, 2]
        v[ok] = hom[:, 1] / hom[:, 2]
    ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    return u, v, z, ok


def boxes_in_camera(boxes, calib, w, h):
    """Boxes [n,7] whose centre projects into the image (the region where painting exists)."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    return project(np.asarray(boxes)[:, :3], calib, w, h)[3]


def load_semantic(root, fid):
    """CARLA semantic camera image -> uint8 [H, W] class ids (red channel; OpenCV loads BGR) or None."""
    import cv2
    p = os.path.join(root, 'semantic', fid + '.png')
    if not os.path.exists(p):
        return None
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img[:, :, 2] if img.ndim == 3 else img


def image_size(root, fid):
    """(w, h) of image_2/<id>.png."""
    import cv2
    p = os.path.join(root, 'image_2', fid + '.png')
    img = cv2.imread(p)
    if img is None:
        raise SystemExit(f'cannot read {p}')
    return img.shape[1], img.shape[0]


def runtime_camera_mismatch(calib, w, h, fov_deg=90.0, offset=1.5):
    """The Driving stack paints live points with a fixed camera model (perception_scheduler.py:
    fov 90 deg, 1.5 m ahead of the LiDAR, same height, unrotated). Returns the largest pixel
    difference between that model and this calib on a grid of points (0 = identical)."""
    xs, ys, zs = np.meshgrid(np.linspace(3, 50, 6), np.linspace(-20, 20, 5), np.linspace(-2.4, 1.0, 3))
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], 1)
    u1, v1, _, ok1 = project(pts, calib, w, h)
    f = w / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    X, Y, Z = -pts[:, 1], -pts[:, 2], pts[:, 0] - offset
    u2, v2 = f * X / Z + w / 2, f * Y / Z + h / 2
    if not ok1.any():
        return float('inf')
    return float(np.max(np.hypot(u1[ok1] - u2[ok1], v1[ok1] - v2[ok1])))


# ------------------------------------------------------------- box helpers
def points_in_box(pts, box):
    """Mask of points inside one [x, y, z, l, w, h, yaw] box."""
    x, y, z, l, w, h, yaw = box
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2) & (np.abs(pts[:, 2] - z) <= h / 2)


def read_labels_raw(path):
    """labels_3d lines -> list of (collector class name, box [7]) without any filtering."""
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) >= 8:
                    try:
                        out.append((p[0].lower(), np.array([float(v) for v in p[1:8]])))
                    except ValueError:
                        continue
    return out


# ------------------------------------------------------------ sanity report
def _points_in_boxes(pts, boxes):
    """Number of points inside each [x, y, z, l, w, h, yaw] box."""
    return np.array([int(points_in_box(pts, b).sum()) for b in boxes], int)


def dataset_report(root, label_dir, n_probe=20):
    """Prints what the data looks like and whether labels sit on points. Returns False only for a
    hard failure (unreadable files, or labels that fit the y-mirrored points better).
    Same as Detection_LIDAR/data_setup.py."""
    ids = frame_ids(root)
    ldir = os.path.join(root, label_dir)
    n_lab = sum(os.path.exists(os.path.join(ldir, i + '.txt')) for i in ids)
    print(f'[data] root {root}')
    print(f'[data] {len(ids)} velodyne frames, {n_lab} with a {label_dir}/<id>.txt')
    if n_lab < len(ids):
        print(f'[data] WARNING: {len(ids) - n_lab} frames have no label file. They are trained on as '
              f'EMPTY scenes (every object in them becomes a false-positive example).')
    ds = C.LidarDataset(root, ids, label_dir, train=False)
    probe = [ids[k] for k in np.unique(np.linspace(0, len(ids) - 1, min(n_probe, len(ids))).astype(int))]
    n_pts, z_low, z_med, i_min, i_max, r99 = [], [], [], [], [], []
    inside_as, inside_mir, first_line = [], [], None
    for i in probe:
        try:
            pts = C.load_points(ds._pts_path(i))
        except Exception as e:
            print(f'[data] FAIL: cannot read the points of frame {i}: {e}\n'
                  f'       velodyne files must be float32 with 4 values per point (x, y, z, intensity)')
            return False
        if pts.ndim != 2 or pts.shape[1] != 4:
            print(f'[data] FAIL: frame {i} points have shape {pts.shape}, expected [N, 4]')
            return False
        if len(pts):
            n_pts.append(len(pts))
            z_low.append(float(np.percentile(pts[:, 2], 1)))
            z_med.append(float(np.median(pts[:, 2])))
            i_min.append(float(pts[:, 3].min()))
            i_max.append(float(pts[:, 3].max()))
            r99.append(float(np.percentile(np.hypot(pts[:, 0], pts[:, 1]), 99)))
        lp = os.path.join(ldir, i + '.txt')
        if not os.path.exists(lp):
            continue
        if first_line is None:
            with open(lp) as f:
                for line in f:
                    if line.strip():
                        first_line = line.strip()
                        break
        try:
            boxes, _ = C.load_labels(lp)
        except ValueError as e:
            print(f'[data] FAIL: {e}')
            return False
        if len(boxes) and len(pts):
            near = boxes[np.hypot(boxes[:, 0], boxes[:, 1]) < 25.0]
            if len(near):
                mir = near.copy()
                mir[:, 1] *= -1
                mir[:, 6] *= -1
                inside_as.extend(_points_in_boxes(pts, near))
                inside_mir.extend(_points_in_boxes(pts, mir))
    if n_pts:
        print(f'[data] points per frame (probe of {len(probe)}): min {min(n_pts)}  median '
              f'{int(np.median(n_pts))}  max {max(n_pts)}')
        print(f'[data] lowest 1% of points at z = {np.median(z_low):.2f} m (median over frames), median '
              f'point z = {np.median(z_med):.2f} m, 99% of points within {np.median(r99):.0f} m, '
              f'intensity {min(i_min):.2f} .. {max(i_max):.2f}')
    if first_line:
        print(f'[data] first label line ({len(first_line.split())} fields): {first_line}')
    ok = True
    if inside_as:
        a, m = np.array(inside_as), np.array(inside_mir)
        fa, fm = float((a >= 3).mean()), float((m >= 3).mean())
        print(f'[data] boxes within 25 m: {len(a)}. With >= 3 LiDAR points inside: {100 * fa:.0f}% as '
              f'labelled, {100 * fm:.0f}% if mirrored left/right (median points {np.median(a):.0f} vs '
              f'{np.median(m):.0f})')
        if fm > fa + 0.1:
            print('[data] FAIL: the labels fit the MIRRORED points better: y is flipped between '
                  'velodyne/ and labels_3d/')
            ok = False
        elif fa < 0.6:
            print('[data] WARNING: most boxes have almost no points inside. Check Z_IS_BOTTOM / YAW_SIGN / '
                  'YAW_OFFSET in pp_common.py and look at the BEV images from --check')
        else:
            print('[data] OK: labels sit on the points')
    else:
        print('[data] no boxes within 25 m in the probed frames: labels-on-points test not run')
    return ok
