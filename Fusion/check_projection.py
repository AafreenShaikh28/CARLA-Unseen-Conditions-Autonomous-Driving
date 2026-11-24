"""
check_projection.py - verify the LiDAR -> camera projection BEFORE any PointPainting.

Uses each frame's calib/<id>.txt (Tr_velo_to_cam, R0_rect, P2 written by collect_kitti_lidar.py),
exactly what paint_points.py uses. Three checks:
  1. NUMERIC (needs semantic/ and labels_3d/): LiDAR points inside a labelled vehicle / pedestrian box
     must land on vehicle / pedestrian pixels of CARLA's semantic camera, and low points near the car
     on road / sidewalk / ground pixels. The same test with the points mirrored left/right is the
     control: it must score far lower. (The old version compared with depth/*.npy, which the new
     collector does not save.)
  2. RUNTIME CAMERA: the Driving stack paints live points with a fixed camera model (90 deg FOV,
     1.5 m ahead of the LiDAR). The calib must give the same pixels, or the fusion model would see
     different painting in the car than in training.
  3. VISUAL: runs_fusion/check_projection/proj_<id>.png. Left: camera image with points coloured by
     distance and the labels_3d boxes. Right: CARLA's semantic image with the tested points (green =
     the pixel has the expected class, red = it does not).

Run from anywhere (defaults: LIDAR_only/kitti_carla/training):
    python check_projection.py
    python check_projection.py --n 12
    python check_projection.py --ids 000000 001234

to_cam() and project() below are kept unchanged: the Driving stack (perception_scheduler.py) loads
this file by path to paint live points. That load does not put Fusion/ on the import path, so this
module imports data_setup only when run as a script (in main), never at import time.
"""
import os
import argparse

import numpy as np
import cv2

D = None            # data_setup, imported in main(): see the note above


# ---------------------------------------------------------------------------------------------
# Fixed camera model used LIVE by the Driving stack (do not change the signatures)
def load_cloud(root, fid):
    """returns (points [N,4], native y convention). velodyne/*.bin from the collector is y-LEFT;
    a raw CARLA .ply is y-RIGHT."""
    p = os.path.join(root, 'velodyne', fid + '.bin')
    if os.path.exists(p):
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4), 'left'
    for sub in ('velodyne', 'velodyne_ply'):
        p = os.path.join(root, sub, fid + '.ply')
        if os.path.exists(p):
            lines = open(p).read().splitlines()
            i = lines.index('end_header')
            arr = np.array([[float(v) for v in l.split()] for l in lines[i + 1:] if l.strip()], np.float32)
            return arr, 'right'
    raise FileNotFoundError('no velodyne/%s.bin or .ply under %s' % (fid, root))


def to_cam(xyz, ysign, offset):
    """saved LiDAR frame -> standard camera frame (X right, Y down, Z forward); camera `offset` m ahead."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    X = -y if ysign == 'left' else y
    return np.stack([X, -z, x - offset], 1)


def project(cam, f, cx, cy, w, h):
    Z = cam[:, 2]
    ok = Z > 0.5
    u = np.full(len(cam), -1.0)
    v = np.full(len(cam), -1.0)
    u[ok] = f * cam[ok, 0] / Z[ok] + cx
    v[ok] = f * cam[ok, 1] / Z[ok] + cy
    ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    return u, v, Z, ok


# ---------------------------------------------------------------------------------------------
CLS_COL = {'car': (255, 128, 0), 'van': (255, 128, 0), 'truck': (0, 128, 255), 'bus': (0, 128, 255),
           'bicycle': (0, 220, 0), 'motorcycle': (0, 220, 0), 'person': (0, 0, 255)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
FRONT = [(0, 1), (1, 5), (5, 4), (4, 0)]          # +x (heading) face is drawn thicker
VEHICLES = ('car', 'van', 'truck', 'bus')
RIDDEN = ('bicycle', 'motorcycle')                # box holds the bike AND its rider


def box_corners(x, y, z, dx, dy, dz, yaw):
    sx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * dx / 2
    sy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * dy / 2
    sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * dz / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + sx * c - sy * s, y + sx * s + sy * c, z + sz], 1)


def expected_tags(name):
    if name in VEHICLES:
        return D.VEHICLE_TAGS
    if name == 'person':
        return D.PERSON_TAGS
    if name in RIDDEN:
        return D.VEHICLE_TAGS + D.PERSON_TAGS
    return None


def test_points(pts, labels):
    """-> (mask of tested points, list of allowed tag tuples per tested point group, kinds)
    Object points: inside a labelled box. Ground points: low, 4-25 m away, outside every box."""
    groups = []                                    # (mask, allowed tags, kind)
    in_any = np.zeros(len(pts), bool)
    for name, b in labels:
        m = D.points_in_box(pts, b)
        in_any |= m
        tags = expected_tags(name)
        if tags is not None and m.sum() >= 3:
            groups.append((m, tags, 'object'))
    z_road = np.percentile(pts[:, 2], 2) if len(pts) else 0.0
    r = np.hypot(pts[:, 0], pts[:, 1])
    ground = (pts[:, 2] < z_road + 0.15) & (r > 4) & (r < 25) & ~in_any
    groups.append((ground, D.GROUND_TAGS, 'ground'))
    return groups


def score(pts, labels, calib, sem, w, h):
    """fraction of tested points whose pixel in the semantic image has the expected class."""
    u, v, _, ok = D.project(pts[:, :3], calib, w, h)
    res = {'object': [0, 0], 'ground': [0, 0]}
    hits = np.zeros(len(pts), np.int8)            # 0 untested, 1 agree, -1 disagree (for drawing)
    tag = np.full(len(pts), -1)
    tag[ok] = sem[v[ok].astype(int), u[ok].astype(int)]
    for m, tags, kind in test_points(pts, labels):
        m = m & ok
        if not m.any():
            continue
        good = np.isin(tag[m], tags)
        res[kind][0] += int(good.sum())
        res[kind][1] += int(m.sum())
        hits[np.flatnonzero(m)[good]] = 1
        hits[np.flatnonzero(m)[~good]] = -1
    return res, u, v, ok, hits


def sem_vis(sem):
    """muted colours (BGR) so the bright green / red test markers stand out"""
    rng = np.random.default_rng(3)
    pal = rng.integers(60, 160, (256, 3)).astype(np.float32)
    for t, c in ((1, (110, 80, 110)), (2, (150, 110, 150)), (24, (120, 150, 120)), (11, (150, 120, 90)),
                 (12, (200, 160, 60)), (13, (200, 160, 60)), (14, (130, 60, 40)), (15, (130, 60, 40)),
                 (16, (130, 60, 40)), (18, (160, 100, 160)), (19, (160, 100, 160))):
        pal[t] = c
    return (pal * 0.7 + 30).astype(np.uint8)[sem]


def main():
    global D
    import data_setup
    D = data_setup
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=D.DEFAULT_DATA_ROOT)
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=8, help='evenly spaced frames to check')
    ap.add_argument('--no_boxes', action='store_true')
    ap.add_argument('--out_dir', default=os.path.join(D.FUSION_RUNS, 'check_projection'))
    args = ap.parse_args()
    root = D.resolve_data_root(args.data_root, args.label_dir)
    D.need_dir(root, 'calib', 'written by collect_kitti_lidar.py')
    D.need_dir(root, 'image_2', 'written by collect_kitti_lidar.py')
    os.makedirs(args.out_dir, exist_ok=True)

    all_ids = D.frame_ids(root)
    ids = args.ids or [all_ids[k] for k in np.unique(np.linspace(0, len(all_ids) - 1, args.n).astype(int))]
    print(f'[proj] {root}\n[proj] checking {len(ids)} frames')

    tot = {'as': {'object': [0, 0], 'ground': [0, 0]}, 'mirror': {'object': [0, 0], 'ground': [0, 0]}}
    cam_err, calib_ref, calib_same = [], None, True
    print('\nframe      object points on their class           low ground points on')
    print('           as saved        mirrored (control)     road / sidewalk / ground')
    for fid in ids:
        pts, _ = load_cloud(root, fid)
        calib = D.frame_calib(root, fid)
        if calib_ref is None:
            calib_ref = calib
        elif any(not np.allclose(calib[k], calib_ref[k]) for k in calib):
            calib_same = False
        img = cv2.imread(os.path.join(root, 'image_2', fid + '.png'))
        if img is None:
            raise SystemExit(f'cannot read image_2/{fid}.png')
        h, w = img.shape[:2]
        cam_err.append(D.runtime_camera_mismatch(calib, w, h))
        labels = D.read_labels_raw(os.path.join(root, args.label_dir, fid + '.txt'))
        sem = D.load_semantic(root, fid)

        hits = None
        if sem is not None:
            r_as, u, v, ok, hits = score(pts, labels, calib, sem, w, h)
            mp = pts.copy()
            mp[:, 1] *= -1
            ml = [(n, b * np.array([1, -1, 1, 1, 1, 1, -1])) for n, b in labels]
            r_m = score(mp, ml, calib, sem, w, h)[0]
            for k in ('object', 'ground'):
                for which, r in (('as', r_as), ('mirror', r_m)):
                    tot[which][k][0] += r[k][0]
                    tot[which][k][1] += r[k][1]
            pct = lambda r: f"{100 * r[0] / r[1]:5.1f}% of {r[1]:<6d}" if r[1] else '   -          '
            print(f"{fid}   {pct(r_as['object'])}   {pct(r_m['object'])}   {pct(r_as['ground'])}")
        else:
            u, v, _, ok = D.project(pts[:, :3], calib, w, h)
            print(f'{fid}   no semantic/{fid}.png: numeric check skipped')

        # ---- overlay ----
        cmap = getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET)
        dist = np.hypot(pts[:, 0], pts[:, 1])
        order = np.flatnonzero(ok)[np.argsort(-dist[ok])]
        cols = cv2.applyColorMap(np.clip(dist[order] / 60.0 * 255, 0, 255).astype(np.uint8).reshape(-1, 1),
                                 cmap).reshape(-1, 3)
        out = img.copy()
        for k, c_ in zip(order, cols):
            cv2.circle(out, (int(u[k]), int(v[k])), 2, tuple(int(t) for t in c_), -1)
        if not args.no_boxes:
            for name, b in labels:
                cc = D.velo_to_cam(box_corners(*b), calib)
                if (cc[:, 2] < 0.5).any():
                    continue
                P = calib['P2']
                hom = cc @ P[:, :3].T + P[:, 3]
                px = hom[:, :2] / hom[:, 2:3]
                if (np.abs(px) > 4000).any():
                    continue
                col = CLS_COL.get(name, (255, 255, 255))
                for i, j in EDGES:
                    th = 3 if (i, j) in FRONT or (j, i) in FRONT else 1
                    cv2.line(out, tuple(int(t) for t in px[i]), tuple(int(t) for t in px[j]), col, th)
                cv2.putText(out, name, (int(px[:, 0].min()), max(int(px[:, 1].min()) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(out, f'{fid}  points in view: {int(ok.sum())}', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels = [out]
        if sem is not None:
            sv = sem_vis(sem)
            for k in np.flatnonzero(hits != 0):
                cv2.circle(sv, (int(u[k]), int(v[k])), 2, (0, 255, 0) if hits[k] > 0 else (0, 0, 255), -1)
            cv2.putText(sv, 'semantic GT: green = point on expected class, red = not', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(sv)
        cv2.imwrite(os.path.join(args.out_dir, f'proj_{fid}.png'), np.hstack(panels))

    # ---- verdict ----
    print()
    ok_all = True
    o_as, o_m = tot['as']['object'], tot['mirror']['object']
    if o_as[1]:
        a, m = o_as[0] / o_as[1], (o_m[0] / o_m[1] if o_m[1] else 0.0)
        g = tot['as']['ground']
        gs = f", ground points on ground pixels {100 * g[0] / g[1]:.1f}%" if g[1] else ''
        print(f'[proj] object points on the right class: {100 * a:.1f}% as saved vs {100 * m:.1f}% mirrored{gs}')
        if a >= 0.6 and a - m >= 0.2:
            print('[proj] OK: the calib projection puts LiDAR points on the right pixels')
        else:
            ok_all = False
            print('[proj] FAIL: object points do not land on their class (or the mirrored control scores as '
                  'well). Do not paint: check calib/ and velodyne/ come from the same collector run.')
    else:
        print('[proj] numeric check not run (no semantic/ images or no labelled objects in view)')
    worst = max(cam_err) if cam_err else float('nan')
    print(f'[proj] Driving-stack camera model (90 deg, 1.5 m ahead) vs calib: max difference {worst:.3f} px')
    if not worst < 0.5:
        ok_all = False
        print('[proj] WARNING: the live painting in perception_scheduler.py (CAM_OFFSET, the fov in the data '
              'collection CFG) does not match this calib. A fusion model trained on this data would get '
              'differently painted points in the car. Make the Driving camera match the collector camera.')
    if not calib_same:
        print('[proj] note: calib differs between frames (fine for painting, which reads each frame\'s file)')
    print(f'[proj] overlays saved to {args.out_dir}')
    print('       Left: points coloured by distance (blue near, red far) must sit on the objects; boxes:')
    print('       orange car/van, blue truck/bus, green bike, red person; thick face = front.')
    return ok_all


if __name__ == '__main__':
    main()
