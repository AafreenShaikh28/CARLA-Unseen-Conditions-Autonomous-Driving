"""
check_paint.py - objective check of the painted channels (run after paint_points.py).

1. Inside GT boxes: for every labels_3d box, the painted points inside it (camera view only):
   mean probability of 'vehicle' and 'pedestrian' per object class, versus background points.
2. Against CARLA's semantic camera (semantic/<id>.png): for every painted point in view, is the
   predicted class (argmax) the class CARLA rendered at that pixel? Per class: share of points of
   that GT class painted correctly, and the most common wrong class. This is the painting
   accuracy the fusion model will see (SegFormer errors + projection errors).

Run from anywhere (defaults: LIDAR_only/kitti_carla/training, velodyne_paint):
    python check_paint.py --n 50                 # 50 evenly spaced painted frames
    python check_paint.py --ids 000000 001234
"""
import os
import json
import argparse
import collections

import numpy as np

import data_setup as D

# segmentation class name -> CARLA 0.9.15 semantic tags it covers (names are normalised:
# lower case, spaces / hyphens -> '_'). Classes not listed here are reported as "not checked".
NAME_TO_TAGS = {
    'road': (1,), 'sidewalk': (2,), 'building': (3,), 'wall': (4,), 'fence': (5,), 'pole': (6,),
    'traffic_light': (7,), 'traffic_sign': (8,), 'vegetation': (9,), 'terrain': (10,), 'sky': (11,),
    'pedestrian': (12, 13), 'person': (12, 13), 'rider': (13,),
    'vehicle': (14, 15, 16, 17, 18, 19), 'car': (14,), 'truck': (15,), 'bus': (16,), 'train': (17,),
    'motorcycle': (18,), 'bicycle': (19,), 'static': (20,), 'dynamic': (21,), 'other': (22,),
    'water': (23,), 'road_line': (24,), 'roadline': (24,), 'lane_marking': (24,), 'ground': (25,),
    'bridge': (26,), 'rail_track': (27,), 'guard_rail': (28,), 'guardrail': (28,),
}


def norm(name):
    return name.lower().replace(' ', '_').replace('-', '_')


def tag_to_class(names):
    """CARLA tag -> index of the segmentation class covering it. The most specific class wins
    (e.g. 'rider' takes tag 13 from 'pedestrian' when the model has both)."""
    t2c = {}
    for k in sorted(range(len(names)), key=lambda k: len(NAME_TO_TAGS.get(norm(names[k]), ())) or 99):
        for t in NAME_TO_TAGS.get(norm(names[k]), ()):
            t2c.setdefault(t, k)
    return t2c


def find(names, *cands):
    for c in cands:
        for k, n in enumerate(names):
            if norm(n) == c:
                return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=D.DEFAULT_DATA_ROOT)
    ap.add_argument('--paint_name', default=D.DEFAULT_PAINT_NAME)
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--n', type=int, default=50, help='evenly spaced painted frames (0 = all)')
    args = ap.parse_args()

    root = D.resolve_data_root(args.data_root, args.label_dir)
    paint_dir = os.path.join(root, args.paint_name)
    mp = os.path.join(paint_dir, 'paint_meta.json')
    if not os.path.exists(mp):
        raise SystemExit(f'{mp} not found: run paint_points.py first')
    with open(mp) as f:
        names = json.load(f)['classes']
    iv, ip = find(names, 'vehicle', 'car'), find(names, 'pedestrian', 'person')
    t2c = tag_to_class(names)
    unchecked = [n for n in names if norm(n) not in NAME_TO_TAGS]

    if args.ids:
        ids = args.ids
    else:
        ids = sorted((f[:-4] for f in os.listdir(paint_dir) if f.endswith('.npy')), key=lambda s: (len(s), s))
        if args.n and len(ids) > args.n:
            ids = [ids[k] for k in np.linspace(0, len(ids) - 1, args.n).astype(int)]
    if not ids:
        raise SystemExit(f'no painted frames in {paint_dir}')

    acc = collections.defaultdict(lambda: dict(boxes=0, pts=0, pv=0.0, pp=0.0, av=0, ap=0))
    bg = dict(pts=0, pv=0.0, pp=0.0, av=0, ap=0)
    conf = np.zeros((len(names), len(names)), np.int64)      # rows: GT class, cols: painted argmax
    n_sem = 0
    for fid in ids:
        pts = np.fromfile(os.path.join(root, 'velodyne', fid + '.bin'), dtype=np.float32).reshape(-1, 4)[:, :3]
        ex = np.load(os.path.join(paint_dir, fid + '.npy'))
        if len(ex) != len(pts):
            print(f'  {fid}: painted file has {len(ex)} points, velodyne {len(pts)}: stale, skipped (repaint)')
            continue
        incam = ex[:, -1] > 0
        probs = ex[:, :-1].astype(np.float32) / 255.0
        am = probs.argmax(1)

        # ---- 1. inside GT boxes ----
        in_any = np.zeros(len(pts), bool)
        for name, box in D.read_labels_raw(os.path.join(root, args.label_dir, fid + '.txt')):
            inside = D.points_in_box(pts, box) & incam
            in_any |= inside
            k = int(inside.sum())
            if k == 0:
                continue
            a = acc[name]
            a['boxes'] += 1
            a['pts'] += k
            if iv is not None:
                a['pv'] += float(probs[inside, iv].sum()); a['av'] += int((am[inside] == iv).sum())
            if ip is not None:
                a['pp'] += float(probs[inside, ip].sum()); a['ap'] += int((am[inside] == ip).sum())
        b = incam & ~in_any
        bg['pts'] += int(b.sum())
        if iv is not None:
            bg['pv'] += float(probs[b, iv].sum()); bg['av'] += int((am[b] == iv).sum())
        if ip is not None:
            bg['pp'] += float(probs[b, ip].sum()); bg['ap'] += int((am[b] == ip).sum())

        # ---- 2. against the semantic camera ----
        sem = D.load_semantic(root, fid)
        if sem is None:
            continue
        n_sem += 1
        h, w = sem.shape
        u, v, _, ok = D.project(pts, D.frame_calib(root, fid), w, h)
        ok &= incam
        tags = sem[v[ok].astype(int), u[ok].astype(int)]
        gt = np.array([t2c.get(int(t), -1) for t in tags])
        m = gt >= 0
        np.add.at(conf, (gt[m], am[ok][m]), 1)

    print(f'\n{len(ids)} frames. 1) Painted points INSIDE GT boxes (camera view only):\n')
    print('%-12s %6s %7s %10s %10s %11s %11s' % ('object', 'boxes', 'points', 'P(vehicle)', 'P(ped)',
                                                  'argmax=veh', 'argmax=ped'))
    for k in sorted(acc, key=lambda n: -acc[n]['pts']):
        a = acc[k]
        n = max(a['pts'], 1)
        print('%-12s %6d %7d %10.2f %10.2f %10.0f%% %10.0f%%' %
              (k, a['boxes'], a['pts'], a['pv'] / n, a['pp'] / n, 100 * a['av'] / n, 100 * a['ap'] / n))
    n = max(bg['pts'], 1)
    print('%-12s %6s %7d %10.2f %10.2f %10.0f%% %10.0f%%' %
          ('BACKGROUND', '-', bg['pts'], bg['pv'] / n, bg['pp'] / n, 100 * bg['av'] / n, 100 * bg['ap'] / n))
    print('Expect: car/van/truck/bus -> high P(vehicle); bicycle/motorcycle -> a mix of vehicle and '
          'pedestrian (the rider is a pedestrian pixel); BACKGROUND -> both low.')

    print(f'\n2) Painting vs CARLA\'s semantic camera ({n_sem} frames with semantic/):')
    if conf.sum() == 0:
        print('   not run (no semantic/ images, or no class names that map to CARLA tags)')
    else:
        print('%-16s %9s %10s   %s' % ('GT class', 'points', 'correct', 'most common wrong class'))
        for k in np.argsort(-conf.sum(1)):
            tot = int(conf[k].sum())
            if tot == 0:
                continue
            wrong = conf[k].copy()
            wrong[k] = 0
            j = int(wrong.argmax())
            ws = f'{names[j]} ({100 * wrong[j] / tot:.0f}%)' if wrong[j] else '-'
            print('%-16s %9d %9.1f%%   %s' % (names[k], tot, 100 * conf[k, k] / tot, ws))
        print('%-16s %9d %9.1f%%' % ('ALL', int(conf.sum()), 100 * np.trace(conf) / conf.sum()))
        if unchecked:
            print(f'   not checked (no CARLA tag mapping for these names): {unchecked}')
        print('   Vehicle and pedestrian rows matter most for the detector. Low numbers there mean SegFormer is weak on '
              'this data (or check_projection.py fails); the fusion model can then only match LiDAR-only, not beat it.')


if __name__ == '__main__':
    main()
