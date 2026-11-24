"""
pose_eval.py - does the detector find people LYING, SITTING and CROUCHING? Per pose and distance, on the
validation pose frames (Town05, collect_pose_data.py), for any PointPillars checkpoint: the old 5-class
models (before) and the fine-tuned ones (after). This is the before / after table for the report.

Data with a hazards\\ folder (collect_kitti_lidar.py with posed pedestrians: every walker is 'person' in
labels_3d, the pose of the posed ones is in hazards\\<id>.txt by actor id): all frames of the split are used.

Put the SAME file in Detection_LIDAR\\ and in Fusion\\ (it uses the pp_common.py next to it; for a 5-class
checkpoint it uses pp_common_5class.py, the copy add_pose_class.py keeps).
    python pose_eval.py --ckpt runs_lidar\\pp_kitti_carla\\best.pt                 # before
    python pose_eval.py --ckpt runs_lidar\\pp_kitti_carla_pose\\best.pt            # after
    add --fov_only for the fusion vs LiDAR-only comparison (camera view, where painting exists)

found        a box of ANY class with score >= --score (0.3 = the driving stack's decision score) on the
             person: its centre within the person's box + 0.5 m, or the person's centre within its box
             + 0.3 m. Any class counts, because the driving stack stops for anything in its path.
right class  that box is LyingPerson for a lying person, Pedestrian for the others (lying too, when the
             pp_common next to it maps person_lying to Pedestrian: add_pose_class.py --lying-as-pedestrian)
people       ground-truth people with >= 5 LiDAR hits (fewer: not counted, as in training)
false alarms LyingPerson / Pedestrian boxes on nothing (no labelled object within 1 m) per 100 frames of
             the ordinary validation frames (no pose data in them)
"""
import os
import json
import math
import argparse
import importlib.util
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_NAMES = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
POSE = {'person': 'standing', 'pedestrian': 'standing', 'walker': 'standing', 'person_sitting': 'sitting',
        'person_crouching': 'crouching', 'person_lying': 'lying'}
POSES = ('lying', 'sitting', 'crouching', 'standing')
BINS = ((0.0, 15.0), (15.0, 30.0), (30.0, 50.0))
VEHICLES = ('Car', 'Truck', 'Bus', 'Cyclist')


def load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def pp_for(ckpt):
    """the pp_common module whose classes match the checkpoint"""
    names = list(ckpt.get('class_names') or OLD_NAMES)
    tried = []
    for fname in ('pp_common.py', 'pp_common_5class.py'):
        path = os.path.join(HERE, fname)
        if not os.path.isfile(path):
            continue
        spec = importlib.util.spec_from_file_location('ppe_' + fname[:-3], path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        if list(m.CLASS_NAMES) == names:
            return m
        tried.append(f'{fname}: {m.CLASS_NAMES}')
    raise SystemExit(f'no pp_common in {HERE} has the checkpoint classes {names} ({"; ".join(tried)})')


def read_gt(path, m):
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path):
        f = line.split()
        if len(f) < 8:
            continue
        try:
            x, y, z, l, w, h, ry = map(float, f[1:8])
        except ValueError:
            continue
        if m.Z_IS_BOTTOM:
            z += h / 2
        n_sem = int(f[12]) if len(f) >= 13 and int(f[12]) >= 0 else (int(float(f[8])) if len(f) > 8 else 99)
        in_img = int(f[10]) if len(f) >= 11 else 1
        out.append(dict(tok=f[0].lower(), box=np.array([x, y, z, l, w, h, m.YAW_SIGN * ry + m.YAW_OFFSET]),
                        n=n_sem, img=in_img, aid=f[9] if len(f) >= 10 else None))
    return out


def read_hazards(path):
    """collect_kitti_lidar.py (posed version) hazards/<id>.txt: actor_id pose place n_body n_ghost labelled
    x y z l w h yaw -> {actor_id: pose}. Its labels_3d calls every walker 'person'; the pose is only here."""
    out = {}
    if os.path.isfile(path):
        for line in open(path):
            f = line.split()
            if len(f) >= 2 and f[1] in ('lying', 'sitting', 'crouching'):
                out[f[0]] = f[1]
    return out


def inside(px, py, box, margin):
    x, y, _, l, w, _, yaw = box
    dx, dy = px - x, py - y
    c, s = math.cos(yaw), math.sin(yaw)
    return abs(dx * c + dy * s) <= l / 2 + margin and abs(-dx * s + dy * c) <= w / 2 + margin


def run_model(m, model, ds, dev, batch, score):
    out = {}
    dl = DataLoader(ds, batch, shuffle=False, num_workers=0, collate_fn=m.collate)
    with torch.no_grad():
        for b in dl:
            preds = m.decode_predictions(model(b['points']), model, score_thr=score)
            for i, p in zip(b['id'], preds):
                out[i] = p
    return out


def main():
    ap = argparse.ArgumentParser(description='found / not found per pose and distance')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', default=os.path.join(os.path.dirname(HERE), 'LIDAR_only', 'kitti_carla', 'training'))
    ap.add_argument('--split_file', default='')
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--first_id', type=int, default=500000, help='pose frames have ids >= this')
    ap.add_argument('--score', type=float, default=0.3, help='the driving stack acts on boxes >= 0.3 (--decision-score)')
    ap.add_argument('--fov_only', action='store_true', help='only people visible in the camera image')
    ap.add_argument('--fp_frames', type=int, default=300, help='ordinary val frames for the false-alarm count (0 = skip)')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--out', default='', help='JSON summary (default: next to the checkpoint)')
    args = ap.parse_args()

    ckpt = load_ckpt(args.ckpt)
    m = pp_for(ckpt)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = m.build_model_from_ckpt(ckpt, dev)
    names = list(m.CLASS_NAMES)
    lying_as_ped = 'Pedestrian' in names and m.CLASS_ALIASES.get('person_lying') == names.index('Pedestrian')
    root = args.data_root if os.path.isdir(os.path.join(args.data_root, 'velodyne')) else os.path.join(args.data_root, 'training')
    with open(args.split_file or os.path.join(root, 'split_lidar.json')) as f:
        ids = json.load(f)[args.split]
    hz_dir = os.path.join(root, 'hazards')
    if os.path.isdir(hz_dir):                  # posed pedestrians recorded in every frame (collect_kitti_lidar.py)
        pose_ids = [i for i in ids if os.path.isfile(os.path.join(hz_dir, i + '.txt'))]
        other = []
        print(f'[eval] poses from {hz_dir} ({len(pose_ids)} of {len(ids)} {args.split} frames have a hazards file)')
    else:
        pose_ids = [i for i in ids if i.isdigit() and int(i) >= args.first_id]
        other = [i for i in ids if i not in set(pose_ids)]
    if not pose_ids:
        raise SystemExit(f'no pose frames (ids >= {args.first_id}, or a hazards/ folder) in the {args.split} split')
    n_feats = int(ckpt.get('num_pt_feats', 4))
    paint_dir = None
    if n_feats > 4:
        paint_dir = os.path.join(root, ckpt.get('paint_name') or 'velodyne_paint')
        miss = [i for i in pose_ids if not os.path.isfile(os.path.join(paint_dir, i + '.npy'))]
        if miss:
            raise SystemExit(f'{len(miss)} pose frames not painted (e.g. {miss[:3]}): run Fusion\\paint_points.py')
    if args.fp_frames and other:
        other = [other[k] for k in np.unique(np.linspace(0, len(other) - 1, min(args.fp_frames, len(other))).astype(int))]
        if paint_dir:
            other = [i for i in other if os.path.isfile(os.path.join(paint_dir, i + '.npy'))]
    else:
        other = []

    def ds(idl):
        return (m.LidarDataset(root, idl, 'labels_3d', train=False, paint_dir=paint_dir) if paint_dir
                else m.LidarDataset(root, idl, 'labels_3d', train=False))

    print(f'[eval] {args.ckpt}\n       classes {names}, {"painted" if paint_dir else "LiDAR only"}, {len(pose_ids)} pose '
          f'frames + {len(other)} ordinary frames ({args.split}), score >= {args.score}, device {dev}'
          + (', camera view only' if args.fov_only else ''))
    preds = run_model(m, model, ds(pose_ids + other), dev, args.batch, min(args.score, 0.1))

    rows = defaultdict(lambda: dict(n=0, found=0, right=0, cls=Counter(), scores=[]))
    hidden = Counter()
    for i in pose_ids:
        gt = read_gt(os.path.join(root, 'labels_3d', i + '.txt'), m)
        p = preds[i]
        keep = p['scores'] >= args.score
        pb, ps, pc = p['boxes'][keep], p['scores'][keep], p['cls'][keep]
        hz = read_hazards(os.path.join(hz_dir, i + '.txt'))
        for g in gt:
            pose = POSE.get(g['tok'])
            if pose is None:
                continue
            if pose == 'standing' and g['aid'] in hz:
                pose = hz[g['aid']]
            if g['n'] < 5 or (args.fov_only and g['img'] != 1):
                hidden[pose] += 1
                continue
            dist = math.hypot(g['box'][0], g['box'][1])
            b = next((k for k, (lo, hi) in enumerate(BINS) if lo <= dist < hi), None)
            if b is None:
                continue
            r = rows[(pose, b)]
            r['n'] += 1
            hit = [k for k in range(len(pb)) if inside(pb[k][0], pb[k][1], g['box'], 0.5)
                   or inside(g['box'][0], g['box'][1], pb[k], 0.3)]
            if hit:
                k = max(hit, key=lambda q: ps[q])
                name = names[int(pc[k])]
                r['found'] += 1
                r['cls'][name] += 1
                r['scores'].append(float(ps[k]))
                want = 'LyingPerson' if pose == 'lying' and 'LyingPerson' in names else 'Pedestrian'
                r['right'] += any(names[int(pc[q])] == want for q in hit)

    print(f"\n{'pose':<10}{'range m':>9}{'people':>8}{'found':>8}{'right class':>13}{'as Ped':>8}{'as Lying':>10}"
          f"{'as vehicle':>12}{'score':>7}")
    summary = {'checkpoint': os.path.abspath(args.ckpt), 'classes': names, 'score': args.score,
               'fov_only': args.fov_only, 'split': args.split, 'rows': [], 'not_counted_few_hits': dict(hidden)}
    for pose in POSES:
        tot = dict(n=0, found=0, right=0)
        for b, (lo, hi) in enumerate(BINS):
            r = rows.get((pose, b))
            if not r or not r['n']:
                continue
            n = r['n']
            veh = sum(v for k, v in r['cls'].items() if k in VEHICLES)
            right = r['right'] if (pose != 'lying' or 'LyingPerson' in names or lying_as_ped) else None
            print(f"{pose:<10}{f'{lo:.0f}-{hi:.0f}':>9}{n:>8}{100 * r['found'] / n:>7.0f}%"
                  f"{('n/a' if right is None else f'{100 * right / n:.0f}%'):>13}"
                  f"{100 * r['cls']['Pedestrian'] / n:>7.0f}%{100 * r['cls']['LyingPerson'] / n:>9.0f}%"
                  f"{100 * veh / n:>11.0f}%{(np.median(r['scores']) if r['scores'] else 0):>7.2f}")
            summary['rows'].append(dict(pose=pose, range=[lo, hi], people=n, found=r['found'], right_class=right,
                                        classes=dict(r['cls']), median_score=float(np.median(r['scores'])) if r['scores'] else None))
            for k in tot:
                tot[k] += r[k] if k != 'right' or right is not None else 0
        if tot['n']:
            rc = 'n/a' if pose == 'lying' and 'LyingPerson' not in names and not lying_as_ped else '%.0f%%' % (100.0 * tot['right'] / tot['n'])
            print(f"{pose:<10}{'all':>9}{tot['n']:>8}{100 * tot['found'] / tot['n']:>7.0f}%{rc:>13}")
            summary.setdefault('total', {})[pose] = tot

    if not other and os.path.isdir(hz_dir) and args.fp_frames:   # no frames without people: count on the same ones
        other = pose_ids[:args.fp_frames]
        fp_what = 'frames'
    else:
        fp_what = 'ordinary frames'
    if other:
        fp = Counter()
        for i in other:
            gt = read_gt(os.path.join(root, 'labels_3d', i + '.txt'), m)
            p = preds[i]
            for k in np.nonzero(p['scores'] >= args.score)[0]:
                name = names[int(p['cls'][k])]
                if name not in ('Pedestrian', 'LyingPerson'):
                    continue
                if not any(inside(p['boxes'][k][0], p['boxes'][k][1], g['box'], 1.0) for g in gt):
                    fp[name] += 1
        per = {k: 100.0 * fp[k] / len(other) for k in ('Pedestrian', 'LyingPerson') if k in names}
        print(f'\nfalse alarms per 100 {fp_what} ({len(other)} frames, boxes on no labelled object): '
              + ', '.join(f'{k} {v:.1f}' for k, v in per.items()))
        summary['false_alarms_per_100_frames'] = per
    if hidden:
        print('not counted (fewer than 5 LiDAR hits' + (' or outside the image' if args.fov_only else '') + '): '
              + ', '.join(f'{k} {v}' for k, v in hidden.items()))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   f"pose_eval_{args.split}{'_fov' if args.fov_only else ''}.json")
    with open(out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'saved {out}')


if __name__ == '__main__':
    main()