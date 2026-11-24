"""
check_kitti.py - verify a collect_kitti_lidar.py dataset before training on it.

    python check_kitti.py kitti_carla                 # stats + 6 overlay images
    python check_kitti.py kitti_carla --split val --n 12

It reads the KITTI files the way OpenPCDet does (label_2 + calib -> LiDAR boxes with
OpenPCDet's own formula; KITTI-devkit corners -> P2 for the image), so if the checks pass
here, OpenPCDet / mmdet3d see the same boxes.

Checks
  1. label_2 vs labels_3d: every real KITTI label converted back to the LiDAR frame must
     match its 360-deg LiDAR-frame twin (centre / size / heading error ~0.01 = rounding).
  2. points inside each converted box (the LiDAR really sees what is labelled).
  3. KITTI difficulty mix (OpenPCDet eval rules) and DontCare share.
  4. overlay images in <root>/check/: 3D boxes drawn on image_2 (green), DontCare (grey),
     bird's-eye view with the same boxes. Boxes must sit on the cars in BOTH views.
"""
import os, glob, argparse
from collections import Counter
import numpy as np
import kitti_utils as K

DIFF = {'easy': (40, 0, 0.15), 'moderate': (25, 1, 0.30), 'hard': (25, 2, 0.50)}   # min h px, max occ, max trunc


def load_l3d(path):
    out = []
    for line in open(path):
        f = line.split()
        if len(f) == 13 and f[10] == '1':
            out.append(np.array(f[1:8], float))
    return out


def count_in_box(pts, box, bottom_skip=0.1):
    x, y, z, l, w, h, yaw = box
    d = pts - np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return int(((np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)
                & (d[:, 2] >= -h / 2 + bottom_skip) & (d[:, 2] <= h / 2)).sum())


def draw(root, fid, objs, calib, pts, out_dir):
    import cv2
    img = cv2.imread(os.path.join(root, 'training', 'image_2', fid + '.png'))
    P2 = calib['P2']
    for o in objs:
        b = o['bbox'].astype(int)
        if o['name'] == 'DontCare':
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (128, 128, 128), 1)
            continue
        c = K.kitti_corners_cam(o)
        if (c[:, 2] < 0.1).any():
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
            continue
        uv = (P2 @ np.hstack([c, np.ones((8, 1))]).T).T
        uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
        for a, e in K.BOX_EDGES:
            cv2.line(img, tuple(uv[a]), tuple(uv[e]), (0, 255, 0), 1)
        cv2.putText(img, o['name'][:3], tuple(uv[4]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    S, R = 700, 60.0                                   # BEV: 700 px for +-60 m, x up, y LEFT on the left
    bev = np.zeros((S, S, 3), np.uint8)
    col = ((R - pts[:, 1]) * S / (2 * R)).astype(int)
    row = ((R - pts[:, 0]) * S / (2 * R)).astype(int)
    m = (col >= 0) & (col < S) & (row >= 0) & (row < S)
    bev[row[m], col[m]] = (160, 160, 160)
    for o in objs:
        if o['name'] == 'DontCare':
            continue
        box = K.kitti_to_velo_box(o, calib)
        cr = K.box_corners_velo(box)[:4]
        poly = np.stack([(R - cr[:, 1]) * S / (2 * R), (R - cr[:, 0]) * S / (2 * R)], 1).astype(np.int32)
        cv2.polylines(bev, [poly], True, (0, 255, 0), 1)
        head = box[:2] + np.array([np.cos(box[6]), np.sin(box[6])]) * box[3] / 2
        p0 = (int((R - box[1]) * S / (2 * R)), int((R - box[0]) * S / (2 * R)))
        p1 = (int((R - head[1]) * S / (2 * R)), int((R - head[0]) * S / (2 * R)))
        cv2.line(bev, p0, p1, (0, 255, 0), 1)
    cv2.circle(bev, (S // 2, S // 2), 3, (255, 255, 0), -1)
    h = bev.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
    cv2.imwrite(os.path.join(out_dir, fid + '.png'), np.hstack([img, bev]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--split', default='all', help='train / val / all / a run name')
    ap.add_argument('--n', type=int, default=6, help='overlay images to write')
    ap.add_argument('--max_frames', type=int, default=3000, help='frames used for the statistics')
    a = ap.parse_args()

    tr = os.path.join(a.root, 'training')
    if a.split == 'all':
        ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(tr, 'label_2', '*.txt')))
    else:
        ids = [l.strip() for l in open(os.path.join(a.root, 'ImageSets', a.split + '.txt')) if l.strip()]
    if not ids:
        raise SystemExit('no frames found')
    step = max(1, len(ids) // a.max_frames)
    stat_ids = ids[::step]

    names, diff, err_c, err_s, err_y = Counter(), Counter(), [], [], []
    unmatched, zero_pts, n_real, n_dc, missing, bad_order = 0, 0, 0, 0, Counter(), 0
    for fid in stat_ids:
        paths = {d: os.path.join(tr, d, fid + ext) for d, ext in
                 (('label_2', '.txt'), ('calib', '.txt'), ('velodyne', '.bin'), ('image_2', '.png'),
                  ('labels_3d', '.txt'), ('planes', '.txt'))}
        miss = [d for d, p in paths.items() if not os.path.isfile(p)]
        for d in miss:
            missing[d] += 1
        if 'label_2' in miss or 'calib' in miss or 'velodyne' in miss:
            continue
        calib = K.read_calib(paths['calib'])
        objs = K.read_label2(paths['label_2'])
        pts = np.fromfile(paths['velodyne'], np.float32).reshape(-1, 4)[:, :3]
        twins = load_l3d(paths['labels_3d']) if 'labels_3d' not in miss else []
        is_dc = [o['name'] == 'DontCare' for o in objs]
        if any(is_dc) and not all(is_dc[is_dc.index(True):]):
            bad_order += 1          # a real label after a DontCare: OpenPCDet would read wrong boxes
        for o in objs:
            if o['name'] == 'DontCare':
                n_dc += 1
                continue
            n_real += 1
            names[o['name']] += 1
            hgt = o['bbox'][3] - o['bbox'][1]
            level = next((k for k, (hm, om, tm) in DIFF.items()
                          if hgt >= hm and o['occ'] <= om and o['trunc'] <= tm), 'beyond hard')
            diff[level] += 1
            box = K.kitti_to_velo_box(o, calib)
            if count_in_box(pts, box) == 0:
                zero_pts += 1
            if twins:
                d = [np.linalg.norm(t[:3] - box[:3]) for t in twins]
                j = int(np.argmin(d))
                if d[j] > 0.5:
                    unmatched += 1
                    continue
                t = twins[j]
                err_c.append(d[j]); err_s.append(np.abs(t[3:6] - box[3:6]).max())
                err_y.append(abs(K.norm_angle(t[6] - box[6])))

    n = len(stat_ids)
    print(f'\n{len(ids)} frames ({n} used for statistics)')
    if missing:
        print('MISSING files:', dict(missing))
    print(f'real labels: {n_real} ({n_real / n:.1f}/frame)   DontCare: {n_dc} ({n_dc / n:.1f}/frame)')
    if bad_order:
        print(f'ERROR: {bad_order} label files have a real label after a DontCare line. OpenPCDet / '
              'mmdet3d assume DontCare lines are last and would read wrong boxes.')
    print('by class:', dict(names.most_common()))
    print('KITTI difficulty (OpenPCDet rules):', {k: diff[k] for k in ('easy', 'moderate', 'hard', 'beyond hard')})
    print(f'labels with 0 LiDAR points inside: {zero_pts} ({100 * zero_pts / max(n_real, 1):.1f}%)  '
          '<- should be ~0; if not, label_2 and velodyne disagree')
    if err_c:
        print(f'label_2 -> LiDAR vs labels_3d: max centre err {max(err_c):.3f} m, max size err '
              f'{max(err_s):.3f} m, max heading err {max(err_y):.3f} rad, unmatched {unmatched}  '
              '<- expect <= ~0.02 (2-decimal rounding)')
    if n_real and diff['easy'] + diff['moderate'] + diff['hard'] == 0:
        print('WARNING: no label falls in any KITTI difficulty -> KITTI AP will be empty')

    import cv2  # noqa: F401  (only needed for the images)
    out = os.path.join(a.root, 'check')
    os.makedirs(out, exist_ok=True)
    for fid in ids[::max(1, len(ids) // max(a.n, 1))][:a.n]:
        draw(a.root, fid, K.read_label2(os.path.join(tr, 'label_2', fid + '.txt')),
             K.read_calib(os.path.join(tr, 'calib', fid + '.txt')),
             np.fromfile(os.path.join(tr, 'velodyne', fid + '.bin'), np.float32).reshape(-1, 4), out)
    print(f'overlay images: {out}  (camera left, bird\'s-eye right; boxes must sit on the objects)')


if __name__ == '__main__':
    main()
