"""
lidar_val.py - evaluate a trained PointPillars checkpoint (BEV AP and 3D AP, KITTI R40).

    python lidar_val.py --ckpt runs_lidar/pp_town04/best.pt
    python lidar_val.py --ckpt runs_lidar/pp_town04/best.pt --vis 20
    # different town = the honest generalisation number, using every frame:
    python lidar_val.py --ckpt ... --data_root ../data_collection_lidar/dataset/Town05 --split all

Changes vs the previous version:
  - ignore boxes (too few LiDAR hits) are passed to the AP: detections on them are not FPs
  - table shows GT count per class and AP at the class's standard IoU (EVAL_IOU)
  - mAP only averages classes with at least --min_gt boxes; the others are listed, not hidden
  - --split all on the training folder now warns (that measures memorisation, not accuracy)
"""
import os, json, time, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import pp_common as C


RANGE_BINS = [(0, 20), (20, 40), (40, 51.2)]
KEYS = ['BEV@0.5', 'BEV@0.7', '3D@0.5', '3D@0.7']


def ap_table(preds, gts, title, min_gt):
    """Print and return the per-class AP table (KITTI R40) for one evaluation region."""
    ious = [C.pair_ious(p['boxes'], g['boxes']) for p, g in zip(preds, gts)]
    n_ign = np.bincount(np.concatenate([g['ign_cls'] for g in gts]).astype(np.int64),
                        minlength=C.NUM_CLASSES)
    res = {}
    print('\n' + '=' * 92 + f'\n{title}')
    print(f"{'class':<12}{'GT':>7}{'ignored':>9}{'IoU':>6}{'BEV@std':>9}{'3D@std':>8}"
          + ''.join(f'{k:>10}' for k in KEYS))
    print('-' * 92)
    for c, name in enumerate(C.CLASS_NAMES):
        row = {}
        for mode, mname in ((0, 'BEV'), (1, '3D')):
            for thr in (0.5, 0.7):
                row[f'{mname}@{thr}'], npos = C.average_precision(preds, gts, ious, c, thr, mode)
            row[f'{mname}@std'], _ = C.average_precision(preds, gts, ious, c, C.EVAL_IOU[c], mode)
        row['gt'], row['ignored'] = npos, int(n_ign[c])
        res[name] = row
        flag = '' if npos >= min_gt else '   (not in mAP: too few GT)'
        print(f"{name:<12}{npos:>7}{int(n_ign[c]):>9}{C.EVAL_IOU[c]:>6.1f}"
              f"{row['BEV@std']:>9.2f}{row['3D@std']:>8.2f}"
              + ''.join(f'{row[k]:>10.2f}' for k in KEYS) + flag)
    used = [n for n in C.CLASS_NAMES if res[n]['gt'] >= min_gt]
    mean = {k: float(np.nanmean([res[n][k] for n in used])) if used else float('nan')
            for k in ['BEV@std', '3D@std'] + KEYS}
    res['mAP'] = dict(mean, classes=used)
    print('-' * 92)
    print(f"{'mAP':<12}{'':>7}{'':>9}{'':>6}{mean['BEV@std']:>9.2f}{mean['3D@std']:>8.2f}"
          + ''.join(f'{mean[k]:>10.2f}' for k in KEYS))
    print(f'mAP over: {used}')
    return res


def restrict(preds, gts, keep):
    """Evaluate one region only. keep(boxes, frame_index) -> bool mask on box centres.
    Predictions outside the region are dropped; GT outside it become IGNORE boxes, so a
    detection that sits on the border and matches one of them is not a false positive."""
    P, G = [], []
    for f, (p, g) in enumerate(zip(preds, gts)):
        mp, mg = keep(p['boxes'], f), keep(g['boxes'], f)
        P.append({k: v[mp] for k, v in p.items()})
        G.append({'boxes': g['boxes'][mg], 'cls': g['cls'][mg],
                  'ign_boxes': np.concatenate([g['ign_boxes'], g['boxes'][~mg]]),
                  'ign_cls': np.concatenate([g['ign_cls'], g['cls'][~mg]])})
    return P, G


def load_calib(path):
    """Tr_velo_to_cam (3x4) and P2 (3x4) from a KITTI calib file."""
    v = {}
    for line in open(path):
        if ':' in line:
            k, s = line.split(':', 1)
            v[k.strip()] = np.array(s.split(), float)
    return v['Tr_velo_to_cam'].reshape(3, 4), v['P2'].reshape(3, 4)


def in_image(boxes, Tr, P2):
    """Box centre in front of the camera and inside the image. Image size = 2 x principal
    point, true for collect_kitti_lidar.py (centred principal point)."""
    if len(boxes) == 0:
        return np.zeros(0, bool)
    cam = boxes[:, :3] @ Tr[:, :3].T + Tr[:, 3]
    uvw = cam @ P2[:, :3].T + P2[:, 3]
    z = np.maximum(uvw[:, 2], 1e-6)
    u, v = uvw[:, 0] / z, uvw[:, 1] / z
    return (cam[:, 2] > 0.1) & (u >= 0) & (u < 2 * P2[0, 2]) & (v >= 0) & (v < 2 * P2[1, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs_lidar/pp_kitti_carla/best.pt')
    ap.add_argument('--data_root', default='../LIDAR_only/kitti_carla/training')
    ap.add_argument('--fov_only', action='store_true', help='also report AP inside the camera image only')
    ap.add_argument('--range_bins', action='store_true', help='also report AP for 0-20 / 20-40 / 40-51.2 m')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--score_thr', type=float, default=0.1)
    ap.add_argument('--min_gt', type=int, default=30, help='classes with fewer GT boxes are left out of mAP')
    ap.add_argument('--vis', type=int, default=0, help='save N BEV images (GT green, pred red, ignore yellow)')
    ap.add_argument('--save_preds', action='store_true')
    ap.add_argument('--out_dir', default='')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model = C.build_model_from_ckpt(ckpt, dev)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), 'eval_' + args.split)
    os.makedirs(out_dir, exist_ok=True)

    train_root = ckpt.get('args', {}).get('data_root')
    if args.split in ('all', 'train') and train_root and \
            os.path.abspath(train_root) == os.path.abspath(args.data_root):
        print(f'WARNING: --split {args.split} on the folder this model was trained on. '
              'These frames were seen in training, so the AP will be too high.')

    if args.split == 'all':
        ids = C.get_frame_ids(args.data_root)
    else:
        sp = os.path.join(args.data_root, 'split_lidar.json')
        if not os.path.exists(sp):
            raise SystemExit(f'{sp} not found - run lidar_train.py first (it creates the split).')
        ids = json.load(open(sp))[args.split]
    ds = C.LidarDataset(args.data_root, ids, args.label_dir, train=False)
    loader = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=C.collate, persistent_workers=args.workers > 0)
    print(f'[val] {len(ds)} frames  ckpt epoch {ckpt["epoch"]+1}')

    preds, gts, frame_ids, vis_done = [], [], [], 0
    t_inf, n_frames = 0.0, 0
    with torch.no_grad():
        for b in loader:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(b['points'])
            res = C.decode_predictions(out, model, score_thr=args.score_thr)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t_inf += time.time() - t0; n_frames += len(b['id'])
            for k in range(len(b['id'])):
                g = {'boxes': b['boxes'][k].numpy(), 'cls': b['cls'][k].numpy(),
                     'ign_boxes': b['ign_boxes'][k].numpy(), 'ign_cls': b['ign_cls'][k].numpy()}
                preds.append(res[k]); gts.append(g); frame_ids.append(b['id'][k])
                if vis_done < args.vis:
                    import cv2
                    img = C.draw_bev(b['points'][k].numpy(), gt=g['boxes'], pred=res[k]['boxes'],
                                     ign=g['ign_boxes'])
                    cv2.imwrite(os.path.join(out_dir, f"bev_{b['id'][k]}.png"), img)
                    vis_done += 1
                if args.save_preds:
                    pdir = os.path.join(out_dir, 'pred_txt'); os.makedirs(pdir, exist_ok=True)
                    with open(os.path.join(pdir, b['id'][k] + '.txt'), 'w') as f:
                        for bx, s, c in zip(res[k]['boxes'], res[k]['scores'], res[k]['cls']):
                            f.write(f"{C.CLASS_NAMES[c]} {s:.4f} " + ' '.join(f'{v:.3f}' for v in bx) + '\n')

    results = {'all (360 deg)': ap_table(preds, gts, 'all (360 deg)', args.min_gt)}
    if args.fov_only:                                    # same region a camera can see
        calib_dir = os.path.join(args.data_root, 'calib')
        if not os.path.isdir(calib_dir):
            raise SystemExit(f'--fov_only needs {calib_dir} (KITTI-format data from collect_kitti_lidar.py)')
        cal = [load_calib(os.path.join(calib_dir, i + '.txt')) for i in frame_ids]
        keep = lambda boxes, f: in_image(boxes, *cal[f])
        P, G = restrict(preds, gts, keep)
        results['camera FOV'] = ap_table(P, G, 'camera FOV only', args.min_gt)
    for lo, hi in RANGE_BINS if args.range_bins else []:  # detection quality vs distance
        keep = lambda boxes, f, lo=lo, hi=hi: (np.hypot(boxes[:, 0], boxes[:, 1]) >= lo) & \
                                              (np.hypot(boxes[:, 0], boxes[:, 1]) < hi)
        P, G = restrict(preds, gts, keep)
        results[f'{lo}-{hi} m'] = ap_table(P, G, f'range {lo}-{hi} m', args.min_gt)
    fps = n_frames / max(t_inf, 1e-9)
    results['fps'] = fps
    print(f'Inference speed: {fps:.1f} frames/s  ({1000/fps:.1f} ms/frame, incl. voxelisation + NMS)')
    json.dump(results, open(os.path.join(out_dir, 'metrics.json'), 'w'), indent=2)
    print(f'Saved metrics to {out_dir}/metrics.json')


if __name__ == '__main__':
    main()
