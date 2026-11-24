"""
Lidar_val.py (Fusion copy) - evaluate a checkpoint (BEV AP and 3D AP, KITTI R40).
Works for BOTH the painted fusion model and the LiDAR-only model (4 features), so the two are scored
by the same code on the same frames and region: the fusion vs LiDAR-only table for the report.

Files needed next to it: pp_common.py and data_setup.py (the Fusion copies).

Region filters (applied to ground truth, ignore boxes AND predictions alike):
    --fov_only               objects whose centre projects into the camera image (each frame's calib):
                             painting only exists there, so this is the fair comparison region
    --dist_min / --dist_max  objects whose centre is within this distance band (metres)

    python Lidar_val.py --fov_only                                       # fusion: runs_lidar/pp_painted_kitti_carla/best.pt
    python Lidar_val.py --ckpt ../Detection_LIDAR/runs_lidar/pp_kitti_carla/best.pt --fov_only   # LiDAR-only, same region
    python Lidar_val.py --fov_only --dist_min 30 --dist_max 60            # far objects, where fusion should help
    python Lidar_val.py --vis 20                                          # BEV images (GT green, pred red, ignore yellow)

Same as Detection_LIDAR/Lidar_val.py (table, mAP rule, speed, metrics.json), plus: painted input read
from the checkpoint's paint folder, the region filters, and a warning if the split is not the one the
checkpoint was trained with.
"""
import os
import json
import time
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import pp_common as C
import data_setup as D


def region_mask(boxes, calib, wh, fov_only, dmin, dmax):
    if len(boxes) == 0:
        return np.zeros(0, bool)
    m = np.ones(len(boxes), bool)
    if fov_only:
        m &= D.boxes_in_camera(boxes, calib, *wh)
    d = np.hypot(boxes[:, 0], boxes[:, 1])
    m &= (d >= dmin) & (d < dmax)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(D.DEFAULT_OUT_DIR, 'best.pt'))
    ap.add_argument('--data_root', default=D.DEFAULT_DATA_ROOT)
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split_file', default='', help='default: <data_root>/split_lidar.json')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--score_thr', type=float, default=0.1)
    ap.add_argument('--min_gt', type=int, default=30, help='classes with fewer GT boxes are left out of mAP')
    ap.add_argument('--vis', type=int, default=0, help='save N BEV images (GT green, pred red, ignore yellow)')
    ap.add_argument('--save_preds', action='store_true')
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--fov_only', action='store_true')
    ap.add_argument('--dist_min', type=float, default=0.0)
    ap.add_argument('--dist_max', type=float, default=1e9)
    ap.add_argument('--paint_name', default=None, help='override; normally read from the checkpoint')
    args = ap.parse_args()

    args.ckpt = os.path.abspath(args.ckpt)
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f'checkpoint not found: {args.ckpt}')
    args.data_root = D.resolve_data_root(args.data_root, args.label_dir)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = D.load_ckpt(args.ckpt)
    model = C.build_model_from_ckpt(ckpt, dev)
    n_feats = ckpt.get('num_pt_feats', 4)

    paint_dir, pname = None, None
    if n_feats != 4:
        pname = args.paint_name or ckpt.get('paint_name') or D.DEFAULT_PAINT_NAME
        paint_dir, n_paint, names = C.paint_info(args.data_root, pname)
        if n_paint != n_feats:
            raise SystemExit(f'{args.ckpt} expects {n_feats} features per point, the painted folder {paint_dir} '
                             f'gives {n_paint}: painted with a different segmentation model?')
        if ckpt.get('paint_classes') is not None and list(ckpt['paint_classes']) != names:
            raise SystemExit(f"{args.ckpt} was trained on paint classes {ckpt['paint_classes']}, "
                             f"{paint_dir} has {names}")

    tag = ('_fov' if args.fov_only else '')
    if args.dist_min > 0 or args.dist_max < 1e8:
        tag += '_d%g-%g' % (args.dist_min, args.dist_max)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), 'eval_' + args.split + tag)
    os.makedirs(out_dir, exist_ok=True)

    train_root = ckpt.get('args', {}).get('data_root')
    if args.split in ('all', 'train') and train_root and \
            os.path.normcase(os.path.realpath(train_root)) == os.path.normcase(os.path.realpath(args.data_root)):
        print(f'WARNING: --split {args.split} on the folder this model was trained on. '
              'These frames were seen in training, so the AP will be too high.')

    if args.split == 'all':
        ids = D.frame_ids(args.data_root)
    else:
        split_path = args.split_file or os.path.join(args.data_root, 'split_lidar.json')
        split = D.read_split(args.data_root, split_path)
        ids = split[args.split]
        if ckpt.get('split_sha1') not in (None, D.split_sha1(split)):
            print(f"WARNING: {args.ckpt} was trained with split {ckpt['split_sha1']}, {split_path} is now "
                  f"{D.split_sha1(split)}: the val frames may include training frames.")
    if paint_dir:
        missing = [i for i in ids if not os.path.exists(os.path.join(paint_dir, i + '.npy'))]
        if missing:
            raise SystemExit(f'{len(missing)} frames are not painted (e.g. {missing[:3]}): run paint_points.py')
    ds = C.LidarDataset(args.data_root, ids, args.label_dir, train=False, paint_dir=paint_dir)
    loader = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=C.collate, persistent_workers=args.workers > 0)
    regional = args.fov_only or args.dist_min > 0 or args.dist_max < 1e8
    wh = D.image_size(args.data_root, ids[0]) if args.fov_only else None
    print(f'[val] {len(ds)} frames  ckpt epoch {ckpt.get("epoch", -1) + 1}  device {dev}  '
          f'painted={bool(paint_dir)} ({n_feats} features)  fov_only={args.fov_only}  '
          f'dist=[{args.dist_min:g},{args.dist_max:g})')

    preds, gts, vis_done = [], [], 0
    t_inf, n_timed = 0.0, 0
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()
            out = model(b['points'])
            res = C.decode_predictions(out, model, score_thr=args.score_thr)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            if bi > 0:                                   # batch 0 includes CUDA / cuDNN warm-up
                t_inf += time.time() - t0
                n_timed += len(b['id'])
            for k in range(len(b['id'])):
                g = {'boxes': b['boxes'][k].numpy(), 'cls': b['cls'][k].numpy(),
                     'ign_boxes': b['ign_boxes'][k].numpy(), 'ign_cls': b['ign_cls'][k].numpy()}
                p = res[k]
                if regional:                             # same region for GT, ignore boxes and predictions
                    calib = D.frame_calib(args.data_root, b['id'][k]) if args.fov_only else None
                    gm = region_mask(g['boxes'], calib, wh, args.fov_only, args.dist_min, args.dist_max)
                    im = region_mask(g['ign_boxes'], calib, wh, args.fov_only, args.dist_min, args.dist_max)
                    pm = region_mask(p['boxes'], calib, wh, args.fov_only, args.dist_min, args.dist_max)
                    g = {'boxes': g['boxes'][gm], 'cls': g['cls'][gm],
                         'ign_boxes': g['ign_boxes'][im], 'ign_cls': g['ign_cls'][im]}
                    p = {'boxes': p['boxes'][pm], 'scores': p['scores'][pm], 'cls': p['cls'][pm]}
                preds.append(p)
                gts.append(g)
                if vis_done < args.vis:
                    import cv2
                    img = C.draw_bev(b['points'][k].numpy(), gt=g['boxes'], pred=p['boxes'], ign=g['ign_boxes'])
                    cv2.imwrite(os.path.join(out_dir, f"bev_{b['id'][k]}.png"), img)
                    vis_done += 1
                if args.save_preds:
                    pdir = os.path.join(out_dir, 'pred_txt')
                    os.makedirs(pdir, exist_ok=True)
                    with open(os.path.join(pdir, b['id'][k] + '.txt'), 'w') as f:
                        for bx, s, c in zip(p['boxes'], p['scores'], p['cls']):
                            f.write(f"{C.CLASS_NAMES[c]} {s:.4f} " + ' '.join(f'{v:.3f}' for v in bx) + '\n')

    print('[val] computing IoUs ...')
    ious = [C.pair_ious(p['boxes'], g['boxes']) for p, g in zip(preds, gts)]
    n_ign = np.bincount(np.concatenate([g['ign_cls'] for g in gts]).astype(np.int64), minlength=C.NUM_CLASSES)

    results = {'checkpoint': args.ckpt, 'painted': bool(paint_dir), 'num_pt_feats': n_feats,
               'region': {'fov_only': args.fov_only, 'dist_min': args.dist_min, 'dist_max': args.dist_max},
               'split': args.split, 'frames': len(ids)}
    keys = ['BEV@0.5', 'BEV@0.7', '3D@0.5', '3D@0.7']
    print('\n' + '=' * 92)
    print(f"{'class':<12}{'GT':>7}{'ignored':>9}{'IoU':>6}{'BEV@std':>9}{'3D@std':>8}"
          + ''.join(f'{k:>10}' for k in keys))
    print('-' * 92)
    for c, name in enumerate(C.CLASS_NAMES):
        row, npos = {}, 0
        for mode, mname in ((0, 'BEV'), (1, '3D')):
            for thr in (0.5, 0.7):
                row[f'{mname}@{thr}'], npos = C.average_precision(preds, gts, ious, c, thr, mode)
            row[f'{mname}@std'], npos = C.average_precision(preds, gts, ious, c, C.EVAL_IOU[c], mode)
        row['gt'], row['ignored'] = npos, int(n_ign[c])
        results[name] = row
        flag = '' if npos >= args.min_gt else '   (not in mAP: too few GT)'
        print(f"{name:<12}{npos:>7}{int(n_ign[c]):>9}{C.EVAL_IOU[c]:>6.1f}"
              f"{row['BEV@std']:>9.2f}{row['3D@std']:>8.2f}"
              + ''.join(f'{row[k]:>10.2f}' for k in keys) + flag)
    used = [n for n in C.CLASS_NAMES if results[n]['gt'] >= args.min_gt]
    mean = {k: float(np.nanmean([results[n][k] for n in used])) if used else float('nan')
            for k in ['BEV@std', '3D@std'] + keys}
    results['mAP'] = dict(mean, classes=used)
    print('-' * 92)
    print(f"{'mAP':<12}{'':>7}{'':>9}{'':>6}{mean['BEV@std']:>9.2f}{mean['3D@std']:>8.2f}"
          + ''.join(f'{mean[k]:>10.2f}' for k in keys))
    print(f'mAP over: {used}')
    print('=' * 92)
    if n_timed > 0 and t_inf > 0:
        fps = n_timed / t_inf
        results['fps'] = fps
        print(f'Inference speed: {fps:.1f} frames/s  ({1000 / fps:.1f} ms/frame, incl. voxelisation + NMS, '
              f'first batch excluded; painting / SegFormer time not included)')
    else:
        results['fps'] = None
        print('Inference speed: not measured (needs at least 2 batches)')
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved metrics to {out_dir}/metrics.json')


if __name__ == '__main__':
    main()
