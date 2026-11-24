"""
paint_points.py - PointPainting step 1: attach SegFormer class probabilities to every LiDAR point.

For each frame:  image_2 -> SegFormer -> per-class probabilities;  velodyne points -> camera with that
frame's calib/<id>.txt (verified by check_projection.py) -> probabilities sampled at the pixel under
each point.

Output (velodyne/ is not touched):
    <data_root>/velodyne_paint/<id>.npy   uint8 [N, n_classes + 1], same point order as velodyne/<id>.bin
        columns 0..n_classes-1 : class probability * 255   (0 for points outside the camera view)
        last column            : 255 if the point projects inside the image, else 0
    <data_root>/velodyne_paint/paint_meta.json   class names, segmentation model, settings
This is the same format the Driving stack paints live (perception_scheduler.py paint_frame).

Run from anywhere (defaults: LIDAR_only/kitti_carla/training and Detection_YOLO/runs/seg/best):
    python check_projection.py                       # 1) the projection check must pass first
    python paint_points.py --limit 20 --preview 6    # 2) quick look: runs_fusion/paint_preview/
    python paint_points.py                           # 3) every frame (resumable: done frames are skipped)
    python check_paint.py --n 50                     # 4) painting vs CARLA's semantic camera

Changes vs the Town03 version:
  - kitti_carla/training layout; projection from each frame's calib file instead of a fixed
    1.5 m offset and 90 deg FOV
  - SegFormer weights are loaded strictly: transformers renamed the decode-head keys between
    versions, and a partly random decode head used to load with only a warning
  - painting with a different segmentation model into an existing folder is refused (mixing two
    models' paint in one training set); a frame whose point count changed is repainted
  - files are written atomically (Ctrl+C never leaves a half-written .npy)
"""
import os
import json
import time
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import data_setup as D

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE_RGB = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [153, 153, 153], [250, 170, 30],
                        [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
                        [0, 0, 142], [255, 255, 255], [110, 190, 160]], dtype=np.uint8)
FORMAT = 'uint8 [N, n_classes+1]: probs*255 per class, last col = in-camera flag (255/0)'

# transformers >= 5.9 renamed SegformerDecodeHead.linear_c -> linear_projections (same tensors)
SEG_KEY_RENAMES = (('decode_head.linear_projections.', 'decode_head.linear_c.'),
                   ('decode_head.linear_c.', 'decode_head.linear_projections.'))


def _read_state_dict(seg_dir):
    st = os.path.join(seg_dir, 'model.safetensors')
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    pt = os.path.join(seg_dir, 'pytorch_model.bin')
    if os.path.isfile(pt):
        return torch.load(pt, map_location='cpu')
    raise SystemExit(f'{seg_dir}: no model.safetensors or pytorch_model.bin')


def load_segformer(seg_dir):
    """from_pretrained loads and, on transformers 5.x, converts the old layer names itself; its
    own loading report is the strict check. Only if that report shows missing or unused tensors
    (e.g. transformers <= 5.8 reading a decode head saved as linear_projections) are the raw
    weights re-loaded with the rename above. Stops if anything is still missing.
    Same loader as perception_scheduler.py, so training and the car use identical weights."""
    from transformers import SegformerForSemanticSegmentation
    from transformers.utils import logging as hf_logging
    if not os.path.isdir(seg_dir):
        raise SystemExit(f'segmentation model folder not found: {seg_dir} (pass --seg_model)')
    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model, info = SegformerForSemanticSegmentation.from_pretrained(seg_dir, output_loading_info=True)
    finally:
        hf_logging.set_verbosity(prev)
    missing = [k for k in info.get('missing_keys', []) if not k.endswith('num_batches_tracked')]
    unexpected = list(info.get('unexpected_keys', []))
    mismatched = list(info.get('mismatched_keys', []))
    renamed, report = 0, (missing, unexpected)
    if missing or unexpected:
        want = set(model.state_dict().keys())
        fixed = {}
        for k, v in _read_state_dict(seg_dir).items():
            if k not in want:
                for a, b in SEG_KEY_RENAMES:
                    if k.startswith(a) and b + k[len(a):] in want:
                        k, renamed = b + k[len(a):], renamed + 1
                        break
            fixed[k] = v
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        missing = [k for k in missing if not k.endswith('num_batches_tracked')]
        if missing or unexpected:
            missing, unexpected = report
    if missing or unexpected or mismatched:
        raise SystemExit(f'SegFormer checkpoint {seg_dir} does not load fully in this transformers version.\n'
                         f'  missing: {missing}\n  unused: {unexpected}\n  wrong shape: {mismatched}')
    print(f'[seg] {seg_dir}: all {len(model.state_dict())} tensors loaded strictly'
          + (f' ({renamed} decode-head keys renamed)' if renamed else ''))
    return model.eval()


def n_points(root, fid):
    return os.path.getsize(os.path.join(root, 'velodyne', fid + '.bin')) // 16


def is_done(out_dir, root, fid):
    p = os.path.join(out_dir, fid + '.npy')
    if not os.path.exists(p):
        return False
    try:
        return np.load(p, mmap_mode='r').shape[0] == n_points(root, fid)     # header only
    except (ValueError, OSError):
        return False


def save_atomic(path, arr):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seg_model', default=D.DEFAULT_SEG_MODEL, help='folder saved by seg_train.py')
    ap.add_argument('--data_root', default=D.DEFAULT_DATA_ROOT)
    ap.add_argument('--out_name', default=D.DEFAULT_PAINT_NAME)
    ap.add_argument('--img_size', type=int, nargs=2, default=[384, 768], metavar=('H', 'W'),
                    help='SegFormer input size; must equal --seg-img-size of the Driving stack')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--split', default='all', choices=['all', 'train', 'val'],
                    help='which frames (all = every frame in velodyne/, needed for training + val)')
    ap.add_argument('--ids', nargs='*', default=None)
    ap.add_argument('--limit', type=int, default=0, help='only the first N frames (0 = all)')
    ap.add_argument('--overwrite', action='store_true', help='repaint frames that are already done')
    ap.add_argument('--preview', type=int, default=0, help='save N preview images of painted points')
    ap.add_argument('--preview_dir', default=os.path.join(D.FUSION_RUNS, 'paint_preview'))
    args = ap.parse_args()

    root = D.resolve_data_root(args.data_root)
    D.need_dir(root, 'image_2', 'the camera images SegFormer runs on')
    D.need_dir(root, 'calib', 'per-frame projection matrices')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    half = dev.type == 'cuda'
    model = load_segformer(args.seg_model).to(dev)
    n_cls = model.config.num_labels
    names = [model.config.id2label[i] for i in range(n_cls)]
    print(f'[seg] {n_cls} classes: {names}   device {dev}')

    if args.ids:
        ids = args.ids
    elif args.split == 'all':
        ids = D.frame_ids(root)
    else:
        ids = D.read_split(root, os.path.join(root, 'split_lidar.json'))[args.split]
    if args.limit:
        ids = ids[:args.limit]
    out_dir = os.path.join(root, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    if args.preview:
        os.makedirs(args.preview_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, 'paint_meta.json')
    meta = {'classes': names, 'ysign': 'left', 'projection': 'per-frame calib/<id>.txt (P2, R0_rect, '
            'Tr_velo_to_cam)', 'seg_model': os.path.abspath(args.seg_model), 'img_size': args.img_size,
            'format': FORMAT}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        clash = [k for k in ('classes', 'img_size') if old.get(k) != meta[k]]
        if os.path.normcase(os.path.abspath(old.get('seg_model', ''))) != os.path.normcase(meta['seg_model']):
            clash.append('seg_model')
        if clash and not args.overwrite:
            raise SystemExit(f'{out_dir} was painted with a different {clash} ({old.get("seg_model")}).\n'
                             f'Mixing two models in one training set is wrong. Use --out_name <new folder>, '
                             f'or --overwrite to repaint every frame with this model.')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    todo_all = [i for i in ids if args.overwrite or not is_done(out_dir, root, i)]
    print(f'[paint] {len(ids)} frames, {len(ids) - len(todo_all)} already painted, {len(todo_all)} to do '
          f'-> {out_dir}')
    if args.preview and not todo_all:
        todo_all = ids[:args.preview]                     # previews of done frames
    H_in, W_in = args.img_size
    n_done, in_cam_frac, argmax_counts, previews = 0, [], np.zeros(n_cls, np.int64), 0
    cam_warned = False
    t0 = time.time()
    for s in range(0, len(todo_all), args.batch):
        chunk = todo_all[s:s + args.batch]
        batch_imgs, bgr_imgs = [], []
        for fid in chunk:
            bgr = cv2.imread(os.path.join(root, 'image_2', fid + '.png'))
            if bgr is None:
                raise SystemExit(f'cannot read image_2/{fid}.png')
            bgr_imgs.append(bgr)
            x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W_in, H_in), interpolation=cv2.INTER_LINEAR)
            x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            batch_imgs.append(x.transpose(2, 0, 1))
        batch = torch.from_numpy(np.stack(batch_imgs)).to(dev)
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=half):
            logits = model(pixel_values=batch).logits.float()            # [B, C, h, w] (1/4 resolution)

        for b, fid in enumerate(chunk):
            pts = np.fromfile(os.path.join(root, 'velodyne', fid + '.bin'), dtype=np.float32).reshape(-1, 4)
            calib = D.frame_calib(root, fid)
            Hh, Ww = bgr_imgs[b].shape[:2]
            if not cam_warned and D.runtime_camera_mismatch(calib, Ww, Hh) >= 0.5:
                cam_warned = True
                print('[paint] WARNING: this calib differs from the camera model the Driving stack paints '
                      'with live (90 deg, 1.5 m ahead): run check_projection.py')
            u, v, _, ok = D.project(pts[:, :3], calib, Ww, Hh)
            extra = np.zeros((len(pts), n_cls + 1), np.uint8)
            probs = None
            if ok.any():
                gx = (u[ok] + 0.5) / Ww * 2 - 1
                gy = (v[ok] + 0.5) / Hh * 2 - 1
                grid = torch.from_numpy(np.stack([gx, gy], 1)).float().to(dev).view(1, 1, -1, 2)
                samp = F.grid_sample(logits[b:b + 1], grid, mode='bilinear', padding_mode='border',
                                     align_corners=False)                   # [1, C, 1, K]
                probs = torch.softmax(samp[0, :, 0, :], dim=0).T.cpu().numpy()   # [K, C]
                extra[ok, :n_cls] = np.round(probs * 255).astype(np.uint8)
                extra[ok, n_cls] = 255
                argmax_counts += np.bincount(probs.argmax(1), minlength=n_cls)
            save_atomic(os.path.join(out_dir, fid + '.npy'), extra)
            in_cam_frac.append(float(ok.mean()) if len(ok) else 0.0)
            n_done += 1

            if args.preview and previews < args.preview and probs is not None:
                img = bgr_imgs[b].copy()
                cls = probs.argmax(1)
                for x_, y_, c_ in zip(u[ok], v[ok], cls):
                    col = PALETTE_RGB[c_ % len(PALETTE_RGB)][::-1]
                    cv2.circle(img, (int(x_), int(y_)), 3, tuple(int(t) for t in col), -1)
                for k, nm in enumerate(names):
                    cv2.putText(img, nm, (8, 20 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                tuple(int(t) for t in PALETTE_RGB[k % len(PALETTE_RGB)][::-1]), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(args.preview_dir, f'paint_{fid}.png'), img)
                previews += 1
        if (s // args.batch) % 20 == 0:
            el = time.time() - t0
            print(f'{n_done}/{len(todo_all)} frames  ({n_done / max(el, 1e-6):.1f} frames/s)')

    print(f'\npainted {n_done} frames in {time.time() - t0:.0f}s; '
          f'{100 * np.mean(in_cam_frac) if in_cam_frac else 0:.1f}% of points are inside the camera view on average')
    tot = max(int(argmax_counts.sum()), 1)
    print('share of in-view points by predicted class:')
    for k in np.argsort(-argmax_counts):
        print(f'  {names[k]:<14} {100 * argmax_counts[k] / tot:5.1f}%')
    left = [i for i in D.frame_ids(root) if not is_done(out_dir, root, i)]
    print(f'saved to {out_dir}')
    if left:
        print(f'{len(left)} frames in velodyne/ are not painted yet (e.g. {left[:3]}): Lidar_train.py needs '
              f'every train and val frame painted')
    else:
        print('every frame is painted. Next: python check_paint.py --n 50, then python Lidar_train.py --check')


if __name__ == '__main__':
    main()
