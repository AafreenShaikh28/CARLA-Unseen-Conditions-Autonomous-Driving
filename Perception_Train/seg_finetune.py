"""
seg_finetune.py - fine-tune your SegFormer (runs\\seg\\best) on the camera collector's frames
(dataset_pose\\<Town>\\image_2 + semantic).

Same input as the car (Driving\\perception_scheduler.py, FusionPP): RGB resized to 384 x 768, /255, ImageNet
mean / std. Ground truth: the CARLA tag (red channel of semantic\\<id>.png) mapped to the MODEL'S OWN classes
(config.id2label) with the name table of Fusion\\check_paint.py / Driving\\eval_camera_models.py (the most
specific class wins). Tags no class covers are ignored (255), never trained as some other class.
--check prints that mapping and the old model's IoU per class: a class the old model gets badly wrong
while it is common in the data means your original training mapped it differently. Fix NAME_TO_TAGS first.

    python seg_finetune.py --data-root <...\\dataset_pose> --check                 # mapping + BEFORE (val)
    python seg_finetune.py --data-root <...\\dataset_pose>                         # -> runs\\seg\\best_pose
    python seg_finetune.py --data-root <...\\dataset_pose> --eval-only --seg-model runs\\seg\\best_pose   # AFTER

Split by town (default Town04 train, Town01 val, like the LiDAR models). The best epoch by val mIoU is saved
with save_pretrained to --out, a NEW folder: runs\\seg\\best is left alone, because the fusion PointPillars
was trained on points painted by it. Validation is at full image size (800 x 600), like eval_camera_models.py.
"""
import os
import glob
import json
import math
import time
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IGNORE = 255
# segmentation class name -> CARLA 0.9.15 semantic tags (same table as Fusion/check_paint.py)
NAME_TO_TAGS = {
    'road': (1,), 'sidewalk': (2,), 'building': (3,), 'wall': (4,), 'fence': (5,), 'pole': (6,),
    'traffic_light': (7,), 'traffic_sign': (8,), 'vegetation': (9,), 'terrain': (10,), 'sky': (11,),
    'pedestrian': (12, 13), 'person': (12, 13), 'rider': (13,),
    'vehicle': (14, 15, 16, 17, 18, 19), 'car': (14,), 'truck': (15,), 'bus': (16,), 'train': (17,),
    'motorcycle': (18,), 'bicycle': (19,), 'static': (20,), 'dynamic': (21,), 'other': (22,),
    'water': (23,), 'road_line': (24,), 'roadline': (24,), 'lane_marking': (24,), 'ground': (25,),
    'bridge': (26,), 'rail_track': (27,), 'guard_rail': (28,), 'guardrail': (28,),
}
TAG_NAMES = ['unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light',
             'traffic_sign', 'vegetation', 'terrain', 'sky', 'pedestrian', 'rider', 'car', 'truck', 'bus',
             'train', 'motorcycle', 'bicycle', 'static', 'dynamic', 'other', 'water', 'roadline', 'ground',
             'bridge', 'railtrack', 'guardrail']


def norm(name):
    return name.lower().replace(' ', '_').replace('-', '_')


def tag_lut(names):
    """256 entries: CARLA tag -> class index (255 = no class covers it). Most specific class wins."""
    lut = np.full(256, IGNORE, np.uint8)
    for k in sorted(range(len(names)), key=lambda k: len(NAME_TO_TAGS.get(norm(names[k]), ())) or 99):
        for t in NAME_TO_TAGS.get(norm(names[k]), ()):
            if lut[t] == IGNORE:
                lut[t] = k
    return lut


# ------------------------------------------------------------------ model (strict loading)
RENAMES = (('decode_head.linear_projections.', 'decode_head.linear_c.'),
           ('decode_head.linear_c.', 'decode_head.linear_projections.'))


def load_segformer(seg_dir):
    """from_pretrained, trusting its own loading report. Only if it reports layers it left random (the
    transformers 5.9 rename decode_head.linear_c <-> linear_projections, seen with older versions) are the
    weights loaded again by hand with that rename. Stops if anything is still missing."""
    from transformers import SegformerForSemanticSegmentation
    from transformers.utils import logging as hf_logging
    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model, info = SegformerForSemanticSegmentation.from_pretrained(seg_dir, output_loading_info=True)
    finally:
        hf_logging.set_verbosity(prev)
    missing = [k for k in info.get('missing_keys', []) if not k.endswith('num_batches_tracked')]
    if missing:
        st = os.path.join(seg_dir, 'model.safetensors')
        if os.path.isfile(st):
            from safetensors.torch import load_file
            sd = load_file(st)
        else:
            sd = torch.load(os.path.join(seg_dir, 'pytorch_model.bin'), map_location='cpu')
        want, fixed = set(model.state_dict()), {}
        for k, v in sd.items():
            if k not in want:
                for a, b in RENAMES:
                    if k.startswith(a) and b + k[len(a):] in want:
                        k = b + k[len(a):]
                        break
            fixed[k] = v
        miss2, _ = model.load_state_dict(fixed, strict=False)
        miss2 = [k for k in miss2 if not k.endswith('num_batches_tracked')]
        if miss2:
            raise SystemExit(f'{seg_dir}: {len(miss2)} layers would be random (e.g. {miss2[:3]}): '
                             f'this transformers version cannot load it')
        print(f'  loaded with the linear_c / linear_projections rename ({len(missing)} tensors)')
    model.config.semantic_loss_ignore_index = IGNORE
    return model


# ------------------------------------------------------------------ data
def frames(root, towns):
    out = []
    for t in towns:
        d = os.path.join(root, t)
        if not os.path.isdir(d):
            raise SystemExit(f'{d} not found')
        meta = os.path.isdir(os.path.join(d, 'metadata'))          # metadata = written last by the collector
        pat = os.path.join(d, 'metadata', '*.json') if meta else os.path.join(d, 'image_2', '*.png')
        for p in sorted(glob.glob(pat)):
            fid = os.path.splitext(os.path.basename(p))[0]
            if all(os.path.isfile(os.path.join(d, s, fid + '.png')) for s in ('image_2', 'semantic')):
                out.append((d, fid))
    return out


def read_tags(path):
    sem = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return sem[:, :, 2] if sem.ndim == 3 else sem             # CARLA tag in the red channel (BGR load)


class SegData(Dataset):
    def __init__(self, items, lut, hw, train):
        self.items, self.lut, self.hw, self.train = items, lut, hw, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, fid = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(d, 'image_2', fid + '.png')), cv2.COLOR_BGR2RGB)
        lab = self.lut[read_tags(os.path.join(d, 'semantic', fid + '.png'))]
        h, w = self.hw
        x = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        if self.train:
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
            if np.random.rand() < 0.5:                                         # mirror
                x, lab = x[:, ::-1], lab[:, ::-1]
            x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
        x = (x - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(lab)).long())


def evaluate(model, items, lut, hw, n_cls, dev, batch, workers, limit=0):
    if limit and len(items) > limit:
        items = [items[i] for i in np.linspace(0, len(items) - 1, limit).astype(int)]
    dl = DataLoader(SegData(items, lut, hw, False), batch, num_workers=workers)
    conf = torch.zeros(n_cls, n_cls, dtype=torch.long, device=dev)
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                logits = model(pixel_values=x).logits
            pred = F.interpolate(logits.float(), size=y.shape[-2:], mode='bilinear', align_corners=False).argmax(1)
            m = y != IGNORE
            conf += torch.bincount(y[m] * n_cls + pred[m], minlength=n_cls * n_cls).view(n_cls, n_cls)
    conf = conf.cpu().numpy().astype(np.float64)
    tp, gt, pr = np.diag(conf), conf.sum(1), conf.sum(0)
    iou = np.where(gt > 0, tp / np.maximum(gt + pr - tp, 1), np.nan)
    return dict(miou=float(np.nanmean(iou)), pixel_acc=float(tp.sum() / max(conf.sum(), 1)),
                iou=iou.tolist(), gt_share=(gt / max(gt.sum(), 1)).tolist(), frames=len(items),
                conf=conf.tolist())


PALETTE = {'road': (128, 64, 128), 'sidewalk': (244, 35, 232), 'building': (70, 70, 70), 'wall': (102, 102, 156),
           'fence': (190, 153, 153), 'pole': (153, 153, 153), 'traffic_light': (250, 170, 30),
           'traffic_sign': (220, 220, 0), 'vegetation': (107, 142, 35), 'terrain': (152, 251, 152),
           'sky': (70, 130, 180), 'pedestrian': (220, 20, 60), 'person': (220, 20, 60), 'vehicle': (0, 0, 142),
           'car': (0, 0, 142), 'road_line': (157, 234, 50), 'roadline': (157, 234, 50), 'other': (255, 255, 255)}


def colour(lab, names):
    lut = np.zeros((256, 3), np.uint8)                               # 255 (ignored) stays black
    for k, nm in enumerate(names):
        lut[k] = PALETTE.get(norm(nm), ((37 * k) % 255, (91 * k) % 255, (151 * k) % 255))
    return lut[lab]                                                  # RGB


def diagnose(model, items, lut, hw, names, dev, out_dir, n_vis):
    """(1) the ground truth and the prediction in the patch right in front of the car (the ego's own
    road: should be road / road_line almost always); (2) n_vis pictures: RGB | GT | prediction | GT over RGB"""
    n = len(names)
    pick = [items[i] for i in np.linspace(0, len(items) - 1, min(len(items), 200)).astype(int)]
    gt_h, pr_h, tag_pred = np.zeros(257), np.zeros(n), np.zeros((256, n))
    vis = set(np.linspace(0, len(pick) - 1, min(n_vis, len(pick))).astype(int).tolist()) if n_vis else set()
    if vis:
        os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for j, (d, fid) in enumerate(pick):
        bgr = cv2.imread(os.path.join(d, 'image_2', fid + '.png'))
        tags = read_tags(os.path.join(d, 'semantic', fid + '.png'))
        if bgr is None or tags is None or tags.shape != bgr.shape[:2]:
            print(f'    {d}\\{fid}: image / semantic missing or of different size {None if tags is None else tags.shape}')
            continue
        h, w = hw
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - MEAN) / STD
        with torch.no_grad():
            lo = model(pixel_values=torch.from_numpy(x.transpose(2, 0, 1))[None].to(dev)).logits
        pred = F.interpolate(lo.float(), size=tags.shape, mode='bilinear', align_corners=False).argmax(1)[0]
        pred = pred.cpu().numpy().astype(np.uint8)
        H, W = tags.shape
        ys, xs = slice(int(0.88 * H), H), slice(int(0.4 * W), int(0.6 * W))   # bottom centre
        gt_h += np.bincount(tags[ys, xs].ravel(), minlength=257)[:257]
        tag_pred += np.bincount(tags.ravel().astype(np.int64) * n + pred.ravel(), minlength=256 * n).reshape(256, n)
        pr_h += np.bincount(pred[ys, xs].ravel(), minlength=n)[:n]
        if j in vis:
            gt = lut[tags]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            over = (0.5 * rgb + 0.5 * colour(gt, names)).astype(np.uint8)
            grid = np.vstack([np.hstack([rgb, colour(gt, names)]), np.hstack([colour(pred, names), over])])
            cv2.imwrite(os.path.join(out_dir, f'{os.path.basename(d)}_{fid}.jpg'),
                        cv2.cvtColor(cv2.resize(grid, (W, H)), cv2.COLOR_RGB2BGR))
    tot = tag_pred.sum()
    print(f'\nwhat the model predicts on each CARLA tag ({len(pick)} frames; for a model trained elsewhere this shows '
          f'the mapping it learned, ignored tags included):')
    for t in np.argsort(-tag_pred.sum(1)):
        row = tag_pred[t]
        if row.sum() < 0.002 * tot:
            break
        cls = names[lut[t]] if lut[t] != IGNORE else 'ignored'
        top = np.argsort(-row)[:2]
        print(f'    {TAG_NAMES[t] if t < len(TAG_NAMES) else t:<14}{100 * row.sum() / tot:5.1f}% of pixels, GT class '
              f'{cls:<14}-> ' + ', '.join(f'{names[k]} {100 * row[k] / row.sum():.0f}%' for k in top if row[k] > 0))
    gs, ps = gt_h / max(gt_h.sum(), 1), pr_h / max(pr_h.sum(), 1)
    print('\nright in front of the car (bottom centre of the image, should be the ego road):')
    print('    ground truth (CARLA tag): ' + ', '.join(
        f'{TAG_NAMES[t] if t < len(TAG_NAMES) else t} {100 * gs[t]:.0f}%' for t in np.argsort(-gs)[:4] if gs[t] > 0.01))
    print('    model prediction:         ' + ', '.join(
        f'{names[k]} {100 * ps[k]:.0f}%' for k in np.argsort(-ps)[:4] if ps[k] > 0.01))
    if vis:
        print(f'    pictures: {out_dir}  (top: RGB | ground truth; bottom: prediction | ground truth over RGB)')


def confusion_lines(names, r):
    conf = np.array(r['conf'], float)
    print('\nwhere the pixels of each ground-truth class went (top 3 predictions):')
    for k, nm in enumerate(names):
        row = conf[k]
        if row.sum() == 0 or r['gt_share'][k] < 0.002:
            continue
        top = np.argsort(-row)[:3]
        print(f'    {nm:<16}-> ' + ', '.join(f'{names[t]} {100 * row[t] / row.sum():.0f}%' for t in top if row[t] > 0))


def show(names, r, title, warn_low=False):
    print(f"\n{title}: mIoU {100 * r['miou']:.1f}  pixel acc {100 * r['pixel_acc']:.1f}  ({r['frames']} frames)")
    for k, n in enumerate(names):
        if r['gt_share'][k] > 0:
            iou = r['iou'][k]
            warn = ('   <-- low for a common class: check its mapping' if warn_low and iou < 0.2
                    and r['gt_share'][k] > 0.005 else '')
            print(f'    {n:<16}{100 * iou:6.1f}   ({100 * r["gt_share"][k]:.1f}% of pixels){warn}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--seg-model', default=os.path.join(HERE, 'runs', 'seg', 'best'))
    ap.add_argument('--out', default=os.path.join(HERE, 'runs', 'seg', 'best_pose'))
    ap.add_argument('--train-towns', nargs='+', default=['Town04'])
    ap.add_argument('--val-towns', nargs='+', default=['Town01'])
    ap.add_argument('--img-size', type=int, nargs=2, default=[384, 768], metavar=('H', 'W'),
                    help='must stay what the car uses (perception_scheduler --seg-img-size)')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=6e-5)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--val-frames', type=int, default=300, help='frames per epoch check (0 = all); final = all')
    ap.add_argument('--check', action='store_true', help='mapping, pixel shares, BEFORE IoU; no training')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--vis', type=int, default=0, help='with --eval-only: save N pictures, print the diagnostics')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_segformer(args.seg_model).to(dev)
    names = [model.config.id2label[k] for k in range(model.config.num_labels)]
    lut = tag_lut(names)
    tr, va = frames(args.data_root, args.train_towns), frames(args.data_root, args.val_towns)
    print(f'{args.seg_model}: {len(names)} classes on {dev}; train {len(tr)} frames {args.train_towns}, '
          f'val {len(va)} frames {args.val_towns}')
    if not va:
        raise SystemExit('no validation frames (--val-towns)')
    hw, n = tuple(args.img_size), len(names)

    if args.check:
        print('\nclass <- CARLA tags (ignored tags: no class)')
        for k, nm in enumerate(names):
            tags = [TAG_NAMES[t] if t < len(TAG_NAMES) else str(t) for t in range(256) if lut[t] == k]
            print(f'    {k:>2} {nm:<16}<- {", ".join(tags) or "NOTHING (no tag maps here: never in the GT)"}')
        hist = np.zeros(256)
        for d, fid in [tr[i] for i in np.linspace(0, len(tr) - 1, min(len(tr), 100)).astype(int)] if tr else []:
            hist += np.bincount(read_tags(os.path.join(d, 'semantic', fid + '.png')).ravel(), minlength=256)
        if hist.sum():
            ign = [(TAG_NAMES[t] if t < len(TAG_NAMES) else str(t), hist[t] / hist.sum()) for t in range(256)
                   if hist[t] and lut[t] == IGNORE]
            print('    ignored pixels (train sample): ' + (', '.join(f'{a} {100 * b:.1f}%' for a, b in ign) or 'none'))
        show(names, evaluate(model, va, lut, hw, n, dev, args.batch, args.workers, args.val_frames),
             f'BEFORE, {args.seg_model} on val', warn_low=True)
        return
    if args.eval_only:
        r = evaluate(model, va, lut, hw, n, dev, args.batch, args.workers, 0)
        show(names, r, f'{args.seg_model} on val (all frames)')
        confusion_lines(names, r)
        if args.vis:
            diagnose(model, va, lut, hw, names, dev,
                     os.path.join(args.seg_model, 'vis_' + '_'.join(os.path.basename(t) for t in args.val_towns)),
                     args.vis)
        with open(os.path.join(args.seg_model, f'eval_{"_".join(args.val_towns)}.json'), 'w') as f:
            json.dump(dict(r, names=names), f, indent=1)
        return
    if not tr:
        raise SystemExit('no training frames (--train-towns)')

    dl = DataLoader(SegData(tr, lut, hw, True), args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total, warm = args.epochs * len(dl), min(300, args.epochs * len(dl) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = (torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda') if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=dev.type == 'cuda'))
    r0 = evaluate(model, va, lut, hw, n, dev, args.batch, args.workers, args.val_frames)
    print(f'epoch 0 (before): val mIoU {100 * r0["miou"]:.1f}')
    best, log = r0['miou'], [dict(epoch=0, miou=r0['miou'])]
    os.makedirs(args.out, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for it, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                loss = model(pixel_values=x, labels=y).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if it % 50 == 0:
                print(f'  epoch {ep} {it}/{len(dl)} loss {loss.item():.4f}', end='\r', flush=True)
        r = evaluate(model, va, lut, hw, n, dev, args.batch, args.workers, args.val_frames)
        log.append(dict(epoch=ep, loss=run / len(dl), miou=r['miou']))
        mark = ''
        if r['miou'] > best:
            best, mark = r['miou'], '  saved (best)'
            model.save_pretrained(args.out)
        print(f'epoch {ep}: loss {run / len(dl):.4f}  val mIoU {100 * r["miou"]:.1f}  ({time.time() - t0:.0f}s){mark}')
        with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
            json.dump(dict(log=log, init=args.seg_model, names=names, img_size=list(hw),
                           train_towns=args.train_towns, val_towns=args.val_towns), f, indent=1)
    if best <= r0['miou']:
        print(f'\nno epoch beat the starting model ({100 * r0["miou"]:.1f}): nothing saved as a model in {args.out}')
        return
    best_model = load_segformer(args.out).to(dev)
    show(names, evaluate(best_model, va, lut, hw, n, dev, args.batch, args.workers, 0), f'AFTER, {args.out}, all val frames')


if __name__ == '__main__':
    main()