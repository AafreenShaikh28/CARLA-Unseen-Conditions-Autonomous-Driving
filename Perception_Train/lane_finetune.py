"""
lane_finetune.py - fine-tune your lane model (DeepLabV3-MobileNetV3 with 1 output channel: the build of
Driving\\perception_scheduler.py LaneNet / lane_validate.py) on the camera collector's frames.

Same input as the car: RGB resized to the checkpoint's img_size (384 x 768), /255, ImageNet mean / std;
output = lane probability per pixel (sigmoid). The checkpoint it writes has the same keys ('model',
'img_size'), so perception_scheduler --lane-ckpt loads it unchanged.

Data, augmentation and loss are the same as your lane_train.py (no mirror, brightness / contrast jitter,
BCE with pos_weight 10 + dice, GT resized by area then > 0.25), and the strict scores are computed exactly
like its evaluate(): P / R / F1 / IoU are directly comparable with lane_train.py's epoch lines.

Ground truth, --gt:
  roadline  the painted lane markings: CARLA tag 24 (RoadLines) in semantic\\<id>.png (what lane_train.py uses)
  centers   the lane CENTRE lines in lanes\\<id>.json (ego / left / right lane), drawn --line-px wide
These are different targets (markings vs centres). --check scores your current model against both: train on
the one it already agrees with, otherwise the fine-tune changes what the model predicts.

    python lane_finetune.py --lane-ckpt <your lane best.pt> --data-root <...\\dataset_pose> --check
    python lane_finetune.py --lane-ckpt <your lane best.pt> --data-root <...\\dataset_pose> --gt roadline
    python lane_finetune.py --lane-ckpt runs_lane\\pose\\best.pt --data-root <...\\dataset_pose> --eval-only --gt roadline

Scores (val, threshold 0.5, at the model's resolution): pixel precision / recall / F1, strict and within
3 px (a predicted pixel counts if a GT pixel is within 3 px, and the other way round). The best epoch by
F1 within 3 px is saved. Split by town (default Town04 train, Town01 val, like the other models).
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
ROADLINE = 24
TOL_PX = 3


def build(ckpt_path):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    try:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:                                            # torch < 1.13
        ck = torch.load(ckpt_path, map_location='cpu')
    m = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None)
    m.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    m.aux_classifier = None
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith('aux_classifier.')})
    hw = tuple(ck.get('img_size', [384, 768])) if isinstance(ck, dict) else (384, 768)
    return m, hw, ck if isinstance(ck, dict) else {}


def frames(root, towns, need_json):
    out = []
    for t in towns:
        d = os.path.join(root, t)
        if not os.path.isdir(d):
            raise SystemExit(f'{d} not found')
        for p in sorted(glob.glob(os.path.join(d, 'metadata', '*.json'))):     # metadata = written last
            fid = os.path.basename(p)[:-5]
            ok = all(os.path.isfile(os.path.join(d, s, fid + '.png')) for s in ('image_2', 'semantic'))
            if ok and (not need_json or os.path.isfile(os.path.join(d, 'lanes', fid + '.json'))):
                out.append((d, fid))
    return out


def gt_mask(d, fid, gt, line_px, shape):
    """binary lane mask at the camera's resolution"""
    if gt == 'roadline':
        sem = cv2.imread(os.path.join(d, 'semantic', fid + '.png'), cv2.IMREAD_UNCHANGED)
        tag = sem[:, :, 2] if sem.ndim == 3 else sem               # CARLA tag in the red channel (BGR load)
        return (tag == ROADLINE).astype(np.uint8)
    m = np.zeros(shape, np.uint8)
    for lane in json.load(open(os.path.join(d, 'lanes', fid + '.json'))):
        run = []
        for p in lane.get('image_points', []) + [None]:            # a None (behind the camera) ends a run
            if p is None or not np.isfinite(p).all():
                if len(run) >= 2:
                    cv2.polylines(m, [np.round(np.array(run)).astype(np.int32)], False, 1, line_px)
                run = []
            else:
                run.append(p)
    return m


def to_model_res(mask, hw):
    """thin lines survive the resize: area average, then > 0.25 (as lane_train.py)"""
    h, w = hw
    return (cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.25).astype(np.float32)


class LaneData(Dataset):
    def __init__(self, items, hw, gt, line_px, train):
        self.items, self.hw, self.gt, self.line_px, self.train = items, hw, gt, line_px, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, fid = self.items[i]
        bgr = cv2.imread(os.path.join(d, 'image_2', fid + '.png'))
        y = to_model_res(gt_mask(d, fid, self.gt, self.line_px, bgr.shape[:2]), self.hw)
        h, w = self.hw
        x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LINEAR)
        x = x.astype(np.float32) / 255.0
        if self.train:                                                  # as lane_train.py: no mirror
            x = np.clip((x - 0.5) * np.random.uniform(0.8, 1.2) + 0.5 + np.random.uniform(-0.1, 0.1), 0, 1)
        x = (x - MEAN) / STD
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y))[None])


def evaluate(model, items, hw, gt, line_px, dev, batch, workers, limit=0):
    if limit and len(items) > limit:
        items = [items[i] for i in np.linspace(0, len(items) - 1, limit).astype(int)]
    dl = DataLoader(LaneData(items, hw, gt, line_px, False), batch, num_workers=workers)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL_PX + 1, 2 * TOL_PX + 1))
    s = dict(tp=0, p=0, g=0, p_near=0, g_near=0)
    model.eval()
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(dev))['out']                         # fp32, as lane_train.py's evaluate()
            pred = (torch.sigmoid(out.float()) > 0.5).cpu().numpy()[:, 0].astype(np.uint8)
            y = y.numpy()[:, 0].astype(np.uint8)
            for pm, gm in zip(pred, y):
                s['tp'] += int((pm & gm).sum())
                s['p'] += int(pm.sum())
                s['g'] += int(gm.sum())
                s['p_near'] += int((pm & cv2.dilate(gm, k)).sum())
                s['g_near'] += int((gm & cv2.dilate(pm, k)).sum())

    def f1(p, r):
        return 2 * p * r / max(p + r, 1e-9)
    P, R = s['tp'] / max(s['p'], 1), s['tp'] / max(s['g'], 1)
    Pt, Rt = s['p_near'] / max(s['p'], 1), s['g_near'] / max(s['g'], 1)
    n_px = len(items) * hw[0] * hw[1]
    return dict(P=P, R=R, F1=f1(P, R), IoU=s['tp'] / max(s['p'] + s['g'] - s['tp'], 1), P_3px=Pt, R_3px=Rt, F1_3px=f1(Pt, Rt), frames=len(items),
                gt_share=s['g'] / max(n_px, 1), pred_share=s['p'] / max(n_px, 1))


def line(r):
    return (f"F1 {100 * r['F1']:.1f} (P {100 * r['P']:.1f}, R {100 * r['R']:.1f}, IoU {100 * r['IoU']:.1f})   within {TOL_PX} px: "
            f"F1 {100 * r['F1_3px']:.1f} (P {100 * r['P_3px']:.1f}, R {100 * r['R_3px']:.1f})   "
            f"lane pixels GT {100 * r['gt_share']:.2f}% / predicted {100 * r['pred_share']:.2f}%  ({r['frames']} frames)")


def freeze_bn(model):
    """BatchNorm keeps the trained model's running statistics (its scale / shift still learn): with
    fine-tuning batches the statistics would drift away from what the trained layers expect"""
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()


def dice_bce(logits, y, pos_weight=10.0):
    """as lane_train.py: BCE with lane pixels weighted pos_weight (they are rare) + dice"""
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor([pos_weight], device=logits.device))
    p = torch.sigmoid(logits)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lane-ckpt', required=True, help='your trained lane model (.pt with "model", "img_size")')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--gt', choices=['roadline', 'centers'], default='roadline')
    ap.add_argument('--line-px', type=int, default=8, help='--gt centers: line width at 800 x 600')
    ap.add_argument('--out-dir', default=os.path.join(HERE, 'runs_lane', 'pose'))
    ap.add_argument('--train-towns', nargs='+', default=['Town04'])
    ap.add_argument('--val-towns', nargs='+', default=['Town01'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--val-frames', type=int, default=300, help='frames per epoch check (0 = all); final = all')
    ap.add_argument('--check', action='store_true', help='score the current model against both GT kinds')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--pos-weight', type=float, default=10.0, help='as lane_train.py')
    ap.add_argument('--bn-train', action='store_true', help='also update BatchNorm statistics (default: kept)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, hw, ck = build(args.lane_ckpt)
    model = model.to(dev)
    print(f'{args.lane_ckpt}: input {hw[0]} x {hw[1]}' + (f", trained on gt '{ck['gt']}'" if ck.get('gt') else '')
          + f' ({dev})')
    need_json = args.check or args.gt == 'centers'
    tr, va = frames(args.data_root, args.train_towns, need_json), frames(args.data_root, args.val_towns, need_json)
    print(f'train {len(tr)} frames {args.train_towns}, val {len(va)} frames {args.val_towns}')
    if not va:
        raise SystemExit('no validation frames (--val-towns)')

    if args.check:
        res = {}
        for g in ('roadline', 'centers'):
            res[g] = evaluate(model, va, hw, g, args.line_px, dev, args.batch, args.workers, args.val_frames)
            print(f'\nBEFORE vs --gt {g}:\n    {line(res[g])}')
        best = max(res, key=lambda g: res[g]['F1_3px'])
        if res[best]['F1_3px'] < 0.3:
            print(f'\nneither matches (best F1 within {TOL_PX} px {100 * res[best]["F1_3px"]:.1f}): your model was '
                  f'trained on another lane definition. Do not fine-tune on these until that is known '
                  f'(send the lane training / lane_validate.py script).')
        else:
            print(f'\nyour model agrees most with --gt {best} (F1 within {TOL_PX} px '
                  f'{100 * res[best]["F1_3px"]:.1f}): fine-tune with --gt {best}')
        return
    if args.eval_only:
        r = evaluate(model, va, hw, args.gt, args.line_px, dev, args.batch, args.workers, 0)
        print(f'\n{args.lane_ckpt} vs --gt {args.gt}, all val frames:\n    {line(r)}')
        return
    if not tr:
        raise SystemExit('no training frames (--train-towns)')

    dl = DataLoader(LaneData(tr, hw, args.gt, args.line_px, True), args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True, persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total, warm = args.epochs * len(dl), min(300, args.epochs * len(dl) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = (torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda') if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=dev.type == 'cuda'))
    r0 = evaluate(model, va, hw, args.gt, args.line_px, dev, args.batch, args.workers, args.val_frames)
    print(f'epoch 0 (before): {line(r0)}')
    best, log, saved = r0['F1_3px'], [dict(epoch=0, **r0)], False
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'best.pt')
    for ep in range(1, args.epochs + 1):
        model.train()
        if not args.bn_train:
            freeze_bn(model)
        t0, run = time.time(), 0.0
        for it, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                logits = model(x)['out']
            loss = dice_bce(logits.float(), y, args.pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            if it % 50 == 0:
                print(f'  epoch {ep} {it}/{len(dl)} loss {loss.item():.4f}', end='\r', flush=True)
        r = evaluate(model, va, hw, args.gt, args.line_px, dev, args.batch, args.workers, args.val_frames)
        log.append(dict(epoch=ep, loss=run / len(dl), **r))
        mark = ''
        if r['F1_3px'] > best:
            best, mark, saved = r['F1_3px'], '  saved (best)', True
            torch.save({'model': model.state_dict(), 'img_size': list(hw), 'gt': args.gt, 'epoch': ep,
                        'val': r, 'init': os.path.abspath(args.lane_ckpt),
                        'train_towns': args.train_towns, 'val_towns': args.val_towns}, out)
        print(f'epoch {ep}: loss {run / len(dl):.4f} ({time.time() - t0:.0f}s)  {line(r)}{mark}')
        with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
            json.dump(log, f, indent=1)
    if not saved:
        print(f'\nno epoch beat the starting model (F1 within {TOL_PX} px {100 * r0["F1_3px"]:.1f}): nothing saved')
        return
    m2, _, _ = build(out)
    r = evaluate(m2.to(dev), va, hw, args.gt, args.line_px, dev, args.batch, args.workers, 0)
    print(f'\nAFTER, {out}, all val frames:\n    {line(r)}')


if __name__ == '__main__':
    main()