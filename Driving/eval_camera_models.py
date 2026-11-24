"""
eval_camera_models.py - offline metrics of the two camera models, per test set (e.g. per weather):
  YOLO        AP50, AP50-95, precision / recall per class, mAP   (vs the 2D boxes in label_2)
  SegFormer   IoU per class, mIoU, pixel accuracy                 (vs CARLA's semantic camera)
The same model files and the same pre/post-processing the car uses (perception_scheduler.py).
Put it in the Driving folder.

    python eval_camera_models.py --seg-model <seg dir>                 # YOLO + SegFormer, 200 frames per set
    python eval_camera_models.py --seg-model <seg dir> --n 0           # every frame
    python eval_camera_models.py --only yolo                           # YOLO only (no SegFormer needed)

Test sets (default): every <Weather>/training under ..\\LIDAR_only\\kitti_weather (collect_weather_tests.py),
plus the original test split (kitti_carla, split_lidar.json 'val' = Town05 clear noon) as 'TestSplit'.
--sets NAME=PATH adds any other KITTI-layout training folder.

YOLO scoring (KITTI 'moderate' rules, COCO-style AP):
  - classes: car (Car + Van), truck, bus, person (Pedestrian), two-wheeler (Cyclist <-> YOLO bicycle /
    motorcycle), and 'vehicle' = car + truck + bus with the class ignored (what the planner needs:
    a van called 'truck' is still an obstacle).
  - scored boxes: height >= 25 px, occlusion <= 1, truncation <= 0.3. Other labelled boxes are 'ignore':
    a detection on them is neither right nor wrong, missing them costs nothing. Detections smaller
    than 25 px, inside DontCare regions, or a 'person' on a cyclist's rider are ignored too.
    With this camera (800 x 600, 90 deg FOV, focal 400 px) 25 px = a car within ~24 m, a pedestrian
    within ~29 m. --min-height 10 scores up to the 60 m label range.
  - GT boxes: label_2 holds the projected 3D box; YOLO draws tight boxes around the visible pixels.
    By default each GT box is tightened to the pixels of its class inside it (semantic camera, same
    pose and size as the RGB camera). The untouched projected boxes are scored too ('projected').
  - AP: 101-point interpolation (COCO), max 300 detections per image; AP50-95 = mean over IoU
    0.50:0.05:0.95. Precision / recall at the car's own threshold (0.25). mAP over classes with
    >= 20 scored boxes.
SegFormer scoring: argmax at full image size; CARLA tags mapped to the model's classes with the table of
Fusion/check_paint.py (most specific class wins); pixels with a tag no class covers are ignored.
mIoU over classes present in the ground truth of that set.

Writes runs_eval_camera/<set>/metrics.json, runs_eval_camera/camera_models.csv,
runs_eval_camera/yolo_by_set.png and runs_eval_camera/segformer_by_set.png.
"""
import os
import csv
import json
import types
import argparse
from collections import OrderedDict

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
KITTI_TO_EVAL = {'Car': 'car', 'Van': 'car', 'Truck': 'truck', 'Bus': 'bus', 'Pedestrian': 'person',
                 'Cyclist': 'two-wheeler'}
YOLO_TO_EVAL = {'car': 'car', 'truck': 'truck', 'bus': 'bus', 'person': 'person',
                'bicycle': 'two-wheeler', 'motorcycle': 'two-wheeler'}
EVAL_CLASSES = ['car', 'truck', 'bus', 'person', 'two-wheeler']
VEHICLES = ('car', 'truck', 'bus')
# for tightening GT boxes: the class's own tag first, then any vehicle tag (a van may be tagged truck).
# Two-wheeler = the bike only, like YOLO's COCO 'bicycle' / 'motorcycle' box; its rider is 'person'
CLASS_TAGS = {'car': [(14,), (14, 15, 16, 17)], 'truck': [(15,), (14, 15, 16, 17)],
              'bus': [(16,), (14, 15, 16, 17)], 'person': [(12,)], 'two-wheeler': [(18, 19)]}
MAX_OCC, MAX_TRUNC = 1, 0.30
OP_CONF = 0.25
MAX_DET = 300
MIN_MAP_GT = 20

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


def norm(name):
    return name.lower().replace(' ', '_').replace('-', '_')


def tag_lut(names):
    """256-entry table: CARLA tag -> model class index (255 = no class covers it: ignored).
    Same rule as check_paint.tag_to_class: the most specific class wins."""
    lut = np.full(256, 255, np.uint8)
    for k in sorted(range(len(names)), key=lambda k: len(NAME_TO_TAGS.get(norm(names[k]), ())) or 99):
        for t in NAME_TO_TAGS.get(norm(names[k]), ()):
            if lut[t] == 255:
                lut[t] = k
    return lut


# ------------------------------------------------------------------------- 2D boxes
def _overlap(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def _area(z):
    return (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])


def iou(a, b):
    """a [N,4], b [M,4] xyxy -> [N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    inter = _overlap(a, b)
    return inter / np.maximum(_area(a)[:, None] + _area(b)[None, :] - inter, 1e-9)


def ioa(a, b):
    """share of each a box that lies inside each b box -> [N,M]"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return _overlap(a, b) / np.maximum(_area(a)[:, None], 1e-9)


def tighten(box, cls, sem, min_px=10):
    """shrink a projected 3D box to the pixels of its class inside it; unchanged if < min_px of them"""
    if sem is None:
        return box
    h, w = sem.shape
    x1, y1 = int(max(np.floor(box[0]), 0)), int(max(np.floor(box[1]), 0))
    x2, y2 = int(min(np.ceil(box[2]), w)), int(min(np.ceil(box[3]), h))
    if x2 <= x1 or y2 <= y1:
        return box
    for tags in CLASS_TAGS[cls]:
        m = np.isin(sem[y1:y2, x1:x2], tags)
        if m.sum() >= min_px:
            ys, xs = np.nonzero(m)
            return [x1 + xs.min(), y1 + ys.min(), x1 + xs.max() + 1.0, y1 + ys.max() + 1.0]
    return box


def read_label2(path, sem=None, min_h=25.0):
    """-> {'projected': gt, 'tight': gt}; gt = dict(boxes, cls, ignore, dontcare, cyclists)"""
    rows, dontcare = [], []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                f = line.split()
                if len(f) < 8:
                    continue
                b = [float(v) for v in f[4:8]]
                if f[0] == 'DontCare':
                    dontcare.append(b)
                elif f[0] in KITTI_TO_EVAL:
                    rows.append((KITTI_TO_EVAL[f[0]], float(f[1]), int(float(f[2])), b))
    a = lambda x: np.asarray(x, np.float64).reshape(-1, 4)              # noqa: E731
    riders = a([b for c, _, _, b in rows if c == 'two-wheeler'])     # full box: bike + rider
    out = {}
    for kind in ('projected', 'tight'):
        boxes = [b if kind == 'projected' else tighten(b, c, sem) for c, _, _, b in rows]
        ign = [(bb[3] - bb[1] < min_h) or occ > MAX_OCC or tr > MAX_TRUNC
               for (c, tr, occ, _), bb in zip(rows, boxes)]
        out[kind] = dict(boxes=a(boxes), cls=np.asarray([r[0] for r in rows], dtype='<U12'), ignore=np.asarray(ign, bool),
                         dontcare=a(dontcare), cyclists=riders)
    return out


def as_vehicle(gt, det):
    """car / truck / bus -> one class 'vehicle' (class-agnostic), duplicates across classes removed"""
    g = dict(gt)
    g['cls'] = np.where(np.isin(gt['cls'], VEHICLES), 'vehicle', gt['cls'])
    b, c, s = det
    m = np.isin(c, VEHICLES)
    vb, vs = b[m], s[m]
    if len(vb):
        keep = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in vb],
                                vs.astype(float).tolist(), 0.0, 0.7)
        keep = np.asarray(keep, int).reshape(-1)
        vb, vs = vb[keep], vs[keep]
    return g, (np.concatenate([b[~m], vb]), np.concatenate([c[~m], np.full(len(vb), 'vehicle', dtype='<U12')]),
               np.concatenate([s[~m], vs]))


def ap101(rec, prec):
    """COCO 101-point interpolated AP, in %"""
    if len(rec) == 0:
        return 0.0
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    mrec = np.concatenate([[0.0], rec, [1.0]])
    idx = np.searchsorted(mrec, np.linspace(0, 1, 101), side='left')
    return 100.0 * float(np.mean(mpre[np.minimum(idx, len(mpre) - 1)]))


def score_class(c, frames, thr, min_h):
    """frames: list of (gt, (boxes, cls, scores)). -> dict(AP, precision, recall at OP_CONF, gt) or None"""
    recs, n_pos = [], 0
    for gt, (db, dc, ds) in frames:
        m = gt['cls'] == c
        g_boxes, g_ign = gt['boxes'][m], gt['ignore'][m]
        n_pos += int((~g_ign).sum())
        d = (dc == c) & ((db[:, 3] - db[:, 1]) >= min_h)       # KITTI: too-small detections are ignored
        b, s = db[d], ds[d]
        if len(b) == 0:
            continue
        order = np.argsort(-s, kind='stable')
        b, s = b[order], s[order]
        ov = iou(b, g_boxes)
        used = np.zeros(len(g_boxes), bool)
        extra = ioa(b, gt['dontcare'])
        if c == 'person' and len(gt['cyclists']):
            extra = np.concatenate([extra, ioa(b, gt['cyclists'])], axis=1)
        for i in range(len(b)):
            free = (~g_ign) & (~used) & (ov[i] >= thr)               # real boxes first
            if free.any():
                j = int(np.argmax(np.where(free, ov[i], -1.0)))
                used[j] = True
                recs.append((s[i], 1))
            elif not ((g_ign & (ov[i] >= thr)).any() or (extra.shape[1] and extra[i].max() >= 0.5)):
                recs.append((s[i], 0))
    if n_pos == 0:
        return None
    recs.sort(key=lambda r: -r[0])
    hit = np.array([r[1] for r in recs], float)
    tp, fp = np.cumsum(hit), np.cumsum(1.0 - hit)
    rec, prec = tp / n_pos, tp / np.maximum(tp + fp, 1e-9)
    k = sum(1 for r in recs if r[0] >= OP_CONF)
    return dict(AP=ap101(rec, prec), precision=100.0 * prec[k - 1] if k else 0.0,
                recall=100.0 * rec[k - 1] if k else 0.0, gt=n_pos)


def yolo_metrics(frames, min_h):
    """frames: list of (gt, det). -> OrderedDict per class + 'vehicle' + 'mAP'"""
    out = OrderedDict()
    veh = [as_vehicle(g, d) for g, d in frames]
    for c in EVAL_CLASSES + ['vehicle']:
        fr = veh if c == 'vehicle' else frames
        r50 = score_class(c, fr, 0.5, min_h)
        if r50 is None:
            out[c] = dict(gt=0)
            continue
        ap5095 = float(np.mean([score_class(c, fr, t, min_h)['AP'] for t in np.arange(0.5, 0.951, 0.05)]))
        out[c] = dict(gt=r50['gt'], AP50=r50['AP'], AP50_95=ap5095, precision=r50['precision'],
                      recall=r50['recall'])
    set_map(out, [c for c in EVAL_CLASSES if out[c].get('gt', 0) >= MIN_MAP_GT])
    return out


def set_map(y, classes):
    """mAP = mean AP over these classes (the same list for every set, or sets are not comparable)"""
    used = [c for c in classes if y[c].get('gt', 0) > 0]
    y['mAP'] = dict(classes=used,
                    AP50=float(np.mean([y[c]['AP50'] for c in used])) if used else None,
                    AP50_95=float(np.mean([y[c]['AP50_95'] for c in used])) if used else None)


def seg_metrics(conf, names):
    """confusion matrix [gt, pred] -> IoU / accuracy per class present, mIoU, pixel accuracy"""
    tp = np.diag(conf).astype(np.float64)
    gt = conf.sum(1).astype(np.float64)
    pr = conf.sum(0).astype(np.float64)
    out = OrderedDict()
    present = [k for k in range(len(names)) if gt[k] > 0]
    for k in present:
        out[names[k]] = dict(IoU=100.0 * tp[k] / max(gt[k] + pr[k] - tp[k], 1.0),
                             accuracy=100.0 * tp[k] / gt[k], pixels=int(gt[k]))
    out['mIoU'] = float(np.mean([out[names[k]]['IoU'] for k in present])) if present else None
    out['pixel_accuracy'] = 100.0 * tp.sum() / max(conf.sum(), 1.0)
    return out


# ------------------------------------------------------------------------- test sets
def list_ids(root):
    d = os.path.join(root, 'image_2')
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith('.png'))


def find_sets(args):
    sets = OrderedDict()
    sp = os.path.join(args.clear_split, 'split_lidar.json')
    if os.path.exists(sp):
        with open(sp) as fh:
            sets['TestSplit'] = (args.clear_split, list(json.load(fh)['val']))
    if os.path.isdir(args.weather_root):
        for w in sorted(os.listdir(args.weather_root)):
            tr = os.path.join(args.weather_root, w, 'training')
            if os.path.isdir(os.path.join(tr, 'image_2')):
                sets[w] = (tr, list_ids(tr))
    for item in args.sets:
        name, _, path = item.partition('=')
        if not path or not os.path.isdir(os.path.join(path, 'image_2')):
            raise SystemExit(f'--sets {item}: expected NAME=<folder with image_2>')
        sets[name] = (path, list_ids(path))
    return sets


def evenly(ids, n):
    if n <= 0 or len(ids) <= n:
        return list(ids)
    return [ids[int(round(i))] for i in np.linspace(0, len(ids) - 1, n)]


# ------------------------------------------------------------------------- charts
ORDER = ['TestSplit', 'ClearNoon', 'CloudySunset', 'HardRainNoon', 'ClearNight', 'HardRainNight', 'Fog']
SHORT = {'TestSplit': 'Test split\n(clear)', 'ClearNoon': 'Clear noon', 'CloudySunset': 'Sunset',
         'HardRainNoon': 'Heavy rain', 'ClearNight': 'Night', 'HardRainNight': 'Rain + night', 'Fog': 'Fog'}
COLOR, INK, INK2, GRID, SURF = '#6250d6', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def chart(rows, panels, title, path, note):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 14, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                         'axes.labelcolor': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                         'savefig.facecolor': SURF})
    names = sorted(rows, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))
    panels = [p for p in panels if any(rows[s].get(p[0]) is not None for s in names)]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 9), squeeze=False)
    for ax, (key, sub) in zip(axes[0], panels):
        vals = [rows[s].get(key) for s in names]
        xs = [i for i, v in enumerate(vals) if v is not None]
        ax.bar(xs, [vals[i] for i in xs], 0.65, color=COLOR, edgecolor=SURF, linewidth=2)
        for i in xs:
            ax.text(i, vals[i] + 1, f'{vals[i]:.1f}', ha='center', va='bottom', fontsize=11, color=INK2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT.get(s, s) for s in names], rotation=30, ha='right', fontsize=12)
        ax.set_title(sub, loc='left', fontsize=15)
        ax.set_ylim(0, 108)
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel('%')
    fig.suptitle(title, x=0.02, y=0.985, ha='left', fontsize=20)
    fig.text(0.02, 0.015, note, fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.955))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------- main
def fmt(v, w=8):
    return f'{v:{w}.1f}' if isinstance(v, (int, float)) and v == v else f'{"-":>{w}}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--weather-root', default=os.path.join(CODE, 'LIDAR_only', 'kitti_weather'))
    ap.add_argument('--clear-split', default=os.path.join(CODE, 'LIDAR_only', 'kitti_carla', 'training'),
                    help="original dataset folder; its split_lidar.json 'val' ids become set 'TestSplit'")
    ap.add_argument('--sets', nargs='*', default=[], metavar='NAME=PATH', help='more KITTI training folders')
    ap.add_argument('--yolo', default=os.path.join(CODE, 'Detection_YOLO', 'runs', 'detect', 'runs', 'detect',
                                                   'carla_finetune-10', 'weights', 'best.onnx'))
    ap.add_argument('--yolo-provider', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--seg-model', default=os.path.join(CODE, 'Detection_YOLO', 'runs', 'seg', 'best'))
    ap.add_argument('--seg-img-size', type=int, nargs=2, default=[384, 768], metavar=('H', 'W'),
                    help='must equal the Driving stack / paint_points.py size')
    ap.add_argument('--n', type=int, default=200, help='frames per set, evenly spaced (0 = all)')
    ap.add_argument('--min-height', type=float, default=25.0, help='px; KITTI moderate = 25')
    ap.add_argument('--only', choices=['yolo', 'seg'], default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'runs_eval_camera'))
    args = ap.parse_args()

    sets = find_sets(args)
    if not sets:
        raise SystemExit(f'no test sets: nothing in {args.weather_root} and no '
                         f'{os.path.join(args.clear_split, "split_lidar.json")}')
    print('test sets: ' + ', '.join(f'{k} ({len(v[1])} frames)' for k, v in sets.items()))
    import perception_scheduler as ps

    yolo = seg = names = lut = dev = torch = None
    if args.only != 'seg':
        if not os.path.exists(args.yolo):
            raise SystemExit(f'YOLO model not found: {args.yolo} (pass --yolo)')
        yolo = ps.YoloOnnx(args.yolo, provider=args.yolo_provider, conf=0.001, iou=0.6)   # the whole PR curve
        print(f'YOLO: {args.yolo} ({yolo.active})')
    if args.only != 'yolo':
        if not os.path.isdir(args.seg_model):
            raise SystemExit(f'SegFormer folder not found: {args.seg_model} (pass --seg-model, or --only yolo)')
        import torch
        dev = ps.get_device()
        seg, _ = ps.load_segformer(args.seg_model)
        seg = seg.to(dev).eval()
        names = [seg.config.id2label[i] for i in range(len(seg.config.id2label))]
        lut = tag_lut(names)
        unmapped = [n for n in names if norm(n) not in NAME_TO_TAGS]
        note = f'; never in the ground truth: {unmapped}' if unmapped else ''
        print(f'SegFormer: {len(names)} classes on {dev}{note}')

    os.makedirs(args.out, exist_ok=True)
    table, no_sem = OrderedDict(), []
    for name, (root, all_ids) in sets.items():
        ids = evenly(all_ids, args.n)
        frames = {'tight': [], 'projected': []}
        conf = np.zeros((len(names), len(names)), np.int64) if seg is not None else None
        n_sem = 0
        for i, fid in enumerate(ids):
            bgr = cv2.imread(os.path.join(root, 'image_2', fid + '.png'))
            if bgr is None:
                continue
            sp = os.path.join(root, 'semantic', fid + '.png')
            sem = cv2.imread(sp, cv2.IMREAD_UNCHANGED) if os.path.exists(sp) else None
            if sem is not None and sem.ndim == 3:
                sem = sem[:, :, 2]                                   # CARLA tag in the red channel (BGR load)
            if sem is not None and sem.shape != bgr.shape[:2]:
                sem = None
            if yolo is not None:
                pred, _ = yolo(types.SimpleNamespace(bgr=bgr))
                ev = np.array([YOLO_TO_EVAL.get(ps.DRIVING_YOLO_CLASSES.get(int(c), ''), '') for c in pred['classes']],
                              dtype='<U12')
                m = ev != ''
                b, c, s = pred['boxes'][m].astype(np.float64).reshape(-1, 4), ev[m], pred['scores'][m]
                top = np.argsort(-s, kind='stable')[:MAX_DET]
                det = (b[top], c[top], s[top])
                gts = read_label2(os.path.join(root, 'label_2', fid + '.txt'), sem, args.min_height)
                for kind in frames:
                    frames[kind].append((gts[kind], det))
            if seg is not None and sem is not None:
                gt = lut[sem]
                h_in, w_in = args.seg_img_size
                x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w_in, h_in), interpolation=cv2.INTER_LINEAR)
                x = (x.astype(np.float32) / 255.0 - ps.IMAGENET_MEAN) / ps.IMAGENET_STD
                with torch.no_grad():
                    lg = seg(pixel_values=torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(dev)).logits
                    lg = torch.nn.functional.interpolate(lg.float(), size=gt.shape, mode='bilinear',
                                                         align_corners=False)
                    pr = lg.argmax(1)[0].cpu().numpy().astype(np.int64)
                ok = gt != 255
                conf += np.bincount(gt[ok].astype(np.int64) * len(names) + pr[ok],
                                    minlength=len(names) ** 2).reshape(len(names), len(names))
                n_sem += 1
            if sem is None and name not in no_sem:
                no_sem.append(name)
            if (i + 1) % 50 == 0:
                print(f'  {name}: {i + 1}/{len(ids)}')
        res = OrderedDict(set=name, root=root, frames=len(ids), min_height_px=args.min_height)
        if yolo is not None:
            res['yolo'] = yolo_metrics(frames['tight'], args.min_height)
            res['yolo_projected_gt'] = yolo_metrics(frames['projected'], args.min_height)
        if seg is not None:
            res['segformer'] = seg_metrics(conf, names) if n_sem else None
            res['segformer_frames'] = n_sem
        table[name] = res
        y, s = res.get('yolo'), res.get('segformer')
        print(f'{name:<14}' + (f" YOLO vehicle AP50 {fmt(y['vehicle'].get('AP50'), 5)}  pedestrian AP50 "
                               f"{fmt(y['person'].get('AP50'), 5)}" if y else '')
              + (f"   SegFormer mIoU {fmt(s['mIoU'], 5)}" if s else ''))
    if no_sem:
        print(f'\n[note] no semantic/ images in {no_sem}: GT boxes there are not tightened'
              + (' and SegFormer is not scored' if seg is not None else ''))
    if yolo is not None:            # one mAP class list for all sets: classes with >= 20 boxes in every set
        common = [c for c in EVAL_CLASSES if all(r['yolo'][c].get('gt', 0) >= MIN_MAP_GT for r in table.values())]
        if not common:
            common = [c for c in EVAL_CLASSES if all(r['yolo'][c].get('gt', 0) > 0 for r in table.values())]
            print(f'[note] no class has >= {MIN_MAP_GT} scored boxes in every set (use more frames, --n): '
                  f'mAP over {common}')
        for r in table.values():
            set_map(r['yolo'], common)
            set_map(r['yolo_projected_gt'], common)
    for name, res in table.items():
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
        with open(os.path.join(args.out, name, 'metrics.json'), 'w') as fh:
            json.dump(res, fh, indent=2)

    # ---- flat table, CSV, console
    flat = OrderedDict()
    for name, res in table.items():
        r = OrderedDict(set=name, frames=res['frames'])
        y = res.get('yolo')
        if y:
            r['yolo_mAP50'], r['yolo_mAP50_95'] = y['mAP']['AP50'], y['mAP']['AP50_95']
            r['yolo_mAP50_projected_gt'] = res['yolo_projected_gt']['mAP']['AP50']
            for c in EVAL_CLASSES + ['vehicle']:
                for k in ('AP50', 'AP50_95', 'precision', 'recall', 'gt'):
                    r[f'yolo_{c}_{k}'] = y[c].get(k)
        s = res.get('segformer')
        if s:
            r['seg_mIoU'], r['seg_pixel_acc'] = s['mIoU'], s['pixel_accuracy']
            for n in names:
                if n in s:
                    r[f'seg_{norm(n)}_IoU'] = s[n]['IoU']
        flat[name] = r
    cols = []
    for r in flat.values():
        cols += [c for c in r if c not in cols]
    out_csv = os.path.join(args.out, 'camera_models.csv')
    try:
        fh = open(out_csv, 'w', newline='')
    except PermissionError:
        out_csv = out_csv[:-4] + '_new.csv'
        print(f'[note] camera_models.csv is open in another program: writing {out_csv}')
        fh = open(out_csv, 'w', newline='')
    with fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(flat.values())

    order = sorted(flat, key=lambda s: (ORDER.index(s) if s in ORDER else 99, s))
    if yolo is not None:
        print(f"\nYOLO (AP50 %, IoU 0.5; boxes >= {args.min_height:.0f} px)\n{'set':<14}{'mAP50':>8}{'mAP50-95':>10}"
              f"{'vehicle':>9}{'car':>8}{'truck':>8}{'bus':>8}{'person':>8}{'2-wheel':>9}"
              f"{'P@0.25':>8}{'R@0.25':>8}{'proj.GT':>9}")
        for n in order:
            r = flat[n]
            print(f"{n:<14}{fmt(r['yolo_mAP50'])}{fmt(r['yolo_mAP50_95'], 10)}{fmt(r['yolo_vehicle_AP50'], 9)}"
                  + ''.join(fmt(r[f'yolo_{c}_AP50']) for c in ('car', 'truck', 'bus', 'person'))
                  + fmt(r['yolo_two-wheeler_AP50'], 9) + fmt(r['yolo_vehicle_precision'])
                  + fmt(r['yolo_vehicle_recall']) + fmt(r['yolo_mAP50_projected_gt'], 9))
        print(f"  mAP over {', '.join(common)}. P / R = vehicle precision / recall at the car's threshold 0.25. "
              f"proj.GT = mAP50 against the untightened projected 3D boxes. Box counts in the CSV.")
    if seg is not None:
        key_cls = [n for n in names if norm(n) in ('road', 'sidewalk', 'road_line', 'roadline', 'lane_marking',
                                                   'vehicle', 'car', 'pedestrian', 'person', 'rider',
                                                   'traffic_light', 'traffic_sign', 'pole')]
        print(f"\nSegFormer (IoU %)\n{'set':<14}{'mIoU':>8}{'pix acc':>9}" + ''.join(f'{n[:9]:>10}' for n in key_cls))
        for n in order:
            r = flat[n]
            if 'seg_mIoU' in r:
                print(f"{n:<14}{fmt(r['seg_mIoU'])}{fmt(r['seg_pixel_acc'], 9)}"
                      + ''.join(fmt(r.get(f'seg_{norm(c)}_IoU'), 10) for c in key_cls))

    # ---- charts (one model per chart, one bar per set)
    made = []
    if yolo is not None:
        made.append(chart(flat, [('yolo_mAP50', 'mAP50 (all classes)'), ('yolo_vehicle_AP50', 'Vehicle AP50'),
                                 ('yolo_person_AP50', 'Pedestrian AP50'),
                                 ('yolo_two-wheeler_AP50', 'Two-wheeler AP50')],
                          'YOLO 2D detection by test set', os.path.join(args.out, 'yolo_by_set.png'),
                          f'IoU 0.5, COCO 101-point AP. Scored: boxes >= {args.min_height:.0f} px tall, occlusion '
                          '<= 1, truncation <= 0.3; DontCare regions ignored. Vehicle = car + truck + bus.'))
    if seg is not None:
        def key(*cands):
            for c in cands:
                if f'seg_{c}_IoU' in cols:
                    return f'seg_{c}_IoU'
            return None
        pan = [('seg_mIoU', 'mIoU')] + [(k, t) for k, t in ((key('road'), 'Road IoU'),
                                                            (key('vehicle', 'car'), 'Vehicle IoU'),
                                                            (key('pedestrian', 'person'), 'Pedestrian IoU')) if k]
        made.append(chart(flat, pan, 'SegFormer segmentation by test set',
                          os.path.join(args.out, 'segformer_by_set.png'),
                          'Every pixel vs the CARLA semantic camera at 800 x 600; pixels whose tag no model class '
                          'covers are ignored.'))
    print(f'\nSaved {out_csv}' + ''.join(f'\n      {p}' for p in made if p)
          + f'\n      {os.path.join(args.out, "<set>", "metrics.json")}')


if __name__ == '__main__':
    main()
