"""
lidar_compare.py - side-by-side visual comparison of ground truth vs PointPillars predictions
(like YOLO's val_batch_labels / val_batch_pred).

Each output image:   [ camera image | GT (BEV) | Prediction (BEV) ]
with a header giving per-frame counts: GT, Pred, matched (TP), false positives, misses.

Examples (run from Detection_LIDAR/):
    python lidar_compare.py --ckpt runs_lidar/pp_town03/best.pt
    python lidar_compare.py --ckpt runs_lidar/pp_town03/best.pt --split val --n 30 --mode random
    python lidar_compare.py --ckpt runs_lidar/pp_town03/best.pt --mode worst --n 20     # most errors first
    python lidar_compare.py --ckpt runs_lidar/pp_town03/best.pt --ids 008100 008500 --range 30
"""
import os, csv, json, argparse, random
import numpy as np
import cv2
import torch
import pp_common as C

COLORS = [(255, 140, 0), (0, 140, 255), (0, 220, 0), (220, 0, 220)]   # BGR per class index


def render(pts, boxes, cls, scores, size, R, title):
    s = size / (2 * R)
    img = np.zeros((size, size, 3), np.uint8)
    for r in range(10, int(R) + 1, 10):                                  # range rings
        cv2.circle(img, (size // 2, size // 2), int(r * s), (45, 45, 45), 1)
    u = ((pts[:, 1] + R) * s).astype(int)
    v = ((R - pts[:, 0]) * s).astype(int)
    m = (u >= 0) & (u < size - 1) & (v >= 0) & (v < size - 1)
    for dy in (0, 1):
        for dx in (0, 1):
            img[v[m] + dy, u[m] + dx] = (150, 150, 150)
    for k, b in enumerate(boxes):
        c = int(cls[k])
        col = COLORS[c % len(COLORS)]
        cr = C.bev_corners(b)
        px = np.stack([(cr[:, 1] + R) * s, (R - cr[:, 0]) * s], 1).astype(np.int32)
        cv2.polylines(img, [px], True, col, 2)
        reach = b[3] / 2 + 1.0
        fx, fy = b[0] + reach * np.cos(b[6]), b[1] + reach * np.sin(b[6])
        c0 = (int((b[1] + R) * s), int((R - b[0]) * s))
        cv2.arrowedLine(img, c0, (int((fy + R) * s), int((R - fx) * s)), col, 2, tipLength=0.4)
        label = C.CLASS_NAMES[c] + (f' {scores[k]:.2f}' if scores is not None else '')
        cv2.putText(img, label, (int(px[:, 0].min()), max(int(px[:, 1].min()) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    cv2.circle(img, (size // 2, size // 2), 5, (255, 255, 0), -1)         # ego
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for k, name in enumerate(C.CLASS_NAMES):                              # legend
        cv2.putText(img, name, (10, size - 12 - 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLORS[k % len(COLORS)], 2, cv2.LINE_AA)
    return img


def match_stats(gt_b, gt_c, pr_b, pr_c, pr_s, thr=0.5):
    """greedy BEV matching (same class, IoU>=thr) -> (tp, fp, fn)"""
    bev, _ = C.pair_ious(pr_b, gt_b)
    used = np.zeros(len(gt_b), bool)
    tp = 0
    for k in np.argsort(-pr_s):
        best, bi = -1, thr
        for j in range(len(gt_b)):
            if not used[j] and gt_c[j] == pr_c[k] and bev[k, j] >= bi:
                best, bi = j, bev[k, j]
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, len(pr_b) - tp, len(gt_b) - tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs_lidar/pp_town03/best.pt')
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split', default='val', choices=['val', 'train', 'all'])
    ap.add_argument('--n', type=int, default=20, help='number of images to save')
    ap.add_argument('--mode', default='even', choices=['even', 'random', 'worst'])
    ap.add_argument('--ids', nargs='*', default=None, help='specific frame ids, e.g. 008100')
    ap.add_argument('--score_thr', type=float, default=0.3, help='min score of predictions to draw/count')
    ap.add_argument('--range', type=float, default=51.2, help='half-size of the BEV view in metres')
    ap.add_argument('--size', type=int, default=800)
    ap.add_argument('--out_dir', default='')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = C.build_model_from_ckpt(torch.load(args.ckpt, map_location='cpu'), dev)
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), f'compare_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    if args.split == 'all':
        all_ids = C.get_frame_ids(args.data_root)
    else:
        all_ids = json.load(open(os.path.join(args.data_root, 'split_lidar.json')))[args.split]
    ds_all = C.LidarDataset(args.data_root, all_ids, args.label_dir, train=False)
    index = {i: k for k, i in enumerate(all_ids)}

    if args.ids:
        cand = args.ids
    elif args.mode == 'even':
        cand = [all_ids[k] for k in np.linspace(0, len(all_ids) - 1, args.n).astype(int)]
    elif args.mode == 'random':
        random.seed(0)
        cand = random.sample(all_ids, min(args.n, len(all_ids)))
    else:
        cand = all_ids                                   # 'worst': score every frame first
    print(f'[compare] running model on {len(cand)} frames ...')

    stats = {}
    with torch.no_grad():
        for s in range(0, len(cand), 4):
            chunk = cand[s:s + 4]
            batch = C.collate([ds_all[index[i]] for i in chunk])
            res = C.decode_predictions(model(batch['points']), model, score_thr=args.score_thr)
            for k, i in enumerate(chunk):
                gb, gc = batch['boxes'][k].numpy(), batch['cls'][k].numpy()
                tp, fp, fn = match_stats(gb, gc, res[k]['boxes'], res[k]['cls'], res[k]['scores'])
                stats[i] = {'pred': res[k], 'gt_n': len(gb), 'tp': tp, 'fp': fp, 'fn': fn}

    # summary csv for all processed frames
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'n_gt', 'n_pred', 'tp@0.5', 'false_pos', 'missed'])
        for i, st in stats.items():
            w.writerow([i, st['gt_n'], len(st['pred']['boxes']), st['tp'], st['fp'], st['fn']])
    tot = {k: sum(s[k] for s in stats.values()) for k in ('gt_n', 'tp', 'fp', 'fn')}
    print(f"[compare] over {len(stats)} frames: GT {tot['gt_n']}  matched {tot['tp']}  "
          f"false-pos {tot['fp']}  missed {tot['fn']}")

    if args.mode == 'worst' and not args.ids:
        pick = sorted(stats, key=lambda i: -(stats[i]['fp'] + stats[i]['fn']))[:args.n]
    else:
        pick = list(stats)[:args.n] if not args.ids else args.ids

    for i in pick:
        st = stats[i]
        sample = ds_all[index[i]]
        pts, gb, gc = sample['points'].numpy(), sample['boxes'].numpy(), sample['cls'].numpy()
        pr = st['pred']
        gt_img = render(pts, gb, gc, None, args.size, args.range, f'GROUND TRUTH ({len(gb)})')
        pr_img = render(pts, pr['boxes'], pr['cls'], pr['scores'], args.size, args.range,
                        f"PREDICTION ({len(pr['boxes'])})")
        panels = []
        cam_path = os.path.join(args.data_root, 'image_2', i + '.png')
        if os.path.exists(cam_path):
            cam = cv2.imread(cam_path)
            h = args.size
            cam = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
            cv2.putText(cam, 'CAMERA', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(cam)
        panels += [gt_img, pr_img]
        body = np.hstack(panels)
        head = np.zeros((44, body.shape[1], 3), np.uint8)
        txt = (f"frame {i}   GT {st['gt_n']}   pred {len(pr['boxes'])}   matched {st['tp']}   "
               f"false-pos {st['fp']}   missed {st['fn']}   (score>={args.score_thr}, BEV IoU>=0.5)")
        cv2.putText(head, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f'cmp_{i}.png'), np.vstack([head, body]))

    print(f'[compare] saved {len(pick)} images + summary.csv to {out_dir}')


if __name__ == '__main__':
    main()