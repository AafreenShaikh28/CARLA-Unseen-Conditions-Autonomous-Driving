"""
prepare_yolo_data.py - turn the camera collector's output (dataset_pose\\<Town>\\image_2 + labels_yolo) into
the folder layout Ultralytics trains on, then fine-tune with your own fine_tune.py.

Ultralytics finds a label by replacing 'images' with 'labels' in the image path, so image_2 + labels_yolo
cannot be used directly. This builds (hard links: no extra disk on the same drive; copies otherwise):

    yolo_pose\\images\\train\\Town04_000123.png    yolo_pose\\labels\\train\\Town04_000123.txt
    yolo_pose\\images\\val\\Town01_002611.png      yolo_pose\\labels\\val\\Town01_002611.txt
    yolo_pose\\data.yaml                           class names copied from --weights (so the head is kept)

Split by town (frames of one run are near-duplicates: never split them at random). Default: Town04 train,
Town01 val, the same split as the LiDAR models. Only complete frames (metadata\\<id>.json is the last file
the collector writes for a frame).

    python prepare_yolo_data.py --data-root <...\\dataset_pose> --check     # counts only, writes nothing
    python prepare_yolo_data.py --data-root <...\\dataset_pose>
"""
import os
import glob
import shutil
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(HERE, 'runs', 'detect', 'runs', 'detect', 'carla_finetune-10', 'weights', 'best.pt')


def frames(root, town):
    d = os.path.join(root, town)
    if not os.path.isdir(d):
        have = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
        raise SystemExit(f'{d} not found. Towns in {root}: {have}')
    out = []
    for p in sorted(glob.glob(os.path.join(d, 'metadata', '*.json'))):
        fid = os.path.basename(p)[:-5]
        if (os.path.isfile(os.path.join(d, 'image_2', fid + '.png'))
                and os.path.isfile(os.path.join(d, 'labels_yolo', fid + '.txt'))):
            out.append(fid)
    return d, out


def read_labels(path, nc):
    good, bad = [], 0
    for line in open(path):
        f = line.split()
        if not f:
            continue
        try:
            c, v = int(f[0]), [float(x) for x in f[1:]]
        except ValueError:
            bad += 1
            continue
        if len(v) != 4 or not 0 <= c < nc or not all(0.0 <= x <= 1.0 for x in v) or v[2] <= 0 or v[3] <= 0:
            bad += 1
            continue
        good.append((c, line.strip()))
    return good, bad


def place(src, dst, copy):
    if os.path.exists(dst):
        os.remove(dst)
    if not copy:
        try:
            os.link(src, dst)
            return 'link'
        except OSError:                          # another drive, or a file system without hard links
            pass
    shutil.copy2(src, dst)
    return 'copy'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True, help='the collector output folder (CFG out, e.g. dataset_pose)')
    ap.add_argument('--train-towns', nargs='+', default=['Town04'])
    ap.add_argument('--val-towns', nargs='+', default=['Town01'])
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS, help='the model you will fine-tune (class names)')
    ap.add_argument('--out', default=os.path.join(HERE, 'yolo_pose'))
    ap.add_argument('--copy', action='store_true', help='copy images instead of hard links')
    ap.add_argument('--check', action='store_true', help='print the counts, write nothing')
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(f'{args.weights} not found: pass --weights <your fine-tuned best.pt>')
    from ultralytics import YOLO
    names = YOLO(args.weights).names                      # {id: name}, the model's own classes
    nc = len(names)
    print(f'{args.weights}: {nc} classes')
    if set(args.train_towns) & set(args.val_towns):
        raise SystemExit('a town is in both --train-towns and --val-towns')

    plan, stats = {}, {}
    for split, towns in (('train', args.train_towns), ('val', args.val_towns)):
        cnt, n_bad, n_empty, items = Counter(), 0, 0, []
        for t in towns:
            d, ids = frames(args.data_root, t)
            for fid in ids:
                good, bad = read_labels(os.path.join(d, 'labels_yolo', fid + '.txt'), nc)
                cnt.update(c for c, _ in good)
                n_bad += bad
                n_empty += not good
                items.append((os.path.join(d, 'image_2', fid + '.png'), f'{t}_{fid}', [l for _, l in good]))
        plan[split], stats[split] = items, (cnt, n_bad, n_empty)
        print(f'\n{split}: {len(items)} frames from {towns} ({n_empty} with no box, {n_bad} bad label lines dropped)')
        for c, n in sorted(cnt.items()):
            print(f'    {c:>3} {names[c]:<14}{n:>8} boxes')
    if not plan['val']:
        raise SystemExit('\nno validation frames: collect a validation town or pass --val-towns')
    labelled = set(stats['train'][0])
    never = [names[c] for c in (9, 11) if c < nc and c not in labelled]
    if never:
        print(f'\n[note] never labelled in this data: {never}. The collector labels vehicles and walkers only, '
              f'so the fine-tune learns these as background and the model gets worse at them.')
    if args.check:
        print('\n--check: nothing written')
        return

    how = Counter()
    for split, items in plan.items():
        di, dl = os.path.join(args.out, 'images', split), os.path.join(args.out, 'labels', split)
        for p in (di, dl):
            if os.path.isdir(p):
                shutil.rmtree(p)                          # a re-run must not keep frames of an older split
            os.makedirs(p)
        for src, name, lines in items:
            how[place(src, os.path.join(di, name + '.png'), args.copy)] += 1
            with open(os.path.join(dl, name + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
    out = os.path.abspath(args.out)
    with open(os.path.join(out, 'data.yaml'), 'w') as f:
        f.write(f"path: {out.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n")
        for k in range(nc):
            f.write(f"  {k}: '{names[k]}'\n")
    print(f'\nwritten: {out} ({how["link"]} hard links, {how["copy"]} copies), data.yaml with {nc} classes')


if __name__ == '__main__':
    main()