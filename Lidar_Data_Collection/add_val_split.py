"""
add_val_split.py - the town01_cloudysunset run stopped before it finished: its frames (2500 .. 3976 in your
kitti_carla) are on disk but in no split, and split_lidar.json has val = 0. Lidar_finetune.py, Lidar_val.py and
pose_eval.py need a validation set. This makes those frames the validation set:

  - only frames that are complete (metadata\\<id>.json is the last file the collector writes) and whose town
    is NOT a training town (the split stays by town: no near-identical frames on both sides)
  - split_lidar.json: val = those frames (train unchanged); the old file is kept as split_lidar.before_val.json
  - progress.json: the run is recorded (split 'val'), so collect_kitti_lidar.py continues AFTER these frames
    and never overwrites them (old file kept as progress.before_val.json)

    python add_val_split.py --dry        # LIDAR_only: show what it would do
    python add_val_split.py
"""
import os
import json
import glob
import shutil
import argparse
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def complete(tr, fid):
    """-> metadata dict if every file of the frame is there, else None"""
    try:
        v = os.path.getsize(os.path.join(tr, 'velodyne', fid + '.bin'))
        if v == 0 or v % 16 or not os.path.isfile(os.path.join(tr, 'labels_3d', fid + '.txt')):
            return None
        with open(os.path.join(tr, 'metadata', fid + '.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(HERE, 'kitti_carla'))
    ap.add_argument('--dry', action='store_true', help='only print what would change')
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    tr = os.path.join(root, 'training')
    sp_path, pr_path = os.path.join(tr, 'split_lidar.json'), os.path.join(root, 'progress.json')
    split = json.load(open(sp_path))
    prog = json.load(open(pr_path)) if os.path.isfile(pr_path) else {'next_idx': 0, 'runs': {}}

    recorded = set()
    for r in prog['runs'].values():
        recorded |= set(range(r['first'], r['last'] + 1))
    train_towns = {r.get('town') for r in prog['runs'].values() if r.get('split') == 'train'}
    ids = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(tr, 'labels_3d', '*.txt'))
                 if os.path.basename(p)[:-4].isdigit())
    extra = [i for i in ids if i not in recorded]
    print(f'{root}\n  split_lidar.json: train {len(split.get("train", []))}, val {len(split.get("val", []))}')
    recs = ', '.join('%s %d..%d (%s)' % (k, v['first'], v['last'], v['split']) for k, v in prog['runs'].items())
    print(f'  progress.json runs: {recs}')
    if not extra:
        raise SystemExit('  no frames outside the recorded runs: nothing to add')

    runs, bad = defaultdict(list), []
    for i in extra:
        meta = complete(tr, f'{i:06d}')
        if meta is None:
            bad.append(i)
        else:
            runs[(meta.get('run', '?'), meta.get('town', '?'), meta.get('weather', '?'))].append(i)
    print(f'  frames on disk outside the recorded runs: {len(extra)} ({extra[0]} .. {extra[-1]}), '
          f'{len(bad)} incomplete (skipped){": " + str(bad[:5]) if bad else ""}')
    val, new_runs = [], {}
    for (run, town, weather), fr in sorted(runs.items(), key=lambda kv: kv[1][0]):
        same = town in train_towns
        print(f'    {run}: {town} / {weather}: {len(fr)} complete frames ({fr[0]} .. {fr[-1]})'
              + ('   SKIPPED: same town as the training frames' if same else '   -> val'))
        if not same:
            val += fr
            new_runs[run] = dict(first=fr[0], last=fr[-1], split='val', town=town, weather=weather,
                                 note='partial run, made the validation set by add_val_split.py')
    if not runs:
        raise SystemExit('  no complete frames outside the recorded runs: nothing to add')
    if not val:
        raise SystemExit('  no usable frames from another town: collect one (collect_kitti_lidar.py --runs town05_clearnoon)')
    old_val = set(split.get('val', []))
    split['val'] = sorted(old_val | {f'{i:06d}' for i in val})
    last = max(extra)                          # incomplete tail frames too: the collector must not reuse their ids
    nxt = max(prog.get('next_idx', 0), last + 1)
    towns = Counter(t for (_, t, _), fr in runs.items() for _ in fr if t not in train_towns)
    print(f'  new split: train {len(split["train"])}, val {len(split["val"])} ({dict(towns)}); '
          f'progress next_idx {prog.get("next_idx")} -> {nxt}')
    if args.dry:
        print('  --dry: nothing written')
        return
    shutil.copy2(sp_path, os.path.join(tr, 'split_lidar.before_val.json'))
    if os.path.isfile(pr_path):
        shutil.copy2(pr_path, os.path.join(root, 'progress.before_val.json'))
    with open(sp_path, 'w') as f:
        json.dump(split, f)
    for run, r in new_runs.items():
        if run in prog['runs']:
            run += '_val'
        prog['runs'][run] = r
    prog['next_idx'] = nxt
    with open(pr_path, 'w') as f:
        json.dump(prog, f, indent=2)
    ims = os.path.join(root, 'ImageSets')
    if os.path.isdir(ims):
        with open(os.path.join(ims, 'val.txt'), 'w') as f:
            f.write('\n'.join(split['val']) + '\n')
    print('  written: split_lidar.json, progress.json (old copies: *.before_val.json). '
          'Paint these frames too (Fusion\\paint_points.py) before the fusion fine-tune.')


if __name__ == '__main__':
    main()