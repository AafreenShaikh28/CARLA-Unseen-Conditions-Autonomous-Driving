"""
label_lying.py - give the people lying on the road their own label name, for a separate LyingPerson class.

collect_kitti_lidar.py writes every walker as 'person' in labels_3d (lying ones included, with the box from
their bones); their pose is only in hazards\\<id>.txt (by actor id). This rewrites 'person' -> 'person_lying'
in labels_3d for the walkers that hazards\\ lists as lying. Sitting and crouching stay 'person' (Pedestrian).
Then add_pose_class.py (without --lying-as-pedestrian) maps person_lying to the class LyingPerson.

    python label_lying.py --dry      # LIDAR_only: count only
    python label_lying.py            # labels_3d is copied to labels_3d_before_lying first (once)
    python label_lying.py --undo     # put labels_3d_before_lying back

Only labels_3d changes; label_2 (camera labels, YOLO) keeps 'Pedestrian'. Frames collected later: run it again
(already rewritten lines are left as they are).
"""
import os
import glob
import shutil
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BACKUP = 'labels_3d_before_lying'


def lying_ids(path):
    out = set()
    if os.path.isfile(path):
        for line in open(path):
            f = line.split()
            if len(f) >= 2 and f[1] == 'lying':
                out.add(f[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.join(HERE, 'kitti_carla'))
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--undo', action='store_true')
    args = ap.parse_args()
    tr = os.path.join(os.path.abspath(args.root), 'training')
    lab, bak, hz = os.path.join(tr, 'labels_3d'), os.path.join(tr, BACKUP), os.path.join(tr, 'hazards')
    if args.undo:
        if not os.path.isdir(bak):
            raise SystemExit(f'no {bak}: nothing to undo')
        shutil.rmtree(lab)
        shutil.copytree(bak, lab)
        print(f'labels_3d restored from {BACKUP}')
        return
    if not os.path.isdir(hz):
        raise SystemExit(f'no {hz}: this data has no posed pedestrians (hazards folder)')
    files = sorted(glob.glob(os.path.join(lab, '*.txt')))
    plan, cnt = {}, Counter()
    for p in files:
        fid = os.path.basename(p)[:-4]
        ids = lying_ids(os.path.join(hz, fid + '.txt'))
        lines = open(p).read().split('\n')
        new, changed = [], 0
        for line in lines:
            f = line.split()
            if len(f) >= 10 and f[0] == 'person' and f[9] in ids:
                line = 'person_lying' + line[len('person'):]
                changed += 1
            elif len(f) >= 10 and f[0] == 'person_lying':
                cnt['already'] += 1
            new.append(line)
        cnt['lying in hazards'] += len(ids)
        if changed:
            plan[p] = '\n'.join(new)
            cnt['rewritten'] += changed
    missing = cnt['lying in hazards'] - cnt['rewritten'] - cnt['already']
    print(f'{len(files)} label files: {cnt["lying in hazards"]} lying people in hazards\\, {cnt["rewritten"]} lines '
          f'to rewrite person -> person_lying in {len(plan)} files, {cnt["already"]} already person_lying'
          + (f', {missing} not found in labels_3d (beyond 60 m or removed that tick)' if missing > 0 else ''))
    if args.dry or not plan:
        print('--dry: nothing written' if args.dry else 'nothing to do')
        return
    if not os.path.isdir(bak):
        shutil.copytree(lab, bak)
        print(f'backup: {bak}')
    for p, text in plan.items():
        with open(p, 'w') as f:
            f.write(text)
    print('done. Next: python add_pose_class.py --undo, then python add_pose_class.py (LyingPerson class), '
          'then python show_classes.py --data kitti_carla')


if __name__ == '__main__':
    main()