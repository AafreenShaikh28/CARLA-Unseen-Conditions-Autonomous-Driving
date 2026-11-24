"""
show_classes.py - which classes do the code, the checkpoints and the pose labels use? (no CARLA, no GPU)

    python show_classes.py
    python show_classes.py --data ..\\LIDAR_only\\kitti_pose\\training --lidar-ckpt <best.pt> --fusion-ckpt <best.pt>

Prints, for both pp_common.py copies: CLASS_NAMES and what person_lying / person_sitting / person_crouching map
to; whether pp_common_5class.py (the backup add_pose_class.py keeps) is there; the classes stored in the two
checkpoints; and the label names in labels_3d of the pose data, with how many boxes each.
"""
import os
import re
import glob
import argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)


def config_of(path):
    """CLASS_NAMES and CLASS_ALIASES read from the file text (no import, so no torch needed)"""
    src = open(path, encoding='utf-8').read()
    m = re.search(r"^CLASS_NAMES\s*=\s*\[([^\]]*)\]", src, re.M)
    names = re.findall(r"'([^']+)'", m.group(1)) if m else None
    m = re.search(r"^CLASS_ALIASES\s*=\s*\{(.*?)\}", src, re.M | re.S)
    aliases = {k: int(v) for k, v in re.findall(r"'([^']+)'\s*:\s*(\d+)", m.group(1))} if m else {}
    return names, aliases


def find_dirs(name, depth=4):
    """folders called <name> up to <depth> levels under python_code (fast: no full walk)"""
    out = []
    for d in range(1, depth + 1):
        out += glob.glob(os.path.join(CODE, *(['*'] * d), name))
    return sorted(p for p in out if os.path.isdir(p))


ANCHORS = {}        # checkpoint -> {class: [l, w, h, z]} (the anchor = mean training box of that class)


def load_names(torch, p):
    """class names of a PointPillars checkpoint; None for anything else (YOLO, SegFormer, unreadable)"""
    try:
        try:
            ck = torch.load(p, map_location='cpu', weights_only=False)
        except TypeError:
            ck = torch.load(p, map_location='cpu')
    except Exception:                       # e.g. a YOLO best.pt needs a matching ultralytics version
        return None
    if not isinstance(ck, dict) or 'anchor_sizes' not in ck:
        return None
    names = list(ck.get('class_names') or [])
    ANCHORS[p] = dict(zip(names, [list(map(float, a)) for a in ck['anchor_sizes']]))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'kitti_pose', 'training'))
    ap.add_argument('--lidar-ckpt', default=os.path.join(CODE, 'Detection_LIDAR', 'runs_lidar', 'pp_kitti_carla', 'best.pt'))
    ap.add_argument('--fusion-ckpt', default=os.path.join(CODE, 'Fusion', 'runs_lidar', 'pp_painted_kitti_carla', 'best.pt'))
    args = ap.parse_args()

    state, ck_names = {}, {}
    print('== pp_common.py')
    for folder in ('Detection_LIDAR', 'Fusion'):
        p = os.path.join(CODE, folder, 'pp_common.py')
        if not os.path.isfile(p):
            print(f'  {folder}: no pp_common.py')
            continue
        names, al = config_of(p)
        maps = {t: (names[al[t]] if names and t in al and al[t] < len(names) else 'NOT MAPPED (= background)')
                for t in ('person_lying', 'person_sitting', 'person_crouching')}
        backup = os.path.isfile(os.path.join(CODE, folder, 'pp_common_5class.py'))
        state[folder] = (names, maps['person_lying'])
        print(f'  {folder}: CLASS_NAMES {names}')
        print('      ' + ', '.join(f'{t} -> {v}' for t, v in maps.items()) + f'   pp_common_5class.py: {"yes" if backup else "no"}')

    print('== checkpoints')
    try:
        import torch
        for name, p in (('LiDAR-only', args.lidar_ckpt), ('fusion', args.fusion_ckpt)):
            if not os.path.isfile(p):
                print(f'  {name}: {p} not found (pass --lidar-ckpt / --fusion-ckpt)')
                continue
            names = load_names(torch, p)
            if names is None:
                print(f'  {name}: {p} is not a PointPillars checkpoint')
                continue
            ck_names[name] = names
            ped = ANCHORS.get(p, {}).get('Pedestrian')
            print(f"  {name}: {names}  ({p})")
            if ped:
                print(f'      Pedestrian anchor l {ped[0]:.2f} w {ped[1]:.2f} h {ped[2]:.2f} m = the mean Pedestrian box '
                      f'it was trained on (a lying person: ~1.8 x 0.6 x 0.4)')
        if len(ck_names) < 2:
            found = sorted(p for folder in ('Detection_LIDAR', 'Fusion')
                           for p in glob.glob(os.path.join(CODE, folder, 'runs*', '**', 'best.pt'), recursive=True))
            print('  PointPillars best.pt files in Detection_LIDAR and Fusion: ' + ('none' if not found else ''))
            for p in found[:20]:
                names = load_names(torch, p)
                if names is None:
                    print(f'    {p}   (not a PointPillars checkpoint, skipped)')
                    continue
                folder = os.path.relpath(p, CODE).split(os.sep)[0]
                code = (state.get(folder) or [None])[0]
                print(f'    {p}   {names}' + ('   <- does NOT load with this pp_common.py' if code and names != code else ''))
    except ImportError:
        print('  torch not installed here: run this in the carla env')

    data = args.data
    if not os.path.isdir(os.path.join(data, 'labels_3d')) and os.path.isdir(os.path.join(data, 'training', 'labels_3d')):
        data = os.path.join(data, 'training')
    print(f'== labels_3d in {data}')
    files = glob.glob(os.path.join(data, 'labels_3d', '*.txt'))
    cnt, dims, shape = Counter(), {}, Counter()
    for f in files:
        for line in open(f):
            t = line.split()
            if len(t) >= 8:
                cnt[t[0]] += 1
                l, w, h = float(t[4]), float(t[5]), float(t[6])
                dims.setdefault(t[0], []).append((max(l, w), h))       # length, height
                if t[0].startswith('person'):
                    shape['lying-shaped (h < 0.6 m, l >= 1.3 m)' if h < 0.6 and max(l, w) >= 1.3 else
                          'flat but short (h < 0.6 m, l < 1.3 m)' if h < 0.6 else
                          'low (0.6 - 1.4 m tall)' if h < 1.4 else 'upright (>= 1.4 m tall)'] += 1
    if not files:
        print('  no label files there. labels_3d folders under python_code:')
        for d in find_dirs('labels_3d'):
            print(f'    {d}   {len(glob.glob(os.path.join(d, "*.txt")))} files')
        print('  pass the one with the pose frames (ids 500000+) as --data <its parent folder>')
        verdict(state, ck_names, cnt, 0.0)
        return
    ids = sorted(int(os.path.basename(f)[:-4]) for f in files if os.path.basename(f)[:-4].isdigit())
    pose_ids = sum(i >= 500000 for i in ids)
    print(f'  {len(files)} frames, ids {ids[0] if ids else "-"} .. {ids[-1] if ids else "-"}; '
          f'{pose_ids} with id >= 500000 (collect_pose_data.py frames)')
    split_info(data)
    print(f"  {'name':<18}{'boxes':>8}{'length m (median)':>19}{'height m (median / min)':>25}{'boxes under 1.2 m tall':>24}")
    for k, v in cnt.most_common():
        ls, hs = sorted(d[0] for d in dims[k]), sorted(d[1] for d in dims[k])
        low = sum(h < 1.2 for h in hs)
        print(f"  {k:<18}{v:>8}{ls[len(ls) // 2]:>19.2f}{hs[len(hs) // 2]:>15.2f} / {hs[0]:<7.2f}{100 * low / v:>15.0f}%")
    tot = sum(shape.values())
    if tot:
        print('  person boxes by shape (all person* names):')
        for k in ('upright (>= 1.4 m tall)', 'low (0.6 - 1.4 m tall)', 'flat but short (h < 0.6 m, l < 1.3 m)',
                  'lying-shaped (h < 0.6 m, l >= 1.3 m)'):
            print(f'    {k:<40}{shape[k]:>9}  {100 * shape[k] / tot:5.1f}%')
    has_hz = bool(glob.glob(os.path.join(data, 'hazards', '*.txt')))
    if not any(k.startswith('person_') for k in cnt) and not has_hz:
        lying = shape['lying-shaped (h < 0.6 m, l >= 1.3 m)']
        print('  no person_lying / person_sitting / person_crouching names and no hazards/ folder. '
              + (f'{lying} lying-shaped person boxes: lying people may be in here, labelled "person".'
                 if tot and lying >= max(5, 0.02 * tot) else
                 'Almost no lying-shaped person boxes: this data has (next to) no people lying down.'))
    hazards(data, dims)
    verdict(state, ck_names, cnt, pose_ids / max(len(ids), 1))


def split_info(data):
    p = os.path.join(data, 'split_lidar.json')
    if not os.path.isfile(p):
        print('  no split_lidar.json: Lidar_train / Lidar_finetune need one (train / val ids)')
        return
    import json
    sp = json.load(open(p))
    tr, va = sp.get('train', []), sp.get('val', [])
    print(f'  split_lidar.json: train {len(tr)}, val {len(va)}'
          + ('   [!] val is EMPTY: the val run (Town05) was not collected or not finished' if not va else ''))
    prog = os.path.join(os.path.dirname(data), 'progress.json')
    if os.path.isfile(prog):
        runs = json.load(open(prog)).get('runs', {})
        print('  runs in progress.json: ' + (', '.join(f"{k} {v['last'] - v['first'] + 1} ({v['split']})"
                                                       for k, v in runs.items()) or 'none'))


def hazards(data, dims):
    """collect_kitti_lidar.py with posed pedestrians: hazards/<id>.txt, one line per posed person:
    actor_id pose place n_body_pts n_ghost_pts labelled x y z l w h yaw"""
    files = glob.glob(os.path.join(data, 'hazards', '*.txt'))
    if not files:
        return
    rows = {}
    for f in files:
        for line in open(f):
            t = line.split()
            if len(t) >= 13:
                rows.setdefault(t[1], []).append((int(t[3]), int(t[4]), int(t[5]),
                                                  max(float(t[9]), float(t[10])), float(t[11])))
    if not rows:
        print(f'  hazards/: {len(files)} files, no posed pedestrian lines')
        return
    n_person = len(dims.get('person', []))
    n_posed = sum(len(v) for v in rows.values())
    print(f'== posed pedestrians (hazards/, {len(files)} frames): {n_posed} of {n_person} person boxes '
          f'({100 * n_posed / max(n_person, 1):.0f}%) are posed, all labelled "person" in labels_3d')
    print(f"  {'pose':<11}{'boxes':>8}{'>=5 body hits':>15}{'median body hits':>18}{'median ghost hits':>19}"
          f"{'length m':>10}{'height m':>10}")
    import statistics as st
    for pose in ('lying', 'sitting', 'crouching'):
        r = rows.get(pose)
        if not r:
            continue
        body, ghost = [a[0] for a in r], [a[1] for a in r]
        print(f'  {pose:<11}{len(r):>8}{100 * sum(b >= 5 for b in body) / len(r):>14.0f}%{st.median(body):>18.0f}'
              f'{st.median(ghost):>19.0f}{st.median(a[3] for a in r):>10.2f}{st.median(a[4] for a in r):>10.2f}')
    seen = [a for v in rows.values() for a in v if a[0] >= 5]
    if seen:
        g = st.median(a[1] / max(a[0], 1) for a in seen)
        print(f'  ghost / body hits (people with >= 5 body hits): median {g:.2f}'
              + ('   [!] the LiDAR also hits the invisible upright capsule a lot: a standing-shaped cluster sits '
                 'over these boxes' if g > 0.5 else '   (low: the LiDAR sees the posed body, not the upright capsule)'))


def verdict(state, ck_names, cnt, pose_share=0.0):
    print('== what this means')
    if len(state) == 2 and len({tuple(v[0] or []) + (v[1],) for v in state.values()}) > 1:
        print('  the two pp_common.py copies differ: make them the same before training (add_pose_class.py does both)')
    for folder, (names, lying) in state.items():
        names = names or []
        ck = ck_names.get('LiDAR-only' if folder == 'Detection_LIDAR' else 'fusion')
        if 'LyingPerson' in names:
            print(f'  {folder}: the CODE has a separate LyingPerson class (add_pose_class.py was run on it)'
                  + (', the checkpoint does not: agent.py / Lidar_val.py refuse it with this file until '
                     'Lidar_finetune.py --init <it> makes a 6-class model from it (or, lying people as Pedestrian '
                     'instead: python add_pose_class.py --lying-as-pedestrian)'
                     if ck is not None and 'LyingPerson' not in ck else ''))
        elif lying == 'Pedestrian':
            print(f'  {folder}: lying / sitting / crouching are Pedestrian (5 classes): fine-tune with Lidar_finetune.py '
                  + ('(pose frames only: --freeze auto trains only the Pedestrian outputs)' if pose_share >= 0.9 else
                     '(this data has every class: --freeze auto trains all of the model)'))
        elif cnt.get('person_lying'):
            print(f'  {folder}: person_lying is not mapped: Lidar_finetune.py stops on unknown names. '
                  'python add_pose_class.py --lying-as-pedestrian')
        if ck is not None and names and ck != names and 'LyingPerson' not in names:
            print(f'  {folder}: checkpoint classes {ck} != pp_common.py {names}')


if __name__ == '__main__':
    main()