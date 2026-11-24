"""
add_pose_class.py - adds the class LyingPerson (a person lying on the road) to BOTH pp_common.py copies
(Detection_LIDAR and Fusion: they must stay identical), and the label names collect_pose_data.py writes:

    person_lying                         -> LyingPerson (new class 5, its own anchor ~1.75 x 0.6 x 0.4 m)
    person_sitting, person_crouching     -> Pedestrian (same class: from above they are pedestrian-sized)

Separate class or Pedestrian? Measured on the anchor grid (0.4 m, axis-aligned bird's-eye overlap, random
heading): a lying person gets 1.0 positive anchor as Pedestrian, 1.2 as LyingPerson, a standing person 1.3
(the best anchor is always made positive, so every person is learned). The real costs of Pedestrian:
one box head has to output both 1.8 m tall and 0.4 m tall people, and Lidar_train.py FROM SCRATCH sets the
Pedestrian anchor to the mean box, which lying people pull longer and lower (Lidar_finetune.py keeps the
trained anchor). The gain: no class change, every 5-class checkpoint still loads everywhere.

    python add_pose_class.py            # patch both files (a copy of each original: pp_common_5class.py)
    python add_pose_class.py --undo     # put the originals back
    python add_pose_class.py --files <path\\pp_common.py> ...   (other locations)
    python add_pose_class.py --lying-as-pedestrian   # NO new class: lying, sitting, crouching all -> Pedestrian
                                                     # (also takes a LyingPerson class added before back out)

Only the CONFIG block changes (CLASS_NAMES, CLASS_ALIASES, POS_THR, NEG_THR, EVAL_IOU, DEFAULT_ANCHORS).
A file whose config block does not look as expected is left untouched. After patching, the patched
module is imported once to check it (needs torch: run it in the carla env).
Checkpoints trained with 5 classes no longer load with the patched file (build_model_from_ckpt stops
with "checkpoint classes ... != CLASS_NAMES"): use Lidar_finetune.py to turn them into 6-class models.
pose_eval.py reads the old ones through pp_common_5class.py.
"""
import os
import re
import sys
import shutil
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
DEFAULT_FILES = [os.path.join(CODE, 'Detection_LIDAR', 'pp_common.py'), os.path.join(CODE, 'Fusion', 'pp_common.py')]
NEW = 'LyingPerson'
NEW_THR = dict(POS_THR='0.50', NEG_THR='0.35', EVAL_IOU='0.50')        # as Pedestrian
NEW_ANCHOR = '[1.75, 0.6, 0.4, -2.2]'      # [l, w, h, z_centre]; only used if the data has no LyingPerson box
BACKUP = 'pp_common_5class.py'
BACKUP6 = 'pp_common_6class.py'          # --lying-as-pedestrian on a file that had LyingPerson


def patch_text(src):
    """-> (new text, None) or (None, reason)"""
    m = re.search(r"^CLASS_NAMES\s*=\s*\[([^\]]*)\]", src, re.M)
    if not m:
        return None, 'CLASS_NAMES = [...] not found'
    names = re.findall(r"'([^']+)'", m.group(1))
    if NEW in names:
        return None, 'already has LyingPerson'
    n = len(names)
    out = src[:m.end(1)] + f", '{NEW}'" + src[m.end(1):]

    m = re.search(r"^CLASS_ALIASES\s*=\s*\{(.*?)\}", out, re.M | re.S)
    if not m:
        return None, 'CLASS_ALIASES = {...} not found'
    body = m.group(1)
    if "'pedestrian'" not in body:
        return None, "CLASS_ALIASES has no 'pedestrian'"
    ped = int(re.search(r"'pedestrian'\s*:\s*(\d+)", body).group(1))
    add = (f",\n                 'person_sitting': {ped}, 'person_crouching': {ped},   # collect_pose_data.py"
           f"\n                 'person_lying': {n}, 'lyingperson': {n}, 'lying_person': {n}")
    out = out[:m.end(1)] + add + out[m.end(1):]

    for key, val in NEW_THR.items():
        m = re.search(r"^(%s\s*=\s*\()([^)]*)(\))" % key, out, re.M)
        if not m:
            return None, f'{key} = (...) not found'
        if len([v for v in m.group(2).split(',') if v.strip()]) != n:
            return None, f'{key} does not have {n} values'
        out = out[:m.end(2)] + f', {val}' + out[m.end(2):]

    m = re.search(r"^DEFAULT_ANCHORS\s*=\s*\[(.*?\])\]", out, re.M | re.S)
    if not m:
        return None, 'DEFAULT_ANCHORS = [[...]] not found'
    if len(re.findall(r'\[[^\[\]]*\]', m.group(1))) != n:
        return None, f'DEFAULT_ANCHORS does not have {n} anchors'
    out = out[:m.end(1)] + f', {NEW_ANCHOR}' + out[m.end(1):]

    out = re.sub(r"^(#\s+Car\s+Truck\s+Bus\s+Ped\s+Cyclist)[ \t]*$", r"\1   LyingP", out, count=1, flags=re.M)
    out = re.sub(r"^(CLASS_NAMES\s*=)", "# LyingPerson added by add_pose_class.py (people lying on the road, "
                                        "collect_pose_data.py)\n\\1", out, count=1, flags=re.M)
    return out, None


def unpatch_text(src):
    """exact reverse of patch_text (keeps any other edit made to the file since). -> (text, None) or (None, why)"""
    m = re.search(r"^CLASS_NAMES\s*=\s*\[([^\]]*)\]", src, re.M)
    names = re.findall(r"'([^']+)'", m.group(1)) if m else []
    if not names or names[-1] != NEW:
        return None, f'{NEW} is not the last name in CLASS_NAMES'
    n = len(names) - 1                                              # index of LyingPerson
    out = src[:m.start(1)] + re.sub(r",\s*'%s'" % NEW, '', m.group(1), count=1) + src[m.end(1):]
    out, k = re.subn(r",\s*'person_sitting'\s*:\s*\d+\s*,\s*'person_crouching'\s*:\s*\d+\s*,[^\n]*\n\s*"
                     r"'person_lying'\s*:\s*%d\s*,\s*'lyingperson'\s*:\s*%d\s*,\s*'lying_person'\s*:\s*%d" % (n, n, n),
                     '', out, count=1)
    if not k:
        return None, 'the pose names in CLASS_ALIASES are not as add_pose_class.py wrote them'
    for key in NEW_THR:
        m = re.search(r"^(%s\s*=\s*\()([^)]*)(\))" % key, out, re.M)
        parts = m.group(2).split(',') if m else []
        if len([v for v in parts if v.strip()]) != n + 1:
            return None, f'{key} does not have {n + 1} values'
        out = out[:m.start(2)] + ','.join(parts[:-1]) + out[m.end(2):]
    m = re.search(r"^DEFAULT_ANCHORS\s*=\s*\[(.*?\])\]", out, re.M | re.S)
    if not m or len(re.findall(r'\[[^\[\]]*\]', m.group(1))) != n + 1:
        return None, f'DEFAULT_ANCHORS does not have {n + 1} anchors'
    inner = re.sub(r",\s*\[[^\[\]]*\]$", '', m.group(1), count=1)
    out = out[:m.start(1)] + inner + out[m.end(1):]
    out = re.sub(r"^# LyingPerson added by add_pose_class\.py[^\n]*\n", '', out, count=1, flags=re.M)
    out = re.sub(r"^(#\s+Car\s+Truck\s+Bus\s+Ped\s+Cyclist)   LyingP[ \t]*$", r"\1", out, count=1, flags=re.M)
    m = re.search(r"^CLASS_ALIASES\s*=\s*\{(.*?)\}", out, re.M | re.S)
    if not m or any(int(v) >= n for v in re.findall(r":\s*(\d+)", m.group(1))):
        return None, f'CLASS_ALIASES still maps a name to class {n}'
    return out, None


def patch_text_ped(src):
    """--lying-as-pedestrian: only aliases, no new class (a LyingPerson class added before is taken out
    first). -> (new text, None, removed LyingPerson?) or (None, reason, False)"""
    m = re.search(r"^CLASS_NAMES\s*=\s*\[([^\]]*)\]", src, re.M)
    if not m:
        return None, 'CLASS_NAMES = [...] not found', False
    removed = NEW in re.findall(r"'([^']+)'", m.group(1))
    if removed:
        src, why = unpatch_text(src)
        if src is None:
            return None, why, False
    m = re.search(r"^CLASS_ALIASES\s*=\s*\{(.*?)\}", src, re.M | re.S)
    if not m or "'pedestrian'" not in m.group(1):
        return None, "CLASS_ALIASES with 'pedestrian' not found", False
    if "'person_lying'" in m.group(1):
        return None, 'already maps person_lying', False
    ped = int(re.search(r"'pedestrian'\s*:\s*(\d+)", m.group(1)).group(1))
    add = (",\n                 # collect_pose_data.py poses: all Pedestrian (add_pose_class.py --lying-as-pedestrian)"
           f"\n                 'person_sitting': {ped}, 'person_crouching': {ped}, 'person_lying': {ped}")
    return src[:m.end(1)] + add + src[m.end(1):], None, removed


def import_check(path):
    """import the patched file in a fresh Python and check the class tables agree"""
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('ppc', sys.argv[1]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(m.CLASS_NAMES, m.NUM_CLASSES, m.CLASS_ALIASES['person_lying'], m.CLASS_ALIASES['person_sitting'])")
    r = subprocess.run([sys.executable, '-c', code, path], capture_output=True, text=True)
    err = r.stderr.strip().splitlines()
    return r.returncode == 0, r.stdout.strip() or (err[-1] if err else '')


def main():
    ap = argparse.ArgumentParser(description='add the LyingPerson class to pp_common.py (both copies)')
    ap.add_argument('--files', nargs='*', default=DEFAULT_FILES)
    ap.add_argument('--undo', action='store_true', help=f'restore the originals from {BACKUP}')
    ap.add_argument('--no-import-check', action='store_true')
    ap.add_argument('--lying-as-pedestrian', action='store_true',
                    help='no new class: person_lying / person_sitting / person_crouching all become Pedestrian')
    args = ap.parse_args()

    bad, done = 0, 0
    for path in args.files:
        path = os.path.abspath(path)
        backup = os.path.join(os.path.dirname(path), BACKUP)
        if not os.path.isfile(path):
            print(f'[skip] {path}: not found')
            bad += 1
            continue
        if args.undo:
            if os.path.isfile(backup):
                shutil.copy2(backup, path)
                print(f'[undo] {path} restored from {BACKUP}')
            else:
                print(f'[undo] {path}: no {BACKUP} next to it, nothing to restore')
            continue
        with open(path, encoding='utf-8') as f:
            src = f.read()
        removed = False
        if args.lying_as_pedestrian:
            new, why, removed = patch_text_ped(src)
            if new is None and why not in ('already maps person_lying', 'CLASS_NAMES = [...] not found') \
                    and os.path.isfile(backup):            # a hand-edited 6-class file: start from the 5-class original
                with open(backup, encoding='utf-8') as f:
                    new, why2, _ = patch_text_ped(f.read())
                if new is not None:
                    print(f'       {path}: {why}; starting from {BACKUP} instead (edits made after it are lost)')
                    removed, why = True, None
        else:
            new, why = patch_text(src)
        if new is None:
            print(f'[skip] {path}: {why}')
            bad += why not in ('already has LyingPerson', 'already maps person_lying')
            if why == 'already maps person_lying':
                done += 1
            if why == 'already has LyingPerson':
                done += 1
                if not os.path.isfile(backup):
                    print(f'       no {BACKUP} next to it: pose_eval.py cannot read 5-class checkpoints from this '
                          f'folder (only an issue for the "before" numbers)')
            continue
        if removed:                                    # keep the 6-class file too (to go back: copy it over pp_common.py)
            shutil.copy2(path, os.path.join(os.path.dirname(path), BACKUP6))
        elif not os.path.isfile(backup):
            shutil.copy2(path, backup)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new)
        what = (f'{NEW} class taken out, lying / sitting / crouching -> Pedestrian (6-class file kept as {BACKUP6})'
                if removed else 'patched' + (f' (original kept as {BACKUP})' if os.path.isfile(backup) else ''))
        if args.no_import_check:
            print(f'[ok]   {path}: {what}')
            continue
        ok, msg = import_check(path)
        if ok:
            print(f'[ok]   {path}: {what}; import check: {msg}')
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(src)
            print(f'[FAIL] {path}: the patched file does not import ({msg}); file left as it was')
            bad += 1
    if bad:
        sys.exit(1)
    if not args.undo and done == len(args.files):
        print(f"\n{'Both copies' if len(args.files) == 2 else 'All files'} already " + ('map the pose names to Pedestrian' if args.lying_as_pedestrian
                                            else 'have LyingPerson') + ': nothing to do.')
        return
    if args.lying_as_pedestrian and not args.undo:
        print('\nLying, sitting and crouching people are now Pedestrian in both copies (5 classes, old checkpoints '
              'still load everywhere).')
        return
    if not args.undo:
        print('\nBoth copies must be patched (the fusion vs LiDAR-only comparison needs identical class tables).\n'
              'Old 5-class checkpoints now only load in pose_eval.py; turn them into 6-class models with\n'
              'Lidar_finetune.py (Detection_LIDAR and Fusion). Delete __pycache__ folders if Python still sees the old file.')


if __name__ == '__main__':
    main()