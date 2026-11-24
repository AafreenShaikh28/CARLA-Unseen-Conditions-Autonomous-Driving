"""
Lidar_finetune.py - continue training an ALREADY TRAINED PointPillars model (LiDAR-only or fusion) on new data
(e.g. the people lying, sitting and crouching), starting from its weights: much faster than Lidar_train.py
from scratch. Lidar_train.py --resume cannot do this: it refuses a changed split or class list, and a
finished run has nothing left to do. Works with or without new classes (without: all weights copied).

Put the SAME file in Detection_LIDAR\\ and in Fusion\\: it uses the pp_common.py next to it.
Before: collect_pose_data.py -> merge_pose_data.py -> add_pose_class.py (both pp_common.py copies)
        -> Fusion\\paint_points.py (paints the new frames; fusion only)
Lying people as Pedestrian (no new class): add_pose_class.py --lying-as-pedestrian instead; then nothing
is transferred (all weights copied) and keep_old trains only the Pedestrian outputs.

    Detection_LIDAR> python Lidar_finetune.py --init runs_lidar\\pp_kitti_carla\\best.pt --check
    Detection_LIDAR> python Lidar_finetune.py --init runs_lidar\\pp_kitti_carla\\best.pt
    Fusion>          python Lidar_finetune.py --init runs_lidar\\pp_painted_kitti_carla\\best.pt
Same --epochs --lr --batch --seed for both (the defaults): the LiDAR-only vs fusion comparison stays fair.
Output: <init folder>_pose\\best.pt / last.pt, same format as Lidar_train.py, so Lidar_val.py, pose_eval.py
and the Driving stack (--lidar-ckpt / --fusion-ckpt) load them.

What happens to the weights
  - every layer of the old model is copied (point net, backbone, the heads of Car ... Cyclist)
  - the new class (LyingPerson): its box and direction heads start from the Pedestrian ones (similar
    point clusters), its classifier starts fresh (prior 1%), its anchor size comes from the data
  - training uses the whole merged split, all classes, so the old classes are not forgotten
Painted (fusion) models: the number of point features, the painted folder and its segmentation classes
come from the checkpoint and must match the data (every train / val frame painted, by one model).

Only the pose data, not the old training data? (--freeze, default auto)
  Trained on pose frames alone, the model would forget what it barely sees there (trucks, buses, cyclists,
  moving traffic). --freeze keep_old trains only the outputs of the new class (LyingPerson) and Pedestrian
  (sitting / crouching people are Pedestrians); the feature layers and the Car / Truck / Bus / Cyclist
  outputs stay exactly as trained, so those classes give the same boxes as before.
    none       train everything (the data includes the old training frames)
    backbone   train all outputs, keep the feature layers (batch-norm statistics too)
    keep_old   as backbone, and only --train_classes outputs learn (default: new classes + LyingPerson + Pedestrian)
    auto       keep_old if 90% or more of the training frames are pose frames (ids >= 500000), else none
"""
import os
import sys
import json
import time
import random
import hashlib
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pp_common as C          # noqa: E402  (the copy next to this file)

OLD_NAMES = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
HEADS = {'cls_head': 1, 'box_head': 7, 'dir_head': 2}           # output channels per anchor


# ----------------------------------------------------------------------------- helpers
def load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


def resolve_root(path):
    p = os.path.abspath(os.path.expanduser(path))
    for cand in (p, os.path.join(p, 'training')):
        if os.path.isdir(os.path.join(cand, 'velodyne')):
            return cand
    raise SystemExit(f"no velodyne/ folder in {p} (or {os.path.join(p, 'training')}): pass --data_root")


def split_sha1(split):
    """same fingerprint as data_setup.split_sha1"""
    s = json.dumps({'train': list(split['train']), 'val': list(split['val'])}, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def read_split(root, path):
    if not os.path.isfile(path):
        raise SystemExit(f'{path} not found')
    with open(path) as f:
        split = json.load(f)
    for part in ('train', 'val'):
        miss = [i for i in split.get(part, []) if not os.path.exists(os.path.join(root, 'velodyne', i + '.bin'))
                and not os.path.exists(os.path.join(root, 'velodyne', i + '.npy'))]
        if miss:
            raise SystemExit(f'{len(miss)} {part} ids of {path} have no velodyne file (e.g. {miss[:3]})')
    return split


def check_paint_files(paint_dir, root, ids, n_probe=30):
    """as Fusion/Lidar_train.py: every frame painted, painted point count = velodyne point count"""
    missing = [i for i in ids if not os.path.exists(os.path.join(paint_dir, i + '.npy'))]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(ids)} train/val frames have no painted file (e.g. {missing[:3]}). '
                         f'Run paint_points.py (it skips frames already done).')
    probe = [ids[k] for k in np.unique(np.linspace(0, len(ids) - 1, min(n_probe, len(ids))).astype(int))]
    stale = [i for i in probe if np.load(os.path.join(paint_dir, i + '.npy'), mmap_mode='r').shape[0]
             != os.path.getsize(os.path.join(root, 'velodyne', i + '.bin')) // 16]
    if stale:
        raise SystemExit(f'painted files do not match velodyne/ for {stale[:3]}: run paint_points.py again')


def unknown_label_names(root, ids, label_dir, skip=('dontcare', 'ignore', 'misc')):
    """label names in the training labels that CLASS_ALIASES does not map (they would silently become background)"""
    seen = {}
    for i in ids:
        p = os.path.join(root, label_dir, i + '.txt')
        if not os.path.isfile(p):
            continue
        for line in open(p):
            f = line.split()
            if f:
                seen[f[0].lower()] = seen.get(f[0].lower(), 0) + 1
    return {k: n for k, n in seen.items() if k not in C.CLASS_ALIASES and k not in skip}


def class_counts(root, ids, label_dir):
    gt_n, ig_n = np.zeros(C.NUM_CLASSES, int), np.zeros(C.NUM_CLASSES, int)
    for i in ids:
        _, c, _, ic = C.load_labels(os.path.join(root, label_dir, i + '.txt'), return_ignore=True)
        gt_n += np.bincount(c, minlength=C.NUM_CLASSES)
        ig_n += np.bincount(ic, minlength=C.NUM_CLASSES)
    return gt_n, ig_n


def transfer(model, ckpt, old_names):
    """copy the old weights into the new model. Returns (n copied tensors, source class per new class)."""
    new_names = list(C.CLASS_NAMES)
    sd_old, sd = ckpt['model'], model.state_dict()
    n = 0
    for k, v in sd.items():
        if k.split('.')[0] in HEADS:
            continue
        if k not in sd_old or sd_old[k].shape != v.shape:
            raise SystemExit(f'{k}: the checkpoint does not fit this model ({tuple(sd_old[k].shape) if k in sd_old else "missing"}'
                             f' vs {tuple(v.shape)}); is --init a PointPillars checkpoint of this folder?')
        sd[k] = sd_old[k].clone()
        n += 1
    src = {}
    for c, name in enumerate(new_names):
        if name in old_names:
            src[c] = (old_names.index(name), True)
        elif 'Pedestrian' in old_names:
            src[c] = (old_names.index('Pedestrian'), False)
    for head, per in HEADS.items():
        for part in ('weight', 'bias'):
            k = f'{head}.{part}'
            new_t, old_t = sd[k], sd_old[k]
            if old_t.shape[0] != 2 * len(old_names) * per:
                raise SystemExit(f'{k}: {old_t.shape[0]} outputs, expected {2 * len(old_names) * per} for '
                                 f'{len(old_names)} classes')
            for c, (co, same) in src.items():
                if head == 'cls_head' and not same:
                    continue                                   # a new class: fresh classifier (prior bias)
                for r in range(2):                             # two anchor rotations per class
                    a, ao = 2 * c + r, 2 * co + r
                    new_t[a * per:(a + 1) * per] = old_t[ao * per:(ao + 1) * per]
            n += 1
    model.load_state_dict(sd)
    return n, {new_names[c]: (old_names[co] + ('' if same else ' (box / direction only)')) for c, (co, same) in src.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='fine-tune a trained PointPillars on the merged pose data')
    ap.add_argument('--init', default='', help='trained checkpoint to start from (Lidar_train.py best.pt)')
    ap.add_argument('--data_root', default=os.path.join(os.path.dirname(HERE), 'LIDAR_only', 'kitti_carla', 'training'))
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split_file', default='', help='default: <data_root>/split_lidar.json (after merge_pose_data.py)')
    ap.add_argument('--paint_name', default=None, help='fusion: painted folder (default: from the checkpoint)')
    ap.add_argument('--out_dir', default='', help='default: <folder of --init>_pose')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4, help='peak learning rate (one-cycle); Lidar_train.py uses 2e-3')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default='', help='continue this fine-tuning run (its last.pt)')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--freeze', default='auto', choices=['auto', 'none', 'backbone', 'keep_old'],
                    help='what stays as trained (see the top of this file); auto: keep_old for pose data alone')
    ap.add_argument('--train_classes', nargs='*', default=None,
                    help='keep_old: the classes whose outputs learn (default: new classes + LyingPerson + Pedestrian)')
    ap.add_argument('--first_id', type=int, default=500000, help='pose frames have ids >= this (for auto)')
    ap.add_argument('--check', action='store_true', help='print the data per class, save BEV images of pose frames, exit')
    args = ap.parse_args()

    if not args.init and not args.resume:
        raise SystemExit('pass --init <trained best.pt> (or --resume <this run\'s last.pt>)')
    seed_all(args.seed)
    root = resolve_root(args.data_root)
    args.data_root = root
    args.split_file = os.path.abspath(args.split_file or os.path.join(root, 'split_lidar.json'))
    split = read_split(root, args.split_file)
    sha = split_sha1(split)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = dev.type == 'cuda' and not args.no_amp
    print(f'[setup] device {dev}  amp {use_amp}  classes {C.CLASS_NAMES}')
    print(f"[split] {args.split_file}: train {len(split['train'])}  val {len(split['val'])}  (fingerprint {sha})")
    unknown = unknown_label_names(root, split['train'], args.label_dir)
    if unknown:
        raise SystemExit(f"labels_3d has names pp_common.py does not know: {unknown}. They would be trained as "
                         f"background. For the pose data (person_lying / person_sitting / person_crouching): run "
                         f"LIDAR_only\\add_pose_class.py first (patches both pp_common.py copies; "
                         f"add --lying-as-pedestrian to count them all as Pedestrian, no new class).")

    resume = load_ckpt(args.resume) if args.resume else None
    if resume is not None:          # the one-cycle schedule is fixed to the run's epoch count
        args.epochs = resume.get('args', {}).get('epochs', args.epochs)
    init = resume if resume is not None else load_ckpt(args.init)
    if resume is None:
        args.init = os.path.abspath(args.init)
    old_names = list(init.get('class_names') or OLD_NAMES)
    if resume is not None and old_names != list(C.CLASS_NAMES):
        raise SystemExit(f'--resume {args.resume}: classes {old_names} != {C.CLASS_NAMES}')
    if resume is None and old_names == list(C.CLASS_NAMES):
        print('[init] the checkpoint already has these classes: plain fine-tuning, all weights copied')
    lost = [n for n in old_names if n not in C.CLASS_NAMES]
    if lost:
        raise SystemExit(f'classes {lost} of the checkpoint are not in pp_common.CLASS_NAMES {C.CLASS_NAMES}')
    n_feats = int(init.get('num_pt_feats', 4))
    painted = n_feats > 4
    paint_dir, paint_meta, seg_names = None, {}, None
    if painted:
        pname = args.paint_name or init.get('paint_name') or 'velodyne_paint'
        paint_dir, n_paint, seg_names = C.paint_info(root, pname)
        if n_paint != n_feats:
            raise SystemExit(f'the checkpoint expects {n_feats} features per point, {paint_dir} gives {n_paint}')
        if init.get('paint_classes') is not None and list(init['paint_classes']) != list(seg_names):
            raise SystemExit(f"the checkpoint was painted with classes {init['paint_classes']}, {paint_dir} has {seg_names}")
        with open(os.path.join(paint_dir, 'paint_meta.json')) as f:
            paint_meta = json.load(f)
        same = lambda a_, b_: (a_ is None or b_ is None or
                               os.path.basename(os.path.normpath(str(a_))) == os.path.basename(os.path.normpath(str(b_))))
        if not same(init.get('paint_seg_model'), paint_meta.get('seg_model')) or \
                (init.get('paint_img_size') and paint_meta.get('img_size')
                 and list(init['paint_img_size']) != list(paint_meta['img_size'])):
            print(f"[paint] WARNING: the checkpoint was painted by {init.get('paint_seg_model')} at "
                  f"{init.get('paint_img_size')}, this data by {paint_meta.get('seg_model')} at "
                  f"{paint_meta.get('img_size')}: paint with the same model and --img_size, or the painted "
                  f"channels mean something else")
        check_paint_files(paint_dir, root, split['train'] + split['val'])
        args.paint_name = pname
        print(f'[paint] {paint_dir}: {len(seg_names)} segmentation classes -> {n_feats} features per point')
    out_dir = os.path.abspath(args.out_dir or (os.path.dirname(os.path.abspath(args.resume)) if args.resume
                                              else os.path.dirname(args.init) + '_pose'))
    args.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)

    def dataset(ids, train):
        if painted:
            return C.LidarDataset(root, ids, args.label_dir, train=train, paint_dir=paint_dir)
        return C.LidarDataset(root, ids, args.label_dir, train=train)

    train_ds, val_ds = dataset(split['train'], True), dataset(split['val'], False)
    gt_n, ig_n = class_counts(root, split['train'], args.label_dir)
    print('[data] train boxes per class (kept / ignored = fewer than MIN_PTS hits):')
    for k, name in enumerate(C.CLASS_NAMES):
        print(f'        {name:<12}{gt_n[k]:>8} /{ig_n[k]:>7}' + ('   <-- too few to learn' if gt_n[k] < 200 else ''))
    new_cls = [c for c, n in enumerate(C.CLASS_NAMES) if n not in old_names]
    if resume is None and any(gt_n[c] == 0 for c in new_cls):
        raise SystemExit(f'no {[C.CLASS_NAMES[c] for c in new_cls]} box in the training split: merge the new data '
                         f'(merge_pose_data.py) and check its labels_3d')

    if args.check:
        import cv2
        d = os.path.join(out_dir, 'check')
        os.makedirs(d, exist_ok=True)
        pose_ids = [i for i in split['train'] if i.isdigit() and int(i) >= 500000][:6] or split['train'][:6]
        plain = dataset(pose_ids, False)
        for k in range(len(plain)):
            s = plain[k]
            img = C.draw_bev(s['points'].numpy(), gt=s['boxes'].numpy(), ign=s['ign_boxes'].numpy(), size=1600)
            cv2.imwrite(os.path.join(d, f"{s['id']}.png"), img)
        print(f'[check] BEV images of {len(plain)} pose frames in {d}: the flat lying people need green boxes on '
              f'small point clusters (yellow = too few hits)')
        return

    # ---------------- model: old weights in, new class added ----------------
    if resume is not None:
        anchor_sizes = resume['anchor_sizes']
    else:
        stats, _ = train_ds.anchor_stats()
        old_sizes = init['anchor_sizes']
        anchor_sizes = [list(old_sizes[old_names.index(n)]) if n in old_names else [float(v) for v in stats[c]]
                        for c, n in enumerate(C.CLASS_NAMES)]
        print('[anchors] [l,w,h,z]: ' + ', '.join(f'{n} {[round(v, 2) for v in a]}' + ('' if n in old_names else ' (new, from data)')
                                                  for n, a in zip(C.CLASS_NAMES, anchor_sizes)))
    model = C.PointPillars(anchor_sizes, num_pt_feats=n_feats).to(dev)
    if resume is not None:
        model.load_state_dict(resume['model'])            # before the frozen values are taken below
    else:
        n_t, src = transfer(model, init, old_names)
        print(f'[init] {n_t} tensors copied from {args.init}; heads: ' + ', '.join(f'{k} <- {v}' for k, v in src.items()))
        model.to(dev)

    # ---------------- what learns ----------------
    pose_share = np.mean([i.isdigit() and int(i) >= args.first_id for i in split['train']]) if split['train'] else 0.0
    freeze = resume.get('freeze', args.freeze) if resume is not None else args.freeze
    if freeze == 'auto':
        freeze = 'keep_old' if pose_share >= 0.9 else 'none'
        print(f'[freeze] auto: {100 * pose_share:.0f}% of the training frames are pose frames -> --freeze {freeze}'
              + ('  (pass --freeze none to train everything)' if freeze == 'keep_old' else ''))
    elif freeze == 'none' and pose_share >= 0.9:
        print(f'[freeze] WARNING: {100 * pose_share:.0f}% of the training frames are pose frames and everything trains: '
              f'the model can forget trucks, buses, cyclists and traffic (--freeze keep_old avoids it)')
    args.freeze = freeze
    pose_cls = [n for n in ('LyingPerson', 'Pedestrian') if n in C.CLASS_NAMES]   # what the pose data teaches
    learn = list((resume or {}).get('learned_classes') or args.train_classes
                 or dict.fromkeys([n for n in C.CLASS_NAMES if n not in old_names] + pose_cls))
    bad = [n for n in learn if n not in C.CLASS_NAMES]
    if bad:
        raise SystemExit(f'--train_classes {bad} not in {C.CLASS_NAMES}')
    frozen_mods = []
    keep_rows = {}                         # head parameter -> (row mask to restore, copy of the trained values)
    if freeze in ('backbone', 'keep_old'):
        for name, mod in model.named_children():
            if name not in HEADS:
                frozen_mods.append(mod)
                for prm in mod.parameters():
                    prm.requires_grad_(False)
    if freeze == 'keep_old':
        on = set(C.CLASS_NAMES.index(n) for n in learn)
        for head, per in HEADS.items():
            for part in ('weight', 'bias'):
                prm = getattr(getattr(model, head), part)
                rows = torch.ones(prm.shape[0], dtype=torch.bool, device=prm.device)
                for c in on:
                    rows[2 * c * per:(2 * c + 2) * per] = False          # these rows learn
                keep_rows[prm] = (rows, prm.detach().clone())
        print(f"[freeze] keep_old: only the {', '.join(learn)} outputs learn; feature layers and the "
              f"{', '.join(n for n in C.CLASS_NAMES if n not in learn)} outputs stay exactly as trained")
    elif freeze == 'backbone':
        print('[freeze] backbone: the feature layers stay as trained, all outputs learn')

    def set_train_mode():
        model.train()
        for mod in frozen_mods:
            mod.eval()                     # frozen layers keep their batch-norm statistics

    def restore_frozen_rows():
        with torch.no_grad():
            for prm, (rows, val) in keep_rows.items():
                prm[rows] = val[rows]

    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, collate_fn=C.collate,
                           drop_last=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init, generator=g)
    va_loader = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers, collate_fn=C.collate,
                           persistent_workers=args.workers > 0, worker_init_fn=worker_init)
    if len(tr_loader) == 0:
        raise SystemExit(f'{len(train_ds)} training frames < --batch {args.batch}')
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(tr_loader), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_ep, best = 0, float('inf')
    if resume is not None:
        opt.load_state_dict(resume['opt'])
        sched.load_state_dict(resume['sched'])
        start_ep, best = resume['epoch'] + 1, resume['best_val']
        args.init = resume.get('init_ckpt', '')
        print(f'[resume] from epoch {start_ep}')
        if start_ep >= args.epochs:
            print(f'[resume] this run already finished its {args.epochs} epochs: nothing to do')
            return

    def save(path, ep):
        d = {'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': ep,
             'best_val': best, 'anchor_sizes': anchor_sizes, 'class_names': C.CLASS_NAMES, 'num_pt_feats': n_feats,
             'split_sha1': sha, 'init_ckpt': args.init, 'finetuned': True, 'freeze': freeze, 'learned_classes': learn,
             'args': vars(args)}
        if painted:
            d.update(paint_name=args.paint_name, paint_classes=list(seg_names),
                     paint_seg_model=paint_meta.get('seg_model'), paint_img_size=paint_meta.get('img_size'))
        torch.save(d, path + '.tmp')
        os.replace(path + '.tmp', path)

    log_path = os.path.join(out_dir, 'training_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,seconds,skipped_batches\n')
    for ep in range(start_ep, args.epochs):
        set_train_mode()
        t0, run, n_ok, n_skip = time.time(), 0.0, 0, 0
        for it, b in enumerate(tr_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(b['points'])
            loss, parts = C.compute_loss(out, model, b['boxes'], b['cls'], b['ign_boxes'], b['ign_cls'])
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                n_skip += 1
                print(f"  [warn] non-finite loss at ep {ep + 1} it {it} (frames {b['id']}): batch skipped")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            restore_frozen_rows()
            run += float(loss)
            n_ok += 1
            if it % 50 == 0:
                print(f"ep {ep + 1}/{args.epochs} it {it}/{len(tr_loader)} loss {float(loss):.3f} (cls {parts['cls']:.3f} "
                      f"box {parts['box']:.3f} dir {parts['dir']:.3f}) pos {parts['npos']} lr {sched.get_last_lr()[0]:.2e}")
        tr_loss = run / max(n_ok, 1)
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for b in va_loader:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(b['points'])
                l, _ = C.compute_loss(out, model, b['boxes'], b['cls'], b['ign_boxes'], b['ign_cls'])
                if torch.isfinite(l):
                    vl += float(l) * len(b['id'])
                    n += len(b['id'])
        val_loss = vl / n if n else float('nan')
        secs = time.time() - t0
        print(f'=== epoch {ep + 1}: train {tr_loss:.4f}  val {val_loss:.4f}  ({secs:.0f}s)'
              + (f'  [{n_skip} batches skipped]' if n_skip else ''))
        with open(log_path, 'a') as f:
            f.write(f'{ep + 1},{tr_loss:.5f},{val_loss:.5f},{secs:.0f},{n_skip}\n')
        if n and val_loss < best:
            best = val_loss
            save(os.path.join(out_dir, 'best.pt'), ep)
            print('    saved best.pt')
        save(os.path.join(out_dir, 'last.pt'), ep)

    best_path = os.path.join(out_dir, 'best.pt')
    print(f'Done: {best_path}\nEvaluate (same commands for both models):')
    print(f'    python Lidar_val.py --ckpt {best_path}' + (' --fov_only' if painted else '')
          + ('      (AP per class, LyingPerson included)' if 'LyingPerson' in C.CLASS_NAMES else
             '      (AP per class; lying people count as Pedestrian)'))
    print(f'    python pose_eval.py --ckpt {best_path}      (lying / sitting / crouching: found or not)')
    flag = '--fusion-ckpt' if painted else '--lidar-ckpt'
    print(f'Driving stack: {flag} {best_path}')


if __name__ == '__main__':      # required on Windows for DataLoader workers
    main()