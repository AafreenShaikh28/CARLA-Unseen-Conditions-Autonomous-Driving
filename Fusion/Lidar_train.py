"""
Lidar_train.py (Fusion copy) - train the PointPainting fusion model: PointPillars on LiDAR points that
carry the SegFormer class probabilities from paint_points.py.

Files needed next to it: pp_common.py and data_setup.py (the Fusion copies).
Works from any working directory; the defaults point at
    data   python_code/LIDAR_only/kitti_carla/training   (velodyne/ + labels_3d/ + split_lidar.json
                                                          + velodyne_paint/ from paint_points.py)
    runs   python_code/Fusion/runs_lidar/pp_painted_kitti_carla

    python Lidar_train.py --check                 # 1) labels on points + painted channels present
    python Lidar_train.py                         # 2) train (same settings as the LiDAR-only run)
    python Lidar_train.py --resume runs_lidar/pp_painted_kitti_carla/last.pt

Everything that is not painting is identical to Detection_LIDAR/Lidar_train.py (model, anchors from the
same labels, loss, schedule, seed, split), so the two models are a fair comparison. Differences:
  - points have 4 + n_classes + 1 features (the painted channels), num_pt_feats is stored in the checkpoint
  - the split is READ ONLY: <data_root>/split_lidar.json, the one the LiDAR-only model trained on.
    No --resplit here; a missing or invalid split stops the run. Its fingerprint goes into the checkpoint.
  - the LiDAR-only checkpoint (if found) is compared: different split file / epochs / batch / lr / seed
    are printed as warnings, because they would make the comparison unfair
  - every train and val frame must be painted, by one segmentation model (checked before training)
"""
import os
import json
import time
import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import pp_common as C
import data_setup as D

SAME_AS_LIDAR = ('epochs', 'batch', 'lr', 'seed', 'label_dir')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def worker_init(worker_id):
    # each worker gets its own numpy stream derived from torch's per-worker seed
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


def check_paint_files(paint_dir, root, ids, n_probe=30):
    """Every frame painted; for a probe of frames the painted file matches the point count."""
    missing = [i for i in ids if not os.path.exists(os.path.join(paint_dir, i + '.npy'))]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(ids)} train/val frames have no painted file '
                         f'(e.g. {missing[:3]}). Run paint_points.py (it skips frames already done).')
    probe = [ids[k] for k in np.unique(np.linspace(0, len(ids) - 1, min(n_probe, len(ids))).astype(int))]
    stale, incam = [], []
    for i in probe:
        ex = np.load(os.path.join(paint_dir, i + '.npy'), mmap_mode='r')
        n = os.path.getsize(os.path.join(root, 'velodyne', i + '.bin')) // 16
        if ex.shape[0] != n:
            stale.append(i)
        else:
            incam.append(float((np.asarray(ex[:, -1]) > 0).mean()))
    if stale:
        raise SystemExit(f'painted files do not match velodyne/ for {stale[:3]} (data re-collected?). '
                         f'Run paint_points.py again: it repaints changed frames.')
    return float(np.mean(incam)) if incam else 0.0


def compare_with_lidar_run(args, split_sha):
    """Warn about settings that differ from the LiDAR-only model (unfair comparison)."""
    if not os.path.isfile(args.lidar_ckpt):
        print(f'[fair] LiDAR-only checkpoint not found ({args.lidar_ckpt}): settings not compared. '
              f'Train it with the same split, epochs, batch, lr and seed.')
        return
    la = D.load_ckpt(args.lidar_ckpt).get('args', {})
    diffs = [f'{k}: LiDAR-only {la.get(k)!r} vs fusion {getattr(args, k)!r}' for k in SAME_AS_LIDAR
             if k in la and la.get(k) != getattr(args, k)]
    l_split = la.get('split_file')
    if l_split and os.path.normcase(os.path.realpath(l_split)) != os.path.normcase(os.path.realpath(args.split_file)):
        diffs.append(f'split file: LiDAR-only {l_split} vs fusion {args.split_file}')
    if diffs:
        print('[fair] WARNING: settings differ from the LiDAR-only run, so the comparison is not like for like:')
        for d in diffs:
            print(f'        - {d}')
    else:
        print(f'[fair] same split file and settings as {args.lidar_ckpt} (split {split_sha})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=D.DEFAULT_DATA_ROOT,
                    help='folder with velodyne/, labels_3d/ and the painted folder (kitti_carla/training)')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--split_file', default='', help='default: <data_root>/split_lidar.json (read only)')
    ap.add_argument('--paint_name', default=D.DEFAULT_PAINT_NAME, help='folder written by paint_points.py')
    ap.add_argument('--out_dir', default=D.DEFAULT_OUT_DIR)
    ap.add_argument('--lidar_ckpt', default=D.DEFAULT_LIDAR_CKPT,
                    help='LiDAR-only checkpoint, only used to check the settings match')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4)          # lower to 2 if you run out of VRAM
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume', default='')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--painted', action='store_true', help='ignored (painting is always on here; kept so '
                                                           'old commands still run)')
    ap.add_argument('--check', action='store_true',
                    help='report data + painting statistics, dump BEV images of GT boxes and exit')
    args = ap.parse_args()

    seed_all(args.seed)
    args.data_root = D.resolve_data_root(args.data_root, args.label_dir)
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (dev.type == 'cuda') and not args.no_amp
    print(f'[setup] device {dev}' + (f' ({torch.cuda.get_device_name(0)})' if dev.type == 'cuda' else '')
          + f'  amp {use_amp}')

    # ---------------- split: the LiDAR-only model's, read only ----------------
    args.split_file = os.path.abspath(args.split_file or os.path.join(args.data_root, 'split_lidar.json'))
    split = D.read_split(args.data_root, args.split_file)
    split_sha = D.split_sha1(split)
    print(f"[split] loaded {args.split_file}: train={len(split['train'])} frames  val={len(split['val'])} "
          f"frames  (fingerprint {split_sha})")

    # ---------------- painted channels ----------------
    paint_dir, n_feats, seg_names = C.paint_info(args.data_root, args.paint_name)
    with open(os.path.join(paint_dir, 'paint_meta.json')) as f:
        paint_meta = json.load(f)
    incam = check_paint_files(paint_dir, args.data_root, split['train'] + split['val'])
    print(f'[paint] {paint_dir}: {len(seg_names)} classes -> {n_feats} features per point; '
          f'{100 * incam:.0f}% of points are in the camera view (painted)')
    compare_with_lidar_run(args, split_sha)

    # ---------------- label / data sanity check ----------------
    if args.check:
        import cv2
        D.dataset_report(args.data_root, args.label_dir, n_probe=30)
        d = os.path.join(args.out_dir, 'check')
        os.makedirs(d, exist_ok=True)
        plain = C.LidarDataset(args.data_root, split['train'], args.label_dir, train=False, paint_dir=paint_dir)
        pillars = []
        for k in np.unique(np.linspace(0, len(plain) - 1, 8).astype(int)):
            s = plain[int(k)]
            pts = s['points'].numpy()
            if pts.shape[1] != n_feats:
                raise SystemExit(f"frame {s['id']}: {pts.shape[1]} features per point, expected {n_feats}")
            pillars.append(C.count_pillars(pts))
            img = C.draw_bev(pts, gt=s['boxes'].numpy(), ign=s['ign_boxes'].numpy(), size=1600)
            cv2.imwrite(os.path.join(d, f"{s['id']}.png"), img)
        print(f'[check] wrote BEV images to {d}')
        print('        Green boxes must sit on the point clusters, arrow pointing in the driving')
        print('        direction; yellow = ignore boxes (too few hits). Left in the image = left')
        print('        of the car. If not, edit Z_IS_BOTTOM / YAW_SIGN / YAW_OFFSET in BOTH pp_common.py files')
        print(f'[check] non-empty pillars per frame: min {min(pillars)}  max {max(pillars)}  '
              f'(MAX_PILLARS_TRAIN = {C.MAX_PILLARS_TRAIN}, MAX_PILLARS_TEST = {C.MAX_PILLARS_TEST})')
        if max(pillars) > C.MAX_PILLARS_TRAIN:
            print('        WARNING: frames have more pillars than MAX_PILLARS_TRAIN -> random '
                  'pillars (including object pillars) get dropped during training. Raise it.')
        gt_n, ig_n = np.zeros(C.NUM_CLASSES, int), np.zeros(C.NUM_CLASSES, int)
        for i in split['train']:
            lp = os.path.join(args.data_root, args.label_dir, i + '.txt')
            _, c, _, ic = C.load_labels(lp, return_ignore=True)
            gt_n += np.bincount(c, minlength=C.NUM_CLASSES)
            ig_n += np.bincount(ic, minlength=C.NUM_CLASSES)
        print('[check] train boxes per class (kept / ignored, inside PC_RANGE):')
        for k, n in enumerate(C.CLASS_NAMES):
            note = '   <-- too few to learn this class' if gt_n[k] < 200 else ''
            print(f'        {n:<11}{gt_n[k]:>8} /{ig_n[k]:>7}{note}')
        print('[check] painting quality itself: python check_projection.py and python check_paint.py')
        return

    D.dataset_report(args.data_root, args.label_dir, n_probe=8)   # short version, never stops training

    train_ds = C.LidarDataset(args.data_root, split['train'], args.label_dir, train=True, paint_dir=paint_dir)
    val_ds = C.LidarDataset(args.data_root, split['val'], args.label_dir, train=False, paint_dir=paint_dir)
    if len(train_ds) < args.batch:
        raise SystemExit(f'{len(train_ds)} training frames < --batch {args.batch}')

    # ---------------- model ----------------
    ckpt = D.load_ckpt(args.resume) if args.resume else None
    if ckpt is not None:
        if list(ckpt.get('class_names', [])) != C.CLASS_NAMES:
            raise SystemExit(f"checkpoint classes {ckpt.get('class_names')} != {C.CLASS_NAMES}")
        if ckpt.get('num_pt_feats', 4) != n_feats or list(ckpt.get('paint_classes', [])) != seg_names:
            raise SystemExit(f"checkpoint was trained on {ckpt.get('num_pt_feats')} features / segmentation "
                             f"classes {ckpt.get('paint_classes')}; the painted data now has {n_feats} / "
                             f"{seg_names}. Start a new run (new --out_dir).")
        if ckpt.get('split_sha1') not in (None, split_sha):
            raise SystemExit(f"checkpoint used split {ckpt.get('split_sha1')}, {args.split_file} is now {split_sha}")
        ck_epochs = ckpt.get('args', {}).get('epochs')
        if ck_epochs is not None and ck_epochs != args.epochs:
            # the saved learning-rate schedule is built for the original epoch count; a different
            # --epochs would crash OneCycleLR after the last original step
            print(f'[resume] --epochs {args.epochs} ignored: the run was started with {ck_epochs} epochs and its '
                  f'learning-rate schedule is fixed to that. Start a fresh run (new --out_dir) for more epochs.')
            args.epochs = ck_epochs
        anchor_sizes = ckpt['anchor_sizes']
    else:
        anchor_sizes, counts = train_ds.anchor_stats()
        print('[anchors] [l,w,h,z] per class:', [[round(v, 2) for v in a] for a in anchor_sizes],
              ' GT counts:', dict(zip(C.CLASS_NAMES, counts)))
        if sum(counts) == 0:
            raise SystemExit('no usable training boxes found. Run --check; the class names in '
                             'labels_3d must match CLASS_ALIASES in pp_common.py')
    model = C.PointPillars(anchor_sizes, num_pt_feats=n_feats).to(dev)

    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                           collate_fn=C.collate, drop_last=True, persistent_workers=args.workers > 0,
                           worker_init_fn=worker_init, generator=g)
    va_loader = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers,
                           collate_fn=C.collate, persistent_workers=args.workers > 0,
                           worker_init_fn=worker_init)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(tr_loader), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_ep, best = 0, float('inf')
    if ckpt is not None:
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched'])
        start_ep, best = ckpt['epoch'] + 1, ckpt['best_val']
        print(f'[resume] from epoch {start_ep} (learning-rate schedule and epoch count come from the '
              f'checkpoint)')
        if start_ep >= args.epochs:
            print(f'[resume] checkpoint already finished {start_ep} epochs >= --epochs {args.epochs}: '
                  f'nothing to do (raise --epochs only together with a fresh run)')
            return

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.out_dir, 'tb'))
    except Exception:
        tb = None

    def save(path, ep):
        tmp = path + '.tmp'
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'best_val': best, 'anchor_sizes': anchor_sizes,
                    'class_names': C.CLASS_NAMES, 'num_pt_feats': n_feats,
                    'paint_name': args.paint_name, 'paint_classes': seg_names, 'split_sha1': split_sha,
                    'paint_seg_model': paint_meta.get('seg_model'), 'paint_img_size': paint_meta.get('img_size'),
                    'args': vars(args)}, tmp)
        os.replace(tmp, path)

    log_path = os.path.join(args.out_dir, 'training_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,seconds,skipped_batches\n')

    # ---------------- train ----------------
    for ep in range(start_ep, args.epochs):
        model.train()
        t0, run, n_ok, n_skip = time.time(), 0.0, 0, 0
        for it, b in enumerate(tr_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(b['points'])
            loss, parts = C.compute_loss(out, model, b['boxes'], b['cls'], b['ign_boxes'], b['ign_cls'])
            if not torch.isfinite(loss):
                n_skip += 1
                print(f"  [warn] non-finite loss at ep {ep + 1} it {it} (frames {b['id']}): batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            n_ok += 1
            if it % 50 == 0:
                print(f'ep {ep + 1}/{args.epochs} it {it}/{len(tr_loader)} loss {float(loss):.3f} '
                      f"(cls {parts['cls']:.3f} box {parts['box']:.3f} dir {parts['dir']:.3f}) "
                      f"pos {parts['npos']} lr {sched.get_last_lr()[0]:.2e}")
        tr_loss = run / max(n_ok, 1)

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for b in va_loader:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(b['points'])
                l, _ = C.compute_loss(out, model, b['boxes'], b['cls'], b['ign_boxes'], b['ign_cls'])
                if not torch.isfinite(l):
                    continue
                vl += float(l) * len(b['id'])
                n += len(b['id'])
        val_loss = vl / n if n else float('nan')
        secs = time.time() - t0
        print(f'=== epoch {ep + 1}: train {tr_loss:.4f}  val {val_loss:.4f}  ({secs:.0f}s)'
              + (f'  [{n_skip} batches skipped]' if n_skip else ''))
        with open(log_path, 'a') as f:
            f.write(f'{ep + 1},{tr_loss:.5f},{val_loss:.5f},{secs:.0f},{n_skip}\n')
        if tb:
            tb.add_scalar('loss/train', tr_loss, ep)
            if n:
                tb.add_scalar('loss/val', val_loss, ep)
        if n and val_loss < best:
            best = val_loss
            save(os.path.join(args.out_dir, 'best.pt'), ep)
            print('    saved best.pt')
        save(os.path.join(args.out_dir, 'last.pt'), ep)

    best_path = os.path.join(args.out_dir, 'best.pt')
    print('Done. Compare on the camera view (where painting exists), same command for both models:')
    print('    python Lidar_val.py --ckpt', best_path, '--fov_only')
    print('    python Lidar_val.py --ckpt', args.lidar_ckpt, '--fov_only')
    hw = paint_meta.get('img_size') or ['?', '?']
    print('Use in the Driving stack with the SAME segmentation model and input size as the painting:')
    print(f"    --fusion-ckpt {best_path} --seg-model {paint_meta.get('seg_model')} --seg-img-size {hw[0]} {hw[1]}")


if __name__ == '__main__':      # required on Windows for DataLoader workers
    main()
