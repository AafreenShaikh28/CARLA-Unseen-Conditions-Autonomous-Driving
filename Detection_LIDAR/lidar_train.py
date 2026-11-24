"""
lidar_train.py - train PointPillars on the CARLA LiDAR dataset (labels_3d, LiDAR frame).

Run from python_code/Detection_LIDAR/ :
    python lidar_train.py --check                      # 1) labels vs points + pillar budget
    python lidar_train.py --data_root ../data_collection_lidar/dataset/Town04 --resplit
    python lidar_train.py --resume runs_lidar/pp_town04/last.pt

Changes vs the previous version: see the FIX list at the top of pp_common.py, plus
  - ignore boxes are passed to the loss (training and validation loss)
  - seeds for python/numpy/torch and for every DataLoader worker, so a run can be repeated
    and workers never share one numpy random state (identical "random" augmentations)
  - --check also prints the pillar count per frame and GT / ignore counts per class
  - class names are saved in the checkpoint and checked on --resume
"""
import os, json, time, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pp_common as C


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def worker_init(worker_id):
    # each worker gets its own numpy stream derived from torch's per-worker seed
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../LIDAR_only/kitti_carla/training')
    ap.add_argument('--label_dir', default='labels_3d')
    ap.add_argument('--out_dir', default='runs_lidar/pp_kitti_carla')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=4)          # lower to 2 if you run out of VRAM
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--val_frac', type=float, default=0.2)
    ap.add_argument('--gap', type=int, default=100)
    ap.add_argument('--resplit', action='store_true')
    ap.add_argument('--resume', default='')
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--check', action='store_true', help='dump BEV images of GT boxes and exit')
    args = ap.parse_args()

    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (dev.type == 'cuda') and not args.no_amp

    # ---------------- split (saved to disk so lidar_val.py uses the same one) ----------------
    split_path = os.path.join(args.data_root, 'split_lidar.json')
    if args.resplit or not os.path.exists(split_path):
        split = C.make_split(args.data_root, split_path, args.val_frac, args.gap)
        print(f'[split] created {split_path}')
        print('[split] WARNING: temporal split inside ONE recording -> val is near the training scenes, '
              'so AP is optimistic. collect_kitti_lidar.py writes a town-based split_lidar.json; '
              'never use --resplit on that folder.')
    else:
        split = json.load(open(split_path))
    print(f"[split] train={len(split['train'])} frames  val={len(split['val'])} frames")

    train_ds = C.LidarDataset(args.data_root, split['train'], args.label_dir, train=True)
    val_ds = C.LidarDataset(args.data_root, split['val'], args.label_dir, train=False)

    # ---------------- label sanity check ----------------
    if args.check:
        import cv2
        d = os.path.join(args.out_dir, 'check'); os.makedirs(d, exist_ok=True)
        plain = C.LidarDataset(args.data_root, split['train'], args.label_dir, train=False)
        pillars = []
        for k in np.linspace(0, len(plain) - 1, 8).astype(int):
            s = plain[k]
            pts = s['points'].numpy()
            pillars.append(C.count_pillars(pts))
            img = C.draw_bev(pts, gt=s['boxes'].numpy(), ign=s['ign_boxes'].numpy(), size=1600)
            cv2.imwrite(os.path.join(d, f"{s['id']}.png"), img)
        print(f'[check] wrote BEV images to {d}')
        print('        Green boxes must sit on the point clusters, arrow pointing in the driving')
        print('        direction; yellow = ignore boxes (too few hits). Left in the image = left')
        print('        of the car. If not, edit Z_IS_BOTTOM / YAW_SIGN / YAW_OFFSET in pp_common.py')
        print(f'[check] non-empty pillars per frame: min {min(pillars)}  max {max(pillars)}  '
              f'(MAX_PILLARS_TRAIN = {C.MAX_PILLARS_TRAIN}, MAX_PILLARS_TEST = {C.MAX_PILLARS_TEST})')
        if max(pillars) > C.MAX_PILLARS_TRAIN:
            print('        WARNING: frames have more pillars than MAX_PILLARS_TRAIN -> random '
                  'pillars (including object pillars) get dropped during training. Raise it.')
        gt_n, ig_n = np.zeros(C.NUM_CLASSES, int), np.zeros(C.NUM_CLASSES, int)
        for i in split['train']:
            _, c, _, ic = C.load_labels(os.path.join(args.data_root, args.label_dir, i + '.txt'),
                                        return_ignore=True)
            gt_n += np.bincount(c, minlength=C.NUM_CLASSES)
            ig_n += np.bincount(ic, minlength=C.NUM_CLASSES)
        print('[check] train boxes per class (kept / ignored, inside PC_RANGE):')
        for k, n in enumerate(C.CLASS_NAMES):
            note = '   <-- too few to learn this class' if gt_n[k] < 200 else ''
            print(f'        {n:<11}{gt_n[k]:>8} /{ig_n[k]:>7}{note}')
        return

    # ---------------- model ----------------
    ckpt = torch.load(args.resume, map_location='cpu') if args.resume else None
    if ckpt is not None:
        if list(ckpt.get('class_names', [])) != C.CLASS_NAMES:
            raise SystemExit(f"checkpoint classes {ckpt.get('class_names')} != {C.CLASS_NAMES}")
        anchor_sizes = ckpt['anchor_sizes']
    else:
        anchor_sizes, counts = train_ds.anchor_stats()
        print('[anchors] [l,w,h,z] per class:', [[round(v, 2) for v in a] for a in anchor_sizes],
              ' GT counts:', dict(zip(C.CLASS_NAMES, counts)))
    model = C.PointPillars(anchor_sizes, num_pt_feats=4).to(dev)

    g = torch.Generator(); g.manual_seed(args.seed)
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
        model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['opt'])
        sched.load_state_dict(ckpt['sched']); start_ep = ckpt['epoch'] + 1; best = ckpt['best_val']
        print(f'[resume] from epoch {start_ep}')

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(args.out_dir, 'tb'))
    except Exception:
        tb = None

    def save(path, ep):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'best_val': best, 'anchor_sizes': anchor_sizes,
                    'class_names': C.CLASS_NAMES, 'num_pt_feats': 4, 'args': vars(args)}, path)

    # ---------------- train ----------------
    for ep in range(start_ep, args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for it, b in enumerate(tr_loader):
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(b['points'])
            loss, parts = C.compute_loss(out, model, b['boxes'], b['cls'], b['ign_boxes'], b['ign_cls'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += float(loss)
            if it % 50 == 0:
                print(f'ep {ep+1}/{args.epochs} it {it}/{len(tr_loader)} loss {float(loss):.3f} '
                      f"(cls {parts['cls']:.3f} box {parts['box']:.3f} dir {parts['dir']:.3f}) "
                      f"pos {parts['npos']} lr {sched.get_last_lr()[0]:.2e}")
        tr_loss = run / max(len(tr_loader), 1)

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for b in va_loader:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(b['points'])
                l, _ = C.compute_loss(out, model, b['boxes'], b['cls'], b['ign_boxes'], b['ign_cls'])
                vl += float(l) * len(b['id']); n += len(b['id'])
        val_loss = vl / max(n, 1)
        print(f'=== epoch {ep+1}: train {tr_loss:.4f}  val {val_loss:.4f}  ({time.time()-t0:.0f}s)')
        if tb:
            tb.add_scalar('loss/train', tr_loss, ep); tb.add_scalar('loss/val', val_loss, ep)
        if val_loss < best:
            best = val_loss
            save(os.path.join(args.out_dir, 'best.pt'), ep)
            print('    saved best.pt')
        save(os.path.join(args.out_dir, 'last.pt'), ep)

    print('Done. Evaluate with: python lidar_val.py --ckpt', os.path.join(args.out_dir, 'best.pt'))


if __name__ == '__main__':      # required on Windows for DataLoader workers
    main()
