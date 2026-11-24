"""
compare_offline.py - offline comparison table and charts: LiDAR-only vs fusion 3D AP, and painting
accuracy (SegFormer labels on LiDAR points vs CARLA's semantic camera), per weather.
Put it in the Fusion folder. It only reads results; run the evaluations first:


    $L = "..\\Detection_LIDAR\\runs_lidar\\pp_kitti_carla\\best.pt"
    foreach ($w in "ClearNoon", "HardRainNoon", "ClearNight", "HardRainNight", "Fog") {
      $root = "..\\LIDAR_only\\kitti_weather\\$w\\training"
      New-Item -ItemType Directory -Force "runs_eval\\$w" | Out-Null
      python paint_points.py --data_root $root
      python check_paint.py --data_root $root --n 100 | Tee-Object "runs_eval\\$w\\check_paint.txt"
      python Lidar_val.py --data_root $root --split all --fov_only --out_dir "runs_eval\\$w\\fusion"
      python Lidar_val.py --data_root $root --split all --fov_only --ckpt $L --out_dir "runs_eval\\$w\\lidar"
    }
    python compare_offline.py


Reads runs_eval/<Weather>/{lidar,fusion}/metrics.json and runs_eval/<Weather>/check_paint.txt, writes
runs_eval/offline_comparison.csv, offline_ap_by_weather.png, offline_paint_by_weather.png.


With the LiDAR rain / fog model (Driving/lidar_weather.py): make the degraded copy, then the same loop with
$root = "..\\LIDAR_only\\kitti_weather_lidarwx\\$w\\training" and "runs_eval_lidarwx\\..." folders
(skip check_paint: painting accuracy depends on the camera, not on the LiDAR), then
    python compare_offline.py --eval runs_eval_lidarwx --lidar-weather
"""
import os
import re
import csv
import json
import glob
import argparse
import textwrap


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
COL = {'lidar': '#2a78d6', 'fusion': '#eb6834'}
NAME = {'lidar': 'LiDAR only', 'fusion': 'Fusion (PointPainting)'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
ORDER = ['ClearNoon', 'CloudySunset', 'HardRainNoon', 'ClearNight', 'HardRainNight', 'Fog']
SHORT = {'ClearNoon': 'Clear\n(seen)', 'CloudySunset': 'Sunset\n(seen)', 'HardRainNoon': 'Heavy\nrain',
         'ClearNight': 'Night', 'HardRainNight': 'Rain +\nnight', 'Fog': 'Fog'}
MIN_FRAMES = 50          # sets with fewer frames are left out of the charts (still in the CSV and table)
plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.facecolor': SURF,
                     'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'legend.frameon': False,
                     'axes.spines.top': False, 'axes.spines.right': False})




def read_text(path):
    """check_paint output saved by PowerShell's Tee-Object is UTF-16; by cmd / bash it is UTF-8"""
    with open(path, 'rb') as fh:
        raw = fh.read()
    for enc in (('utf-16',) if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else ()) + ('utf-8', 'cp1252'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')




def paint_rows(path):
    """{class name: % of LiDAR points painted with the right class} from section 2 of check_paint.py"""
    if not os.path.exists(path):
        return {}
    text = read_text(path)
    i = text.find('2) Painting vs CARLA')
    out = {}
    for line in text[i:].splitlines()[1:] if i >= 0 else []:
        m = re.match(r'^(\S.*?)\s+(\d+)\s+([\d.]+)%', line.strip())
        if m:
            out[m.group(1).strip()] = float(m.group(3))
    return out




def load(eval_dir):
    rows = []
    for wdir in sorted(glob.glob(os.path.join(eval_dir, '*'))):
        if not os.path.isdir(wdir):
            continue
        w = os.path.basename(wdir)
        paint = paint_rows(os.path.join(wdir, 'check_paint.txt'))
        for src in ('lidar', 'fusion'):
            mp = os.path.join(wdir, src, 'metrics.json')
            if not os.path.exists(mp):
                continue
            with open(mp) as fh:
                m = json.load(fh)
            r = {'weather': w, 'source': src, 'frames': m.get('frames'),
                 'mAP_3D': m.get('mAP', {}).get('3D@std'), 'mAP_BEV': m.get('mAP', {}).get('BEV@std'),
                 'mAP_classes': ' '.join(m.get('mAP', {}).get('classes', [])), 'fps': m.get('fps')}
            for c, v in m.items():
                if isinstance(v, dict) and '3D@std' in v and c != 'mAP':
                    r[f'{c}_3D_AP'] = v.get('3D@std')
                    r[f'{c}_gt'] = v.get('gt')
            if src == 'fusion':
                for k, v in paint.items():
                    r[f'paint_{k}_pct'] = v
            rows.append(r)
    return rows




def bars(ax, rows, key, title, ylabel, fmt='{:.1f}'):
    ws = sorted({r['weather'] for r in rows if r.get(key) is not None},
                key=lambda w: (ORDER.index(w) if w in ORDER else 99, w))
    srcs = [s for s in ('lidar', 'fusion') if any(r['source'] == s and r.get(key) is not None for r in rows)]
    if not ws:
        ax.set_visible(False)
        return
    width = 0.8 / len(srcs)
    for j, s in enumerate(srcs):
        first = True
        for i, w in enumerate(ws):
            v = [r[key] for r in rows if r['source'] == s and r['weather'] == w and r.get(key) is not None]
            if not v or v[0] != v[0]:          # missing or NaN
                continue
            x = i - 0.4 + width * (j + 0.5)
            ax.bar(x, v[0], width, color=COL[s], edgecolor=SURF, linewidth=2, label=NAME[s] if first else None)
            ax.text(x, v[0], fmt.format(v[0]), ha='center', va='bottom', fontsize=9, color=INK2)
            first = False
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([SHORT.get(w, w) for w in ws], fontsize=11)
    ax.set_title(title, loc='left')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(ax.get_ylim()[1] * 1.08, 1.0))




def figure(rows, panels, title, path, note):
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 9))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        bars(ax, rows, *p)
    handles, labels = [], []
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    if not handles:
        plt.close(fig)
        return None
    if len(labels) > 1:                    # one series: the title names it, no legend box
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.015, 0.935), ncol=len(labels), fontsize=13)
    fig.suptitle(title, x=0.02, y=0.985, ha='left', fontsize=20)
    lines = textwrap.wrap(note, 190)
    fig.text(0.02, 0.012, '\n'.join(lines), fontsize=11, color=INK2, va='bottom')
    fig.tight_layout(rect=(0, 0.02 + 0.025 * len(lines), 1, 0.87))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', default=os.path.join(HERE, 'runs_eval'), help='folder with <Weather>/ subfolders')
    ap.add_argument('--lidar-weather', action='store_true',
                    help='the sets were degraded with lidar_weather.py (LiDAR rain / fog model): changes the notes')
    ap.add_argument('--min-frames', type=int, default=MIN_FRAMES,
                    help='leave sets with fewer frames out of the charts (they stay in the table and CSV)')
    args = ap.parse_args()
    lidar_note = ('LiDAR degraded per weather with lidar_weather.py (rain / fog model); both models get the same '
                  'points.' if args.lidar_weather else
                  'CARLA LiDAR ignores weather, so LiDAR-only should not change.')
    rows = load(args.eval)
    if not rows:
        raise SystemExit(f'no <Weather>/lidar|fusion/metrics.json found in {args.eval}')


    cols = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    out_csv = os.path.join(args.eval, 'offline_comparison.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


    # console table
    classes = [c[:-6] for c in cols if c.endswith('_3D_AP')]
    print(f"\n{'weather':<15}{'model':<9}{'frames':>7}{'mAP 3D':>9}" + ''.join(f'{c:>12}' for c in classes)
          + f"{'paint ALL %':>13}")
    for r in sorted(rows, key=lambda r: (ORDER.index(r['weather']) if r['weather'] in ORDER else 99, r['source'])):
        def f(v):
            return f'{v:.1f}' if isinstance(v, (int, float)) and v == v else '-'
        print(f"{r['weather']:<15}{r['source']:<9}{r.get('frames') or '-':>7}{f(r.get('mAP_3D')):>9}"
              + ''.join(f"{f(r.get(c + '_3D_AP')):>12}" for c in classes)
              + f"{f(r.get('paint_ALL_pct')):>13}")


    small = sorted({r['weather'] for r in rows if (r.get('frames') or 0) < args.min_frames})
    shown = [r for r in rows if r['weather'] not in small]
    left_out = (' Left out (too few frames): ' + ', '.join(
        f"{SHORT.get(w, w).replace(chr(10), ' ')} ({min(r['frames'] for r in rows if r['weather'] == w)})"
        for w in small) + '.') if small else ''
    if small:
        print(f'\n[note] charts leave out {small}: fewer than {args.min_frames} frames (--min-frames)')
    panels = [('mAP_3D', 'mAP (3D)', 'AP %')] + [(f'{c}_3D_AP', f'{c} (3D AP)', 'AP %') for c in classes
                                                  if c in ('Car', 'Pedestrian', 'Cyclist')]
    p1 = figure(shown, panels[:4], 'Offline 3D detection in seen and unseen weather: LiDAR only vs fusion'
                + (' (LiDAR weather model on)' if args.lidar_weather else ''),
                os.path.join(args.eval, 'offline_ap_by_weather.png'),
                'Town05 test drive per weather (same start and seed; the traffic differs between sets), camera '
                'view only (--fov_only). KITTI IoU: 0.7 vehicles, 0.5 pedestrian / cyclist. ' + lidar_note
                + left_out)
    paint_keys = [c for c in cols if c.startswith('paint_')]
    pick = [k for k in paint_keys if k == 'paint_ALL_pct'] + \
           [k for k in paint_keys if re.search('vehic|car', k, re.I)][:1] + \
           [k for k in paint_keys if re.search('pedes|person', k, re.I)][:1]
    p2 = None
    if pick:
        p2 = figure([r for r in shown if r['source'] == 'fusion'],
                    [(k, k[6:-4].replace('_', ' ').capitalize() + ' points labelled right', '%') for k in pick],
                    'Fusion input: painting accuracy by weather (SegFormer labels on LiDAR points vs CARLA)',
                    os.path.join(args.eval, 'offline_paint_by_weather.png'),
                    'From check_paint.py (section 2): share of LiDAR points in the camera view that got the '
                    'right class. This is the input fusion depends on.' + left_out)
    print(f'\nSaved {out_csv}' + ''.join(f'\n      {p}' for p in (p1, p2) if p))




if __name__ == '__main__':
    main()