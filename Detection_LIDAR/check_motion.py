"""
check_motion.py - how much does the ego vehicle move in the train vs validation frames?
Reads ego_location / ego_speed from metadata/<id>.json (every 20th frame = every 4 seconds).

    python check_motion.py
"""
import json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='../Data_collection_LIDAR/dataset/Town03')
    ap.add_argument('--step', type=int, default=20)
    args = ap.parse_args()
    r = args.data_root
    split = json.load(open(r + '/split_lidar.json'))
    for k in ('train', 'val'):
        ids = split[k][::args.step]
        M = np.array([[m['ego_location']['x'], m['ego_location']['y'], m['ego_speed']]
                      for m in (json.load(open('%s/metadata/%s.json' % (r, i))) for i in ids)])
        path = float(np.hypot(np.diff(M[:, 0]), np.diff(M[:, 1])).sum())
        cells = len(set(map(tuple, np.floor(M[:, :2] / 20).astype(int))))
        print('%-5s %5d sampled frames | path length %6.0f m | distinct 20x20 m map cells visited: %3d | '
              'median speed %.2f m/s | frames with speed < 0.3 m/s: %.0f%%'
              % (k, len(ids), path, cells, np.median(M[:, 2]), 100 * (M[:, 2] < 0.3).mean()))
        print('      first position (%.0f, %.0f)   last position (%.0f, %.0f)   x range %.0f..%.0f   y range %.0f..%.0f'
              % (M[0, 0], M[0, 1], M[-1, 0], M[-1, 1], M[:, 0].min(), M[:, 0].max(), M[:, 1].min(), M[:, 1].max()))


if __name__ == '__main__':
    main()