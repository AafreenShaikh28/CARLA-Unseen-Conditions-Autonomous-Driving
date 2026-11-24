"""
ped_check.py - why are pedestrians DontCare? Reads labels_3d/ only (no CARLA needed).

    python ped_check.py kitti_carla

labels_3d fields used:  8 = n_box  (ray-cast LiDAR points inside the box, lowest 0.1 m skipped)
                        11 = distance (m)      12 = n_sem (semantic-LiDAR hits with this actor's id)
A box is DontCare in label_2 when n_sem < 5. Comparing n_box with n_sem tells you which of
three causes it is (the verdict is printed at the end).
"""
import glob, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else 'kitti_carla'
rows = {'person': [], 'car': []}
for p in glob.glob(os.path.join(root, 'training', 'labels_3d', '*.txt')):
    for line in open(p):
        f = line.split()
        if len(f) == 13 and f[0] in rows:
            rows[f[0]].append((float(f[11]), int(f[8]), int(f[12])))
if not rows['person']:
    raise SystemExit('no pedestrian boxes in labels_3d at all -> spawning problem, not a labelling one')

bins = [(0, 10), (10, 20), (20, 40), (40, 60)]
print(f"{'class':<8}{'range m':>9}{'boxes':>7}{'median n_box':>14}{'median n_sem':>14}{'n_sem<5':>9}")
for cls, r in rows.items():
    a = np.array(r)
    for lo, hi in bins:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.any():
            print(f'{cls:<8}{f"{lo}-{hi}":>9}{m.sum():>7}{np.median(a[m, 1]):>14.0f}'
                  f'{np.median(a[m, 2]):>14.0f}{100 * np.mean(a[m, 2] < 5):>8.0f}%')

p = np.array(rows['person'])
near = p[p[:, 0] < 20]
print()
if len(near) < 10:
    print('VERDICT: fewer than 10 pedestrians within 20 m -> they are mostly far away, where a few')
    print('         hits is normal. DontCare is correct; spawn walkers closer (near_fraction) if needed.')
elif np.median(near[:, 1]) >= 10 and np.median(near[:, 2]) < 5:
    print('VERDICT: ID MISMATCH. The LiDAR does hit nearby pedestrians (n_box high) but the semantic')
    print('         LiDAR hits do not carry the walker actor id (n_sem ~0). The labels are wrong, not')
    print('         the pedestrians: use the fixed collect_kitti_lidar.py (tag-in-box fallback) and recollect.')
elif np.median(near[:, 1]) < 5:
    print('VERDICT: the LiDAR is NOT hitting nearby pedestrians (n_box and n_sem both ~0). Check one frame')
    print('         in the bird\'s-eye view: if there is no point cluster at a pedestrian box, the sensor')
    print('         does not see walkers (CARLA collision issue), and DontCare is the honest label.')
else:
    print('VERDICT: nearby pedestrians are fine (real labels). The DontCare ones are far or occluded -')
    print('         that is the expected behaviour of the min_points rule.')
