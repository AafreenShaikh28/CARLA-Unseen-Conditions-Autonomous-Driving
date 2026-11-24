"""
kitti_utils.py - numpy-only KITTI conversions shared by collect_kitti_lidar.py and check_kitti.py.

Frames
  velodyne (saved .bin, KITTI convention) : x forward, y LEFT,  z up   (right-handed)
  CARLA sensor (native)                    : x forward, y RIGHT, z up   (left-handed)
  KITTI camera = rect camera (R0_rect = I) : x right,  y down,  z forward
KITTI label_2 boxes live in the camera frame: location = BOTTOM centre, dims = h w l,
rotation_y about the camera y axis. OpenPCDet converts them back to LiDAR boxes with
    xyz = rect_to_lidar(location); xyz.z += h / 2; heading = -rotation_y - pi / 2
which only holds when the camera and LiDAR are mounted level (no pitch/roll). The
collector refuses tilted mounts for that reason.
"""
import numpy as np

CARLA_TO_CV = np.array([[0, 1, 0, 0],      # CARLA camera axes -> OpenCV/KITTI camera axes
                        [0, 0, -1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1.0]])
VELO_TO_CARLA = np.diag([1.0, -1.0, 1.0, 1.0])   # saved velodyne (y left) -> CARLA lidar (y right)

# corner order: bottom face 0-3, top face 4-7 (same footprint order)
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def intrinsics(w, h, fov_deg):
    f = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def velo_to_cam_matrix(T_vehicle_lidar, T_vehicle_cam):
    """4x4 matrix: saved velodyne coords -> KITTI camera coords.
    Inputs are carla.Transform(...).get_matrix() of the two mounts (relative to the ego)."""
    return CARLA_TO_CV @ np.linalg.inv(np.asarray(T_vehicle_cam)) @ np.asarray(T_vehicle_lidar) @ VELO_TO_CARLA


def _fmt(m):
    return ' '.join(f'{v:.12e}' for v in np.asarray(m, dtype=float).ravel())


def calib_text(K, Tr):
    """KITTI calib file. OpenPCDet reads it BY LINE NUMBER (P2 = line 3, R0 = line 5,
    Tr_velo_to_cam = line 6), so the order below must not change. One camera only, so
    P0..P3 are identical and the rect frame is the camera frame."""
    P = np.hstack([K, np.zeros((3, 1))])
    lines = [f'P{i}: {_fmt(P)}' for i in range(4)]
    lines.append(f'R0_rect: {_fmt(np.eye(3))}')
    lines.append(f'Tr_velo_to_cam: {_fmt(np.asarray(Tr)[:3, :])}')
    lines.append(f'Tr_imu_to_velo: {_fmt(np.hstack([np.eye(3), np.zeros((3, 1))]))}')
    return '\n'.join(lines) + '\n'


def plane_text(cam_height):
    """training/planes/*.txt (used by OpenPCDet's GT sampler when USE_ROAD_PLANE: True):
    road plane a*x + b*y + c*z + d = 0 in camera coords; flat road cam_height below the camera."""
    return f'# Plane\nWidth 4\nHeight 1\n0.000000e+00 -1.000000e+00 0.000000e+00 {cam_height:.6e}\n'


def read_calib(path):
    vals = {}
    for line in open(path):
        if ':' in line:
            k, v = line.split(':', 1)
            vals[k.strip()] = np.array([float(x) for x in v.split()])
    return {'P2': vals['P2'].reshape(3, 4), 'R0': vals['R0_rect'].reshape(3, 3),
            'Tr': vals['Tr_velo_to_cam'].reshape(3, 4)}


def norm_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def transform(T, pts):
    T = np.asarray(T)
    return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]


def box_corners_velo(box):
    """box = [x, y, z_centre, l, w, h, yaw] in the velodyne frame -> 8x3 corners."""
    x, y, z, l, w, h, yaw = box
    xs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * l / 2
    ys = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    zs = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * h / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * xs - s * ys, y + s * xs + c * ys, z + zs], 1)


def project_clipped(corners_cam, K, img_w, img_h, near=0.1):
    """Project 8 camera-frame corners, clipping box edges at the near plane so a box
    that straddles the camera is not shrunk. Returns (full_box, image_box, z_min) where
    full_box = [x1,y1,x2,y2] before image clipping and image_box is clipped to the image
    (None if nothing is inside). Returns None if the box is entirely behind the camera."""
    pts = [p for p in corners_cam if p[2] > near]
    for a, b in BOX_EDGES:
        pa, pb = corners_cam[a], corners_cam[b]
        if (pa[2] > near) != (pb[2] > near):
            t = (near - pa[2]) / (pb[2] - pa[2])
            pts.append(pa + t * (pb - pa))
    if not pts:
        return None
    pts = np.array(pts)
    u = K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2]
    v = K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]
    full = [u.min(), v.min(), u.max(), v.max()]
    img = [max(full[0], 0.0), max(full[1], 0.0), min(full[2], img_w - 1.0), min(full[3], img_h - 1.0)]
    if img[2] <= img[0] or img[3] <= img[1]:
        img = None
    return full, img, float(pts[:, 2].min())


def velo_box_to_kitti(box, Tr):
    """velodyne box -> (location (bottom centre, camera frame), (h, w, l), rotation_y, alpha)"""
    x, y, z, l, w, h, yaw = box
    loc = transform(Tr, np.array([[x, y, z - h / 2.0]]))[0]
    d = np.asarray(Tr)[:3, :3] @ np.array([np.cos(yaw), np.sin(yaw), 0.0])   # heading in camera frame
    ry = norm_angle(np.arctan2(-d[2], d[0]))
    alpha = norm_angle(ry - np.arctan2(loc[0], loc[2]))
    return loc, (h, w, l), ry, alpha


def kitti_line(name, trunc, occ, alpha, bbox, dims, loc, ry):
    h, w, l = dims
    return (f'{name} {trunc:.2f} {int(occ)} {alpha:.2f} '
            f'{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} '
            f'{h:.2f} {w:.2f} {l:.2f} {loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} {ry:.2f}')


def dontcare_line(bbox):
    return (f'DontCare -1 -1 -10 {bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} '
            f'-1 -1 -1 -1000 -1000 -1000 -10')


def read_label2(path):
    objs = []
    for line in open(path):
        f = line.split()
        if len(f) < 15:
            continue
        objs.append({'name': f[0], 'trunc': float(f[1]), 'occ': int(float(f[2])), 'alpha': float(f[3]),
                     'bbox': np.array(f[4:8], float), 'hwl': np.array(f[8:11], float),
                     'loc': np.array(f[11:14], float), 'ry': float(f[14])})
    return objs


def kitti_to_velo_box(obj, calib):
    """The conversion OpenPCDet applies when it loads label_2 (boxes3d_kitti_camera_to_lidar)."""
    h, w, l = obj['hwl']
    R0 = np.eye(4); R0[:3, :3] = calib['R0']
    Tr = np.eye(4); Tr[:3, :] = calib['Tr']
    p = np.linalg.inv(R0 @ Tr) @ np.append(obj['loc'], 1.0)
    return np.array([p[0], p[1], p[2] + h / 2, l, w, h, -obj['ry'] - np.pi / 2])


def kitti_corners_cam(obj):
    """KITTI devkit corner construction (camera frame), independent of box_corners_velo."""
    h, w, l = obj['hwl']
    x = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    c, s = np.cos(obj['ry']), np.sin(obj['ry'])
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.vstack([x, y, z])).T + obj['loc']
