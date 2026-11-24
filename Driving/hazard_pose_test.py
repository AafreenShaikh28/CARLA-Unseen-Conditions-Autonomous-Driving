"""
hazard_pose_test.py - Indian road hazards in CARLA, step 1: a person LYING on the road, SITTING on it
(legs stretched out) and CROUCHING on it, and what each sensor sees of them. Put it in the Driving folder.

CARLA has no lying / sitting / crouching animation, so the walker's skeleton is posed through CARLA's
bone API (Walker.get_pose_from_animation, set_bones, blend_pose; CARLA 0.9.13+). The axes of the crl_
skeleton's bones are not documented, so the script does not guess them: for each pose it tries the
axis / sign options, measures the result from the bone positions (get_bones), and keeps the one that
gives the right shape (head height, hip height, knee direction ...). It then checks each sensor:

    python hazard_pose_test.py                                   # CARLA server running, Town05
    python hazard_pose_test.py --distance 15 --poses lying       # one pose, farther away

Per pose (standing = reference) it prints and saves:
  shape     head / hip height and body length from the bones: is it really lying / sitting / crouching?
  LiDAR     points on the body; 'ghost' points where the upright walker would be (if the LiDAR hits an
            invisible upright capsule instead of the posed body, the pose is useless for LiDAR tests)
  semantic  pedestrian pixels on the body (the ground truth SegFormer / painting is compared with)
  YOLO      best box on the body, its class and score (if perception_scheduler.py and the model load)
  images    <pose>.png (ego camera, body outline green, YOLO box orange) and <pose>_zoom.png
  recipe    hazard_poses.json: the bone changes that made each pose (agent.py will read this file)
Everything goes to hazard_test\\<place>\\ (lane, edge or footpath).

The ego is a parked car with the same camera / LiDAR mounts as the data collector, looking at the
person --distance m ahead. Where the person is (--place):
    lane      in the ego's lane, at the lane centre, lying across the lane (default): must be detected AND stopped for
    edge      at the right edge of the ego's lane, lying along the road: partly in the car's path
    footpath  on the footpath to the right of the road, lying along it: should be detected, but the car should not brake
    python hazard_pose_test.py --place footpath
--yaw overrides the heading (0 = along the road, 90 = across), --offset moves the person sideways (m, + = right).
The ground height is measured at the spot (the footpath is a kerb higher than the road), so the body rests on it.
"""
import os
import sys
import json
import math
import queue
import types
import argparse
from collections import OrderedDict

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
AXES = ('pitch', 'yaw', 'roll')
CAM = dict(w=800, h=600, fov=90.0, xyz=(1.5, 0.0, 2.4))                  # same as collect_kitti_lidar.py
LIDAR = dict(xyz=(0.0, 0.0, 2.4), channels=64, range=60.0, pps=1300000, upper=2.0, lower=-24.8)
PED_TAG = 12


# ----------------------------------------------------------------------------- bones
def bone_names(walker):
    return [b.name for b in walker.get_bones().bone_transforms]


def find_bone(names, *keys):
    keys = [k.lower() for k in keys]
    for n in names:
        if all(k in n.lower() for k in keys):
            return n
    return None


def skeleton(names):
    """the bones the shape checks need, by name (crl_ skeleton: crl_thigh__L, crl_leg__L = shin ...)"""
    sk = dict(root=find_bone(names, 'root'), hips=find_bone(names, 'hips'), head=find_bone(names, 'head'),
              spine=find_bone(names, 'spine__'))
    for side in ('l', 'r'):
        sk['thigh_' + side] = find_bone(names, 'thigh__' + side)
        sk['leg_' + side] = find_bone(names, '_leg__' + side) or find_bone(names, 'leg__' + side)
        sk['foot_' + side] = find_bone(names, 'foot__' + side)
    missing = [k for k, v in sk.items() if v is None]
    if missing:
        raise SystemExit(f'bones not found: {missing}. Bones of this walker: {names}')
    return sk


def world_bones(walker):
    out = {}
    for b in walker.get_bones().bone_transforms:
        loc = b.world.location
        out[b.name] = np.array([loc.x, loc.y, loc.z], float)
    return out


def relative_pose(walker, carla):
    out = OrderedDict()
    for b in walker.get_bones().bone_transforms:
        r, loc = b.relative.rotation, b.relative.location
        out[b.name] = carla.Transform(carla.Location(loc.x, loc.y, loc.z),
                                      carla.Rotation(pitch=r.pitch, yaw=r.yaw, roll=r.roll))
    return out


def set_pose(carla, world, walker, base, changes, ticks=3):
    """base: bone -> relative Transform (the idle pose); changes: bone -> ([d_pitch, d_yaw, d_roll], [dx, dy, dz]).
    Every bone is sent, so bones that are not changed keep the idle pose."""
    pose = []
    for name, tf in base.items():
        if name in changes:
            (dp, dyw, dr), (dx, dy, dz) = changes[name]
            r, loc = tf.rotation, tf.location
            tf = carla.Transform(carla.Location(loc.x + dx, loc.y + dy, loc.z + dz),
                                 carla.Rotation(pitch=r.pitch + dp, yaw=r.yaw + dyw, roll=r.roll + dr))
        pose.append((name, tf))
    ctl = carla.WalkerBoneControlIn()
    ctl.bone_transforms = pose
    walker.set_bones(ctl)
    walker.blend_pose(1.0)
    for _ in range(ticks):
        world.tick()
    return world_bones(walker)


def rot(axis, deg):
    d = [0.0, 0.0, 0.0]
    d[AXES.index(axis)] = float(deg)
    return d


def merged(*parts):
    out = {}
    for p in parts:
        for k, (r, t) in p.items():
            r0, t0 = out.get(k, ([0.0] * 3, [0.0] * 3))
            out[k] = ([a + b for a, b in zip(r0, r)], [a + b for a, b in zip(t0, t)])
    return out


# ----------------------------------------------------------------------------- shape
def shape(bw, sk, ground_z, yaw_deg):
    """heights above the road and directions in the walker's own frame (fwd = where it faces)"""
    yaw = math.radians(yaw_deg)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    pts = np.array([p for n, p in bw.items() if n != sk['root']])      # the root bone is not a body part
    z = pts[:, 2] - ground_z
    hip_xy = bw[sk['hips']][:2]
    flat = pts[:, :2]
    length = float(np.max(np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2))) if len(flat) > 1 else 0.0

    def h(name):
        return float(bw[sk[name]][2] - ground_z)

    def f(name):
        return float(np.dot(bw[sk[name]][:2] - hip_xy, fwd))

    return dict(head_z=h('head'), hips_z=h('hips'), min_z=float(z.min()), max_z=float(z.max()), length=length,
                knee_z=(h('leg_l') + h('leg_r')) / 2, knee_fwd=(f('leg_l') + f('leg_r')) / 2,
                ankle_z=(h('foot_l') + h('foot_r')) / 2, ankle_fwd=(f('foot_l') + f('foot_r')) / 2)


def direction_error(bw, a, b, want):
    """1 - cos(angle) between bone a -> bone b and the wanted 3D direction (0 = exact)"""
    v = bw[b] - bw[a]
    n = np.linalg.norm(v)
    return 2.0 if n < 1e-6 else float(1.0 - np.dot(v / n, want / np.linalg.norm(want)))


def wanted(yaw_deg, fwd, up):
    yaw = math.radians(yaw_deg)
    return np.array([math.cos(yaw) * fwd, math.sin(yaw) * fwd, up])


def pose_ok(name, s):
    """the final check for each pose (heights in m above the road)"""
    if name == 'lying':
        return s['head_z'] < 0.45 and s['hips_z'] < 0.45 and s['max_z'] < 0.6 and s['min_z'] > -0.1 and s['length'] > 1.2
    if name == 'sitting':
        return (s['hips_z'] < 0.35 and 0.6 < s['head_z'] < 1.15 and s['knee_z'] < 0.4 and s['ankle_fwd'] > 0.5
                and s['min_z'] > -0.1)
    if name == 'crouching':
        return (0.2 < s['hips_z'] < 0.7 and 0.7 < s['head_z'] < 1.35 and s['ankle_z'] < 0.25 and s['knee_fwd'] > 0.15
                and s['min_z'] > -0.1)
    return True


# ----------------------------------------------------------------------------- pose search
class Poser:
    def __init__(self, carla, world, walker, ground_z, yaw):
        self.carla, self.world, self.walker = carla, world, walker
        self.ground_z, self.yaw = ground_z, yaw
        self.walker.get_pose_from_animation()
        for _ in range(2):
            world.tick()
        self.base = relative_pose(walker, carla)
        self.sk = skeleton(list(self.base))
        self.tried = 0

    def apply(self, changes):
        self.tried += 1
        return set_pose(self.carla, self.world, self.walker, self.base, changes)

    def best(self, options, cost):
        """options: list of (label, changes); returns (cost, label, changes, bones) with the lowest cost"""
        scored = []
        for label, ch in options:
            bw = self.apply(ch)
            scored.append((cost(bw), label, ch, bw))
        scored.sort(key=lambda t: t[0])
        return scored[0]

    def limb_options(self, bones_lr, deg, base_changes):
        """rotate a left / right bone pair by deg about each axis, both signs, same or mirrored sign"""
        out = []
        for axis in AXES:
            for sign in (1, -1):
                for mirror in (1, -1):
                    ch = merged(base_changes, {bones_lr[0]: (rot(axis, sign * deg), [0, 0, 0]),
                                               bones_lr[1]: (rot(axis, sign * mirror * deg), [0, 0, 0])})
                    out.append((f'{axis}{"+" if sign > 0 else "-"}{"" if mirror > 0 else " mirrored"}', ch))
        return out

    def drop_to(self, changes, target_low, low_bones):
        """lower the whole body (root bone) so the lowest of low_bones ends at target_low m above the road"""
        root = self.sk['root']
        bw = self.apply(changes)
        d = min(bw[b][2] for b in low_bones) - self.ground_z - target_low
        best = None
        for i in range(3):
            for sign in (1, -1):
                t = [0.0, 0.0, 0.0]
                t[i] = -sign * d
                ch = merged(changes, {root: ([0, 0, 0], t)})
                bw2 = self.apply(ch)
                err = abs(min(bw2[b][2] for b in low_bones) - self.ground_z - target_low)
                if best is None or err < best[0]:
                    best = (err, ch, bw2, 'xyz'[i] + ('-' if sign > 0 else '+'))
        return best

    def lying(self):
        root, sk = self.sk['root'], self.sk
        opts = []
        for axis in AXES:
            for sign in (1, -1):
                opts.append((f'root {axis}{"+" if sign > 0 else "-"}90', {root: (rot(axis, sign * 90), [0, 0, 0])}))

        def cost(bw):
            s = shape(bw, sk, self.ground_z, self.yaw)
            return s['head_z'] + s['hips_z'] + max(0.0, 1.3 - s['length'])
        c, label, ch, bw = self.best(opts, cost)
        err, ch2, bw2, how = self.drop_to(ch, 0.10, [n for n in self.base if n != root])   # lowest bone 10 cm up
        return ch2, f'{label}, root moved along {how} so the body rests on the road'

    def legs(self, thigh_deg, shin_deg, shin_want, spine_deg=0.0):
        sk = self.sk
        thighs, shins = (sk['thigh_l'], sk['thigh_r']), (sk['leg_l'], sk['leg_r'])
        horiz = wanted(self.yaw, 1.0, 0.0)

        def thigh_cost(bw):
            return sum(direction_error(bw, t, k, horiz) for t, k in zip(thighs, shins))
        c1, l1, ch, _ = self.best(self.limb_options(thighs, thigh_deg, {}), thigh_cost)
        label = f'thighs {l1} {thigh_deg:g}'
        if shin_deg:
            def shin_cost(bw):
                return sum(direction_error(bw, k, f, shin_want) for k, f in zip(shins, (sk['foot_l'], sk['foot_r'])))
            c2, l2, ch, _ = self.best(self.limb_options(shins, shin_deg, ch), shin_cost)
            label += f', shins {l2} {shin_deg:g}'
        if spine_deg:
            lean = wanted(self.yaw, math.sin(math.radians(spine_deg)), math.cos(math.radians(spine_deg)))
            opts = [(f'{a}{"+" if s > 0 else "-"}', merged(ch, {sk['spine']: (rot(a, s * spine_deg), [0, 0, 0])}))
                    for a in AXES for s in (1, -1)]
            c3, l3, ch, _ = self.best(opts, lambda bw: direction_error(bw, sk['hips'], sk['head'], lean))
            label += f', spine {l3} {spine_deg:g}'
        return ch, label

    def sitting(self):
        ch, label = self.legs(90, 0, None)
        err, ch, bw, how = self.drop_to(ch, 0.12, [self.sk['thigh_l'], self.sk['thigh_r']])
        return ch, label + f', body lowered along root {how} to the road'

    def crouching(self):
        back_down = wanted(self.yaw, -0.5, -0.87)                           # shin from knee down and back
        ch, label = self.legs(90, 120, back_down, spine_deg=25)
        err, ch, bw, how = self.drop_to(ch, 0.08, [self.sk['foot_l'], self.sk['foot_r']])
        return ch, label + f', body lowered along root {how} until the feet are on the road'


# ----------------------------------------------------------------------------- sensors
class Sync:
    def __init__(self, sensors):
        self.q = {n: queue.Queue() for n in sensors}
        for n, s in sensors.items():
            s.listen(self.q[n].put)

    def get(self, frame, timeout=10.0):
        out = {}
        for n, q in self.q.items():
            item = None
            while item is None or item.frame < frame:
                item = q.get(timeout=timeout)
            out[n] = item
        return out

    def drain(self):
        for q in self.q.values():
            while not q.empty():
                q.get_nowait()


def body_frame(points_world, hips_xy, yaw_deg):
    yaw = math.radians(yaw_deg)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    lat = np.array([-math.sin(yaw), math.cos(yaw)])
    d = points_world[:, :2] - hips_xy
    return d @ fwd, d @ lat


def lidar_counts(meas, bw, actor_xy, ground_z, yaw):
    raw = np.frombuffer(meas.raw_data, dtype=np.float32).reshape(-1, 4)
    m = np.array(meas.transform.get_matrix())
    pw = (np.c_[raw[:, :3], np.ones(len(raw))] @ m.T)[:, :3]
    bones = np.array([p for n, p in bw.items() if 'root' not in n.lower()])
    hips_xy = bones[:, :2].mean(0)
    bf, bl = body_frame(bones, hips_xy, yaw)
    pf, pl = body_frame(pw, hips_xy, yaw)
    pz = pw[:, 2] - ground_z
    in_body = ((pf > bf.min() - 0.2) & (pf < bf.max() + 0.2) & (pl > bl.min() - 0.2) & (pl < bl.max() + 0.2)
               & (pz > 0.05) & (pz < (bones[:, 2] - ground_z).max() + 0.2))
    col = (np.linalg.norm(pw[:, :2] - actor_xy, axis=1) < 0.35) & (pz > 0.6) & (pz < 2.0) & ~in_body
    return int(in_body.sum()), int(col.sum())


def project(points_world, cam_tf, w, h, fov):
    inv = np.array(cam_tf.get_inverse_matrix())
    pc = (np.c_[points_world, np.ones(len(points_world))] @ inv.T)[:, :3]           # x fwd, y right, z up
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    ok = pc[:, 0] > 0.1
    u = w / 2.0 + f * pc[:, 1] / np.maximum(pc[:, 0], 0.1)
    v = h / 2.0 - f * pc[:, 2] / np.maximum(pc[:, 0], 0.1)
    return u, v, pc[:, 0], ok, f


def body_box_2d(bw, cam_tf):
    pts = np.array([p for n, p in bw.items() if 'root' not in n.lower()])
    u, v, depth, ok, f = project(pts, cam_tf, CAM['w'], CAM['h'], CAM['fov'])
    if not ok.any():
        return None
    m = 0.15 * f / depth[ok]                                     # ~15 cm of body around the bone centres
    x1, x2 = float((u[ok] - m).min()), float((u[ok] + m).max())
    y1, y2 = float((v[ok] - m).min()), float((v[ok] + m * 1.5).max())
    return [max(x1, 0), max(y1, 0), min(x2, CAM['w'] - 1), min(y2, CAM['h'] - 1)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------- main
def footpath_next_to(carla, wp, max_steps=5):
    """the footpath (sidewalk lane) to the right of this lane, or None"""
    cur = wp
    for _ in range(max_steps):
        cur = cur.get_right_lane()
        if cur is None:
            return None
        if cur.lane_type == carla.LaneType.Sidewalk:
            return cur
    return None


def straight_spot(carla, cmap, spawns, start, distance, place):
    for i in list(range(start, len(spawns))) + list(range(0, start)):
        wp0 = cmap.get_waypoint(spawns[i].location)
        nxt = wp0.next(distance)
        if not nxt or wp0.is_junction:
            continue
        wp = nxt[0]
        dyaw = abs((wp.transform.rotation.yaw - wp0.transform.rotation.yaw + 180) % 360 - 180)
        if not wp.is_junction and wp.road_id == wp0.road_id and dyaw < 3 and wp.lane_id == wp0.lane_id:
            side = footpath_next_to(carla, wp) if place == 'footpath' else wp
            if side is not None:
                return i, wp0, wp, side
    raise SystemExit('no straight lane' + (' with a footpath on its right' if place == 'footpath' else '')
                     + ' found ahead of any spawn point; try another --town or --distance')


def ground_height(world, carla, loc, fallback):
    """height of whatever is under loc (road, kerb, footpath); falls back to the waypoint height"""
    try:
        hit = world.ground_projection(carla.Location(loc.x, loc.y, loc.z + 2.0), 6.0)
        if hit is not None:
            return float(hit.location.z), str(hit.label)
    except (AttributeError, RuntimeError):
        pass
    return float(fallback), 'waypoint height'


def main():
    ap = argparse.ArgumentParser(description='pose lying / sitting / crouching pedestrians and check what the sensors see')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    ap.add_argument('--town', default='Town05')
    ap.add_argument('--spawn-index', type=int, default=0, help='first spawn point to try for a straight lane')
    ap.add_argument('--distance', type=float, default=10.0, help='m from the ego to the person')
    ap.add_argument('--place', choices=['lane', 'edge', 'footpath'], default='lane',
                    help='lane: centre of the ego lane; edge: right edge of the ego lane; footpath: beside the road')
    ap.add_argument('--offset', type=float, default=0.0, help='extra m to the right (+) or left (-) of that spot')
    ap.add_argument('--yaw', type=float, default=None,
                    help='person heading vs the road, deg: 90 = across (lane default), 0 = along (edge / footpath default)')
    ap.add_argument('--poses', nargs='+', default=['standing', 'lying', 'sitting', 'crouching'],
                    choices=['standing', 'lying', 'sitting', 'crouching'])
    ap.add_argument('--blueprint', default='walker.pedestrian.0001')
    ap.add_argument('--yolo', default=os.path.join(CODE, 'Detection_YOLO', 'runs', 'detect', 'runs', 'detect',
                                                   'carla_finetune-10', 'weights', 'best.onnx'))
    ap.add_argument('--no-yolo', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'hazard_test'))
    args = ap.parse_args()
    import carla
    args.out = os.path.join(args.out, args.place)            # hazard_test\\lane, hazard_test\\footpath ...
    os.makedirs(args.out, exist_ok=True)

    yolo, names = None, {}
    if not args.no_yolo and os.path.exists(args.yolo):
        try:
            sys.path.insert(0, HERE)
            import perception_scheduler as ps
            yolo, names = ps.YoloOnnx(args.yolo, conf=0.25), ps.DRIVING_YOLO_CLASSES
            print(f'YOLO: {args.yolo}')
        except Exception as e:                                   # noqa: BLE001
            print(f'[note] YOLO not loaded ({e}); camera detection is not checked')
    elif not args.no_yolo:
        print(f'[note] no YOLO model at {args.yolo} (pass --yolo); camera detection is not checked')

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if not world.get_map().name.endswith(args.town):
        print(f'loading {args.town} ...')
        world = client.load_world(args.town)
    original = world.get_settings()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    actors, results, recipes = [], OrderedDict(), OrderedDict()
    try:
        cmap, bp = world.get_map(), world.get_blueprint_library()
        idx, wp0, wp, side = straight_spot(carla, cmap, cmap.get_spawn_points(), args.spawn_index, args.distance,
                                           args.place)
        ego_tf = carla.Transform(wp0.transform.location + carla.Location(z=0.3), wp0.transform.rotation)
        ego = world.try_spawn_actor((list(bp.filter('vehicle.lincoln.mkz_2020')) or list(bp.filter('vehicle.*')))[0], ego_tf)
        if ego is None:
            raise SystemExit(f'could not spawn the ego at spawn point {idx}: try --spawn-index {idx + 1}')
        actors.append(ego)
        ego.set_simulate_physics(False)
        cam_bp = bp.find('sensor.camera.rgb')
        sem_bp = bp.find('sensor.camera.semantic_segmentation')
        for b in (cam_bp, sem_bp):
            b.set_attribute('image_size_x', str(CAM['w']))
            b.set_attribute('image_size_y', str(CAM['h']))
            b.set_attribute('fov', str(CAM['fov']))
        lid_bp = bp.find('sensor.lidar.ray_cast')
        for k, v in (('channels', LIDAR['channels']), ('range', LIDAR['range']), ('points_per_second', LIDAR['pps']),
                     ('rotation_frequency', 20), ('upper_fov', LIDAR['upper']), ('lower_fov', LIDAR['lower']),
                     ('dropoff_general_rate', 0.0)):
            lid_bp.set_attribute(k, str(v))
        cam_at = carla.Transform(carla.Location(*CAM['xyz']))
        sensors = OrderedDict(rgb=world.spawn_actor(cam_bp, cam_at, attach_to=ego),
                              sem=world.spawn_actor(sem_bp, cam_at, attach_to=ego),
                              lidar=world.spawn_actor(lid_bp, carla.Transform(carla.Location(*LIDAR['xyz'])),
                                                      attach_to=ego))
        actors += list(sensors.values())
        sync = Sync(sensors)
        right = wp.transform.get_right_vector()
        side_m = {'lane': 0.0, 'edge': wp.lane_width / 2.0 - 0.3}.get(args.place, 0.0) + args.offset
        base = side.transform.location
        spot = base + carla.Location(x=right.x * side_m, y=right.y * side_m)
        ground_z, ground_what = ground_height(world, carla, spot, base.z)
        rel_yaw = args.yaw if args.yaw is not None else (90.0 if args.place == 'lane' else 0.0)
        yaw = wp.transform.rotation.yaw + rel_yaw
        lateral = (spot.x - wp.transform.location.x) * right.x + (spot.y - wp.transform.location.y) * right.y
        print(f'spawn point {idx}: person on the {args.place} {args.distance:g} m ahead, {lateral:.1f} m right of the '
              f'lane centre, {"across" if abs(rel_yaw) % 180 > 45 else "along"} the road; ground '
              f'{ground_z - wp.transform.location.z:+.2f} m vs the road ({ground_what})')
        if args.place == 'footpath' and ground_z - wp.transform.location.z < 0.05:
            print('  [note] the footpath measured no higher than the road: check in the images that the person is on it')

        for pose in args.poses:
            walker = None
            for dz in (1.0, 1.3, 1.6):
                walker = world.try_spawn_actor(bp.find(args.blueprint), carla.Transform(
                    carla.Location(spot.x, spot.y, ground_z + dz), carla.Rotation(yaw=yaw)))
                if walker is not None:
                    break
            if walker is None:
                raise SystemExit(f'could not spawn {args.blueprint} on the road; try --offset or --spawn-index')
            for _ in range(15):                                         # let it settle on the road
                world.tick()
            walker.set_collisions(True)                                 # no AI controller: stays visible to LiDAR
            poser = Poser(carla, world, walker, ground_z, yaw)
            how = 'idle animation'
            changes = {}
            if pose != 'standing':
                changes, how = getattr(poser, pose)()
            bw = poser.apply(changes)
            for _ in range(3):
                world.tick()
            sync.drain()
            frame = world.tick()
            data = sync.get(frame)
            bw = world_bones(walker)
            sh = shape(bw, poser.sk, ground_z, yaw)
            loc = walker.get_transform().location
            n_body, n_ghost = lidar_counts(data['lidar'], bw, np.array([loc.x, loc.y]), ground_z, yaw)
            img = np.frombuffer(data['rgb'].raw_data, dtype=np.uint8).reshape(CAM['h'], CAM['w'], 4)[:, :, :3].copy()
            sem = np.frombuffer(data['sem'].raw_data, dtype=np.uint8).reshape(CAM['h'], CAM['w'], 4)[:, :, 2]
            box = body_box_2d(bw, data['rgb'].transform)
            sem_px = 0
            if box is not None:
                x1, y1, x2, y2 = [int(round(t)) for t in box]
                sem_px = int((sem[y1:y2 + 1, x1:x2 + 1] == PED_TAG).sum())
            det = None
            if yolo is not None and box is not None:
                pred, _ = yolo(types.SimpleNamespace(bgr=img))
                for b, c, sc in zip(pred['boxes'], pred['classes'], pred['scores']):
                    o = iou(box, list(map(float, b)))
                    if o >= 0.2 and (det is None or sc > det['score']):
                        det = dict(cls=names.get(int(c), str(int(c))), score=float(sc), iou=round(o, 2),
                                   box=[float(t) for t in b])
            ok = pose_ok(pose, sh)
            results[pose] = OrderedDict(
                pose_ok=ok, how=how, tried=poser.tried, head_z=round(sh['head_z'], 2), hips_z=round(sh['hips_z'], 2),
                body_length=round(sh['length'], 2), top_z=round(sh['max_z'], 2), lowest_z=round(sh['min_z'], 2),
                lidar_points_on_body=n_body, lidar_ghost_points=n_ghost, semantic_ped_pixels=sem_px,
                box_px=None if box is None else [round(t) for t in box],
                yolo=det if yolo is not None else 'not checked')
            if pose != 'standing':
                recipes[pose] = dict(changes=changes, how=how, pose_ok=ok, measured=results[pose])

            vis = img.copy()
            if box is not None:
                cv2.rectangle(vis, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (60, 200, 60), 2)
            if det:
                b = det['box']
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (40, 120, 240), 2)
                cv2.putText(vis, f"YOLO {det['cls']} {det['score']:.2f}", (int(b[0]), max(int(b[1]) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 120, 240), 1, cv2.LINE_AA)
            cv2.putText(vis, f'{pose} on {args.place} at {args.distance:g} m  LiDAR {n_body} pts  ped px {sem_px}', (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(args.out, f'{pose}.png'), vis)
            if box is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                half = max(box[2] - box[0], box[3] - box[1]) * 0.9 + 20
                x1, y1 = int(max(cx - half, 0)), int(max(cy - half, 0))
                x2, y2 = int(min(cx + half, CAM['w'])), int(min(cy + half, CAM['h']))
                crop = vis[y1:y2, x1:x2]
                if crop.size:
                    cv2.imwrite(os.path.join(args.out, f'{pose}_zoom.png'),
                                cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            print(f"  {pose:<10} {'OK  ' if ok else 'BAD '} head {sh['head_z']:.2f} m  hips {sh['hips_z']:.2f} m  "
                  f"length {sh['length']:.2f} m | LiDAR {n_body} on body, {n_ghost} ghost | ped px {sem_px} | "
                  f"YOLO {(det['cls'] + ' %.2f' % det['score']) if det else ('-' if yolo else 'not checked')}")
            print(f'             {how}')
            walker.destroy()
            world.tick()
    finally:
        for a in reversed(actors):
            try:
                if a.type_id.startswith('sensor'):
                    a.stop()
                a.destroy()
            except RuntimeError:
                pass
        world.apply_settings(original)

    with open(os.path.join(args.out, 'hazard_poses.json'), 'w') as fh:
        json.dump(dict(blueprint=args.blueprint, distance_m=args.distance, place=args.place, yaw_vs_road=rel_yaw,
                       poses=recipes),
                  fh, indent=2)
    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\n{'pose':<10}{'shape':>7}{'head m':>8}{'hips m':>8}{'len m':>7}{'LiDAR':>7}{'ghost':>7}{'ped px':>8}  YOLO")
    for p, r in results.items():
        y = r['yolo']
        ytxt = 'not checked' if y == 'not checked' else ('missed' if y is None else f"{y['cls']} {y['score']:.2f}")
        print(f"{p:<10}{'ok' if r['pose_ok'] else 'BAD':>7}{r['head_z']:>8.2f}{r['hips_z']:>8.2f}{r['body_length']:>7.2f}"
              f"{r['lidar_points_on_body']:>7}{r['lidar_ghost_points']:>7}{r['semantic_ped_pixels']:>8}  {ytxt}")
    print(f'\nimages, results.json and hazard_poses.json in {args.out}')
    print('ghost > ~20 with a lying / sitting / crouching pose means the LiDAR hits an invisible upright capsule,')
    print('not the posed body: say so before using that pose for LiDAR or fusion results.')


if __name__ == '__main__':
    main()