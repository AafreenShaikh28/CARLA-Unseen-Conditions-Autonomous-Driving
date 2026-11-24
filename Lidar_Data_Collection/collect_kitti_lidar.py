"""
collect_kitti_lidar.py - CARLA 0.9.15 LiDAR + camera dataset in KITTI format
(project plan section 3.1: synchronous mode, ego + RGB + ray-cast LiDAR + semantic camera,
30-80 vehicles + pedestrians on Traffic Manager, 3D boxes < 60 m, fully occluded actors
dropped, split by town/run, KITTI layout so OpenPCDet / mmdet3d load it without custom code).

    python collect_kitti_lidar.py --quick        # 40 frames per run: smoke test first
    python collect_kitti_lidar.py                # full collection (edit CFG['runs'])
    python check_kitti.py kitti_carla            # then verify before training

Output (frame ids are global and unique across runs):
  kitti_carla/
    ImageSets/train.txt val.txt test.txt(empty) <run_name>.txt
    training/
      velodyne/000000.bin   float32 [N,4] x,y,z,intensity  (x fwd, y left, z up)
      image_2/000000.png    RGB camera
      calib/000000.txt      P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo
      label_2/000000.txt    KITTI labels (camera frame, objects in the image) + DontCare
      planes/000000.txt     road plane for OpenPCDet's GT sampler (USE_ROAD_PLANE)
      semantic/000000.png   raw CARLA class ids in the red channel (for PointPainting)
      labels_3d/000000.txt  360 deg LiDAR-frame boxes, same 13 fields as before (pp_common.py)
      metadata/000000.json  town, weather, ego pose / speed / control, light state
      split_lidar.json      town-based train/val ids for lidar_train.py
  kitti_carla/progress.json  resume bookkeeping

Label rules (one place to change them: make_labels)
  - vehicles and pedestrians within label_max_dist of the LiDAR (horizontal)
  - label_2 real label: projects into the image AND >= min_points hits from the semantic
    LiDAR (CARLA's own per-actor hit ids). occluded 0/1/2 from the depth camera,
    truncated = share of the projected box outside the image.
  - label_2 DontCare: in the image but < min_points hits (fully hidden from the LiDAR).
    OpenPCDet's KITTI eval ignores detections there instead of calling them false positives.
  - labels_3d: every actor in range, all around the car, with both hit counts.
  - parked cars baked into the map (no actor, so no label) are switched off.
"""
import os, json, math, queue, random, time, argparse
from collections import Counter
import numpy as np
import carla
import kitti_utils as K

CFG = dict(
    host='localhost', port=2000, tm_port=8000, timeout=60.0, seed=0,
    out='kitti_carla',
    # One entry per recording. Split by town/run, never by random frame.
    # Weather presets: ClearNoon, CloudyNoon, WetNoon, HardRainNoon, ClearSunset,
    # CloudySunset, ClearNight, ... (ray-cast LiDAR ignores weather; the camera does not).
    runs=[
        dict(name='town04_clearnoon', town='Town04', weather='ClearNoon', frames=2500, split='train'),
        dict(name='town01_cloudysunset', town='Town01', weather='CloudySunset', frames=2500, split='train'),
        dict(name='town05_clearnoon', town='Town05', weather='ClearNoon', frames=1500, split='val'),
    ],
    fixed_delta=0.05,        # 20 Hz simulation
    tick_skip=4,             # save every 4th tick = 5 Hz (consecutive ticks are near-identical)
    warmup_ticks=60,
    max_static_frames=10,    # stop saving after this many saved frames with the ego standing still

    # sensors, relative to the ego origin (which sits on the road). Keep them level:
    # KITTI boxes have no pitch/roll, so a tilted camera or LiDAR breaks the conversion.
    img_w=800, img_h=600, fov=90.0,
    cam_xyz=(1.5, 0.0, 2.4),
    lidar_xyz=(0.0, 0.0, 2.4),
    lidar_channels=64, lidar_range=60.0, lidar_pps=1300000,
    lidar_upper_fov=2.0, lidar_lower_fov=-24.8,
    lidar_noise_stddev=0.0, lidar_dropoff=0.0,   # CARLA default dropoff 0.45 deletes 45% of points

    # traffic (plan: 30-80 vehicles + pedestrians)
    # first dataset: 299 pedestrians vs 16,869 cyclists (>= 5 hits) -> fewer forced two-wheelers,
    # more walkers, and walkers kept near the ego (walker_respawn_*, ego_speed_diff below)
    n_vehicles=70, n_walkers=150, min_bicycles=3, min_motorcycles=3,
    near_radius=80.0,        # m: "near the ego"
    near_fraction=0.6,       # share of vehicles / walkers placed near the ego
    pedestrian_cross_factor=0.5,   # share of walkers allowed to cross roads (in front of the ego)
    tm_speed_diff=10.0,      # % slower than the speed limit -> denser traffic
    recycle_ticks=200,       # move vehicles left far behind to spots ahead (0 = off)
    walker_retarget_ticks=600,
    walker_respawn_ticks=100,      # every 5 s: walkers > walker_far_m away are replaced near the ego (0 = off)
    walker_far_m=60.0,
    walker_spawn_band=(8.0, 35.0), # m from the ego for the replacements
    walker_respawn_max=30,         # per call
    ego_speed_diff=40.0,           # EGO only: % below the speed limit, so pedestrians stay in range longer
    # CARLA 0.9.15 bug (issue #7092): controller.ai.walker switches off the walker's collision,
    # and ray-cast sensors skip collision-free actors -> pedestrians get ZERO LiDAR points.
    # Workaround from that thread: turn collisions (and physics) back on after start().
    walker_lidar_fix=True,
    walker_physics=True,     # set False if walkers stop walking / jitter with physics on
    disable_map_vehicles=True,

    # labels
    label_max_dist=60.0,
    min_points=5,            # semantic-LiDAR hits needed for a real label (else DontCare)
    occ_partly=0.8,          # visible share >= this -> occluded 0
    occ_largely=0.4,         # visible share >= this -> occluded 1, below -> 2
    occlusion_margin=1.0,    # m of depth slack in the visibility test
    min_dontcare_px=8,
    near_plane=0.1,
    save_semantic=True,
)

KITTI_NAME = {'car': 'Car', 'van': 'Van', 'truck': 'Truck', 'bus': 'Bus',
              'person': 'Pedestrian', 'bicycle': 'Cyclist', 'motorcycle': 'Cyclist'}
BICYCLES = {'vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets'}
MOTORCYCLES = {'vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja',
               'vehicle.vespa.zx125', 'vehicle.yamaha.yzf'}
BUSES = {'vehicle.mitsubishi.fusorosa'}
VANS = {'vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2', 'vehicle.volkswagen.t2_2021',
        'vehicle.ford.ambulance'}
TRUCKS = {'vehicle.carlamotors.carlacola', 'vehicle.carlamotors.firetruck',
          'vehicle.carlamotors.european_hgv', 'vehicle.tesla.cybertruck'}
SEM_LIDAR_DTYPE = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32),
                            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)])


# ----------------------------------------------------------------------------------- helpers
def classify(actor):
    t = actor.type_id
    if t.startswith('walker.pedestrian'):
        return 'person'
    if not t.startswith('vehicle.'):
        return None
    base = actor.attributes.get('base_type', '').lower()      # CARLA 0.9.14+
    if base in ('car', 'van', 'truck', 'bus', 'motorcycle', 'bicycle'):
        return base
    for group, name in ((BICYCLES, 'bicycle'), (MOTORCYCLES, 'motorcycle'), (BUSES, 'bus'),
                        (VANS, 'van'), (TRUCKS, 'truck')):
        if t in group:
            return name
    return 'motorcycle' if actor.attributes.get('number_of_wheels') == '2' else 'car'


def agent_tags():
    """Semantic tag ids of road users, read from the installed CARLA (ids differ across versions)."""
    names = ('Pedestrians', 'Rider', 'Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle')
    return np.array([int(getattr(carla.CityObjectLabel, n)) for n in names
                     if hasattr(carla.CityObjectLabel, n)], dtype=np.uint32)


def alive(actors):
    out = []
    for a in actors:
        try:
            if a.is_alive:
                out.append(a)
        except RuntimeError:
            pass
    return out


def mount(xyz):
    return carla.Transform(carla.Location(x=xyz[0], y=xyz[1], z=xyz[2]))


def decode_depth(img):
    a = np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4).astype(np.float32)
    return 1000.0 * (a[:, :, 2] + a[:, :, 1] * 256.0 + a[:, :, 0] * 65536.0) / (256.0 ** 3 - 1)


def visible_share(depth, box, z_min, margin, grid=5):
    """Share of a grid of pixels inside the 2D box where nothing is clearly closer than the
    object's nearest corner. Depth camera = planar depth = camera-frame z."""
    xs = np.linspace(box[0], box[2], grid + 2)[1:-1].astype(int)
    ys = np.linspace(box[1], box[3], grid + 2)[1:-1].astype(int)
    patch = depth[np.ix_(ys, xs)]
    return float((patch >= z_min - margin).mean())


def count_in_box(pts, box, bottom_skip=0.1):
    """Points inside an upright box; the lowest 0.1 m is skipped so road returns don't count."""
    x, y, z, l, w, h, yaw = box
    d = pts - np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return int(((np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)
                & (d[:, 2] >= -h / 2 + bottom_skip) & (d[:, 2] <= h / 2)).sum())


class SensorSync:
    """Hands out one measurement per sensor for exactly the requested frame."""
    def __init__(self, queues):
        self.queues, self.stash = queues, {}

    def drain(self):
        for q in self.queues.values():
            while not q.empty():
                q.get_nowait()
        self.stash.clear()

    def get(self, frame, timeout=5.0):
        out = {}
        for name, q in self.queues.items():
            item = self.stash.pop(name, None)
            try:
                while item is None or item.frame < frame:
                    item = q.get(timeout=timeout)
            except queue.Empty:
                print(f'\n  [warn] {name}: nothing for frame {frame} within {timeout}s')
                return None
            if item.frame > frame:
                self.stash[name] = item
                return None
            out[name] = item
        return out


# ---------------------------------------------------------------------------------- spawning
def disable_map_vehicles(world):
    """Parked cars that are part of the map are not actors, so they would be in the point
    cloud with no label. Switch them off."""
    ids = set()
    for name in ('Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle', 'Vehicles'):
        label = getattr(carla.CityObjectLabel, name, None)
        if label is not None:
            ids |= {o.id for o in world.get_environment_objects(label)}
    if ids:
        world.enable_environment_objects(ids, False)
    return len(ids)


def pick_ego_spawn(world):
    """Random choice among the 10% of spawn points with the most other spawn points nearby."""
    sps = world.get_map().get_spawn_points()
    dens = [sum(o.location.distance(sp.location) <= 100.0 for o in sps) for sp in sps]
    cut = sorted(dens, reverse=True)[max(0, len(sps) // 10 - 1)]
    return random.choice([sp for sp, d in zip(sps, dens) if d >= cut])


def spawn_vehicles(world, bp, tm, ego_loc):
    sps = world.get_map().get_spawn_points()
    random.shuffle(sps)
    near = sorted([s for s in sps if s.location.distance(ego_loc) <= CFG['near_radius']],
                  key=lambda s: s.location.distance(ego_loc))
    far = [s for s in sps if s.location.distance(ego_loc) > CFG['near_radius']]
    all_bps = list(bp.filter('vehicle.*'))
    bikes = [b for b in all_bps if b.id in BICYCLES]
    motos = [b for b in all_bps if b.id in MOTORCYCLES]
    plan = ([random.choice(bikes) for _ in range(CFG['min_bicycles'])] if bikes else []) + \
           ([random.choice(motos) for _ in range(CFG['min_motorcycles'])] if motos else [])
    plan += [random.choice(all_bps) for _ in range(CFG['n_vehicles'] - len(plan))]
    n_near = int(len(plan) * CFG['near_fraction'])
    points = near[:n_near] + far + near[n_near:]          # two-wheelers get the closest points
    vehicles, i = [], 0
    for b in plan:
        if b.has_attribute('role_name'):
            b.set_attribute('role_name', 'autopilot')
        while i < len(points):
            v = world.try_spawn_actor(b, points[i]); i += 1
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                vehicles.append(v)
                break
    print(f'  vehicles: {len(vehicles)}/{CFG["n_vehicles"]}')
    return vehicles


def nav_point(world, ego_loc, near=True, tries=30):
    loc = None
    for _ in range(tries):
        loc = world.get_random_location_from_navigation()
        if loc is not None and (not near or loc.distance(ego_loc) <= CFG['near_radius']):
            return loc
    return loc


def make_walkers_lidar_visible(walkers):
    """Undo what controller.ai.walker does in CARLA 0.9.15 (issue #7092): without collision
    the LiDAR rays pass straight through pedestrians. Call after every controller start()."""
    if not CFG['walker_lidar_fix']:
        return
    for w in alive(walkers):
        try:
            w.set_collisions(True)
            if CFG['walker_physics']:
                w.set_simulate_physics(True)
        except (RuntimeError, AttributeError) as e:
            print(f'  [warn] could not re-enable walker collision ({e}); pedestrians may get no LiDAR points')
            return


def spawn_walkers(world, bp, ego_loc):
    world.set_pedestrians_cross_factor(CFG['pedestrian_cross_factor'])
    wbps = list(bp.filter('walker.pedestrian.*'))
    walkers = []
    for k in range(CFG['n_walkers'] * 3):
        if len(walkers) >= CFG['n_walkers']:
            break
        loc = nav_point(world, ego_loc, near=random.random() < CFG['near_fraction'])
        if loc is None:
            continue
        b = random.choice(wbps)
        if b.has_attribute('is_invincible'):
            b.set_attribute('is_invincible', 'false')
        w = world.try_spawn_actor(b, carla.Transform(loc))
        if w is not None:
            walkers.append(w)
    world.tick()                                     # walkers must exist before controllers
    cbp = bp.find('controller.ai.walker')
    ctrls = []
    for w in walkers:
        c = world.try_spawn_actor(cbp, carla.Transform(), attach_to=w)
        if c is not None:
            ctrls.append(c)
    world.tick()                                     # controllers must exist before start()
    for c in ctrls:
        try:
            c.start()
            dest = nav_point(world, ego_loc)
            if dest is not None:
                c.go_to_location(dest)
            c.set_max_speed(3.0 if random.random() < 0.2 else 1.4)
        except RuntimeError:
            pass
    make_walkers_lidar_visible(walkers)
    print(f'  walkers: {len(walkers)}/{CFG["n_walkers"]} ({len(ctrls)} controllers)')
    return walkers, ctrls


def recycle_vehicles(world, vehicles, ego_tf):
    """Teleport vehicles far behind the ego to free spawn points 65-130 m ahead of it
    (beyond LiDAR range, so nothing pops up inside the sensor's view)."""
    vehicles[:] = alive(vehicles)
    ego_loc, fwd = ego_tf.location, ego_tf.get_forward_vector()
    locs = {}
    for v in vehicles:
        try:
            locs[v.id] = v.get_transform().location
        except RuntimeError:
            pass
    movers = [v for v in vehicles if v.id in locs and locs[v.id].distance(ego_loc) > 1.5 * CFG['near_radius']]
    cands = []
    for sp in world.get_map().get_spawn_points():
        d = sp.location.distance(ego_loc)
        if 65.0 <= d <= 130.0:
            dx, dy = sp.location.x - ego_loc.x, sp.location.y - ego_loc.y
            if (dx * fwd.x + dy * fwd.y) / d >= 0.5:
                cands.append(sp)
    random.shuffle(cands)
    taken, moved = list(locs.values()), 0
    for v in movers[:40]:
        while cands:
            sp = cands.pop()
            if all(sp.location.distance(p) >= 8.0 for p in taken):
                try:
                    v.set_transform(sp)
                    taken.append(sp.location); moved += 1
                except RuntimeError:
                    pass
                break
    return moved


def respawn_walkers(world, bp, walkers, ctrls, actors, ego_loc):
    """Replace walkers more than walker_far_m from the ego with new walkers 8-35 m from it, so
    pedestrians stay in LiDAR range while the ego drives on (walkers at 1.4 m/s can't follow).
    Ticks the world twice: call it only AFTER the frame of this tick has been written."""
    walkers[:] = alive(walkers)
    ctrls[:] = alive(ctrls)
    far = []
    for w in walkers:
        try:
            if w.get_location().distance(ego_loc) > CFG['walker_far_m']:
                far.append(w)
        except RuntimeError:
            pass
    far = far[:CFG['walker_respawn_max']]
    if not far:
        return 0
    lo, hi = CFG['walker_spawn_band']
    pts = []
    for _ in range(40 * len(far)):
        if len(pts) >= len(far):
            break
        loc = world.get_random_location_from_navigation()
        if loc is not None and lo <= loc.distance(ego_loc) <= hi:
            pts.append(loc)
    far = far[:len(pts)]
    if not far:
        return 0
    far_ids = {w.id for w in far}
    for c in list(ctrls):                                   # controllers first, then their walkers
        try:
            if c.parent is not None and c.parent.id in far_ids:
                c.stop()
                c.destroy()
                ctrls.remove(c)
        except RuntimeError:
            pass
    for w in far:
        try:
            w.destroy()
        except RuntimeError:
            pass
    walkers[:] = [w for w in walkers if w.id not in far_ids]

    wbps = list(bp.filter('walker.pedestrian.*'))
    new = []
    for loc in pts:
        b = random.choice(wbps)
        if b.has_attribute('is_invincible'):
            b.set_attribute('is_invincible', 'false')
        w = world.try_spawn_actor(b, carla.Transform(loc))
        if w is not None:
            new.append(w)
    if not new:
        return 0
    world.tick()                                            # walkers must exist before controllers
    cbp = bp.find('controller.ai.walker')
    new_ctrls = []
    for w in new:
        c = world.try_spawn_actor(cbp, carla.Transform(), attach_to=w)
        if c is not None:
            new_ctrls.append(c)
    world.tick()                                            # controllers must exist before start()
    for c in new_ctrls:
        try:
            c.start()
            dest = nav_point(world, ego_loc)
            if dest is not None:
                c.go_to_location(dest)
            c.set_max_speed(3.0 if random.random() < 0.2 else 1.4)
        except RuntimeError:
            pass
    make_walkers_lidar_visible(new)                         # the 0.9.15 walker LiDAR fix
    walkers.extend(new)
    ctrls.extend(new_ctrls)
    actors.extend(new + new_ctrls)
    return len(new)


def spawn_sensors(world, bp, ego):
    cam_tf, lid_tf = mount(CFG['cam_xyz']), mount(CFG['lidar_xyz'])

    def camera(kind):
        b = bp.find(kind)
        for k, v in (('image_size_x', CFG['img_w']), ('image_size_y', CFG['img_h']), ('fov', CFG['fov'])):
            b.set_attribute(k, str(v))
        return world.spawn_actor(b, cam_tf, attach_to=ego)

    sensors = {'rgb': camera('sensor.camera.rgb'), 'depth': camera('sensor.camera.depth')}
    if CFG['save_semantic']:
        sensors['sem'] = camera('sensor.camera.semantic_segmentation')
    attrs = {'channels': CFG['lidar_channels'], 'range': CFG['lidar_range'],
             'points_per_second': CFG['lidar_pps'],
             'rotation_frequency': round(1.0 / CFG['fixed_delta']),   # one full sweep per tick
             'upper_fov': CFG['lidar_upper_fov'], 'lower_fov': CFG['lidar_lower_fov'],
             'dropoff_general_rate': CFG['lidar_dropoff'], 'dropoff_zero_intensity': CFG['lidar_dropoff'],
             'noise_stddev': CFG['lidar_noise_stddev']}
    for name, kind in (('lidar', 'sensor.lidar.ray_cast'), ('lidar_sem', 'sensor.lidar.ray_cast_semantic')):
        b = bp.find(kind)
        for k, v in attrs.items():
            if b.has_attribute(k):
                b.set_attribute(k, str(v))
        sensors[name] = world.spawn_actor(b, lid_tf, attach_to=ego)   # same pose, same pattern
    queues = {}
    for name, s in sensors.items():
        q = queue.Queue()
        s.listen(q.put)
        queues[name] = q
    return sensors, queues


# ------------------------------------------------------------------------------------ labels
def make_labels(world, ego, data, velo_xyz, Tr, Kmat):
    """-> (label_2 lines, labels_3d lines, Counter)"""
    w2l = np.array(data['lidar'].transform.get_inverse_matrix())    # world -> CARLA lidar
    R = w2l[:3, :3]
    sem = np.frombuffer(data['lidar_sem'].raw_data, dtype=SEM_LIDAR_DTYPE)
    ids, cnt = np.unique(sem['ObjIdx'], return_counts=True)
    hits = dict(zip(ids.tolist(), cnt.tolist()))
    # Fallback when a hit does not carry the actor's id (seen for pedestrians): semantic points
    # tagged as a road user (pedestrian, rider, car, truck, bus, motorcycle, bicycle) inside the box.
    agent = np.isin(sem['ObjTag'], agent_tags())
    agent_xyz = np.stack([sem['x'][agent], -sem['y'][agent], sem['z'][agent]], 1)   # velodyne frame
    depth = decode_depth(data['depth'])
    W, H = CFG['img_w'], CFG['img_h']
    label2, dontcare, l3d, stats = [], [], [], Counter()

    actors = world.get_actors()
    for actor in list(actors.filter('vehicle.*')) + list(actors.filter('walker.pedestrian.*')):
        if actor.id == ego.id:
            continue
        cls = classify(actor)
        if cls is None:
            continue
        try:
            tf, bb = actor.get_transform(), actor.bounding_box
        except RuntimeError:                                   # removed by the simulator this tick
            continue
        cw = tf.transform(bb.location)                         # box centre, world
        c = w2l @ np.array([cw.x, cw.y, cw.z, 1.0])
        x, y, z = c[0], -c[1], c[2]                            # velodyne frame (y left)
        dist = math.hypot(x, y)
        if dist > CFG['label_max_dist']:
            continue
        fw = tf.get_forward_vector()
        f = R @ np.array([fw.x, fw.y, fw.z])
        yaw = math.atan2(-f[1], f[0])
        box = np.array([x, y, z, 2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z, yaw])
        n_sem = hits.get(actor.id, 0)
        if n_sem == 0 and len(agent_xyz):
            n_sem = count_in_box(agent_xyz, box, bottom_skip=0.0)
            if n_sem:
                stats['id_fallback_' + cls] += 1
        n_box = count_in_box(velo_xyz, box)

        proj = K.project_clipped(K.transform(Tr, K.box_corners_velo(box)), Kmat, W, H, CFG['near_plane'])
        labelled = 0
        if proj is not None and proj[1] is not None:
            full, img, z_min = proj
            area = lambda b: max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
            trunc = float(np.clip(1.0 - area(img) / max(area(full), 1e-6), 0.0, 1.0))
            if n_sem >= CFG['min_points']:
                vis = visible_share(depth, img, z_min, CFG['occlusion_margin'])
                occ = 0 if vis >= CFG['occ_partly'] else (1 if vis >= CFG['occ_largely'] else 2)
                loc, dims, ry, alpha = K.velo_box_to_kitti(box, Tr)
                label2.append(K.kitti_line(KITTI_NAME[cls], trunc, occ, alpha, img, dims, loc, ry))
                stats[KITTI_NAME[cls]] += 1
                labelled = 1
            elif min(img[2] - img[0], img[3] - img[1]) >= CFG['min_dontcare_px']:
                dontcare.append(K.dontcare_line(img))
                stats['DontCare'] += 1
        l3d.append(f'{cls} {x:.3f} {y:.3f} {z:.3f} {box[3]:.3f} {box[4]:.3f} {box[5]:.3f} '
                   f'{yaw:.4f} {n_box} {actor.id} {labelled} {dist:.2f} {n_sem}')
        stats['3d_' + cls] += 1
        if cls == 'person' and n_sem >= CFG['min_points']:
            stats['3d_person_seen'] += 1                       # pedestrian the LiDAR actually hit
    # DontCare MUST come last: OpenPCDet and mmdet3d take the first N lines (N = non-DontCare
    # count) as the real boxes, exactly like the original KITTI files are ordered.
    return label2 + dontcare, l3d, stats


def write_frame(tr, fid, data, world, ego, Tr, Kmat, calib_txt, plane_txt, run, frame):
    pts = np.frombuffer(data['lidar'].raw_data, dtype=np.float32).reshape(-1, 4).copy()
    pts[:, 1] *= -1.0                                          # CARLA y-right -> KITTI y-left
    pts.tofile(f'{tr}/velodyne/{fid}.bin')
    data['rgb'].save_to_disk(f'{tr}/image_2/{fid}.png')
    if 'sem' in data:
        data['sem'].save_to_disk(f'{tr}/semantic/{fid}.png')  # raw ids, no colour converter
    with open(f'{tr}/calib/{fid}.txt', 'w') as fh:
        fh.write(calib_txt)
    with open(f'{tr}/planes/{fid}.txt', 'w') as fh:
        fh.write(plane_txt)
    label2, l3d, stats = make_labels(world, ego, data, pts[:, :3], Tr, Kmat)
    with open(f'{tr}/label_2/{fid}.txt', 'w') as fh:
        fh.write('\n'.join(label2) + ('\n' if label2 else ''))
    with open(f'{tr}/labels_3d/{fid}.txt', 'w') as fh:
        fh.write('\n'.join(l3d))
    tf, v, ctl = ego.get_transform(), ego.get_velocity(), ego.get_control()
    meta = {'frame_id': fid, 'sim_frame': frame, 'run': run['name'], 'town': run['town'],
            'weather': run['weather'], 'seed': CFG['seed'],
            'ego_location': [tf.location.x, tf.location.y, tf.location.z],
            'ego_rotation': [tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll],
            'ego_speed': math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
            'control': {'throttle': ctl.throttle, 'steer': ctl.steer, 'brake': ctl.brake},
            'at_traffic_light': ego.is_at_traffic_light(),
            'traffic_light_state': str(ego.get_traffic_light_state()),
            'n_points': int(len(pts)), 'labels': dict(stats),
            'sensor_frames': {k: d.frame for k, d in data.items()}}
    with open(f'{tr}/metadata/{fid}.json', 'w') as fh:
        json.dump(meta, fh)
    return stats


# ------------------------------------------------------------------------------------ a run
def collect_run(client, run, start_idx, tr, Tr, Kmat, calib_txt, plane_txt):
    print(f"\n=== {run['name']}: {run['town']} / {run['weather']} / {run['frames']} frames ===")
    client.load_world(run['town'])
    world = client.get_world()
    original = world.get_settings()
    tm = client.get_trafficmanager(CFG['tm_port'])
    actors, sensors, ctrls = [], {}, []
    totals, skipped, respawned = Counter(), 0, 0
    try:
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = True, CFG['fixed_delta']
        world.apply_settings(s)
        tm.set_synchronous_mode(True)                 # after the world is synchronous
        tm.set_random_device_seed(CFG['seed'])        # needed again after every load_world
        tm.global_percentage_speed_difference(CFG['tm_speed_diff'])
        world.set_weather(getattr(carla.WeatherParameters, run['weather']))
        if CFG['disable_map_vehicles']:
            print(f'  switched off {disable_map_vehicles(world)} parked vehicles baked into the map')
        world.tick()

        bp = world.get_blueprint_library()
        ego_bp = bp.find('vehicle.tesla.model3')
        ego_bp.set_attribute('role_name', 'hero')
        ego = None
        for _ in range(20):                           # a busy spawn point can be occupied
            ego = world.try_spawn_actor(ego_bp, pick_ego_spawn(world))
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError('could not spawn the ego vehicle')
        actors.append(ego)
        world.tick()
        ego.set_autopilot(True, tm.get_port())
        tm.vehicle_percentage_speed_difference(ego, CFG['ego_speed_diff'])
        ego_loc = ego.get_transform().location
        vehicles = spawn_vehicles(world, bp, tm, ego_loc)
        actors += vehicles
        walkers, ctrls = spawn_walkers(world, bp, ego_loc)
        actors += walkers + ctrls
        sensors, queues = spawn_sensors(world, bp, ego)
        actors += list(sensors.values())

        for _ in range(CFG['warmup_ticks']):
            world.tick()
        sync = SensorSync(queues)
        sync.drain()

        saved, tick, static = start_idx, 0, 0
        max_ticks = run['frames'] * CFG['tick_skip'] * 5          # safety net if the ego gets stuck
        while saved < start_idx + run['frames']:
            if tick >= max_ticks:
                print(f'\n  [warn] stopping early: {max_ticks} ticks without enough moving frames')
                break
            frame = world.tick()
            tick += 1
            data = sync.get(frame)
            if data is None:
                skipped += 1
            elif tick % CFG['tick_skip'] == 0:
                v = ego.get_velocity()
                static = static + 1 if math.sqrt(v.x ** 2 + v.y ** 2) < 0.5 else 0
                if static <= CFG['max_static_frames']:
                    totals += write_frame(tr, f'{saved:06d}', data, world, ego, Tr, Kmat,
                                          calib_txt, plane_txt, run, frame)
                    saved += 1
                    print(f"  {saved - start_idx}/{run['frames']}", end='\r')
            # maintenance AFTER the frame is written: teleports only show up next tick
            if CFG['recycle_ticks'] and tick % CFG['recycle_ticks'] == 0:
                recycle_vehicles(world, vehicles, ego.get_transform())
            if CFG['walker_respawn_ticks'] and tick % CFG['walker_respawn_ticks'] == 0:
                respawned += respawn_walkers(world, bp, walkers, ctrls, actors,
                                             ego.get_transform().location)
            if CFG['walker_retarget_ticks'] and tick % CFG['walker_retarget_ticks'] == 0:
                ctrls[:] = alive(ctrls)
                ego_loc = ego.get_transform().location
                for c in ctrls:
                    try:
                        dest = nav_point(world, ego_loc)
                        if dest is not None:
                            c.go_to_location(dest)
                    except RuntimeError:
                        pass
                make_walkers_lidar_visible(walkers)          # in case the controller turned it off again

        n = saved - start_idx
        print(f'\n  saved {n} frames ({skipped} ticks skipped for incomplete sensor data, '
              f'{respawned} walkers respawned near the ego)')
        print('  label_2 objects per frame: ' + ', '.join(
            f'{k} {v / max(n, 1):.1f}' for k, v in sorted(totals.items())
            if not k.startswith(('3d_', 'id_fallback_'))))
        print('  labels_3d (360 deg) per frame: ' + ', '.join(
            f'{k[3:]} {v / max(n, 1):.1f}' for k, v in sorted(totals.items()) if k.startswith('3d_')))
        if totals['3d_person'] and not totals['3d_person_seen']:
            print('  [WARN] no pedestrian got >= min_points LiDAR hits. The walker collision fix did '
                  'not work on this CARLA build: pedestrians are invisible to the LiDAR.')
        fb = {k[12:]: v for k, v in totals.items() if k.startswith('id_fallback_')}
        if fb:
            print(f'  hits found by tag-in-box because the actor id was missing: {fb}  '
                  '(large for "person" = CARLA does not put walker ids on LiDAR hits)')
        return saved
    finally:
        for s_ in sensors.values():
            try:
                s_.stop()
            except RuntimeError:
                pass
        for c in alive(ctrls):
            try:
                c.stop()
            except RuntimeError:
                pass
        live = alive(actors)
        if live:
            client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in live], True)
        tm.set_synchronous_mode(False)
        world.apply_settings(original)


# ------------------------------------------------------------------------------- bookkeeping
def write_splits(root, progress):
    ims = os.path.join(root, 'ImageSets')
    split_ids = {'train': [], 'val': []}
    for name, r in progress['runs'].items():
        ids = [f'{i:06d}' for i in range(r['first'], r['last'] + 1)]
        with open(os.path.join(ims, f'{name}.txt'), 'w') as fh:
            fh.write('\n'.join(ids) + '\n')
        split_ids.setdefault(r['split'], []).extend(ids)
    for split in ('train', 'val'):
        with open(os.path.join(ims, f'{split}.txt'), 'w') as fh:
            fh.write('\n'.join(split_ids[split]) + ('\n' if split_ids[split] else ''))
    open(os.path.join(ims, 'test.txt'), 'w').close()          # OpenPCDet's info script expects it
    with open(os.path.join(root, 'training', 'split_lidar.json'), 'w') as fh:
        json.dump({'train': split_ids['train'], 'val': split_ids['val']}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='40 frames per run (smoke test)')
    ap.add_argument('--runs', nargs='*', help='only these run names')
    ap.add_argument('--fresh', action='store_true', help='ignore progress.json and start at frame 0')
    args = ap.parse_args()

    for xyz in (CFG['cam_xyz'], CFG['lidar_xyz']):
        assert len(xyz) == 3
    random.seed(CFG['seed']); np.random.seed(CFG['seed'])
    root = CFG['out'] + ('_quick' if args.quick else '')     # smoke test never mixes with real data
    tr = os.path.join(root, 'training')
    for d in ('velodyne', 'image_2', 'calib', 'label_2', 'planes', 'semantic', 'labels_3d', 'metadata'):
        os.makedirs(os.path.join(tr, d), exist_ok=True)
    os.makedirs(os.path.join(root, 'ImageSets'), exist_ok=True)

    # calibration is fixed by the mounts, so it is computed once from them (no sim timing involved)
    Kmat = K.intrinsics(CFG['img_w'], CFG['img_h'], CFG['fov'])
    Tr = K.velo_to_cam_matrix(np.array(mount(CFG['lidar_xyz']).get_matrix()),
                              np.array(mount(CFG['cam_xyz']).get_matrix()))
    calib_txt = K.calib_text(Kmat, Tr)
    plane_txt = K.plane_text(CFG['cam_xyz'][2])
    print('Tr_velo_to_cam:\n', np.round(Tr, 3))

    prog_path = os.path.join(root, 'progress.json')
    progress = {'next_idx': 0, 'runs': {}}
    if os.path.isfile(prog_path) and not args.fresh:
        progress = json.load(open(prog_path))
        print(f"resuming at frame {progress['next_idx']}; done: {list(progress['runs'])}")

    client = carla.Client(CFG['host'], CFG['port'])
    client.set_timeout(CFG['timeout'])
    idx = progress['next_idx']
    for run in CFG['runs']:
        if args.runs and run['name'] not in args.runs:
            continue
        if run['name'] in progress['runs']:
            print(f"skip {run['name']} (already collected)")
            continue
        run = dict(run, frames=40 if args.quick else run['frames'])
        start = idx
        idx = collect_run(client, run, start, tr, Tr, Kmat, calib_txt, plane_txt)
        progress['runs'][run['name']] = {'first': start, 'last': idx - 1, 'split': run['split'],
                                         'town': run['town'], 'weather': run['weather']}
        progress['next_idx'] = idx
        with open(prog_path, 'w') as fh:
            json.dump(progress, fh, indent=2)
        write_splits(root, progress)
    print(f'\ndone: {idx} frames in {root}. Next: python check_kitti.py {root}')


if __name__ == '__main__':
    main()
