import carla, os, queue, random, json, time, math
from collections import Counter
import numpy as np

CFG = dict(
    towns           = ['Town04', 'Town01', 'Town05'],
    frames_per_town = 10000,          # raise to 7000 once a short run passes
    tick_skip       = 4,
    n_vehicles      = 150,            # FIX: was 120 -- see two-wheeler floor below
    n_walkers       = 100,
    # Minimum number of each two-wheeler class to force-spawn per town,
    # since random.choice over all vehicle.* blueprints under-represents
    # them badly (CARLA's default content has far more car blueprints than
    # bicycle/motorcycle ones). These come out of the n_vehicles budget.
    # FIX: was 50/50, i.e. ~83% of a 120-vehicle budget was two-wheelers,
    # which starves the car/truck/bus classes. 20/20 of 150 is ~27%.
    min_bicycles    = 20,
    min_motorcycles = 20,
    # Distance (meters) within which two-wheeler spawn points are preferred
    # over random ones, so more instances land big enough to actually learn
    # from instead of being unusable 6px specks at the far end of lidar_range.
    two_wheeler_bias_radius = 25.0,
    # Drop GT 2D boxes smaller than this many pixels in width or height --
    # they carry ~no learnable signal and just add label noise to training.
    min_box_px = 12,
    img_w = 800, img_h = 600, fov = 90,

    # FIX (pedestrians with no LiDAR returns): denser, KITTI-like sweep and
    # NO random point dropout/noise. CARLA's default dropoff_general_rate is
    # 0.45, i.e. ~45% of points were silently deleted, which is brutal for
    # thin objects like pedestrians.
    lidar_range = 60.0, lidar_channels = 64, lidar_pps = 1300000,
    lidar_upper_fov = 2.0, lidar_lower_fov = -24.8,
    lidar_dropoff_general_rate = 0.0,
    lidar_dropoff_zero_intensity = 0.0,
    lidar_noise_stddev = 0.0,
    weather = carla.WeatherParameters.ClearNoon,
    out = 'dataset',

    # NEW: pedestrian spawning (CFG['n_walkers'] existed before but was
    # never actually used anywhere -- no walkers were ever spawned).
    pedestrian_running_ratio  = 0.2,   # fraction of walkers set to "running" speed
    pedestrian_max_speed_walk = 1.4,   # m/s
    pedestrian_max_speed_run  = 3.0,   # m/s
    pedestrian_spawn_retries  = 3,     # x n_walkers = max spawn attempts

    # NEW: crowd density where the sensors actually look. Spawning uniformly
    # over the whole map leaves only a handful of actors inside the ~60 m /
    # 90 deg slice the sensors cover, and the ego drives away from them. These
    # knobs concentrate actors around the ego; the remainder stays map-wide so
    # the far field isn't empty.
    crowd_radius             = 100.0,  # m around the ego that counts as "near"
    crowd_vehicle_fraction   = 0.6,    # share of the general vehicle fill spawned near the ego
    crowd_walker_fraction    = 0.7,    # share of walkers spawned / re-targeted near the ego
    walker_pool_size         = 1500,   # nav-mesh points sampled per spawn / re-target round
    walker_redispatch_ticks  = 600,    # re-target walkers every N ticks (0 = never; walkers
                                       # otherwise stop for good at their first destination)
    pedestrian_cross_factor  = 0.3,    # fraction of pedestrians allowed to cross roads
    tm_speed_diff            = 10.0,   # Traffic Manager global speed difference, %.
                                       # POSITIVE = slower than the limit (denser traffic),
                                       # negative = faster. 0 = unchanged.
    tm_distance_to_leading   = 1.0,    # TM global gap to the vehicle ahead, m (CARLA default 1.0)
    ego_spawn_dense          = True,   # start the ego in a spawn-point-rich area, not a random one

    # NEW: recycling. Spawn-time bias alone can't beat the number of spawn
    # points near the ego (~25-30 per 100 m circle), and the ego drives away
    # from its initial crowd. Every `recycle_ticks`, vehicles/walkers that are
    # farther than recycle_far_factor x crowd_radius from the ego are
    # moved to free spots AHEAD of it (inside its forward cone, beyond
    # lidar range so they never pop in inside the sensor's view).
    # Vehicles are teleported; walkers are destroyed and respawned (see
    # recycle_walkers() for why).
    recycle_ticks        = 200,    # 0 = off
    recycle_far_factor   = 1.5,
    recycle_min_dist     = 65.0,   # m; keep > lidar_range so nothing appears inside it
    recycle_max_dist     = 130.0,  # m
    recycle_ahead_cos    = 0.5,    # min cos(angle to ego heading); 0.5 = +-60 deg (camera FOV is +-45)
    recycle_min_gap      = 8.0,    # m between a recycled vehicle and any other vehicle
    recycle_max_per_call = 40,     # cap per call (bounds the RPC cost)

    # NEW: perception-pipeline range/filter knobs, decoupled per your spec
    # (section 3 vs section 6) even though both default to 60 m today.
    yolo_max_distance = 60.0,   # max distance (m) from ego for a 2D YOLO GT box

    # FIX: was 1. Boxes with few/zero LiDAR returns are exactly the hard
    # pedestrian cases; dropping them at collection time throws away
    # information you can never get back. Keep everything in range, store
    # num_points (and the exact semantic-LiDAR count) per box, and filter
    # per class in the dataloader instead (e.g. pedestrian >= 3 to train,
    # KITTI-style easy/moderate/hard at evaluation).
    min_points        = 0,

    # FIX: points counted as "inside" a box are tested against the box
    # inflated by this margin (meters, each side) -- pedestrians are thin and
    # sweep-motion skew at ego speed is comparable to their width.
    point_count_margin = 0.12,

    # FIX: also spawn a sensor.lidar.ray_cast_semantic at the same pose.
    # It reports the actor id of every point, so we get an exact per-actor
    # point count and can tell "truly occluded" (0) from "just sparse".
    # Used for LABELS/METADATA ONLY -- never as detector input.
    use_semantic_lidar = True,

    # NEW: occlusion / visibility filtering for 2D YOLO GT (section 5).
    # Practical depth-buffer sampling, NOT a per-pixel raycast -- see
    # is_visible_via_depth() docstring for exactly what this does and does
    # not catch.
    visibility_check = True,
    occlusion_margin  = 1.0,    # meters of slack before calling a pixel "occluded"
    # FIX: keep a box if at least this fraction of sampled pixels is NOT
    # blocked by something closer than the actor (was: any 1 of 5).
    occlusion_min_visible_ratio = 0.2,
    near_plane = 0.1,           # FIX: boxes are clipped against this camera-space depth (m)

    # NEW: optional debug/duplicate outputs, off by default to avoid
    # doubling storage use.
    save_lidar_ply      = False,  # also dump the old .ply alongside .bin
    save_depth_raw_png  = False,  # also dump a visualization depth PNG
    save_semantic_visual = True,  # also dump CityScapes-colored semantic PNG
    # FIX: float32 800x600 .npy is ~1.9 MB/frame (~57 GB for 30k frames).
    # float16 halves it (~3 cm resolution at 60 m, fine for painting checks).
    save_depth_fp16     = True,

    # FIX: label_2/ was a copy of labels_3d/ but is NOT KITTI format (9-13
    # fields vs the 15 KITTI/OpenPCDet loaders expect, and there is no calib/
    # per frame). Off by default so a KITTI loader can't silently mis-parse
    # it. Point a custom dataset class at labels_3d/ + calib.json instead.
    # Set True only if other code of yours needs the directory populated.
    write_label_2 = False,

    # NEW: lane GT (section 7).
    lane_sample_distance = 2.0,   # meters between sampled waypoints along a lane
    lane_max_distance    = 60.0,  # how far forward to sample lanes

    # NEW: run/versioning + split bookkeeping (sections 9, 11).
    run_id = 'run_0001',
    weather_name = 'ClearNoon',   # human-readable label for metadata; keep in
                                   # sync with CFG['weather'] above -- carla's
                                   # WeatherParameters object isn't JSON-safe.
    # Not a random per-frame split -- split by town/run, exactly because
    # consecutive frames are highly correlated. Configurable, not hard-coded.
    split_config = {'Town01': 'train', 'Town04': 'train', 'Town05': 'val'},

    validate_after_run = True,    # run validate_dataset() at the end of each town

    # FIX: reproducibility + crash recovery.
    seed   = 0,
    resume = False,               # True = continue from progress.json after a crash
)


# ---------------------------------------------------------------------------
# FIX: raw CARLA semantic segmentation tag IDs -> this project's class names.
# ---------------------------------------------------------------------------
# The previous table (1=building, 4=pedestrian, 7=road, 10=vehicle ...) is the
# OLD pre-0.9.11 scheme. CARLA 0.9.11+ uses the Cityscapes-style set below
# (carla.CityObjectLabel). These IDs are stored in the red channel of the RAW
# semantic PNG (semantic/000001.png, saved WITHOUT a ColorConverter).
#
# Versions differ, so check_semantic_table() (called from __main__) compares a
# few labels against your installed carla.CityObjectLabel and prints a warning
# if this table doesn't match -- rebuild it from `print(list(vars(
# carla.CityObjectLabel).items()))` in that case.
#
# Edit THIS table (not the raw PNGs) to change SegFormer's training class
# set later; the raw PNGs always keep CARLA's original numeric IDs so
# nothing here is destructive.
CARLA_SEMANTIC_TO_PROJECT = {
    0:  'unlabeled',
    1:  'road',
    2:  'sidewalk',
    3:  'building',
    4:  'wall',
    5:  'fence',
    6:  'pole',
    7:  'trafficlight',
    8:  'trafficsign',
    9:  'vegetation',
    10: 'terrain',
    11: 'sky',
    12: 'pedestrian',
    13: 'rider',
    14: 'car',
    15: 'truck',
    16: 'bus',
    17: 'train',
    18: 'motorcycle',
    19: 'bicycle',
    20: 'static',
    21: 'dynamic',
    22: 'other',
    23: 'water',
    24: 'roadline',
    25: 'ground',
    26: 'bridge',
    27: 'railtrack',
    28: 'guardrail',
}


def check_semantic_table():
    """Warn if CARLA_SEMANTIC_TO_PROJECT disagrees with the installed
    carla.CityObjectLabel (enum member name -> expected project name)."""
    expected = {
        'Roads': 'road', 'Sidewalks': 'sidewalk', 'Buildings': 'building',
        'Pedestrians': 'pedestrian', 'Rider': 'rider', 'Car': 'car',
        'Truck': 'truck', 'Bus': 'bus', 'Motorcycle': 'motorcycle',
        'Bicycle': 'bicycle', 'RoadLines': 'roadline', 'Vegetation': 'vegetation',
    }
    bad = []
    for enum_name, project_name in expected.items():
        label = getattr(carla.CityObjectLabel, enum_name, None)
        if label is None:
            bad.append(f'{enum_name} (not in this CARLA version)')
            continue
        if CARLA_SEMANTIC_TO_PROJECT.get(int(label)) != project_name:
            bad.append(f'{enum_name}={int(label)} (table says '
                       f'{CARLA_SEMANTIC_TO_PROJECT.get(int(label))!r})')
    if bad:
        print('[warn] CARLA_SEMANTIC_TO_PROJECT does not match this CARLA '
              'install for: ' + ', '.join(bad) +
              ' -- rebuild the table before trusting semantic GT.')
    else:
        print('semantic tag table matches carla.CityObjectLabel')
# ---------------------------------------------------------------------------

# raw_data layout of sensor.lidar.ray_cast_semantic (per CARLA docs)
SEMANTIC_LIDAR_DTYPE = np.dtype([
    ('x', np.float32), ('y', np.float32), ('z', np.float32),
    ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32),
])


def _inv_matrix(transform, sensor):
    """World->sensor 4x4. Prefers the transform captured WITH the measurement
    (data['rgb'].transform) over sensor.get_transform() (same in sync mode,
    but tied to the capture itself)."""
    t = transform if transform is not None else sensor.get_transform()
    return np.array(t.get_inverse_matrix())


def _set_lidar_attrs(lid):
    """Apply CFG lidar attributes; attributes a blueprint doesn't have
    (e.g. dropoff/noise on the semantic lidar) are skipped."""
    attrs = {
        'range': CFG['lidar_range'],
        'channels': CFG['lidar_channels'],
        'points_per_second': CFG['lidar_pps'],
        'rotation_frequency': 20,        # = 1 / fixed_delta_seconds
        'upper_fov': CFG['lidar_upper_fov'],
        'lower_fov': CFG['lidar_lower_fov'],
        'dropoff_general_rate': CFG['lidar_dropoff_general_rate'],
        'dropoff_intensity_limit': 1.0,
        'dropoff_zero_intensity': CFG['lidar_dropoff_zero_intensity'],
        'noise_stddev': CFG['lidar_noise_stddev'],
    }
    for k, v in attrs.items():
        if lid.has_attribute(k):
            lid.set_attribute(k, str(v))


def setup_sensors(world, vehicle, bp):
    sensors = {}
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
    lidar_tf = carla.Transform(carla.Location(z=2.4))

    rgb = bp.find('sensor.camera.rgb')
    rgb.set_attribute('image_size_x', str(CFG['img_w']))
    rgb.set_attribute('image_size_y', str(CFG['img_h']))
    rgb.set_attribute('fov', str(CFG['fov']))
    sensors['rgb'] = world.spawn_actor(rgb, cam_tf, attach_to=vehicle)

    sem = bp.find('sensor.camera.semantic_segmentation')
    sem.set_attribute('image_size_x', str(CFG['img_w']))
    sem.set_attribute('image_size_y', str(CFG['img_h']))
    sem.set_attribute('fov', str(CFG['fov']))
    sensors['sem'] = world.spawn_actor(sem, cam_tf, attach_to=vehicle)

    # NEW: depth sensor -- identical resolution/FOV/transform to RGB so every
    # pixel lines up 1:1, both for the metric depth.npy export and for the
    # occlusion check in extract_2d_bboxes_yolo(). Synchronized the same way
    # as every other sensor here (own queue, picked up generically by
    # grab_synced() below -- no changes needed there).
    depth = bp.find('sensor.camera.depth')
    depth.set_attribute('image_size_x', str(CFG['img_w']))
    depth.set_attribute('image_size_y', str(CFG['img_h']))
    depth.set_attribute('fov', str(CFG['fov']))
    sensors['depth'] = world.spawn_actor(depth, cam_tf, attach_to=vehicle)

    lid = bp.find('sensor.lidar.ray_cast')
    _set_lidar_attrs(lid)
    sensors['lidar'] = world.spawn_actor(lid, lidar_tf, attach_to=vehicle)

    # FIX: semantic LiDAR at the SAME pose/pattern, for exact per-actor point
    # counts (labels/metadata only, never detector input).
    if CFG.get('use_semantic_lidar'):
        lid_sem = bp.find('sensor.lidar.ray_cast_semantic')
        _set_lidar_attrs(lid_sem)
        sensors['lidar_sem'] = world.spawn_actor(lid_sem, lidar_tf, attach_to=vehicle)

    queues = {}
    for name, s in sensors.items():
        q = queue.Queue()
        s.listen(q.put)
        queues[name] = q
    return sensors, queues


def compute_calibration(sensors):
    """NEW: fixed camera<->LiDAR calibration for point painting, in the SAVED
    LiDAR frame (x forward, y left, z up -- velodyne/*.bin) -> OpenCV camera
    frame (x right, y down, z forward) -> pixels. Both sensors are rigidly
    attached to the ego, so this is constant for the whole run: written once
    per town to calib.json.

    Painting a saved point p = (x, y, z, 1):
        pc = Tr_velo_to_cam @ p            # camera coords
        u = (K @ pc[:3])[0] / pc[2]; v = (K @ pc[:3])[1] / pc[2]
    keep only pc[2] > 0 and 0 <= u < img_w, 0 <= v < img_h.
    """
    K = build_projection_matrix(CFG['img_w'], CFG['img_h'], CFG['fov'])
    lidar_to_world = np.array(sensors['lidar'].get_transform().get_matrix())
    world_to_cam = np.array(sensors['rgb'].get_transform().get_inverse_matrix())
    F = np.diag([1.0, -1.0, 1.0, 1.0])            # saved (y-left) -> native CARLA lidar (y-right)
    S = np.array([[0, 1, 0, 0],                   # CARLA cam (x fwd,y right,z up) -> OpenCV
                  [0, 0, -1, 0],
                  [1, 0, 0, 0],
                  [0, 0, 0, 1.0]])
    Tr = (S @ world_to_cam @ lidar_to_world @ F)[:3, :]
    P2 = np.hstack([K, np.zeros((3, 1))])
    return {
        'K': K.tolist(),
        'P2': P2.tolist(),
        'Tr_velo_to_cam': Tr.tolist(),
        'img_w': CFG['img_w'], 'img_h': CFG['img_h'], 'fov': CFG['fov'],
        'lidar_frame': 'x forward, y left, z up (velodyne/*.bin as saved)',
        'camera_frame': 'OpenCV: x right, y down, z forward',
    }


def drain_queues(queues):
    """Empty out any backlog (e.g. from warm-up ticks) so the recording
    loop starts from a clean, synchronized state."""
    for name, q in queues.items():
        n = 0
        while True:
            try:
                q.get_nowait()
                n += 1
            except queue.Empty:
                break
        if n:
            print(f'  drained {n} stale "{name}" frames before recording')


class FrameDropped(Exception):
    """Raised when a sensor genuinely never produced a frame for frame_id
    (as opposed to just arriving slightly out of order)."""
    pass


def grab_synced(queues, frame_id, buffers, timeout=5.0):
    data = {}
    for name, q in queues.items():
        buffered = buffers.get(name)
        if buffered is not None and buffered.frame == frame_id:
            data[name] = buffered
            del buffers[name]
            continue
        if buffered is not None and buffered.frame > frame_id:
            raise FrameDropped(f'{name} has no data for frame {frame_id} '
                                f'(next available is {buffered.frame})')
        if buffered is not None and buffered.frame < frame_id:
            del buffers[name]   # stale leftover from an earlier overshoot, discard
            buffered = None
        while True:
            try:
                item = q.get(timeout=timeout)
            except queue.Empty:
                raise FrameDropped(f'{name} produced nothing for frame '
                                    f'{frame_id} within {timeout}s (stalled/dead sensor)')
            if item.frame == frame_id:
                data[name] = item
                break
            if item.frame > frame_id:
                buffers[name] = item
                raise FrameDropped(f'{name} overshot frame {frame_id} (got {item.frame})')
            # item.frame < frame_id: stale frame (e.g. from the extra ticks
            # inside recycle_walkers()) -- discard and keep reading
    return data


# extract_labels() below was always a no-op stub (returned [] unconditionally)
# -- extract_3d_bboxes_lidar() further down was already doing the real
# KITTI-style 3D GT generation and writing it to labels_3d/. Since the new
# directory layout only has labels_3d/ + label_2/, and the labels/yolo,
# label_2/, the dead call to this stub was removed from collect_town(); the
# function itself is left here unchanged in case you were relying on its
# signature/docstring elsewhere.
def extract_labels(world, lidar_sensor):
    """YOUR JOB — see the CARLA bounding-box tutorial.
    For each vehicle/walker actor:
      1. read actor.bounding_box (extent + transform)
      2. transform world frame -> LiDAR frame
      3. drop actors beyond lidar_range or with no returns
      4. return KITTI-style rows: class, dims, location, rotation_y
    """
    return []          # stub so the pipeline runs; replace this


# ---------------------------------------------------------------------------
# 2D bounding-box (YOLO) ground truth
# ---------------------------------------------------------------------------
# Maps CARLA actor type prefixes -> YOLO class ids. Extend/renumber to match
# whatever your model's data.yaml expects.
YOLO_CLASSES = {
    'person': 0,
    'bicycle': 1,
    'car': 2,
    'motorcycle': 3,
    'bus': 5,
    'truck': 7,
    'traffic_light': 9,
    'stop_sign': 11,
}

# ---------------------------------------------------------------------------
# CARLA actor -> COCO/YOLO class mapping
# ---------------------------------------------------------------------------
# CARLA type_ids never look like 'car...' or 'person...' — they look like
# 'vehicle.tesla.model3' or 'walker.pedestrian.0004' — so we can't prefix-
# match against YOLO_CLASSES directly. Classify by structure instead.

# Known non-"car" vehicle blueprints (CARLA 0.9.13/0.9.14 default content).
# Anything vehicle.* not listed here defaults to 'car'.
BICYCLE_BLUEPRINTS = {
    'vehicle.bh.crossbike',
    'vehicle.diamondback.century',
    'vehicle.gazelle.omafiets',
}
MOTORCYCLE_BLUEPRINTS = {
    'vehicle.harley-davidson.low_rider',
    'vehicle.kawasaki.ninja',
    'vehicle.vespa.zx125',
    'vehicle.yamaha.yzf',
}
BUS_BLUEPRINTS = {
    'vehicle.mitsubishi.fusorosa',
}
TRUCK_BLUEPRINTS = {
    'vehicle.carlamotors.carlacola',
    'vehicle.carlamotors.firetruck',
    'vehicle.carlamotors.european_hgv',
    'vehicle.tesla.cybertruck',
    'vehicle.mercedes.sprinter',
    'vehicle.volkswagen.t2',
    'vehicle.volkswagen.t2_2021',
    'vehicle.ford.ambulance',
}


def classify_actor(actor):
    """Map a CARLA actor to a (class_name, class_id) pair from YOLO_CLASSES,
    or (None, None) if it's not a type we're labeling."""
    type_id = actor.type_id

    if type_id.startswith('walker.pedestrian'):
        return 'person', YOLO_CLASSES['person']

    if not type_id.startswith('vehicle.'):
        return None, None

    # Prefer CARLA's own base_type attribute when present (0.9.14+) --
    # it already speaks car/truck/bus/motorcycle/bicycle.
    base_type = actor.attributes.get('base_type')
    if base_type in YOLO_CLASSES:
        return base_type, YOLO_CLASSES[base_type]

    if type_id in BICYCLE_BLUEPRINTS:
        return 'bicycle', YOLO_CLASSES['bicycle']
    if type_id in MOTORCYCLE_BLUEPRINTS:
        return 'motorcycle', YOLO_CLASSES['motorcycle']
    if type_id in BUS_BLUEPRINTS:
        return 'bus', YOLO_CLASSES['bus']
    if type_id in TRUCK_BLUEPRINTS:
        return 'truck', YOLO_CLASSES['truck']

    # Fallback for any blueprint not in the tables above (e.g. a future
    # CARLA content update): use wheel count as a last resort.
    n_wheels = actor.attributes.get('number_of_wheels')
    if n_wheels == '2':
        return 'motorcycle', YOLO_CLASSES['motorcycle']

    return 'car', YOLO_CLASSES['car']
# ---------------------------------------------------------------------------


def build_projection_matrix(w, h, fov):
    """Pinhole camera intrinsics matrix for CARLA's RGB camera."""
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(location, K, world_2_camera):
    """Project a carla.Location (world space) into 2D image pixel coords.
    Returns (x, y, depth); depth <= 0 means the point is behind the camera."""
    point = np.array([location.x, location.y, location.z, 1.0])
    point_camera = np.dot(world_2_camera, point)

    # CARLA/UE4 world axes are (x-forward, y-right, z-up); standard camera
    # convention is (x-right, y-down, z-forward). Swap accordingly.
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])

    point_img = np.dot(K, point_camera)
    depth = point_img[2]
    if depth != 0:
        point_img[0] /= depth
        point_img[1] /= depth
    return point_img[0], point_img[1], depth


# FIX: near-plane clipping for box projection. The old code skipped vertices
# with depth <= 0 and built the box from the rest, which shrinks the 2D box
# of any actor straddling the camera plane (the camera sits at x=1.5, so
# every neighbouring car passes through this case).
# Edge list for carla.BoundingBox.get_world_vertices() ordering.
BOX_EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def project_box_clipped(verts, K, world_2_camera, near=0.1):
    """Project the 8 world-space box vertices into the image, clipping edges
    that cross the near plane (camera-space depth == near). Returns
    (xs, ys, depths) of every projected point, or None if the whole box is
    behind the camera."""
    cam = []
    for v in verts:
        p = np.dot(world_2_camera, np.array([v.x, v.y, v.z, 1.0]))
        cam.append(np.array([p[1], -p[2], p[0]]))     # OpenCV: x right, y down, z fwd

    pts = [p for p in cam if p[2] > near]
    for a, b in BOX_EDGES:
        pa, pb = cam[a], cam[b]
        if (pa[2] > near) != (pb[2] > near):
            t = (near - pa[2]) / (pb[2] - pa[2])
            pts.append(pa + t * (pb - pa))
    if not pts:
        return None

    xs = [K[0, 0] * p[0] / p[2] + K[0, 2] for p in pts]
    ys = [K[1, 1] * p[1] / p[2] + K[1, 2] for p in pts]
    ds = [p[2] for p in pts]
    return xs, ys, ds


# NEW: metric depth decoding.
def decode_depth(depth_image):
    """Decode a carla.Image from sensor.camera.depth into metric depth
    (meters), shape (H, W) float32.

    CARLA encodes depth across all three color channels for precision:
        normalized = (R + G*256 + B*256^2) / (256^3 - 1)
        depth_m    = 1000.0 * normalized      # 1000 m = camera far plane

    We decode the raw BGRA buffer directly rather than going through
    carla.ColorConverter.Depth, because that converter produces an 8-bit
    grayscale *visualization* image (lossy, not a numeric training target --
    see section 2's explicit warning about this).
    """
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = array.reshape((depth_image.height, depth_image.width, 4)).astype(np.float32)
    B, G, R = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    depth_m = 1000.0 * normalized
    return depth_m.astype(np.float32)


# FIX: practical (non-raycast) occlusion check for 2D YOLO GT.
def is_visible_via_depth(depth_array, img_w, img_h, x_min, y_min, x_max, y_max,
                          actor_d_min, margin=1.0, min_visible_ratio=0.2, grid=3):
    """Section 5 visibility rule.

    Samples a grid x grid lattice of pixels strictly inside the projected 2D
    box against the depth camera's decoded metric buffer. A sample is
    "blocked" if the buffer shows something clearly CLOSER than the nearest
    point of the actor's own 3D box (actor_d_min - margin); otherwise it is
    clear. The box is kept if at least `min_visible_ratio` of samples are
    clear (partial occlusion -- a rider half behind a parked car -- is still
    a valid training example; a fully hidden actor is dropped).

    FIX: the old version compared the buffer against the MEAN corner depth
    (~ the box centre). The depth buffer sees the nearest SURFACE, which for
    a bus/truck is several metres closer than its centre, so clearly visible
    large vehicles were rejected as "occluded". Comparing against the box's
    NEAREST depth removes that bias.

    LIMITATION: this is a 9-point sample, not a per-pixel mask. A thin
    occluder (pole, sign edge) can slip between the samples. If you need
    pixel-perfect occlusion later, use the semantic/ buffer for a per-pixel
    class mask check -- not done here to avoid per-actor raycasts.
    """
    xs = np.linspace(x_min, x_max, grid + 2)[1:-1]
    ys = np.linspace(y_min, y_max, grid + 2)[1:-1]
    n_clear, n_total = 0, 0
    for sy in ys:
        for sx in xs:
            px = int(np.clip(sx, 0, img_w - 1))
            py = int(np.clip(sy, 0, img_h - 1))
            n_total += 1
            if depth_array[py, px] >= actor_d_min - margin:
                n_clear += 1
    return (n_clear / max(n_total, 1)) >= min_visible_ratio


def extract_2d_bboxes_yolo(world, rgb_sensor, ego_vehicle, img_w, img_h, fov,
                            max_distance=None,
                            depth_array=None, visibility_check=False,
                            occlusion_margin=1.0, rgb_transform=None):
    """Project every vehicle/walker actor's 3D bounding box into the RGB
    camera's image plane and return YOLO-format ground-truth lines:

        class_id x_center y_center width height      (all normalized 0..1)

    Notes / limitations:
      - This is a geometric projection PLUS an optional depth-buffer
        visibility check (see is_visible_via_depth() for exactly what that
        does and doesn't catch). It is still not a full per-pixel
        occlusion/segmentation mask.
      - Boxes are clipped to the image bounds; actors that are entirely
        outside the frame or entirely behind the camera are dropped.
        Actors straddling the camera plane are clipped against the near
        plane first so their box isn't truncated.
      - max_distance only controls whether a LABEL is generated -- actors
        beyond it are still rendered normally in the RGB image and still
        exist in the CARLA world (section 3 requirement).
    """
    K = build_projection_matrix(img_w, img_h, fov)
    world_2_camera = _inv_matrix(rgb_transform, rgb_sensor)

    all_actors = world.get_actors()
    actors = list(all_actors.filter('vehicle.*')) + \
             list(all_actors.filter('walker.pedestrian.*'))

    ego_loc = ego_vehicle.get_transform().location
    lines = []

    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        if max_distance is not None:
            if actor.get_transform().location.distance(ego_loc) > max_distance:
                continue

        cls_name, cls_id = classify_actor(actor)
        if cls_id is None:
            continue  # unmapped actor type, skip

        bb = actor.bounding_box
        verts = bb.get_world_vertices(actor.get_transform())

        proj = project_box_clipped(verts, K, world_2_camera, near=CFG['near_plane'])
        if proj is None:
            continue  # entirely behind the camera
        xs, ys, depths = proj

        x_min, x_max = max(min(xs), 0.0), min(max(xs), img_w - 1.0)
        y_min, y_max = max(min(ys), 0.0), min(max(ys), img_h - 1.0)

        if x_max <= x_min or y_max <= y_min:
            continue  # off-screen or degenerate after clipping to the frame

        if (x_max - x_min) < CFG['min_box_px'] or (y_max - y_min) < CFG['min_box_px']:
            continue  # too small to carry real training signal, just noise

        # visibility / occlusion filtering (section 5). Only runs if a
        # decoded depth buffer was actually handed in -- callers that don't
        # care (or don't have a depth frame available) just get the old
        # geometry-only behavior.
        if visibility_check and depth_array is not None:
            if not is_visible_via_depth(depth_array, img_w, img_h,
                                         x_min, y_min, x_max, y_max,
                                         min(depths), margin=occlusion_margin,
                                         min_visible_ratio=CFG['occlusion_min_visible_ratio']):
                continue  # no meaningfully visible pixels -- likely fully occluded

        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        width    = (x_max - x_min) / img_w
        height   = (y_max - y_min) / img_h

        lines.append((cls_name,
                      f'{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'))

    return lines
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3D bounding boxes, in the LiDAR sensor's own frame (PointPillars / OpenPCDet
# style) -- for assessing a point-cloud 3D detector rather than an image one.
# ---------------------------------------------------------------------------
def world_to_lidar_matrix(lidar_sensor, lidar_transform=None):
    """4x4 matrix mapping a world-space point into the LiDAR's local frame --
    the same frame the raw point cloud (saved to velodyne/*.bin) is already
    expressed in, EXCEPT for the y-flip both are given below (see BUGFIX
    note in extract_3d_bboxes_lidar and the y-flip in collect_town's .bin
    writer -- they must both flip y the same way or point counts and box
    positions land in mismatched frames)."""
    return _inv_matrix(lidar_transform, lidar_sensor)


def extract_3d_bboxes_lidar(world, lidar_sensor, lidar_measurement, ego_vehicle,
                             max_distance=None, min_points=0,
                             rgb_transform=None, lidar_transform=None,
                             sem_measurement=None, count_margin=0.0,
                             depth_array=None):
    """Ground-truth 3D boxes in the LiDAR sensor's coordinate frame.

    Each line:
        class x y z dx dy dz yaw num_points actor_id in_cam_fov distance num_points_sem

      x, y, z    -- box center, meters, in the LiDAR frame
                    (x forward, y left, z up -- right-handed, KITTI/velodyne
                    convention; CARLA itself is left-handed with y to the
                    right, so y is negated below to convert).
      dx, dy, dz -- box size along x, y, z (length, width, height), meters.
      yaw        -- heading around the vertical (z) axis, radians,
                    counter-clockwise from the LiDAR's +x axis.
      num_points -- raw points from THIS frame's point cloud that fall
                    inside the box (inflated by count_margin per side).
                    Doubles as a free occlusion/visibility proxy and as the
                    usual KITTI easy/moderate/hard signal. Boxes below
                    `min_points` are dropped (default 0 = keep everything in
                    range; filter per class in the dataloader).
      actor_id   -- CARLA actor id (joins to semantic lidar / other logs).
      in_cam_fov -- 1 if the box is both inside the image frame AND passes
                    the same depth-buffer occlusion check used for the 2D
                    YOLO GT (is_visible_via_depth()) -- i.e. it is actually
                    visible in image_2/<frame>.png, not just geometrically
                    inside the frustum. If no `depth_array` is passed to this
                    call, falls back to the old pure frustum/clip test
                    (ignores occlusion -- useful for a frustum-only/painting
                    pipeline that wants "box touches the image" rather than
                    "box is visible"). 0 if outside the frame / fully behind
                    the camera, -1 if no camera transform was supplied at all.
      distance   -- horizontal distance from the LiDAR origin, meters.
      num_points_sem -- exact per-actor point count from the semantic LiDAR
                    (-1 if the semantic lidar is disabled). 0 here with
                    num_points == 0 means truly occluded, not just sparse.

    IMPORTANT: coordinate-frame conversions (handedness, axis order, sign of
    yaw) are the single easiest thing to get subtly wrong here, and it can't
    be verified without a running simulator. Before trusting a full dataset
    run, sanity-check a handful of frames: plot a few of these boxes over
    their velodyne/*.bin point cloud (e.g. with Open3D) and confirm they
    actually wrap the vehicles, with the heading arrow pointing the way the
    car is facing.
    """
    w2l = world_to_lidar_matrix(lidar_sensor, lidar_transform)  # world -> lidar-local, 4x4
    R = w2l[:3, :3]

    raw = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    cloud_xyz = raw[:, :3].copy()
    # BUGFIX (section 6 coordinate-conversion check): the box center below is
    # converted to the y-left (KITTI) frame via `y = -center_lidar[1]`, but
    # this containment test was comparing that flipped center against the
    # RAW (y-right, native CARLA) cloud -- i.e. two points expressed in
    # mirrored frames, which silently shifted the "inside box" test to
    # y = -true_y and under/over-counted `num_points` for anything not
    # essentially on the y=0 line. Flipping the cloud once here, up front,
    # puts it in the same y-left frame as the box center used below so the
    # containment math is actually comparing like with like. This does NOT
    # change the previously-working `x, y, z` box output convention at all.
    cloud_xyz[:, 1] = -cloud_xyz[:, 1]

    # exact per-actor counts from the semantic lidar (same pose, same pattern)
    sem_counts = None
    if sem_measurement is not None:
        sem_raw = np.frombuffer(sem_measurement.raw_data, dtype=SEMANTIC_LIDAR_DTYPE)
        uniq, cnts = np.unique(sem_raw['ObjIdx'], return_counts=True)
        sem_counts = dict(zip(uniq.tolist(), cnts.tolist()))

    # camera projection for the in_cam_fov flag
    K = cam_w2c = None
    if rgb_transform is not None:
        K = build_projection_matrix(CFG['img_w'], CFG['img_h'], CFG['fov'])
        cam_w2c = np.array(rgb_transform.get_inverse_matrix())

    all_actors = world.get_actors()
    actors = list(all_actors.filter('vehicle.*')) + \
             list(all_actors.filter('walker.pedestrian.*'))

    ego_loc = ego_vehicle.get_transform().location
    lines = []

    def to_saved(vec):
        """Rotate a world-space direction into the LiDAR-local frame, then
        flip y to the saved (x fwd, y left, z up) convention."""
        out = R.dot(np.array([vec.x, vec.y, vec.z]))
        out[1] = -out[1]
        return out

    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        if max_distance is not None:
            if actor.get_transform().location.distance(ego_loc) > max_distance:
                continue

        cls_name, cls_id = classify_actor(actor)
        if cls_id is None:
            continue  # unmapped actor type, skip

        tf = actor.get_transform()
        bb = actor.bounding_box

        # bb.location is an offset from the actor's own origin (non-zero for
        # some pedestrian rigs), so transform it through the actor's pose
        # first to get the box center in world space.
        center_world = tf.transform(bb.location)
        center_h = np.array([center_world.x, center_world.y, center_world.z, 1.0])
        center_lidar = w2l.dot(center_h)[:3]
        x, y, z = center_lidar[0], -center_lidar[1], center_lidar[2]

        dx = bb.extent.x * 2.0   # length, along the actor's forward axis
        dy = bb.extent.y * 2.0   # width
        dz = bb.extent.z * 2.0   # height

        # FIX: heading and box axes from the actor's full orientation vectors
        # rotated through the lidar's world->local transform, instead of a
        # yaw-difference only. Handles ego pitch/roll on slopes (Town03).
        f = to_saved(tf.get_forward_vector())
        r = to_saved(tf.get_right_vector())
        u = to_saved(tf.get_up_vector())
        yaw = float(np.arctan2(f[1], f[0]))   # CCW from +x in the y-left frame

        # raw points inside the (rotated) box -- project the cloud onto the
        # box's own axes, then do a simple axis-aligned containment test.
        # `cloud_xyz` is y-flipped to match (x, y, z) above -- see BUGFIX
        # note where cloud_xyz is built. Box is inflated by count_margin.
        rel = cloud_xyz - np.array([x, y, z])
        local_x = rel.dot(f)
        local_y = rel.dot(r)
        local_z = rel.dot(u)
        inside = (
            (np.abs(local_x) <= dx / 2.0 + count_margin) &
            (np.abs(local_y) <= dy / 2.0 + count_margin) &
            (np.abs(local_z) <= dz / 2.0 + count_margin)
        )
        num_points = int(inside.sum())

        if num_points < min_points:
            continue

        # UPDATED: in_cam_fov is now occlusion-aware when a depth_array is
        # supplied (reuses the same is_visible_via_depth() check used for
        # the 2D YOLO GT, so the two stay consistent with each other).
        # Falls back to a pure frustum/clip test if depth_array is None.
        in_fov = -1
        if K is not None:
            proj = project_box_clipped(bb.get_world_vertices(tf), K, cam_w2c,
                                        near=CFG['near_plane'])
            if proj is None:
                in_fov = 0
            else:
                pxs, pys, pds = proj
                fx_min, fx_max = max(min(pxs), 0.0), min(max(pxs), CFG['img_w'] - 1.0)
                fy_min, fy_max = max(min(pys), 0.0), min(max(pys), CFG['img_h'] - 1.0)
                geom_hit = fx_max > fx_min and fy_max > fy_min
                if geom_hit and depth_array is not None:
                    in_fov = int(is_visible_via_depth(
                        depth_array, CFG['img_w'], CFG['img_h'],
                        fx_min, fy_min, fx_max, fy_max, min(pds),
                        margin=CFG['occlusion_margin'],
                        min_visible_ratio=CFG['occlusion_min_visible_ratio']))
                else:
                    in_fov = int(geom_hit)

        n_sem = sem_counts.get(actor.id, 0) if sem_counts is not None else -1
        distance = float(np.hypot(x, y))

        lines.append(f'{cls_name} {x:.3f} {y:.3f} {z:.3f} '
                     f'{dx:.3f} {dy:.3f} {dz:.3f} {yaw:.4f} {num_points} '
                     f'{actor.id} {in_fov} {distance:.2f} {n_sem}')

    return lines
# ---------------------------------------------------------------------------


# NEW: lane ground truth (section 7).
def extract_lane_gt(world, ego_vehicle, rgb_sensor, img_w, img_h, fov,
                     max_distance=60.0, sample_distance=2.0, rgb_transform=None):
    """Collects nearby lane waypoints (ego lane + immediate left/right
    neighbours) from CARLA's map/waypoint API, samples points forward along
    each lane up to max_distance, and projects them into the RGB camera's
    image plane.

    This is deliberately a clean, MODEL-INDEPENDENT representation (world
    points + projected image points + marking metadata) -- it is NOT the
    literal Ultra-Fast-Lane-Detection v2 row-anchor annotation format.
    UFLD-v2 needs lane x-positions resampled onto a fixed row-anchor grid at
    its own training resolution, which is a deterministic, purely-numeric
    post-processing step you can run on this JSON later -- it depends on
    training config choices, not on anything the simulator can decide, so
    it isn't invented here.

    FIX: points behind the camera (depth <= 0, e.g. the waypoint under the
    ego) are stored as null in image_points instead of a mirrored/garbage
    projection. Their `visible` flag is False as before.

    LIMITATION (documented per instructions): only the ego's current lane
    and its immediate left/right neighbors are captured, not the full lane
    graph (adjacent-but-not-touching lanes, opposing traffic lanes further
    over, lanes beyond an intersection, etc. are not included). If your
    lane detector needs more than 3 lanes in view, extend `lane_sources`
    below using further get_left_lane()/get_right_lane() calls.
    """
    carla_map = world.get_map()
    ego_loc = ego_vehicle.get_transform().location
    K = build_projection_matrix(img_w, img_h, fov)
    world_2_camera = _inv_matrix(rgb_transform, rgb_sensor)

    start_wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
    if start_wp is None:
        return []  # ego off-road / no nearby lane -- document, don't fabricate

    lane_sources = {'ego': start_wp}
    left = start_wp.get_left_lane()
    if left is not None and left.lane_type == carla.LaneType.Driving:
        lane_sources['left'] = left
    right = start_wp.get_right_lane()
    if right is not None and right.lane_type == carla.LaneType.Driving:
        lane_sources['right'] = right

    lanes = []
    for lane_label, wp0 in lane_sources.items():
        world_pts, img_pts, visible_flags = [], [], []
        wp = wp0
        traveled = 0.0
        while wp is not None and traveled <= max_distance:
            loc = wp.transform.location
            world_pts.append([loc.x, loc.y, loc.z])
            x, y, depth = get_image_point(loc, K, world_2_camera)
            vis = bool(depth > 0 and 0 <= x <= img_w and 0 <= y <= img_h)
            img_pts.append([float(x), float(y)] if depth > 0 else None)
            visible_flags.append(vis)

            nxt = wp.next(sample_distance)
            if not nxt:
                break
            wp = nxt[0]
            traveled += sample_distance

        marking = wp0.right_lane_marking if lane_label == 'right' else wp0.left_lane_marking
        lanes.append({
            'lane_id': lane_label,
            'marking_type': str(marking.type) if marking is not None else None,
            'marking_color': str(marking.color) if marking is not None else None,
            'world_points': world_pts,
            'image_points': img_pts,
            'visible': visible_flags,
        })

    return lanes
# ---------------------------------------------------------------------------


def spawn_traffic(world, bp, tm, ego_location=None):
    """Spawn CFG['n_vehicles'] total vehicles, but force a minimum number of
    bicycles and motorcycles first (CFG['min_bicycles']/['min_motorcycles']),
    biased toward spawn points near the ego vehicle.

    Why bias by distance: even after guaranteeing a minimum count, a
    two-wheeler spawned at a random point across the whole map is usually
    far from the ego's camera for most/all of the frames it's ever seen in
    -- producing GT boxes too small to be a useful training example (see
    min_box_px filtering above). Sorting candidate spawn points by distance
    to the ego and preferring the closest ones for two-wheelers means more
    instances end up at a size the model can actually learn from.
    """
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    if ego_location is not None:
        # Points within the bias radius, closest-first; everything else
        # after, still in its (already shuffled) random order.
        near = sorted(
            (sp for sp in spawn_points
             if sp.location.distance(ego_location) <= CFG['two_wheeler_bias_radius']),
            key=lambda sp: sp.location.distance(ego_location))
        far = [sp for sp in spawn_points if sp not in near]
        two_wheeler_spawn_points = near + far
    else:
        two_wheeler_spawn_points = spawn_points

    all_vehicle_bps = list(bp.filter('vehicle.*'))
    bike_bps = [b for b in all_vehicle_bps if b.id in BICYCLE_BLUEPRINTS]
    moto_bps = [b for b in all_vehicle_bps if b.id in MOTORCYCLE_BLUEPRINTS]

    def spawn_forced(blueprint_pool, n_target, label, candidate_points):
        """Try spawn points one at a time, closest-to-ego first, until
        n_target actors of this pool spawn or we run out of points. Points
        that get tried (spawned or not) are removed from the shared pool so
        the general fill-in pass below doesn't reuse them."""
        spawned = 0
        while spawned < n_target and candidate_points:
            sp = candidate_points.pop(0)
            if sp in spawn_points:
                spawn_points.remove(sp)
            chosen_bp = random.choice(blueprint_pool)
            v = world.try_spawn_actor(chosen_bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                vehicles.append(v)
                spawned += 1
        if spawned < n_target:
            print(f'  [warn] only spawned {spawned}/{n_target} {label} '
                  f'(ran out of free spawn points near the ego -- try a '
                  f'larger two_wheeler_bias_radius or fewer n_vehicles)')
        return spawned

    if bike_bps:
        spawn_forced(bike_bps, CFG['min_bicycles'], 'bicycles',
                     list(two_wheeler_spawn_points))
    else:
        print('  [warn] no bicycle blueprints found on this CARLA install -- '
              'check BICYCLE_BLUEPRINTS against your CARLA version\'s content')

    if moto_bps:
        spawn_forced(moto_bps, CFG['min_motorcycles'], 'motorcycles',
                     [sp for sp in two_wheeler_spawn_points if sp in spawn_points])
    else:
        print('  [warn] no motorcycle blueprints found on this CARLA install -- '
              'check MOTORCYCLE_BLUEPRINTS against your CARLA version\'s content')

    # Fill the remaining budget with the general vehicle pool (cars, trucks,
    # buses, and occasionally more two-wheelers by chance).
    # UPDATED: a share (CFG['crowd_vehicle_fraction']) goes on spawn points
    # within CFG['crowd_radius'] of the ego so traffic is dense where the
    # sensors look; the rest stays map-wide. A failed spawn now moves on to the
    # next point (the old `spawn_points[:remaining]` slice silently under-filled
    # whenever a spawn collided).
    remaining = max(0, CFG['n_vehicles'] - len(vehicles))

    def fill(points, target):
        """Try points in order until `target` vehicles spawned or points run
        out. Returns (n_spawned, untried_points)."""
        n, i = 0, 0
        while i < len(points) and n < target:
            sp = points[i]
            i += 1
            v = world.try_spawn_actor(random.choice(all_vehicle_bps), sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                vehicles.append(v)
                n += 1
        return n, points[i:]

    n_near = 0
    if ego_location is not None:
        r = CFG['crowd_radius']
        near_pts = [sp for sp in spawn_points if sp.location.distance(ego_location) <= r]
        far_pts = [sp for sp in spawn_points if sp.location.distance(ego_location) > r]
        n_near_target = int(round(remaining * CFG['crowd_vehicle_fraction']))
        n_near, near_left = fill(near_pts, n_near_target)
        n_far, far_left = fill(far_pts, remaining - n_near)
        short = remaining - n_near - n_far
        if short > 0:                       # backfill from whatever points are left
            n_extra, _ = fill(near_left + far_left, short)
            n_near += n_extra               # (approximate: leftovers are mostly near points)
    else:
        fill(spawn_points, remaining)

    print(f'  spawned {len(vehicles)}/{CFG["n_vehicles"]} vehicles '
          f'(incl. forced two-wheelers; ~{n_near} of the general fill within '
          f'{CFG["crowd_radius"]:.0f} m of the ego)')
    if len(vehicles) < CFG['n_vehicles']:
        print(f'  [warn] {CFG["n_vehicles"] - len(vehicles)} vehicles short of n_vehicles '
              f'(spawn collisions, or n_vehicles exceeds this town\'s '
              f'{len(world.get_map().get_spawn_points())} spawn points)')

    return vehicles


# NEW: crowd helpers (near-ego sampling, dense ego spawn, walker re-targeting).
def sample_nav_pool(world, n, ego_location=None):
    """Sample n random navigation-mesh points and split them into
    (near, far) by distance to the ego (everything is 'far' if no ego given)."""
    near, far = [], []
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        if ego_location is not None and loc.distance(ego_location) <= CFG['crowd_radius']:
            near.append(loc)
        else:
            far.append(loc)
    return near, far


# FIX (destroyed actors): the stored actor lists (traffic / walkers /
# controllers) are built once at spawn time, but CARLA can remove actors
# mid-run -- pedestrians are spawned with is_invincible=false so traffic
# kills them, and vehicles can be removed too. Any method call on a removed
# actor raises "RuntimeError: trying to operate on a destroyed actor", which
# is what crashed recycle_vehicles(). Every function that touches a stored
# list in a later tick now filters it through _alive() first.
def _alive(actors):
    """Return only the actors the simulator has not destroyed."""
    out = []
    for a in actors:
        try:
            if a.is_alive:
                out.append(a)
        except RuntimeError:
            pass
    return out


def _safe_loc(actor):
    """Actor location, or None if the actor is gone."""
    try:
        return actor.get_transform().location
    except RuntimeError:
        return None


def redispatch_walkers(world, controllers, ego_location):
    """Give every walker a fresh destination -- CFG['crowd_walker_fraction']
    of them near the ego. Without this each walker stops for good at its
    first destination and the crowd thins out over a long run.

    NOTE: on towns with disconnected sidewalk areas (Town04 is mostly
    highway) some random destinations are unreachable from a walker's
    current spot, and CARLA prints "WARNING: NAV: Failed to set request to
    go to ...". That walker just keeps its old target / stands still until
    the next redispatch. Harmless.
    """
    controllers[:] = _alive(controllers)     # FIX: drop controllers of dead walkers
    near, far = sample_nav_pool(world, CFG['walker_pool_size'], ego_location)
    pool_all = near + far
    if not pool_all:
        return
    for c in controllers:
        pool = near if (near and random.random() < CFG['crowd_walker_fraction']) else pool_all
        try:
            c.go_to_location(random.choice(pool))
        except RuntimeError:
            pass    # controller already destroyed/stopped -- skip


def _is_ahead(ego_tf, loc):
    """True if `loc` lies inside the ego's forward cone (CFG['recycle_ahead_cos'])."""
    f = ego_tf.get_forward_vector()
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    d = math.hypot(dx, dy)
    return d > 1e-6 and (dx * f.x + dy * f.y) / d >= CFG['recycle_ahead_cos']


def recycle_vehicles(world, vehicles, ego_tf):
    """Teleport vehicles that are far from the ego onto free spawn points
    ahead of it (autopilot keeps driving them). Returns how many moved.

    FIX: prunes destroyed vehicles from `vehicles` IN PLACE first, so the
    caller's `traffic` list stays clean for every later call too. No world
    tick happens in here, so the current frame's labels are unaffected
    (actor poses read by the extract_* functions come from the last tick's
    snapshot; the teleport only shows up next tick, and only beyond lidar
    range anyway).
    """
    ego_loc = ego_tf.location
    far_d = CFG['recycle_far_factor'] * CFG['crowd_radius']

    vehicles[:] = _alive(vehicles)
    locs = {v.id: _safe_loc(v) for v in vehicles}

    movers = [v for v in vehicles
              if locs[v.id] is not None and locs[v.id].distance(ego_loc) > far_d]
    if not movers:
        return 0

    positions = [loc for loc in locs.values() if loc is not None]
    cands = [sp for sp in world.get_map().get_spawn_points()
             if CFG['recycle_min_dist'] <= sp.location.distance(ego_loc) <= CFG['recycle_max_dist']
             and _is_ahead(ego_tf, sp.location)]
    random.shuffle(cands)

    moved = 0
    for v in movers:
        if moved >= CFG['recycle_max_per_call'] or not cands:
            break
        while cands:
            sp = cands.pop()
            if all(sp.location.distance(p) >= CFG['recycle_min_gap'] for p in positions):
                try:
                    v.set_transform(sp)
                except RuntimeError:
                    cands.append(sp)    # vehicle died just now; keep the spot free
                    break
                positions.append(sp.location)
                moved += 1
                break
    return moved


def recycle_walkers(world, bp, walkers, controllers, actors, ego_tf):
    """Replace walkers left far behind with fresh walkers spawned ahead of
    the ego (inside the forward cone, beyond lidar range). Returns how many
    were replaced.

    FIX 1: prunes destroyed walkers/controllers IN PLACE before use (same
           crash as recycle_vehicles -- walkers killed by traffic).
    FIX 2: AI-controlled walkers are no longer teleported. set_transform()
           moves the walker but its navigation agent can stay at the old
           spot, so the follow-up go_to_location() fails with
           "WARNING: NAV: Failed to set request to go to ..." or the walker
           snaps back. Destroy + respawn uses the same
           spawn -> tick -> controller -> tick -> start pattern as
           spawn_walkers(), which already works in this script.

    IMPORTANT: this calls world.tick() twice. If it returns > 0, the world
    has moved on from the frame the caller just grabbed sensor data for, so
    the caller must NOT save that frame (collect_town() skips it). The extra
    sensor frames produced by these ticks are discarded by grab_synced().
    """
    ego_loc = ego_tf.location
    far_d = CFG['recycle_far_factor'] * CFG['crowd_radius']

    walkers[:] = _alive(walkers)
    controllers[:] = _alive(controllers)

    movers = []
    for w in walkers:
        loc = _safe_loc(w)
        if loc is not None and loc.distance(ego_loc) > far_d:
            movers.append(w)
    movers = movers[:CFG['recycle_max_per_call']]
    if not movers:
        return 0

    # candidate nav-mesh points: ahead of the ego, beyond lidar range
    pts = []
    for _ in range(CFG['walker_pool_size']):
        loc = world.get_random_location_from_navigation()
        if (loc is not None
                and CFG['recycle_min_dist'] <= loc.distance(ego_loc) <= CFG['recycle_max_dist']
                and _is_ahead(ego_tf, loc)):
            pts.append(loc)
    if not pts:
        return 0
    random.shuffle(pts)

    # --- 1. stop + destroy the far walkers and their controllers ----------
    mover_ids = {w.id for w in movers}
    doomed_ctrls = []
    for c in controllers:
        try:
            if c.parent is not None and c.parent.id in mover_ids:
                c.stop()
                doomed_ctrls.append(c)
        except RuntimeError:
            pass
    doomed_ctrl_ids = {c.id for c in doomed_ctrls}

    for a in doomed_ctrls + movers:          # controllers first, then walkers
        try:
            if a.is_alive:
                a.destroy()
        except RuntimeError:
            pass

    walkers[:] = [w for w in walkers if w.id not in mover_ids]
    controllers[:] = [c for c in controllers if c.id not in doomed_ctrl_ids]

    # --- 2. spawn replacements ahead of the ego ---------------------------
    walker_bps = list(bp.filter('walker.pedestrian.*'))
    if not walker_bps:
        return 0
    new_walkers = []
    for _ in range(len(movers)):
        if not pts:
            break
        wbp = random.choice(walker_bps)
        if wbp.has_attribute('is_invincible'):
            wbp.set_attribute('is_invincible', 'false')
        w = world.try_spawn_actor(wbp, carla.Transform(pts.pop()))
        if w is not None:
            new_walkers.append(w)
    if not new_walkers:
        return 0

    world.tick()                             # walkers must exist before controllers

    ctrl_bp = bp.find('controller.ai.walker')
    new_ctrls = []
    for w in new_walkers:
        c = world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        if c is not None:
            new_ctrls.append(c)

    world.tick()                             # controllers must exist before start()

    run_ratio = CFG.get('pedestrian_running_ratio', 0.2)
    walk_speed = CFG.get('pedestrian_max_speed_walk', 1.4)
    run_speed = CFG.get('pedestrian_max_speed_run', 3.0)
    for c in new_ctrls:
        try:
            c.start()
            # destination from the same forward window -> same area, so it is
            # far more likely to be reachable than a random map-wide point
            dest = random.choice(pts) if pts else world.get_random_location_from_navigation()
            if dest is not None:
                c.go_to_location(dest)
            c.set_max_speed(run_speed if random.random() < run_ratio else walk_speed)
        except RuntimeError:
            pass

    # --- 3. register for later recycling + final cleanup -------------------
    walkers.extend(new_walkers)
    controllers.extend(new_ctrls)
    actors.extend(new_walkers + new_ctrls)
    return len(new_walkers)


def pick_ego_spawn(world):
    """Ego spawn transform. With CFG['ego_spawn_dense'], pick randomly among
    the top ~10% of spawn points by number of other spawn points within
    CFG['crowd_radius'] -- i.e. busy intersections/city blocks rather than an
    empty road at the edge of the map."""
    sps = world.get_map().get_spawn_points()
    if not CFG.get('ego_spawn_dense'):
        return random.choice(sps)
    r = CFG['crowd_radius']
    density = [sum(1 for o in sps if o.location.distance(sp.location) <= r) for sp in sps]
    cutoff = sorted(density, reverse=True)[max(0, len(sps) // 10 - 1)]
    top = [sp for sp, d in zip(sps, density) if d >= cutoff]
    return random.choice(top)


# NEW: pedestrian spawning. CFG['n_walkers'] existed in the original CFG but
# nothing ever spawned a walker -- spawn_traffic() only ever handles
# vehicle.* blueprints. This fixes that gap: every collect_town() run now
# actually gets pedestrians, with retries so the count is close to
# n_walkers rather than best-effort.
def spawn_walkers(world, bp, n_walkers, ego_location=None):
    """Spawn n_walkers pedestrians with wandering AI controllers, CARLA's
    standard walker pattern (same approach as CARLA's own
    generate_traffic.py example):

      1. try_spawn_actor() a walker.pedestrian.* blueprint at a random
         point from the navigation mesh.
      2. tick once (AI walker controllers are unreliable if start() is
         called in the same tick the walker was spawned in).
      3. spawn a controller.ai.walker attached to each walker, tick again,
         then start() it and send it to another random nav-mesh point.

    Returns (walkers, controllers) -- both must be appended to the town's
    `actors` list for cleanup, and controllers must have .stop() called on
    them BEFORE destroying actors (handled in collect_town's finally block).
    """
    walker_bps = list(bp.filter('walker.pedestrian.*'))
    if not walker_bps:
        print('  [warn] no walker.pedestrian.* blueprints found on this '
              'CARLA install -- no pedestrians will be spawned')
        return [], []

    # NEW: by default pedestrians stay on sidewalks and (almost) never enter
    # the road, which is exactly where the camera looks.
    try:
        world.set_pedestrians_cross_factor(CFG.get('pedestrian_cross_factor', 0.0))
    except Exception as e:
        print(f'  [warn] could not set pedestrian cross factor: {e}')

    # NEW: sample one pool of nav-mesh points, split into near/far of the ego.
    # Near points are used for CFG['crowd_walker_fraction'] of the walkers;
    # the full pool is kept for choosing walk destinations below.
    near_pool, far_pool = sample_nav_pool(world, CFG['walker_pool_size'], ego_location)
    near_all, far_all = list(near_pool), list(far_pool)
    pool_all = near_all + far_all
    n_near_target = (int(round(n_walkers * CFG['crowd_walker_fraction']))
                     if ego_location is not None else 0)
    n_near = 0

    walkers = []
    attempts = 0
    max_attempts = n_walkers * max(1, CFG.get('pedestrian_spawn_retries', 3))

    while len(walkers) < n_walkers and attempts < max_attempts:
        attempts += 1
        from_near = False
        if n_near < n_near_target and near_pool:
            loc, from_near = near_pool.pop(), True
        elif far_pool:
            loc = far_pool.pop()
        elif near_pool:
            loc, from_near = near_pool.pop(), True
        else:
            loc = world.get_random_location_from_navigation()   # pool exhausted
        if loc is None:
            continue
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
        if walker is not None:
            walkers.append(walker)
            if from_near:
                n_near += 1

    if not walkers:
        print('  [warn] spawned 0 pedestrians -- nav mesh may be missing '
              'for this town, or n_walkers/pedestrian_spawn_retries is too low')
        return [], []

    world.tick()  # let the walkers exist server-side before attaching controllers

    controller_bp = bp.find('controller.ai.walker')
    controllers = []
    for walker in walkers:
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            controllers.append(controller)

    world.tick()  # let controllers exist before calling start()/go_to_location()

    running_ratio = CFG.get('pedestrian_running_ratio', 0.2)
    walk_speed = CFG.get('pedestrian_max_speed_walk', 1.4)
    run_speed = CFG.get('pedestrian_max_speed_run', 3.0)

    for controller in controllers:
        try:
            controller.start()
            # UPDATED: destination drawn from the sampled pool, biased toward
            # the ego (falls back to a plain random nav point if the pool is empty).
            if near_all and random.random() < CFG['crowd_walker_fraction']:
                dest = random.choice(near_all)
            elif pool_all:
                dest = random.choice(pool_all)
            else:
                dest = world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(run_speed if random.random() < running_ratio else walk_speed)
        except RuntimeError as e:
            print(f'  [warn] walker controller failed to start: {e}')

    if len(walkers) < n_walkers:
        print(f'  [warn] only spawned {len(walkers)}/{n_walkers} pedestrians '
              f'(nav mesh exhausted or spawn collisions -- this town may just '
              f'have less walkable area; raise pedestrian_spawn_retries to try harder)')

    print(f'  spawned {len(walkers)}/{n_walkers} pedestrians '
          f'({len(controllers)} AI controllers, {n_near} placed within '
          f'{CFG["crowd_radius"]:.0f} m of the ego)')

    return walkers, controllers


# NEW: metadata writer (section 8).
def write_metadata(path, frame_id, town, run_id, ego,
                    n_yolo, n_3d, n_lanes, n_lidar_points, sync_status,
                    sensor_frame_ids):
    tf = ego.get_transform()
    vel = ego.get_velocity()
    speed = float(np.linalg.norm([vel.x, vel.y, vel.z]))
    meta = {
        'frame_id': frame_id,
        'town': town,
        'run_id': run_id,
        'timestamp': time.time(),
        'ego_location': {'x': tf.location.x, 'y': tf.location.y, 'z': tf.location.z},
        'ego_rotation': {'pitch': tf.rotation.pitch, 'yaw': tf.rotation.yaw, 'roll': tf.rotation.roll},
        'ego_velocity': {'x': vel.x, 'y': vel.y, 'z': vel.z},
        'ego_speed': speed,
        'rgb_frame_id': sensor_frame_ids.get('rgb'),
        'semantic_frame_id': sensor_frame_ids.get('sem'),
        'depth_frame_id': sensor_frame_ids.get('depth'),
        'lidar_frame_id': sensor_frame_ids.get('lidar'),
        'lidar_sem_frame_id': sensor_frame_ids.get('lidar_sem'),
        'weather': CFG.get('weather_name'),
        'camera_resolution': [CFG['img_w'], CFG['img_h']],
        'camera_fov': CFG['fov'],
        'lidar_range': CFG['lidar_range'],
        'num_yolo_objects': n_yolo,
        'num_3d_objects': n_3d,
        'num_lanes': n_lanes,
        'num_lidar_points': n_lidar_points,
        'sync_status': sync_status,
        'seed': CFG.get('seed'),
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


# UPDATED: directory structure -- kept FLAT and matching the original
# script's own directory names (dataset/<Town>/<dir>/...), not the nested
# dataset/<Town>/<run_id>/... layout from the first pass, because other
# code on top of this collector already reads image_2/velodyne/semantic/
# labels_yolo/labels_3d directly. run_id is still recorded inside every
# metadata/*.json for provenance -- it just no longer changes the path.
#
#   image_2/        RGB PNGs                    (was rgb/)
#   velodyne/        LiDAR, KITTI-style .bin     [N,4] float32 x,y,z,intensity
#                    (was lidar/ -- .bin now lives under the ORIGINAL
#                    velodyne/ name, matching common KITTI convention where
#                    velodyne/ holds .bin files. The debug .ply, if enabled,
#                    goes in the separate velodyne_ply/ so it never collides
#                    with the .bin the downstream code expects here.)
#   semantic/        raw numeric CARLA class-ID PNGs (the actual GT mask)
#   semantic_visual/ CityScapes-colored PNGs, visualization only, optional
#   depth/           decoded metric depth .npy   (H, W) float16/32, meters
#   depth_raw/       optional visualization depth PNG
#   label_2/         OPTIONAL (CFG['write_label_2']) copy of labels_3d/ --
#                    NOT real KITTI format, off by default (see CFG note).
#   labels_yolo/     2D YOLO GT .txt (was labels/yolo/)
#   labels_3d/       3D box lines: class x y z dx dy dz yaw num_points
#                    actor_id in_cam_fov distance num_points_sem
#   lanes/           lane GT .json
#   metadata/        per-frame metadata .json
#   calib.json       camera/LiDAR calibration for point painting (per town)
def get_run_root(town):
    return f"{CFG['out']}/{town}"


def ensure_dataset_dirs(root):
    subdirs = [
        'image_2', 'velodyne',
        'semantic', 'semantic_visual',
        'depth', 'depth_raw',
        'labels_yolo', 'labels_3d',
        'lanes', 'metadata',
    ]
    if CFG.get('write_label_2'):
        subdirs.append('label_2')
    if CFG.get('save_lidar_ply'):
        subdirs.append('velodyne_ply')
    for d in subdirs:
        os.makedirs(f'{root}/{d}', exist_ok=True)


# NEW: validation / safety checks (section 13).
def validate_dataset(root, town, sample_limit=None):
    """Walk the saved frames for one town/run and sanity-check every
    modality. Prints the summary table your spec asked for. Returns the
    summary dict in case a caller wants to act on it programmatically."""
    rgb_dir = f'{root}/image_2'
    if not os.path.isdir(rgb_dir):
        print(f'  [validate] no image_2/ directory found under {root}, skipping validation')
        return None

    frame_ids = sorted(f[:-4] for f in os.listdir(rgb_dir) if f.endswith('.png'))
    if sample_limit:
        frame_ids = frame_ids[:sample_limit]
    n = len(frame_ids)

    counts = Counter()
    yolo_class_counts = Counter()
    box3d_class_counts = Counter()
    ped_zero_points = 0
    ped_total = 0
    lidar_point_totals = []
    lane_totals = []
    bad_frames = []

    if not os.path.isfile(f'{root}/calib.json'):
        print('  [validate] calib.json missing -- point painting will have no calibration')

    for fid in frame_ids:
        ok = True

        rgb_path = f'{root}/image_2/{fid}.png'
        if os.path.isfile(rgb_path):
            counts['rgb'] += 1
        else:
            ok = False

        sem_path = f'{root}/semantic/{fid}.png'
        if os.path.isfile(sem_path):
            counts['semantic'] += 1
        else:
            ok = False

        depth_path = f'{root}/depth/{fid}.npy'
        if os.path.isfile(depth_path):
            try:
                d = np.load(depth_path)
                if d.ndim == 2 and np.isfinite(d).all():
                    counts['depth'] += 1
                else:
                    ok = False
            except Exception:
                ok = False
        else:
            ok = False

        lidar_path = f'{root}/velodyne/{fid}.bin'
        if os.path.isfile(lidar_path):
            pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            if pts.dtype == np.float32 and pts.shape[1] == 4:
                counts['lidar'] += 1
                lidar_point_totals.append(pts.shape[0])
            else:
                ok = False
        else:
            ok = False

        yolo_path = f'{root}/labels_yolo/{fid}.txt'
        if os.path.isfile(yolo_path):
            valid = True
            with open(yolo_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        valid = False
                        break
                    cls_id, xc, yc, w, h = parts
                    vals = [float(xc), float(yc), float(w), float(h)]
                    if not all(0.0 <= v <= 1.0 for v in vals):
                        valid = False
                        break
                    yolo_class_counts[cls_id] += 1
            counts['yolo'] += 1 if valid else 0
            ok = ok and valid
        else:
            ok = False

        kitti_path = f'{root}/labels_3d/{fid}.txt'
        if os.path.isfile(kitti_path):
            valid = True
            with open(kitti_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 13:     # FIX: 9 -> 13 fields (see extract_3d_bboxes_lidar)
                        valid = False
                        break
                    cls_name = parts[0]
                    nums = [float(p) for p in parts[1:]]
                    if any(np.isnan(v) or np.isinf(v) for v in nums):
                        valid = False
                        break
                    box3d_class_counts[cls_name] += 1
                    if cls_name == 'person':
                        ped_total += 1
                        # FIX: num_points is nums[7] (x y z dx dy dz yaw
                        # num_points actor_id ...). nums[8] is actor_id, so
                        # the old check counted almost nothing.
                        if int(nums[7]) == 0:
                            ped_zero_points += 1
            counts['3d_labels'] += 1 if valid else 0
            ok = ok and valid
        else:
            ok = False

        lane_path = f'{root}/lanes/{fid}.json'
        if os.path.isfile(lane_path):
            try:
                with open(lane_path) as f:
                    lanes = json.load(f)
                valid = isinstance(lanes, list)
                if valid:
                    for lane in lanes:
                        # FIX: points behind the camera are null; points far
                        # off to the side legitimately project outside the
                        # image, so only structure/finite-ness is checked.
                        for pt in lane.get('image_points', []):
                            if pt is None:
                                continue
                            if len(pt) != 2 or not all(np.isfinite(p) for p in pt):
                                valid = False
                                break
                counts['lanes'] += 1 if valid else 0
                lane_totals.append(len(lanes) if valid else 0)
                ok = ok and valid
            except Exception:
                ok = False
        else:
            ok = False

        meta_path = f'{root}/metadata/{fid}.json'
        if os.path.isfile(meta_path):
            try:
                json.load(open(meta_path))
                counts['metadata'] += 1
            except Exception:
                ok = False
        else:
            ok = False

        if not ok:
            bad_frames.append(fid)

    print(f'\nFrames saved: {n}\n')
    for label, key in [('RGB', 'rgb'), ('LiDAR', 'lidar'), ('Semantic', 'semantic'),
                        ('Depth', 'depth'), ('YOLO', 'yolo'), ('3D labels', '3d_labels'),
                        ('Lane GT', 'lanes'), ('Metadata', 'metadata')]:
        print(f'{label + ":":<12}{counts.get(key, 0)}/{n}')

    print('\nYOLO instances by class:',
          dict(yolo_class_counts) if yolo_class_counts else '(none)')
    print('3D instances by class:',
          dict(box3d_class_counts) if box3d_class_counts else '(none)')
    if ped_total:
        print(f'pedestrian 3D boxes with 0 LiDAR points: {ped_zero_points}/{ped_total} '
              f'({100.0 * ped_zero_points / ped_total:.1f}%)')
    print(f'frames with a validation issue: {len(bad_frames)}'
          + (f' (e.g. {bad_frames[:5]})' if bad_frames else ''))
    if lidar_point_totals:
        print(f'average LiDAR points/frame: {np.mean(lidar_point_totals):.0f}')
    if lane_totals:
        print(f'average lanes/frame: {np.mean(lane_totals):.2f}')

    return {
        'n_frames': n, 'counts': dict(counts),
        'yolo_class_counts': dict(yolo_class_counts),
        'box3d_class_counts': dict(box3d_class_counts),
        'bad_frames': bad_frames,
    }


def collect_town(client, town, start_idx):
    print(f'\nloading {town} ...')
    client.load_world(town)
    time.sleep(10)
    world = client.get_world()                      # re-fetch: episode restarted
    world.set_weather(CFG['weather'])

    bp = world.get_blueprint_library()
    tm = client.get_trafficmanager()
    tm.set_random_device_seed(CFG['seed'])          # FIX: reproducible traffic

    original = world.get_settings()
    sensors, queues, actors, controllers = {}, {}, [], []   # defined before try
    class_counts = Counter()                        # GT instance counts, this town
    crowd = Counter()                               # NEW: per-frame crowd-density accumulators

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)                # enable TM sync AFTER world sync
        # NEW: traffic-density tuning (CFG['tm_speed_diff'] > 0 = slower flow)
        tm.global_percentage_speed_difference(CFG['tm_speed_diff'])
        tm.set_global_distance_to_leading_vehicle(CFG['tm_distance_to_leading'])

        # UPDATED: ego starts in a busy area (CFG['ego_spawn_dense']) instead of
        # a uniformly random spawn point.
        ego = world.spawn_actor(
            bp.filter('vehicle.tesla.model3')[0],
            pick_ego_spawn(world))
        ego.set_autopilot(True, tm.get_port())
        actors.append(ego)

        traffic = spawn_traffic(world, bp, tm, ego_location=ego.get_transform().location)
        actors += traffic

        # NEW: pedestrians -- previously n_walkers was configured but never
        # actually spawned anywhere.
        walkers, controllers = spawn_walkers(world, bp, CFG['n_walkers'],
                                             ego_location=ego.get_transform().location)
        actors += walkers
        actors += controllers

        sensors, queues = setup_sensors(world, ego, bp)
        actors += list(sensors.values())

        # UPDATED: new directory layout (section 9) -- dataset/<town>/...
        root = get_run_root(town)
        ensure_dataset_dirs(root)

        for _ in range(60):                         # let traffic settle
            world.tick()

        # NEW: calibration for point painting (fixed rigid mounts -> once per town)
        with open(f'{root}/calib.json', 'w') as f:
            json.dump(compute_calibration(sensors), f, indent=2)

        drain_queues(queues)                        # clear warm-up backlog

        buffers = {}
        dropped_frames = 0
        saved, tick = start_idx, 0
        while saved < start_idx + CFG['frames_per_town']:
            frame_id = world.tick()
            try:
                data = grab_synced(queues, frame_id, buffers)
            except FrameDropped as e:
                dropped_frames += 1
                print(f'\n  [warn] {e} -- skipping this frame '
                      f'({dropped_frames} dropped so far)')
                continue

            tick += 1
            # NEW: keep walkers moving (and drifting toward the ego) -- they
            # otherwise stop for good at their first destination.
            if (controllers and CFG['walker_redispatch_ticks']
                    and tick % CFG['walker_redispatch_ticks'] == 0):
                redispatch_walkers(world, controllers, ego.get_transform().location)
            # NEW: move actors the ego has left far behind to spots ahead of it.
            if CFG['recycle_ticks'] and tick % CFG['recycle_ticks'] == 0:
                ego_tf = ego.get_transform()
                crowd['recycled_vehicles'] += recycle_vehicles(world, traffic, ego_tf)
                if walkers:
                    n_w = recycle_walkers(world, bp, walkers, controllers, actors, ego_tf)
                    crowd['recycled_walkers'] += n_w
                    if n_w:
                        # FIX: recycle_walkers() ticked the world twice, so
                        # actor poses no longer match this frame's sensor
                        # data -- skip saving it instead of writing
                        # misaligned labels.
                        continue
            if tick % CFG['tick_skip']:
                continue

            base = f'{saved:06d}'

            # --- RGB -----------------------------------------------------
            data['rgb'].save_to_disk(f'{root}/image_2/{base}.png')

            # --- Semantic (UPDATED: raw class-ID PNG is now the ground
            # truth, saved into the ORIGINAL semantic/ dir; CityScapes
            # color version is a separate, optional visualization in its
            # own semantic_visual/ dir, never the training target itself) -
            data['sem'].save_to_disk(f'{root}/semantic/{base}.png')
            if CFG.get('save_semantic_visual'):
                data['sem'].save_to_disk(f'{root}/semantic_visual/{base}.png',
                                         carla.ColorConverter.CityScapesPalette)

            # --- NEW: Depth ------------------------------------------------
            depth_array = decode_depth(data['depth'])      # float32, used in-memory for occlusion
            np.save(f'{root}/depth/{base}.npy',
                    depth_array.astype(np.float16) if CFG.get('save_depth_fp16') else depth_array)
            if CFG.get('save_depth_raw_png'):
                data['depth'].save_to_disk(f'{root}/depth_raw/{base}.png',
                                           carla.ColorConverter.Depth)

            # --- NEW: LiDAR .bin (UPDATED representation; .ply optional) --
            # Saved into the ORIGINAL velodyne/ dir (as .bin instead of
            # .ply) -- matches common KITTI convention where velodyne/
            # holds .bin point clouds, and keeps the path any downstream
            # code already points at.
            lidar_points = np.frombuffer(data['lidar'].raw_data, dtype=np.float32).reshape(-1, 4).copy()
            # Flip y to the same x-forward,y-left,z-up (KITTI/right-handed)
            # frame the 3D box labels below use -- see the BUGFIX note in
            # extract_3d_bboxes_lidar for why points and boxes must share a
            # frame. Native CARLA lidar points come out x-forward,y-right,
            # z-up (left-handed), matching the world/vertex convention used
            # everywhere else in this file.
            lidar_points[:, 1] = -lidar_points[:, 1]
            lidar_points.tofile(f'{root}/velodyne/{base}.bin')
            if CFG.get('save_lidar_ply'):
                data['lidar'].save_to_disk(f'{root}/velodyne_ply/{base}.ply')

            # --- 2D YOLO GT (UPDATED: yolo_max_distance + visibility) ----
            yolo_pairs = extract_2d_bboxes_yolo(
                world, sensors['rgb'], ego,
                CFG['img_w'], CFG['img_h'], CFG['fov'],
                max_distance=CFG['yolo_max_distance'],
                depth_array=depth_array,
                visibility_check=CFG['visibility_check'],
                occlusion_margin=CFG['occlusion_margin'],
                rgb_transform=data['rgb'].transform)
            for cls_name, _ in yolo_pairs:
                class_counts[cls_name] += 1
            with open(f'{root}/labels_yolo/{base}.txt', 'w') as f:
                f.write('\n'.join(line for _, line in yolo_pairs))

            # --- 3D GT (UPDATED: no min_points filtering by default,
            # per-actor semantic-lidar counts, occlusion-aware in_cam_fov
            # flag via depth_array) ------------------------------------
            lidar_3d_lines = extract_3d_bboxes_lidar(
                world, sensors['lidar'], data['lidar'], ego,
                max_distance=CFG['lidar_range'], min_points=CFG['min_points'],
                rgb_transform=data['rgb'].transform,
                lidar_transform=data['lidar'].transform,
                sem_measurement=data.get('lidar_sem'),
                count_margin=CFG['point_count_margin'],
                depth_array=depth_array)
            with open(f'{root}/labels_3d/{base}.txt', 'w') as f:
                f.write('\n'.join(lidar_3d_lines))
            if CFG.get('write_label_2'):
                with open(f'{root}/label_2/{base}.txt', 'w') as f:
                    f.write('\n'.join(lidar_3d_lines))

            # --- NEW: crowd-density bookkeeping (printed at the end of the town).
            # "in range" = every vehicle/walker within lidar_range with a 3D
            # label (assumes min_points=0, otherwise it undercounts).
            crowd['frames'] += 1
            crowd['in_range'] += len(lidar_3d_lines)
            crowd['peds_in_range'] += sum(1 for l in lidar_3d_lines if l.startswith('person '))
            crowd['in_cam_fov'] += sum(1 for l in lidar_3d_lines if l.split()[10] == '1')
            crowd['yolo_labels'] += len(yolo_pairs)

            # --- NEW: Lane GT ---------------------------------------------
            lanes = extract_lane_gt(
                world, ego, sensors['rgb'], CFG['img_w'], CFG['img_h'], CFG['fov'],
                max_distance=CFG['lane_max_distance'],
                sample_distance=CFG['lane_sample_distance'],
                rgb_transform=data['rgb'].transform)
            with open(f'{root}/lanes/{base}.json', 'w') as f:
                json.dump(lanes, f)

            # --- NEW: Metadata ----------------------------------------------
            sensor_frame_ids = {name: item.frame for name, item in data.items()}
            sync_ok = len(set(sensor_frame_ids.values())) == 1
            write_metadata(
                f'{root}/metadata/{base}.json', frame_id, town, CFG['run_id'], ego,
                n_yolo=len(yolo_pairs), n_3d=len(lidar_3d_lines), n_lanes=len(lanes),
                n_lidar_points=lidar_points.shape[0],
                sync_status='ok' if sync_ok else 'mismatch',
                sensor_frame_ids=sensor_frame_ids)

            saved += 1
            print(f'{town}: {saved - start_idx}/{CFG["frames_per_town"]}', end='\r')

        print(f'\n{town} done ({dropped_frames} frames dropped/skipped)')
        # FIX: operator precedence -- the old `'...' + ', '.join(...) or '(none)'`
        # bound `or` to the whole concatenation, so '(none)' could never print.
        count_summary = ', '.join(f'{name}={count}' for name, count in
                                   sorted(class_counts.items(), key=lambda kv: -kv[1]))
        print(f'{town} GT instance counts (2D labels): ' + (count_summary or '(none)'))
        if crowd['frames']:
            nf = crowd['frames']
            print(f'{town} avg per frame -- actors within {CFG["lidar_range"]:.0f} m: '
                  f'{crowd["in_range"] / nf:.1f} (pedestrians {crowd["peds_in_range"] / nf:.1f}), '
                  f'in camera FOV: {crowd["in_cam_fov"] / nf:.1f}, '
                  f'2D labels kept: {crowd["yolo_labels"] / nf:.1f}')
            print(f'{town} recycled {crowd["recycled_vehicles"]} vehicle teleports and '
                  f'{crowd["recycled_walkers"]} walker respawns ahead of the ego')

        if CFG.get('validate_after_run'):
            validate_dataset(root, town)

        return saved

    finally:
        print()                                     # FIX: don't overwrite the progress line
        for s in sensors.values():                  # stop before destroy
            try:
                if s.is_alive:
                    s.stop()
            except Exception:
                pass
        for c in controllers:                        # stop walker AI before destroy
            try:
                if c.is_alive:
                    c.stop()
            except Exception:
                pass

        # Only ask the server to destroy actors the client still believes are
        # alive, in one batched RPC. This avoids spamming a raw
        # "ERROR: failed to destroy actor N : unable to destroy actor: not
        # found" line -- printed by the CARLA client library itself, not a
        # catchable Python exception -- for every actor the simulator already
        # removed on its own (e.g. a non-invincible pedestrian killed by traffic,
        # or a walker replaced by recycle_walkers()).
        alive = []
        for a in actors:
            try:
                if a.is_alive:
                    alive.append(a)
            except Exception:
                pass
        if alive:
            responses = client.apply_batch_sync(
                [carla.command.DestroyActor(a.id) for a in alive], True)
            missing = [r.actor_id for r in responses if r.has_error()]
            if missing:
                print(f'  [info] {len(missing)} actor(s) were already gone '
                      f'before cleanup (likely pedestrians killed by traffic '
                      f'mid-run): {missing}')

        tm.set_synchronous_mode(False)
        world.apply_settings(original)
        time.sleep(3)
        print(f'{town} cleaned up')


if __name__ == '__main__':
    random.seed(CFG['seed'])                         # FIX: reproducibility
    np.random.seed(CFG['seed'])

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    check_semantic_table()

    # FIX: progress.json was written but never read. With CFG['resume'] = True
    # a restart skips towns that already finished and continues numbering
    # from the saved index. (A town that crashed mid-way is redone from its
    # own start index, overwriting its partial files.)
    progress_path = 'progress.json'
    progress = {'next_idx': 0, 'done_towns': [], 'ranges': {}}
    if CFG['resume'] and os.path.isfile(progress_path):
        progress.update(json.load(open(progress_path)))
        print(f"resuming: done towns {progress['done_towns']}, next idx {progress['next_idx']}")

    idx = progress['next_idx']
    for town in CFG['towns']:
        if town in progress['done_towns']:
            print(f'skipping {town} (already done)')
            continue
        start = idx
        idx = collect_town(client, town, start)
        progress['next_idx'] = idx
        progress['done_towns'].append(town)
        progress['ranges'][town] = [start, idx - 1]
        json.dump(progress, open(progress_path, 'w'))

        # FIX: split_config was never written anywhere. Split by town (not by
        # random frame -- consecutive frames are highly correlated).
        splits = {t: {'split': CFG['split_config'].get(t, 'train'),
                      'first_frame': r[0], 'last_frame': r[1]}
                  for t, r in progress['ranges'].items()}
        os.makedirs(CFG['out'], exist_ok=True)
        json.dump(splits, open(f"{CFG['out']}/splits.json", 'w'), indent=2)

    print(f'\ntotal frames: {idx}')