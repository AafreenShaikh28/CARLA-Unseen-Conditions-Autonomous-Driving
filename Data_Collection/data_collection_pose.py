import carla, os, queue, random, json, time, math, argparse, faulthandler, shutil
from collections import Counter, OrderedDict, deque
import numpy as np

# A crash inside the CARLA client library ends Python with no traceback. faulthandler writes the Python
# stack of such a crash to this file (appended: a restart must not erase it). On Windows it also logs
# harmless C++ exceptions that CARLA catches itself (code 0xe06d7363): only the LAST entry matters.
CRASH_LOG = open('collector_crash_trace.txt', 'a')
CRASH_LOG.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
CRASH_LOG.flush()
faulthandler.enable(file=CRASH_LOG, all_threads=True)

# ---------------------------------------------------------------------------
# UPDATED (scene upkeep, ported from the LiDAR collector collect_lidar_pose.py): posed pedestrians,
# walkers and traffic stay around the ego until the LAST frame.
#   1. Walker navigation mesh: the CARLA client loads it ONCE, for the map the server had when the
#      script connected, and never again (Town04 got Nav/Town05.bin, Town01 got Town04's). Each town
#      now gets a NEW client after its map is loaded (load_town); >= 85% of the mesh must have ground.
#   2. Unreal updates a walker's skeleton only while a camera renders it: calibration has its own camera
#      (and is retried per town if it failed), the cameras are spawned BEFORE any pedestrian is posed,
#      and 3 small cameras render left / back / right of the ego (keepalive_cams). An empty get_bones()
#      no longer crashes pose_hazard_batch / hazard_actor_box_2d.
#   3. The ego was moved to ANY spawn point when stuck (everything else stayed at the old spot -> empty
#      frames) and blocking cars were DESTROYED (traffic shrank over the run). Now: remove posed
#      pedestrians ahead -> remove walkers ahead -> move the ego 30-200 m to a street with footpaths and
#      rebuild the scene there (repopulate: vehicles 20-70 m away, walkers 8-50 m, posed pedestrians
#      topped up to n_hazard_pedestrians), 2 s settle without saving. A red light is not a jam.
#   4. Walkers: recycle_walkers put replacements 65-130 m ahead (outside the 60 m labels) and destroyed
#      the old ones first, so every failed spawn was a walker lost for good (the LiDAR runs drained to 0).
#      respawn_walkers now spawns 8-35 m from the ego FIRST, then removes as many far walkers, and tops
#      the population up to n_walkers. No walker within 40 m for 15 s (e.g. the Town04 highway, no
#      footpaths) -> the ego is moved back to a street with footpaths.
#   5. Fewer jams: at most hazard_max_road posed pedestrians on lanes / lane edges, none within
#      hazard_junction_gap of a junction, none on the lanes the ego can reach next (ego_route).
#   6. Too few vehicles within crowd_radius -> far ones are moved 20-70 m from the ego.
#   7. Unattended runs: progress.json (now in CFG['out']) is saved every 50 frames and the same command
#      continues where it stopped (--fresh starts over); stops cleanly when the disk is almost full;
#      end-of-town cleanup no longer waits for a tick (it timed out after 60 s in the LiDAR runs).
#   8. A [scene] line every census_every frames: vehicles / walkers within 40 m, posed pedestrians, and in
#      how many of the last frames a posed pedestrian was in the labels.
#   Frames are now written BEFORE the upkeep jobs of a tick (they tick the world), not skipped after them.
#
# UPDATED (latest): HAZARD PLACEMENT RELATIVE TO THE EGO, SPREAD OVER DISTANCE
#   Only ~34 of 10,000 images had a posed pedestrian: replacements were put 65-130 m
#   ahead (labels stop at 60 m) and only replaced once > 150 m away. Now:
#   - every hazard is placed at a random distance in hazard_band (8-55 m) from the ego along
#     the road. New ones are placed in STRATIFIED distance bins (band split into n equal parts,
#     one random distance per part), so the distances are spread over the whole band, not bunched;
#   - as the ego drives, hazards it has passed (hazard_behind_m behind) or that are farther than
#     hazard_far_m are removed and the set is topped up to n_hazard_pedestrians (even if only
#     one pedestrian is missing; spawning ticks the world);
#   - never in the ego's own lane / path ahead (the Traffic Manager stops for a person there):
#     hazard_keep_ego_lane_clear; the ego keeps its lane;
#   - each walker's ground height comes from its own lowest bone after it settled; a pose that fails
#     its check gets 8 more ticks of re-sending before it is dropped;
#   - the run prints how many hazard labels / frames with a hazard were saved, per distance band.
#
# UPDATED (earlier) -- on top of the previous one (hazard-pose pedestrians
# added, LiDAR removed):
#   1. HAZARD POSES NEVER APPLIED (log: lying/sitting/crouching all printed the
#      standing numbers, head 1.64 m, length 0.50 m) -- fixed:
#      a. set_pose() and pose_hazard_batch() now RE-SEND set_bones + blend_pose
#         on every tick (like CARLA's own bone example) instead of once; a
#         one-shot pose is overwritten by the animation update.
#         (Best diagnosis without running CARLA; see the [diag] line below.)
#      b. calibrate_hazard_recipes() spawns the calibration walker on a flat,
#         non-junction road (waypoint height, no ground_projection guess), waits
#         until it is at rest, takes the ground as the lowest bone of the
#         STANDING walker, prints a "standing" line (head ~1.6 m, hips ~1.0 m)
#         and prints a [diag] line if the pose did not change at all.
#         The second run of the log (head -6.47 m) was a walker that fell off
#         the map after being spawned on a random navmesh point.
#      c. Calibration runs at most ONCE per script run (HAZARD_CALIBRATION_TRIED);
#         before, a failed first attempt left the cache empty and
#         spawn_hazard_pedestrians() ran it a second time.
#   2. EGO STUCK ("stopping early: 1000 ticks ... ego stuck?") -- fixed:
#      with no hazards, nothing ever unstuck the ego. New unstick_ticks:
#      ego below 0.3 m/s for this long (20 s, longer than a red light) ->
#      unstick_ego() removes cars in its path ahead, or, if there are none,
#      moves the ego to a free spawn point.
#
# EARLIER UPDATES (kept):
#   - Hazard pedestrians were spawned nearest-first around the ego, so one lay
#     across the ego's lane and the TM stopped for good. Poses are measured once
#     (calibrate_hazard_recipes) and replayed in a batch; the ego's autopilot is
#     switched on only right before the warm-up; no initial hazard in the ego's
#     path; at most max_static_frames saved while the ego stands still;
#     max_ticks stops a run that cannot get moving.
#   - HAZARD RESPAWN: a hazard that traffic ran into/over, shoved, or that was
#     removed is destroyed and replaced 65-130 m ahead of the ego
#     (maintain_hazards). Every spawn/respawn is logged in hazard_pedestrians.json.
#   - frames_per_town=50 at tick_skip=4 is only 10 s of driving: raise it for
#     the real run.
# ---------------------------------------------------------------------------

CFG = dict(
    host = 'localhost', port = 2000, timeout = 60.0, tm_port = 8000,
    towns           = ['Town04','Town01'],
    frames_per_town = 2500,       # a short test: 50 (= only 10 s of driving)
    tick_skip       = 4,
    n_vehicles      = 120,
    n_walkers       = 100,
    min_bicycles    = 20,
    min_motorcycles = 20,
    two_wheeler_bias_radius = 25.0,
    min_box_px = 12,
    img_w = 800, img_h = 600, fov = 90,

    weather = carla.WeatherParameters.ClearNoon,
    out = 'dataset_pose',

    # pedestrian spawning (regular foot traffic)
    pedestrian_running_ratio  = 0.2,
    pedestrian_max_speed_walk = 1.4,
    pedestrian_max_speed_run  = 3.0,
    pedestrian_spawn_retries  = 3,

    crowd_radius             = 100.0,
    crowd_vehicle_fraction   = 0.6,
    crowd_walker_fraction    = 0.7,
    walker_pool_size         = 1500,
    walker_redispatch_ticks  = 600,
    pedestrian_cross_factor  = 0.3,
    tm_speed_diff            = 10.0,
    tm_distance_to_leading   = 1.0,
    ego_spawn_dense          = True,

    recycle_ticks        = 200,    # 0 = off
    recycle_far_factor   = 1.5,
    recycle_min_dist     = 65.0,
    recycle_max_dist     = 130.0,
    recycle_ahead_cos    = 0.5,
    recycle_min_gap      = 8.0,
    recycle_max_per_call = 40,

    yolo_max_distance = 60.0,

    # keep the dataset from filling up with frames of a standing ego.
    max_static_frames = 10,     # stop saving after this many saved-candidate frames with the ego standing still
    stuck_ticks       = 100,    # ego speed < 0.3 m/s this many ticks (5 s) AND a hazard pedestrian in its path
                                #   -> that hazard is respawned elsewhere (a red light alone does not trigger it)
    unstick_ticks     = 200,    # UPDATED: ego standing still 10 s (not at a red light) -> one escalation step:
                                #   remove posed pedestrians ahead -> remove walkers ahead -> move the ego (0 = off)
    max_ticks_factor  = 5,      # safety net: a town stops after frames_per_town * tick_skip * this ticks
    ego_speed_diff    = 30.0,   # NEW: ego only, % below the speed limit: posed pedestrians stay in view longer

    # NEW: scene upkeep (see the UPDATED notes at the top)
    walker_respawn_ticks = 100,          # every 5 s: walkers > walker_far_m away are replaced near the ego (0 = off)
    walker_far_m         = 60.0,
    walker_spawn_band    = (8.0, 35.0),  # m from the ego for the replacements
    walker_respawn_max   = 30,           # per call
    min_vehicles_near    = 20,           # fewer vehicles than this within crowd_radius -> far ones moved closer
    vehicle_place_band   = (20.0, 70.0), # m from the ego for those
    teleport_band        = (30.0, 200.0),# a stuck ego is moved this far, to a street with footpaths
    settle_ticks         = 40,           # 2 s without saving after the scene was rebuilt around a moved ego
    empty_move_ticks     = 300,          # no walker within 40 m for 15 s -> move the ego to a street with footpaths
    tm_reset_on_move     = True,         # autopilot off / on around a teleport (the TM plans a new route)
    keepalive_cams       = True,         # 3 small cameras (left / back / right) so posed walkers get rendered
    keepalive_res        = (320, 240),
    spectator_follow     = True,         # the server window follows the ego (and renders the area around it)
    census_every         = 100,          # print what is around the ego every N saved frames (0 = off)
    teleport_test        = 0,            # set by --teleport-test: move the ego every N ticks (test the rebuild)
    nav_min_ground       = 0.85,         # share of navigation points that must have ground under them
    min_free_gb          = 3.0,          # stop cleanly (progress saved) below this much free disk space

    # hazard-pose pedestrians (lying / sitting / crouching), see hazard_pose_test.py
    n_hazard_pedestrians  = 20,       # kept alive around the ego at any time (topped up as it drives)
    hazard_poses          = {'lying': 0.5, 'sitting': 0.25, 'crouching': 0.25},
    hazard_places         = {'lane': 0.3, 'edge': 0.3, 'footpath': 0.4},
    hazard_band           = (8.0, 55.0),   # m from the ego along the road: placement distances are spread over
                                           # this band (keep hi below yolo_max_distance so they get labelled)
    hazard_front_fraction = 1.0,           # share placed AHEAD of the ego (the camera faces forward)
    hazard_far_m          = 60.0,          # farther than this from the ego -> removed and replaced
    hazard_behind_m       = 8.0,           # passed by the ego by this much -> removed and replaced
    hazard_min_batch      = 1,             # top up after any loss (spawning ticks the world)
    hazard_respawn_max    = 15,            # at most this many per top-up
    hazard_keep_ego_lane_clear = True,     # no lane / edge hazard in the ego's lane or path ahead
    hazard_max_road       = 8,             # NEW: at most this many on lanes / lane edges (each stops the cars behind it)
    hazard_verbose        = False,         # NEW: print every [hazard-spawn] / [hazard-maintenance] line (all are
                                           #      in hazard_pedestrians.json anyway; the [scene] line sums them up)
    hazard_junction_gap   = 15.0,          # NEW: m; no lane / edge hazard this close to a junction
    hazard_blueprint_pool = None,
    hazard_check_ticks    = 10,      # hit / shoved / gone / far / behind / blocking check + top-up every N ticks (0 = off)
    hazard_respawn        = True,    # replace those pedestrians (False = only log them)
    hazard_shove_m        = 0.5,     # moved farther than this from where it was posed -> counts as hit
    # a vehicle closer than this (m) to the hazard's centre counts as having hit it (vehicle box vs disc)
    hazard_hit_radius     = {'lying': 0.6, 'sitting': 0.5, 'crouching': 0.4},

    visibility_check = True,
    occlusion_margin  = 1.0,
    occlusion_min_visible_ratio = 0.2,
    near_plane = 0.1,

    save_depth_raw_png  = False,
    save_semantic_visual = True,
    save_depth_fp16     = True,

    lane_sample_distance = 2.0,
    lane_max_distance    = 60.0,

    run_id = 'run_0001',
    weather_name = 'ClearNoon',
    split_config = {'Town01': 'train', 'Town04': 'train', 'Town05': 'val'},

    validate_after_run = True,

    seed   = 0,
    resume = True,      # UPDATED: continue from CFG['out']/progress.json (python ... --fresh starts over)
)


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


def _inv_matrix(transform, sensor):
    """World->sensor 4x4. Prefers the transform captured WITH the measurement
    (data['rgb'].transform) over sensor.get_transform()."""
    t = transform if transform is not None else sensor.get_transform()
    return np.array(t.get_inverse_matrix())


def setup_sensors(world, vehicle, bp):
    """UPDATED: + keepalive_cams (left / back / right of the ego). Unreal only updates a walker's skeleton
    while a camera renders it, so a posed pedestrian beside or behind the ego kept its standing bones and
    failed the pose check. The extra cameras are in `sensors` (stopped and destroyed with the others) but
    not in `queues`: grab_synced never waits for them and their images are dropped by the empty callback."""
    sensors = {}
    cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))

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

    depth = bp.find('sensor.camera.depth')
    depth.set_attribute('image_size_x', str(CFG['img_w']))
    depth.set_attribute('image_size_y', str(CFG['img_h']))
    depth.set_attribute('fov', str(CFG['fov']))
    sensors['depth'] = world.spawn_actor(depth, cam_tf, attach_to=vehicle)

    queues = {}
    for name, s in sensors.items():
        q = queue.Queue()
        s.listen(q.put)
        queues[name] = q
    if CFG['keepalive_cams']:
        w, h = CFG['keepalive_res']
        for yaw in (90.0, 180.0, -90.0):
            b = bp.find('sensor.camera.rgb')
            for k, v in (('image_size_x', w), ('image_size_y', h), ('fov', 100.0)):
                b.set_attribute(k, str(v))
            cam = world.spawn_actor(b, carla.Transform(carla.Location(z=2.4), carla.Rotation(yaw=yaw)),
                                    attach_to=vehicle)
            cam.listen(lambda _img: None)
            sensors[f'keepalive_{int(yaw)}'] = cam
    return sensors, queues


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
            # inside recycle_walkers() / maintain_hazards()) -- discard and keep reading
    return data


def extract_labels(world, lidar_sensor):
    """Left unchanged: always was a no-op stub (see earlier notes)."""
    return []


# ---------------------------------------------------------------------------
# 2D bounding-box (YOLO) ground truth
# ---------------------------------------------------------------------------
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
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])

    point_img = np.dot(K, point_camera)
    depth = point_img[2]
    if depth != 0:
        point_img[0] /= depth
        point_img[1] /= depth
    return point_img[0], point_img[1], depth


BOX_EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
             [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def project_box_clipped(verts, K, world_2_camera, near=0.1):
    """Project the 8 world-space box vertices into the image, clipping edges
    that cross the near plane. Returns (xs, ys, depths) or None if the whole
    box is behind the camera."""
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


def decode_depth(depth_image):
    """Decode a carla.Image from sensor.camera.depth into metric depth
    (meters), shape (H, W) float32."""
    array = np.frombuffer(depth_image.raw_data, dtype=np.uint8)
    array = array.reshape((depth_image.height, depth_image.width, 4)).astype(np.float32)
    B, G, R = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    normalized = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 ** 3 - 1.0)
    depth_m = 1000.0 * normalized
    return depth_m.astype(np.float32)


def is_visible_via_depth(depth_array, img_w, img_h, x_min, y_min, x_max, y_max,
                          actor_d_min, margin=1.0, min_visible_ratio=0.2, grid=3):
    """Samples a grid x grid lattice of pixels inside the projected 2D box
    against the depth buffer; a sample is "blocked" if something clearly
    closer than the actor's nearest depth is there. Kept if at least
    `min_visible_ratio` of samples are clear. 9-point sample, not a per-pixel
    mask: a thin occluder can slip between samples."""
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
                            occlusion_margin=1.0, rgb_transform=None,
                            hazard_actor_ids=None):
    """Project every vehicle/walker actor's 3D bounding box into the RGB
    camera's image plane and return (class_name, yolo_line, is_hazard, min_depth_m)
    tuples; yolo_line is

        class_id x_center y_center width height      (all normalized 0..1)

    Actors in `hazard_actor_ids` (lying/sitting/crouching pedestrians) get
    their box from their CURRENT bone positions (hazard_actor_box_2d())
    instead of actor.bounding_box, which stays an upright capsule whatever
    the pose. `hazard_actor_ids` is a set that is updated in place when
    hazards are respawned, so always pass the same set object.
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
            continue

        if hazard_actor_ids and actor.id in hazard_actor_ids:
            box = hazard_actor_box_2d(actor, K, world_2_camera, img_w, img_h,
                                       near=CFG['near_plane'])
            if box is None:
                continue
            x_min, y_min, x_max, y_max, min_depth = box
        else:
            bb = actor.bounding_box
            verts = bb.get_world_vertices(actor.get_transform())

            proj = project_box_clipped(verts, K, world_2_camera, near=CFG['near_plane'])
            if proj is None:
                continue
            xs, ys, depths = proj

            x_min, x_max = max(min(xs), 0.0), min(max(xs), img_w - 1.0)
            y_min, y_max = max(min(ys), 0.0), min(max(ys), img_h - 1.0)
            min_depth = min(depths)

        if x_max <= x_min or y_max <= y_min:
            continue

        if (x_max - x_min) < CFG['min_box_px'] or (y_max - y_min) < CFG['min_box_px']:
            continue

        if visibility_check and depth_array is not None:
            if not is_visible_via_depth(depth_array, img_w, img_h,
                                         x_min, y_min, x_max, y_max,
                                         min_depth, margin=occlusion_margin,
                                         min_visible_ratio=CFG['occlusion_min_visible_ratio']):
                continue

        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        width    = (x_max - x_min) / img_w
        height   = (y_max - y_min) / img_h

        is_hazard = bool(hazard_actor_ids and actor.id in hazard_actor_ids)
        lines.append((cls_name,
                      f'{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}',
                      is_hazard, float(min_depth)))

    return lines
# ---------------------------------------------------------------------------


def extract_lane_gt(world, ego_vehicle, rgb_sensor, img_w, img_h, fov,
                     max_distance=60.0, sample_distance=2.0, rgb_transform=None):
    """Nearby lane waypoints (ego lane + immediate left/right neighbours),
    sampled forward and projected into the RGB image. Points behind the
    camera are stored as null in image_points."""
    carla_map = world.get_map()
    ego_loc = ego_vehicle.get_transform().location
    K = build_projection_matrix(img_w, img_h, fov)
    world_2_camera = _inv_matrix(rgb_transform, rgb_sensor)

    start_wp = carla_map.get_waypoint(ego_loc, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
    if start_wp is None:
        return []

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
    """Spawn CFG['n_vehicles'] total vehicles, forcing a minimum number of
    bicycles and motorcycles first, biased toward spawn points near the ego."""
    vehicles = []
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    if ego_location is not None:
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

    remaining = max(0, CFG['n_vehicles'] - len(vehicles))

    def fill(points, target):
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
        if short > 0:
            n_extra, _ = fill(near_left + far_left, short)
            n_near += n_extra
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
    of them near the ego."""
    controllers[:] = _alive(controllers)
    near, far = sample_nav_pool(world, CFG['walker_pool_size'], ego_location)
    pool_all = near + far
    if not pool_all:
        return
    for c in controllers:
        pool = near if (near and random.random() < CFG['crowd_walker_fraction']) else pool_all
        try:
            c.go_to_location(random.choice(pool))
        except RuntimeError:
            pass


def _is_ahead(ego_tf, loc):
    """True if `loc` lies inside the ego's forward cone (CFG['recycle_ahead_cos'])."""
    f = ego_tf.get_forward_vector()
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    d = math.hypot(dx, dy)
    return d > 1e-6 and (dx * f.x + dy * f.y) / d >= CFG['recycle_ahead_cos']


def recycle_vehicles(world, vehicles, ego_tf, tm=None, band=None, far_m=None, max_n=None):
    """Teleport vehicles farther than far_m (default recycle_far_factor x crowd_radius) from the ego to
    free spawn points band m from it (default recycle_min_dist-recycle_max_dist), ahead of it first.
    UPDATED: a closer band (vehicle_place_band) refills an empty scene, e.g. after the ego was moved; the
    moved vehicles' autopilot is switched off and on again (one tick in between) so the Traffic Manager
    plans a new route from the new spot, and their speed is set to 0. Returns how many moved."""
    vehicles[:] = _alive(vehicles)
    far_m = CFG['recycle_far_factor'] * CFG['crowd_radius'] if far_m is None else far_m
    lo, hi = (CFG['recycle_min_dist'], CFG['recycle_max_dist']) if band is None else band
    max_n = CFG['recycle_max_per_call'] if max_n is None else max_n
    ego_loc = ego_tf.location
    locs = {v.id: _safe_loc(v) for v in vehicles}
    movers = sorted((v for v in vehicles if locs[v.id] is not None and locs[v.id].distance(ego_loc) > far_m),
                    key=lambda v: -locs[v.id].distance(ego_loc))[:max_n]          # farthest first
    if not movers:
        return 0
    ahead, other = [], []
    for sp in world.get_map().get_spawn_points():
        if lo <= sp.location.distance(ego_loc) <= hi:
            (ahead if _is_ahead(ego_tf, sp.location) else other).append(sp)
    random.shuffle(ahead)
    random.shuffle(other)
    cands = other + ahead                                     # pop() takes from the end: ahead first
    taken = [p for p in locs.values() if p is not None] + [ego_loc]
    plan = []
    for v in movers:
        while cands:
            sp = cands.pop()
            if all(sp.location.distance(p) >= CFG['recycle_min_gap'] for p in taken):
                plan.append((v, sp))
                taken.append(sp.location)
                break
    port = tm.get_port() if tm is not None and CFG['tm_reset_on_move'] else None
    moved = []
    for v, sp in plan:
        try:
            if port is not None:
                v.set_autopilot(False, port)
            v.set_transform(sp)
            v.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            moved.append(v)
        except RuntimeError:
            pass
    if moved and port is not None:
        world.tick()                                          # the Traffic Manager drops the old routes
        for v in moved:
            try:
                v.set_autopilot(True, port)
            except RuntimeError:
                pass
    return len(moved)


def ground_below(world, loc):
    """True if there is static geometry under loc (a navigation point of the wrong mesh has none)."""
    try:
        return world.ground_projection(carla.Location(loc.x, loc.y, loc.z + 2.0), 6.0) is not None
    except (AttributeError, RuntimeError):
        return True                                           # API not available: cannot check


def nav_point(world, ego_loc, near=True, tries=30):
    """A navigation point with ground under it, within crowd_radius of the ego if near (else anywhere)."""
    fallback = None
    for _ in range(tries):
        loc = world.get_random_location_from_navigation()
        if loc is None or not ground_below(world, loc):
            continue
        if not near or loc.distance(ego_loc) <= CFG['crowd_radius']:
            return loc
        fallback = loc
    return fallback


def spawn_walker_at(world, b, loc):
    """try_spawn_actor right on a navigation point often fails (the capsule touches the kerb or the
    ground): also try 0.5 m and 1 m higher; the walker drops onto the ground."""
    for dz in (0.0, 0.5, 1.0):
        w = world.try_spawn_actor(b, carla.Transform(carla.Location(loc.x, loc.y, loc.z + dz)))
        if w is not None:
            return w
    return None


def sidewalk_points_near(world, ego_loc, n, band, avoid=()):
    """Up to n points on the footpaths beside the roads around the ego, band[0]-band[1] m away (map lanes
    of type Sidewalk): fast in big towns, where random navigation points rarely land near the ego."""
    wp0 = world.get_map().get_waypoint(ego_loc)
    if wp0 is None:
        return []
    lo, hi = band
    out, avoid = [], list(avoid)
    for _ in range(12 * n):
        if len(out) >= n:
            break
        d = random.uniform(max(lo, 1.0), hi)
        cands = wp0.next(d) if random.random() < 0.6 else wp0.previous(d)
        if not cands:
            continue
        wp = random.choice(cands)
        if random.random() < 0.4:                              # sometimes the road beside this lane
            adj = wp.get_left_lane() if random.random() < 0.5 else wp.get_right_lane()
            wp = adj if adj is not None else wp
        side = wp if wp.lane_type == carla.LaneType.Sidewalk else sidewalk_beside(wp, random.choice((1, -1)))
        if side is None:
            continue
        r, off = side.transform.get_right_vector(), random.uniform(-0.6, 0.6)
        loc = side.transform.location + carla.Location(x=r.x * off, y=r.y * off, z=1.0)
        if not lo <= loc.distance(ego_loc) <= hi + 5.0:
            continue
        if any(loc.distance(p) < 1.5 for p in avoid + out) or not ground_below(world, loc):
            continue
        out.append(loc)
    return out


def walker_points_near(world, ego_loc, n, band, tries_per=40):
    """n walkable points band[0]-band[1] m from the ego: random navigation points first (on the
    navigation mesh by construction), footpath points from the map for the rest."""
    lo, hi = band
    pts, tries = [], 0
    while len(pts) < n and tries < tries_per * n:
        tries += 1
        if tries > 300 and len(pts) < 0.01 * tries:            # big town: hardly any land near the ego
            break
        loc = world.get_random_location_from_navigation()
        if loc is not None and lo <= loc.distance(ego_loc) <= hi and ground_below(world, loc):
            pts.append(loc)
    if len(pts) < n:
        pts += sidewalk_points_near(world, ego_loc, n - len(pts), band, avoid=pts)
    return pts


def respawn_walkers(world, bp, walkers, controllers, actors, ego_loc, far_m=None, band=None, max_n=None):
    """REPLACES recycle_walkers. Keep pedestrians in range while the ego drives on (walkers at 1.4 m/s
    can't follow): new walkers band m (walker_spawn_band, 8-35 m: inside the 60 m labels) from the ego
    replace walkers more than far_m (walker_far_m) away. The new walkers are spawned FIRST and only as
    many far walkers as are needed are removed afterwards; the population is topped up to n_walkers
    (walkers removed by the unstick steps come back too). recycle_walkers destroyed first and put the
    replacements 65-130 m ahead: every failed spawn was a walker lost for good.
    Ticks the world twice: call it only AFTER the frame of this tick has been written. -> number spawned"""
    far_m = CFG['walker_far_m'] if far_m is None else far_m
    band = CFG['walker_spawn_band'] if band is None else band
    max_n = CFG['walker_respawn_max'] if max_n is None else max_n
    walkers[:] = _alive(walkers)
    controllers[:] = _alive(controllers)
    dist = {}
    for w in walkers:
        loc = _safe_loc(w)
        if loc is not None:
            dist[w.id] = loc.distance(ego_loc)
    far = sorted((w for w in walkers if dist.get(w.id, 0.0) > far_m), key=lambda w: -dist[w.id])[:max_n]
    deficit = max(0, CFG['n_walkers'] - len(walkers))
    want = len(far) + deficit
    if not want:
        return 0
    pts = walker_points_near(world, ego_loc, want, band)
    walker_bps = list(bp.filter('walker.pedestrian.*'))
    new = []
    for loc in pts:                                           # 1. spawn the new walkers
        b = random.choice(walker_bps)
        if b.has_attribute('is_invincible'):
            b.set_attribute('is_invincible', 'false')
        w = spawn_walker_at(world, b, loc)
        if w is not None:
            new.append(w)
    if not new:
        return 0
    n_old = min(len(far), max(0, len(walkers) + len(new) - CFG['n_walkers']))
    old_ids = {w.id for w in far[:n_old]}                     # 2. then remove only that many far ones
    for c in list(controllers):                               # controllers first, then their walkers
        try:
            if c.parent is not None and c.parent.id in old_ids:
                c.stop()
                if c.is_alive:
                    c.destroy()
                controllers.remove(c)
        except RuntimeError:
            pass
    for w in far[:n_old]:
        try:
            if w.is_alive:
                w.destroy()
        except RuntimeError:
            pass
    walkers[:] = [w for w in walkers if w.id not in old_ids]
    world.tick()                                              # walkers must exist before controllers
    ctrl_bp = bp.find('controller.ai.walker')
    new_ctrls = []
    for w in new:
        c = world.try_spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
        if c is not None:
            new_ctrls.append(c)
    world.tick()                                              # controllers must exist before start()
    run_ratio = CFG.get('pedestrian_running_ratio', 0.2)
    for c in new_ctrls:
        try:
            c.start()
            dest = nav_point(world, ego_loc)
            if dest is not None:
                c.go_to_location(dest)
            c.set_max_speed(CFG['pedestrian_max_speed_run'] if random.random() < run_ratio
                            else CFG['pedestrian_max_speed_walk'])
        except RuntimeError:
            pass
    walkers.extend(new)
    controllers.extend(new_ctrls)
    actors.extend(new + new_ctrls)
    return len(new)


def count_near(actors, loc, r):
    n = 0
    for a in _alive(actors):
        p = _safe_loc(a)
        if p is not None and p.distance(loc) <= r:
            n += 1
    return n





def footpath_spawn_points(world, min_n=10):
    """NEW. Spawn points with a footpath (map lane of type Sidewalk) beside their road, [] if the town has
    fewer than min_n (then nothing is filtered). Walkers and posed pedestrians on footpaths can only be
    placed there: on the Town04 highway loop the LiDAR run had 0 walkers within 40 m."""
    cmap = world.get_map()
    out = []
    for sp in cmap.get_spawn_points():
        wp = cmap.get_waypoint(sp.location)
        if wp is not None and (sidewalk_beside(wp, 1) is not None or sidewalk_beside(wp, -1) is not None):
            out.append(sp)
    return out if len(out) >= min_n else []


def pick_ego_spawn(world):
    """Ego spawn transform. With CFG['ego_spawn_dense'], pick randomly among the top ~10% of spawn points
    by number of other spawn points nearby. UPDATED: among spawn points with a footpath beside them."""
    sps = footpath_spawn_points(world) or world.get_map().get_spawn_points()
    if not CFG.get('ego_spawn_dense'):
        return random.choice(sps)
    r = CFG['crowd_radius']
    density = [sum(1 for o in sps if o.location.distance(sp.location) <= r) for sp in sps]
    cutoff = sorted(density, reverse=True)[max(0, len(sps) // 10 - 1)]
    top = [sp for sp, d in zip(sps, density) if d >= cutoff]
    return random.choice(top)


def spawn_walkers(world, bp, n_walkers, ego_location=None):
    """Spawn n_walkers pedestrians with wandering AI controllers (CARLA's
    standard walker pattern). Returns (walkers, controllers)."""
    walker_bps = list(bp.filter('walker.pedestrian.*'))
    if not walker_bps:
        print('  [warn] no walker.pedestrian.* blueprints found on this '
              'CARLA install -- no pedestrians will be spawned')
        return [], []

    try:
        world.set_pedestrians_cross_factor(CFG.get('pedestrian_cross_factor', 0.0))
    except Exception as e:
        print(f'  [warn] could not set pedestrian cross factor: {e}')

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
            loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker = spawn_walker_at(world, walker_bp, loc)       # UPDATED: also 0.5 / 1 m higher
        if walker is not None:
            walkers.append(walker)
            if from_near:
                n_near += 1

    if not walkers:
        print('  [warn] spawned 0 pedestrians -- nav mesh may be missing '
              'for this town, or n_walkers/pedestrian_spawn_retries is too low')
        return [], []

    world.tick()

    controller_bp = bp.find('controller.ai.walker')
    controllers = []
    for walker in walkers:
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller is not None:
            controllers.append(controller)

    world.tick()

    running_ratio = CFG.get('pedestrian_running_ratio', 0.2)
    walk_speed = CFG.get('pedestrian_max_speed_walk', 1.4)
    run_speed = CFG.get('pedestrian_max_speed_run', 3.0)

    for controller in controllers:
        try:
            controller.start()
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


# ---------------------------------------------------------------------------
# Hazard-pose pedestrians -- person lying / sitting / crouching on or beside
# the road. CARLA has no such animation; the pose is set directly on the
# walker's skeleton via the bone API (CARLA 0.9.13+). The pose-search helpers
# (AXES through Poser) are ported from hazard_pose_test.py.
# The pose search runs ONCE per script run (calibrate_hazard_recipes) and the
# resulting bone changes are replayed on every hazard walker in a batch
# (pose_hazard_batch).
# UPDATED: set_bones + blend_pose are re-sent on every tick in set_pose() and
# pose_hazard_batch(); a one-shot pose was overwritten by the animation update.
# ---------------------------------------------------------------------------
AXES = ('pitch', 'yaw', 'roll')
HAZARD_RECIPES = {}     # pose -> dict(changes=..., how=...), filled by calibrate_hazard_recipes()
HAZARD_SK = {}          # skeleton bone names of the crl_ walker, filled with it
HAZARD_CALIBRATION_TRIED = False   # calibration runs at most once per script run


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


def relative_pose(walker):
    out = OrderedDict()
    for b in walker.get_bones().bone_transforms:
        r, loc = b.relative.rotation, b.relative.location
        out[b.name] = carla.Transform(carla.Location(loc.x, loc.y, loc.z),
                                       carla.Rotation(pitch=r.pitch, yaw=r.yaw, roll=r.roll))
    return out


def build_ctl(base, changes):
    """base: bone -> relative Transform (the idle pose); changes: bone ->
    ([d_pitch, d_yaw, d_roll], [dx, dy, dz]). Every bone is sent, so bones
    that are not changed keep the idle pose."""
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
    return ctl


def set_pose(world, walker, base, changes, ticks=4):
    """Re-send the pose on every tick (like CARLA's bone example): a pose sent
    once is overwritten by the next animation update."""
    ctl = build_ctl(base, changes)
    for _ in range(ticks):
        walker.set_bones(ctl)
        walker.blend_pose(1.0)
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


def shape(bw, sk, ground_z, yaw_deg):
    """heights above the road and directions in the walker's own frame (fwd = where it faces)"""
    yaw = math.radians(yaw_deg)
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    pts = np.array([p for n, p in bw.items() if n != sk['root']])
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


class Poser:
    """Works out the bone changes for a lying / sitting / crouching pose by
    trial and error (the crl_ skeleton's bone axes aren't documented).
    Ported from hazard_pose_test.py's Poser; used only for calibration."""

    def __init__(self, world, walker, ground_z, yaw):
        self.world, self.walker = world, walker
        self.ground_z, self.yaw = ground_z, yaw
        self.walker.get_pose_from_animation()
        for _ in range(2):
            world.tick()
        self.base = relative_pose(walker)
        self.sk = skeleton(list(self.base))
        self.tried = 0

    def apply(self, changes):
        self.tried += 1
        return set_pose(self.world, self.walker, self.base, changes)

    def best(self, options, cost):
        scored = []
        for label, ch in options:
            bw = self.apply(ch)
            scored.append((cost(bw), label, ch, bw))
        scored.sort(key=lambda t: t[0])
        return scored[0]

    def limb_options(self, bones_lr, deg, base_changes):
        out = []
        for axis in AXES:
            for sign in (1, -1):
                for mirror in (1, -1):
                    ch = merged(base_changes, {bones_lr[0]: (rot(axis, sign * deg), [0, 0, 0]),
                                               bones_lr[1]: (rot(axis, sign * mirror * deg), [0, 0, 0])})
                    out.append((f'{axis}{"+" if sign > 0 else "-"}{"" if mirror > 0 else " mirrored"}', ch))
        return out

    def drop_to(self, changes, target_low, low_bones):
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
        err, ch2, bw2, how = self.drop_to(ch, 0.10, [n for n in self.base if n != root])
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
        back_down = wanted(self.yaw, -0.5, -0.87)
        ch, label = self.legs(90, 120, back_down, spine_deg=25)
        err, ch, bw, how = self.drop_to(ch, 0.08, [self.sk['foot_l'], self.sk['foot_r']])
        return ch, label + f', body lowered along root {how} until the feet are on the road'


def footpath_next_to(wp, max_steps=5):
    """the footpath (sidewalk lane) to the right of this lane, or None."""
    cur = wp
    for _ in range(max_steps):
        cur = cur.get_right_lane()
        if cur is None:
            return None
        if cur.lane_type == carla.LaneType.Sidewalk:
            return cur
    return None


def ground_height(world, loc, fallback):
    """height of whatever is under loc (road, kerb, footpath); falls back
    to `fallback` if World.ground_projection isn't available."""
    try:
        hit = world.ground_projection(carla.Location(loc.x, loc.y, loc.z + 2.0), 6.0)
        if hit is not None:
            return float(hit.location.z), str(hit.label)
    except (AttributeError, RuntimeError):
        pass
    return float(fallback), 'waypoint height'


def has_bones(walker):
    try:
        return len(world_bones(walker)) > 0
    except RuntimeError:
        return False


def look_at_camera(world, bp, wp, dist=5.0, height=1.5):
    """NEW. A small camera (and the spectator) 5 m along the lane from waypoint wp, looking back at it:
    Unreal only updates a walker's skeleton while something renders it."""
    b = bp.find('sensor.camera.rgb')
    b.set_attribute('image_size_x', '320')
    b.set_attribute('image_size_y', '240')
    t, f = wp.transform.location, wp.transform.get_forward_vector()
    tf = carla.Transform(carla.Location(t.x + dist * f.x, t.y + dist * f.y, t.z + height),
                         carla.Rotation(pitch=-15.0, yaw=wp.transform.rotation.yaw + 180.0))
    cam = world.spawn_actor(b, tf)
    cam.listen(lambda _img: None)
    try:
        world.get_spectator().set_transform(tf)
    except RuntimeError:
        pass
    return cam


def _calibrate_at(world, bp, wbp, wp):
    """One calibration attempt with the walker dropped on waypoint wp.
    -> True (at least one pose passed), False (none passed), None (the walker could not be spawned)."""
    walker, cam, loc = None, None, wp.transform.location
    for dz in (0.8, 1.1, 1.4):
        walker = world.try_spawn_actor(wbp, carla.Transform(carla.Location(loc.x, loc.y, loc.z + dz),
                                                            carla.Rotation(yaw=0.0)))
        if walker is not None:
            break
    if walker is None:
        return None
    try:
        cam = look_at_camera(world, bp, wp)                    # NEW: no camera, no bone updates
        walker.set_collisions(True)
        for _ in range(30):                                    # let it fall and settle
            world.tick()
        for _ in range(60):                                    # ... and wait until it is at rest
            z0 = walker.get_transform().location.z
            world.tick()
            if abs(walker.get_transform().location.z - z0) < 0.005:
                break
        else:
            raise RuntimeError('calibration walker never came to rest (falling through the map?)')
        walker.get_pose_from_animation()
        for _ in range(10):                                    # bones appear once it has been rendered
            world.tick()
            if has_bones(walker):
                break
        if not has_bones(walker):
            raise RuntimeError('get_bones() is empty for the calibration walker, even with a camera on it')
        ground_z = min(p[2] for p in world_bones(walker).values())   # lowest bone of the STANDING walker
        poser = Poser(world, walker, ground_z, 0.0)
        stand = shape(world_bones(walker), poser.sk, ground_z, 0.0)
        print(f"    standing   head {stand['head_z']:.2f} m  hips {stand['hips_z']:.2f} m "
              f"(should be ~1.6 / ~1.0)")
        sh = stand
        for pose in ('lying', 'sitting', 'crouching'):
            changes, how = getattr(poser, pose)()
            sh = shape(poser.apply(changes), poser.sk, ground_z, 0.0)
            ok = pose_ok(pose, sh)
            print(f"    {pose:<10} {'OK ' if ok else 'BAD'} head {sh['head_z']:.2f} m  "
                  f"hips {sh['hips_z']:.2f} m  length {sh['length']:.2f} m  ({how})")
            if ok:
                HAZARD_RECIPES[pose] = dict(changes=changes, how=how)
        if HAZARD_RECIPES:
            HAZARD_SK.update(poser.sk)
            return True
        if abs(sh['head_z'] - stand['head_z']) < 0.05:
            print('  [diag] the pose did not change at all: set_bones / blend_pose had no effect on this walker.')
        return False
    except (Exception, SystemExit) as e:                       # no bone API, unknown skeleton, fell
        print(f'  [warn] hazard pose calibration failed here ({e})')
        HAZARD_RECIPES.clear()
        return False
    finally:
        if cam is not None:
            try:
                cam.stop()
                cam.destroy()
            except RuntimeError:
                pass
        try:
            if walker.is_alive:
                walker.destroy()
            world.tick()
        except RuntimeError:
            pass


def calibrate_hazard_recipes(world, bp, attempts=2):
    """Measure the lying / sitting / crouching bone recipes on one temporary adult walker facing yaw 0, on
    a flat non-junction road. They do not depend on the town: once they exist they are reused.
    UPDATED: a small camera (and the spectator) looks at the calibration walker (Unreal only updates a
    skeleton while something renders it: the pose 'did not change' in towns where nothing looked at that
    spot); a second spot is tried if no pose passes; a town where it failed tries again in the next
    town (HAZARD_CALIBRATION_TRIED is reset per town in collect_town)."""
    global HAZARD_CALIBRATION_TRIED
    if HAZARD_RECIPES or HAZARD_CALIBRATION_TRIED:
        return
    HAZARD_CALIBRATION_TRIED = True
    print('  calibrating hazard poses ...')
    cmap = world.get_map()
    wbp = bp.find('walker.pedestrian.0001')
    if wbp.has_attribute('is_invincible'):
        wbp.set_attribute('is_invincible', 'true')
    sps = cmap.get_spawn_points()
    random.shuffle(sps)
    tried = 0
    for sp in sps:                                             # flat, non-junction road
        wp = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        res = _calibrate_at(world, bp, wbp, wp)
        if res is None:                                        # spot occupied: next spawn point
            continue
        tried += 1
        if res or tried >= attempts:
            break
        print('  trying the calibration again at another spot ...')
    if not HAZARD_RECIPES:
        print('  [warn] no hazard pose passed its check: no hazard pedestrians in this town')


def pose_hazard_batch(world, items):
    """Pose a group of freshly settled walkers with the calibrated recipes (~7-15 ticks for the group).
    Each walker's ground height is the lowest bone of ITS OWN standing pose. The pose is re-sent on
    every tick; walkers that fail the check get 8 more ticks.
    UPDATED: waits up to 6 ticks for get_bones() (empty until a camera has rendered the walker); a walker
    without bones is dropped instead of crashing min() (the Town01 'empty sequence' crash).
    items: dicts with walker / pose / yaw. Sets item['ok'] and item['ground_z']."""
    for it in items:
        try:
            it['walker'].get_pose_from_animation()
        except RuntimeError:
            it['ok'] = False
    for _ in range(2):
        world.tick()
    for _ in range(6):
        if all(it.get('ok') is False or has_bones(it['walker']) for it in items):
            break
        world.tick()
    for it in items:
        if it.get('ok') is False:
            continue
        try:
            bw = world_bones(it['walker'])
            if not bw:
                it['ok'] = False
                continue
            it['ground_z'] = min(p[2] for p in bw.values())
            it['ctl'] = build_ctl(relative_pose(it['walker']), HAZARD_RECIPES[it['pose']]['changes'])
        except RuntimeError:
            it['ok'] = False

    def send(group, ticks):
        for _ in range(ticks):                  # re-send every tick, like CARLA's bone example
            for it in group:
                if it.get('ok') is False:
                    continue
                try:
                    it['walker'].set_bones(it['ctl'])
                    it['walker'].blend_pose(1.0)
                except RuntimeError:
                    it['ok'] = False
            world.tick()

    def check(it):
        if it.get('ok') is False:
            return False
        try:
            return pose_ok(it['pose'], shape(world_bones(it['walker']), HAZARD_SK, it['ground_z'], it['yaw']))
        except (RuntimeError, ValueError, KeyError, IndexError):
            return False

    send(items, 4)
    failed = [it for it in items if not check(it)]
    if failed:
        send(failed, 8)
    for it in items:
        it['ok'] = check(it)





def sidewalk_beside(wp, side, max_steps=5):
    """the footpath (sidewalk lane) on the right (side > 0) or left of this lane, or None"""
    cur = wp
    for _ in range(max_steps):
        cur = cur.get_right_lane() if side > 0 else cur.get_left_lane()
        if cur is None:
            return None
        if cur.lane_type == carla.LaneType.Sidewalk:
            return cur
    return None


def stratified_distances(n, lo, hi):
    """n distances spread over [lo, hi]: the band is cut into n equal parts and one random distance
    is drawn in each (shuffled), so the placements cover the whole band instead of bunching."""
    edges = np.linspace(lo, hi, n + 1)
    d = [random.uniform(edges[i], edges[i + 1]) for i in range(n)]
    random.shuffle(d)
    return d


def near_junction(wp, gap, step=3.0):
    """True if a junction is within gap m ahead of or behind wp along its lane."""
    d = step
    while d <= gap + 1e-6:
        for w in wp.next(d) + wp.previous(d):
            if w.is_junction:
                return True
        d += step
    return False


def ego_route(world, ego, dist=45.0, step=3.0):
    """NEW. [(x, y)] of the lane centres the ego can reach within dist m, every branch at junctions: the
    road ahead whichever way the autopilot turns (client-side map only)."""
    try:
        wp = world.get_map().get_waypoint(ego.get_transform().location)
    except RuntimeError:
        return []
    if wp is None:
        return []
    out, frontier, d = [], [wp], 0.0
    while frontier and d < dist:
        nxt = []
        for w in frontier:
            for n in w.next(step):
                out.append((n.transform.location.x, n.transform.location.y))
                nxt.append(n)
        frontier, d = nxt[:12], d + step
    return out


def hazard_spot(world, ego_tf, place, d, allow_road=True, route=()):
    """-> (location, spawn_z, along_yaw, place) about d metres from the ego ALONG THE ROAD, or None.
    lane: in a driving lane; edge: at the right edge of a lane; footpath: on the footpath beside
    the road (falls back to edge where there is none). With hazard_keep_ego_lane_clear a lane / edge
    spot in the ego's lane or path ahead is rejected: the Traffic Manager would stop the ego for good.
    UPDATED: lane / edge spots also keep hazard_junction_gap m from junctions and 2.8 m from the lanes the
    ego can reach next (route = ego_route()); allow_road=False (road quota full): footpath only."""
    cmap = world.get_map()
    wp0 = cmap.get_waypoint(ego_tf.location)
    if wp0 is None:
        return None
    ahead = random.random() < CFG['hazard_front_fraction']
    cands = wp0.next(d) if ahead else wp0.previous(d)
    if not cands:
        return None
    wp = random.choice(cands)
    if wp.is_junction:
        return None
    if random.random() < 0.4:                                  # sometimes the neighbouring lane
        adj = wp.get_left_lane() if random.random() < 0.5 else wp.get_right_lane()
        if adj is not None and adj.lane_type == carla.LaneType.Driving and not adj.is_junction:
            wp = adj
    kind, base, off = place, wp, 0.0
    if kind == 'footpath':
        side = sidewalk_beside(wp, 1) or sidewalk_beside(wp, -1)
        if side is not None:
            base, off = side, random.uniform(-0.5, 0.5)
        elif not allow_road:
            return None
        else:
            kind = 'edge'
    if kind == 'edge':
        off = wp.lane_width / 2.0 - 0.3 + random.uniform(-0.2, 0.2)
    elif kind == 'lane':
        off = random.uniform(-0.5, 0.5)
    r = base.transform.get_right_vector()
    loc = base.transform.location + carla.Location(x=r.x * off, y=r.y * off)
    if kind != 'footpath':
        if CFG['hazard_junction_gap'] and near_junction(wp, CFG['hazard_junction_gap']):
            return None
        rt = np.array(route, float).reshape(-1, 2)
        if len(rt) and np.min(np.hypot(rt[:, 0] - loc.x, rt[:, 1] - loc.y)) < 2.8:
            return None                                        # on a lane the ego can take next
    if CFG['hazard_keep_ego_lane_clear'] and ahead and kind != 'footpath':
        f = ego_tf.get_forward_vector()
        dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
        same_lane = wp.road_id == wp0.road_id and wp.lane_id == wp0.lane_id
        in_path = dx * f.x + dy * f.y > 0 and abs(-dx * f.y + dy * f.x) < 3.2
        if same_lane or in_path:
            return None
    return loc, base.transform.location.z, wp.transform.rotation.yaw, kind


def hazard_actor_box_2d(walker, K, world_2_camera, img_w, img_h, near=0.1):
    """2D box for a hazard-posed pedestrian, built from its CURRENT bone
    world positions (a walker's collision capsule stays upright whatever the
    pose). Ported from hazard_pose_test.py's body_box_2d().
    Returns (x_min, y_min, x_max, y_max, min_depth), or None.
    UPDATED: None (no crash) when get_bones() is empty or the walker is gone."""
    try:
        bw = world_bones(walker)
    except RuntimeError:
        return None
    pts = np.array([p for n, p in bw.items() if 'root' not in n.lower()])
    if pts.ndim != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        return None
    cam = []
    for p in pts:
        c = np.dot(world_2_camera, np.array([p[0], p[1], p[2], 1.0]))
        cam.append(np.array([c[1], -c[2], c[0]]))
    cam = np.array(cam)
    ok = cam[:, 2] > near
    if not ok.any():
        return None
    f = K[0, 0]
    u = K[0, 2] + f * cam[ok, 0] / cam[ok, 2]
    v = K[1, 2] + f * cam[ok, 1] / cam[ok, 2]
    margin = 0.15 * f / cam[ok, 2]
    x_min, x_max = float((u - margin).min()), float((u + margin).max())
    y_min, y_max = float((v - margin).min()), float((v + margin * 1.5).max())
    x_min, x_max = max(x_min, 0.0), min(x_max, img_w - 1.0)
    y_min, y_max = max(y_min, 0.0), min(y_max, img_h - 1.0)
    return x_min, y_min, x_max, y_max, float(cam[ok, 2].min())


def _in_ego_path(ego_tf, loc, ahead_m, lateral=3.5):
    """True if loc is within ahead_m in front of the ego and within `lateral`
    m of its heading line (roughly: in the lane the autopilot is about to use)."""
    f = ego_tf.get_forward_vector()
    dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
    a = dx * f.x + dy * f.y
    lat = abs(-dx * f.y + dy * f.x)
    return 0.0 < a < ahead_m and lat < lateral


def spawn_hazard_pedestrians(world, bp, n, ego_tf, n_road=0, route=()):
    """Spawn n pedestrians frozen in a lying / sitting / crouching pose on or beside the road,
    each at a random distance from the ego (stratified over CFG['hazard_band'], measured along the
    road). They get NO AI controller (it would overwrite the pose), are invincible, and have
    physics off once posed. A walker whose pose fails its shape check is destroyed.
    Ticks the world (~20 ticks): callers must not save the frame they have in hand.

    UPDATED: n_road = how many existing ones are on lanes / lane edges; at most hazard_max_road in all
    (the rest go on footpaths); route = ego_route(): no lane / edge spot on a lane the ego can take next.

    Returns (walkers, recipe): recipe = list of {actor_id, pose, place, location, ground_z, yaw,
    dist_target} dicts (index-aligned)."""
    if n <= 0:
        return [], []
    calibrate_hazard_recipes(world, bp)
    poses = [p for p in CFG['hazard_poses'] if p in HAZARD_RECIPES]
    if not poses:
        if CFG['hazard_verbose']:
            print('  [warn] no hazard pose is available: no hazard pedestrians spawned')
        return [], []
    pose_weights = [CFG['hazard_poses'][p] for p in poses]
    place_names, place_weights = zip(*CFG['hazard_places'].items())
    lo, hi = CFG['hazard_band']

    walker_bps_all = list(bp.filter('walker.pedestrian.*'))
    pool_ids = CFG.get('hazard_blueprint_pool')
    walker_bps = [b for b in walker_bps_all if b.id in pool_ids] if pool_ids else walker_bps_all
    if not walker_bps:
        print('  [warn] no walker.pedestrian.* blueprints found (or hazard_blueprint_pool '
              'matched none of them) -- no hazard pedestrians spawned')
        print(f'  [hazard-spawn] requested={n} actor_spawned=0 pose_valid=0 registered=0')
        return [], []

    pending = []
    for d_target in stratified_distances(n, lo, hi):
        for _attempt in range(4):                              # a few tries at about this distance
            allow_road = n_road < CFG['hazard_max_road']
            place = random.choices(place_names, weights=place_weights)[0] if allow_road else 'footpath'
            pose = random.choices(poses, weights=pose_weights)[0]
            spot = hazard_spot(world, ego_tf, place, float(np.clip(d_target + random.uniform(-2.0, 2.0), lo, hi)),
                               allow_road, route)
            if spot is None:
                continue
            loc, gz, along, kind = spot
            if pose == 'lying':                                # across the lane / along the kerb
                yaw = along + (random.choice((90.0, -90.0)) if kind == 'lane' else random.choice((0.0, 180.0)))
                yaw += random.uniform(-10.0, 10.0)
            else:
                yaw = random.uniform(-180.0, 180.0)
            walker_bp = random.choice(walker_bps)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'true')
            walker = None
            for dz in (0.6, 1.0, 1.4):
                walker = world.try_spawn_actor(
                    walker_bp, carla.Transform(carla.Location(loc.x, loc.y, gz + dz),
                                               carla.Rotation(yaw=yaw)))
                if walker is not None:
                    break
            if walker is None:
                continue
            pending.append(dict(walker=walker, pose=pose, place=kind, yaw=yaw, d_target=round(d_target, 1)))
            if kind != 'footpath':
                n_road += 1
            break

    if not pending:
        if CFG['hazard_verbose']:
            print(f'  [hazard-spawn] requested={n} actor_spawned=0 pose_valid=0 registered=0')
        return [], []

    for _ in range(12):                      # settle onto the ground, once for the whole group
        world.tick()
    for it in pending:
        try:
            it['walker'].set_collisions(True)
        except RuntimeError:
            pass
    pose_hazard_batch(world, pending)

    spawned, recipe = [], []
    for it in pending:
        w = it['walker']
        if not it.get('ok'):
            try:
                w.destroy()
            except RuntimeError:
                pass
            continue
        try:
            w.set_simulate_physics(False)    # lock the pose in place
            floc = w.get_transform().location
        except RuntimeError:
            continue
        spawned.append(w)
        recipe.append(dict(actor_id=w.id, pose=it['pose'], place=it['place'],
                           location={'x': floc.x, 'y': floc.y, 'z': floc.z},
                           ground_z=it['ground_z'], yaw=it['yaw'], dist_target=it['d_target']))
    if CFG['hazard_verbose']:
        print(f'  [hazard-spawn] requested={n} actor_spawned={len(pending)} '
              f'pose_valid={len(spawned)} registered={len(spawned)}')
    return spawned, recipe


def _register_hazards(hazard_info, hazard_actor_ids, walkers, recipe):
    for w, r in zip(walkers, recipe):
        hazard_info[r['actor_id']] = dict(actor=w, pose=r['pose'], place=r['place'],
                                          x=r['location']['x'], y=r['location']['y'],
                                          z=r['location']['z'], ground_z=r['ground_z'])
        hazard_actor_ids.add(r['actor_id'])


def find_hit_hazards(world, ego, traffic, hazard_info):
    """-> {hazard_id: reason} for hazard pedestrians that traffic ran into or
    over ('hit'), that were pushed away from where they were posed ('moved'),
    or that no longer exist ('gone').

    Uses one world snapshot (a single RPC) and the vehicles' bounding boxes,
    not physics callbacks: these walkers have physics switched off, so
    collision events are not reliable for them. A vehicle counts as having hit
    one when its box comes within CFG['hazard_hit_radius'][pose] of the
    pedestrian's centre."""
    snap = world.get_snapshot()
    rows = []
    for v in [ego] + list(traffic):
        s = snap.find(v.id)
        if s is None:
            continue
        tf = s.get_transform()
        bb = v.bounding_box
        c = tf.transform(bb.location)
        rows.append((c.x, c.y, c.z, math.radians(tf.rotation.yaw), bb.extent.x, bb.extent.y))
    V = np.array(rows, float) if rows else np.zeros((0, 6))

    bad = {}
    for hid, h in hazard_info.items():
        s = snap.find(hid)
        if s is None:
            bad[hid] = 'gone'
            continue
        p = s.get_transform().location
        if math.hypot(p.x - h['x'], p.y - h['y']) > CFG['hazard_shove_m']:
            bad[hid] = 'moved'
            continue
        if len(V):
            dx, dy = h['x'] - V[:, 0], h['y'] - V[:, 1]
            c, s_ = np.cos(V[:, 3]), np.sin(V[:, 3])
            lx, ly = dx * c + dy * s_, -dx * s_ + dy * c
            ex = np.maximum(np.abs(lx) - V[:, 4], 0.0)
            ey = np.maximum(np.abs(ly) - V[:, 5], 0.0)
            hit = (np.hypot(ex, ey) < CFG['hazard_hit_radius'].get(h['pose'], 0.6)) & \
                  (np.abs(V[:, 2] - h['ground_z']) < 3.0)
            if hit.any():
                bad[hid] = 'hit'
    return bad


def maintain_hazards(world, bp, ego, traffic, hazard_info, hazard_actor_ids, actors,
                      hazard_log, ego_tf, stuck, stats, max_new=None):
    """Keep about n_hazard_pedestrians posed pedestrians around the ego:
      remove  - hit / shoved / removed ('hit' / 'moved' / 'gone'),
              - farther than hazard_far_m from the ego ('far'),
              - passed by the ego, more than hazard_behind_m behind it ('behind'),
              - blocking a stuck ego ('blocking'; only when `stuck` is True);
      top up  - to n_hazard_pedestrians with new ones at random distances spread over hazard_band
                (when at least hazard_min_batch are missing: spawning ticks the world).
    Returns how many were removed or added. May tick the world: call it only AFTER the frame of this
    tick has been written. UPDATED: new ones respect hazard_max_road and stay off the lanes the ego can
    take next (ego_route); max_new (default hazard_respawn_max) lets repopulate() fill the whole set."""
    active_before = len(hazard_info)
    el = ego_tf.location
    f = ego_tf.get_forward_vector()
    reasons = dict(find_hit_hazards(world, ego, traffic, hazard_info))
    for hid, h in hazard_info.items():
        if hid in reasons:
            continue
        dx, dy = h['x'] - el.x, h['y'] - el.y
        if math.hypot(dx, dy) > CFG['hazard_far_m']:
            reasons[hid] = 'far'
        elif dx * f.x + dy * f.y < -CFG['hazard_behind_m']:
            reasons[hid] = 'behind'
    if stuck:
        for hid, h in hazard_info.items():
            if hid not in reasons and _in_ego_path(ego_tf, carla.Location(h['x'], h['y'], h['z']), 30.0):
                reasons[hid] = 'blocking'

    ids = list(reasons)[:CFG['recycle_max_per_call']]
    for hid in ids:
        h = hazard_info.pop(hid)
        hazard_actor_ids.discard(hid)
        stats['hazard_' + reasons[hid]] += 1
        hazard_log.append(dict(event='removed', actor_id=hid, reason=reasons[hid], pose=h['pose'],
                               place=h['place'], location={'x': h['x'], 'y': h['y'], 'z': h['z']}))
        try:
            if h['actor'].is_alive:
                h['actor'].destroy()
        except RuntimeError:
            pass
    if not CFG['hazard_respawn']:
        stats['hazard_checks'] += 1
        hazard_log.append(dict(event='maintenance_check', active_before=active_before,
                               removed=len(ids), requested=0, registered=0,
                               active_after=len(hazard_info), target=CFG['n_hazard_pedestrians']))
        return len(ids)

    missing = CFG['n_hazard_pedestrians'] - len(hazard_info)
    added = 0
    requested = 0
    registered = 0
    if missing >= CFG['hazard_min_batch']:
        requested = min(missing, CFG['hazard_respawn_max'] if max_new is None else max_new)
        n_road = sum(1 for h in hazard_info.values() if h['place'] != 'footpath')
        new_w, new_r = spawn_hazard_pedestrians(world, bp, requested, ego_tf, n_road, ego_route(world, ego))
        _register_hazards(hazard_info, hazard_actor_ids, new_w, new_r)
        actors.extend(new_w)
        for r in new_r:
            r['event'] = 'respawned'
            hazard_log.append(r)
        stats['hazard_respawned'] += len(new_w)
        stats['hazard_spawn_requested'] += requested
        stats['hazard_spawn_registered'] += len(new_w)
        stats['hazard_spawn_shortfall'] += requested - len(new_w)
        added = len(new_w)
        registered = len(new_w)
    stats['hazard_checks'] += 1
    hazard_log.append(dict(event='maintenance_check', active_before=active_before,
                           removed=len(ids), missing_before_spawn=missing,
                           requested=requested, registered=registered,
                           active_after=len(hazard_info), target=CFG['n_hazard_pedestrians']))
    if CFG['hazard_verbose'] and (requested or ids):
        print(f'  [hazard-maintenance] active={active_before}->{len(hazard_info)} '
              f'removed={len(ids)} requested={requested} registered={registered}')
    return len(ids) + added


# ---------------------------------------------------------------------------


def blockers_ahead(world, ego, dist, lat):
    """vehicles / walkers in a corridor in front of the ego -> [(actor, metres ahead)], nearest first"""
    tf = ego.get_transform()
    loc, fwd = tf.location, tf.get_forward_vector()
    actors = world.get_actors()
    out = []
    for a in list(actors.filter('vehicle.*')) + list(actors.filter('walker.pedestrian.*')):
        if a.id == ego.id:
            continue
        p = _safe_loc(a)
        if p is None:
            continue
        dx, dy = p.x - loc.x, p.y - loc.y
        ahead, lateral = dx * fwd.x + dy * fwd.y, abs(-dx * fwd.y + dy * fwd.x)
        if 0.0 < ahead < dist and lateral < lat:
            out.append((a, ahead))
    out.sort(key=lambda t: t[1])
    return out


def clear_blocking_hazards(hazard_info, hazard_actor_ids, ego_tf, hazard_log, stats):
    """Unstick step 1: remove the posed pedestrians within 40 m in front of the ego (any place: edge
    hazards block the cars ahead of the ego too)."""
    n = 0
    for hid, h in list(hazard_info.items()):
        if not _in_ego_path(ego_tf, carla.Location(h['x'], h['y'], h['z']), 40.0, 4.0):
            continue
        hazard_info.pop(hid)
        hazard_actor_ids.discard(hid)
        stats['hazard_unstick'] += 1
        hazard_log.append(dict(event='removed', actor_id=hid, reason='unstick', pose=h['pose'],
                               place=h['place'], location={'x': h['x'], 'y': h['y'], 'z': h['z']}))
        try:
            if h['actor'].is_alive:
                h['actor'].destroy()
        except RuntimeError:
            pass
        n += 1
    return n


def remove_walkers_ahead(world, ego, walkers, controllers, hazard_actor_ids, dist=25.0, lat=3.5):
    """Unstick step 2: remove ordinary walkers (and their controllers) standing in the ego's path.
    respawn_walkers tops the population up again."""
    ids = {a.id for a, _ in blockers_ahead(world, ego, dist, lat)
           if a.type_id.startswith('walker.pedestrian') and a.id not in hazard_actor_ids}
    if not ids:
        return 0
    for c in _alive(controllers):
        try:
            if c.parent is not None and c.parent.id in ids:
                c.stop()
                c.destroy()
        except RuntimeError:
            pass
    for w in _alive(walkers):
        if w.id in ids:
            try:
                w.destroy()
            except RuntimeError:
                pass
    walkers[:] = _alive(walkers)
    controllers[:] = _alive(controllers)
    return len(ids)


def configure_ego_autopilot(ego, tm):
    """Autopilot + the ego's own Traffic Manager settings (needed again after it was switched off / on)."""
    ego.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(ego, CFG['ego_speed_diff'])
    if CFG['hazard_keep_ego_lane_clear']:
        tm.auto_lane_change(ego, False)             # stays in the lane that is kept clear of hazards


def teleport_ego(world, ego, tm):
    """Unstick step 3 (and the empty-street move): a free spawn point with a footpath beside it,
    teleport_band (30-200 m) from where the ego is (was: ANY spawn point, far from all the traffic,
    walkers and posed pedestrians), with no vehicle within 15 m and none in the 30 m in front of it.
    The autopilot is switched off during the move so the Traffic Manager forgets the old route.
    Ticks once. -> distance moved (m), or None if no spot was found."""
    old = ego.get_transform().location
    others = []
    for a in world.get_actors().filter('vehicle.*'):
        if a.id != ego.id:
            p = _safe_loc(a)
            if p is not None:
                others.append(p)
    foot = footpath_spawn_points(world)
    sps = world.get_map().get_spawn_points()
    random.shuffle(foot)
    random.shuffle(sps)
    lo, hi = CFG['teleport_band']

    def free(sp, gap):
        if any(sp.location.distance(o) <= gap for o in others):
            return False
        f = sp.get_forward_vector()
        for o in others:                                      # nothing parked in the lane just ahead
            dx, dy = o.x - sp.location.x, o.y - sp.location.y
            if 0.0 < dx * f.x + dy * f.y < 30.0 and abs(-dx * f.y + dy * f.x) < 2.5:
                return False
        return True
    # a street with footpaths 30-200 m away, then one anywhere, then any road 30-200 m away, then anywhere
    cands = [sp for sp in foot if lo <= sp.location.distance(old) <= hi and free(sp, 15.0)]
    if not cands:
        cands = [sp for sp in foot[:80] if free(sp, 12.0)]
    if not cands:
        cands = [sp for sp in sps if lo <= sp.location.distance(old) <= hi and free(sp, 15.0)]
    if not cands:
        cands = [sp for sp in sps[:80] if free(sp, 12.0)]
    if not cands:
        return None
    sp = cands[0]
    if CFG['tm_reset_on_move']:
        ego.set_autopilot(False, tm.get_port())
    ego.set_transform(sp)
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    world.tick()                                              # the move takes effect; TM drops the route
    if CFG['tm_reset_on_move']:
        configure_ego_autopilot(ego, tm)
    return sp.location.distance(old)


def unstick_ego(world, ego, tm, hazard_info, hazard_actor_ids, walkers, controllers, level, hazard_log, stats):
    """REPLACES the old unstick_ego (which destroyed the cars in front, so traffic shrank over the run, or
    moved the ego anywhere). The ego has not moved for unstick_ticks: say what is in front of it, then
    escalate one step: 0 = remove posed pedestrians ahead, 1 = remove walkers ahead, 2 = move the ego.
    -> (the level to start from next time, True if the ego was moved: then call repopulate)."""
    tf = ego.get_transform()
    near = blockers_ahead(world, ego, 30.0, 3.5)
    desc = ', '.join(f"{a.type_id.split('.', 1)[1]}{' (posed)' if a.id in hazard_actor_ids else ''} {d:.0f} m"
                     for a, d in near[:4]) or 'nothing within 30 m'
    print(f'\n  [stuck] ego standing still. Ahead: {desc}.')
    for lvl in range(level, 3):
        if lvl == 0:
            n, what = clear_blocking_hazards(hazard_info, hazard_actor_ids, tf, hazard_log, stats), \
                'removed posed pedestrians ahead'
        elif lvl == 1:
            n, what = remove_walkers_ahead(world, ego, walkers, controllers, hazard_actor_ids), 'removed walkers ahead'
        else:
            dist = teleport_ego(world, ego, tm)
            n, what = int(dist is not None), f'moved the ego {dist or 0:.0f} m to a free spawn point'
        if n:
            print(f'    -> {what}' + (f' ({n})' if lvl < 2 else ''))
            return min(lvl + 1, 2), lvl == 2
    return 2, False


def follow_spectator(world, ego):
    """Server window: look at the ego from behind and above."""
    if not CFG['spectator_follow']:
        return
    try:
        tf = ego.get_transform()
        f = tf.get_forward_vector()
        loc = tf.location + carla.Location(x=-12.0 * f.x, y=-12.0 * f.y, z=8.0)
        world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-25.0, yaw=tf.rotation.yaw)))
    except RuntimeError:
        pass


def census(ego, traffic, walkers, hazard_info, hz_recent=()):
    """One line: road users near the ego and how often a posed pedestrian made it into the labels."""
    loc = ego.get_transform().location
    out = (f'within 40 m: {count_near(traffic, loc, 40.0)} vehicles, {count_near(walkers, loc, 40.0)} walkers '
           f'(of {len(_alive(walkers))}); posed pedestrians: {len(hazard_info)} active')
    if len(hz_recent):
        out += f', in the labels of {100.0 * sum(hz_recent) / len(hz_recent):.0f}% of the last {len(hz_recent)} frames'
    return out


def repopulate(world, bp, tm, ego, traffic, walkers, controllers, actors, hazard_info, hazard_actor_ids,
               hazard_log, stats):
    """NEW. After the ego was moved, bring the scene to it (everything was around the old spot): far
    vehicles to free spawn points vehicle_place_band (20-70 m) from the ego, far walkers replaced 8-50 m
    from it, posed pedestrians out of range replaced (the whole set at once), then settle_ticks without
    saving (moved cars pull away, new walkers land). -> one-line summary"""
    tf = ego.get_transform()
    nv = recycle_vehicles(world, traffic, tf, tm, band=CFG['vehicle_place_band'], far_m=CFG['crowd_radius'],
                          max_n=int(CFG['n_vehicles'] * CFG['crowd_vehicle_fraction']))
    lo, hi = CFG['walker_spawn_band']
    nw = respawn_walkers(world, bp, walkers, controllers, actors, tf.location, band=(lo, hi + 15.0),
                         max_n=int(CFG['n_walkers'] * CFG['crowd_walker_fraction']))
    before = len(hazard_info)
    if CFG['n_hazard_pedestrians']:
        maintain_hazards(world, bp, ego, traffic, hazard_info, hazard_actor_ids, actors, hazard_log,
                         ego.get_transform(), False, stats, max_new=CFG['n_hazard_pedestrians'])
    for _ in range(CFG['settle_ticks']):
        world.tick()
    return (f'scene rebuilt: {nv} vehicles moved, {nw} walkers spawned near the ego, posed pedestrians '
            f'{before} -> {len(hazard_info)}. Now {census(ego, traffic, walkers, hazard_info)}')


# ------------------------------------------------------------------- connection / map / disk
CLIENTS = []             # every client made, kept until the script ends (one per town, see load_town)


def connect():
    c = carla.Client(CFG['host'], CFG['port'])
    c.set_timeout(CFG['timeout'])
    CLIENTS.append(c)          # kept alive: dropping a client whose Traffic Manager is still known can crash
    return c


def nav_matches(world, n=60):
    """-> (ok, share of random navigation points with ground under them). A correct mesh: ~all of them;
    Town04 with Town05's mesh had 52%."""
    ground = tot = 0
    for _ in range(n):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        tot += 1
        try:
            ground += world.ground_projection(carla.Location(loc.x, loc.y, loc.z + 2.0), 6.0) is not None
        except (AttributeError, RuntimeError):
            return True, 1.0                                   # cannot check
    if not tot:
        return False, 0.0
    return ground / tot >= CFG['nav_min_ground'], ground / tot


def load_town(client, town):
    """-> (client to use for this town, world). NEW: the CARLA client loads the walkers' navigation mesh
    only ONCE, for the map the server has when the client first connects, and never again for later
    load_world calls. So `client` only loads the map, then a NEW client connects: its mesh is this map's."""
    client.load_world(town)
    time.sleep(5.0)
    g = 0.0
    for attempt in range(3):
        fresh = connect()
        world = fresh.get_world()
        ok, g = nav_matches(world)
        if ok:
            print(f'  navigation mesh matches {town} ({g:.0%} of its points have ground under them)')
            return fresh, world
        print(f'  navigation mesh does not match {town} ({g:.0%}): connecting again ({attempt + 1}/3) ...')
        time.sleep(3.0)
    raise SystemExit(f'Stopped: the walkers have no navigation mesh for {town} ({g:.0%}). Restart CARLA and '
                     f'run this town alone (--towns {town}).')


class DiskFull(Exception):
    pass


def free_gb(path):
    return shutil.disk_usage(os.path.abspath(path)).free / 1e9


def save_progress(path, progress):
    tmp = path + '.tmp'                                       # write, then rename: never half a file
    with open(tmp, 'w') as fh:
        json.dump(progress, fh, indent=2)
    os.replace(tmp, path)
# ---------------------------------------------------------------------------


def write_metadata(path, frame_id, town, run_id, ego,
                    n_yolo, n_lanes, sync_status,
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
        'weather': CFG.get('weather_name'),
        'camera_resolution': [CFG['img_w'], CFG['img_h']],
        'camera_fov': CFG['fov'],
        'num_yolo_objects': n_yolo,
        'num_lanes': n_lanes,
        'sync_status': sync_status,
        'seed': CFG.get('seed'),
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


# Directory structure (flat, unchanged):
#   image_2/ semantic/ semantic_visual/ depth/ depth_raw/ labels_yolo/ lanes/ metadata/
#   hazard_pedestrians.json  every hazard spawn / removal / respawn with its reason
def get_run_root(town):
    return f"{CFG['out']}/{town}"


def ensure_dataset_dirs(root):
    subdirs = [
        'image_2',
        'semantic', 'semantic_visual',
        'depth', 'depth_raw',
        'labels_yolo',
        'lanes', 'metadata',
    ]
    for d in subdirs:
        os.makedirs(f'{root}/{d}', exist_ok=True)


def validate_dataset(root, town, sample_limit=None):
    """Walk the saved frames for one town/run and sanity-check every
    modality. Prints the summary table. Returns the summary dict."""
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
    lane_totals = []
    bad_frames = []

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

        lane_path = f'{root}/lanes/{fid}.json'
        if os.path.isfile(lane_path):
            try:
                with open(lane_path) as f:
                    lanes = json.load(f)
                valid = isinstance(lanes, list)
                if valid:
                    for lane in lanes:
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
    for label, key in [('RGB', 'rgb'), ('Semantic', 'semantic'),
                        ('Depth', 'depth'), ('YOLO', 'yolo'),
                        ('Lane GT', 'lanes'), ('Metadata', 'metadata')]:
        print(f'{label + ":":<12}{counts.get(key, 0)}/{n}')

    print('\nYOLO instances by class:',
          dict(yolo_class_counts) if yolo_class_counts else '(none)')
    print(f'frames with a validation issue: {len(bad_frames)}'
          + (f' (e.g. {bad_frames[:5]})' if bad_frames else ''))
    if lane_totals:
        print(f'average lanes/frame: {np.mean(lane_totals):.2f}')

    return {
        'n_frames': n, 'counts': dict(counts),
        'yolo_class_counts': dict(yolo_class_counts),
        'bad_frames': bad_frames,
    }


def write_frame(root, base, data, world, ego, sensors, frame_id, town, hazard_actor_ids):
    """Save one frame (RGB, semantic, depth, YOLO labels, lanes, metadata). -> the YOLO label tuples."""
    data['rgb'].save_to_disk(f'{root}/image_2/{base}.png')
    data['sem'].save_to_disk(f'{root}/semantic/{base}.png')
    if CFG.get('save_semantic_visual'):
        data['sem'].save_to_disk(f'{root}/semantic_visual/{base}.png', carla.ColorConverter.CityScapesPalette)
    depth_array = decode_depth(data['depth'])
    np.save(f'{root}/depth/{base}.npy',
            depth_array.astype(np.float16) if CFG.get('save_depth_fp16') else depth_array)
    if CFG.get('save_depth_raw_png'):
        data['depth'].save_to_disk(f'{root}/depth_raw/{base}.png', carla.ColorConverter.Depth)

    yolo_pairs = extract_2d_bboxes_yolo(
        world, sensors['rgb'], ego, CFG['img_w'], CFG['img_h'], CFG['fov'],
        max_distance=CFG['yolo_max_distance'], depth_array=depth_array,
        visibility_check=CFG['visibility_check'], occlusion_margin=CFG['occlusion_margin'],
        rgb_transform=data['rgb'].transform, hazard_actor_ids=hazard_actor_ids)
    with open(f'{root}/labels_yolo/{base}.txt', 'w') as f:
        f.write('\n'.join(p[1] for p in yolo_pairs))

    lanes = extract_lane_gt(world, ego, sensors['rgb'], CFG['img_w'], CFG['img_h'], CFG['fov'],
                            max_distance=CFG['lane_max_distance'], sample_distance=CFG['lane_sample_distance'],
                            rgb_transform=data['rgb'].transform)
    with open(f'{root}/lanes/{base}.json', 'w') as f:
        json.dump(lanes, f)

    sensor_frame_ids = {name: item.frame for name, item in data.items()}
    write_metadata(f'{root}/metadata/{base}.json', frame_id, town, CFG['run_id'], ego,
                   n_yolo=len(yolo_pairs), n_lanes=len(lanes),
                   sync_status='ok' if len(set(sensor_frame_ids.values())) == 1 else 'mismatch',
                   sensor_frame_ids=sensor_frame_ids)
    return yolo_pairs


def collect_town(client, town, start_idx, n_frames, ckpt=None, continuing=False):
    """Record n_frames frames of `town`, ids from start_idx. UPDATED: own client per town (navigation
    mesh), cameras before the posed pedestrians, frames written BEFORE the upkeep jobs of a tick, and the
    scene kept around the ego until the last frame (see the UPDATED notes at the top). ckpt(next_idx) is
    called every 50 saved frames (progress.json); continuing=True appends to this town's hazard log."""
    global HAZARD_CALIBRATION_TRIED
    print(f'\n=== {town}: {n_frames} frames, ids from {start_idx:06d} ===')
    client, world = load_town(client, town)
    world.set_weather(CFG['weather'])
    bp = world.get_blueprint_library()
    tm = client.get_trafficmanager(CFG['tm_port'])
    tm.set_random_device_seed(CFG['seed'])
    HAZARD_CALIBRATION_TRIED = False                  # a town where calibration failed tries again

    original = world.get_settings()
    sensors, queues, actors, controllers = {}, {}, [], []
    class_counts, crowd = Counter(), Counter()
    root = get_run_root(town)
    ensure_dataset_dirs(root)
    log_path = f'{root}/hazard_pedestrians.json'
    hazard_log = json.load(open(log_path)) if continuing and os.path.isfile(log_path) else []

    def write_hazard_log():
        if hazard_log:
            with open(log_path, 'w') as f:
                json.dump(hazard_log, f, indent=2)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)
        tm.global_percentage_speed_difference(CFG['tm_speed_diff'])
        tm.set_global_distance_to_leading_vehicle(CFG['tm_distance_to_leading'])
        world.tick()

        if CFG['n_hazard_pedestrians']:
            calibrate_hazard_recipes(world, bp)       # needs its own camera; cached for later towns

        # ego in a busy street with footpaths; its autopilot goes on right before the warm-up
        ego_bp = bp.filter('vehicle.tesla.model3')[0]
        if ego_bp.has_attribute('role_name'):
            ego_bp.set_attribute('role_name', 'hero')
        ego = None
        for _ in range(20):
            ego = world.try_spawn_actor(ego_bp, pick_ego_spawn(world))
            if ego is not None:
                break
        if ego is None:
            raise RuntimeError('could not spawn the ego vehicle')
        actors.append(ego)
        world.tick()

        # UPDATED: cameras BEFORE any pedestrian is posed (a skeleton only updates while rendered)
        sensors, queues = setup_sensors(world, ego, bp)
        actors += list(sensors.values())
        follow_spectator(world, ego)

        traffic = spawn_traffic(world, bp, tm, ego_location=ego.get_transform().location)
        actors += traffic
        walkers, controllers = spawn_walkers(world, bp, CFG['n_walkers'],
                                             ego_location=ego.get_transform().location)
        actors += walkers
        actors += controllers

        # Hazard pedestrians: kept out of `walkers`/`controllers` (no AI controller)
        hazard_actor_ids = set()        # same set object is passed to the 2D-label code for the whole town
        hazard_info = {}                # actor id -> dict(actor, pose, place, x, y, z, ground_z)
        hz_w, hz_r = spawn_hazard_pedestrians(world, bp, CFG['n_hazard_pedestrians'], ego.get_transform(),
                                              0, ego_route(world, ego))
        _register_hazards(hazard_info, hazard_actor_ids, hz_w, hz_r)
        actors += hz_w
        hazard_log += [dict(r, event='spawned') for r in hz_r]
        print(f'  spawned {len(hz_w)}/{CFG["n_hazard_pedestrians"]} hazard pedestrians '
              f'({", ".join(f"{k}={v}" for k, v in Counter(r["pose"] for r in hz_r).items()) or "none"})')

        configure_ego_autopilot(ego, tm)
        for _ in range(60):                           # let traffic settle
            world.tick()
        write_hazard_log()
        drain_queues(queues)
        print(f'  [scene] start: {census(ego, traffic, walkers, hazard_info)}')

        buffers, dropped_frames = {}, 0
        saved, tick = start_idx, 0
        still = moving = static_saved = level = empty = 0
        hz_recent = deque(maxlen=100)                 # did each of the last saved frames have a posed label?
        max_ticks = n_frames * CFG['tick_skip'] * CFG['max_ticks_factor']
        while saved < start_idx + n_frames:
            if tick >= max_ticks:
                print(f'\n  [warn] stopping early: {max_ticks} ticks without enough moving frames (ego stuck?)')
                break
            frame_id = world.tick()
            tick += 1
            try:
                data = grab_synced(queues, frame_id, buffers)
            except FrameDropped as e:
                dropped_frames += 1
                print(f'\n  [warn] {e} -- skipping this frame ({dropped_frames} dropped so far)')
                data = None
            vel = ego.get_velocity()
            speed = math.hypot(vel.x, vel.y)

            # 1. the frame of THIS tick, before anything below ticks the world
            if data is not None and tick % CFG['tick_skip'] == 0:
                static_saved = static_saved + 1 if speed < 0.5 else 0      # no dataset of a standing ego
                if static_saved <= CFG['max_static_frames']:
                    yolo_pairs = write_frame(root, f'{saved:06d}', data, world, ego, sensors, frame_id, town,
                                             hazard_actor_ids)
                    has_hz = False
                    for cls_name, _, is_hz, dmin in yolo_pairs:
                        class_counts[cls_name] += 1
                        if is_hz:                           # hazard labels per distance band
                            b = int(min(dmin, CFG['yolo_max_distance'] - 0.1) // 15)
                            crowd[f'hz_labels_{15 * b}_{15 * b + 15}m'] += 1
                            crowd['hz_labels'] += 1
                            has_hz = True
                    crowd['hz_frames'] += has_hz
                    hz_recent.append(has_hz)
                    crowd['frames'] += 1
                    crowd['yolo_labels'] += len(yolo_pairs)
                    saved += 1
                    k = saved - start_idx
                    print(f'  {town}: {k}/{n_frames}', end='\n' if k % 50 == 0 else '\r', flush=True)
                    if CFG['census_every'] and k % CFG['census_every'] == 0:
                        print(f'  [scene] frame {k}: {census(ego, traffic, walkers, hazard_info, hz_recent)}')
                    if k % 50 == 0:
                        write_hazard_log()
                        if ckpt:
                            ckpt(saved)                     # a crash now loses at most 50 frames
                        if free_gb(root) < CFG['min_free_gb']:
                            raise DiskFull(f'{free_gb(root):.1f} GB free')

            # 2. upkeep AFTER the frame is written (these jobs tick the world)
            if tick % 5 == 0:
                follow_spectator(world, ego)
            still = still + 1 if speed < 0.3 else 0
            moving = moving + 1 if speed > 2.0 else 0
            if moving >= 20:
                level = 0                             # driving again: the escalation starts from step 1
            moved = False
            if CFG['unstick_ticks'] and still >= CFG['unstick_ticks']:
                if ego.is_at_traffic_light() and ego.get_traffic_light_state() == carla.TrafficLightState.Red:
                    still = 0                         # a red light is not a jam
                else:
                    level, moved = unstick_ego(world, ego, tm, hazard_info, hazard_actor_ids, walkers,
                                               controllers, level, hazard_log, crowd)
                    still = 0
            if CFG['empty_move_ticks'] and tick % 20 == 0 and not moved:      # no pedestrians around
                empty = empty + 20 if count_near(walkers, ego.get_transform().location, 40.0) == 0 else 0
                if empty >= CFG['empty_move_ticks']:
                    dist = teleport_ego(world, ego, tm)
                    print(f'\n  [empty] no walker within 40 m for {empty * 0.05:.0f} s (road without footpaths?): '
                          f'ego moved {dist or 0:.0f} m to a street with footpaths')
                    moved, empty = dist is not None, 0
            if CFG['teleport_test'] and tick % CFG['teleport_test'] == 0 and not moved:
                dist = teleport_ego(world, ego, tm)
                print(f'\n  [teleport-test] tick {tick}: ego moved {dist or 0:.0f} m')
                moved = dist is not None
            if moved:                                 # everything was around the old spot
                print(f'    -> {repopulate(world, bp, tm, ego, traffic, walkers, controllers, actors, hazard_info, hazard_actor_ids, hazard_log, crowd)}')
                crowd['rebuilds'] += 1
                still = static_saved = moving = 0
                follow_spectator(world, ego)
                continue
            etf = ego.get_transform()
            if controllers and CFG['walker_redispatch_ticks'] and tick % CFG['walker_redispatch_ticks'] == 0:
                redispatch_walkers(world, controllers, etf.location)
            # hazard pedestrians hit by traffic / shoved / gone / too far / passed / blocking a stuck ego
            # -> replaced ahead of the ego
            if CFG['hazard_check_ticks'] and tick % CFG['hazard_check_ticks'] == 0 and HAZARD_RECIPES:
                traffic[:] = _alive(traffic)
                stuck = still >= CFG['stuck_ticks']
                if maintain_hazards(world, bp, ego, traffic, hazard_info, hazard_actor_ids, actors,
                                    hazard_log, etf, stuck, crowd) and stuck:
                    still = 0
            if CFG['recycle_ticks'] and tick % CFG['recycle_ticks'] == 0:
                n_near = count_near(traffic, etf.location, CFG['crowd_radius'])
                if n_near < CFG['min_vehicles_near']:  # too few around the ego -> bring far ones in range
                    crowd['recycled_vehicles'] += recycle_vehicles(
                        world, traffic, etf, tm, band=CFG['vehicle_place_band'], far_m=CFG['crowd_radius'],
                        max_n=CFG['min_vehicles_near'] - n_near)
                else:
                    crowd['recycled_vehicles'] += recycle_vehicles(world, traffic, etf, tm)
            if CFG['walker_respawn_ticks'] and tick % CFG['walker_respawn_ticks'] == 0:
                crowd['respawned_walkers'] += respawn_walkers(world, bp, walkers, controllers, actors, etf.location)

        n = saved - start_idx
        print(f'\n{town} done: {n} frames ({dropped_frames} dropped/skipped, {crowd["rebuilds"]} scene rebuilds '
              f'after moving the ego, {crowd["respawned_walkers"]} walkers respawned near the ego)')
        count_summary = ', '.join(f'{name}={count}' for name, count in
                                  sorted(class_counts.items(), key=lambda kv: -kv[1]))
        print(f'{town} GT instance counts (2D labels): ' + (count_summary or '(none)'))
        write_hazard_log()
        if hazard_log:
            spawned_events = [r for r in hazard_log if r.get('event') in ('spawned', 'respawned')]
            pose_counts = Counter(r['pose'] for r in spawned_events)
            print(f'{town} hazard pedestrians: {len(spawned_events)} spawned in total '
                  f'({", ".join(f"{k}={v}" for k, v in pose_counts.items())}); '
                  f'replaced: hit={crowd["hazard_hit"]}, moved={crowd["hazard_moved"]}, '
                  f'gone={crowd["hazard_gone"]}, far={crowd["hazard_far"]}, '
                  f'behind={crowd["hazard_behind"]}, blocking={crowd["hazard_blocking"]}, '
                  f'unstick={crowd["hazard_unstick"]}; respawned={crowd["hazard_respawned"]}')
            print(f'{town} hazard maintenance: checks={crowd["hazard_checks"]}, '
                  f'respawn requested={crowd["hazard_spawn_requested"]}, '
                  f'registered={crowd["hazard_spawn_registered"]}, '
                  f'shortfall={crowd["hazard_spawn_shortfall"]}')
        if crowd['frames']:
            nf = crowd['frames']
            print(f'{town} avg per frame -- 2D labels kept: {crowd["yolo_labels"] / nf:.1f}')
            hz_bands = [(k, v) for k, v in crowd.items() if k.startswith('hz_labels_')]   # FIX: sort these only
            bands = ', '.join(f'{k[10:]} {v}' for k, v in sorted(hz_bands, key=lambda kv: int(kv[0].split('_')[2])))
            print(f'{town} posed-pedestrian labels: {crowd["hz_labels"]} in {crowd["hz_frames"]}/{nf} frames '
                  f'({100.0 * crowd["hz_frames"] / nf:.0f}%); by distance: {bands or "(none)"}')
            print(f'{town} recycled {crowd["recycled_vehicles"]} vehicle teleports')

        if CFG.get('validate_after_run'):
            validate_dataset(root, town)
        return saved

    finally:
        # UPDATED: world and Traffic Manager back to asynchronous FIRST, then destroy without waiting for a
        # tick (apply_batch_sync(..., True) timed out after 60 s in the LiDAR runs). Errors are printed.
        print()
        write_hazard_log()
        for s in sensors.values():
            try:
                if s.is_alive:
                    s.stop()
            except Exception:
                pass
        for c in controllers:
            try:
                if c.is_alive:
                    c.stop()
            except Exception:
                pass
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original)
        except RuntimeError as e:
            print(f'  [warn] could not switch CARLA back to asynchronous mode ({e})')
        try:
            alive = _alive(actors)
            if alive:
                client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in alive])
            time.sleep(1.0)
        except RuntimeError as e:
            print(f'  [warn] cleanup failed ({e}). The frames of this town are saved. If the CARLA window is '
                  f'frozen or closed, restart CARLA before the next town.')
        print(f'{town} cleaned up')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fresh', action='store_true', help='ignore progress.json and start at frame 0')
    ap.add_argument('--towns', nargs='*', help='only these towns (default: CFG towns)')
    ap.add_argument('--frames', type=int, help='frames per town (overrides frames_per_town)')
    ap.add_argument('--no-hazards', action='store_true', help='no lying / sitting / crouching pedestrians')
    ap.add_argument('--teleport-test', type=int, default=0, metavar='TICKS',
                    help='move the ego every TICKS ticks, to test the scene rebuild (e.g. 400)')
    args = ap.parse_args()
    if args.no_hazards:
        CFG['n_hazard_pedestrians'] = 0
    CFG['teleport_test'] = args.teleport_test
    random.seed(CFG['seed'])
    np.random.seed(CFG['seed'])

    client = connect()                    # only loads maps; each town gets its own client (load_town)
    check_semantic_table()
    os.makedirs(CFG['out'], exist_ok=True)
    progress_path = f"{CFG['out']}/progress.json"             # UPDATED: in the output folder
    progress = {'next_idx': 0, 'done_towns': [], 'ranges': {}}
    if CFG['resume'] and not args.fresh and os.path.isfile(progress_path):
        progress.update(json.load(open(progress_path)))
        print(f"resuming: done towns {progress['done_towns']}, next idx {progress['next_idx']}")

    idx = progress['next_idx']
    for town in CFG['towns']:
        if args.towns and town not in args.towns:
            continue
        if town in progress['done_towns']:
            print(f'skipping {town} (already done)')
            continue
        n_town = args.frames or CFG['frames_per_town']
        part = progress.get('partial')
        if part and part['town'] == town and part['first'] == idx:   # continue a town that stopped
            first, start, continuing = part['first'], part['next'], True
            print(f'\ncontinuing {town}: {start - first} of {n_town} frames were saved before it stopped')
        else:
            if part:
                print(f"\n[note] {part['town']} had stopped half way; it starts again from its first frame")
                progress.pop('partial')
            first = start = idx
            continuing = False
        progress['partial'] = {'town': town, 'first': first, 'next': start}
        save_progress(progress_path, progress)

        def ckpt(next_idx, town=town, first=first):
            progress['partial'] = {'town': town, 'first': first, 'next': next_idx}
            save_progress(progress_path, progress)
        todo = n_town - (start - first)
        if free_gb(CFG['out']) < CFG['min_free_gb'] + 1.0:
            print(f'\nSTOPPED before {town}: only {free_gb(CFG["out"]):.1f} GB free on this disk. '
                  f'Free some space and run the same command again.')
            return
        try:
            idx = collect_town(client, town, start, todo, ckpt, continuing) if todo > 0 else start
        except (DiskFull, OSError) as e:
            if not isinstance(e, DiskFull) and free_gb(CFG['out']) >= CFG['min_free_gb']:
                raise                                          # an OSError that is not a full disk
            print(f'\nSTOPPED in {town}: the disk is (almost) full ({free_gb(CFG["out"]):.1f} GB free). Progress '
                  f'is saved: free some space and run the same command again, it continues this town.')
            return
        progress.pop('partial', None)
        progress['next_idx'] = idx
        progress['done_towns'].append(town)
        progress['ranges'][town] = [first, idx - 1]
        save_progress(progress_path, progress)

        splits = {t: {'split': CFG['split_config'].get(t, 'train'),
                      'first_frame': r[0], 'last_frame': r[1]}
                  for t, r in progress['ranges'].items()}
        json.dump(splits, open(f"{CFG['out']}/splits.json", 'w'), indent=2)

    print(f'\ntotal frames: {idx}')


if __name__ == '__main__':
    main()