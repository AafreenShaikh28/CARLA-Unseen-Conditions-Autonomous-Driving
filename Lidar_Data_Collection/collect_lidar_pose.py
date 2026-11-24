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
      hazards/000000.txt    one line per posed (lying / sitting / crouching) pedestrian in range:
                            actor_id pose place n_body_pts n_ghost_pts labelled x y z l w h yaw
      metadata/000000.json  town, weather, ego pose / speed / control, light state
      split_lidar.json      town-based train/val ids for lidar_train.py
  kitti_carla/progress.json  resume bookkeeping
  kitti_carla/hazard_poses.json  the bone changes that made each pose (from calibration)

Label rules (one place to change them: make_labels)
  - vehicles and pedestrians within label_max_dist of the LiDAR (horizontal)
  - label_2 real label: projects into the image AND >= min_points hits from the semantic
    LiDAR (CARLA's own per-actor hit ids). occluded 0/1/2 from the depth camera,
    truncated = share of the projected box outside the image.
  - label_2 DontCare: in the image but < min_points hits (fully hidden from the LiDAR).
    OpenPCDet's KITTI eval ignores detections there instead of calling them false positives.
  - labels_3d: every actor in range, all around the car, with both hit counts.
  - parked cars baked into the map (no actor, so no label) are switched off.

Posed pedestrians (lying / sitting / crouching; ported from hazard_pose_test.py)
  CARLA has no such animations, so a few walkers per frame are posed through the bone API
  (Walker.set_bones / blend_pose). The bone recipes are found at the start of EVERY run by measuring
  the skeleton (calibrate_poses), then applied to walkers placed on the footpath, at the lane edge or in
  the lane near the ego (spawn_hazards / refresh_hazards). They have no AI controller (they stay put).
  CARLA keeps the walker's bounding box UPRIGHT whatever the pose, so for these walkers the 3D box is
  rebuilt from the bone positions (posed_box_world) and the LiDAR hits are counted inside THAT box.
  Hits that land in the old upright capsule but not on the body are counted as 'ghost' points.
  If the LiDAR only ever hits the capsule, posed walkers get no hits on the body, so they come out as
  DontCare / labelled=0 and the run prints a [WARN] instead of silently training on wrong boxes.

Earlier fixes (kept):
  1. Town04: calibration walker fell for ever -> it is spawned on a flat non-junction driving lane,
     waits until at rest, ground = lowest bone of the STANDING walker, 'standing' line printed.
  2. Town01: ground height of each posed walker comes from its own lowest bone after it settled;
     the pose is re-sent on every tick; failures get 8 more ticks.
  3. Navigation points are only used if there is ground under them (ground_below).
  4. Hazards are not placed in the ego's lane / path ahead (hazard_keep_ego_lane_clear).
  5. load_town() checks that the navigation mesh matches the map and reloads it.

UPDATED (this version), from the second --quick run (Town04: 10 frames in 800 ticks, Town01: crash):
  6. Ego standing still (Town04). Removing posed pedestrians was not enough: something else was in
     the way (the log never said what). Now, after unstick_ticks (10 s) without moving, the script
     PRINTS what is ahead (vehicle / walker / posed walker + distance, traffic light state) and
     escalates one step per stuck period: (1) remove posed pedestrians ahead, (2) remove ordinary
     walkers ahead (and their controllers), (3) move the ego to a free spawn point. The step resets
     once the ego drives again. A red light is not treated as a jam.
     Optional: CFG['ego_ignore_walkers_pct'] > 0 makes the ego's Traffic Manager ignore pedestrians.
  7. Town01 crash 'min() arg is an empty sequence': get_bones() can be empty for a freshly spawned
     walker. world_bones() is now checked everywhere (wait up to 6 ticks, then drop that walker).
  8. Town01: 0/30 posed pedestrians, the measured shape was the STANDING one, i.e. the bone pose did
     not take effect in that world. The recipes are now re-measured at the start of every run (so a
     run that cannot pose pedestrians says so right at the start), and after hazard_fail_limit
     refreshes in a row with 0 kept, posed pedestrians are switched off for that run instead of
     wasting ~26 ticks every 5 s.
  9. 'ERROR: failed to destroy actor ... not found': actors were destroyed twice. All destroys go
     through destroy_actor(), which checks is_alive first.
 10. load_town(): if two reloads do not fix the navigation mesh, load another town first and then the
     wanted one (third attempt).

UPDATED (scene rebuild), from the run where frames went empty after the ego was moved and Town01 had no
posed pedestrians:
 11. Unstick step 3 moves the ego to a free spawn point 30-200 m away (was: anywhere in the town, far from
     everything spawned around the old spot), then repopulate() rebuilds the scene there: far vehicles to
     free spawn points 20-70 m away, far walkers replaced 8-50 m away, posed pedestrians re-spawned,
     2 s settle without saving. Moved vehicles get their autopilot switched off / on (new TM route).
 12. Every recycle_ticks: fewer than min_vehicles_near vehicles within near_radius -> far ones moved in.
 13. Posed pedestrians were switched off for the whole run after 3 refreshes with 0 kept, which also
     happened while the ego was stuck and no free spot existed. Now only if the pose never worked.
 14. Unreal updates a walker's skeleton only while a camera renders it (empty get_bones(), pose 'did
     not change' in Town01): calibration has its own camera, 3 small cameras render left / back / right
     of the ego (keepalive_cams), and the sensors are spawned before the road users.
 15. Lane / edge hazards: at most hazard_max_road, >= hazard_junction_gap from junctions, off the Traffic
     Manager's route for the ego (fewer jams, so fewer teleports).
 16. [scene] census line every census_every frames, spectator follows the ego, --frames N and
     --teleport-test TICKS for testing.
 17. (after the first real test) Town04 got Town05's navigation mesh (0 walkers near the ego) and the
     check passed it: >= 85% of the points now need ground. (Loading twice did not help: see 21.)
 18. The test stopped silently (a crash inside the CARLA client library): faulthandler now writes the
     Python stack to collector_crash_trace.txt. TrafficManager.get_all_actions() is no longer called
     (ego_route uses the client-side map); tm_reset_on_move switches off the autopilot toggling.
 19. Progress: a full line every 50 frames, so it is not overwritten when the script ends.
 20. (second test) Town04: the ego drove onto the highway loop (no footpaths, 0 walkers within 40 m, no
     walker could be respawned). The ego now starts and is moved only to spawn points with a footpath
     beside them, and after empty_move_ticks (15 s) with no walker within 40 m it is moved back to such a
     street and the scene is rebuilt there.
 21. Navigation mesh, found in the third test: Town01.bin IS in CARLA's Nav folder. The CARLA client loads
     the walkers' mesh once, for the map the server had when the script connected (Town05 -> Town04 run,
     Town01 -> Town04 run, Town04 -> also used for Town01), and reloads do not change it. load_town() now
     loads the map with the main client and then connects a NEW client for the run (connect()).
 22. End of run: back to asynchronous mode first, then destroy without a tick (the tick after the destroy
     timed out after 60 s and stopped the script). Cleanup errors are printed, not raised.
"""
import os, json, math, queue, random, time, argparse, faulthandler
from collections import Counter, OrderedDict
import numpy as np
import carla
import kitti_utils as K

# A crash inside the CARLA client library ends Python with no traceback (the last run just stopped).
# faulthandler writes the Python stack of such a crash to this file. On Windows it also logs harmless
# C++ exceptions that CARLA catches itself (code 0xe06d7363): only the LAST entry matters.
CRASH_LOG = open('collector_crash_trace.txt', 'w')
faulthandler.enable(file=CRASH_LOG, all_threads=True)

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
    # Unstick the ego (the Traffic Manager stops for any person / vehicle in its path):
    unstick_ticks=200,             # ego standing still this long (10 s, not at a red light) -> escalate
    ego_ignore_walkers_pct=0.0,    # 0-100: % of pedestrians the ego's autopilot ignores (0 = stops for all).
                                   # >0 stops the jams for good but the ego may hit a walker.
    # NEW: keep the scene populated around the ego (frames were empty after the ego was moved)
    teleport_band=(30.0, 200.0),   # unstick step 3: new ego spot this far (m) from where it was stuck, not
                                   # anywhere in the town (all traffic was spawned around the old spot)
    min_vehicles_near=15,          # every recycle_ticks: fewer vehicles than this within near_radius ->
    vehicle_place_band=(20.0, 70.0),   # far ones are moved to free spawn points this far from the ego
    settle_ticks=40,               # 2 s without saving after the scene is rebuilt (cars start, walkers land)
    census_every=100,              # print what is around the ego every N saved frames (0 = off)
    spectator_follow=True,         # the server window follows the ego (and renders the area around it)
    teleport_test=0,               # set by --teleport-test: move the ego every N ticks to test the rebuild
    tm_reset_on_move=True,         # switch the autopilot off / on around a teleport (new route); False if
                                   # collector_crash_trace.txt ever points at set_autopilot
    empty_move_ticks=300,          # no walker within 40 m of the ego for this long (15 s; e.g. the Town04
                                   # highway: no footpaths, so no walker can be placed) -> move the ego back
                                   # to a street with footpaths and rebuild the scene there (0 = off)
    nav_min_ground=0.85,           # share of navigation points that must have ground under them (a correct
                                   # mesh: ~all; Town04 with Town05's mesh: 52%)
    # NEW: Unreal only updates a walker's skeleton while some camera renders it. A posed walker nothing
    # looks at keeps its standing bones (pose check fails; get_bones() is even empty for a new walker).
    # These 3 small cameras (left / back / right of the ego) exist only to render; images are thrown away.
    keepalive_cams=True,
    keepalive_res=(320, 240),
    # CARLA 0.9.15 bug (issue #7092): controller.ai.walker switches off the walker's collision,
    # and ray-cast sensors skip collision-free actors -> pedestrians get ZERO LiDAR points.
    # Workaround from that thread: turn collisions (and physics) back on after start().
    walker_lidar_fix=True,
    walker_physics=True,     # set False if walkers stop walking / jitter with physics on
    disable_map_vehicles=True,
    nav_ground_check=True,   # only use navigation points that have ground under them (see ground_below)

    # posed pedestrians: lying / sitting / crouching on the road (Indian road hazards)
    n_hazards=30,            # posed pedestrians kept near the ego at any time (0 = feature off)
    hazard_poses=dict(lying=0.40, sitting=0.30, crouching=0.30),    # sampling weights
    hazard_places=dict(footpath=0.40, edge=0.25, lane=0.35),        # where they are put (weights)
    hazard_band=(10.0, 45.0),      # m from the ego (along the road) for new ones
    hazard_front_fraction=0.7,     # share placed ahead of the ego, the rest behind it
    hazard_far_m=60.0,             # farther than this from the ego -> replaced (= label_max_dist)
    hazard_respawn_ticks=100,      # every 5 s, together with the walker respawn (0 = never replace)
    hazard_reapply_ticks=10,       # re-send the bone pose; set 1 if poses flicker back to standing
    hazard_fail_limit=3,           # this many refreshes in a row with 0 kept -> no posed walkers for this run
    hazard_keep_ego_lane_clear=True,  # never put a lane / edge hazard in the ego's lane ahead of it: the
                                   # Traffic Manager stops for a person in its path, so the ego never moves on
    hazard_max_road=8,             # NEW: at most this many on lanes / lane edges at a time: each one stops the
                                   # Traffic Manager cars behind it for good -> jams -> stuck ego -> teleport
    hazard_junction_gap=15.0,      # NEW: m; no lane / edge hazard this close to a junction (cars stopped
                                   # inside a junction block every direction)
    hazard_adults_only=True,       # the pose recipes are measured on an adult skeleton
    hazard_box_margin=0.12,        # m added around the bone centres to get the body box
    hazard_bottom_skip=0.08,       # m: lowest part of the body box ignored when counting hits (road)
    # KITTI class written for each pose. 'Pedestrian' trains one pedestrian class that also knows
    # lying / sitting people. KITTI itself has 'Person_sitting' if you want to split them off.
    pose_kitti_name=dict(lying='Pedestrian', sitting='Pedestrian', crouching='Pedestrian'),

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
POSES = ('lying', 'sitting', 'crouching')
AXES = ('pitch', 'yaw', 'roll')
# filled at the start of every run by calibrate_poses(): pose -> dict(changes=..., how=..., measured=...)
# why / last_fail: why posed pedestrians were not kept (printed when none are)
# fail_streak / disabled: circuit breaker, see spawn_hazards
HAZARD = dict(recipes={}, sk=None, why=Counter(), last_fail=None, fail_streak=0, disabled=False)
NAV = Counter()          # navigation points accepted / rejected by ground_below
CLIENTS = []             # every client made, kept until the script ends (one per run, see load_town)


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


def destroy_actor(a):
    """Destroy an actor once. Destroying one that is already gone makes CARLA print
    'failed to destroy actor ... not found', so check is_alive first."""
    try:
        if a.is_alive:
            a.destroy()
    except RuntimeError:
        pass


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


def in_box(pts, box, bottom_skip=0.1):
    """Mask of the points inside an upright box; the lowest bottom_skip m is skipped so road
    returns don't count."""
    x, y, z, l, w, h, yaw = box
    d = pts - np.array([x, y, z])
    c, s = np.cos(yaw), np.sin(yaw)
    lx, ly = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return ((np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2)
            & (d[:, 2] >= -h / 2 + bottom_skip) & (d[:, 2] <= h / 2))


def count_in_box(pts, box, bottom_skip=0.1):
    """Points inside an upright box; the lowest 0.1 m is skipped so road returns don't count."""
    return int(in_box(pts, box, bottom_skip).sum())


def velo_box(w2l, R, centre, fwd, dims):
    """World-frame box (centre, forward vector, l/w/h) -> [x,y,z,l,w,h,yaw] in the velodyne frame."""
    c = w2l @ np.array([centre[0], centre[1], centre[2], 1.0])
    f = R @ np.array(fwd, float)
    return np.array([c[0], -c[1], c[2], dims[0], dims[1], dims[2], math.atan2(-f[1], f[0])])


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


def footpath_spawn_points(world, min_n=10):
    """NEW. Spawn points with a footpath (map lane of type Sidewalk) beside their road, [] if the town has
    fewer than min_n (then nothing is filtered). Walkers and posed pedestrians can only be placed near
    footpaths: on the Town04 highway loop the test run had 0 walkers within 40 m."""
    cmap = world.get_map()
    out = []
    for sp in cmap.get_spawn_points():
        wp = cmap.get_waypoint(sp.location)
        if wp is not None and (sidewalk_beside(wp, 1) is not None or sidewalk_beside(wp, -1) is not None):
            out.append(sp)
    return out if len(out) >= min_n else []


def pick_ego_spawn(world):
    """Random choice among the 10% of spawn points with the most other spawn points nearby.
    UPDATED: among the spawn points with a footpath beside them (see footpath_spawn_points)."""
    sps = footpath_spawn_points(world) or world.get_map().get_spawn_points()
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


def ground_below(world, loc):
    """True if there is static geometry under loc. A navigation point with nothing under it (the
    log showed Town04 loading 'Nav/Town01.bin') would drop a walker into the void."""
    if not CFG['nav_ground_check']:
        return True
    try:
        ok = world.ground_projection(carla.Location(loc.x, loc.y, loc.z + 2.0), 6.0) is not None
    except (AttributeError, RuntimeError):
        return True                                      # API not available: cannot check
    NAV['ok' if ok else 'no_ground'] += 1
    return ok


def nav_point(world, ego_loc, near=True, tries=30):
    fallback = None
    for _ in range(tries):
        loc = world.get_random_location_from_navigation()
        if loc is None or not ground_below(world, loc):
            continue
        if not near or loc.distance(ego_loc) <= CFG['near_radius']:
            return loc
        fallback = loc
    return fallback


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


# ------------------------------------------------- NEW: keeping the scene populated around the ego
def sidewalk_points_near(world, ego_loc, n, band, avoid=()):
    """Up to n points on the footpaths beside the roads around the ego, band[0]-band[1] m away (map
    lanes of type Sidewalk). Fast in big towns, where random navigation points rarely land near the
    ego. The walkers' navigation mesh covers the footpaths, so their AI controller can start there."""
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
        loc = side.transform.location + carla.Location(x=r.x * off, y=r.y * off, z=1.0)   # dropped from 1 m
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


def count_near(actors, loc, r):
    n = 0
    for a in alive(actors):
        try:
            n += a.get_location().distance(loc) <= r
        except RuntimeError:
            pass
    return n


def configure_ego_autopilot(ego, tm):
    """Autopilot + the ego's own Traffic Manager settings (needed again after it was switched off / on)."""
    ego.set_autopilot(True, tm.get_port())
    tm.vehicle_percentage_speed_difference(ego, CFG['ego_speed_diff'])
    if CFG['ego_ignore_walkers_pct'] > 0:
        tm.ignore_walkers_percentage(ego, CFG['ego_ignore_walkers_pct'])
    if CFG['hazard_keep_ego_lane_clear']:
        tm.auto_lane_change(ego, False)               # stays in the lane that is kept clear of hazards


def ego_route(world, ego, dist=45.0, step=3.0):
    """[(x, y)] of the lane centres the ego can reach within dist m, every branch at junctions: the road
    ahead whichever way the autopilot turns. UPDATED: client-side map only. It used
    TrafficManager.get_all_actions(), a call into the Traffic Manager that runs inside this process
    (a suspect for the silent stop) and that only returns the points where the road option changes."""
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


def census(ego, vehicles, walkers, hazards, r=40.0):
    """One line: road users near the ego, to see at once when the scene empties."""
    loc = ego.get_transform().location
    posed = count_near([h['actor'] for h in hazards.values()], loc, CFG['hazard_far_m'])
    off = ' (posed pedestrians OFF for this run)' if HAZARD['disabled'] else ''
    return (f'within {r:.0f} m: {count_near(vehicles, loc, r)} vehicles, {count_near(walkers, loc, r)} walkers; '
            f'posed within {CFG["hazard_far_m"]:.0f} m: {posed}{off}')


def look_at_camera(world, bp, wp, dist=5.0, height=1.5):
    """A small camera (and the spectator) 5 m along the lane from waypoint wp, looking back at it:
    Unreal only updates a walker's skeleton while something renders it. Along the lane, not along
    world x, so the camera is not inside a building."""
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


def spawn_walkers(world, bp, ego_loc):
    world.set_pedestrians_cross_factor(CFG['pedestrian_cross_factor'])
    wbps = list(bp.filter('walker.pedestrian.*'))
    walkers = []
    NAV.clear()
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
    if NAV['no_ground']:
        tot = NAV['ok'] + NAV['no_ground']
        print(f"  navigation points without ground under them: {NAV['no_ground']}/{tot} rejected"
              + ('  [WARN] most of this town\'s navigation mesh does not match its geometry'
                 if NAV['no_ground'] > tot / 2 else ''))
    return walkers, ctrls


def recycle_vehicles(world, vehicles, ego_tf, tm=None, band=(65.0, 130.0), far_m=None, max_n=40):
    """Teleport vehicles farther than far_m (default 1.5 x near_radius) from the ego to free spawn points
    band m from it, ahead of it first. Default band 65-130 m: traffic keeps coming without popping up in
    LiDAR range. A closer band (vehicle_place_band) refills an empty scene, e.g. after the ego was moved.
    UPDATED: the moved vehicles' autopilot is switched off and on again (one tick in between), so the
    Traffic Manager plans a new route from the new spot instead of steering back to the old one, and
    their speed is set to 0. Was: only spots 65-130 m ahead, so after a teleport nothing came in range."""
    vehicles[:] = alive(vehicles)
    far_m = 1.5 * CFG['near_radius'] if far_m is None else far_m
    ego_loc, fwd = ego_tf.location, ego_tf.get_forward_vector()
    locs = {}
    for v in vehicles:
        try:
            locs[v.id] = v.get_transform().location
        except RuntimeError:
            pass
    movers = sorted((v for v in vehicles if v.id in locs and locs[v.id].distance(ego_loc) > far_m),
                    key=lambda v: -locs[v.id].distance(ego_loc))[:max_n]       # farthest first
    if not movers:
        return 0
    lo, hi = band
    ahead, other = [], []
    for sp in world.get_map().get_spawn_points():
        d = sp.location.distance(ego_loc)
        if lo <= d <= hi:
            dx, dy = sp.location.x - ego_loc.x, sp.location.y - ego_loc.y
            (ahead if (dx * fwd.x + dy * fwd.y) / d >= 0.5 else other).append(sp)
    random.shuffle(ahead)
    random.shuffle(other)
    cands = other + ahead                                     # pop() takes from the end: ahead first
    taken, plan = list(locs.values()) + [ego_loc], []
    for v in movers:
        while cands:
            sp = cands.pop()
            if all(sp.location.distance(p) >= 8.0 for p in taken):
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


def respawn_walkers(world, bp, walkers, ctrls, actors, ego_loc, far_m=None, band=None, max_n=None):
    """Replace walkers more than far_m (walker_far_m) from the ego with new walkers band m
    (walker_spawn_band) from it, so pedestrians stay in LiDAR range while the ego drives on (walkers at
    1.4 m/s can't follow). UPDATED: repopulate() calls it with a larger max_n after the ego was moved, and
    the points come from walker_points_near (random navigation points, then footpaths near the ego).
    Ticks the world twice: call it only AFTER the frame of this tick has been written."""
    far_m = CFG['walker_far_m'] if far_m is None else far_m
    band = CFG['walker_spawn_band'] if band is None else band
    max_n = CFG['walker_respawn_max'] if max_n is None else max_n
    walkers[:] = alive(walkers)
    ctrls[:] = alive(ctrls)
    far = []
    for w in walkers:
        try:
            if w.get_location().distance(ego_loc) > far_m:
                far.append(w)
        except RuntimeError:
            pass
    far = far[:max_n]
    if not far:
        return 0
    pts = walker_points_near(world, ego_loc, len(far), band)
    far = far[:len(pts)]
    if not far:
        return 0
    far_ids = {w.id for w in far}
    for c in list(ctrls):                                   # controllers first, then their walkers
        try:
            if c.parent is not None and c.parent.id in far_ids:
                c.stop()
                destroy_actor(c)
                ctrls.remove(c)
        except RuntimeError:
            pass
    for w in far:
        destroy_actor(w)
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


# ------------------------------------------------------------------- posed pedestrians: bones
# Ported from hazard_pose_test.py. The axes of the crl_ skeleton's bones are not documented, so
# the Poser tries the axis / sign options and keeps the one whose measured shape is right.
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
        raise RuntimeError(f'bones not found: {missing}. Bones of this walker: {names}')
    return sk


def world_bones(walker):
    """bone name -> world position. EMPTY if the simulator has no skeleton for the walker (yet)."""
    out = {}
    for b in walker.get_bones().bone_transforms:
        loc = b.world.location
        out[b.name] = np.array([loc.x, loc.y, loc.z], float)
    return out


def has_bones(walker):
    try:
        return len(world_bones(walker)) > 0
    except RuntimeError:
        return False


def relative_pose(walker):
    out = OrderedDict()
    for b in walker.get_bones().bone_transforms:
        r, loc = b.relative.rotation, b.relative.location
        out[b.name] = carla.Transform(carla.Location(loc.x, loc.y, loc.z),
                                      carla.Rotation(pitch=r.pitch, yaw=r.yaw, roll=r.roll))
    return out


def build_ctl(base, changes):
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
    return ctl


def set_pose(world, walker, base, changes, ticks=4):
    """Re-send the pose on every tick (like CARLA's bone example): a pose sent once can be
    overwritten by the next animation update."""
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


class Poser:
    def __init__(self, world, walker, ground_z, yaw):
        self.world, self.walker = world, walker
        self.ground_z, self.yaw = ground_z, yaw
        self.walker.get_pose_from_animation()
        for _ in range(2):
            world.tick()
        self.base = relative_pose(walker)
        self.sk = skeleton(list(self.base))

    def apply(self, changes):
        return set_pose(self.world, self.walker, self.base, changes)

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
        gz = min(p[2] for p in world_bones(walker).values())   # ground = lowest bone of the standing walker
        poser = Poser(world, walker, gz, 0.0)
        stand = shape(world_bones(walker), poser.sk, gz, 0.0)
        print(f"    standing   head {stand['head_z']:.2f} m  hips {stand['hips_z']:.2f} m  (should be ~1.6 / ~1.0)")
        sh = stand
        for pose in POSES:
            changes, how = getattr(poser, pose)()
            sh = shape(poser.apply(changes), poser.sk, gz, 0.0)
            ok = pose_ok(pose, sh)
            print(f"    {pose:<10} {'OK ' if ok else 'BAD'} head {sh['head_z']:.2f} m  hips {sh['hips_z']:.2f} m  "
                  f"length {sh['length']:.2f} m  ({how})")
            if ok:
                HAZARD['recipes'][pose] = dict(changes=changes, how=how,
                                               measured={k: round(v, 3) for k, v in sh.items()})
        if HAZARD['recipes']:
            HAZARD['sk'] = poser.sk
            return True
        if abs(sh['head_z'] - stand['head_z']) < 0.05:
            print('  [diag] the pose did not change at all: set_bones / blend_pose had no effect on this walker.')
        return False
    except (RuntimeError, AttributeError, ValueError) as e:    # no bone API (< 0.9.13), unknown skeleton, fell
        print(f'  [warn] pose calibration failed here ({e})')
        HAZARD['recipes'].clear()
        return False
    finally:
        if cam is not None:
            try:
                cam.stop()
            except RuntimeError:
                pass
            destroy_actor(cam)
        destroy_actor(walker)
        try:
            world.tick()
        except RuntimeError:
            pass


def calibrate_poses(world, bp, near_loc, attempts=2):
    """Measure the lying / sitting / crouching bone recipes. Done at the start of EVERY run: it
    proves that the bone API works in THIS world. Uses one temporary adult walker, facing yaw 0, on a
    flat non-junction DRIVING LANE (road geometry always exists, unlike a navigation point on a
    mismatched mesh). The ground is the lowest bone of the STANDING walker once it is at rest, so every
    later height is measured the same way. -> True if at least one pose passed its shape check.
    UPDATED: a small camera (and the spectator) looks at the calibration walker. Unreal only updates a
    walker's skeleton while something renders it: in Town01 nothing looked at that spot, so get_bones()
    stayed empty or 'standing' and every pose failed. If no pose passes, a second spot is tried."""
    HAZARD['recipes'].clear()
    HAZARD['sk'] = None
    print('  calibrating lying / sitting / crouching bone poses ...')
    cmap = world.get_map()
    wbp = bp.find('walker.pedestrian.0001')
    if wbp.has_attribute('is_invincible'):
        wbp.set_attribute('is_invincible', 'true')
    sps = cmap.get_spawn_points()
    random.shuffle(sps)
    tried = 0
    for sp in sps:
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
    if not tried:
        print('  [warn] could not spawn the calibration walker anywhere')
    if not HAZARD['recipes']:
        print('  [warn] no pose passed its shape check: posed pedestrians are switched off for this run')
        return False
    return True


def save_hazard_recipes(root):
    if HAZARD['recipes']:
        with open(os.path.join(root, 'hazard_poses.json'), 'w') as fh:
            json.dump(dict(blueprint='walker.pedestrian.0001', poses=HAZARD['recipes']), fh, indent=2)


# ------------------------------------------------------------- posed pedestrians: placement
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


def near_junction(wp, gap, step=3.0):
    """True if a junction is within gap m ahead of or behind wp along its lane."""
    d = step
    while d <= gap + 1e-6:
        for w in wp.next(d) + wp.previous(d):
            if w.is_junction:
                return True
        d += step
    return False


def hazard_spot(world, ego_tf, place, allow_road=True, route=()):
    """-> (location, spawn_z, yaw, place) near the ego, or None.
    lane: in a driving lane (lying people across it); edge: at the right edge of a lane;
    footpath: on the footpath beside the road (falls back to edge where there is none).
    spawn_z is the waypoint height: only used to drop the walker from; its real ground height is
    measured later from its own bones.
    UPDATED: lane / edge spots keep hazard_junction_gap m from junctions (cars stopped in a junction
    block every direction) and 2.8 m from the route the Traffic Manager plans for the ego (route =
    ego_route(); after a turn the ego's lane has another road id, so the same-lane test missed it).
    allow_road=False (road quota full): footpath only, no fallback to the lane edge."""
    cmap = world.get_map()
    wp0 = cmap.get_waypoint(ego_tf.location)
    if wp0 is None:
        return None
    lo, hi = CFG['hazard_band']
    route = np.array(route, float).reshape(-1, 2)
    for _ in range(10):
        d = random.uniform(lo, hi)
        ahead = random.random() < CFG['hazard_front_fraction']
        cands = wp0.next(d) if ahead else wp0.previous(d)
        if not cands:
            continue
        wp = random.choice(cands)
        if wp.is_junction:
            continue
        if random.random() < 0.4:                              # sometimes the neighbouring lane
            adj = wp.get_left_lane() if random.random() < 0.5 else wp.get_right_lane()
            if adj is not None and adj.lane_type == carla.LaneType.Driving and not adj.is_junction:
                wp = adj
        kind, base, off = place, wp, 0.0
        if kind == 'footpath':
            side = sidewalk_beside(wp, 1) or sidewalk_beside(wp, -1)
            if side is not None:
                base, off = side, random.uniform(-0.5, 0.5)
            elif not allow_road:
                continue
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
                continue
            if len(route) and np.min(np.hypot(route[:, 0] - loc.x, route[:, 1] - loc.y)) < 2.8:
                continue                                       # on the ego's planned route
        if CFG['hazard_keep_ego_lane_clear'] and ahead and kind != 'footpath':
            f = ego_tf.get_forward_vector()
            dx, dy = loc.x - ego_tf.location.x, loc.y - ego_tf.location.y
            same_lane = wp.road_id == wp0.road_id and wp.lane_id == wp0.lane_id
            in_path = dx * f.x + dy * f.y > 0 and abs(-dx * f.y + dy * f.x) < 3.2
            if same_lane or in_path:
                continue                                       # would block the ego for good
        along = wp.transform.rotation.yaw
        return loc, base.transform.location.z, along, kind
    return None


def _send_pose(world, items, ticks):
    """Re-send each walker's pose on every tick, like CARLA's bone example."""
    for _ in range(ticks):
        for h in items:
            if h.get('bad'):
                continue
            try:
                h['actor'].set_bones(h['ctl'])
                h['actor'].blend_pose(1.0)
            except RuntimeError:
                h['bad'] = True
        world.tick()


def _pose_check(h):
    if h.get('bad'):
        return False
    try:
        bw = world_bones(h['actor'])
        if not bw:                                             # skeleton not available (yet)
            HAZARD['why']['no_bones_at_check'] += 1
            return False
        sh = shape(bw, HAZARD['sk'], h['ground_z'], h['yaw'])
    except (RuntimeError, ValueError, KeyError):
        return False
    ok = pose_ok(h['pose'], sh)
    if not ok:
        HAZARD['last_fail'] = (h['pose'], h['place'], {k: round(v, 2) for k, v in sh.items()})
    return ok


def spawn_hazards(world, bp, hazards, actors, ego_tf, n, route=()):
    """Spawn up to n posed pedestrians near the ego and register them in `hazards` (actor id -> info).
    Ticks the world ~25 times: call it only AFTER the frame of this tick has been written.
    UPDATED: at most hazard_max_road of them on lanes / lane edges; the circuit breaker only switches
    posed pedestrians off when the pose has NEVER worked in this run (was: 3 refreshes in a row with 0
    kept, which also happened when the ego was stuck and no free spot was found -> off for the whole
    run, so nothing came back after the ego was moved)."""
    recipes = HAZARD['recipes']
    if n <= 0 or not recipes or HAZARD['disabled']:
        return 0
    why = HAZARD['why']
    poses = list(recipes)
    pose_w = [CFG['hazard_poses'].get(p, 0.0) or 1e-6 for p in poses]
    places = list(CFG['hazard_places'])
    place_w = [CFG['hazard_places'][p] for p in places]
    wbps = list(bp.filter('walker.pedestrian.*'))
    if CFG['hazard_adults_only']:
        adults = [b for b in wbps if not (b.has_attribute('age') and b.get_attribute('age').as_str() == 'child')]
        wbps = adults or wbps
    n_road = sum(1 for h in hazards.values() if h['place'] != 'footpath')
    new = []
    for _ in range(n * 4):
        if len(new) >= n:
            break
        pose = random.choices(poses, pose_w)[0]
        allow_road = n_road < CFG['hazard_max_road']
        spot = hazard_spot(world, ego_tf, random.choices(places, place_w)[0] if allow_road else 'footpath',
                           allow_road, route)
        if spot is None:
            why['no_spot'] += 1
            continue
        loc, gz, along, place = spot
        if pose == 'lying':                                    # across the lane / along the kerb
            yaw = along + (random.choice((90.0, -90.0)) if place == 'lane' else random.choice((0.0, 180.0)))
            yaw += random.uniform(-10.0, 10.0)
        else:
            yaw = random.uniform(-180.0, 180.0)
        b = random.choice(wbps)
        if b.has_attribute('is_invincible'):
            b.set_attribute('is_invincible', 'false')
        w = None
        for dz in (0.6, 1.0, 1.4):
            w = world.try_spawn_actor(b, carla.Transform(carla.Location(loc.x, loc.y, gz + dz),
                                                         carla.Rotation(yaw=yaw)))
            if w is not None:
                break
        if w is None:
            why['could_not_spawn'] += 1
            continue
        try:
            w.set_collisions(True)                             # no AI controller: stays visible to LiDAR
        except (RuntimeError, AttributeError):
            pass
        new.append(dict(actor=w, pose=pose, place=place, yaw=yaw, root=HAZARD['sk']['root']))
        if place != 'footpath':
            n_road += 1
    if not new:
        return 0
    for _ in range(12):                                        # fall onto the ground
        world.tick()
    for h in new:
        try:
            h['actor'].get_pose_from_animation()
        except RuntimeError:
            h['bad'] = True
    for _ in range(2):
        world.tick()
    for _ in range(6):                                         # get_bones() can be empty for a fresh walker
        if all(h.get('bad') or has_bones(h['actor']) for h in new):
            break
        world.tick()
    for h in new:                                              # measure from the walker's OWN standing pose
        if h.get('bad'):
            continue
        try:
            bw = world_bones(h['actor'])
            if not bw:
                h['bad'] = True
                why['no_bones'] += 1
                continue
            h['ground_z'] = min(p[2] for p in bw.values())
            h['ctl'] = build_ctl(relative_pose(h['actor']), recipes[h['pose']]['changes'])
        except RuntimeError:
            h['bad'] = True
    n_checked = sum(1 for h in new if not h.get('bad'))        # walkers that really got a pose to check
    _send_pose(world, new, 4)
    failed = [h for h in new if not _pose_check(h)]
    if failed:                                                 # some builds need longer before the pose shows
        _send_pose(world, [h for h in failed if not h.get('bad')], 8)
    kept = 0
    for h in new:
        w = h['actor']
        ok = _pose_check(h)
        if ok and h in failed:
            why['pose_needed_retry'] += 1
        if ok:
            hazards[w.id] = h
            actors.append(w)
            kept += 1
            why['kept'] += 1
        else:
            why['pose_check_failed'] += 1
            destroy_actor(w)
    # circuit breaker: only when the pose never takes in this world (not when spots were just scarce)
    if kept:
        HAZARD['fail_streak'] = 0
    elif n_checked and not why['kept']:
        HAZARD['fail_streak'] += 1
        if HAZARD['fail_streak'] >= CFG['hazard_fail_limit']:
            HAZARD['disabled'] = True
            print(f"\n  [WARN] {HAZARD['fail_streak']} attempts in a row and not one walker took the pose in "
                  f"this run: posed pedestrians are switched off for the rest of it. "
                  f"last failed check: {HAZARD['last_fail']}")
    elif n_checked >= 3:
        why['refresh_all_failed'] += 1                         # worked before, not this time: counted only
    return kept


def refresh_hazards(world, bp, hazards, actors, ego_tf, route=()):
    """Drop posed pedestrians that are out of range and top up to n_hazards. -> number spawned."""
    ego_loc = ego_tf.location
    for i, h in list(hazards.items()):
        try:
            far = h['actor'].get_location().distance(ego_loc) > CFG['hazard_far_m']
        except RuntimeError:
            hazards.pop(i)
            continue
        if far:
            destroy_actor(h['actor'])
            hazards.pop(i)
    missing = CFG['n_hazards'] - len(hazards)
    return spawn_hazards(world, bp, hazards, actors, ego_tf, missing, route) if missing > 0 else 0





def reapply_hazards(hazards):
    """Re-send the bone pose so it cannot fall back to the idle animation."""
    for i, h in list(hazards.items()):
        try:
            h['actor'].set_bones(h['ctl'])
            h['actor'].blend_pose(1.0)
        except RuntimeError:
            hazards.pop(i)


# ------------------------------------------------------------------------- unsticking the ego
def blockers_ahead(world, ego, dist, lat):
    """vehicles / walkers in a corridor in front of the ego -> [(actor, metres ahead)], nearest first"""
    tf = ego.get_transform()
    loc, fwd = tf.location, tf.get_forward_vector()
    actors = world.get_actors()
    out = []
    for a in list(actors.filter('vehicle.*')) + list(actors.filter('walker.pedestrian.*')):
        if a.id == ego.id:
            continue
        try:
            p = a.get_location()
        except RuntimeError:
            continue
        dx, dy = p.x - loc.x, p.y - loc.y
        ahead, lateral = dx * fwd.x + dy * fwd.y, abs(-dx * fwd.y + dy * fwd.x)
        if 0.0 < ahead < dist and lateral < lat:
            out.append((a, ahead))
    out.sort(key=lambda t: t[1])
    return out


def clear_blocking_hazards(hazards, ego_tf):
    """The autopilot stops for a person in its lane: remove the posed pedestrians within 40 m in
    front of the ego (any place: edge hazards block too)."""
    loc, fwd = ego_tf.location, ego_tf.get_forward_vector()
    n = 0
    for i, h in list(hazards.items()):
        try:
            if not h['actor'].is_alive:
                hazards.pop(i)
                continue
            p = h['actor'].get_location()
            dx, dy = p.x - loc.x, p.y - loc.y
            ahead, lateral = dx * fwd.x + dy * fwd.y, abs(-dx * fwd.y + dy * fwd.x)
            if 0.0 < ahead < 40.0 and lateral < 4.0:
                destroy_actor(h['actor'])
                hazards.pop(i)
                n += 1
        except RuntimeError:
            hazards.pop(i)
    return n


def remove_walkers_ahead(world, ego, walkers, ctrls, hazards, dist=25.0, lat=3.5):
    """remove ordinary walkers (and their controllers) standing in the ego's path"""
    ids = {a.id for a, _ in blockers_ahead(world, ego, dist, lat) if a.type_id.startswith('walker.pedestrian')}
    if not ids:
        return 0
    for c in alive(ctrls):
        try:
            if c.parent is not None and c.parent.id in ids:
                c.stop()
                destroy_actor(c)
        except RuntimeError:
            pass
    for w in alive(walkers):
        if w.id in ids:
            destroy_actor(w)
    for i in ids:
        hazards.pop(i, None)
    walkers[:] = alive(walkers)
    ctrls[:] = alive(ctrls)
    return len(ids)


def teleport_ego(world, ego, tm):
    """Unstick step 3 (and the empty-street move). UPDATED: the new spot is a free spawn point with a
    footpath beside it, teleport_band (30-200 m) from where the ego is (was: any spawn point in the town, far from all the traffic, walkers and posed
    pedestrians, which were spawned around the old spot), with no vehicle within 15 m and none in the
    30 m in front of it. The autopilot is switched off during the move so the Traffic Manager forgets
    the old route. Ticks once. -> distance moved (m), or None if no spot was found."""
    old = ego.get_transform().location
    others = []
    for a in world.get_actors().filter('vehicle.*'):
        if a.id != ego.id:
            try:
                others.append(a.get_location())
            except RuntimeError:
                pass
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
    if not cands:                                             # as before: anywhere free
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


def unstick_ego(world, ego, tm, hazards, walkers, ctrls, level):
    """The ego has not moved for unstick_ticks. Say what is in front of it, then escalate one step:
    0 = remove posed pedestrians ahead, 1 = remove ordinary walkers ahead, 2 = move the ego.
    -> (the level to start from next time, True if the ego was moved: then call repopulate)."""
    tf = ego.get_transform()
    near = blockers_ahead(world, ego, 30.0, 3.5)
    desc = ', '.join(f"{a.type_id.split('.', 1)[1]}{' (posed)' if a.id in hazards else ''} {d:.0f} m"
                     for a, d in near[:4]) or 'nothing within 30 m'
    light = str(ego.get_traffic_light_state()) if ego.is_at_traffic_light() else 'not at a light'
    print(f'\n  [stuck] ego standing still. Ahead: {desc}. Traffic light: {light}.')
    for lvl in range(level, 3):
        if lvl == 0:
            n, what = clear_blocking_hazards(hazards, tf), 'removed posed pedestrians ahead'
        elif lvl == 1:
            n, what = remove_walkers_ahead(world, ego, walkers, ctrls, hazards), 'removed walkers ahead'
        else:
            dist = teleport_ego(world, ego, tm)
            n, what = int(dist is not None), f'moved the ego {dist or 0:.0f} m to a free spawn point'
        if n:
            print(f'    -> {what}' + (f' ({n})' if lvl < 2 else ''))
            return min(lvl + 1, 2), lvl == 2
    return 2, False


def repopulate(world, bp, tm, ego, vehicles, walkers, ctrls, actors, hazards):
    """NEW. After the ego was moved, bring the scene to it: everything was spawned around the old spot.
    Far vehicles go to free spawn points vehicle_place_band (20-70 m) from the ego, far walkers are
    replaced 8-50 m from it, posed pedestrians out of range are replaced, then settle_ticks pass without
    saving (moved cars pull away, new walkers land, poses are re-sent). -> one-line summary"""
    tf = ego.get_transform()
    nv = recycle_vehicles(world, vehicles, tf, tm, band=CFG['vehicle_place_band'], far_m=CFG['near_radius'],
                          max_n=int(CFG['n_vehicles'] * CFG['near_fraction']))
    lo, hi = CFG['walker_spawn_band']
    nw = respawn_walkers(world, bp, walkers, ctrls, actors, tf.location, band=(lo, hi + 15.0),
                         max_n=int(CFG['n_walkers'] * CFG['near_fraction']))
    nh = 0
    if CFG['n_hazards'] and HAZARD['recipes']:
        HAZARD['fail_streak'] = 0
        if HAZARD['why']['kept']:
            HAZARD['disabled'] = False                        # the pose worked in this run: try again here
        nh = refresh_hazards(world, bp, hazards, actors, tf, ego_route(world, ego))
    for t in range(CFG['settle_ticks']):
        world.tick()
        if CFG['hazard_reapply_ticks'] and t % CFG['hazard_reapply_ticks'] == 0:
            reapply_hazards(hazards)
    return (f'scene rebuilt: {nv} vehicles moved, {nw} walkers and {nh} posed pedestrians respawned near '
            f'the ego. Now {census(ego, vehicles, walkers, hazards)}')





def posed_box_world(walker, info):
    """Body box of a posed walker from its bone positions (CARLA's own box stays upright).
    -> (centre xyz, forward xyz, (l, w, h)) in world coordinates, or None.
    The long axis comes from a principal-axis fit of the bones, pointing the way the walker faces."""
    try:
        bw = world_bones(walker)
        fv = walker.get_transform().get_forward_vector()
    except RuntimeError:
        return None
    pts = np.array([p for n, p in bw.items() if n != info['root']])
    # get_bones() can come back empty for a walker the simulator is removing or has not updated yet
    if pts.ndim != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        return None
    xy = pts[:, :2]
    c = xy.mean(0)
    axis = np.linalg.svd(xy - c, full_matrices=False)[2][0]
    if axis @ np.array([fv.x, fv.y]) < 0:
        axis = -axis
    lat = np.array([-axis[1], axis[0]])
    a, b = (xy - c) @ axis, (xy - c) @ lat
    m = CFG['hazard_box_margin']
    a0, a1, b0, b1 = a.min() - m, a.max() + m, b.min() - m, b.max() + m
    l, w = max(a1 - a0, 0.5), max(b1 - b0, 0.5)
    cxy = c + axis * (a0 + a1) / 2 + lat * (b0 + b1) / 2
    z0, z1 = info['ground_z'], float(pts[:, 2].max()) + m
    return (cxy[0], cxy[1], (z0 + z1) / 2), (axis[0], axis[1], 0.0), (l, w, z1 - z0)


def spawn_sensors(world, bp, ego):
    """UPDATED: + keepalive_cams (left / back / right). Unreal only updates a walker's skeleton while a
    camera renders it, so posed pedestrians beside or behind the ego kept the standing bones and failed
    the pose check. They are in `sensors` (stopped and destroyed with the others) but not in `queues`
    (SensorSync never waits for them; their images are dropped by the empty callback)."""
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
    if CFG['keepalive_cams']:
        w, h = CFG['keepalive_res']
        x, y, z = CFG['lidar_xyz']
        for yaw in (90.0, 180.0, -90.0):
            b = bp.find('sensor.camera.rgb')
            for k, v in (('image_size_x', w), ('image_size_y', h), ('fov', 100.0)):
                b.set_attribute(k, str(v))
            cam = world.spawn_actor(b, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)),
                                    attach_to=ego)
            cam.listen(lambda _img: None)
            sensors[f'keepalive_{int(yaw)}'] = cam
    return sensors, queues


# ------------------------------------------------------------------------------------ labels
def make_labels(world, ego, data, velo_xyz, Tr, Kmat, posed=None):
    """-> (label_2 lines, labels_3d lines, hazard lines, Counter)"""
    posed = posed or {}
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
    label2, dontcare, l3d, hz, stats = [], [], [], [], Counter()

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
        fw = tf.get_forward_vector()
        box = velo_box(w2l, R, (cw.x, cw.y, cw.z), (fw.x, fw.y, fw.z),
                       (2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z))
        x, y, z = box[:3]
        dist = math.hypot(x, y)
        if dist > CFG['label_max_dist']:
            continue

        pose, capsule = None, None
        info = posed.get(actor.id)
        if info is not None:                                   # posed walker: box from the bones
            pb = posed_box_world(actor, info)
            if pb is None:
                continue
            capsule, pose = box, info['pose']                  # CARLA's upright box = the 'ghost' capsule
            box = velo_box(w2l, R, pb[0], pb[1], pb[2])
            x, y, z = box[:3]
        yaw = box[6]

        n_ghost = 0
        if pose:
            on_body = in_box(agent_xyz, box, 0.0)
            n_sem = int(on_body.sum())
            n_ghost = int((in_box(agent_xyz, capsule, 0.0) & ~on_body).sum())
        else:
            n_sem = hits.get(actor.id, 0)
            if n_sem == 0 and len(agent_xyz):
                n_sem = count_in_box(agent_xyz, box, bottom_skip=0.0)
                if n_sem:
                    stats['id_fallback_' + cls] += 1
        n_box = count_in_box(velo_xyz, box, CFG['hazard_bottom_skip'] if pose else 0.1)
        name = CFG['pose_kitti_name'][pose] if pose else KITTI_NAME[cls]

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
                label2.append(K.kitti_line(name, trunc, occ, alpha, img, dims, loc, ry))
                stats[name] += 1
                if pose:
                    stats['hz_' + pose] += 1
                labelled = 1
            elif min(img[2] - img[0], img[3] - img[1]) >= CFG['min_dontcare_px']:
                dontcare.append(K.dontcare_line(img))
                stats['DontCare'] += 1
        l3d.append(f'{cls} {x:.3f} {y:.3f} {z:.3f} {box[3]:.3f} {box[4]:.3f} {box[5]:.3f} '
                   f'{yaw:.4f} {n_box} {actor.id} {labelled} {dist:.2f} {n_sem}')
        stats['3d_' + cls] += 1
        if cls == 'person' and n_sem >= CFG['min_points']:
            stats['3d_person_seen'] += 1                       # pedestrian the LiDAR actually hit
        if pose:
            stats['3d_hz_' + pose] += 1
            stats['ghostpts_' + pose] += n_ghost
            if n_sem >= CFG['min_points']:
                stats['seen_hz_' + pose] += 1
            hz.append(f"{actor.id} {pose} {info['place']} {n_sem} {n_ghost} {labelled} "
                      f"{x:.3f} {y:.3f} {z:.3f} {box[3]:.3f} {box[4]:.3f} {box[5]:.3f} {yaw:.4f}")
    # DontCare MUST come last: OpenPCDet and mmdet3d take the first N lines (N = non-DontCare
    # count) as the real boxes, exactly like the original KITTI files are ordered.
    return label2 + dontcare, l3d, hz, stats


def write_frame(tr, fid, data, world, ego, Tr, Kmat, calib_txt, plane_txt, run, frame, posed=None):
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
    label2, l3d, hz, stats = make_labels(world, ego, data, pts[:, :3], Tr, Kmat, posed)
    with open(f'{tr}/label_2/{fid}.txt', 'w') as fh:
        fh.write('\n'.join(label2) + ('\n' if label2 else ''))
    with open(f'{tr}/labels_3d/{fid}.txt', 'w') as fh:
        fh.write('\n'.join(l3d))
    with open(f'{tr}/hazards/{fid}.txt', 'w') as fh:
        fh.write('\n'.join(hz))
    tf, v, ctl = ego.get_transform(), ego.get_velocity(), ego.get_control()
    meta = {'frame_id': fid, 'sim_frame': frame, 'run': run['name'], 'town': run['town'],
            'weather': run['weather'], 'seed': CFG['seed'],
            'ego_location': [tf.location.x, tf.location.y, tf.location.z],
            'ego_rotation': [tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll],
            'ego_speed': math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2),
            'control': {'throttle': ctl.throttle, 'steer': ctl.steer, 'brake': ctl.brake},
            'at_traffic_light': ego.is_at_traffic_light(),
            'traffic_light_state': str(ego.get_traffic_light_state()),
            'n_points': int(len(pts)), 'labels': dict(stats), 'n_posed': len(posed or {}),
            'sensor_frames': {k: d.frame for k, d in data.items()}}
    with open(f'{tr}/metadata/{fid}.json', 'w') as fh:
        json.dump(meta, fh)
    return stats


# ------------------------------------------------------------------------------------ a run
def nav_matches(world, n=60):
    """-> (ok, share of random navigation points with ground under them). A correct mesh has ground
    under ~all its points. UPDATED: the last run loaded Town04 with Town05's mesh (client printed
    'Nav/Town05.bin') and that still had ground under 52% of its points, so the old 50% test and the
    'next to a lane' test (50%) both passed it. Now nav_min_ground (85%)."""
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


def connect():
    c = carla.Client(CFG['host'], CFG['port'])
    c.set_timeout(CFG['timeout'])
    CLIENTS.append(c)          # kept alive: dropping a client whose Traffic Manager is still known can crash
    return c


def load_town(client, town):
    """-> (client to use for this run, world).
    UPDATED (third test): the CARLA client loads the walkers' navigation mesh only ONCE, for the map the
    server has when the client first connects, and never again for later load_world / reload_world:
    run 1 got Nav/Town05.bin for Town04, run 2 Nav/Town04.bin (by luck, Town04 was still loaded) and no Nav
    file at all for Town01, run 3 Nav/Town01.bin for Town04 even after three reloads. So `client` only
    loads the map, then a NEW client connects: its navigation mesh is the one of this map.
    If the mesh is still wrong, the run stops instead of recording frames without proper pedestrians."""
    client.load_world(town)
    time.sleep(2.0)
    g = 0.0
    for attempt in range(3):
        fresh = connect()                                     # its navigation mesh = the map loaded now
        world = fresh.get_world()
        ok, g = nav_matches(world)
        if ok:
            print(f'  navigation mesh matches {town} ({g:.0%} of its points have ground under them)')
            return fresh, world
        print(f'  navigation mesh does not match {town} ({g:.0%} of its points have ground under them): '
              f'connecting again ({attempt + 1}/3) ...')
        time.sleep(3.0)
    raise SystemExit(f'Stopped: the walkers have no navigation mesh for {town} ({g:.0%}). Look for a line '
                     f'"INFO: Found the required file in cache! Carla/Maps/Nav/{town}.bin" above: if a Nav line '
                     f'names another town, restart CARLA and run this town alone (--runs <run name>).')


def collect_run(client, run, start_idx, tr, Tr, Kmat, calib_txt, plane_txt):
    print(f"\n=== {run['name']}: {run['town']} / {run['weather']} / {run['frames']} frames ===")
    client, world = load_town(client, run['town'])      # UPDATED: a new client for each run (navigation mesh)
    original = world.get_settings()
    tm = client.get_trafficmanager(CFG['tm_port'])
    actors, sensors, ctrls = [], {}, []
    totals, skipped, respawned, hz_spawned, rebuilt = Counter(), 0, 0, 0, 0
    HAZARD['why'].clear()
    HAZARD['last_fail'] = None
    HAZARD['fail_streak'] = 0
    HAZARD['disabled'] = False
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
        if CFG['n_hazards']:
            calibrate_poses(world, bp, ego.get_transform().location)   # every run: proves set_bones works HERE
        configure_ego_autopilot(ego, tm)
        # UPDATED: sensors BEFORE the road users: the cameras must be rendering when the posed
        # pedestrians are posed (Unreal updates a skeleton only while something renders it)
        sensors, queues = spawn_sensors(world, bp, ego)
        actors += list(sensors.values())
        follow_spectator(world, ego)
        ego_loc = ego.get_transform().location
        vehicles = spawn_vehicles(world, bp, tm, ego_loc)
        actors += vehicles
        walkers, ctrls = spawn_walkers(world, bp, ego_loc)
        actors += walkers + ctrls
        hazards = {}                                  # posed pedestrians: actor id -> info
        if CFG['n_hazards'] and HAZARD['recipes']:
            hz_spawned += spawn_hazards(world, bp, hazards, actors, ego.get_transform(), CFG['n_hazards'],
                                        ego_route(world, ego))
            print(f"  posed pedestrians: {len(hazards)}/{CFG['n_hazards']} "
                  f"({', '.join(sorted(Counter(h['pose'] for h in hazards.values()).elements()))})")
            if len(hazards) < CFG['n_hazards']:
                print(f"    why not all: {dict(HAZARD['why'])}")
                if HAZARD['last_fail']:
                    print(f"    last failed pose check (pose, place, measured m): {HAZARD['last_fail']}")

        for _ in range(CFG['warmup_ticks']):
            world.tick()
        sync = SensorSync(queues)
        sync.drain()
        print(f'  [scene] start: {census(ego, vehicles, walkers, hazards)}')

        saved, tick, static, still, moving, level, empty = start_idx, 0, 0, 0, 0, 0, 0
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
                                          calib_txt, plane_txt, run, frame, hazards)
                    saved += 1
                    k = saved - start_idx
                    print(f"  {k}/{run['frames']}", end='\n' if k % 50 == 0 else '\r', flush=True)
                    if CFG['census_every'] and (saved - start_idx) % CFG['census_every'] == 0:
                        print(f"  [scene] frame {saved - start_idx}: {census(ego, vehicles, walkers, hazards)}")
            # maintenance AFTER the frame is written: teleports only show up next tick
            if tick % 5 == 0:
                follow_spectator(world, ego)
            v = ego.get_velocity()                    # the autopilot stops for any person / vehicle in its path
            speed2 = v.x ** 2 + v.y ** 2
            still = still + 1 if speed2 < 0.25 else 0
            moving = moving + 1 if speed2 > 4.0 else 0
            if moving >= 20:
                level = 0                             # driving again: start the escalation from the first step
            moved = False
            if CFG['unstick_ticks'] and still >= CFG['unstick_ticks']:
                if ego.is_at_traffic_light() and ego.get_traffic_light_state() == carla.TrafficLightState.Red:
                    still = 0                         # a red light is not a jam
                else:
                    level, moved = unstick_ego(world, ego, tm, hazards, walkers, ctrls, level)
                    still = 0
            if CFG['empty_move_ticks'] and tick % 20 == 0 and not moved:     # NEW: no pedestrians around
                empty = empty + 20 if count_near(walkers, ego.get_transform().location, 40.0) == 0 else 0
                if empty >= CFG['empty_move_ticks']:
                    dist = teleport_ego(world, ego, tm)
                    print(f'\n  [empty] no walker within 40 m for {empty * CFG["fixed_delta"]:.0f} s (road without '
                          f'footpaths?): ego moved {dist or 0:.0f} m to a street with footpaths')
                    moved, empty = dist is not None, 0
            if CFG['teleport_test'] and tick % CFG['teleport_test'] == 0 and not moved:
                dist = teleport_ego(world, ego, tm)
                print(f'\n  [teleport-test] tick {tick}: ego moved {dist or 0:.0f} m')
                moved = dist is not None
            if moved:                                 # NEW: everything was spawned around the old spot
                print(f'    -> {repopulate(world, bp, tm, ego, vehicles, walkers, ctrls, actors, hazards)}')
                rebuilt += 1
                still = static = moving = 0
                follow_spectator(world, ego)
                continue                              # the periodic jobs below would only repeat it
            etf = ego.get_transform()
            if CFG['recycle_ticks'] and tick % CFG['recycle_ticks'] == 0:
                n_near = count_near(vehicles, etf.location, CFG['near_radius'])
                if n_near < CFG['min_vehicles_near']:  # NEW: too few around the ego -> bring far ones in range
                    recycle_vehicles(world, vehicles, etf, tm, band=CFG['vehicle_place_band'],
                                     far_m=CFG['near_radius'], max_n=CFG['min_vehicles_near'] - n_near)
                else:
                    recycle_vehicles(world, vehicles, etf, tm)
            if CFG['walker_respawn_ticks'] and tick % CFG['walker_respawn_ticks'] == 0:
                respawned += respawn_walkers(world, bp, walkers, ctrls, actors, etf.location)
            if hazards or (CFG['n_hazards'] and HAZARD['recipes'] and not HAZARD['disabled']):
                if CFG['hazard_respawn_ticks'] and tick % CFG['hazard_respawn_ticks'] == 0 and HAZARD['recipes']:
                    hz_spawned += refresh_hazards(world, bp, hazards, actors, etf, ego_route(world, ego))
                if CFG['hazard_reapply_ticks'] and tick % CFG['hazard_reapply_ticks'] == 0:
                    reapply_hazards(hazards)
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
              f'{respawned} walkers respawned near the ego, {hz_spawned} posed pedestrians spawned, '
              f'{rebuilt} scene rebuilds after moving the ego)')
        print('  label_2 objects per frame: ' + ', '.join(
            f'{k} {v / max(n, 1):.1f}' for k, v in sorted(totals.items())
            if not k.startswith(('3d_', 'id_fallback_', 'ghostpts_', 'seen_'))))
        print('  labels_3d (360 deg) per frame: ' + ', '.join(
            f'{k[3:]} {v / max(n, 1):.1f}' for k, v in sorted(totals.items()) if k.startswith('3d_')))
        if totals['3d_person'] and not totals['3d_person_seen']:
            print('  [WARN] no pedestrian got >= min_points LiDAR hits. The walker collision fix did '
                  'not work on this CARLA build: pedestrians are invisible to the LiDAR.')
        for p in POSES:
            n3 = totals['3d_hz_' + p]
            if not n3:
                continue
            print(f"  posed {p}: {n3} in range, {totals['seen_hz_' + p]} with >= min_points hits on the body, "
                  f"{totals['hz_' + p]} in label_2, ghost points per instance {totals['ghostpts_' + p] / n3:.1f}")
            if not totals['seen_hz_' + p]:
                print(f'  [WARN] no posed "{p}" pedestrian got LiDAR hits on its body: the LiDAR only sees the '
                      'upright collision capsule (ghost points), so these cannot be used for LiDAR training.')
        if CFG['n_hazards']:
            print(f"  posed pedestrian bookkeeping: {dict(HAZARD['why'])}"
                  + ('  [OFF: the pose never worked in this run]' if HAZARD['disabled'] else ''))
        fb = {k[12:]: v for k, v in totals.items() if k.startswith('id_fallback_')}
        if fb:
            print(f'  hits found by tag-in-box because the actor id was missing: {fb}  '
                  '(large for "person" = CARLA does not put walker ids on LiDAR hits)')
        return saved
    finally:
        # UPDATED: world and Traffic Manager go back to asynchronous FIRST, then the actors are destroyed
        # without waiting for a tick. apply_batch_sync(..., True) waits for one more synchronous tick: at
        # the end of the Town04 test it never came (time-out after 60 s) and the error stopped the script
        # before progress.json was written. Cleanup problems are now printed, not raised.
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
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original)
        except RuntimeError as e:
            print(f'  [warn] could not switch CARLA back to asynchronous mode ({e})')
        try:
            live = alive(actors)
            if live:
                client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in live])
            time.sleep(1.0)
        except RuntimeError as e:
            print(f'  [warn] cleanup failed ({e}). The frames of this run are saved. If the CARLA window is '
                  f'frozen or closed, restart CARLA before the next run.')


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
    ap.add_argument('--no-hazards', action='store_true', help='no lying / sitting / crouching pedestrians')
    ap.add_argument('--frames', type=int, help='frames per run (overrides CFG and --quick)')
    ap.add_argument('--teleport-test', type=int, default=0, metavar='TICKS',
                    help='move the ego every TICKS ticks, to test the scene rebuild (e.g. 400)')
    args = ap.parse_args()
    if args.no_hazards:
        CFG['n_hazards'] = 0
    CFG['teleport_test'] = args.teleport_test

    for xyz in (CFG['cam_xyz'], CFG['lidar_xyz']):
        assert len(xyz) == 3
    random.seed(CFG['seed']); np.random.seed(CFG['seed'])
    root = CFG['out'] + ('_quick' if args.quick else '')     # smoke test never mixes with real data
    tr = os.path.join(root, 'training')
    for d in ('velodyne', 'image_2', 'calib', 'label_2', 'planes', 'semantic', 'labels_3d', 'hazards', 'metadata'):
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

    client = connect()                                        # only loads maps; each run gets its own client
    idx = progress['next_idx']
    for run in CFG['runs']:
        if args.runs and run['name'] not in args.runs:
            continue
        if run['name'] in progress['runs']:
            print(f"skip {run['name']} (already collected)")
            continue
        run = dict(run, frames=args.frames or (40 if args.quick else run['frames']))
        start = idx
        idx = collect_run(client, run, start, tr, Tr, Kmat, calib_txt, plane_txt)
        progress['runs'][run['name']] = {'first': start, 'last': idx - 1, 'split': run['split'],
                                         'town': run['town'], 'weather': run['weather']}
        progress['next_idx'] = idx
        with open(prog_path, 'w') as fh:
            json.dump(progress, fh, indent=2)
        write_splits(root, progress)
        save_hazard_recipes(root)
    print(f'\ndone: {idx} frames in {root}. Next: python check_kitti.py {root}')


if __name__ == '__main__':
    main()