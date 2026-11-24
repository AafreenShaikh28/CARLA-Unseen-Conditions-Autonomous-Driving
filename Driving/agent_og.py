"""
agent.py -- closed-loop driving in CARLA: perception -> world model -> decision -> control.

Lives next to perception_scheduler.py, route.py, world_model.py, decision.py,
evaluator.py. Needs CARLA's agents package: pass --carla-root <CARLA folder>
(the one that contains PythonAPI) or set CARLA_ROOT; and pip install networkx.

MILESTONES
  M1  --check-boxes       detector boxes (converted to world) vs CARLA's own boxes,
                          plus RGB overlays. Gate before trusting any run where
                          detections drive the car. Run it with --scenario route
                          --traffic 30 --decide-on oracle (the mirror/yaw tests need
                          cars beside and across the ego; static gives NOT TESTED)
  M2  --scenario route    route following in an empty town (perception optional)
  M3  evaluator.py        4 output files per run + project score
  M4  --scenario static   parked car in the ego lane 60 m ahead; must stop 2-10 m short
  S3  --scenario route --traffic 30
                          traffic following: a slower lead car on the ego route
                          (--lead-distance, default 25 m) must be followed >= 10 s,
                          route completed, no ego-caused collision / emergency stop /
                          TTC < 1.5 s outside junctions

  Run M4 three ways, same seed:
    --obstacle-source none    perception off: MUST collide (proves causality)
    --obstacle-source fusion  the evaluated run (lidar = the LiDAR-only baseline)
    --obstacle-source oracle  CARLA ground truth: diagnostic only, disclose as such

GROUND TRUTH, AND WHERE IT IS ALLOWED
  ego pose + speed     read from CARLA (MAP-track-equivalent setup: route and pose given)
  obstacles            only with --obstacle-source oracle
  evaluator, box check scoring only; never fed back into decisions

USAGE (PowerShell, one line each; CARLA server running)
  python agent.py --carla-root C:\\path\\to\\CARLA_0.9.15 --scenario route --obstacle-source none
  python agent.py --carla-root ... --scenario route --traffic 30 --obstacle-source fusion --seg-model SEG --check-boxes --decide-on oracle
  python agent.py --carla-root ... --scenario static --obstacle-source fusion --seg-model SEG
  python agent.py --carla-root ... --scenario static --obstacle-source none
  python agent.py --carla-root ... --scenario static --obstacle-source oracle

Presentation video (recorder.py): add --record to any run -> <run folder>/video.mp4, real speed,
with the route, the perceived boxes, the driving state and the car camera drawn in. --record-view top
films from above (good for traffic following).
  python agent.py --carla-root ... --scenario static --obstacle-source fusion --seg-model SEG --record
"""
import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

try:
    import carla
except ImportError:
    raise SystemExit("the 'carla' package is not importable in this environment")

import perception_scheduler as ps
import lidar_weather as lw
from control import SpeedController, shape_control
from decision import Params, Decider, assess, predict_crossing, ROUTE_COMPLETE, BLOCKED_WAIT, PEDESTRIAN_CLASSES
from evaluator import Evaluator
from route import Route, wrap_pi
from world_model import Obstacle, WorldModel, footprint

DT = ps.DT
EGO_BP = 'vehicle.tesla.model3'   # same as the data collector


# ----------------------------------------------------------------- weather
# Unseen-weather tests. The detectors were trained on ClearNoon (Town04) and CloudySunset (Town01).
# CARLA has no fog preset, so 'Fog' is ClearNoon with dense ground fog.
CUSTOM_WEATHER = {'Fog': ('ClearNoon', dict(fog_density=60.0, fog_distance=10.0, fog_falloff=1.0))}


def check_weather(args):
    """--weather / --weather-set -> (preset, overrides), checked before the models load"""
    base, extra = CUSTOM_WEATHER.get(args.weather, (args.weather, {}))
    preset = getattr(carla.WeatherParameters, base, None)
    if not isinstance(preset, carla.WeatherParameters):
        names = sorted(n for n in dir(carla.WeatherParameters) if not n.startswith('_')
                       and isinstance(getattr(carla.WeatherParameters, n, None), carla.WeatherParameters))
        raise SystemExit(f"--weather {args.weather}: not a weather preset of this CARLA. "
                         f"Choose from: {', '.join(names + sorted(CUSTOM_WEATHER))}")
    overrides = dict(extra)
    for item in filter(None, (args.weather_set or '').split(',')):
        key, _, val = item.partition('=')
        key = key.strip()
        if not hasattr(preset, key) or key.startswith('_'):
            raise SystemExit(f"--weather-set {item}: '{key}' is not a weather field "
                             f"(e.g. fog_density, fog_distance, precipitation, wetness, sun_altitude_angle)")
        try:
            overrides[key] = float(val)
        except ValueError:
            raise SystemExit(f"--weather-set {item}: '{val}' is not a number")
    return preset, overrides


def apply_weather(world, preset, overrides):
    """sets the weather; returns (every weather field that was set, vehicle lights needed).
    Copies the preset into a fresh WeatherParameters, so the preset itself is never modified."""
    w = carla.WeatherParameters()
    fields = {}
    for n in dir(preset):
        v = getattr(preset, n, None)
        if n.startswith('_') or isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        val = float(overrides.get(n, v))
        try:
            setattr(w, n, val)
        except AttributeError:                     # a read-only field: not a weather setting
            continue
        fields[n] = round(val, 3)
    world.set_weather(w)
    lights = (fields.get('sun_altitude_angle', 90.0) < 10.0 or fields.get('precipitation', 0.0) >= 30.0
              or fields.get('fog_density', 0.0) >= 20.0 or fields.get('dust_storm', 0.0) >= 20.0)
    return fields, lights


def headlights_on(vehicles, fog):
    """low beams (+ fog lights) on every moving car, as real drivers would at night / in rain or fog"""
    L = carla.VehicleLightState
    state = L(int(L.Position) | int(L.LowBeam) | (int(L.Fog) if fog else 0))
    for v in vehicles:
        v.set_light_state(state)


# ----------------------------------------------------------------- helpers
def add_agents_path(carla_root):
    for root in (carla_root, os.environ.get('CARLA_ROOT')):
        if root:
            p = os.path.join(root, 'PythonAPI', 'carla')
            if os.path.isdir(os.path.join(p, 'agents')):
                if p not in sys.path:
                    sys.path.insert(0, p)
                return p
    import importlib.util
    if importlib.util.find_spec('agents') is not None:
        return 'already on sys.path'
    raise SystemExit("CARLA's 'agents' package not found. Pass --carla-root <folder that "
                     "contains PythonAPI> or set CARLA_ROOT. It also needs: pip install networkx")


GT_TYPE = {}                      # actor id -> CARLA type_id, for the tick log


def actor_obstacle(a, k):
    """CARLA ground-truth box of an actor, in the same form as detector obstacles."""
    GT_TYPE[a.id] = a.type_id
    tf = a.get_transform()
    bb = a.bounding_box
    c = np.array(tf.get_matrix()) @ np.array([bb.location.x, bb.location.y, bb.location.z, 1.0])
    yaw = math.radians(tf.rotation.yaw + bb.rotation.yaw)
    l, w, h = 2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z
    v = a.get_velocity()
    cls = 'pedestrian' if a.type_id.startswith('walker.') else 'vehicle'
    return Obstacle(cls, 1.0, c[:3], footprint(c[0], c[1], l, w, yaw), (l, w, h), yaw,
                    np.array([v.x, v.y]), a.id, k, 'oracle')


def gt_obstacles(world, ego, k, radius=60.0):
    ego_loc = ego.get_location()
    out = []
    for a in world.get_actors():
        if a.id == ego.id or not (a.type_id.startswith('vehicle.') or a.type_id.startswith('walker.')):
            continue
        if a.get_location().distance(ego_loc) <= radius:
            out.append(actor_obstacle(a, k))
    return out


def disable_parked(world):
    """Map-baked parked cars are not actors: they have no GT box, so the detector
    would see cars the oracle and the box check cannot."""
    n = 0
    for name in ('Car', 'Truck', 'Bus', 'Motorcycle', 'Bicycle'):
        lbl = getattr(carla.CityObjectLabel, name, None)
        if lbl is None:
            continue
        ids = {o.id for o in world.get_environment_objects(lbl)}
        if ids:
            world.enable_environment_objects(ids, False)
            n += len(ids)
    if n:
        print(f"  disabled {n} map-baked parked vehicles")


def spawn_traffic(world, bp, tm, sps, n, keep_clear, radius=15.0):
    """n autopilot cars (4 wheels only: CARLA's two-wheeler boxes are a separate
    question) at spawn points at least `radius` m from the ego start / obstacle."""
    cars = [b for b in bp.filter('vehicle.*')
            if not b.has_attribute('number_of_wheels') or int(b.get_attribute('number_of_wheels')) == 4]
    pts = [p for p in sps if all(p.location.distance(c) > radius for c in keep_clear)]
    random.shuffle(pts)
    out = []
    for p in pts:
        if len(out) >= n:
            break
        b = random.choice(cars)
        if b.has_attribute('role_name'):
            b.set_attribute('role_name', 'autopilot')
        v = world.try_spawn_actor(b, p)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            out.append(v)
    return out


def make_walkers_lidar_visible(walkers):
    """CARLA 0.9.15 bug (issue #7092): controller.ai.walker switches the walker's collision off, and the
    ray-cast LiDAR skips actors without collision, so pedestrians get NO LiDAR points. Same workaround
    as the data collector: collision (and physics) back on after every start() / new destination."""
    for w in walkers:
        if not w.is_alive:
            continue
        try:
            w.set_collisions(True)
            w.set_simulate_physics(True)
        except (RuntimeError, AttributeError) as e:
            print(f"  [warn] could not re-enable pedestrian collision ({e}): the LiDAR may not see them")
            return


def retarget_walkers(ctrls, walkers, pool):
    """new destinations near the route, so pedestrians keep walking during long runs"""
    for c in ctrls:
        if c.is_alive:
            c.go_to_location(random.choice(pool))
    make_walkers_lidar_visible(walkers)


def spawn_walkers(world, bp, n, cross, route_xy, keep_clear, seed, ctrls, actors, near=30.0, radius=15.0):
    """n pedestrians on the sidewalks within `near` m of the ego route, each walking to another such
    place; `cross` (0-1) is the share that may cross the road anywhere, not only at crossings
    (CARLA's pedestrian cross factor). Each walker goes into `actors` and its AI controller into
    `ctrls` as soon as it exists (the cleanup stops and destroys controllers first). Ticks once."""
    if hasattr(world, 'set_pedestrians_seed'):
        world.set_pedestrians_seed(int(seed))
    world.set_pedestrians_cross_factor(float(cross))
    pool = []
    for _ in range(60 * n):                       # random sidewalk points, kept if near the route
        if len(pool) >= 3 * n:
            break
        loc = world.get_random_location_from_navigation()
        if loc is not None and float(np.min(np.hypot(route_xy[:, 0] - loc.x, route_xy[:, 1] - loc.y))) <= near:
            pool.append(loc)
    wbps = list(bp.filter('walker.pedestrian.*'))
    walkers, speeds = [], []
    for loc in pool:
        if len(walkers) >= n:
            break
        if any(loc.distance(c) < radius for c in keep_clear):
            continue
        b = random.choice(wbps)
        if b.has_attribute('is_invincible'):
            b.set_attribute('is_invincible', 'false')     # so that a hit registers as a collision
        w = world.try_spawn_actor(b, carla.Transform(loc))
        if w is None:
            continue
        actors.append(w)
        walkers.append(w)
        try:
            speeds.append(float(b.get_attribute('speed').recommended_values[1]))     # walking
        except (AttributeError, IndexError, ValueError):
            speeds.append(1.4)
    cbp = bp.find('controller.ai.walker')
    for w in walkers:
        ctrls.append(world.spawn_actor(cbp, carla.Transform(), attach_to=w))
    world.tick()                                  # the server must know the walkers before start()
    for c, v in zip(ctrls[-len(walkers):] if walkers else [], speeds):
        c.start()
        c.go_to_location(random.choice(pool))
        c.set_max_speed(v)
    make_walkers_lidar_visible(walkers)
    return walkers, pool


def spawn_lead(world, bp, tm, route, dist, speed_kmh, bp_id):
    """Step 3: a car `dist` m ahead of the ego, on the ego's own route, driven by the
    Traffic Manager along that route at `speed_kmh`. It obeys traffic lights, so the
    ego also gets stop-and-go behind it. Returns (vehicle, info, spawn location)."""
    b = bp.find(bp_id)
    if b.has_attribute('role_name'):
        b.set_attribute('role_name', 'lead')
    wps = route.wps
    s_first = dist + 5.0                           # ego front (~2.4 m) + lead half length (~2.4 m)
    for s in np.arange(s_first, s_first + 30.0, 3.0):
        i = int(min(np.searchsorted(route.s, s), len(wps) - 1))
        t = wps[i].transform
        tf = carla.Transform(carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + 0.5),
                             t.rotation)
        v = world.try_spawn_actor(b, tf)
        if v is not None:
            break
    else:
        return None, None, None
    v.set_autopilot(True, tm.get_port())
    tm.auto_lane_change(v, False)
    tm.distance_to_leading_vehicle(v, 3.0)
    path, last = [], route.s[i]
    for j in range(i + 1, len(wps)):
        if route.s[j] - last >= 10.0 or j == len(wps) - 1:
            path.append(wps[j].transform.location)
            last = route.s[j]
    info = dict(bp=bp_id, start_s_m=round(float(route.s[i]), 1), speed_kmh=speed_kmh,
                follows_ego_route=hasattr(tm, 'set_path') and bool(path))
    if info['follows_ego_route']:                  # else the TM picks its own turns at junctions
        tm.set_path(v, path)
    if hasattr(tm, 'set_desired_speed'):
        tm.set_desired_speed(v, speed_kmh)
    else:                                          # older TM: % below the speed limit (30 km/h in town)
        tm.vehicle_percentage_speed_difference(v, 100.0 * (1.0 - speed_kmh / 30.0))
    return v, info, tf.location


def walk_lane(wp, dist, step):
    out = [wp]
    while (len(out) - 1) * step < dist:
        nxt = out[-1].next(step)
        if not nxt:
            return out, False
        out.append(nxt[0])
    return out, True


def pick_route(args, cmap, sps):
    start = sps[args.start_index % len(sps)]
    if args.goal_index is not None:
        return start, sps[args.goal_index % len(sps)].location, None
    wp0 = cmap.get_waypoint(start.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane, _ = walk_lane(wp0, args.route_length, 2.0)
    print(f"  route scenario: spawn point {args.start_index}, about {args.route_length:.0f} m along the lane")
    return start, lane[-1].transform.location, None


def pick_static(args, cmap, sps):
    """First spawn point (from --start-index) with a straight, junction-free stretch."""
    need = args.obstacle_distance + args.after_obstacle
    for i in range(len(sps)):
        idx = (args.start_index + i) % len(sps)
        wp0 = cmap.get_waypoint(sps[idx].location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
        lane, ok = walk_lane(wp0, need, 1.0)
        if not ok or any(w.is_junction for w in lane):
            continue
        yaw0 = wp0.transform.rotation.yaw
        if max(abs((w.transform.rotation.yaw - yaw0 + 180) % 360 - 180) for w in lane) > 3.0:
            continue
        print(f"  static scenario: spawn point {idx}, {need:.0f} m straight, no junction")
        return sps[idx], lane[-1].transform.location, lane[int(round(args.obstacle_distance))]
    raise SystemExit("no straight, junction-free stretch found; try another --town or --start-index")


class SteerTarget:
    """What VehiclePIDController reads from a waypoint (with offset 0) is only
    .transform.location, so an interpolated point can stand in for one."""

    def __init__(self, xy, z):
        self.transform = carla.Transform(carla.Location(float(xy[0]), float(xy[1]), float(z)))


def _stop_waypoints(tl, cmap):
    """the light's stop-line waypoints; if this CARLA lacks get_stop_waypoints, the same thing
    the way CARLA's own BasicAgent finds it: the trigger volume's lane, walked up to the junction"""
    try:
        return list(tl.get_stop_waypoints())
    except (AttributeError, RuntimeError):
        pass
    try:
        from agents.tools.misc import get_trafficlight_trigger_location
        wp = cmap.get_waypoint(get_trafficlight_trigger_location(tl))
        for _ in range(40):
            nxt = wp.next(1.0)
            if not nxt or nxt[0].is_junction:
                break
            wp = nxt[0]
        return [wp]
    except Exception:
        return []


def route_stop_lines(world, route, cmap=None, max_d=2.0, max_angle_deg=30.0):
    """Traffic-light stop lines that lie on the route, from the map: each light's stop waypoints
    (one per lane it controls), kept if within max_d m of the route and facing the same way.
    Returns [{'s', 'light', 'id', 'ends'}] sorted by route position s (ends = the stop line's
    two 3D end points, for the video)."""
    out = {}
    cos_max = math.cos(math.radians(max_angle_deg))
    for tl in world.get_actors().filter('traffic.traffic_light'):
        wps = _stop_waypoints(tl, cmap)
        for wp in wps:
            loc = wp.transform.location
            j, t, dist, _ = route._nearest(np.array([[loc.x, loc.y]]), 0, len(route.seg_len))
            if dist[0] > max_d:
                continue
            s_line = float(route.s[j[0]] + t[0] * route.seg_len[j[0]])
            yaw = math.radians(wp.transform.rotation.yaw)
            if float(np.dot([math.cos(yaw), math.sin(yaw)], route.tangent_at(s_line))) < cos_max:
                continue                                   # the other direction
            half = 0.5 * float(getattr(wp, 'lane_width', 3.5))
            r = (-math.sin(yaw), math.cos(yaw))            # to the right of the lane, CARLA axes
            ends = [(loc.x - r[0] * half, loc.y - r[1] * half, loc.z + 0.05),
                    (loc.x + r[0] * half, loc.y + r[1] * half, loc.z + 0.05)]
            if tl.id not in out or s_line < out[tl.id]['s']:
                out[tl.id] = {'s': s_line, 'light': tl, 'id': tl.id, 'ends': ends}
    return sorted(out.values(), key=lambda x: x['s'])


def light_state(tl):
    """'Red' / 'Yellow' / 'Green' / 'Off' / 'Unknown' (CARLA ground truth)"""
    try:
        return str(tl.get_state()).split('.')[-1]
    except RuntimeError:
        return 'Unknown'


def follow_spectator(spectator, tf):
    yaw = math.radians(tf.rotation.yaw)
    loc = carla.Location(tf.location.x - 9 * math.cos(yaw), tf.location.y - 9 * math.sin(yaw),
                         tf.location.z + 5)
    spectator.set_transform(carla.Transform(loc, carla.Rotation(pitch=-20, yaw=tf.rotation.yaw)))


# ------------------------------------------------------------- M1 box check
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def box8(o):
    z0, z1 = o.center[2] - o.size[2] / 2, o.center[2] + o.size[2] / 2
    return np.vstack([np.c_[o.corners, np.full(4, z0)], np.c_[o.corners, np.full(4, z1)]])


class BoxCheck:
    """M1 gate: detector boxes (converted to world) vs CARLA's boxes of the same frame.

    Matching: all det-GT pairs within match_radius, nearest first (one-to-one).
    Errors are also expressed in the LiDAR frame at capture, y-LEFT like the
    training data, so an offset has a direction (forward / left).
    Gate statistics use one record per (actor, 2 m distance band, motion state):
    150 ticks stopped behind one car are one observation, not 150.
    Two tests target the conversion code rather than the model:
      mirror test   for every detection more than 1.5 m to the side: is a car
                    nearer to it as detected, or to its mirror image? (y-flip bug)
      yaw test      which (sign, offset) maps detected yaw onto true yaw; only
                    meaningful when centres are right and the objects face
                    several directions (traffic)."""

    def __init__(self, out_dir, w, h, fov, max_images=10, every=20, match_radius=3.0):
        self.dir = os.path.join(out_dir, 'box_check')
        os.makedirs(self.dir, exist_ok=True)
        self.w, self.h = w, h
        self.f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
        self.max_images, self.every, self.match_radius = max_images, every, match_radius
        self.records, self.saved = [], 0
        self.n_dup = self.n_fp = self.near_gt = self.missed_gt = 0
        self.fp_cls = {}
        self.mirror_votes = {'as_is': 0, 'mirrored': 0}

    @staticmethod
    def _local(M_inv, xy, z, yaw_world, lidar_yaw):
        p = M_inv @ np.array([xy[0], xy[1], z, 1.0])
        yaw = -wrap_pi(yaw_world - lidar_yaw)            # y-right -> y-left flips the angle
        return float(p[0]), float(-p[1]), float(yaw)

    def run(self, k, dets, gts, ego_xy, bgr, cam_tf, lidar_tf, moving):
        M_inv = np.array(lidar_tf.get_inverse_matrix())
        lyaw = math.radians(lidar_tf.rotation.yaw)
        # mirror test on ALL detections, matched or not: a mirrored box lands > 3 m from
        # its car and would never be matched, so matched pairs cannot reveal a y-flip
        gl = [self._local(M_inv, g.center[:2], g.center[2], g.yaw, lyaw)[:2] for g in gts]
        for d in dets:
            x, y, _ = self._local(M_inv, d.center[:2], d.center[2], d.yaw, lyaw)
            if abs(y) <= 1.5 or not gl:                 # near the centreline both hypotheses agree
                continue
            near = min(math.hypot(x - gx, y - gy) for gx, gy in gl)
            near_m = min(math.hypot(x - gx, -y - gy) for gx, gy in gl)
            if near <= self.match_radius and near < near_m:
                self.mirror_votes['as_is'] += 1
            elif near_m <= self.match_radius and near_m < near:
                self.mirror_votes['mirrored'] += 1
        pairs = sorted((float(np.hypot(*(d.center[:2] - g.center[:2]))), i, j)
                       for i, d in enumerate(dets) for j, g in enumerate(gts))
        used_d, used_g, match = set(), set(), []
        for dist, i, j in pairs:
            if dist > self.match_radius:
                break
            if i not in used_d and j not in used_g:
                used_d.add(i)
                used_g.add(j)
                match.append((i, j, dist))
        for i, j, dist in match:
            d, g = dets[i], gts[j]
            dx, dy, dyaw = self._local(M_inv, d.center[:2], d.center[2], d.yaw, lyaw)
            gx, gy, gyaw = self._local(M_inv, g.center[:2], g.center[2], g.yaw, lyaw)
            e = abs(float(wrap_pi(d.yaw - g.yaw)))
            self.records.append(dict(
                k=k, gt_id=g.track_id, gt_cls=g.cls, det_cls=d.cls, score=round(d.score, 3),
                moving=bool(moving), dist=round(float(np.hypot(gx, gy)), 2),
                gt_x=round(gx, 3), gt_y=round(gy, 3), det_x=round(dx, 3), det_y=round(dy, 3),
                err_fwd=round(dx - gx, 3), err_left=round(dy - gy, 3), err_xy=round(dist, 3),
                gt_yaw_deg=round(math.degrees(gyaw), 1), det_yaw_deg=round(math.degrees(dyaw), 1),
                err_yaw_deg=round(math.degrees(min(e, math.pi - e)), 1),
                err_z=round(float(d.center[2] - g.center[2]), 3),
                gt_l=round(g.size[0], 2), gt_w=round(g.size[1], 2),
                det_l=round(d.size[0], 2), det_w=round(d.size[1], 2),
                err_l=round(float(d.size[0] - g.size[0]), 3), err_w=round(float(d.size[1] - g.size[1]), 3)))
        for i, d in enumerate(dets):
            if i in used_d:
                continue
            near = [g for g in gts if np.hypot(*(d.center[:2] - g.center[:2])) <= self.match_radius]
            if near:
                self.n_dup += 1                   # a second box on an object that already has one
            else:
                self.n_fp += 1
                self.fp_cls[d.cls] = self.fp_cls.get(d.cls, 0) + 1
        near = [j for j, g in enumerate(gts) if np.hypot(*(g.center[:2] - np.asarray(ego_xy))) < 30.0]
        self.near_gt += len(near)
        self.missed_gt += sum(1 for j in near if j not in used_g)
        if bgr is not None and cam_tf is not None and self.saved < self.max_images and k % self.every == 0:
            self._overlay(k, bgr, cam_tf, dets, gts)

    def _overlay(self, k, bgr, cam_tf, dets, gts):
        import cv2
        img = bgr.copy()
        Minv = np.array(cam_tf.get_inverse_matrix())
        for obs, color in ((gts, (0, 0, 255)), (dets, (0, 255, 0))):
            for o in obs:
                P = np.c_[box8(o), np.ones(8)] @ Minv.T          # camera: x fwd, y right, z up
                if (P[:, 0] < 0.5).any():
                    continue
                u = self.f * P[:, 1] / P[:, 0] + self.w / 2.0
                v = -self.f * P[:, 2] / P[:, 0] + self.h / 2.0
                pts = [(int(round(a)), int(round(b))) for a, b in zip(u, v)]
                for a, b in EDGES:
                    cv2.line(img, pts[a], pts[b], color, 2)
                if color == (0, 255, 0):
                    cv2.putText(img, o.cls, pts[4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, f"k={k}  red = CARLA box, green = detector box (+class)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(self.dir, f"overlay_{k:05d}.png"), img)
        self.saved += 1

    @staticmethod
    def _dedup(r):
        seen, out = set(), []
        for x in r:
            key = (x['gt_id'], int(x['dist'] // 2), x['moving'])
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    def summary(self, yaw_sign=1.0, yaw_offset=0.0):
        r = self.records
        u = self._dedup(r)
        veh = [x for x in u if x['gt_cls'] == 'vehicle']

        def med(key, rows):
            return round(float(np.median([x[key] for x in rows])), 3) if rows else None

        s = dict(records=len(r), observations=len(u), distinct_actors=len({x['gt_id'] for x in u}),
                 duplicate_boxes=self.n_dup, false_positives=self.n_fp, false_positive_classes=self.fp_cls,
                 gt_within_30m=self.near_gt, gt_within_30m_missed=self.missed_gt,
                 median_center_err_m=med('err_xy', u),
                 p90_center_err_m=round(float(np.percentile([x['err_xy'] for x in u], 90)), 3) if u else None,
                 median_err_forward_m=med('err_fwd', u), median_err_left_m=med('err_left', u),
                 median_yaw_err_deg_vehicles=med('err_yaw_deg', veh), median_z_err_m=med('err_z', u),
                 median_length_err_m=med('err_l', veh), median_width_err_m=med('err_w', veh),
                 images=self.saved)
        s['by_detected_class'] = {
            c: dict(n=len(rows), median_center_err_m=med('err_xy', rows),
                    median_det_l=med('det_l', rows), median_det_w=med('det_w', rows),
                    median_yaw_err_deg=med('err_yaw_deg', rows))
            for c in sorted({x['det_cls'] for x in u})
            for rows in [[x for x in u if x['det_cls'] == c]]}

        a_, m_ = self.mirror_votes['as_is'], self.mirror_votes['mirrored']
        s['mirror_votes'] = dict(self.mirror_votes)
        if a_ + m_ < 20:
            s['mirror_test'] = (f'not testable: only {a_ + m_} detections more than 1.5 m to the side '
                                f'near a car (run --scenario route --traffic 30)')
        elif m_ > 2 * a_:
            s['mirror_test'] = 'Y-FLIP: detections sit where their cars would be if mirrored left/right'
        elif a_ > 2 * m_:
            s['mirror_test'] = 'ok'
        else:
            s['mirror_test'] = 'unclear'
        flipped = s['mirror_test'].startswith('Y-FLIP')

        spread = float(np.std([x['gt_yaw_deg'] for x in veh])) if veh else 0.0
        if len(veh) >= 10 and spread > 20.0:
            gyaw = np.radians([x['gt_yaw_deg'] for x in veh])
            dyaw = np.radians([x['det_yaw_deg'] for x in veh])
            # errors are folded to 0-90 deg (a box turned by 180 deg is the same box), so offsets 0/180
            # and +90/-90 cannot be told apart: test 0 and 90 only, and keep the current convention
            # unless another one is better by more than 1 deg (an exact 0/180 tie used to pick 180)
            cand = {}
            for sg in (1.0, -1.0):
                for off in (0.0, 90.0):
                    e = np.abs(wrap_pi(sg * dyaw + math.radians(off) - gyaw))
                    cand[(sg, off)] = float(np.median(np.degrees(np.minimum(e, math.pi - e))))
            (sg, off), m = min(cand.items(), key=lambda kv: kv[1])
            if m > cand[(1.0, 0.0)] - 1.0:
                (sg, off), m = (1.0, 0.0), cand[(1.0, 0.0)]
            boxes_wrong = (s['median_center_err_m'] > 0.5 or abs(s['median_length_err_m'] or 0) > 0.5
                           or abs(s['median_width_err_m'] or 0) > 0.5)
            if sg == 1.0 and off == 0.0:
                s['yaw_test'] = f'ok (median {m:.1f} deg)'
            elif flipped:
                s['yaw_test'] = f'{sg:+.0f} x detected {off:+.0f} deg fits best: explained by the y-flip, fix that first'
            elif boxes_wrong:
                s['yaw_test'] = (f'{sg:+.0f} x detected {off:+.0f} deg fits best, but centre or size is ALSO wrong, '
                                 f'which no yaw convention causes: the boxes themselves are wrong (model or '
                                 f'its training labels). Do not change --yaw-sign/--yaw-offset for this')
            else:
                new_sign = yaw_sign * sg
                new_off = yaw_offset - math.radians(off) * yaw_sign * sg
                s['yaw_test'] = (f'true yaw = {sg:+.0f} x detected {off:+.0f} deg fits best (median {m:.1f} deg). '
                                 f'Only if centres and sizes are right, try --yaw-sign {new_sign:+.0f} '
                                 f'--yaw-offset {new_off:.4f}')
        else:
            s['yaw_test'] = (f'not testable: vehicles all face about the same way (spread {spread:.0f} deg) '
                             f'or < 10 observations (run --scenario route --traffic)')

        s['gate'] = ('on de-duplicated observations: >= 20, median centre < 0.5 m, median vehicle yaw < 10 deg, '
                     '|median z| < 0.3 m, mirror test ok, yaw test ok (heuristic thresholds). The mirror and '
                     'yaw tests need cars beside and across the ego: --scenario route --traffic 30')
        basic = (len(u) >= 20 and s['median_center_err_m'] < 0.5 and
                 (s['median_yaw_err_deg_vehicles'] is None or s['median_yaw_err_deg_vehicles'] < 10.0) and
                 abs(s['median_z_err_m']) < 0.3)
        untested = s['mirror_test'].startswith('not testable') or s['yaw_test'].startswith('not testable')
        failed = (not basic or flipped or s['mirror_test'] == 'unclear'
                  or not (s['yaw_test'].startswith('ok') or s['yaw_test'].startswith('not testable')))
        s['status'] = ('FAIL' if failed else
                       'NOT TESTED (mirror/yaw tests could not run: use --scenario route --traffic 30)'
                       if untested else 'PASS')
        s['passed'] = s['status'] == 'PASS'
        with open(os.path.join(self.dir, 'box_check.json'), 'w') as f:
            json.dump(dict(summary=s, records=r), f, indent=2)
        return s


def rescore_box_checks(folder, yaw_sign=1.0, yaw_offset=0.0):
    """re-applies the M1 verdict rules to saved box checks (box_check.json keeps every record), e.g.
    after a fix to the rules. No CARLA needed. The numbers stay the same; only the verdict can change"""
    import glob
    paths = sorted(glob.glob(os.path.join(folder, '*', 'box_check', 'box_check.json')))
    if not paths:
        raise SystemExit(f"no box_check.json found in {os.path.join(folder, '*', 'box_check')}")
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        old = d['summary']
        bc = BoxCheck.__new__(BoxCheck)            # no folders created, nothing re-collected
        bc.dir, bc.records = os.path.dirname(p), d['records']
        bc.n_dup, bc.n_fp = old['duplicate_boxes'], old['false_positives']
        bc.near_gt, bc.missed_gt = old['gt_within_30m'], old['gt_within_30m_missed']
        bc.fp_cls, bc.mirror_votes, bc.saved = old['false_positive_classes'], old['mirror_votes'], old['images']
        new = bc.summary(yaw_sign, yaw_offset)     # rewrites box_check.json
        run_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        print(f"{run_name}: {old.get('status')} -> {new['status']}" + ('' if new['status'] == old.get('status')
                                                                        else f"   (yaw test: {new['yaw_test']})"))


# ------------------------------------------------------------------- main
def run(args):
    if args.rescore_box_check:
        rescore_box_checks(args.rescore_box_check, args.yaw_sign if args.yaw_sign is not None else 1.0,
                           args.yaw_offset if args.yaw_offset is not None else 0.0)
        return
    add_agents_path(args.carla_root)
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    from agents.navigation.controller import VehiclePIDController

    random.seed(args.seed)
    np.random.seed(args.seed)
    detector = args.obstacle_source in ('lidar', 'fusion')
    if args.check_boxes and not detector:
        raise SystemExit("--check-boxes needs --obstacle-source lidar or fusion")
    if args.check_boxes and args.timing != 'lockstep':
        raise SystemExit("--check-boxes compares with GT of the same frame: use --timing lockstep")
    speed_kmh = args.speed_kmh or (20.0 if args.scenario == 'static' else 30.0)
    params = Params(v0_kmh=speed_kmh)
    weather_preset, weather_overrides = check_weather(args)
    if args.walkers < 0 or not 0.0 <= args.walker_cross <= 1.0:
        raise SystemExit("--walkers must be >= 0 and --walker-cross between 0 and 1")
    if args.walkers and args.scenario == 'static':
        raise SystemExit("--walkers is for --scenario route: in M4 a pedestrian walking into the stopped "
                         "car would end the test")

    # ---------------- perception (only for detector sources) ----------------
    sched = wm = dc = cfg = None
    if detector:
        ps._need_torch()
        import torch
        torch.manual_seed(args.seed)
        ps.resolve_ckpts(args)
        device = ps.get_device()
        print(f"device: {device}\nloading project modules ...")
        mods = {
            'dc': ps.load_module('dc_module', os.path.join(args.data_collection_dir, 'data_collection.py')),
            'pp_lidar': ps.load_module('pp_common_lidar', os.path.join(args.detection_lidar_dir, 'pp_common.py')),
            'pp_fusion': ps.load_module('pp_common_fusion', os.path.join(args.fusion_dir, 'pp_common.py')),
            'projection': ps.load_module('check_projection_live',
                                         os.path.join(args.fusion_dir, 'check_projection.py')),
        }
        dc = mods['dc']
        dc.CFG['use_semantic_lidar'] = False
        cfg = dc.CFG
        table = ps.load_latency_table(args.latency_table)
        qkey = 'p95_ms' if args.latency_quantile == 'p95' else 'median_ms'
        args.calibrate = False
        print("loading trained models ...")
        streams, _ = ps.build_streams(args, device, mods, cfg, table)
        sched = ps.PerceptionScheduler(streams, timing=args.timing, slack_ticks=args.slack_ticks,
                                       sync_fn=lambda: ps.sync_if_cuda(device), strict=not args.lenient)
        ps.print_plan(sched, table, qkey)
        pp = mods['pp_fusion'] if args.obstacle_source == 'fusion' else mods['pp_lidar']
        yaw_sign = args.yaw_sign if args.yaw_sign is not None else float(getattr(pp, 'YAW_SIGN', 1.0))
        yaw_offset = args.yaw_offset if args.yaw_offset is not None else float(getattr(pp, 'YAW_OFFSET', 0.0))
        print(f"  box conventions: classes={pp.CLASS_NAMES}, YAW_SIGN={yaw_sign}, YAW_OFFSET={yaw_offset}")
        wm = WorldModel(pp.CLASS_NAMES, yaw_sign, yaw_offset, min_score=args.decision_score, dt=DT)

    # ---------------- world -------------------------------------------------
    print(f"\nconnecting to CARLA at {args.carla_host}:{args.carla_port} ...")
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(args.carla_timeout)
    client.load_world(args.town)
    world = client.get_world()
    weather, lights = apply_weather(world, weather_preset, weather_overrides)
    cmap = world.get_map()
    print(f"  map loaded: {cmap.name}")
    print(f"  weather: {args.weather}" + (f" + {weather_overrides}" if args.weather_set else '')
          + f" (sun {weather.get('sun_altitude_angle', 0):.0f} deg, rain {weather.get('precipitation', 0):.0f}, "
          f"fog {weather.get('fog_density', 0):.0f})" + (", headlights on" if lights else ''))
    # LiDAR rain / fog on top of CARLA's clean cloud (CARLA's LiDAR ignores the weather). Rain from the
    # preset's precipitation; fog only when we set it (Fog / --weather-set), not the presets' camera haze
    lidar_fx, lidar_r0 = None, lw.DEFAULT_RANGE
    if args.lidar_weather == 'auto':
        if detector:
            lidar_r0 = float(cfg.get('lidar_range', lw.DEFAULT_RANGE))
            prm = lw.profile(weather.get('precipitation', 0.0), weather_overrides.get('fog_density', 0.0))
            lidar_fx = prm if prm['active'] else None
            print(f"  LiDAR weather: {lw.describe(prm, lidar_r0)}")
        else:
            print(f"  LiDAR weather: ignored, --obstacle-source {args.obstacle_source} uses no LiDAR")
    bp = world.get_blueprint_library()
    original_settings = world.get_settings()
    if not args.keep_parked:
        disable_parked(world)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timing = args.timing if detector else None
    lead_dist = (args.lead_distance if args.lead_distance is not None
                 else 25.0 if args.scenario == 'route' and args.traffic > 0 else 0.0)
    run_name = (f"{args.scenario}_{args.obstacle_source}" + (f"_{timing}" if timing else '')
                + (f"_traffic{args.traffic}" if args.traffic > 0 else '')
                + ('_lead' if lead_dist > 0 else '')
                + (f"_walkers{args.walkers}" if args.walkers > 0 else '')
                + ('_lights' if args.lights != 'off' else '') + ('_xing' if args.predict_crossing else '')
                + (f"_{args.weather}" if args.weather != 'ClearNoon' or args.weather_set else '')
                + ('_custom' if args.weather_set else '') + ('_lidarwx' if lidar_fx else '') + f"_{stamp}")
    actors, sensors, queues, buffers = [], {}, {}, {}
    walker_ctrls, walkers, walker_pool = [], [], []
    stop_lines = []
    ev = boxcheck = obstacle = tm = lead = lead_info = rec = None
    extra, done_reason, k = {}, 'not started', 0
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        sps = cmap.get_spawn_points()
        if args.scenario == 'static':
            start_tf, goal_loc, obstacle_wp = pick_static(args, cmap, sps)
        else:
            start_tf, goal_loc, obstacle_wp = pick_route(args, cmap, sps)
        ebp = bp.find(EGO_BP)
        if ebp.has_attribute('role_name'):
            ebp.set_attribute('role_name', 'hero')
        ego = world.try_spawn_actor(ebp, start_tf)
        if ego is None:
            raise RuntimeError("could not spawn the ego at the chosen spawn point (occupied?)")
        actors.append(ego)
        ebb = ego.bounding_box
        ego_front = ebb.location.x + ebb.extent.x
        params.ego_half_width = ebb.extent.y
        if obstacle_wp is not None:
            otf = obstacle_wp.transform
            otf.location.z += 0.3
            obstacle = world.try_spawn_actor(bp.find(args.obstacle_bp), otf)
            if obstacle is None:
                raise RuntimeError(f"could not spawn {args.obstacle_bp} {args.obstacle_distance} m ahead")
            actors.append(obstacle)
            obstacle.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
        grp = GlobalRoutePlanner(cmap, args.route_resolution)
        wps = [w for w, _ in grp.trace_route(start_tf.location, goal_loc)]
        route = Route([[w.transform.location.x, w.transform.location.y] for w in wps], wps)
        print(f"  route: {route.length:.0f} m, {len(route.xy)} points")
        stop_lines = route_stop_lines(world, route, cmap)
        print(f"  traffic lights on the route: {len(stop_lines)}"
              + (" (the ego stops at red; light state from CARLA)" if args.lights == 'oracle'
                 else " (not obeyed: add --lights oracle; red lights run are counted)"))

        if args.traffic > 0 or lead_dist > 0:
            tm = client.get_trafficmanager(args.tm_port)
            tm.set_synchronous_mode(True)              # after world sync, as the TM docs require
            tm.set_random_device_seed(args.seed)
            clear = [start_tf.location] + ([obstacle_wp.transform.location] if obstacle_wp is not None else [])
            if lead_dist > 0:
                lead, lead_info, lead_loc = spawn_lead(world, bp, tm, route, lead_dist,
                                                       args.lead_speed_kmh, args.lead_bp)
                if lead is None:
                    print(f"  [warn] could not spawn the lead car about {lead_dist:.0f} m ahead "
                          f"(occupied?); try another --start-index")
                else:
                    actors.append(lead)
                    clear.append(lead_loc)
                    print(f"  lead car: {args.lead_bp} at {lead_info['start_s_m']:.0f} m along the route, "
                          f"{args.lead_speed_kmh:.0f} km/h, "
                          + ("driven along the ego route (TM set_path)" if lead_info['follows_ego_route']
                             else "[warn] this TM has no set_path: it may turn off the route at a junction"))
            if args.traffic > 0:
                traffic = spawn_traffic(world, bp, tm, sps, args.traffic, clear)
                actors += traffic
                print(f"  traffic: {len(traffic)}/{args.traffic} autopilot cars")
        if args.walkers > 0:
            clear = [start_tf.location] + ([obstacle_wp.transform.location] if obstacle_wp is not None else [])
            walkers, walker_pool = spawn_walkers(world, bp, args.walkers, args.walker_cross, np.asarray(route.xy),
                                                 clear, args.seed, walker_ctrls, actors)
            print(f"  pedestrians: {len(walkers)}/{args.walkers} within 30 m of the route "
                  f"({len(walker_pool)} sidewalk points found), {100 * args.walker_cross:.0f}% may cross the road anywhere"
                  + ("" if len(walkers) >= args.walkers else "  [fewer: not enough free sidewalk near the route]"))
        step3 = args.traffic > 0 or lead is not None
        if lights:
            headlights_on([a for a in actors if a.type_id.startswith('vehicle.') and a is not obstacle],
                          fog=weather.get('fog_density', 0.0) >= 20.0 and weather.get('fog_distance', 99.0) <= 20.0)

        meta = dict(run=run_name, town=args.town, map=cmap.name, seed=args.seed, scenario=args.scenario,
                    steer_target=args.steer_target, speed_control=args.speed_control,
                    lon_gains=args.lon_gains if args.speed_control == 'pi' else None,
                    obstacle_source=args.obstacle_source, timing=timing, speed_kmh=speed_kmh,
                    decide_on=args.decide_on if detector else args.obstacle_source,
                    traffic=args.traffic, lead_car=lead_info,
                    walkers=args.walkers, walker_cross=args.walker_cross if args.walkers > 0 else None,
                    weather=args.weather, weather_set=args.weather_set or None, weather_params=weather,
                    headlights=lights, lights=args.lights, predict_crossing=bool(args.predict_crossing),
                    lights_on_route=len(stop_lines),
                    lidar_weather=dict(lidar_fx, range_m=lidar_r0, effect=lw.describe(lidar_fx, lidar_r0))
                    if lidar_fx else None,
                    decision_score=args.decision_score if detector else None,
                    params=vars(params), latency_table=args.latency_table if detector else None)
        ev = Evaluator(args.out_dir, run_name, meta, route.xy, DT)
        ev.junction_fn = lambda x, y: cmap.get_waypoint(carla.Location(x=x, y=y), project_to_road=True).is_junction
        for name, cb in (('sensor.other.collision', ev.on_collision),
                         ('sensor.other.lane_invasion', ev.on_lane_invasion)):
            s = world.spawn_actor(bp.find(name), carla.Transform(), attach_to=ego)
            s.listen(cb)
            sensors[name] = s
            actors.append(s)

        if detector:
            ps_sensors, queues = dc.setup_sensors(world, ego, bp)
            sensors.update(ps_sensors)
            actors += list(ps_sensors.values())
            if args.drop_gt_cameras:
                for key in ('sem', 'depth'):
                    s = sensors.pop(key, None)
                    queues.pop(key, None)
                    if s is not None:
                        s.stop()
                        s.destroy()
                        actors.remove(s)
            if args.check_boxes:
                boxcheck = BoxCheck(ev.dir, cfg['img_w'], cfg['img_h'], cfg['fov'])

        for _ in range(30):                      # let the cars settle on the road
            world.tick()
        if obstacle is not None:
            ev.polygons.append(actor_obstacle(obstacle, 0).corners)
        if detector:
            dc.drain_queues(queues)
            f0 = world.tick()
            first = ps.decode_inputs(dc.grab_synced(queues, f0, buffers), f0, cfg)
            t0 = time.perf_counter()
            sched.warmup(first, iters=args.warmup_iters)
            print(f"  warm-up {time.perf_counter() - t0:.1f} s")
            dc.drain_queues(queues)
            buffers = {}

        ctrl = VehiclePIDController(ego,
                                    args_lateral={'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': DT},
                                    args_longitudinal={'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0, 'dt': DT},
                                    max_throttle=0.75, max_brake=0.5, max_steering=0.8)
        speed_ctrl = (SpeedController(*args.lon_gains, dt=DT) if args.speed_control == 'pi' else None)
        decider = Decider(params)
        spectator = world.get_spectator()
        if args.record:
            from recorder import Recorder
            rec_source = ({'oracle': 'oracle (CARLA ground truth)', 'none': 'none (perception off)'}
                          .get(args.obstacle_source, args.obstacle_source) + (f' ({timing})' if timing else '')
                          + (', decides on GT' if detector and args.decide_on == 'oracle' else ''))
            rec = Recorder(world, ego, bp, os.path.join(ev.dir, 'video.mp4'), DT, size=args.record_size,
                           view=args.record_view, title=f'{args.scenario} | obstacles: {rec_source}'
                           + (f' | {args.weather}' if args.weather != 'ClearNoon' or args.weather_set else '')
                           + (' + LiDAR weather' if lidar_fx else ''),
                           pip_fov=float(cfg['fov']) if detector else None)
            sensors['recorder'] = rec.sensor
            actors.append(rec.sensor)
        drop_cam, drop_lidar = ps.parse_range(args.drop_camera), ps.parse_range(args.drop_lidar)
        logged = set()
        has_moved = False
        complete_ticks = blocked_ticks = 0
        first_collision_k = None
        crash_k, n_coll_seen, v_prev, front_prev = None, 0, 0.0, None
        max_ticks = int(args.max_seconds / DT)
        print(f"\ndriving: {args.scenario}, obstacles from {args.obstacle_source}, "
              f"v0 {speed_kmh:.0f} km/h, up to {args.max_seconds:.0f} s\n")
        done_reason = 'time limit'

        for k in range(max_ticks):
            if walker_pool and k > 0 and k % 600 == 0:             # every 30 s, as the data collector
                retarget_walkers(walker_ctrls, walkers, walker_pool)
            t0 = time.perf_counter()
            frame = world.tick()
            t1 = time.perf_counter()
            snap = inputs = data = view = None
            healthy = True
            if detector:
                try:
                    data = dc.grab_synced(queues, frame, buffers)
                    inputs = ps.decode_inputs(data, frame, cfg)
                except dc.FrameDropped as e:
                    print(f"  [warn] {e}")
                if inputs is not None:
                    if ps.in_range(k, drop_cam):
                        inputs.bgr = inputs.rgb = None
                    if ps.in_range(k, drop_lidar):
                        inputs.lidar = None
                    if lidar_fx is not None and inputs.lidar is not None:
                        inputs.lidar = lw.degrade(inputs.lidar, lidar_fx, np.random.default_rng([args.seed, k]),
                                                  lidar_r0)
                snap = sched.step(inputs, frame=frame)
                healthy = snap.healthy
                view = snap.get('obstacles')
            t2 = time.perf_counter()

            tf = ego.get_transform()
            vel = ego.get_velocity()
            v = math.hypot(vel.x, vel.y)
            ego_xy = (tf.location.x, tf.location.y)
            s_ego, dev = route.update_progress(ego_xy)
            det_fresh = False
            need_gt = (args.obstacle_source == 'oracle' or step3 or
                       (detector and args.decide_on == 'oracle'))
            gt_now = gt_obstacles(world, ego, k) if need_gt else None
            if args.obstacle_source == 'oracle':
                obstacles, fresh = gt_now, True
            elif detector:
                obstacles, det_fresh = wm.update(view.out if view is not None else None, k)
                fresh = det_fresh
                if args.decide_on == 'oracle':           # detector still runs and is checked
                    obstacles, fresh = gt_now, True
            else:
                obstacles, fresh = [], True
            hits = assess(obstacles, route, params, ego_front)
            if args.predict_crossing:
                hits = sorted(hits + predict_crossing(obstacles, route, params, ego_front), key=lambda h: h.gap)
            # next stop line ahead (map) and its light (CARLA state): for the decision with
            # --lights oracle, and always for the red-light count and the video
            front = s_ego + ego_front
            light = light_ends = None
            for sl in stop_lines:
                if sl['s'] - front > -0.5:
                    if sl['s'] - front <= params.horizon + 20.0:
                        light = (sl['s'] - front, light_state(sl['light']), sl['id'])
                        light_ends = sl['ends']
                    break
            if front_prev is not None:
                for sl in stop_lines:
                    if front_prev < sl['s'] <= front and light_state(sl['light']) == 'Red':
                        ev.event(k, frame, 'red_light_run', light=sl['id'], v=round(v, 2))
            front_prev = front
            # a pedestrian hit while moving: stop for good, end the run 5 s later
            if len(ev._raw_collisions) > n_coll_seen:
                new = ev._raw_collisions[n_coll_seen:]
                n_coll_seen += len(new)
                if (crash_k is None and v_prev > 0.3
                        and any(str(c[2]).startswith('walker') for c in new)):
                    decider.crash('pedestrian')
                    crash_k = k
                    print(f"  [!] k {k}: the ego hit a pedestrian at {v_prev * 3.6:.0f} km/h: stopping for good")
            v_prev = v
            # ground-truth lead and TTC for the evaluator (never used for driving): cars only,
            # so pedestrians crossing do not count as 'the car ahead' (they have their own checks)
            gt_lead_gap = gt_ttc = None
            gt_lead_id = gt_lead_type = gt_lead_d = ''    # which GT car is ahead (diagnostic)
            ego_in_junction = cmap.get_waypoint(tf.location).is_junction  # 3D position
            junction_conflict = False   # nearest car in the path is CROSSING it, at a junction
            if gt_now is not None:
                gh = assess([o for o in gt_now if o.cls not in PEDESTRIAN_CLASSES], route, params, ego_front)
                if gh:
                    gt_lead_gap = gh[0].gap
                    closing = v - gh[0].v_lead
                    gt_ttc = gt_lead_gap / closing if closing > 0.1 and gt_lead_gap > 0 else None
                    ob = gh[0].obstacle
                    gt_lead_id, gt_lead_type = ob.track_id, GT_TYPE.get(ob.track_id, ob.cls)
                    _, dd, _, past = route.project(ob.center[None, :2], params.horizon, return_beyond=True)
                    gt_lead_d = '' if past[0] else round(float(dd[0]), 2)    # its centre, sideways
                    t_hat = route.tangent_at(gh[0].s_abs)
                    crossing = abs(t_hat[0] * math.sin(ob.yaw) - t_hat[1] * math.cos(ob.yaw)) > 0.5
                    if crossing:                               # 30-150 deg off the path direction
                        c = ob.center
                        junction_conflict = ego_in_junction or cmap.get_waypoint(
                            carla.Location(x=float(c[0]), y=float(c[1]), z=float(c[2]))).is_junction
            dec = decider.step(v, s_ego, route.remaining, route.speed_cap(params.a_lat_max, params.b),
                               hits, fresh, healthy, ego_front, DT,
                               light=light if args.lights == 'oracle' else None)
            look = 3.0 + 0.5 * v
            xy = (route.lookahead_point(ego_xy, look) if args.steer_target == 'pursuit'
                  else route.point_at(s_ego + look))
            control = ctrl.run_step(dec.v_target * 3.6, SteerTarget(xy, tf.location.z))
            if speed_ctrl is not None:
                control.throttle, control.brake = speed_ctrl.step(dec.v_target, v, dec.a_cmd,
                                                                  dec.hold_brake)
            control = shape_control(control, dec)
            ego.apply_control(control)
            t3 = time.perf_counter()

            if boxcheck is not None and det_fresh and view is not None and view.out is not None:
                boxcheck.run(k, wm.base, gt_obstacles(world, ego, k, radius=80.0), ego_xy,
                             inputs.bgr if inputs is not None else None,
                             data['rgb'].transform if data is not None else None,
                             view.out.lidar_tf, v > 0.5)

            gt_gap = None
            if obstacle is not None:
                _, _, s_abs = route.project(actor_obstacle(obstacle, k).corners, horizon=500.0)
                gt_gap = float(s_abs.min() - s_ego - ego_front)

            # ---- events (what the M4 log must show) ----
            for e in decider.events:
                ev.event(k, frame, e.pop('type'), **e)
            decider.events.clear()
            if v > 1.0:
                has_moved = True
            if hits and 'corridor' not in logged:
                logged.add('corridor')
                ev.event(k, frame, 'obstacle_in_corridor', gap=round(hits[0].gap, 2),
                         cls=hits[0].obstacle.cls, source=hits[0].obstacle.source,
                         gt_gap=None if gt_gap is None else round(gt_gap, 2))
            if has_moved and control.brake >= 0.05 and 'brake' not in logged:
                logged.add('brake')
                ev.event(k, frame, 'first_brake', brake=round(control.brake, 2), state=dec.state,
                         gt_gap=None if gt_gap is None else round(gt_gap, 2))

            ages = {n: ('' if sv.age_ticks is None else sv.age_ticks)
                    for n, sv in (snap.views.items() if snap is not None else [])}
            ev.tick(k=k, frame=frame, t=round(k * DT, 2), x=round(tf.location.x, 3),
                    y=round(tf.location.y, 3), yaw=round(tf.rotation.yaw, 2), speed=round(v, 3),
                    s=round(s_ego, 2), completion=round(route.completion, 4), lat_dev=round(dev, 3),
                    state=dec.state, v_target_kmh=round(dec.v_target * 3.6, 2), v0_kmh=round(dec.v0 * 3.6, 1),
                    a_cmd=round(dec.a_cmd, 3), throttle=round(control.throttle, 3),
                    steer=round(control.steer, 3), brake=round(control.brake, 3),
                    n_obstacles=len(obstacles), n_in_path=len(hits),
                    lead_gap='' if dec.lead_gap is None else round(dec.lead_gap, 2),
                    lead_ttc='' if dec.lead_ttc is None or math.isinf(dec.lead_ttc) else round(dec.lead_ttc, 2),
                    gt_gap='' if gt_gap is None else round(gt_gap, 2), healthy=int(healthy),
                    gt_lead_gap='' if gt_lead_gap is None else round(gt_lead_gap, 2),
                    gt_ttc='' if gt_ttc is None else round(gt_ttc, 2),
                    gt_lead_id=gt_lead_id, gt_lead_type=gt_lead_type, gt_lead_d=gt_lead_d,
                    ego_in_junction=int(bool(ego_in_junction)), junction_conflict=int(junction_conflict),
                    lead_cls=dec.lead_kind, light_state='' if light is None else light[1],
                    light_gap='' if light is None else round(light[0], 2),
                    obstacles_age=ages.get('obstacles', ''), lights_age=ages.get('lights', ''),
                    server_ms=round((t1 - t0) * 1000, 1), perception_ms=round((t2 - t1) * 1000, 1),
                    decide_ms=round((t3 - t2) * 1000, 1))

            if rec is not None and rec.writer is not None:
                try:
                    rec.frame(frame, k * DT, dec, v, control, obstacles, hits, route, s_ego, ego_xy,
                              len(ev._raw_collisions), source=rec_source,
                              pip=inputs.bgr if inputs is not None else None,
                              pip_tf=data['rgb'].transform if data is not None else None,
                              camera_expected=bool(detector),
                              light=None if light is None else (light[1], light[0], light_ends,
                                                                args.lights == 'oracle'))
                except Exception as e:                   # the video must never stop the run
                    print(f"  [warn] video recording stopped: {type(e).__name__}: {e}")
                    rec.writer.release()
                    rec.writer = None
                    rec.sensor.stop()                    # no more frames piling up in its queue
            if args.spectator:
                follow_spectator(spectator, tf)
            if (k + 1) % args.print_every == 0:
                print(f"k {k:5d} | {dec.state:14s} | v {v * 3.6:5.1f} km/h -> {dec.v_target * 3.6:5.1f} | "
                      f"s {s_ego:6.1f}/{route.length:.0f} m dev {dev:4.2f} | in path {len(hits)}"
                      + (f" gap {dec.lead_gap:5.1f}" if dec.lead_gap is not None else "")
                      + (f" | GT gap {gt_gap:5.1f}" if gt_gap is not None else ""))

            # ---- stop conditions ----
            if crash_k is not None and k - crash_k >= int(5.0 / DT):
                done_reason = 'stopped after hitting a pedestrian'
                break
            complete_ticks = complete_ticks + 1 if dec.state == ROUTE_COMPLETE and v < 0.1 else 0
            if complete_ticks >= 20:
                done_reason = 'route complete'
                break
            blocked_ticks = blocked_ticks + 1 if dec.state == BLOCKED_WAIT else 0
            if args.scenario == 'static' and blocked_ticks * DT >= args.wait_after_stop:
                done_reason = 'stopped for the obstacle'
                break
            if ev.has_collision and not step3:           # traffic runs keep going: every collision
                first_collision_k = k if first_collision_k is None else first_collision_k   # is classified
                if k - first_collision_k >= 40:
                    done_reason = 'collision'
                    break
            if dev > 30.0:
                done_reason = 'left the route by > 30 m'
                break
    except KeyboardInterrupt:
        done_reason = 'interrupted'
        print("\ninterrupted; cleaning up and scoring what ran ...")
    except Exception as e:                       # e.g. a CARLA RPC timeout: clean up, score, re-raise
        done_reason = f'error: {type(e).__name__}: {e}'
        print(f"\n{done_reason}\ncleaning up and scoring what ran ...")
        raise
    finally:
        def _safe(what, fn):
            try:
                fn()
            except BaseException as e:          # incl. Ctrl+C pressed again: skip this step only
                print(f"  [warn] {what}: {type(e).__name__} {e}")

        for sen in sensors.values():
            _safe('could not stop a sensor', lambda sen=sen: sen.stop() if sen.is_alive else None)

        # score first: pure Python, so the results are saved even if CARLA hangs below.
        # If scoring itself fails (or Ctrl+C lands in it), CARLA is still restored first.
        summary, pending = None, None
        if ev is not None:
            try:
                extra['end_reason'] = done_reason
                if args.scenario == 'static':
                    gaps = [r['gt_gap'] for r in ev.rows if r['gt_gap'] != '']
                    extra.update(obstacle_source=args.obstacle_source, obstacle_bp=args.obstacle_bp,
                                 obstacle_distance_m=args.obstacle_distance,
                                 stopped=done_reason == 'stopped for the obstacle',
                                 final_gt_gap_m=gaps[-1] if gaps else None,
                                 min_gt_gap_m=min(gaps) if gaps else None)
                summary = ev.finish(args.scenario, extra)
            except BaseException as e:
                pending = e
                print(f"  [warn] scoring failed ({type(e).__name__}: {e}); restoring CARLA first")

        for c in walker_ctrls:
            _safe('could not stop a pedestrian', lambda c=c: c.stop() if c.is_alive else None)
        alive = []

        def _collect():
            for a in walker_ctrls + actors:          # controllers before their pedestrians
                if a.is_alive:
                    alive.append(a)
        _safe('could not list actors', _collect)
        if alive:
            _safe('could not destroy all actors',
                  lambda: client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in alive], True))
        if tm is not None:
            _safe('could not set the Traffic Manager back to async', lambda: tm.set_synchronous_mode(False))
        _safe('could not restore world settings', lambda: world.apply_settings(original_settings))
        if rec is not None:
            _safe('could not finish the video', rec.close)

        if summary is not None:
            print_summary(summary, ev.dir)
        if boxcheck is not None:
            print_boxcheck(boxcheck.summary(wm.yaw_sign, wm.yaw_offset), boxcheck.dir)
        if pending is not None:
            raise pending


def print_summary(s, out_dir):
    bar = '=' * 78
    print(f"\n{bar}\n{s.get('milestone', s['scenario'])}: {'PASS' if s.get('passed') else 'FAIL'}"
          f"   ({s.get('end_reason')})\n{bar}")
    print(f"  {'map':40s} {s.get('meta', {}).get('map')}   steer={s.get('meta', {}).get('steer_target')}"
          f"  speed control={s.get('meta', {}).get('speed_control')}")
    if s.get('meta', {}).get('lidar_weather'):
        print(f"  {'LiDAR weather':40s} {s['meta']['lidar_weather'].get('effect')}")
    for name in ('sim_seconds', 'route_completion_pct', 'max_lateral_dev_m', 'mean_lateral_dev_m',
                 'steer_reversals_per_min', 'max_speed_kmh', 'final_gt_gap_m', 'min_gt_gap_m',
                 'rms_jerk', 'hard_brake_events', 'pedal_back_and_forth', 'direct_throttle_brake',
                 'brake_while_cruising_pct', 'lead_following_s', 'min_gt_lead_gap_m',
                 'median_time_headway_s', 'stops_behind_a_car',
                 'ego_caused_collisions_outside_junctions', 'ego_caused_collisions_in_junctions',
                 'not_ego_caused_collisions', 'emergency_stops_outside_junctions',
                 'emergency_stops_in_junctions', 'min_gt_ttc_outside_junctions_s',
                 'min_gt_ttc_in_junctions_s', 'ttc_below_1_5s_events_outside_junctions',
                 'emergency_stops_for_pedestrians', 'pedestrian_hits_by_ego', 'red_lights_run',
                 'infraction_penalty', 'project_score'):
        if name in s:
            print(f"  {name[:40]:40s} {s[name]}")
    print(f"  {'collisions':40s} {len(s.get('collisions', []))}")
    for c in s.get('collisions', []):
        where = ('pedestrian: always gated' if c.get('category') == 'pedestrian' and c.get('ego_caused') else
                 'junction conflict, not gated' if c.get('junction_conflict') else
                 {True: 'in junction, gated', False: 'outside junction', None: 'junction unknown'}[c.get('in_junction')])
        print(f"    frame {c['frame']}: {c['other_type']}, {c.get('fault')}, {where}")
    if s.get('note'):
        print(f"  note: {s['note']}")
    print(f"  {'states (ticks)':40s} {s.get('state_ticks')}")
    for name, ok in s.get('checks', {}).items():
        print(f"  [{'x' if ok else ' '}] {name}")
    print(f"  files: {out_dir}  (ticks.csv, events.jsonl, summary.json, trajectory.png)\n{bar}")


def print_boxcheck(s, out_dir):
    print(f"\nM1 BOX CHECK: {s.get('status', 'PASS' if s['passed'] else 'FAIL')}\n  gate: {s['gate']}")
    for key in ('records', 'observations', 'distinct_actors', 'duplicate_boxes', 'false_positives',
                'false_positive_classes', 'gt_within_30m', 'gt_within_30m_missed',
                'median_center_err_m', 'p90_center_err_m', 'median_err_forward_m', 'median_err_left_m',
                'median_yaw_err_deg_vehicles', 'median_z_err_m', 'median_length_err_m',
                'median_width_err_m', 'mirror_votes', 'mirror_test', 'yaw_test', 'images'):
        print(f"  {key:30s} {s[key]}")
    print("  by detected class:")
    for c, d in s['by_detected_class'].items():
        print(f"    {c:12s} n={d['n']:<4d} centre err {d['median_center_err_m']}  "
              f"box l x w {d['median_det_l']} x {d['median_det_w']}  yaw err {d['median_yaw_err_deg']}")
    print(f"  overlays + box_check.json (every record) in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--carla-root', default=None, help='CARLA folder containing PythonAPI (or set CARLA_ROOT)')
    p.add_argument('--scenario', choices=['route', 'static'], default='route')
    p.add_argument('--obstacle-source', choices=['none', 'oracle', 'lidar', 'fusion'], default='none')
    p.add_argument('--check-boxes', action='store_true', help='M1 gate: detector boxes vs CARLA boxes')
    p.add_argument('--town', default='Town05', help='Town05 = the validation town in the plan')
    p.add_argument('--weather', default='ClearNoon',
                   help='CARLA weather preset (ClearNoon, HardRainNoon, ClearNight, HardRainNight, WetCloudyNoon, '
                        'SoftRainSunset, ...) or Fog. Training used ClearNoon and CloudySunset')
    p.add_argument('--weather-set', default='',
                   help="change weather fields on top of --weather, e.g. 'fog_density=80,fog_distance=5'")
    p.add_argument('--lidar-weather', choices=['off', 'auto'], default='off',
                   help="auto: rain / fog also degrade the LiDAR (lidar_weather.py; CARLA's LiDAR ignores "
                        "weather). Same points for LiDAR-only and fusion. off = CARLA's clean cloud")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--speed-kmh', type=float, default=None, help='default 30 (route), 20 (static)')
    p.add_argument('--start-index', type=int, default=0, help='spawn point index to start from / search from')
    p.add_argument('--goal-index', type=int, default=None, help='route scenario: goal spawn point index')
    p.add_argument('--route-length', type=float, default=300.0, help='route scenario without --goal-index')
    p.add_argument('--route-resolution', type=float, default=1.0)
    p.add_argument('--obstacle-distance', type=float, default=60.0)
    p.add_argument('--after-obstacle', type=float, default=30.0, help='route continues this far past it')
    p.add_argument('--obstacle-bp', default='vehicle.audi.a2')
    p.add_argument('--wait-after-stop', type=float, default=5.0, help='static: seconds held before ending')
    p.add_argument('--max-seconds', type=float, default=90.0)
    p.add_argument('--keep-parked', action='store_true', help='keep map-baked parked cars')
    p.add_argument('--traffic', type=int, default=0, help='autopilot cars to spawn (M1: 30 or more)')
    p.add_argument('--tm-port', type=int, default=8000)
    p.add_argument('--lead-distance', type=float, default=None,
                   help='step 3: a car this many m ahead on the ego route (default 25 with --traffic, '
                        'else off; 0 = off)')
    p.add_argument('--lead-speed-kmh', type=float, default=18.0, help='slower than the ego, so it must follow')
    p.add_argument('--lead-bp', default='vehicle.lincoln.mkz_2020')
    p.add_argument('--walkers', type=int, default=0,
                   help='pedestrians on the sidewalks near the route (video / stress runs; the ego has no '
                        'red-light stop and no crossing prediction yet)')
    p.add_argument('--lights', choices=['off', 'oracle'], default='off',
                   help='oracle: stop at red lights. Stop lines from the map, light state from CARLA '
                        '(the YOLO model finds traffic lights but does not classify their colour)')
    p.add_argument('--predict-crossing', action='store_true',
                   help='brake for pedestrians walking towards the path who will be in it within 2 s')
    p.add_argument('--walker-cross', type=float, default=0.2,
                   help='share of pedestrians (0-1) that may cross the road anywhere, not only at crossings')
    p.add_argument('--decide-on', choices=['detector', 'oracle'], default='detector',
                   help='oracle: drive on CARLA ground truth while the detector runs and is checked '
                        '(M1 only; never an evaluated run)')
    p.add_argument('--decision-score', type=float, default=0.3,
                   help='min detector score the planner acts on (heuristic)')
    p.add_argument('--yaw-sign', type=float, default=None, help='override pp_common.YAW_SIGN')
    p.add_argument('--yaw-offset', type=float, default=None, help='override pp_common.YAW_OFFSET (rad)')
    p.add_argument('--rescore-box-check', default='', metavar='RUNS_FOLDER',
                   help='re-apply the M1 verdict rules to every saved box check in this folder (no CARLA needed)')
    p.add_argument('--steer-target', choices=['pursuit', 'path'], default='pursuit',
                   help='pursuit = point at straight-line distance L (smooth); path = s_ego + L (first M2 run)')
    p.add_argument('--speed-control', choices=['pi', 'carla'], default='pi',
                   help="pi = SpeedController below; carla = VehiclePIDController's own pedals (first M2 run)")
    p.add_argument('--lon-gains', type=float, nargs=3, default=[0.5, 0.3, 0.1], metavar=('KP', 'KI', 'KFF'),
                   help='SpeedController gains: per m/s, per m, per m/s^2')
    p.add_argument('--spectator', action='store_true', help='move the CARLA window camera behind the ego')
    p.add_argument('--record', action='store_true',
                   help='presentation video: <run folder>/video.mp4 at real speed (see recorder.py)')
    p.add_argument('--record-view', default='chase', choices=['chase', 'top'],
                   help='chase = behind and above the car, top = straight down')
    p.add_argument('--record-size', type=int, nargs=2, default=[1280, 720], metavar=('W', 'H'))
    p.add_argument('--print-every', type=int, default=20)
    p.add_argument('--carla-host', default='localhost')
    p.add_argument('--carla-port', type=int, default=2000)
    p.add_argument('--carla-timeout', type=float, default=30.0)
    p.add_argument('--out-dir', default='runs')
    ps.add_model_args(p)
    g = p.add_argument_group('perception scheduler (same meaning as in perception_scheduler.py)')
    g.add_argument('--timing', choices=['lockstep', 'emulated'], default='lockstep')
    g.add_argument('--latency-table', default=None)
    g.add_argument('--latency-quantile', choices=['p95', 'median'], default='p95')
    g.add_argument('--obstacle-period', type=int, default=1)
    g.add_argument('--lights-period', type=int, default=4)
    g.add_argument('--lane-period', type=int, default=4)
    g.add_argument('--slack-ticks', type=int, default=2)
    g.add_argument('--camera-loss', choices=['stale', 'unpainted'], default='stale')
    g.add_argument('--warmup-iters', type=int, default=5)
    g.add_argument('--lenient', action='store_true')
    g.add_argument('--drop-camera', default=None, metavar='A:B')
    g.add_argument('--drop-lidar', default=None, metavar='A:B')
    g.add_argument('--drop-gt-cameras', action='store_true')
    args = p.parse_args()
    if args.obstacle_source == 'fusion' and not args.seg_model:
        p.error('--seg-model is required for --obstacle-source fusion')
    return args


if __name__ == '__main__':
    run(parse_args())