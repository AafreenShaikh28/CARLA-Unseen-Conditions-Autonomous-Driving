"""
test_driving_logic.py -- offline checks for route.py, world_model.py, decision.py, manoeuvre.py,
lane_walkers.py.
No CARLA, no GPU:   python test_driving_logic.py

The closed-loop tests drive a point-mass car (speed follows the target with a
0.4 s lag, emergency = -8 m/s^2) through the M4 scenario: a parked car 60 m
ahead, visible to "perception" only inside the detector's 51.2 m range.
They test the decision logic, not CARLA's vehicle physics; M4 in CARLA is
still the real test.
"""
import math
import sys
from types import SimpleNamespace

import numpy as np

from decision import (Params, Decider, assess, idm, predict_crossing, FOLLOW_ROUTE, BLOCKED_WAIT,
                      EMERGENCY_STOP, DEGRADED, ROUTE_COMPLETE, STOP_FOR_LIGHT, CRASH_STOP)
from control import SpeedController
from route import Route
from world_model import Obstacle, WorldModel, footprint, lidar_boxes_to_world

DT = 0.05
FAILS, N = [], [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        FAILS.append(msg)
        print(f"  FAIL  {msg}")


def carla_matrix(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    """Same formula as carla.Transform.get_matrix() (degrees in, local->world)."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
            [sp, -cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0]]


class FakeTf:
    def __init__(self, *a, **kw):
        self.m = carla_matrix(*a, **kw)

    def get_matrix(self):
        return self.m


def straight(length=200.0):
    xs = np.arange(0.0, length + 1e-9, 1.0)
    return Route(np.c_[xs, np.zeros_like(xs)])


def static_car(cx, cy=0.0, l=4.5, w=1.9, yaw=0.0):
    return Obstacle('Car', 0.9, np.array([cx, cy, 0.8]), footprint(cx, cy, l, w, yaw),
                    (l, w, 1.5), yaw)


# ------------------------------------------------------------------ route
def test_route():
    print("route")
    r = straight(100)
    s, dev = r.update_progress((10.3, 0.7))
    check(abs(s - 10.3) < 1e-6 and abs(dev - 0.7) < 1e-6, f"progress on a straight: s={s}, dev={dev}")
    s_rel, d, _ = r.project(np.array([[30.0, 1.5], [30.0, -1.5]]))
    check(np.allclose(s_rel, 19.7) and np.allclose(np.abs(d), 1.5) and d[0] * d[1] < 0,
          f"projection s_rel={s_rel}, d={d}")
    R = 30.0
    th = np.arange(0, np.pi, 1.0 / R)
    circ = Route(np.c_[R * np.sin(th), R - R * np.cos(th)])
    circ.update_progress(circ.xy[10])
    cap = circ.speed_cap(a_lat_max=2.0, b=2.0)
    check(abs(cap - math.sqrt(2.0 * R)) < 0.5, f"curve cap on R=30 m: {cap:.2f} vs {math.sqrt(60):.2f}")
    check(straight(200).speed_cap() == 50.0, "no cap on a straight")
    xs = np.arange(0.0, 120.0, 1.0)                         # trace_route-style near-duplicate point
    pts = np.r_[np.c_[xs[:41], np.zeros(41)], [[40.005, 0.003]], np.c_[xs[41:], np.zeros(len(xs) - 41)]]
    rd = Route(pts)
    caps = []
    for x in np.arange(10.0, 45.0, 1.0):
        rd.update_progress((x, 0.0))
        caps.append(rd.speed_cap())
    check(min(caps) == 50.0 and len(rd.xy) == len(xs), f"near-duplicate point dropped, no false curve "
                                                         f"(min cap {min(caps):.1f} m/s)")
    check(np.allclose(rd.xy[-1], pts[-1]), "route still ends exactly at the goal")


# ------------------------------------------------------------- box -> world
def test_boxes():
    print("detector boxes -> world")
    tf = FakeTf(100.0, 50.0, 2.4, yaw=30.0)
    box = [[10.0, 2.0, -1.6, 4.5, 1.9, 1.5, 0.2]]          # trained frame: 2 m to the LEFT
    c, corners, yaw = lidar_boxes_to_world(box, tf.get_matrix())
    # by hand: y-left (10, 2) -> CARLA sensor (10, -2) -> rotate 30 deg, add (100, 50)
    exp = (100 + 10 * math.cos(math.radians(30)) + 2 * math.sin(math.radians(30)),
           50 + 10 * math.sin(math.radians(30)) - 2 * math.cos(math.radians(30)))
    check(np.allclose(c[0, :2], exp, atol=1e-6), f"centre {c[0, :2]} vs hand {exp}")
    check(abs(c[0, 2] - 0.8) < 1e-6, f"centre z {c[0, 2]} vs 0.8")
    check(abs(yaw[0] - (math.radians(30) - 0.2)) < 1e-6, f"world yaw {yaw[0]:.4f} vs {math.radians(30) - 0.2:.4f}")
    e = np.linalg.norm(np.diff(np.vstack([corners[0], corners[0][:1]]), axis=0), axis=1)
    check(np.allclose(sorted(e), [1.9, 1.9, 4.5, 4.5], atol=1e-6), f"footprint edges {e}")
    c2, _, yaw2 = lidar_boxes_to_world([[10, 2, -1.6, 4.5, 1.9, 1.5, -0.2 + math.pi / 2]],
                                       tf.get_matrix(), yaw_sign=-1.0, yaw_offset=math.pi / 2)
    check(np.allclose(c2, c) and abs(yaw2[0] - yaw[0]) < 1e-6, "YAW_SIGN/YAW_OFFSET are undone")
    tilt = FakeTf(0, 0, 2.4, pitch=5.0)                     # nose-down ramp
    c3, _, _ = lidar_boxes_to_world([[20, 0, 0, 4, 2, 1.5, 0]], tilt.get_matrix())
    check(abs(c3[0, 2] - (2.4 + 20 * math.sin(math.radians(5)))) < 1e-6, f"pitch handled: z={c3[0, 2]:.3f}")


def test_world_model():
    print("world model")
    tf0 = FakeTf(0, 0, 2.4)
    wm = WorldModel(['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'], min_score=0.3, dt=DT)
    rng = np.random.default_rng(0)
    for k in range(20):                                     # car driving +x at 2 m/s
        x = 20 + 2.0 * k * DT + rng.normal(0, 0.05)
        out = SimpleNamespace(value={'boxes': np.array([[x, 0, -1.6, 4.5, 1.9, 1.5, 0],
                                                        [0.3, 0, -1.6, 4.5, 2, 1.5, 0],   # the ego itself
                                                        [30, 5, -1.6, 4.5, 1.9, 1.5, 0]]),
                                     'scores': np.array([0.9, 0.9, 0.2]), 'cls': np.array([0, 0, 1])},
                              capture_k=k, lidar_tf=tf0, source='lidar')
        obs, fresh = wm.update(out, k)
    check(fresh and len(obs) == 1, f"ego box and low-score box dropped, kept {len(obs)}")
    check(abs(obs[0].velocity[0] - 2.0) < 0.3 and abs(obs[0].velocity[1]) < 0.3,
          f"velocity from the 1 s fit: {obs[0].velocity}")
    obs2, fresh2 = wm.update(out, 22)                       # same output (captured k=19), 3 ticks later
    want = obs[0].velocity[0] * 3 * DT
    check(not fresh2 and abs(obs2[0].center[0] - obs[0].center[0] - want) < 1e-9,
          "stale output is predicted forward by v * age")


# --------------------------------------------------------------- corridor
def test_corridor():
    print("corridor")
    r = straight(200)
    r.update_progress((0, 0))
    p = Params()
    cases = [
        ("car in lane", static_car(40), True),
        ("car in the next lane (d=3.5)", static_car(40, 3.5), False),
        ("wide truck, centre at 2.0 m but edge inside", static_car(40, 2.0, 8.0, 2.6), True),
        ("bus across the road, corners at +-5 m", static_car(40, 0.0, 10.0, 2.5, math.pi / 2), True),
        ("car behind", static_car(-20), False),
        ("car beyond the horizon", static_car(90), False),
    ]
    for name, ob, want in cases:
        got = bool(assess([ob], r, p, 2.4))
        check(got == want, f"{name}: in_path={got}, expected {want}")
    h = assess([static_car(40)], r, p, 2.4)[0]
    check(abs(h.gap - (40 - 2.25 - 2.4)) < 1e-6, f"gap is bumper to edge: {h.gap:.2f}")
    # a car BESIDE the ego (CARLA run, k 883): its rear 0.8 m behind the ego's centre, inner
    # edge 0.2 m into the safety margin. The old check called it 'ahead' with gap -3.21 m
    # and commanded full braking. Braking cannot help with a car alongside: not in path.
    r10 = straight(200)
    r10.update_progress((10.0, 0.0))                       # ego mid-route, so nothing clamps at s = 0
    for side in (+1, -1):
        check(not assess([static_car(11.44, side * 2.15)], r10, p, 2.4),
              f"car alongside, reaching into the margin ({'left' if side > 0 else 'right'}): not in path")
    cut = assess([static_car(4.5, 1.2, yaw=-0.436)], r, p, 2.4)      # angled 25 deg into our lane
    check(len(cut) == 1 and 0.0 <= cut[0].gap < 2.0,
          f"car cutting in from beside, nose inside our width: in path, small gap "
          f"({cut[0].gap:.2f} m)" if cut else "car cutting in from beside: NOT detected")
    m = assess([static_car(20, 2.3)], r, p, 2.4)                      # wholly ahead, edge in margin
    check(len(m) == 1 and abs(m[0].gap - (20 - 2.25 - 2.4)) < 1e-6,
          "car ahead reaching into the margin: in path (the margin still applies ahead)")
    every = assess([static_car(x, y, yaw=a) for x in (-3, 0, 2, 5, 12) for y in (-2.5, -1.5, 0, 1.5, 2.5)
                    for a in (0.0, 0.5, -0.5, 1.57)], r, p, 2.4)
    check(all(x.gap >= 0.0 for x in every), f"gap is never negative ({len(every)} hits checked)")
    # near the route end: a car straight ahead PAST the end used to snap onto the end point
    # (left/right corners get opposite signs -> 'in path', parked at the goal)
    for x in range(0, 171, 10):                             # progress is forward-only, 60-point window
        r.update_progress((float(x), 0.0))
    check(abs(r.s_ego - 170.0) < 1e-6, f"ego walked to s={r.s_ego:.1f}")
    check(not assess([static_car(215)], r, p, 2.4), "car straight ahead past the route end: not in path")
    h_end = assess([static_car(200)], r, p, 2.4)
    check(len(h_end) == 1 and abs(h_end[0].gap - (200 - 2.25 - 170 - 2.4)) < 1e-6,
          "car straddling the route end: in path, gap from its rear")


# ------------------------------------------------------------- hysteresis
def test_hysteresis():
    print("hysteresis")
    p = Params(v0_kmh=20)
    r = straight(200)
    r.update_progress((0, 0))
    hit = assess([static_car(40)], r, p, 2.4)
    d = Decider(p)
    for occ in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]:          # 1 in 5: never confirmed
        d._hysteresis(hit if occ else [], True, DT)
    check(d.latched is None, "isolated detections never confirm")
    d = Decider(p)
    seq = [1, 1] + [1] + [0] * 9
    for occ in seq:
        d._hysteresis(hit if occ else [], True, DT)
    check(d.latched is not None, "3 of 5 confirms, and 9 clear frames do not release")
    d._hysteresis([], True, DT)
    check(d.latched is None, "10th clear frame releases")


# ------------------------------------------------------------ closed loop
def drive(obstacle_x=60.0, visible_range=51.2, period=1, latency=0, perception=True,
          lead_speed=0.0, healthy_fn=None, v0_kmh=20.0, seconds=40.0, route_len=200.0):
    """Point-mass closed loop. Returns dict with final gap, collision, states seen."""
    p = Params(v0_kmh=v0_kmh)
    r = straight(route_len)
    dec = Decider(p)
    x, v, ego_front = 0.0, 0.0, 2.4
    ox = obstacle_x
    pending = []
    collided, min_gap, states = False, 1e9, []
    for k in range(int(seconds / DT)):
        if ox is not None:
            ox += lead_speed * DT
        s_ego, _ = r.update_progress((x, 0.0))
        fresh, hits = False, []
        if perception and ox is not None and k % period == 0 and ox - x <= visible_range:
            pending.append((k + latency, ox))                 # captured now, visible later
        ready = [o for kk, o in pending if kk <= k]
        if ready:
            pending = [(kk, o) for kk, o in pending if kk > k]
            ob = static_car(ready[-1])
            ob.velocity = np.array([lead_speed, 0.0])
            ob = ob.shifted(ob.velocity * latency * DT)       # world model predicts to now
            hits = assess([ob], r, p, ego_front)
            fresh = True
        elif perception and k % period == 0 and latency == 0:
            fresh = True                                     # an output arrived, nothing in it
        healthy = True if healthy_fn is None else healthy_fn(k)
        d = dec.step(v, s_ego, r.remaining, r.speed_cap(p.a_lat_max, p.b), hits, fresh,
                     healthy, ego_front, DT)
        states.append(d.state)
        if d.emergency:
            acc = -8.0
        elif d.hold_brake:
            acc = -4.0 if v > 0 else 0.0
        else:
            acc = float(np.clip((d.v_target - v) / 0.4, -5.0, 2.5))
            if d.a_cmd < -3.0:                       # agent.py's brake feed-forward
                acc = min(acc, d.a_cmd)
        v = max(0.0, v + acc * DT)
        x += v * DT
        if ox is not None:
            gap = (ox - 2.25) - (x + ego_front)
            min_gap = min(min_gap, gap)
            if gap <= 0:
                collided = True
                break
    return dict(gap=None if ox is None else (ox - 2.25) - (x + ego_front), min_gap=min_gap,
                collided=collided, v=v, x=x, states=states, final=states[-1])


def test_closed_loop():
    print("closed loop (point mass)")
    r = drive()
    check(not r['collided'] and r['gap'] >= 2.0 and r['v'] < 0.05 and r['final'] == BLOCKED_WAIT,
          f"M4, lockstep: gap={r['gap']:.2f} m, collided={r['collided']}, final={r['final']}")
    check(EMERGENCY_STOP not in r['states'], "M4 lockstep: a smooth IDM stop, no emergency brake")
    r2 = drive(period=2, latency=2)
    check(not r2['collided'] and r2['gap'] >= 2.0 and r2['final'] == BLOCKED_WAIT,
          f"M4, emulated fusion (every 2 ticks, 2 ticks late): gap={r2['gap']:.2f} m")
    r2b = drive(visible_range=1000.0)
    check(not r2b['collided'] and r2b['gap'] >= 2.0 and r2b['final'] == BLOCKED_WAIT
          and r2b['x'] > 40.0,
          f"obstacle visible from standstill (oracle case): drives up and stops, x={r2b['x']:.1f}, "
          f"gap={r2b['gap']:.2f}")
    r3 = drive(perception=False)
    check(r3['collided'], "M4, perception off: must collide (causality)")
    r4 = drive(visible_range=15.0)
    check(not r4['collided'] and EMERGENCY_STOP not in r4['states'] and r4['v'] < 0.05,
          f"first seen at 15 m (TTC 1.9 s): hard IDM stop, no emergency, gap={r4['gap']:.2f}")
    r4b = drive(visible_range=12.0)
    check(not r4b['collided'] and EMERGENCY_STOP in r4b['states'],
          f"first seen at 12 m (TTC 1.3 s): emergency stop, gap={r4b['gap']:.2f}")
    r5 = drive(obstacle_x=30.0, lead_speed=3.0, v0_kmh=30.0, seconds=60.0, route_len=400.0)
    check(not r5['collided'] and r5['min_gap'] > 3.0,
          f"following a 3 m/s leader: min gap {r5['min_gap']:.2f} m, final gap {r5['gap']:.2f} m")
    r6 = drive(healthy_fn=lambda k: not (200 <= k < 260), obstacle_x=None, seconds=30.0)
    st = r6['states']
    check(st[200] == DEGRADED or st[201] == DEGRADED, "perception unhealthy -> DEGRADED")
    check(all(s == DEGRADED for s in st[201:269]) and st[275] == FOLLOW_ROUTE,
          "stays DEGRADED until 10 healthy ticks, then FOLLOW_ROUTE")
    r7 = drive(obstacle_x=None, route_len=100.0, v0_kmh=30.0, seconds=60.0)
    check(r7['final'] == ROUTE_COMPLETE and r7['v'] < 0.05 and abs(r7['x'] - 100.0) < 3.0,
          f"route end: stopped at x={r7['x']:.2f} (route 100 m), state {r7['final']}")
    check(r7['states'][0] == DEGRADED and FOLLOW_ROUTE in r7['states'], "starts held, then drives")


def test_idm():
    print("IDM")
    p = Params(v0_kmh=36)
    check(abs(idm(10.0, 10.0, p)) < 1e-9, "no acceleration at v0 on a free road")
    check(idm(0.0, 10.0, p) == p.a_max, "full comfortable acceleration from rest")
    check(idm(10.0, 10.0, p, gap=5.0, v_lead=0.0) < -p.b, "closing fast on a stopped leader brakes hard")


# ---------------------------------------------- regressions from the first M2 run
def test_lookahead_smooth():
    """In CARLA, s_ego hopped ~0.1 m at each waypoint inside sharp turns; a target
    at s_ego + L hopped with it and kicked the steering PID every 4 ticks."""
    print("steering target in a tight turn")
    pts = [(x, 0.0) for x in np.arange(-30, 0, 1.0)]
    R = 10.0
    pts += [(R * math.sin(t), R - R * math.cos(t)) for t in np.arange(0, math.pi / 2, 1.0 / R)]
    pts += [(R, y) for y in np.arange(R, R + 30, 1.0)]
    Rc = R - 1.2                                         # car cuts the corner by 1.2 m, as in CARLA
    car = [(x, 1.2) for x in np.arange(-10, 0, 0.2)]
    car += [(Rc * math.sin(t), R - Rc * math.cos(t)) for t in np.arange(0, math.pi / 2, 0.2 / Rc)]
    ratio = {}
    for name in ('path', 'pursuit'):
        r, prev, steps = Route(np.array(pts)), None, []
        for (x, y) in car:
            s, _ = r.update_progress((x, y))
            tgt = r.point_at(s + 5.0) if name == 'path' else r.lookahead_point((x, y), 5.0)
            if prev is not None:
                steps.append(float(np.hypot(*(tgt - prev))))
            prev = tgt
        steps = np.array(steps[60:110])
        ratio[name] = steps.max() / steps.min()
    check(ratio['path'] > 1.4, f"old target hops in the turn (max/min step {ratio['path']:.2f})")
    check(ratio['pursuit'] < 1.2, f"pursuit target moves evenly (max/min step {ratio['pursuit']:.2f})")


# car model fitted to the first M2 CARLA log: accel = 7.19*engine - 3.73*brake - 0.66 - 0.271*v,
# engine = throttle through a 0.15 s lag, 1 tick (50 ms) delay. Fit R^2 0.69.
PLANT = dict(delay=1, tau=0.15, gt=7.19, gb=3.73, c0=0.66, c1=0.271)


class OldCarlaSpeedPID:
    """VehiclePIDController's longitudinal half as used in the first M2 run."""
    def __init__(self):
        self.buf = []

    def step(self, vt, v, a_cmd, hold):
        e = (vt - v) * 3.6
        self.buf = (self.buf + [e])[-10:]
        acc = float(np.clip(1.0 * e + 0.05 * sum(self.buf) * DT, -1, 1))
        return (min(acc, 0.75), 0.0) if acc >= 0 else (0.0, min(-acc, 0.5))


def drive_plant(ctrl, route_len=400.0, seconds=70.0):
    from evaluator import comfort_metrics
    p = Params(v0_kmh=30.0)
    dec, r = Decider(p), straight(route_len)
    x = v = eng = 0.0
    q_t, q_b = [0.0] * (PLANT['delay'] + 1), [0.0] * (PLANT['delay'] + 1)
    rows = []
    for _ in range(int(seconds / DT)):
        s, _ = r.update_progress((x, 0.0))
        d = dec.step(v, s, r.remaining, r.speed_cap(), [], True, True, 2.4, DT)
        th, br = ctrl.step(d.v_target, v, d.a_cmd, d.hold_brake)
        if d.hold_brake:
            th, br = 0.0, max(br, 0.6)
        q_t, q_b = q_t[1:] + [th], q_b[1:] + [br]
        eng += (q_t[0] - eng) * DT / PLANT['tau']
        acc = PLANT['gt'] * eng - PLANT['gb'] * q_b[0] - (PLANT['c0'] + PLANT['c1'] * v if v > 0.1 else 0.0)
        v = max(0.0, v + acc * DT)
        x += v * DT
        rows.append(dict(speed=v, throttle=th, brake=br, a_cmd=d.a_cmd, state=d.state))
    return comfort_metrics(rows, DT), x


def walker(cx, cy, vx, vy, cls='pedestrian'):
    return Obstacle(cls, 1.0, np.array([cx, cy, 0.9]), footprint(cx, cy, 0.5, 0.5, 0.0),
                    (0.5, 0.5, 1.8), 0.0, np.array([vx, vy]))


def test_crossing():
    print("crossing pedestrians")
    p = Params()
    r = straight(200)
    # 20 m ahead, 4 m to the side, walking towards the path at 1.4 m/s: inside the corridor
    # (|d| <= 1.0 + 0.4) after (4 - 0.25 - 1.4) / 1.4 = 1.7 s -> predicted hit 20 - 0.25 - 2.4 ahead
    h = predict_crossing([walker(20, 4.0, 0.0, -1.4)], r, p, 2.4)
    check(len(h) == 1 and h[0].kind == 'crossing' and abs(h[0].gap - (20 - 0.25 - 2.4)) < 1e-6,
          f"pedestrian walking towards the path is predicted: {[(x.gap, x.kind) for x in h]}")
    check(not predict_crossing([walker(20, 6.0, 0.0, -1.4)], r, p, 2.4),
          "too far to the side to reach the path within 2 s: ignored")
    check(not predict_crossing([walker(20, 2.2, 1.4, 0.0)], r, p, 2.4),
          "walking along the kerb (no speed towards the path): ignored")
    check(not predict_crossing([walker(20, 3.0, 0.0, 1.4)], r, p, 2.4), "walking away from the path: ignored")
    check(not predict_crossing([walker(20, 3.0, 0.0, -1.4, cls='Car')], r, p, 2.4), "cars are not predicted")
    check(not predict_crossing([walker(1.0, 3.0, 0.0, -1.4)], r, p, 2.4), "beside / behind the bumper: ignored")
    check(predict_crossing([walker(20, -4.0, 0.0, 1.4, cls='Pedestrian')], r, p, 2.4),
          "detector class name, from the other side")
    # closed loop: at 29 km/h a pedestrian steps out 12 m ahead (2.5 m to the side, 1.4 m/s): reacting
    # only once they are in the path is too late (contact), predicting the crossing is not
    d, v, x = Decider(p), 8.0, 0.0
    d.state, d.healthy_count, d.v_ref = FOLLOW_ROUTE, 99, v
    ped = walker(12.0, 2.5, 0.0, -1.4)
    hit = False
    for k in range(400):
        ped = walker(ped.center[0], ped.center[1] + ped.velocity[1] * DT, 0.0, ped.velocity[1]) \
            if ped.center[1] > -6.0 else ped
        r.update_progress((x, 0.0))
        hits = sorted(assess([ped], r, p, 2.4) + predict_crossing([ped], r, p, 2.4), key=lambda q: q.gap)
        dd = d.step(v, r.s_ego, r.remaining, 50.0, hits, True, True, 2.4, DT)
        v = max(0.0, v + (dd.a_cmd if dd.state == FOLLOW_ROUTE else min(dd.a_cmd, -2.0)) * DT) \
            if not dd.emergency else max(0.0, v - 8.0 * DT)
        x += v * DT
        if abs(ped.center[1]) < 1.25 and 0.0 < ped.center[0] - x < 2.65:
            hit = True
    check(not hit, f"closed loop: no contact with a pedestrian stepping out 12 m ahead (car stopped at x={x:.1f})")


def test_lights_and_crash():
    print("traffic lights and the stop after a pedestrian hit")
    p = Params()
    r = straight(400)

    def run(state_at, x0=0.0, v0=8.0, ticks=600, line=60.0):
        d, v, x = Decider(p), v0, x0
        d.state, d.healthy_count, d.v_ref = FOLLOW_ROUTE, 99, v
        states = []
        for k in range(ticks):
            r.update_progress((x, 0.0))
            dd = d.step(v, r.s_ego, r.remaining, 50.0, [], True, True, 2.4, DT,
                        light=(line - x - 2.4, state_at(k * DT), 7))
            v = 0.0 if dd.hold_brake else max(0.0, min(v + dd.a_cmd * DT, dd.v_target + 0.5))
            x += v * DT
            states.append(dd.state)
        return x, v, states, d

    x, v, st, d = run(lambda t: 'Red')
    check(v < 0.05 and 0.3 < 60 - (x + 2.4) < 2.0 and STOP_FOR_LIGHT in st,
          f"red 57 m ahead: stops about 1 m before the line (bumper {60 - x - 2.4:.2f} m before)")
    x, v, st, d = run(lambda t: 'Red' if t < 10 else 'Green', ticks=900)
    check(x > 100 and st[-1] == FOLLOW_ROUTE, f"drives on when it turns green (x={x:.0f})")
    x, v, st, d = run(lambda t: 'Green' if t < 0.05 else 'Red', x0=52.0)
    check(x > 60 and STOP_FOR_LIGHT not in st, f"turns red 5.6 m before the line at 29 km/h: goes through (x={x:.1f})")
    x, v, st, d = run(lambda t: 'Yellow', x0=52.0)
    check(x > 60, "yellow too close to stop comfortably: goes through")
    x, v, st, d = run(lambda t: 'Yellow')
    check(v < 0.05 and x + 2.4 < 60, "yellow far enough ahead: stops")
    x, v, st, d = run(lambda t: 'Yellow' if t < 0.3 else 'Red', x0=44.0, ticks=300)
    check(v < 0.05 and x + 2.4 < 60, f"go on yellow re-checked when it turns red, and it can still stop: stops "
                                     f"(bumper {60 - x - 2.4:.2f} m before)")
    x, v, st, d = run(lambda t: 'Green')
    check(STOP_FOR_LIGHT not in st and x > 100, "green: no stop")

    d = Decider(p)
    d.state, d.healthy_count = FOLLOW_ROUTE, 99
    d.crash('pedestrian')
    dd = d.step(6.0, 10.0, 100.0, 50.0, [], True, True, 2.4, DT)
    check(dd.state == CRASH_STOP and dd.emergency and dd.hold_brake and dd.v_target == 0.0,
          "after hitting a pedestrian: full brake and hold")
    dd = d.step(0.0, 10.0, 100.0, 50.0, [], True, True, 2.4, DT)
    check(dd.state == CRASH_STOP, "and it stays stopped")


def test_evaluator_rules():
    print("evaluator: pedestrian hits and red lights")
    import tempfile
    from evaluator import Evaluator

    def summary(raw, state_fn=lambda k: FOLLOW_ROUTE, lead_cls_fn=lambda k: '', lights='off', red_runs=0,
                junction=False):
        ev = Evaluator(tempfile.mkdtemp(), 'run', dict(traffic=5, lead_car={'x': 1}, lights=lights),
                       np.c_[np.arange(0.0, 101.0), np.zeros(101)], DT)
        for k in range(400):
            ev.tick(k=k, frame=1000 + k, t=round(k * DT, 2), x=k * 0.25, y=0.0, yaw=0.0, speed=5.0, lat_dev=0.1,
                    steer=0.0, v_target_kmh=18.0,                # t, v_target: the trajectory plot needs them
                    state=ROUTE_COMPLETE if k == 399 else state_fn(k), completion=k / 399.0,
                    throttle=0.2, brake=0.0, a_cmd=0.0, gt_lead_gap=12.0, gt_ttc='',
                    ego_in_junction=int(junction), junction_conflict=int(junction), lead_cls=lead_cls_fn(k))
        for i in range(red_runs):
            ev.event(100 + i, 1100 + i, 'red_light_run', light=7, v=5.0)
        ev._raw_collisions = raw
        return ev.finish('route')

    ped = [(1200, 55, 'walker.pedestrian.0002', 190.0, (50.0, 0.0, 0.0, 5.0, 51.0, 0.5, 90.0))]
    s = summary(ped, junction=True)
    check(s['pedestrian_hits_by_ego'] == 1 and not s['passed']
          and not s['checks']['no pedestrian hit by the ego (anywhere, junctions included)'],
          "a pedestrian hit inside a junction fails Step 3 (was exempt as a junction conflict)")
    s = summary([])
    check(s['passed'] and s['red_lights_run'] == 0, f"clean run passes: {[k for k, v in s['checks'].items() if not v]}")
    s = summary([], red_runs=1)
    check(s['passed'] and s['red_lights_run'] == 1, "red light run is counted, not gated without --lights")
    s = summary([], red_runs=1, lights='oracle')
    check(not s['passed'], "with --lights oracle a red light run fails")
    s = summary([], state_fn=lambda k: EMERGENCY_STOP if 100 <= k < 110 else FOLLOW_ROUTE,
                lead_cls_fn=lambda k: 'Pedestrian crossing' if k == 100 else 'Car')
    check(s['passed'] and s['emergency_stops_for_pedestrians'] == 1 and s['emergency_stops_outside_junctions'] == 0,
          "an emergency stop that began for a pedestrian is not a car-following failure")
    s = summary([], state_fn=lambda k: EMERGENCY_STOP if 100 <= k < 110 else FOLLOW_ROUTE,
                lead_cls_fn=lambda k: 'Car')
    check(not s['passed'] and s['emergency_stops_outside_junctions'] == 1, "one for a car still fails")


def test_speed_control():
    print("speed control on the car fitted from the CARLA log")
    m_old, x_old = drive_plant(OldCarlaSpeedPID())
    check(m_old['pedal_switches_per_min'] > 100 and m_old['brake_while_cruising_pct'] > 10,
          f"old PID reproduces the chatter seen in CARLA: {m_old}")
    m_new, x_new = drive_plant(SpeedController())
    check(m_new['pedal_switches_per_min'] < 10 and m_new['brake_while_cruising_pct'] < 1
          and m_new['rms_jerk'] < 2.0, f"new controller is smooth: {m_new}")
    check(abs(x_new - 400.0) < 4.0, f"new controller still stops at the route end: x={x_new:.1f}")


# ------------------------------------------------- manoeuvres (manoeuvre.py)
class Bicycle:
    """kinematic bicycle like the mock CARLA car: steering through CARLA's lateral PID (K_P 1.95, K_I 0.05,
    K_D 0.2, 0.1 per tick rate limit, max 0.8), 0.1 s steering lag, 35 deg wheel at full lock"""
    def __init__(self, x, y, yaw):
        self.x, self.y, self.yaw, self.v, self.wheel, self.past = x, y, yaw, 0.0, 0.0, 0.0
        self.eb = []

    def steer_to(self, tx, ty):
        vv = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        wv = np.array([tx - self.x, ty - self.y])
        n = np.linalg.norm(wv)
        dot = 0.0 if n == 0 else math.acos(float(np.clip(wv @ vv / n, -1, 1)))
        if vv[0] * wv[1] - vv[1] * wv[0] < 0:
            dot = -dot
        self.eb = (self.eb + [dot])[-10:]
        de = (self.eb[-1] - self.eb[-2]) / DT if len(self.eb) >= 2 else 0.0
        cur = float(np.clip(1.95 * dot + 0.2 * de + 0.05 * sum(self.eb) * DT, -1, 1))
        self.past = float(np.clip(min(max(cur, self.past - 0.1), self.past + 0.1), -0.8, 0.8))
        return self.past

    def step(self, steer, acc):
        self.wheel += (steer - self.wheel) * DT / 0.1
        self.v = max(0.0, self.v + acc * DT)
        self.yaw += self.v / 2.9 * math.tan(math.radians(35.0) * self.wheel) * DT
        self.x += self.v * math.cos(self.yaw) * DT
        self.y += self.v * math.sin(self.yaw) * DT


def lanes_for(n, left=0, right=0, w=3.5, left_ok=True, junction=None):
    return dict(width=np.full(n, w), left_kind=np.full(n, left), left_width=np.full(n, w if left else 0.0),
                left_ok=np.full(n, left_ok), right_kind=np.full(n, right), right_width=np.full(n, w if right else 0.0),
                right_ok=np.full(n, bool(right)), junction=np.zeros(n, bool) if junction is None else junction)


def car_at(cx, cy, vx=0.0, vy=0.0, l=4.5, w=1.9, cls='Car', tid=-1):
    yaw = math.atan2(vy, vx) if vx or vy else 0.0
    return Obstacle(cls, 1.0, np.array([cx, cy, 0.8]), footprint(cx, cy, l, w, yaw), (l, w, 1.5), yaw,
                    np.array([vx, vy]), tid)


def mover(x0, y0, vx, vy=0.0, t0=0.0, tid=-1):
    return lambda t: None if t < t0 else car_at(x0 + vx * (t - t0), y0 + vy * (t - t0), vx, vy, tid=tid)


def drive2d(route, lanes, statics=(), movers=(), v0_kmh=30.0, seconds=60.0, **mkw):
    """closed loop: decision.py + manoeuvre.py + the bicycle above, perception = ground truth"""
    from manoeuvre import MParams, Manoeuvres, poly_distance, rect
    p = Params(v0_kmh=v0_kmh)
    p.ego_half_width = 1.05
    dec = Decider(p)
    dec.state, dec.healthy_count = FOLLOW_ROUTE, 99
    mv = Manoeuvres(route, MParams(**mkw), p, 2.4, 2.4, lanes)
    car = Bicycle(route.xy[0, 0], route.xy[0, 1], route.heading[0])
    out = dict(collided=False, min_static=math.inf, min_moving=math.inf, events=[], ys=[], xy=[], states=[],
               t_end=None, mv=mv)
    for k in range(int(seconds / DT)):
        t = k * DT
        obs = list(statics) + [o for o in (f(t) for f in movers) if o is not None]
        route.update_progress((car.x, car.y))
        path = mv.path
        if path is not route:
            path.update_progress((car.x, car.y))
        hits = assess(obs, path, p, 2.4)
        res = mv.step(t, car.v, v0_kmh / 3.6, (car.x, car.y), car.yaw, obs, hits, DT,
                      route_hits=assess(obs, route, p, 2.4) if path is not route else None)
        if res['release']:
            dec.release('manoeuvre')
        path = mv.path
        hits = [h for h in assess(obs, path, p, 2.4) if not any(h.obstacle is o for o in mv.passing)]
        cap = path.speed_cap(p.a_lat_max, p.b)
        if res['v_cap'] is not None:
            cap = min(cap, res['v_cap'])
        d = dec.step(car.v, path.s_ego, route.remaining, cap, hits, True, True, 2.4, DT, stop=res['stop'],
                     static_gap=res['static_gap'])
        steer = car.steer_to(*path.lookahead_point((car.x, car.y), 3.0 + 0.5 * car.v))
        if d.emergency:
            acc = -8.0
        elif d.hold_brake:
            acc = -4.0 if car.v > 0 else 0.0
        else:
            acc = float(np.clip((d.v_target - car.v) / 0.4, -5.0, 2.5))
            acc = min(acc, d.a_cmd) if d.a_cmd < -3.0 else acc
        car.step(steer, acc)
        ego = rect(car.x, car.y, car.yaw, 2.4, 2.4, 1.05)
        for o in obs:
            dist = poly_distance(ego, o.corners)
            out['collided'] |= dist <= 0.0
            key = 'min_static' if any(o is q for q in statics) else 'min_moving'
            out[key] = min(out[key], dist)
        out['events'] += [e['type'] for e in mv.events]
        mv.events.clear()
        dec.events.clear()
        out['xy'].append((car.x, car.y))
        out['states'].append(d.state)
        if out['collided']:
            break
        if d.state == ROUTE_COMPLETE and car.v < 0.1:
            out['t_end'] = t
            break
    return out


def test_manoeuvre_geometry():
    print("manoeuvres: offset path and safety check")
    from manoeuvre import (MParams, Manoeuvres, alpha_at, offset_route, junction_spans, ego_time, ONCOMING,
                           SAME)
    r = straight(200)
    r.update_progress((0.0, 0.0))
    sh = offset_route(r, np.full(len(r.xy), 1.5))
    check(np.allclose(sh.xy[:, 1], 1.5) and np.allclose(sh.s, r.s), "offset +1.5 m = 1.5 m to the right (+y in "
                                                                        "CARLA), same s")
    s, dev = sh.update_progress((50.0, 1.5))
    check(abs(s - 50.0) < 1e-6 and dev < 1e-6, "progress on the offset path keeps the route's s")
    a = alpha_at([(10.0, 0.0), (20.0, 1.0)], np.array([0.0, 10.0, 15.0, 20.0, 30.0]))
    check(np.allclose(a, [0.0, 0.0, 0.5, 1.0, 1.0]), f"quintic ramp 0 -> 1: {a}")
    check(abs(ego_time(10.0, 0.0, 10.0, 2.0) - math.sqrt(10.0)) < 1e-9 and abs(ego_time(20.0, 5.0, 5.0, 1.0) - 4.0)
          < 1e-9, "time to cover a distance (accelerating / at speed)")
    flags = np.zeros(len(r.xy), bool)
    flags[100:115] = True
    j = junction_spans(r, flags)
    check(len(j) == 1 and abs(j[0]['s_in'] - 100.0) < 1e-6 and abs(j[0]['turn_deg']) < 1.0,
          f"junction span from the waypoint flags: {j}")
    p = Params()
    p.ego_half_width = 1.05
    mv = Manoeuvres(r, MParams(lane_change=True), p, 2.4, 2.4, lanes_for(len(r.xy), left=SAME))
    cand = mv._make_path([(0.0, 0.0), (15.0, 1.0)], mv._lane_c('left'))
    check(abs(cand.xy[30, 1] + 3.5) < 1e-6, "lane change path ends on the left lane's centre (-3.5 m)")
    ok, _ = mv.clear(cand, 25.0, 5.0, 8.3, [], (0.0, 0.0), 0.0)
    check(ok, "nothing around: clear")
    ok, why = mv.clear(cand, 25.0, 5.0, 8.3, [car_at(-25.0, -3.5, 12.0)], (0.0, 0.0), 0.0)
    check(not ok, f"a faster car coming from behind in the target lane: not clear ({why})")
    ok, _ = mv.clear(cand, 25.0, 5.0, 8.3, [car_at(-12.0, 0.0, 6.0)], (0.0, 0.0), 0.0)
    check(ok, "a car following in the ego's own lane is not the ego's to check")
    mv2 = Manoeuvres(r, MParams(steer_around=True), p, 2.4, 2.4, lanes_for(len(r.xy), left=ONCOMING))
    kn = [(0.0, 0.0), (10.0, 1.0), (60.0, 1.0), (75.0, 0.0)]
    ok, need, _ = mv2.sight_ok(kn, -2.6, 'left')
    check(not ok and need > 51.2, f"into the oncoming lane up to ~70 m: more than the 51 m the sensors see "
                                  f"(needs {need:.0f} m)")
    ok, need, _ = mv2.sight_ok([(0.0, 0.0), (10.0, 1.0), (16.0, 1.0), (28.0, 0.0)], -2.6, 'left')
    check(ok, f"a short pass: the oncoming lane in view is enough (needs {need:.0f} m)")


def test_steer_around():
    print("manoeuvres: steer around a stopped car (bicycle model, closed loop)")
    from manoeuvre import ONCOMING
    xs = np.arange(0.0, 251.0, 1.0)

    def run(statics=(), movers=(), left=ONCOMING, w=3.5, **kw):
        r = Route(np.c_[xs, np.zeros_like(xs)])
        return drive2d(r, lanes_for(len(r.xy), left=left, w=w), statics, movers, v0_kmh=20.0, seconds=70.0,
                       steer_around=True, **kw)
    park = car_at(60.0, 0.0, l=3.7, w=1.8)
    o = run([park])
    ys = [y for _, y in o['xy']]
    check(not o['collided'] and o['t_end'] is not None and 'steer_around_start' in o['events'],
          f"parked car in the lane: steers around, route completed (min clearance {o['min_static']:.2f} m)")
    check(o['min_static'] >= 0.5 and min(ys) < -2.0 and abs(ys[-1]) < 0.2,
          f"passes it >= 0.5 m away, into the oncoming lane (y {min(ys):.2f}) and back (y {ys[-1]:.2f})")
    o = run([park], left=0)
    check(not o['collided'] and 'steer_around_start' not in o['events'] and o['t_end'] is None,
          f"no lane beside it: waits behind it ({o['mv'].why})")
    o = run([car_at(60.0, 2.0, l=3.7, w=1.8)], left=0, w=4.5)
    check(not o['collided'] and 'steer_around_start' in o['events'] and o['t_end'] is not None
          and min(y for _, y in o['xy']) > -1.75, "a 4.5 m lane without markings, a car half on the kerb: squeezes "
                                                  "past inside its own lane")
    onc = [mover(130.0 + 30.0 * i, -3.5, -8.3) for i in range(3)]
    o = run([park], onc)
    check(not o['collided'] and o['t_end'] is not None and o['min_moving'] > 1.0,
          f"3 oncoming cars: waits for them, then passes (closest to them {o['min_moving']:.2f} m)")
    o = run([park], [mover(100.0, -3.5, -10.0, t0=12.3)])
    check(not o['collided'] and 'manoeuvre_abort' in o['events'] and o['t_end'] is not None,
          "an oncoming car appears just after it started: aborts, back in lane, passes later")
    o = run([car_at(60.0, 0.0, l=0.5, w=0.5, cls='pedestrian')])
    check('steer_around_start' not in o['events'] and not o['collided'], "a person: waits (not without people=True)")
    o = run([car_at(60.0, 0.0, l=1.8, w=0.6, cls='unknown')])
    check('steer_around_start' not in o['events'] and not o['collided'],
          "an 'unknown' object (occupancy check; may be a person): waits too")
    o = run([car_at(60.0, 0.0, l=1.8, w=0.6, cls='pedestrian')], people=True)
    check('steer_around_start' in o['events'] and not o['collided'] and o['min_static'] >= 1.0,
          f"people=True: passes a person lying on the road with >= 1 m (min {o['min_static']:.2f} m)")
    o = run([park, car_at(68.0, 0.0, l=3.7, w=1.8)])
    check('steer_around_start' not in o['events'] and not o['collided'],
          "two stopped cars nose to tail (a queue): traffic, waits")


def test_lane_change():
    print("manoeuvres: lane change past a slow car (bicycle model, closed loop)")
    from manoeuvre import SAME
    xs = np.arange(0.0, 501.0, 1.0)

    def run(movers, junction=None, **kw):
        r = Route(np.c_[xs, np.zeros_like(xs)])
        jn = None
        if junction:
            jn = np.zeros(len(r.xy), bool)
            jn[junction[0]:junction[1]] = True
        lanes = lanes_for(len(r.xy), left=SAME, junction=jn, left_ok=kw.pop('left_ok', True))
        return drive2d(r, lanes, (), movers, v0_kmh=30.0, seconds=45.0, lane_change=True, **kw)
    slow = mover(30.0, 0.0, 3.0, tid=7)
    o = run([slow])
    ev = o['events']
    check(not o['collided'] and ev.count('lane_change_start') == 1 and 'lane_change_done' in ev,
          f"slow car ahead: one change out, past it, one change back ({ev})")
    check(o['min_moving'] > 1.0, f"closest to the slow car {o['min_moving']:.2f} m")
    o = run([slow, mover(-40.0, -3.5, 13.0)])
    check(not o['collided'] and 'lane_change_start' in o['events'] and o['min_moving'] > 1.0,
          f"a faster car coming from behind in the other lane: waits for it, then changes "
          f"(closest {o['min_moving']:.2f} m)")
    o = run([slow], left_ok=False)
    check('lane_change_start' not in o['events'], "the marking forbids a change: stays behind it")
    o = run([slow, mover(20.0, -3.5, 3.2)])
    check('lane_change_start' not in o['events'], "the other lane is just as slow: stays")
    o = run([slow], junction=(80, 95))
    ys_j = [abs(y) for x, y in o['xy'] if 78.0 <= x <= 95.0]
    check(not o['collided'] and max(ys_j) < 0.3, f"junction 80 m ahead: no change before it (max |y| in it "
                                                  f"{max(ys_j):.2f} m)")

    def speeding(t):                                     # speeds up when the ego is beside it
        x = 30.0 + 3.0 * min(t, 10.5) + 8.0 * max(t - 10.5, 0.0)
        return car_at(x, 0.0, 3.0 if t < 10.5 else 8.0, tid=7)
    o = run([speeding], junction=(130, 145))
    ys_j = [abs(y) for x, y in o['xy'] if 128.0 <= x <= 145.0]
    check(not o['collided'] and 'lane_change_back' in o['events'] and ys_j and max(ys_j) < 0.3,
          f"slow car speeds up beside it, junction ahead: waits, gets back behind it before the junction "
          f"(max |y| in it {max(ys_j) if ys_j else None})")


def test_yield():
    print("manoeuvres: yield at a junction (bicycle model, closed loop)")
    from manoeuvre import ONCOMING
    R = 10.0
    pts = [(x, 0.0) for x in np.arange(0.0, 100.0, 1.0)]
    pts += [(100 + R * math.sin(a), -R + R * math.cos(a)) for a in np.arange(0.0, math.pi / 2, 1.0 / R)]
    pts += [(100 + R, y) for y in np.arange(-R, -R - 80.0, -1.0)]

    def run(movers, on=True):
        r = Route(np.array(pts))
        lanes = lanes_for(len(r.xy), left=ONCOMING, junction=(r.s >= 98.0) & (r.s <= 118.0))
        return drive2d(r, lanes, (), movers, v0_kmh=30.0, seconds=45.0, yield_at_junctions=on)
    T = 17.65                       # without traffic the ego reaches the conflict point (107.6, -3.5) at T
    o = run([mover(107.6 + 8.3 * T, -3.5, -8.3)])
    check(not o['collided'] and 'yield_start' in o['events'] and o['t_end'] is not None and o['min_moving'] > 1.0,
          f"left turn, an oncoming car arriving with the ego: yields, then turns (closest {o['min_moving']:.2f} m)")
    check(EMERGENCY_STOP not in o['states'], "... in time: no emergency stop")
    o = run([mover(107.6 + 8.3 * (T + 6.0), -3.5, -8.3)])
    check('yield_start' not in o['events'] and o['t_end'] is not None, "an oncoming car 6 s later: no need to yield")
    o = run([mover(107.6 + 8.3 * (T - 3.0 + 2.5 * i), -3.5, -8.3) for i in range(4)])
    check(not o['collided'] and 'yield_start' in o['events'] and o['min_moving'] > 1.0,
          "a stream of 4 oncoming cars 2.5 s apart: waits for all of them")
    o = run([mover(106.5, -3.5 + 8.3 * T, 0.0, -8.3)])
    check(not o['collided'] and 'yield_start' in o['events'], "a car crossing from the side: yields")
    o = run([mover(-15.0, 0.0, 7.0)])
    check('yield_start' not in o['events'], "a car following the ego: not a reason to yield")
    o = run([mover(107.6 + 8.3 * T, -3.5, -8.3)], on=False)
    check(EMERGENCY_STOP in o['states'] or o['collided'], "without yield the same car forces an emergency stop "
                                                          "(or a crash): the test has teeth")


def test_decision_stops():
    print("decision: yield / merge stop, larger gap to wait, release")
    from decision import YIELD
    p = Params()
    r = straight(300)
    d, v, x = Decider(p), 8.0, 0.0
    d.state, d.healthy_count, d.v_ref = FOLLOW_ROUTE, 99, v
    st = []
    for k in range(600):
        r.update_progress((x, 0.0))
        dd = d.step(v, r.s_ego, r.remaining, 50.0, [], True, True, 2.4, DT, stop=(50.0 - x - 2.4, 'yield'))
        v = 0.0 if dd.hold_brake else max(0.0, min(v + dd.a_cmd * DT, dd.v_target + 0.5))
        x += v * DT
        st.append(dd.state)
    check(v < 0.05 and 0.0 <= 50.0 - (x + 2.4) < 1.5 and YIELD in st and dd.hold_brake,
          f"yield stop 47.6 m ahead: stops at it and holds (bumper {50 - x - 2.4:.2f} m before)")
    hit = assess([static_car(40)], r, p, 2.4)
    for gap_want in (None, 9.5):
        r = straight(300)
        d, v, x = Decider(Params(v0_kmh=20)), 0.0, 0.0
        d.state, d.healthy_count = FOLLOW_ROUTE, 99
        for k in range(900):
            r.update_progress((x, 0.0))
            h = assess([static_car(40)], r, d.p, 2.4)
            dd = d.step(v, r.s_ego, r.remaining, 50.0, h, True, True, 2.4, DT, static_gap=gap_want)
            v = 0.0 if dd.hold_brake else max(0.0, min(v + dd.a_cmd * DT, dd.v_target + 0.5))
            x += v * DT
        gap = 40 - 2.25 - x - 2.4
        want = 4.0 if gap_want is None else 9.5
        check(abs(gap - want) < 1.0 and dd.state == BLOCKED_WAIT,
              f"static_gap {gap_want}: stops {gap:.1f} m behind (about {want} m), BLOCKED_WAIT")
    d = Decider(p)
    for _ in range(5):
        d._hysteresis(hit, True, DT)
    d.release('manoeuvre')
    check(d.latched is None and not d.hist, "release() forgets the latched obstacle")


def test_interactions():
    print("evaluator: post-encroachment time and clearance")
    from evaluator import interactions
    box = dict(front=2.4, rear=2.4, half_width=1.05)
    rows = [dict(k=k, x=k * 0.5, y=0.0, yaw=0.0) for k in range(200)]          # ego along +x, 10 m/s
    cross = [(k, 9, 50.0, -20.0 + k * 0.4 - 0.4 * 70, math.pi / 2, 4.5, 1.9) for k in range(200)]
    o = interactions(rows, cross, box, DT)
    check(o['pet_min_s'] is not None and o['pet_min_s'] < 0.5, f"a car crossing the ego's path about when it "
                                                               f"gets there: small PET ({o['pet_min_s']})")
    late = [(k, 9, 50.0, -20.0 + (k - 60) * 0.4 - 0.4 * 70, math.pi / 2, 4.5, 1.9) for k in range(200)]
    o2 = interactions(rows, late, box, DT)
    check(o2['pet_min_s'] is not None and 2.5 <= o2['pet_min_s'] <= 3.5, f"60 ticks later: PET about 3 s "
                                                                         f"({o2['pet_min_s']})")
    follow = [(k, 5, k * 0.5 - 15.0, 0.0, 0.0, 4.5, 1.9) for k in range(200)]
    o3 = interactions(rows, follow, box, DT)
    check(o3['pet_min_s'] is None and abs(o3['min_vehicle_clearance_m'] - (15.0 - 4.65)) < 0.01,
          "a car following 15 m behind: not a conflict, clearance 10.35 m")


def fake_walker_world(ground='all', lanes=(0.0, -3.5)):
    """a straight road along +x: lane centres at y in `lanes` (CARLA: +y is right of +x), 3.5 m wide;
    ground everywhere, or ('road') only on the lanes. Walkers move with their last WalkerControl."""
    class L:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)

        def distance(self, o):
            return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)

    class T:
        def __init__(self, location=None, rotation=None):
            self.location, self.rotation = location or L(), rotation or SimpleNamespace(yaw=0.0)

    C = SimpleNamespace(Location=L, Vector3D=L, Transform=T, Rotation=lambda yaw=0.0: SimpleNamespace(yaw=yaw),
                        LaneType=SimpleNamespace(Driving='Driving', Sidewalk='Sidewalk'),
                        WalkerControl=lambda d, v: SimpleNamespace(direction=d, speed=v))
    lo, hi = min(lanes) - 1.75, max(lanes) + 1.75

    class Map:
        def get_waypoint(self, loc, project_to_road=True, lane_type=None):
            if project_to_road:
                y = min(lanes, key=lambda c: abs(c - loc.y))
            else:
                y = next((c for c in lanes if abs(loc.y - c) <= 1.75), None)
                if y is None:
                    return None
            return SimpleNamespace(transform=T(L(loc.x, y, 0.0)), lane_width=3.5)

    class Walker:
        def __init__(self, tf, k):
            self.id, self.is_alive, self.ctl, self.phys = 1000 + k, True, None, True
            self.set_transform(tf)

        def set_transform(self, tf):
            self.loc = L(tf.location.x, tf.location.y, tf.location.z)

        def set_simulate_physics(self, on):
            self.phys = on

        def apply_control(self, c):
            self.ctl = c

        def get_location(self):
            return L(self.loc.x, self.loc.y, self.loc.z)

        def step(self, dt):
            if self.phys and self.ctl is not None and self.ctl.speed > 0:
                self.loc.x += self.ctl.direction.x * self.ctl.speed * dt
                self.loc.y += self.ctl.direction.y * self.ctl.speed * dt

    class World:
        def __init__(self):
            self.walkers = []

        def try_spawn_actor(self, b, tf):
            w = Walker(tf, len(self.walkers))
            self.walkers.append(w)
            return w

        def ground_projection(self, loc, dist):
            if ground == 'road' and not lo <= loc.y <= hi:
                return None
            return SimpleNamespace(location=L(loc.x, loc.y, 0.0))

    bp = SimpleNamespace(filter=lambda pat: [SimpleNamespace(has_attribute=lambda k: False)])
    return C, World(), Map(), bp


def walk_drive(ground='all', cross=0.5, n=10, seconds=80.0, seed=0):
    """the ego (point mass, 4.8 x 2.1 m) drives +x at up to 8 m/s and stops for walkers in its lane"""
    from lane_walkers import LaneWalkers
    C, world, cmap, bp = fake_walker_world(ground)
    route = Route(np.c_[np.arange(0.0, 400.5, 1.0), np.zeros(401)])
    lw = LaneWalkers(world, cmap, C, bp, route, n, cross, seed, [C.Location(0, 0, 0)], [], clear_d=1.9)
    x, v, out = 2.4, 0.0, dict(min_clear=99.0, rude=0, inside_far=0, min_abs_y=99.0)
    prev = {}
    for k in range(int(seconds / DT)):
        lw.tick(DT, x, v, (x, 0.0))
        for w in world.walkers:
            w.step(DT)
        block = any(w.phys and abs(w.loc.y) < 1.45 + 0.3 and 0.0 < w.loc.x - (x + 2.4) < 25.0 for w in world.walkers)
        v = max(0.0, v - 4.0 * DT) if block else min(8.0, v + 1.5 * DT)
        x += v * DT
        for w in world.walkers:
            if not w.phys:
                continue
            ay = abs(w.loc.y)
            out['min_abs_y'] = min(out['min_abs_y'], ay)
            dx = max(0.0, abs(w.loc.x - x) - 2.4)
            dy = max(0.0, ay - 1.05)
            out['min_clear'] = min(out['min_clear'], math.hypot(dx, dy))
            if prev.get(w.id, 9.0) >= 1.75 > ay and v > 0.5 and 0.0 < w.loc.x - (x + 2.4) < 2.0 * v:
                out['rude'] += 1                                  # stepped into the ego's lane right in front
            prev[w.id] = ay
        if x > 390.0:
            break
    out.update(lw=lw, x=x)
    return out


def side_pass(v_ego, gap):
    """one walker crossing from the left footpath (y = -6.45) to the right one (y = +2.95) at x = 60; the
    ego starts with its front `gap` m before that and keeps v_ego unless someone is in its lane ahead.
    Returns (closest walker-body to car-body distance, final walker y)."""
    from lane_walkers import LaneWalkers
    C, world, cmap, bp = fake_walker_world()
    route = Route(np.c_[np.arange(0.0, 300.5, 1.0), np.zeros(301)])
    lw = LaneWalkers(world, cmap, C, bp, route, 1, 0.0, 0, [C.Location(0, 0, 0)], [], clear_d=1.9)
    w = lw.walkers[0]
    a = w['actor']
    a.set_transform(C.Transform(C.Location(60.0, -6.45, 1.2)))
    a.set_simulate_physics(True)
    w.update(mode='cross', side=0, dir=1, i=60, crosser=False, target=(60, 2.95))
    x, v, closest = 60.0 - gap - 2.4, v_ego, 99.0
    for _ in range(int(30 / DT)):
        lw.tick(DT, x, v, (x, 0.0))
        a.step(DT)
        block = abs(a.loc.y) < 1.75 and 0.0 < a.loc.x - (x + 2.4) < 25.0
        v = max(0.0, v - 4.0 * DT) if block else min(v_ego, v + 1.5 * DT)
        x += v * DT
        if a.phys:
            closest = min(closest, math.hypot(max(0.0, abs(a.loc.x - x) - 2.4), max(0.0, abs(a.loc.y) - 1.05)) - 0.3)
    return closest, a.loc.y


def jitter_track(wm, xs, ys, every=2, sd=0.15, seed=0):
    """feed detector boxes of one object (true path xs[k], ys[k]) with sd m of centre jitter at 20/every Hz;
    returns [(k, center_xy, velocity)] per update"""
    rng = np.random.default_rng(seed)
    tf0, out = FakeTf(0, 0, 2.4), []
    for k in range(0, len(xs), every):
        x, y = xs[k] + rng.normal(0, sd), ys[k] + rng.normal(0, sd)
        o = SimpleNamespace(value={'boxes': np.array([[x, -y, -1.6, 4.5, 1.9, 1.5, 0]]), 'scores': np.array([0.9]),
                                   'cls': np.array([0])}, capture_k=k, lidar_tf=tf0, source='fusion')
        obs, _ = wm.update(o, k)
        if obs:
            out.append((k, obs[0].center[:2].copy(), obs[0].velocity.copy()))
    return out


def test_detector_jitter():
    print("detector box jitter: a parked car stays parked")
    names = ['Car', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
    n = 1200                                                       # 60 s, detections at 10 Hz
    xs, ys = np.full(n, 20.0), np.zeros(n)
    old = jitter_track(WorldModel(names, dt=DT, history_s=0.5, vel_sig=0.0), xs, ys)
    new = jitter_track(WorldModel(names, dt=DT), xs, ys)
    moving = lambda res: np.mean([np.hypot(*v) >= 0.5 for k, _, v in res if k > 40])
    check(moving(old) > 0.1 and moving(new) < 0.02,
          f"0.15 m box jitter: counted as moving (>= 0.5 m/s) {100 * moving(old):.0f}% of updates before "
          f"(0.5 s fit), {100 * moving(new):.1f}% now (1 s fit, significance test)")
    spread = np.std([c[0] for k, c, _ in new if k > 40])
    check(spread < 0.1, f"a static object is placed at the mean of its positions: centre jitter {spread:.2f} m (raw 0.15)")
    walk = jitter_track(WorldModel(names, dt=DT), np.full(n, 10.0), 5.0 - 1.3 * np.arange(n) * DT,
                        sd=0.15)
    sp = [np.hypot(*v) for k, _, v in walk if 40 < k < 200]
    check(np.mean([abs(s - 1.3) < 0.4 for s in sp]) > 0.95,
          f"a pedestrian crossing at 1.3 m/s keeps its speed (median {np.median(sp):.2f} m/s)")
    # decision: a stopped leader whose speed estimate wobbles 0.3-0.9 m/s
    for rel, want_flips in ((0.5, True), (1.0, False)):
        r = straight(300)
        d, v, x = Decider(Params(v0_kmh=20, static_release=rel)), 0.0, 0.0
        d.state, d.healthy_count = FOLLOW_ROUTE, 99
        rng = np.random.default_rng(1)
        flips, x_stop = 0, None
        for k in range(1200):
            r.update_progress((x, 0.0))
            car = static_car(40)
            if k > 400:
                car.velocity = np.array([rng.uniform(-0.9, -0.3), 0.0])   # jitter that looks like it backs up
            dd = d.step(v, r.s_ego, r.remaining, 50.0, assess([car], r, d.p, 2.4), True, True, 2.4, DT,
                        static_gap=9.5)
            v = 0.0 if dd.hold_brake else max(0.0, min(v + dd.a_cmd * DT, dd.v_target + 0.5))
            x += v * DT
            if k == 400:
                x_stop = x
            flips += k > 400 and dd.state == FOLLOW_ROUTE
        crept = x - x_stop
        if want_flips:
            check(flips > 0, f"static_release 0.5 (before): leaves BLOCKED_WAIT {flips} ticks, creeps {crept:.2f} m")
        else:
            check(flips == 0 and crept < 0.05,
                  f"static_release 1.0 (detector sources now): stays BLOCKED_WAIT, creeps {crept:.2f} m")


def test_lane_walkers():
    from lane_walkers import add_sidewalks
    r = walk_drive()
    lw = r['lw']
    check(np.nanmedian(lw.lines[0]) == 2.95 and np.nanmedian(lw.lines[1]) == -6.45,
          f"walk lines 1.2 m beyond each road edge: {np.nanmedian(lw.lines[0])}, {np.nanmedian(lw.lines[1])}")
    check(lw.crossings >= 3, f"half of them cross in front of the ego ({lw.crossings} crossings)")
    check(lw.placed > 10, f"walkers left behind start again ahead ({lw.placed} placements for 10)")
    check(r['min_clear'] > 0.1, f"no walker touches the ego (closest {r['min_clear']:.2f} m)")
    check(r['rude'] == 0, f"nobody steps into the ego's lane right in front of it ({r['rude']})")
    check(r['x'] > 390.0, f"the ego gets through ({r['x']:.0f} m in 80 s)")
    worst = min(side_pass(v, g)[0] for v in (3.0, 6.0, 8.0) for g in (5.0, 10.0, 14.0, 20.0, 30.0))
    check(worst > 0.5, f"a crosser waits until the passing car is fully by: never walks into its side "
                       f"(closest {worst:.2f} m)")
    c0, y0 = side_pass(0.0, -2.4)
    check(c0 > 0.3 and y0 < -1.0, f"the car stopped across their way: they wait, not walk into it "
                                  f"({c0:.2f} m, y {y0:.2f})")
    r2 = walk_drive(ground='road')
    lw2 = r2['lw']
    check(np.all(np.isnan(lw2.lines[0])) and np.nanmedian(lw2.lines[1]) == -5.0,
          "no ground beside the road: nothing on the ego's side, the far road edge 0.25 m inside")
    check(lw2.crossings == 0 and r2['min_abs_y'] >= 1.9, f"no crossing to a side with nowhere to stand "
                                                        f"({lw2.crossings}, closest |y| {r2['min_abs_y']:.2f})")
    x = ('<OpenDRIVE><road name="a" id="5" junction="-1"><lanes><laneSection s="0"><center><lane id="0" '
         'type="none"/></center><right><lane id="-1" type="driving"/><lane id="-2" type="driving"/></right>'
         '</laneSection></lanes></road><road id="9" junction="3"><lanes><laneSection s="0"><right><lane id="-1" '
         'type="driving"/></right></laneSection></lanes></road><road id="6" junction="-1"><lanes><laneSection '
         's="0"><left><lane id="1" type="driving"/></left><right><lane id="-1" type="driving"/><lane id="-2" '
         'type="sidewalk"/></right></laneSection></lanes></road></OpenDRIVE>')
    y, k = add_sidewalks(x, 2.0)
    check(k == 2 and '<lane id="-3" type="sidewalk"' in y and '<lane id="2" type="sidewalk"' in y,
          f"footpath beside the outer driving lane of each side ({k})")
    check(y.count('type="sidewalk"') == 3 and '<road id="9" junction="3"><lanes><laneSection s="0"><right><lane '
          'id="-1" type="driving"/></right>' in y, "junction roads and sides with a footpath are left alone")
    check(add_sidewalks(y, 2.0) == (y, 0), "a second pass changes nothing")


if __name__ == '__main__':
    for t in (test_idm, test_route, test_boxes, test_world_model, test_corridor,
              test_hysteresis, test_closed_loop, test_lookahead_smooth, test_speed_control,
              test_crossing, test_lights_and_crash, test_evaluator_rules, test_decision_stops,
              test_manoeuvre_geometry, test_steer_around, test_lane_change, test_yield, test_interactions,
              test_lane_walkers, test_detector_jitter):
        t()
    print(f"\n{N[0] - len(FAILS)} of {N[0]} checks passed")
    sys.exit(1 if FAILS else 0)