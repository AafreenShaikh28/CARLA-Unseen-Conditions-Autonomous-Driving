"""
manoeuvre.py -- the sideways half of the driving logic, and yielding at junctions.
Pure python/numpy, no CARLA: agent.py passes in what the map says (lanes beside the route, junctions),
so all of it is tested offline (test_driving_logic.py).

  steer around   pass a stopped obstacle in the lane (parked car, cattle; a person or an 'unknown' object
                 from the occupancy check only with people=True), using the next lane if needed
  lane change    move to a same-direction lane to pass a slow car, and come back
  yield          at a junction, wait for cars whose path crosses the ego's path in it (oncoming cars
                 when turning left, crossing cars)

decision.py stays the speed layer. Every sideways manoeuvre is an OFFSET of the route:
    path point(s) = route point(s) + alpha(s) * c(s) * n(s)
  n(s)      the route's right-hand normal (CARLA: d > 0 is right of the route)
  c(s)      where to go: a fixed offset (steer around) or the next lane's centre (lane change)
  alpha(s)  0 = route lane, 1 = fully over; quintic ramps between knots, so position, heading and
            curvature stay continuous and the steering target moves smoothly
The offset path keeps the route's s, so stop lines, junctions, the route end and every gap decision.py
computes keep their places: agent.py gives decision.py the offset path as "the route".

SAFETY CHECK (clear()), the same for every manoeuvre: the ego is moved along the planned path with a
planned speed profile, every other object along its velocity (constant velocity), in 0.2 s steps until
the manoeuvre ends (at most 12 s). Footprints are inflated: moving objects by lat_margin all round and
by headway_s x their speed ahead of them (room to brake), static ones by the class clearance, the ego by
ego_headway_s x its speed ahead (against moving objects); a moving object also by tail_s x its speed
behind it, unless it goes the ego's way slower than the ego (then the ego's own headway covers it).
To START a manoeuvre both are start_margin_s longer than to keep going, so a marginal gap
does not start one that the next tick aborts. Any overlap = not clear. Objects behind the
ego in its current lane are not checked (keeping their distance is their job; checking them would
forbid every manoeuvre).
WHAT THE SENSORS CANNOT SEE: a path into an ONCOMING lane is used only if a car appearing at the edge of
sensing range (sense_range_m) at phantom_kmh could still stop (1 s reaction, 4 m/s^2) before the
farthest point of that lane the ego will occupy.

STEER AROUND (steer_around): the first obstacle in the path stands still (< 0.5 m/s) for static_wait_s
(people: people_wait_s), its class is allowed (people and 'unknown' objects, which may be a person
lying on the road, only with people=True), it is not in or within 15 m before a junction or a stop
line (a car waiting there is traffic, not an obstruction), not in a queue (another stopped object within
10 m in front of it) and the route goes on > 25 m past it. Offset = its edge + class clearance + half
the ego's width (+ plan_extra_m for tracking error), on the side that needs less (left on a tie:
right-hand traffic), inside the driving lanes the map has (same-direction or oncoming, never a
footpath). While it waits for a gap the car keeps wait_gap_m to the obstacle so it has room to pull
out. Speed while passing: pass_kmh of the class.
LANE CHANGE (lane_change): a car ahead in the path, slower than min(v0 - 2.5 m/s, 0.8 v0), followed for
follow_s within 40 m; a same-direction lane on the left (else right) where the marking allows a change,
that is faster (nothing within 40 m ahead in it slower than the car + 1 m/s), and no junction or route
end for the whole overtake (estimated from the speed difference). Back to the route lane once the slow
car is behind the ego, no slower car is within 30 m ahead in the route lane and the change back is clear,
and always before the next junction: if the way back is not clear by then, the car slows and waits at
the last point it can still get back.
If a change becomes unsafe while the car is still less than half way over, it goes back (abort).
YIELD (yield_at_junctions): before and inside a junction on the route, every moving object that is not
driving along the ego's path is predicted (constant velocity, 8 s); if it will be in the ego's path
through the junction while the ego is, +-gap_s, the ego stops at the junction entry (inside it: before
the conflict) until that has been clear for yield_release_n ticks. Not if it can no longer stop
there at <= commit_decel (it is committed and goes on). It yields whatever the turn (defensive).
A crossing object that is in the ego's path now but will have left it gap_s before the ego gets
there goes into `passing`: agent.py takes it out of the corridor check, so the car does not brake
(or emergency-stop) for a car that is crossing ahead of it and will be gone.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from decision import assess
from route import Route, wrap_pi

NONE_LANE, SAME, ONCOMING = 0, 1, 2
KIND = {'vehicle': 'vehicle', 'Car': 'vehicle', 'Truck': 'vehicle', 'Bus': 'vehicle', 'Van': 'vehicle',
        'Cyclist': 'cyclist', 'Motorcycle': 'cyclist', 'Bicycle': 'cyclist',
        'animal': 'animal', 'Animal': 'animal', 'unknown': 'unknown',
        'pedestrian': 'pedestrian', 'Pedestrian': 'pedestrian', 'LyingPerson': 'pedestrian'}


@dataclass
class MParams:
    # steer around
    steer_around: bool = False
    people: bool = False                   # also around a person lying / sitting on the road, and 'unknown'
    static_wait_s: float = 2.0
    people_wait_s: float = 3.0
    clearance: dict = field(default_factory=lambda: {'vehicle': 0.6, 'cyclist': 1.0, 'animal': 1.2,
                                                     'unknown': 1.5, 'pedestrian': 1.5})
    pass_kmh: dict = field(default_factory=lambda: {'vehicle': 20.0, 'cyclist': 15.0, 'animal': 15.0,
                                                    'unknown': 10.0, 'pedestrian': 10.0})
    wait_gap_m: float = 9.5                # bumper gap kept while waiting for a gap to pass
    plan_extra_m: float = 0.2              # planned clearance = class clearance + this (tracking error)
    ramp_done_m: float = 1.5               # the ego centre is fully over this far before the obstacle
    junction_clear_m: float = 15.0
    queue_m: float = 10.0
    after_m: float = 1.0                   # the ego's rear this far past the obstacle before it comes back
    route_after_m: float = 25.0
    # lane change
    lane_change: bool = False
    slow_dv: float = 2.5                   # m/s
    slow_ratio: float = 0.8
    follow_s: float = 3.0
    follow_range_m: float = 40.0
    return_clear_m: float = 30.0
    min_hold_s: float = 1.0
    # shape of a shift
    l_min: float = 10.0                    # m, shortest ramp (from standstill)
    l_min_change: float = 15.0
    l_max: float = 30.0
    a_lat: float = 1.5                     # m/s^2 peak lateral acceleration of a ramp at the planned speed
    edge_margin: float = 0.2               # the ego's side stays this far inside the drivable lanes
    # safety check
    dt: float = 0.2
    horizon_s: float = 12.0
    lat_margin: float = 0.3
    headway_s: float = 1.0                 # room kept ahead of a moving object (its travel in this time)
    keep_slack_m: float = 0.3              # keep going past a static object down to clearance - this (start:
                                           # clearance; plan: clearance + plan_extra): its detector box moves
    target_moving_speed: float = 1.5       # the object being steered around counts as static below this, m/s:
                                           # detector box jitter / growing boxes are not a car driving off
    tail_s: float = 0.5                    # ... and behind it (the ego crosses its lane only after it passed)
    ego_headway_s: float = 0.5
    a_plan: float = 1.0
    sense_range_m: float = 51.2
    phantom_kmh: float = 30.0
    phantom_react_s: float = 1.0
    phantom_decel: float = 4.0
    cooldown_s: float = 2.0                # after an abort, before trying again
    start_margin_s: float = 1.0            # to START, others need this much more headway (no start-abort flapping)
    # yield
    yield_at_junctions: bool = False
    yield_look_m: float = 50.0
    yield_horizon_s: float = 8.0
    yield_step_s: float = 0.25
    gap_s: float = 1.5
    a_go: float = 1.2
    commit_decel: float = 4.0
    yield_release_n: int = 6
    entry_margin: float = 1.0              # stop the bumper this far before the junction entry


# ------------------------------------------------------------------ geometry
def smooth(u):
    """quintic 0 -> 1 with zero slope and curvature at both ends"""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def alpha_at(knots, s):
    """piecewise quintic through knots [(s, alpha), ...] sorted by s; flat before the first, after the last"""
    s = np.asarray(s, float)
    if not knots:
        return np.zeros_like(s)
    out = np.full(s.shape, float(knots[0][1]))
    for (s0, a0), (s1, a1) in zip(knots[:-1], knots[1:]):
        m = s >= s0
        out[m] = a0 + (a1 - a0) * (smooth((s[m] - s0) / (s1 - s0)) if s1 > s0 else 1.0)
    return out


def ramp_length(offset, v, p, change=False):
    """ramp long enough that its peak curvature 5.77 |c| / L^2 gives <= a_lat at speed v"""
    v = max(float(v), 3.0)
    lo = p.l_min_change if change else p.l_min
    return float(np.clip(math.sqrt(5.77 * abs(offset) * v * v / p.a_lat), lo, p.l_max))


def point_normals(route):
    """unit right-hand normal at each route point (mean of the two segment normals)"""
    h = route.heading
    hp, hn = np.r_[h[0], h], np.r_[h, h[-1]]
    n = np.c_[-(np.sin(hp) + np.sin(hn)), np.cos(hp) + np.cos(hn)]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def offset_route(route, d, normals=None):
    """the route shifted sideways by d (one value per route point, metres, + = right). Keeps the route's
    s, waypoints and progress, so everything measured in s stays where it was."""
    n = point_normals(route) if normals is None else normals
    new = Route.__new__(Route)
    new.xy = route.xy + np.asarray(d, float)[:, None] * n
    new.wps, new.s, new.seg_len, new.length = route.wps, route.s, route.seg_len, route.length
    seg = np.diff(new.xy, axis=0)
    new.heading = np.arctan2(seg[:, 1], seg[:, 0])
    new.idx, new.s_ego = route.idx, route.s_ego
    new.offset = np.asarray(d, float)
    return new


def rect(cx, cy, h, front, rear, hw):
    c, s = math.cos(h), math.sin(h)
    lx = np.array([front, front, -rear, -rear])
    ly = np.array([hw, -hw, -hw, hw])
    return np.c_[cx + c * lx - s * ly, cy + s * lx + c * ly]


def hull(pts):
    """convex hull (monotone chain), counter-clockwise in a right-handed plot"""
    P = sorted(set(map(tuple, np.round(np.asarray(pts, float), 6))))
    if len(P) <= 2:
        return np.array(P)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def overlap(a, b):
    """separating-axis test, convex polygons (k, 2)"""
    for poly in (a, b):
        e = np.roll(poly, -1, axis=0) - poly
        n = np.c_[-e[:, 1], e[:, 0]]
        pa, pb = a @ n.T, b @ n.T
        if np.any((pa.max(0) < pb.min(0)) | (pb.max(0) < pa.min(0))):
            return False
    return True


def poly_distance(a, b):
    """0 if the convex polygons overlap, else the smallest distance between them"""
    if overlap(a, b):
        return 0.0
    best = math.inf
    for P, Q in ((a, b), (b, a)):
        A, B = Q, np.roll(Q, -1, axis=0)
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-12)
        for x in P:
            t = np.clip(((x - A) * AB).sum(1) / L2, 0.0, 1.0)
            best = min(best, float(np.min(np.linalg.norm(A + t[:, None] * AB - x, axis=1))))
    return best


def _strip_part(poly, thr, lo, hi):
    """s-range of the part of polygon [(s, d), ...] inside |d| <= thr and lo <= s <= hi, or None"""
    for a_s, a_d, b in ((0.0, 1.0, thr), (0.0, -1.0, thr), (-1.0, 0.0, -lo), (1.0, 0.0, hi)):
        out = []
        for i in range(len(poly)):
            P, Q = poly[i], poly[(i + 1) % len(poly)]
            fp = a_s * P[0] + a_d * P[1] - b
            fq = a_s * Q[0] + a_d * Q[1] - b
            if fp <= 0.0:
                out.append(P)
            if fp * fq < 0.0:
                t = fp / (fp - fq)
                out.append((P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])))
        poly = out
        if not poly:
            return None
    ss = [q[0] for q in poly]
    return min(ss), max(ss)


def ego_time(dist, v, v_max, a):
    """seconds to cover dist metres from speed v, accelerating at a up to v_max"""
    dist = max(float(dist), 0.0)
    v_max = max(float(v_max), 0.5)
    v = min(max(float(v), 0.0), v_max)
    d_acc = (v_max * v_max - v * v) / (2.0 * a)
    if dist <= d_acc:
        return (-v + math.sqrt(v * v + 2.0 * a * dist)) / a
    return (v_max - v) / a + (dist - d_acc) / v_max


def junction_spans(route, flags, a_lat=2.0):
    """[{s_in, s_out, turn_deg, v_turn}] from per-point is_junction flags. turn > 0 = right
    (CARLA's yaw grows clockwise), < 0 = left. v_turn = curve speed limit through it."""
    flags = np.asarray(flags, bool)
    out, i, n = [], 0, len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        h_in = route.heading[max(i - 1, 0)]
        h_out = route.heading[min(j, len(route.heading) - 1)]
        turn = math.degrees(wrap_pi(h_out - h_in))
        span = max(route.s[j] - route.s[max(i - 1, 0)], 1.0)
        kappa = abs(math.radians(turn)) / span
        out.append(dict(s_in=float(route.s[i]), s_out=float(route.s[j]), turn_deg=round(turn, 1),
                        v_turn=float(min(math.sqrt(a_lat / max(kappa, 1e-6)), 50.0))))
        i = j + 1
    return out


def turn_name(deg):
    return 'left' if deg < -30.0 else 'right' if deg > 30.0 else 'straight'


# ------------------------------------------------------------------ planner
class Manoeuvres:
    """agent.py calls step() every tick before decision.step(); see the top of this file"""

    def __init__(self, route, p: MParams, dp, ego_front, ego_rear, lanes=None, stop_s=()):
        """route: the base Route; dp: decision.Params (ego_half_width, margin, horizon, static_speed);
        lanes: dict of arrays, one value per route point: width, left_kind, left_width, left_ok,
        right_kind, right_width, right_ok, junction (kinds: 0 none, 1 same direction, 2 oncoming;
        *_ok = the marking allows a lane change to that side). None = a single lane, no junctions.
        stop_s: route s of the stop lines (traffic lights)."""
        self.route, self.p, self.dp = route, p, dp
        self.front, self.rear, self.hw = float(ego_front), float(ego_rear), float(dp.ego_half_width)
        n = len(route.xy)
        if lanes is None:
            lanes = dict(width=np.full(n, 3.5), left_kind=np.zeros(n, int), left_width=np.zeros(n),
                         left_ok=np.zeros(n, bool), right_kind=np.zeros(n, int), right_width=np.zeros(n),
                         right_ok=np.zeros(n, bool), junction=np.zeros(n, bool))
        self.lanes = {k: np.asarray(v) for k, v in lanes.items()}
        self.normals = point_normals(route)
        self.jx = junction_spans(route, self.lanes['junction'])
        self.stop_s = sorted(float(x) for x in stop_s)
        self.path = route
        self.mode = None                 # None | 'avoid' | 'change' | 'hold' | 'back'
        self.knots, self.c, self.side = [], 0.0, ''
        self.plan = {}                   # what the current manoeuvre is about
        self.cand = None                 # steer-around candidate
        self.follow_t = 0.0              # time spent behind a slow car
        self.cool_until = -1.0
        self.yielding = None
        self.yield_clear = 0
        self.passing = []                # crossing objects in the path now that will have left it >= gap_s
                                         # before the ego gets there: agent.py does not brake for them
        self.events = []                 # drained by agent.py into the evaluator
        self.status = ''                 # one word for the tick log and the video
        self.why = ''                    # last reason a wanted manoeuvre did not start
        self.stats = dict(steer_arounds=0, lane_changes=0, changes_back=0, aborts=0, yields=0,
                          yield_s=0.0, waited_for_gap_s=0.0)

    # ---------------------------------------------------------------- helpers
    def _event(self, kind, **kw):
        self.events.append(dict(type=kind, **kw))

    def _i(self, s):
        return int(np.clip(np.searchsorted(self.route.s, s), 0, len(self.route.s) - 1))

    def _span(self, s0, s1):
        i0, i1 = self._i(s0), self._i(s1)
        return slice(i0, max(i1 + 1, i0 + 1))

    def alpha_now(self):
        return float(alpha_at(self.knots, [self.route.s_ego])[0]) if self.knots else 0.0

    def offset_now(self):
        if not self.knots:
            return 0.0
        c = self.c[self._i(self.route.s_ego)] if np.ndim(self.c) else self.c
        return self.alpha_now() * float(c)

    def _make_path(self, knots, c):
        d = alpha_at(knots, self.route.s) * (np.asarray(c, float) if np.ndim(c) else float(c))
        return offset_route(self.route, np.nan_to_num(d), self.normals)

    def _next_junction(self, s_from):
        for j in self.jx:
            if j['s_out'] >= s_from:
                return j
        return None

    def _junction_near(self, s0, s1):
        """a junction overlapping [s0, s1]"""
        return any(j['s_in'] <= s1 and j['s_out'] >= s0 for j in self.jx)

    def _set(self, mode, knots=None, c=None):
        self.mode = mode
        if knots is not None:
            self.knots, self.c = knots, c
        self.path = self._make_path(self.knots, self.c) if self.knots and mode else self.route
        if mode is None:
            self.knots, self.c, self.side, self.plan = [], 0.0, '', {}
        self.path.idx, self.path.s_ego = self.route.idx, self.route.s_ego

    # ---------------------------------------------------------- safety check
    def clear(self, cand, s_end, v, v_go, obstacles, ego_xy, ego_yaw, start=True, static_ids=()):
        """(True, '') if driving along path `cand` until the ego centre reaches s_end is clear of every
        object (see the top of this file), else (False, why). start=True: the stricter test to begin a
        manoeuvre (others' headway + start_margin_s); False: the test to keep going (static objects with
        keep_slack_m less clearance). static_ids: track ids of the object being steered around: static
        unless faster than target_moving_speed."""
        p, r = self.p, self.route
        hwy = p.headway_s + (p.start_margin_s if start else 0.0)
        tail = p.tail_s + (p.start_margin_s if start else 0.0)
        ts = np.arange(0.0, p.horizon_s + 1e-9, p.dt)
        vs = np.minimum(v_go, v + p.a_plan * ts) if v < v_go else np.full(len(ts), float(v))
        ss = r.s_ego + np.r_[0.0, np.cumsum(0.5 * (vs[1:] + vs[:-1]) * p.dt)]
        n_t = min(len(ts), int(np.searchsorted(ss, s_end)) + 1)
        ts, vs, ss = ts[:n_t], vs[:n_t], ss[:n_t]
        cxy = np.array([cand.point_at(x) for x in ss])
        hd = np.array([math.atan2(*cand.tangent_at(x)[::-1]) for x in ss])
        ego_r = math.hypot(max(self.front, self.rear), self.hw)
        ego_plain = [rect(cxy[k, 0], cxy[k, 1], hd[k], self.front, self.rear, self.hw) for k in range(n_t)]
        ego_ahead = [rect(cxy[k, 0], cxy[k, 1], hd[k], self.front + p.ego_headway_s * vs[k], self.rear, self.hw)
                     for k in range(n_t)]
        c, s = math.cos(ego_yaw), math.sin(ego_yaw)
        for ob in obstacles:
            rel = ob.center[:2] - np.asarray(ego_xy, float)
            x_rel, y_rel = rel[0] * c + rel[1] * s, -rel[0] * s + rel[1] * c
            l, w = float(ob.size[0]), float(ob.size[1])
            if x_rel + 0.5 * l < -self.rear + 0.5 and abs(y_rel) < self.hw + 1.2:
                continue                                   # behind the ego in its own lane
            if math.hypot(rel[0], rel[1]) > 150.0:
                continue
            vel = np.asarray(ob.velocity, float)[:2]
            sp = float(np.hypot(vel[0], vel[1]))
            kind = KIND.get(ob.cls, 'unknown')
            target = ob.track_id >= 0 and ob.track_id in static_ids
            if sp < (p.target_moving_speed if target else self.dp.static_speed):
                m = max(p.clearance.get(kind, 1.0) - (0.0 if start else p.keep_slack_m), p.lat_margin)
                poly = rect(ob.center[0], ob.center[1], ob.yaw, 0.5 * l + m, 0.5 * l + m, 0.5 * w + m)
                rad = math.hypot(0.5 * l + m, 0.5 * w + m)
                dist = np.hypot(cxy[:, 0] - ob.center[0], cxy[:, 1] - ob.center[1])
                for k in np.nonzero(dist <= rad + ego_r)[0]:
                    if overlap(ego_plain[k], poly):
                        return False, f"{ob.cls} (static) at t+{ts[k]:.1f} s"
                continue
            m = p.lat_margin
            along = float(np.dot(vel, [c, s]))
            tl = 0.0 if along > 0.7 * sp and along < v + 0.5 else tail    # a slower car the ego's way: the
            base = rect(ob.center[0], ob.center[1], ob.yaw, 0.5 * l + m, 0.5 * l + m, 0.5 * w + m)   # ego's own
            grown = hull(np.vstack([base - vel * tl, base + vel * hwy]))                    # headway covers it
            rad = math.hypot(0.5 * l + m, 0.5 * w + m) + 0.5 * sp * (hwy + tl)
            ctr = ob.center[:2][None] + ts[:, None] * vel[None] + 0.5 * (hwy - tl) * vel[None]
            dist = np.hypot(cxy[:, 0] - ctr[:, 0], cxy[:, 1] - ctr[:, 1])
            for k in np.nonzero(dist <= rad + ego_r + p.ego_headway_s * vs)[0]:
                if overlap(ego_ahead[k], grown + vel * ts[k]):
                    onc = float(np.dot(vel, [c, s])) < -0.5 * sp
                    return False, f"{ob.cls}{' oncoming' if onc else ''} at t+{ts[k]:.1f} s"
        return True, ''

    def _target_ids(self, obstacles):
        """track id of the object being steered around (its own id, or the nearest box to where it was:
        a detector track can restart)"""
        ob0 = self.plan.get('obstacle')
        if ob0 is None:
            return ()
        ids = {ob0.track_id} if ob0.track_id >= 0 else set()
        near = [o for o in obstacles if o.track_id >= 0 and float(np.hypot(*(o.center[:2] - ob0.center[:2]))) < 1.5]
        ids.update(o.track_id for o in near)
        return tuple(ids)

    def _no_oncoming_in(self, cand, obstacles):
        """nothing coming towards the ego is in the planned path now (even if it will have left it in time:
        decision.py would brake for it on the new path)"""
        cand.idx, cand.s_ego = self.route.idx, self.route.s_ego
        for h in assess(obstacles, cand, self.dp, self.front):
            if h.v_lead < -self.dp.static_speed:
                return False, f'{h.obstacle.cls} coming towards the car in that lane, {h.gap:.0f} m'
        return True, ''

    def sight_ok(self, knots, c, side):
        """the phantom rule for a path that goes into an oncoming lane: (ok, need_m, why)"""
        p, r, L = self.p, self.route, self.lanes
        sl = self._span(r.s_ego, r.s_ego + 200.0)
        s = r.s[sl]
        a = alpha_at(knots, s) * (np.asarray(c, float)[sl] if np.ndim(c) else float(c))
        kind = L[side + '_kind'][sl]
        edge = np.abs(a) + self.hw > 0.5 * L['width'][sl] + 0.1          # the ego reaches over the lane line
        inside = edge & (kind == ONCOMING)
        if not inside.any():
            return True, 0.0, ''
        far = float(s[inside].max()) + self.front - r.s_ego
        v = p.phantom_kmh / 3.6
        need = far + v * p.phantom_react_s + v * v / (2.0 * p.phantom_decel)
        if need > p.sense_range_m:
            return False, need, (f"sight: needs {need:.0f} m of oncoming lane in view, sensors see "
                                 f"{p.sense_range_m:.0f} m")
        return True, need, ''

    # ---------------------------------------------------------- steer around
    def _avoid(self, t, v, hits, obstacles, ego_xy, ego_yaw):
        """steer-around candidate and plan; returns the bumper gap to keep while waiting, or None"""
        p, r, dp = self.p, self.route, self.dp
        h = hits[0] if hits else None
        if h is None:
            self.cand = None
            return None
        ob = h.obstacle
        sp = float(np.hypot(*np.asarray(ob.velocity, float)[:2]))
        kind = KIND.get(ob.cls)
        lim = dp.static_speed if self.cand is None else p.target_moving_speed     # once a candidate: hysteresis
        if sp >= lim or kind is None or (kind in ('pedestrian', 'unknown') and not p.people):
            self.cand = None
            return None
        c = ob.center[:2]
        if self.cand is None or float(np.hypot(*(c - self.cand['xy']))) > 1.5:
            self.cand = dict(xy=c.copy(), t0=t, cls=ob.cls, kind=kind, stopped_s=0.0)
        self.cand['xy'] = c.copy()
        _, d, s_abs, beyond = r.project(ob.corners, dp.horizon + 20.0, return_beyond=True)
        if beyond.any():
            return None                                         # not all of it in view on the route yet
        s_near, s_far = float(s_abs.min()), float(s_abs.max())
        # traffic, not an obstruction: at / just before a junction or a stop line, or in a queue
        if any(j['s_in'] - p.junction_clear_m <= s_far and s_near <= j['s_out'] for j in self.jx):
            self.why = 'it is at a junction (traffic, not an obstruction)'
            return None
        if any(s_near - 2.0 <= x <= s_far + p.junction_clear_m for x in self.stop_s):
            self.why = 'it is at a stop line (traffic, not an obstruction)'
            return None
        for h2 in hits[1:]:
            o2 = h2.obstacle
            if (float(np.hypot(*np.asarray(o2.velocity, float)[:2])) < dp.static_speed
                    and h2.s_abs - s_far < p.queue_m):
                self.why = 'queue: another stopped object just ahead of it'
                return None
        if s_far + p.route_after_m > r.length:
            self.why = 'the route ends just after it'
            return None
        if v < 0.3 and h.gap < p.wait_gap_m + 3.0:
            self.cand['stopped_s'] += self._dt               # standing behind it, waiting for a gap
        wait = p.people_wait_s if kind == 'pedestrian' else p.static_wait_s
        if t - self.cand['t0'] < wait or t < self.cool_until:
            self.why = f'checking it stays stopped ({t - self.cand["t0"]:.1f} of {wait:.1f} s)'
            return p.wait_gap_m
        plan, why = self._plan_avoid(kind, d, s_near, s_far, v)
        if plan is None:
            self.why = why
            return p.wait_gap_m
        v_pass = p.pass_kmh.get(kind, 15.0) / 3.6
        cand = self._make_path(plan['knots'], plan['c'])
        ok, _, why = self.sight_ok(plan['knots'], plan['c'], plan['side'])
        if ok:
            ok, why = self._no_oncoming_in(cand, obstacles)
        if ok:
            ok, why = self.clear(cand, plan['knots'][-1][0], v, v_pass, obstacles, ego_xy, ego_yaw,
                                 static_ids=(ob.track_id,))
        if not ok:
            self.why = why
            return p.wait_gap_m
        waited = self.cand['stopped_s']
        self.stats['steer_arounds'] += 1
        self.stats['waited_for_gap_s'] += waited
        self.plan = dict(kind='avoid', cls=ob.cls, v_pass=v_pass, s_end=plan['knots'][-1][0],
                         s_far=s_far, obstacle=ob, started=t)
        self.side = plan['side']
        self._set('avoid', plan['knots'], plan['c'])
        self._event('steer_around_start', cls=ob.cls, side=plan['side'], offset_m=round(plan['c'], 2),
                    gap=round(h.gap, 2), stopped_s=round(waited, 1), seen_s=round(t - self.cand['t0'], 1),
                    into_oncoming_lane=plan['oncoming'], ramp_m=round(plan['knots'][1][0] - plan['knots'][0][0], 1))
        self.cand = None
        self.why = ''
        return None

    def _plan_avoid(self, kind, d, s_near, s_far, v):
        p, r, L = self.p, self.route, self.lanes
        d_now = self.offset_now()                       # not 0 only when retrying after an abort
        clr = p.clearance.get(kind, 1.0) + p.plan_extra_m     # the check needs the class clearance
        v_pass = p.pass_kmh.get(kind, 15.0) / 3.6
        options = []
        for side, D in (('left', float(d.min()) - clr - self.hw), ('right', float(d.max()) + clr + self.hw)):
            l_in = ramp_length(D, max(v, 2.0), p)
            l_out = ramp_length(D, max(min(v_pass, math.sqrt(v * v + 2 * p.a_plan * 20.0)), 2.0), p)
            s_b = s_near - p.ramp_done_m
            s_a = max(s_b - l_in, r.s_ego)
            if s_b - s_a < p.l_min:
                s_b = s_a + p.l_min                     # close behind it: finish the ramp later, clear()
                if s_b > s_near + 0.5:                  # checks the front corner against its clearance
                    options.append((None, f'too close to pull out on the {side} ({s_near - r.s_ego:.1f} m)'))
                    continue
            s_c = s_far + self.rear + p.after_m
            s_d = s_c + l_out
            if s_d + 5.0 > r.length:
                options.append((None, 'the route ends before it can get back'))
                continue
            if self._junction_near(s_a, s_d + 5.0):
                options.append((None, 'a junction before it can get back'))
                continue
            sl = self._span(s_b, s_c)
            w = L['width'][sl]
            kind_side, w_side = L[side + '_kind'][sl], L[side + '_width'][sl]
            room = 0.5 * w + np.where(kind_side > 0, w_side, 0.0) - p.edge_margin
            if np.any(abs(D) + self.hw > room):
                options.append((None, f'no room on the {side} (needs {abs(D) + self.hw:.1f} m from the lane '
                                      f'centre, has {float(room.min()):.1f})'))
                continue
            oncoming = bool(np.any((abs(D) + self.hw > 0.5 * w + 0.1) & (kind_side == ONCOMING)))
            if d_now * D < 0.0 and abs(d_now) > 0.1:
                options.append((None, 'the car is still offset to the other side'))
                continue
            a0 = float(np.clip(d_now / D, 0.0, 1.0)) if D else 0.0
            knots = [(s_a, a0), (s_b, 1.0), (s_c, 1.0), (s_d, 0.0)]
            options.append((dict(side=side, c=D, knots=knots, oncoming=oncoming), ''))
        good = [o for o, _ in options if o is not None]
        if not good:
            return None, '; '.join(w for o, w in options if o is None)
        good.sort(key=lambda o: abs(o['c']) + (0.0 if o['side'] == 'left' else 0.3))
        return good[0], ''

    # ---------------------------------------------------------- lane change
    def _lane_c(self, side):
        """offset of the neighbour lane's centre at each route point (NaN where it is not a same-direction
        driving lane)"""
        L = self.lanes
        sgn = -1.0 if side == 'left' else 1.0
        c = sgn * 0.5 * (L['width'] + L[side + '_width'])
        return np.where(L[side + '_kind'] == SAME, c, np.nan)

    def _change(self, t, v, v0, hits, obstacles, ego_xy, ego_yaw):
        p, r, dp = self.p, self.route, self.dp
        h = next((x for x in hits if KIND.get(x.obstacle.cls) in ('vehicle', 'cyclist')), None)
        slow = (h is not None and h.gap <= p.follow_range_m and h.v_lead >= dp.static_speed
                and h.v_lead < min(v0 - p.slow_dv, p.slow_ratio * v0) and h is hits[0])
        self.follow_t = self.follow_t + 1.0 if slow else 0.0          # counted in ticks by step()
        if not slow or self.follow_t * self._dt < p.follow_s or t < self.cool_until:
            return
        i = r.idx
        L = self.lanes
        side = ('left' if L['left_kind'][i] == SAME and L['left_ok'][i] else
                'right' if L['right_kind'][i] == SAME and L['right_ok'][i] else None)
        if side is None:
            self.why = 'no same-direction lane where a change is allowed'
            return
        c = self._lane_c(side)
        ci = float(c[i])
        l_in = ramp_length(ci, v, p, change=True)
        lead_len = float(h.obstacle.size[0])
        rel = max(v0 - h.v_lead, 0.5)
        overtake = (h.gap + lead_len + self.front + self.rear + 10.0) * v0 / rel
        need = l_in + min(overtake, 250.0) + l_in + 10.0
        s0 = r.s_ego
        j = self._next_junction(s0)
        room = min((j['s_in'] if j else math.inf), r.length) - s0
        if room < need:
            self.why = f'not enough road before the next junction / route end ({room:.0f} of {need:.0f} m)'
            return
        if np.any(np.isnan(c[self._span(s0, s0 + l_in + 20.0)])):
            self.why = f'the {side} lane does not continue'
            return
        knots = [(s0, 0.0), (s0 + l_in, 1.0)]
        cand = self._make_path(knots, c)
        for h2 in assess(obstacles, cand, dp, self.front):
            if h2.obstacle is not h.obstacle and h2.gap <= p.follow_range_m and h2.v_lead < h.v_lead + 1.0:
                self.why = f'the {side} lane is no faster ({h2.obstacle.cls} at {h2.v_lead * 3.6:.0f} km/h, {h2.gap:.0f} m)'
                return
        ok, why = self._no_oncoming_in(cand, obstacles)
        if ok:
            ok, why = self.clear(cand, s0 + l_in + 2.0 * max(v, 3.0), v, v0, obstacles, ego_xy, ego_yaw)
        if not ok:
            self.why = why
            return
        self.stats['lane_changes'] += 1
        self.side = side
        self.plan = dict(kind='change', lead=h.obstacle.cls, lead_v=round(h.v_lead, 2), s_start=s0,
                         s_ramp=s0 + l_in, t_over=None, lead_id=h.obstacle.track_id,
                         lead_xy=h.obstacle.center[:2].copy())
        self._set('change', knots, c)
        self._event('lane_change_start', side=side, lead_cls=h.obstacle.cls, lead_v=round(h.v_lead, 2),
                    gap=round(h.gap, 2), ramp_m=round(l_in, 1))
        self.follow_t = 0.0
        self.why = ''

    def _passed_lead(self, obstacles, ego_xy, ego_yaw):
        """the car being overtaken is behind the ego (or no longer seen)"""
        lid, last = self.plan.get('lead_id', -1), self.plan.get('lead_xy')
        lead = None
        for o in obstacles:
            if (lid >= 0 and o.track_id == lid) or (lid < 0 and last is not None
                                                     and float(np.hypot(*(o.center[:2] - last))) < 3.0):
                lead = o
                break
        if lead is None:
            return True
        self.plan['lead_xy'] = lead.center[:2].copy()
        rel = lead.center[:2] - np.asarray(ego_xy, float)
        x_rel = rel[0] * math.cos(ego_yaw) + rel[1] * math.sin(ego_yaw)
        return x_rel < -(self.rear + 0.5 * float(lead.size[0]) + 2.0)

    def _back(self, t, v, v0, obstacles, ego_xy, ego_yaw, force, route_hits):
        """try to change back to the route lane; returns True if it started"""
        p, r = self.p, self.route
        if not force:
            if self.plan.get('t_over') is None or t - self.plan['t_over'] < p.min_hold_s:
                return False
            if not self._passed_lead(obstacles, ego_xy, ego_yaw):
                self.why = 'not past the slow car yet'
                return False
            for h in route_hits:
                if h.gap < p.return_clear_m and h.v_lead < v0 - 1.0:
                    self.why = f'still a slower {h.obstacle.cls} {h.gap:.0f} m ahead in the route lane'
                    return False
        a0 = self.alpha_now()
        ci = float(self.c[self._i(r.s_ego)]) if np.ndim(self.c) else float(self.c)
        l_out = ramp_length(ci * a0, v, p, change=True)
        knots = [(r.s_ego, a0), (r.s_ego + l_out, 0.0)]
        cand = self._make_path(knots, self.c)
        ok, why = self.clear(cand, r.s_ego + l_out + 2.0 * max(v, 3.0), v, max(v, min(v0, 3.0)), obstacles,
                             ego_xy, ego_yaw)
        if not ok:
            self.why = 'way back not clear: ' + why
            return False
        self.stats['changes_back'] += 1
        self._set('back', knots, self.c)
        self._event('lane_change_back', side=self.side, forced=bool(force), ramp_m=round(l_out, 1))
        self.why = ''
        return True

    # ---------------------------------------------------------- yield
    def _yield(self, t, v, v0, obstacles, dt, ego_xy, ego_yaw):
        p, r, dp = self.p, self.route, self.dp
        ce, se = math.cos(ego_yaw), math.sin(ego_yaw)
        front = r.s_ego + self.front
        J = next((j for j in self.jx if j['s_out'] + self.rear >= r.s_ego), None)
        if J is None or J['s_in'] - front > p.yield_look_m:
            self.passing = []
            if self.yielding is not None:
                self._end_yield(t)
            return None
        lo, hi = J['s_in'] - 2.0, J['s_out'] + 3.0
        thr = self.hw + dp.margin
        v_go = max(min(v0, J['v_turn']), 2.0)
        win = hi - r.s_ego + 5.0
        ts = np.arange(0.0, p.yield_horizon_s + 1e-9, p.yield_step_s)
        found = []
        self.passing = []
        for ob in obstacles:
            if KIND.get(ob.cls) == 'pedestrian':
                continue                                   # --predict-crossing handles people
            vel = np.asarray(ob.velocity, float)[:2]
            sp = float(np.hypot(vel[0], vel[1]))
            if sp < dp.static_speed:
                continue                                   # standing: the corridor check handles it
            rel = ob.center[:2] - np.asarray(ego_xy, float)
            if rel[0] * ce + rel[1] * se < 0.0 and vel[0] * ce + vel[1] * se > 0.5 * sp:
                continue                                   # behind the ego, same direction: following it
            if float(np.hypot(*(ob.center[:2] - r.point_at(0.5 * (lo + hi))))) > 30.0 + sp * p.yield_horizon_s:
                continue
            s_c, d_c, _, past = r.project(ob.center[None, :2], win, return_beyond=True)
            tan = r.tangent_at(r.s_ego + float(s_c[0]))
            if not past[0] and abs(float(d_c[0])) < 2.5 and float(np.dot(vel, tan)) > math.cos(math.radians(35)) * sp:
                continue                                   # driving along our path: follow it, not yield
            P = (ob.corners[None] + ts[:, None, None] * vel[None, None]).reshape(-1, 2)
            _, d, s_abs, beyond = r.project(P, win, return_beyond=True)
            d, s_abs, beyond = d.reshape(len(ts), 4), s_abs.reshape(len(ts), 4), beyond.reshape(len(ts), 4)
            t_in = t_out = None
            s_lo, s_hi = math.inf, -math.inf
            for k in range(len(ts)):
                if beyond[k].any():
                    continue
                part = _strip_part(list(zip(s_abs[k], d[k])), thr, lo, hi)
                if part is None:
                    continue
                t_in = ts[k] if t_in is None else t_in
                t_out = ts[k] + p.yield_step_s
                s_lo, s_hi = min(s_lo, part[0]), max(s_hi, part[1])
            if t_in is None:
                continue
            if s_lo - front <= 0.0:
                continue                                   # the ego's bumper is already in that area
            te_in = ego_time(s_lo - front, v, v_go, p.a_go)
            te_out = ego_time(s_hi - front + self.front + self.rear, v, v_go, p.a_go)
            if t_in == 0.0 and t_out <= te_in - p.gap_s:
                self.passing.append(ob)                    # crossing now, gone before the ego arrives
            if t_in < te_out + p.gap_s and t_out > te_in - p.gap_s:
                found.append(dict(ob=ob, t_in=float(t_in), t_out=float(t_out), s_lo=s_lo, te_in=te_in,
                                  te_out=te_out))
        stop_s = None
        for f in found:
            for s_stop in (J['s_in'] - p.entry_margin, f['s_lo'] - 1.5):
                gap = s_stop - front
                if gap < 0.0:
                    continue
                need = v * v / (2.0 * max(gap, 0.1))
                if self.yielding is None and need > p.commit_decel:
                    continue                               # can no longer stop there: committed
                stop_s = s_stop if stop_s is None else min(stop_s, s_stop)
                break
        if stop_s is not None:
            f = min(found, key=lambda x: x['t_in'])
            if self.yielding is None:
                self.stats['yields'] += 1
                self.yielding = dict(t0=t, s_stop=stop_s, turn=turn_name(J['turn_deg']))
                self._event('yield_start', cls=f['ob'].cls, turn=turn_name(J['turn_deg']),
                            other_in_s=round(f['t_in'], 2), ego_in_s=round(f['te_in'], 2),
                            ego_out_s=round(f['te_out'], 2), gap_to_stop=round(stop_s - front, 2))
            self.yielding['s_stop'] = min(self.yielding['s_stop'], stop_s) if self.yielding.get('s_stop') else stop_s
            self.yielding['why'] = f"{f['ob'].cls} in {f['t_in']:.1f} s"
            self.yield_clear = 0
        elif self.yielding is not None:
            self.yield_clear += 1
            if self.yield_clear >= p.yield_release_n:
                self._end_yield(t)
        if self.yielding is None:
            return None
        self.stats['yield_s'] += dt
        return self.yielding['s_stop'] - front

    def _end_yield(self, t):
        self._event('yield_end', waited_s=round(t - self.yielding['t0'], 2))
        self.yielding = None
        self.yield_clear = 0

    # ---------------------------------------------------------- per tick
    def step(self, t, v, v0, ego_xy, ego_yaw, obstacles, hits, dt=0.05, route_hits=None):
        """hits: decision.assess() on the CURRENT path (self.path); route_hits: assess() on the base route
        (for the change back; None = same as hits). Returns dict(path, release, static_gap, stop, v_cap):
          release     a manoeuvre started that passes the obstacle decision.py has latched: release it
          static_gap  bumper gap to keep to a static obstacle (waiting to steer around it) or None
          stop        (gap from the bumper, 'yield' | 'merge') or None
          v_cap       speed limit while passing, m/s, or None"""
        p, r = self.p, self.route
        self._dt = dt
        if self.path is not r:
            self.path.update_progress(ego_xy)
        released, static_gap, stop, v_cap = False, None, None, None
        mode0 = self.mode
        # ---- manoeuvre in progress ----
        if self.mode == 'avoid':
            v_cap = self.plan['v_pass']
            if r.s_ego >= self.plan['s_end']:
                self._event('steer_around_done')
                self._set(None)
            elif self.alpha_now() < 0.5 and r.s_ego < self.knots[1][0]:
                ok, why = self.clear(self.path, self.plan['s_end'], v, self.plan['v_pass'], obstacles, ego_xy,
                                     ego_yaw, start=False, static_ids=self._target_ids(obstacles))
                if not ok:
                    self._abort(t, v, why)
        elif self.mode == 'change':
            if r.s_ego >= self.plan['s_ramp']:
                self.mode = 'hold'
            elif self.alpha_now() < 0.5:
                ok, why = self.clear(self.path, self.plan['s_ramp'] + 2.0 * max(v, 3.0), v, v0, obstacles,
                                     ego_xy, ego_yaw, start=False)
                if not ok:
                    self._abort(t, v, why)
        if self.mode == 'hold':
            rh = hits if route_hits is None else route_hits
            if self.plan.get('t_over') is None:
                self.plan['t_over'] = t
            j = self._next_junction(r.s_ego)
            ci = float(self.c[self._i(r.s_ego)]) if np.ndim(self.c) else float(self.c)
            l_back = ramp_length(ci, max(v, 3.0), p, change=True)
            latest = min((j['s_in'] if j else math.inf), r.length) - l_back - 5.0
            lane_ends = np.isnan(self.c[self._span(r.s_ego, r.s_ego + l_back + 30.0)]).any()
            force = r.s_ego + self.front >= latest - 30.0 or lane_ends
            if not self._back(t, v, v0, obstacles, ego_xy, ego_yaw, force, rh) and force:
                stop = (max(latest - r.s_ego - self.front, 0.0), 'merge')
        elif self.mode == 'back':
            if r.s_ego >= self.knots[-1][0] or (v < 0.3 and abs(self.offset_now()) < 0.1):
                self._event('lane_change_done' if self.plan.get('kind') == 'change' else 'back_in_lane')
                self._set(None)
        # ---- start one (also again after an aborted steer-around, from where the car is) ----
        retry = self.mode == 'back' and self.plan.get('kind') == 'avoid' and mode0 == 'back'
        if (self.mode is None and mode0 is None) or retry:
            if p.steer_around:
                static_gap = self._avoid(t, v, hits, obstacles, ego_xy, ego_yaw)
                released = self.mode == 'avoid' and self.plan.get('started') == t
            if self.mode is None and p.lane_change:
                self._change(t, v, v0, hits, obstacles, ego_xy, ego_yaw)
                released = released or self.mode == 'change'
        if p.yield_at_junctions:
            g = self._yield(t, v, v0, obstacles, dt, ego_xy, ego_yaw)
            if g is not None and (stop is None or g < stop[0]):
                stop = (g, 'yield')
        self.status = (('yield' if stop and stop[1] == 'yield' else 'merge_wait' if stop else '')
                       or {'avoid': 'steer_around', 'change': 'lane_change', 'hold': 'other_lane',
                           'back': 'change_back'}.get(self.mode, '')
                       or ('wait_gap' if static_gap is not None else ''))
        return dict(path=self.path, release=released, static_gap=static_gap, stop=stop, v_cap=v_cap)

    def _abort(self, t, v, why):
        a0 = self.alpha_now()
        ci = float(self.c[self._i(self.route.s_ego)]) if np.ndim(self.c) else float(self.c)
        l_out = ramp_length(ci * a0, v, self.p)
        self.stats['aborts'] += 1
        self._event('manoeuvre_abort', manoeuvre=self.mode, why=why, alpha=round(a0, 2))
        self.plan['aborted'] = True
        self._set('back', [(self.route.s_ego, a0), (self.route.s_ego + l_out, 0.0)], self.c)
        self.cool_until = t + self.p.cooldown_s