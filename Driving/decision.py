"""
decision.py -- what speed to drive at, and why. Pure python/numpy, no CARLA.

Method (the agreed plan):
  corridor   an obstacle blocks the path when part of its footprint lies inside
             |d| <= W/2 + margin AND ahead of the ego's front bumper; the gap is to
             that part (footprint clipped in path coordinates, so a truck whose
             centre is outside the lane and a bus across it are both caught). A car
             already level with the ego (beside it) counts only if part of it ahead
             of the bumper is inside the ego's own width |d| <= W/2 (a cut-in): one
             that only reaches into the margin alongside is not braked for.
  IDM        a = a_max [1 - (v/v0)^4 - (s*/s)^2],  s* = s0 + vT + v dv / (2 sqrt(a_max b))
             Everything that should slow the car is a virtual leader; take the minimum a.
               static obstacle  gap - static_buffer, v_lead 0
               moving obstacle  gap, its speed along the path
               route end        gap = remaining + s0, so IDM settles at the end
               curves           v0 capped by route.speed_cap()
  emergency  TTC = gap / (v - v_lead) < 1.5 s, or gap < 3 m while moving -> full brake
  hysteresis per NEW perception output: occupied in 3 of the last 5 -> confirmed;
             clear 10 in a row -> released. While confirmed, the obstacle stays
             latched at its path position, so one missed detection cannot release it.
  speed ref  v_ref integrates IDM's acceleration (never more than 1 m/s from the
             actual speed); the speed PID tracks v_ref, so it sees a smooth target.
  crossing   (optional) a pedestrian walking towards the path who will be inside it within
             2 s is treated as a static obstacle at the point where it will enter
  lights     (optional) red / yellow light at a stop line on the route: a virtual static
             leader so the car stops 1 m before the line. Decided once per light phase: stop
             if it can (yellow: comfortably, red: <= 4 m/s^2), else go through
  crash      after hitting a pedestrian the car stops for good (CRASH_STOP)
  yield      (optional, manoeuvre.py) a place to stop that is not an obstacle: a junction entry while
             yielding, or the last point to get back into the route lane. A virtual static leader
             like the light; YIELD while it is what slows the car
  manoeuvre  manoeuvre.py may ask for a larger standstill gap to a static obstacle (room to pull out
             around it) and calls release() when a manoeuvre starts that passes the latched obstacle
  FSM        CRASH_STOP > ROUTE_COMPLETE > EMERGENCY_STOP > DEGRADED > STOP_FOR_LIGHT / YIELD >
             BLOCKED_WAIT > FOLLOW_ROUTE
"""
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

FOLLOW_ROUTE = 'FOLLOW_ROUTE'
BLOCKED_WAIT = 'BLOCKED_WAIT'
EMERGENCY_STOP = 'EMERGENCY_STOP'
DEGRADED = 'DEGRADED'
ROUTE_COMPLETE = 'ROUTE_COMPLETE'
STOP_FOR_LIGHT = 'STOP_FOR_LIGHT'
CRASH_STOP = 'CRASH_STOP'
YIELD = 'YIELD'
PEDESTRIAN_CLASSES = ('pedestrian', 'Pedestrian')      # oracle name, detector name


@dataclass
class Params:
    v0_kmh: float = 30.0          # desired cruise speed
    T: float = 1.5                # time headway, s
    s0: float = 3.0               # standstill gap, m
    a_max: float = 1.5            # comfortable acceleration, m/s^2
    b: float = 2.0                # comfortable deceleration, m/s^2
    b_max: float = 8.0            # physical limit used for clipping
    static_buffer: float = 1.0    # extra gap kept to static obstacles, m
    static_speed: float = 0.5     # below this an obstacle counts as static, m/s
    static_release: float = 0.5   # a static leader counts as moving again only above this, m/s (agent.py
                                  # sets 1.0 for detector sources: box jitter, see world_model.py)
    ego_half_width: float = 1.0   # overwritten from the ego's bounding box
    margin: float = 0.4           # lateral safety margin, m
    horizon: float = 60.0         # how far ahead the corridor is checked, m
    ttc_emergency: float = 1.5
    gap_emergency: float = 3.0
    a_lat_max: float = 2.0        # curve speed cap
    confirm_n: int = 3
    confirm_window: int = 5
    release_n: int = 10
    stopped_speed: float = 0.3
    goal_radius: float = 3.0
    degraded_decel: float = 3.0
    healthy_to_recover: int = 10
    ref_lead: float = 1.0         # max |v_ref - v|, m/s
    predict_s: float = 2.0        # crossing pedestrians: look this far into the future, s
    predict_step: float = 0.25
    approach_speed: float = 0.5   # ... only if walking towards the path at least this fast, m/s
    light_margin: float = 1.0     # stop the bumper this far before the stop line, m
    light_decel_max: float = 4.0  # red light: stop if this deceleration is enough, else go through


@dataclass
class Hit:
    s_abs: float      # path position of the obstacle's nearest edge
    gap: float        # ego front bumper to that edge, along the path
    v_lead: float     # obstacle speed along the path
    obstacle: object
    kind: str = 'in_path'     # 'in_path' now, or 'crossing' (predicted to enter the path)


@dataclass
class Decision:
    state: str
    v_target: float               # m/s, for the speed PID
    a_cmd: float                  # m/s^2, what IDM asked for
    hold_brake: bool
    emergency: bool
    v0: float
    lead_gap: Optional[float]
    lead_ttc: Optional[float]
    n_hits: int
    lead_kind: str = ''           # cls (+ ' crossing') of the obstacle the car reacts to
    light: Optional[tuple] = None  # (state, gap to the stop line, stopping for it) or None
    stop: Optional[tuple] = None   # (kind, gap) of a yield / merge stop, or None


def idm(v, v0, p, gap=None, v_lead=0.0):
    free = 1.0 - (v / max(v0, 0.1)) ** 4
    if gap is None:
        return p.a_max * free
    if gap <= 0.1:
        return -p.b_max
    dv = v - v_lead
    s_star = p.s0 + max(0.0, v * p.T + v * dv / (2.0 * math.sqrt(p.a_max * p.b)))
    return p.a_max * (free - (s_star / gap) ** 2)


def _clip(poly, a_s, a_d, b):
    """Part of polygon poly [(s, d), ...] where a_s*s + a_d*d <= b (Sutherland-Hodgman)."""
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
    return out


def _nearest_inside(poly, s_min, half_w):
    """Smallest s of the part of poly with s >= s_min and |d| <= half_w, or None."""
    for a_s, a_d, b in ((-1.0, 0.0, -s_min), (0.0, 1.0, half_w), (0.0, -1.0, half_w)):
        poly = _clip(poly, a_s, a_d, b)
        if not poly:
            return None
    return min(q[0] for q in poly)


def assess(obstacles, route, p, ego_front) -> List[Hit]:
    """Obstacles in the corridor ahead of the front bumper, nearest first. The gap
    is from the bumper to the nearest part of the obstacle inside the corridor, so
    it is never negative (see 'corridor' at the top of this file)."""
    thr = p.ego_half_width + p.margin
    hits = []
    for ob in obstacles:
        s_rel, d, s_abs, beyond = route.project(ob.corners, p.horizon, return_beyond=True)
        if beyond.all():
            continue                                   # wholly past the route end / horizon
        if beyond.any():
            # partly past the route end or horizon, so far ahead: the corners past it
            # do not project, so test the others as an interval
            s_rel, d = s_rel[~beyond], d[~beyond]
            if s_rel.min() < ego_front or d.min() > thr or d.max() < -thr:
                continue
            near = float(s_rel.min())
        else:
            poly = list(zip(s_rel.tolist(), d.tolist()))      # footprint() corners are in order
            if s_rel.min() >= ego_front:                      # wholly ahead of the bumper
                near = _nearest_inside(poly, ego_front, thr)
            else:                                             # level with the ego: cut-ins only
                near = _nearest_inside(poly, ego_front, p.ego_half_width)
            if near is None:
                continue
        if near > p.horizon:
            continue                                   # too far
        s_near = route.s_ego + near
        v_lead = float(np.dot(ob.velocity, route.tangent_at(s_near)))
        hits.append(Hit(s_near, near - ego_front, v_lead, ob))
    hits.sort(key=lambda h: h.gap)
    return hits


def predict_crossing(obstacles, route, p, ego_front) -> List[Hit]:
    """Pedestrians walking towards the path who will be inside it (|d| <= W/2 + margin, ahead of
    the bumper) within p.predict_s seconds, at constant velocity. Each becomes a static Hit at the
    point where it will enter, so the car slows or stops BEFORE the person steps in. People
    walking along the kerb (no speed towards the path) are ignored."""
    thr = p.ego_half_width + p.margin
    hits = []
    for ob in obstacles:
        if ob.cls not in PEDESTRIAN_CLASSES:
            continue
        vel = np.asarray(ob.velocity, float)[:2]
        if float(np.hypot(vel[0], vel[1])) < 0.3:
            continue
        s0_, d0, _, past = route.project(ob.center[None, :2], p.horizon, return_beyond=True)
        if past[0] or s0_[0] < ego_front - 1.0 or abs(d0[0]) <= thr:
            continue                        # beyond the horizon, behind the bumper, or already in path
        dt_probe = 0.5
        _, d1, _, _ = route.project((ob.center[:2] + vel * dt_probe)[None], p.horizon, return_beyond=True)
        if abs(d0[0]) - abs(d1[0]) < p.approach_speed * dt_probe:
            continue                        # not walking towards the path
        for t in np.arange(p.predict_step, p.predict_s + 1e-9, p.predict_step):
            corners = ob.corners + vel * t
            s_rel, d, _, beyond = route.project(corners, p.horizon, return_beyond=True)
            if beyond.any():
                break
            near = _nearest_inside(list(zip(s_rel.tolist(), d.tolist())), ego_front, thr)
            if near is not None:
                hits.append(Hit(route.s_ego + near, near - ego_front, 0.0, ob, 'crossing'))
                break
    return hits


class Decider:
    def __init__(self, p: Params):
        self.p = p
        self.state = DEGRADED            # start held until perception reports healthy
        self.hist = deque(maxlen=p.confirm_window)
        self.clear_count = 0
        self.latched = None              # {'s_abs', 'v_lead', 'cls'}
        self.lead_static = False         # the latched leader counted as static last tick (hysteresis)
        self.healthy_count = 0
        self.v_ref = 0.0
        self.events = []                 # drained by agent.py into the evaluator
        self.crashed = False             # set by crash(): stop for good
        self.light_plan = None           # (light id, 'stop' | 'go', state) for the current red/yellow phase

    def release(self, why):
        """a manoeuvre now passes the latched obstacle (steer around, lane change): forget it"""
        if self.latched is not None:
            self._event('obstacle_released', reason=why)
        self.latched = None
        self.lead_static = False
        self.hist.clear()
        self.clear_count = 0

    def crash(self, what):
        """called by agent.py when the ego hits a pedestrian while moving"""
        if not self.crashed:
            self.crashed = True
            self._event('crash_stop', hit=what)

    def _event(self, kind, **kw):
        self.events.append(dict(type=kind, **kw))

    def _hysteresis(self, hits, fresh, dt):
        p = self.p
        if self.latched is not None:      # a moving leader keeps moving between frames
            self.latched['s_abs'] += self.latched['v_lead'] * dt
        if not fresh:
            return
        occupied = bool(hits)
        self.hist.append(occupied)
        if occupied:
            self.clear_count = 0
            if self.latched is not None or sum(self.hist) >= p.confirm_n:
                h = hits[0]
                if self.latched is None or self.latched.get('track') != h.obstacle.track_id:
                    self.lead_static = False          # another leader: no hysteresis from the last one
                if self.latched is None:
                    self._event('obstacle_confirmed', gap=round(h.gap, 2),
                                cls=h.obstacle.cls, track=h.obstacle.track_id, reason=h.kind)
                self.latched = {'s_abs': h.s_abs, 'v_lead': h.v_lead, 'cls': h.obstacle.cls,
                                'kind': h.kind, 'track': h.obstacle.track_id}
        else:
            self.clear_count += 1
            if self.latched is not None and self.clear_count >= p.release_n:
                self._event('obstacle_released')
                self.latched = None

    def _light(self, v, light):
        """light = (stop-line gap from the bumper, state, light id) or None.
        Returns the distance to stop at (bumper to 1 m before the line) or None."""
        p = self.p
        if light is None:
            self.light_plan = None
            return None
        gap, state, lid = light
        if state not in ('Red', 'Yellow'):
            self.light_plan = None                 # green / off / unknown: drive
            return None
        # decide once per light and phase; a 'go' on yellow is re-checked when it turns red
        if (self.light_plan is None or self.light_plan[0] != lid
                or (self.light_plan[1] == 'go' and self.light_plan[2] != state)):
            room = max(gap - p.light_margin, 0.1)
            need = v * v / (2.0 * room)            # constant deceleration to stop in time
            ok = v < p.stopped_speed or need <= (p.b if state == 'Yellow' else p.light_decel_max)
            self.light_plan = (lid, 'stop' if ok else 'go', state)
            self._event('light', state=state, gap=round(gap, 2), plan=self.light_plan[1],
                        need_decel=round(need, 2))
        if self.light_plan[1] == 'go':
            return None
        return max(gap - p.light_margin, 0.0)

    def step(self, v, s_ego, remaining, v_cap, hits, fresh, healthy, ego_front, dt, light=None, stop=None,
             static_gap=None):
        """stop: (bumper gap to a place to stop at, kind) from manoeuvre.py (yield / merge) or None;
        static_gap: bumper gap to keep to a static obstacle, default s0 + static_buffer"""
        p = self.p
        buf = p.static_buffer if static_gap is None else max(static_gap - p.s0, p.static_buffer)
        self._hysteresis(hits, fresh, dt)
        self.healthy_count = self.healthy_count + 1 if healthy else 0
        v0 = max(min(p.v0_kmh / 3.6, v_cap), 0.5)

        # ---- longitudinal: IDM over every virtual leader --------------------
        a = idm(v, v0, p)
        lead_gap = lead_ttc = None
        static = False
        if self.latched is not None:
            vl = self.latched['v_lead']
            static = abs(vl) < (p.static_release if self.lead_static else p.static_speed)
            lead_gap = self.latched['s_abs'] - s_ego - ego_front
            gap = lead_gap - (buf if static else 0.0)
            a = min(a, idm(v, v0, p, gap, max(vl, 0.0)))
            closing = v - vl
            lead_ttc = lead_gap / closing if closing > 0.1 else math.inf
        self.lead_static = static
        a = min(a, idm(v, v0, p, remaining + p.s0, 0.0))
        stop_at = self._light(v, light)
        a_light = None
        if stop_at is not None:                     # IDM settles at gap s0: bumper 1 m before the line
            a_light = idm(v, v0, p, stop_at + p.s0, 0.0)
            a = min(a, a_light)
        stop_gap = a_stop = None
        if stop is not None:                        # yield / merge: IDM settles with the bumper at it
            stop_gap = max(float(stop[0]), 0.0)
            a_stop = idm(v, v0, p, stop_gap + p.s0, 0.0)
            a = min(a, a_stop)
        a = float(np.clip(a, -p.b_max, p.a_max))

        # ---- state machine -------------------------------------------------
        prev = self.state
        hazard = lead_gap is not None and v > 0.5 and (lead_ttc < p.ttc_emergency or
                                                       lead_gap < p.gap_emergency)   # a stopped car cannot brake harder
        obstacle_stop = None if lead_gap is None else lead_gap - (buf if static else 0.0)
        if self.crashed:
            self.state = CRASH_STOP
        elif self.state == ROUTE_COMPLETE or remaining < p.goal_radius:
            self.state = ROUTE_COMPLETE
        elif hazard:
            self.state = EMERGENCY_STOP
        elif self.state == EMERGENCY_STOP and v > p.stopped_speed:
            pass                                      # stay until actually stopped
        elif not healthy or (self.state == DEGRADED and
                             self.healthy_count < p.healthy_to_recover):
            self.state = DEGRADED
        elif (stop_at is not None and (obstacle_stop is None or stop_at <= obstacle_stop - p.s0)
              and (a_light < (0.3 if self.state == STOP_FOR_LIGHT else -0.1)
                   or (v < p.stopped_speed and stop_at < 1.5))):
            self.state = STOP_FOR_LIGHT               # the light is the nearer place to stop and it
                                                      # slows the car now (or it waits at the line)
        elif (stop_gap is not None and (obstacle_stop is None or stop_gap <= obstacle_stop - p.s0)
              and (a_stop < (0.3 if self.state == YIELD else -0.1)
                   or (v < p.stopped_speed and stop_gap < 1.5))):
            self.state = YIELD                        # the same for a yield / merge stop
        elif static and (self.state == BLOCKED_WAIT or
                         (v < p.stopped_speed and lead_gap - buf < p.s0 + 2.0)):
            self.state = BLOCKED_WAIT                 # stopped BEHIND it, not just stopped
        else:
            self.state = FOLLOW_ROUTE

        # ---- outputs --------------------------------------------------------
        hold = False
        if self.state in (FOLLOW_ROUTE, STOP_FOR_LIGHT, YIELD):
            if prev not in (FOLLOW_ROUTE, STOP_FOR_LIGHT, YIELD):
                self.v_ref = v
            self.v_ref = float(np.clip(self.v_ref + a * dt, 0.0, v0))
            self.v_ref = float(np.clip(self.v_ref, v - p.ref_lead, v + p.ref_lead))
            v_target = max(self.v_ref, 0.0)
            if self.state == STOP_FOR_LIGHT and v < p.stopped_speed and stop_at < 1.5:
                v_target, hold = 0.0, True            # waiting at the line: hold, do not creep
            if self.state == YIELD and v < p.stopped_speed and stop_gap < 1.5:
                v_target, hold = 0.0, True
        else:
            v_target = 0.0
            self.v_ref = v
            if self.state in (EMERGENCY_STOP, CRASH_STOP):
                a = -p.b_max
            elif self.state == DEGRADED:
                a = min(a, -p.degraded_decel)
            else:
                a = min(a, -p.b)
            hold = v < p.stopped_speed or self.state == CRASH_STOP

        if self.state != prev:
            self._event('state', frm=prev, to=self.state, v=round(v, 2),
                        gap=None if lead_gap is None else round(lead_gap, 2),
                        ttc=None if lead_ttc is None or math.isinf(lead_ttc) else round(lead_ttc, 2),
                        healthy=bool(healthy))
        lead_kind = ''
        if self.latched is not None:
            lead_kind = self.latched['cls'] + (' crossing' if self.latched.get('kind') == 'crossing' else '')
        light_info = None if light is None else (light[1], light[0], stop_at is not None)
        stop_info = None if stop is None else (stop[1], stop_gap)
        return Decision(self.state, v_target, a, hold, self.state in (EMERGENCY_STOP, CRASH_STOP), v0,
                        lead_gap, lead_ttc, len(hits), lead_kind, light_info, stop_info)