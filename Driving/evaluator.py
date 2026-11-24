"""
evaluator.py -- ground-truth scoring of one run. Never feeds back into driving.

Writes 4 files into <out_dir>/<run_name>/:
  ticks.csv        one row per tick: pose, speed, state, commands, perception ages, GT gap
  events.jsonl     state changes, obstacle confirmed/released, first brake, collisions
  summary.json     metrics, project score, pass/fail of the scenario's milestone
  trajectory.png   route vs driven path coloured by speed (skipped without matplotlib)

Project score = route completion x infraction penalty, with CARLA Leaderboard 2.0
coefficients per collision (pedestrian 0.50, vehicle 0.60, static 0.65). It is
a "project score with LB 2.0 coefficients", not an official Leaderboard result:
our evaluator, CARLA 0.9.15, no scenario_runner. Red-light (0.70) joins in M5.

Pass thresholds marked (heuristic) are my choices, not from the Leaderboard.

Traffic runs (--traffic N) add:
  collision fault   from the ego's speed and where the other actor was at impact:
                    ego stopped (< 0.1 m/s the tick before) -> 'hit while stopped'; a map
                    object (pole, wall) -> 'ego hit a static object'; other ahead (within
                    60 deg of the ego's heading) -> 'ego drove into it'; other behind ->
                    'hit from behind'; otherwise 'side contact while moving'. Ego-caused =
                    drove into it, static object, side contact, or unknown. A heuristic,
                    labelled as such; the score still applies the LB penalty to every
                    collision, like the LB.
  junction split    the ego ignores traffic lights until step 4 and cannot predict
                    crossing cars until step 5, so junction conflicts (the nearest car
                    in the path is crossing it, 30-150 deg off, with it or the ego in a
                    junction; for collisions also: the ego inside a junction) are
                    reported separately and not gated yet
  min TTC           from CARLA ground truth (gt_ttc column), never from the agent's belief
  following         a run only counts if a car was actually followed: >= 10 s with a
                    ground-truth car in the ego's path within 40 m (agent.py puts a
                    slower lead car on the ego route for this). Without it, 'no collision'
                    and 'TTC None' only mean nothing was ever ahead.

Manoeuvre runs (manoeuvre.py) add, from ground-truth footprints of the cars near the ego:
  clearance         smallest gap between the ego's and another car's footprint at the same tick
  PET               post-encroachment time: the least time between the ego and another car being on
                    the same spot, either order (0 = collision). Only cars whose heading differs from
                    the ego's by > 30 deg there (crossing / oncoming), so following a car is not a conflict
  M6  steer around  (static + --steer-around) passed the obstacle, steered (not only waited), clearance
                    to it >= 0.3 m, back in the route lane, no collision, PET >= 1 s with oncoming cars
  M7  lane change   (overtake + --lane-change) changed lane, passed the slow car, back in the route lane,
                    no collision / emergency stop, clearance to cars >= 0.5 m, TTC >= 1.5 s
  M8  junction      (junction + --yield) through the junction, traffic crossed the ego's path (else
                    yielding was not tested), PET >= 1 s, no collision
  These thresholds are heuristics, like the others marked so.
"""
import bisect
import csv
import json
import math
import os
from collections import Counter

import numpy as np

PENALTY = {'pedestrian': 0.50, 'vehicle': 0.60, 'static': 0.65}
MAX_DEV_M = 2.0                  # M2 exit criterion from the plan
MAX_REVERSALS_PER_MIN = 30.0     # steering oscillation (heuristic)
MIN_STOP_GAP_M = 2.0             # M4 exit criterion from the plan
MAX_STOP_EXTRA_M = 4.0           # M4: final gap at most s0 + static_buffer + this (heuristic)
PEDAL_ON = 0.05                  # pedal threshold for the back-and-forth metrics (heuristic)
MIN_FOLLOW_S = 10.0              # step 3: following must actually happen (heuristic)
FOLLOW_RANGE_M = 40.0            # a car in the path this close counts as being followed
MIN_PASS_CLEAR_M = 0.3           # M6: passing a stopped obstacle (heuristic)
MIN_CAR_CLEAR_M = 0.5            # M7: other cars during a lane change (heuristic)
MIN_PET_S = 1.0                  # M6 / M8: post-encroachment time with crossing / oncoming cars (heuristic)
CONFLICT_PET_S = 5.0             # a car this close in time to the ego's spot = traffic really crossed its path


def steer_reversals(steer, moving, dt, amp=0.05):
    """Direction reversals of the steering command larger than `amp`, counted
    only while moving. Returns (reversals per minute, RMS steering rate per s)."""
    s = np.asarray(steer)[np.asarray(moving)]
    if len(s) < 3:
        return 0.0, 0.0
    rev, trend, ext = 0, 0, s[0]
    for x in s[1:]:
        if trend == 0:
            if abs(x - ext) > amp:
                trend, ext = (1 if x > ext else -1), x
        elif trend == 1:
            if x > ext:
                ext = x
            elif ext - x > amp:
                trend, ext, rev = -1, x, rev + 1
        else:
            if x < ext:
                ext = x
            elif x - ext > amp:
                trend, ext, rev = 1, x, rev + 1
    minutes = len(s) * dt / 60.0
    return rev / minutes, float(np.sqrt(np.mean((np.diff(s) / dt) ** 2)))


def comfort_metrics(rows, dt):
    """The plan's comfort metrics (RMS jerk, hard-brake events) plus two pedal
    diagnostics. Reported, not gated: thresholds are not agreed yet."""
    v = np.array([r['speed'] for r in rows], float)
    if len(v) < 10:
        return {}
    vs = np.convolve(np.pad(v, 2, mode='edge'), np.ones(5) / 5, mode='valid')   # 0.25 s smoothing,
                                                                              # edge-padded (zero padding faked a stop)
    a = np.gradient(vs, dt)
    jerk = np.gradient(a, dt)
    moving = v > 1.0
    hard = (a < -4.0) & moving
    hard_events = int(np.sum(np.diff(np.r_[0, hard.astype(int)]) == 1))
    th = np.array([r['throttle'] for r in rows], float)
    br = np.array([r['brake'] for r in rows], float)
    pedal = np.where(br > 0.01, -1, np.where(th > 0.01, 1, 0))[moving]
    switches = int(np.sum(np.diff(pedal) != 0))
    # what actually matters: A -> B -> A within 1 s (jerky back-and-forth), and
    # throttle <-> brake with no coasting between. Slowing for a curve is neither.
    # A pedal counts as pressed above PEDAL_ON: a throttle flickering 0.009 <-> 0.019
    # while holding a gap is not a pedal change anyone would feel.
    pedal_all = np.where(br > PEDAL_ON, -1, np.where(th > PEDAL_ON, 1, 0))
    back_forth = direct = 0
    prev, last = None, None                     # last = (from, to, time) of the previous switch
    for i in range(len(rows)):
        if not moving[i]:
            prev, last = None, None
            continue
        c = int(pedal_all[i])
        if prev is not None and c != prev:
            t = i * dt
            if last is not None and c == last[0] and prev == last[1] and t - last[2] < 1.0:
                back_forth += 1
            if abs(c - prev) == 2:
                direct += 1
            last = (prev, c, t)
        prev = c
    minutes = max(moving.sum() * dt / 60.0, 1e-9)
    follow = np.array([r['state'] == 'FOLLOW_ROUTE' for r in rows]) & moving
    a_cmd = np.array([r['a_cmd'] for r in rows], float)
    needless = follow & (br > 0.01) & (a_cmd > -0.5)
    return dict(rms_jerk=round(float(np.sqrt(np.mean(jerk[moving] ** 2))) if moving.any() else 0.0, 2),
                hard_brake_events=hard_events,
                pedal_back_and_forth=back_forth, direct_throttle_brake=direct,
                pedal_switches_per_min=round(float(switches / minutes), 1),
                brake_while_cruising_pct=round(float(100.0 * needless.sum() / max(follow.sum(), 1)), 1))


class Evaluator:
    def __init__(self, out_dir, run_name, meta, route_xy, dt):
        self.dir = os.path.join(out_dir, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.meta, self.dt = meta, dt
        self.route_xy = np.asarray(route_xy)
        self.rows, self.events, self.polygons = [], [], []
        self._raw_collisions = []
        self.lane_invasions = 0
        self.junction_fn = None          # (x, y) -> bool, set by agent.py from the CARLA map
        self.others = []                 # (k, id, x, y, yaw rad, length, width): GT cars near the ego

    # ---- live inputs -------------------------------------------------------
    def tick(self, **row):
        self.rows.append(row)

    def event(self, k, frame, kind, **kw):
        self.events.append(dict(k=k, frame=frame, type=kind, **kw))

    def on_collision(self, e):          # CARLA sensor thread
        imp = e.normal_impulse
        other = e.other_actor
        try:                            # pose at impact, for the fault rule
            et, ev, ot = e.actor.get_transform(), e.actor.get_velocity(), other.get_transform()
            pose = (et.location.x, et.location.y, et.rotation.yaw, math.hypot(ev.x, ev.y),
                    ot.location.x, ot.location.y, ot.rotation.yaw)
        except Exception:
            pose = None
        self._raw_collisions.append((e.frame, getattr(other, 'id', -1),
                                     getattr(other, 'type_id', 'unknown'),
                                     math.sqrt(imp.x ** 2 + imp.y ** 2 + imp.z ** 2), pose))

    def on_lane_invasion(self, e):      # diagnostic only (computed client-side by CARLA)
        self.lane_invasions += 1

    @property
    def has_collision(self):
        return bool(self._raw_collisions)

    # ---- scoring -----------------------------------------------------------
    def collisions(self):
        """Merge the several events CARLA sends per contact into incidents (same
        other actor within 1 s = one incident). Map objects (poles, walls) all
        report id 0, so incidents are keyed by (id, type)."""
        out, last = [], {}
        gap = int(round(1.0 / self.dt))
        row_frames = [r['frame'] for r in self.rows]
        for frame, oid, tid, inten, pose in sorted(self._raw_collisions, key=lambda c: (c[0], c[1])):
            key = (oid, tid)
            if key in last and frame - last[key] <= gap:
                last[key] = frame
                continue
            last[key] = frame
            cat = ('pedestrian' if tid.startswith('walker') else
                   'vehicle' if tid.startswith('vehicle') else 'static')
            # the event arrives after the physics step that contained the impact, when
            # the ego may already have been stopped by it: judge from the last tick before
            i = bisect.bisect_left(row_frames, frame) - 1
            r = self.rows[i] if i >= 0 else None
            ego = (r['x'], r['y'], r['yaw'], r['speed']) if r is not None else (pose[:4] if pose else None)
            fault, junction = 'unknown', None
            if ego is not None and ego[3] < 0.1:
                fault = 'hit while stopped'
            elif cat == 'static':
                fault = 'ego hit a static object'        # map objects have no usable position
            elif ego is not None and pose is not None:
                ex, ey, eyaw, _ = ego
                ox, oy = pose[4], pose[5]
                h = math.radians(eyaw)
                fwd = (ox - ex) * math.cos(h) + (oy - ey) * math.sin(h)
                side = -(ox - ex) * math.sin(h) + (oy - ey) * math.cos(h)
                ang = abs(math.degrees(math.atan2(side, fwd)))
                fault = ('ego drove into it' if ang <= 60.0 else 'hit from behind' if ang >= 120.0
                         else 'side contact while moving')
            flagged = False
            if r is not None and r.get('ego_in_junction') is not None:
                junction = bool(r['ego_in_junction'])          # ego inside a junction (3D), tick before
                flagged = bool(r.get('junction_conflict'))     # a crossing car was ahead at a junction
            elif ego is not None and self.junction_fn is not None:
                try:
                    junction = bool(self.junction_fn(ego[0], ego[1]))
                except Exception:
                    junction = None
            # exempt only crossing-type conflicts (the ego runs lights until step 4): the other
            # actor's heading 30-150 deg off the ego's. A rear-end or a static hit is always gated.
            crossing = (cat != 'static' and ego is not None and pose is not None and len(pose) >= 7
                        and abs(math.sin(math.radians(pose[6] - ego[2]))) > 0.5)
            out.append(dict(frame=frame, other_id=oid, other_type=tid, category=cat,
                            intensity=round(inten, 1), fault=fault,
                            ego_caused=fault in ('ego drove into it', 'side contact while moving',
                                                 'ego hit a static object', 'unknown'),
                            in_junction=junction,
                            junction_conflict=bool(crossing and (bool(junction) or flagged))))
        return out

    def finish(self, scenario, extra=None):
        extra = dict(extra or {})
        rows = self.rows
        coll = self.collisions()
        for c in coll:
            self.event(None, c['frame'], 'collision', **{k: v for k, v in c.items() if k != 'frame'})
        if not rows:
            summary = dict(meta=self.meta, scenario=scenario, ticks=0, passed=False,
                           note='no ticks recorded', **extra)
            self._write(summary)
            return summary

        speed = np.array([r['speed'] for r in rows])
        dev = np.array([r['lat_dev'] for r in rows])
        steer = np.array([r['steer'] for r in rows])
        moving = speed > 2.0
        rev_min, steer_rms = steer_reversals(steer, moving, self.dt)
        comfort = comfort_metrics(rows, self.dt)
        completed = any(r['state'] == 'ROUTE_COMPLETE' for r in rows)
        rc = 100.0 if completed else 100.0 * max(r['completion'] for r in rows)
        ip = 1.0
        for c in coll:
            ip *= PENALTY[c['category']]

        inter = interactions(rows, self.others, self.meta.get('ego_box'), self.dt)
        summary = dict(
            meta=self.meta, scenario=scenario, ticks=len(rows),
            sim_seconds=round(len(rows) * self.dt, 2),
            route_length_m=round(float(np.sum(np.linalg.norm(np.diff(self.route_xy, axis=0), axis=1))), 1),
            route_completed=completed, route_completion_pct=round(rc, 1),
            max_lateral_dev_m=round(float(dev.max()), 3),
            mean_lateral_dev_m=round(float(dev.mean()), 3),
            steer_reversals_per_min=round(rev_min, 1), steer_rate_rms=round(steer_rms, 3),
            max_speed_kmh=round(float(speed.max() * 3.6), 1),
            mean_moving_speed_kmh=round(float(speed[moving].mean() * 3.6), 1) if moving.any() else 0.0,
            collisions=coll, lane_invasion_events=self.lane_invasions,
            infraction_penalty=round(ip, 4), project_score=round(rc * ip, 2),
            score_note='project score with CARLA LB 2.0 penalty coefficients; not an official Leaderboard result',
            state_ticks=dict(Counter(r['state'] for r in rows)),
            pedestrian_hits_by_ego=sum(1 for c in coll if c['category'] == 'pedestrian' and c['ego_caused']),
            red_lights_run=sum(1 for e in self.events if e['type'] == 'red_light_run'),
            **inter, **comfort, **extra)
        ms = extra.get('manoeuvre_stats') or {}
        mvm = self.meta.get('manoeuvres') or {}
        off = extra.get('final_offset_m')
        back_in_lane = off is None or off == '' or abs(float(off)) < 0.3
        lights_mode = self.meta.get('lights') or 'off'

        traffic = int(self.meta.get('traffic') or 0) + int((self.meta.get('jam') or {}).get('vehicles') or 0)
        lead_car = self.meta.get('lead_car')
        if traffic > 0 or lead_car:
            junc = np.array([bool(r.get('junction_conflict')) for r in rows])   # crossing car at a junction
            # emergency stops for a pedestrian stepping out are the right reaction, not a
            # car-following failure: counted separately (a pedestrian HIT always fails, see checks)
            # (an episode belongs to a pedestrian if one was the obstacle when it began)
            emerg_all = np.array([r['state'] == 'EMERGENCY_STOP' for r in rows])
            ped = np.zeros(len(rows), bool)
            for i in range(len(rows)):
                if emerg_all[i]:
                    ped[i] = (ped[i - 1] if i > 0 and emerg_all[i - 1] else
                              str(rows[i].get('lead_cls', '')).lower().startswith('pedestrian'))
            emerg = emerg_all & ~ped
            ttc = np.array([float(r['gt_ttc']) if r.get('gt_ttc', '') != '' else np.inf for r in rows])
            ego_coll = [c for c in coll if c['ego_caused']]
            summary.update(
                traffic=traffic,
                ego_caused_collisions_outside_junctions=sum(1 for c in ego_coll if not c['junction_conflict']),
                ego_caused_collisions_in_junctions=sum(1 for c in ego_coll if c['junction_conflict']),
                not_ego_caused_collisions=len(coll) - len(ego_coll),
                emergency_stops_outside_junctions=_split_episodes(emerg, junc)[0],
                emergency_stops_in_junctions=_split_episodes(emerg, junc)[1],
                emergency_stops_for_pedestrians=_episodes(emerg_all & ped),
                min_gt_ttc_outside_junctions_s=_finite_min(ttc[~junc]),
                min_gt_ttc_in_junctions_s=_finite_min(ttc[junc]),
                ttc_below_1_5s_events_outside_junctions=_split_episodes(ttc < 1.5, junc)[0])
            gap = np.array([float(r['gt_lead_gap']) if r.get('gt_lead_gap', '') != '' else np.inf
                            for r in rows])
            follow = gap <= FOLLOW_RANGE_M
            moving_f = follow & (speed > 2.0)
            summary.update(
                lead_car=lead_car,
                lead_following_s=round(float(follow.sum() * self.dt), 1),
                min_gt_lead_gap_m=_finite_min(gap[follow & ~junc]),
                median_time_headway_s=(round(float(np.median(gap[moving_f] / speed[moving_f])), 2)
                                       if moving_f.any() else None),
                stops_behind_a_car=_episodes((speed < 0.3) & (gap <= 12.0)))

        checks = {}
        if scenario == 'route' and (traffic > 0 or lead_car):
            summary['milestone'] = (f'Step 3 traffic following ({traffic} cars'
                                    + (', packed' if self.meta.get('jam') else '')
                                    + (', lead car)' if lead_car else ', no lead car)'))
            m = summary['min_gt_ttc_outside_junctions_s']
            g = summary['min_gt_lead_gap_m']
            checks = {'route completed (a run stuck behind a jam or interrupted is not a pass)': completed,
                      f'followed a car for >= {MIN_FOLLOW_S:.0f} s (else following was not tested)':
                          summary['lead_following_s'] >= MIN_FOLLOW_S,
                      f'min gap to the car ahead outside junctions >= {MIN_STOP_GAP_M} m': g is None or g >= MIN_STOP_GAP_M,
                      'no ego-caused collision outside junctions (heuristic fault rule)':
                          summary['ego_caused_collisions_outside_junctions'] == 0,
                      'no emergency stop outside junctions': summary['emergency_stops_outside_junctions'] == 0,
                      'min TTC outside junctions >= 1.5 s': m is None or m >= 1.5,
                      'no pedestrian hit by the ego (anywhere, junctions included)':
                          summary['pedestrian_hits_by_ego'] == 0}
            if lights_mode != 'off':
                checks['no red light run (--lights ' + lights_mode + ')'] = summary['red_lights_run'] == 0
            if mvm.get('yield_at_junctions'):
                checks['no ego-caused collision in junctions (--yield)'] = \
                    summary['ego_caused_collisions_in_junctions'] == 0
            summary['note'] = ('junction conflicts (a car crossing the ego path at a junction; for '
                               'collisions also: the ego inside a junction) are reported but not gated: '
                               + ('the ego stops at red lights but does not yield to oncoming or crossing '
                                  'cars when turning' if lights_mode != 'off' else
                                  'the ego ignores traffic lights (run with --lights oracle)')
                               + '. Following a car through a junction IS gated; hitting a pedestrian '
                               'fails everywhere')
        elif scenario == 'route':
            checks = {'route completed': completed,
                      f'max lateral deviation <= {MAX_DEV_M} m': summary['max_lateral_dev_m'] <= MAX_DEV_M,
                      f'steering reversals <= {MAX_REVERSALS_PER_MIN:.0f}/min (heuristic)':
                          rev_min <= MAX_REVERSALS_PER_MIN,
                      'no collisions': not coll}
            if lights_mode != 'off':
                checks['no red light run (--lights ' + lights_mode + ')'] = summary['red_lights_run'] == 0
            summary['milestone'] = 'M2 route following'
        elif scenario == 'overtake':
            summary['milestone'] = 'M7 lane change past a slow car'
            clr, m = summary.get('min_vehicle_clearance_m'), summary.get('min_gt_ttc_outside_junctions_s')
            checks = {'route completed': completed,
                      'changed lane (else overtaking was not tested)': ms.get('lane_changes', 0) >= 1,
                      'passed the slow car': bool(extra.get('lead_passed')),
                      'back in the route lane at the end': back_in_lane,
                      f'min clearance to other cars >= {MIN_CAR_CLEAR_M} m (GT footprints)':
                          clr is None or clr >= MIN_CAR_CLEAR_M,
                      'no collisions': not coll,
                      'no emergency stop outside junctions': summary.get('emergency_stops_outside_junctions', 0) == 0,
                      'min TTC outside junctions >= 1.5 s': m is None or m >= 1.5}
        elif scenario == 'junction':
            st = self.meta.get('scenario_traffic') or {}
            summary['milestone'] = (f"M8 junction, {st.get('turn') or '?'} turn"
                                    + ('' if mvm.get('yield_at_junctions') else ' (yield OFF: baseline)'))
            pet = summary.get('pet_min_s')
            summary['emergency_stops'] = _episodes([r['state'] == 'EMERGENCY_STOP' for r in rows])
            checks = {'route completed (through the junction)': completed,
                      'no emergency stop (it yielded in time instead)': summary['emergency_stops'] == 0,
                      f'traffic crossed the ego path (a car within {CONFLICT_PET_S:.0f} s of it, or it yielded; '
                      'else yielding was not tested)': (summary.get('conflicts_within_5s', 0) >= 1
                                                        or ms.get('yields', 0) >= 1),
                      f'min post-encroachment time >= {MIN_PET_S} s (GT)': pet is None or pet >= MIN_PET_S,
                      'no collisions': not coll}
        elif scenario == 'static':
            summary['milestone'] = 'M4 static-obstacle stop'
            hz = extra.get('hazard')
            run_over = bool(extra.get('hazard_run_over'))
            if hz:
                summary['milestone'] = 'Hazard: ' + hz.get('label', f"person {hz.get('pose')} on the {hz.get('place')}")
            steer = mvm.get('steer_around') and (not hz or hz.get('expected') == 'steer around')
            if steer and not hz:
                summary['milestone'] = 'M6 steer around a stopped obstacle'
            if extra.get('obstacle_source') == 'none':
                checks = {'collided (expected with perception off: proves causality)': bool(coll) or run_over}
            elif steer:
                clr, pet = extra.get('min_obstacle_clearance_m'), summary.get('pet_min_s')
                checks = {'passed it (route completed)': completed,
                          'steered around it (not only waited)': ms.get('steer_arounds', 0) >= 1,
                          f'min clearance to it >= {MIN_PASS_CLEAR_M} m (GT footprints)':
                              clr is not None and clr >= MIN_PASS_CLEAR_M,
                          'back in the route lane at the end': back_in_lane,
                          'no collisions': not coll}
                if hz:
                    checks['did not drive onto it'] = not run_over
                if (self.meta.get('scenario_traffic') or {}).get('oncoming'):
                    checks[f'min post-encroachment time with the oncoming cars >= {MIN_PET_S} s (GT)'] = \
                        pet is None or pet >= MIN_PET_S
            elif hz and hz.get('expected') == 'drive past':
                checks = {'drove past (route completed, no stop for it)': completed,
                          'did not drive onto the person': not run_over,
                          'no collisions': not coll}
            elif hz and hz.get('expected') == 'yield':            # crossing in front of the car
                checks = {'let it cross (no contact) and went on, or waited': completed or bool(extra.get('stopped')),
                          'did not drive onto it': not run_over,
                          'no collisions': not coll}
            elif hz and hz.get('expected') == 'stop or pass':     # beside the path, inside the safety margin
                checks = {'stopped behind it or drove past': bool(extra.get('stopped')) or completed,
                          'did not drive onto the person': not run_over,
                          'no collisions': not coll}
            else:
                gap = extra.get('final_gt_gap_m')
                pp = self.meta.get('params') or {}
                max_gap = float(pp.get('s0', 4.0)) + float(pp.get('static_buffer', 2.0)) + MAX_STOP_EXTRA_M
                summary['max_stop_gap_m'] = round(max_gap, 1)
                checks = {'stopped behind the obstacle': bool(extra.get('stopped')),
                          f'final gap >= {MIN_STOP_GAP_M} m':
                              gap is not None and gap >= MIN_STOP_GAP_M,
                          f'final gap <= {max_gap:.0f} m (stopped for the real obstacle, not short of it)':
                              gap is not None and gap <= max_gap,
                          'no collisions': not coll}
                if hz:
                    checks['did not drive onto the person'] = not run_over
        summary['checks'] = checks
        summary['passed'] = bool(checks) and all(checks.values())
        self._write(summary)
        return summary

    # ---- files -------------------------------------------------------------
    def _write(self, summary):
        if self.rows:
            with open(os.path.join(self.dir, 'ticks.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                w.writeheader()
                w.writerows(self.rows)
        with open(os.path.join(self.dir, 'events.jsonl'), 'w') as f:
            for e in self.events:
                f.write(json.dumps(e, default=_jsonable) + '\n')
        with open(os.path.join(self.dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=_jsonable)
        self._plot(os.path.join(self.dir, 'trajectory.png'), summary)

    def _plot(self, path, summary):
        if not self.rows:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('  (matplotlib not installed: trajectory.png skipped)')
            return
        x = np.array([r['x'] for r in self.rows])
        y = np.array([r['y'] for r in self.rows])
        t = np.array([r['t'] for r in self.rows])
        v = np.array([r['speed'] for r in self.rows]) * 3.6
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.3]})

        # top-down path, square limits so a straight run is still readable
        ax.plot(self.route_xy[:, 0], self.route_xy[:, 1], '--', color='0.55', lw=1, label='route')
        sc = ax.scatter(x, y, c=v, s=5, cmap='viridis', label='driven')
        fig.colorbar(sc, ax=ax, label='speed (km/h)', shrink=0.7)
        for poly in self.polygons:
            p = np.vstack([poly, poly[:1]])
            ax.fill(p[:, 0], p[:, 1], color='tab:red', alpha=0.5, label='obstacle (GT)')
        frames = {r['frame']: (r['x'], r['y']) for r in self.rows}
        for c in summary.get('collisions', []):
            if c['frame'] in frames:
                ax.plot(*frames[c['frame']], 'rx', ms=12, mew=3, label='collision')
        ax.plot(x[0], y[0], 'go', ms=8, label='start')
        pts = np.vstack([self.route_xy, np.c_[x, y]])
        mid, half = (pts.max(0) + pts.min(0)) / 2, max(np.ptp(pts, axis=0).max() / 2 + 5, 15)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] + half, mid[1] - half)      # CARLA top view: +y points down the screen
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        h, l = ax.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        ax.legend(uniq.values(), uniq.keys(), loc='best', fontsize=8)

        # time panel: speed, target, and the true gap; events as vertical lines
        ax2.plot(t, v, label='speed (km/h)')
        ax2.plot(t, [r['v_target_kmh'] for r in self.rows], '--', lw=1, label='target (km/h)')
        gt = [(r['t'], r['gt_gap']) for r in self.rows if r.get('gt_gap', '') != '']
        if gt:
            ax2.plot(*zip(*gt), color='tab:red', label='GT gap (m)')
            ax2.axhline(MIN_STOP_GAP_M, color='tab:red', ls=':', lw=1)
        marks = {'obstacle_in_corridor': ('in corridor', 'tab:orange'),
                 'obstacle_confirmed': ('confirmed', 'tab:purple'),
                 'first_brake': ('first brake', 'black'),
                 'steer_around_start': ('steer around', 'tab:green'), 'lane_change_start': ('lane change', 'tab:green'),
                 'lane_change_back': ('change back', 'tab:olive'), 'yield_start': ('yield', 'tab:brown'),
                 'yield_end': ('go', 'tab:cyan'), 'manoeuvre_abort': ('abort', 'tab:red')}
        for e in self.events:
            if e['type'] in marks and e['k'] is not None:
                name, col = marks[e['type']]
                ax2.axvline(e['k'] * self.dt, color=col, ls='--', lw=1, label=name)
        ax2.set_xlabel('time (s)')
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8, loc='best')

        verdict = 'PASS' if summary.get('passed') else 'FAIL'
        if summary.get('min_obstacle_clearance_m') is not None and (self.meta.get('manoeuvres') or {}).get(
                'steer_around'):
            detail = f"passed it at {summary['min_obstacle_clearance_m']} m min clearance"
        elif summary.get('pet_min_s') is not None:
            detail = f"min PET {summary['pet_min_s']} s  RC {summary['route_completion_pct']}%"
        elif summary['scenario'] == 'static':
            g = summary.get('final_gt_gap_m')
            detail = f"final GT gap {g} m" if g is not None else ''
        else:
            detail = f"RC {summary['route_completion_pct']}%  score {summary['project_score']}"
        fig.suptitle(f"{summary.get('milestone', summary['scenario'])}: {verdict}   {detail}")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)


def interactions(rows, others, box, dt, cell=2.0):
    """min clearance to other cars (same tick) and post-encroachment time (PET) with crossing / oncoming
    cars, from ground-truth footprints (see the top of this file). {} without logged cars."""
    if not others or not rows or not box:
        return {}
    from collections import defaultdict
    from manoeuvre import rect, overlap, poly_distance
    ego, yaw = {}, {}
    grid = defaultdict(list)
    for r in rows:
        k, h = r['k'], math.radians(r['yaw'])
        P = rect(r['x'], r['y'], h, box['front'], box['rear'], box['half_width'])
        ego[k], yaw[k] = P, h
        (x0, y0), (x1, y1) = np.floor(P.min(0) / cell).astype(int), np.floor(P.max(0) / cell).astype(int)
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                grid[(cx, cy)].append(k)
    tracks = defaultdict(list)
    for o in others:
        tracks[o[1]].append(o)
    min_clear, pets = math.inf, {}
    for oid, tr in tracks.items():
        best = math.inf
        for k, _, x, y, h, length, width in tr:
            Q = rect(x, y, h, 0.5 * length, 0.5 * length, 0.5 * width)
            if k in ego:
                min_clear = min(min_clear, poly_distance(ego[k], Q))
            (x0, y0), (x1, y1) = np.floor(Q.min(0) / cell).astype(int), np.floor(Q.max(0) / cell).astype(int)
            cand = set()
            for cx in range(x0, x1 + 1):
                for cy in range(y0, y1 + 1):
                    cand.update(grid.get((cx, cy), ()))
            for ke in sorted(cand, key=lambda q: abs(q - k)):
                gap = abs(ke - k) * dt
                if gap >= best:
                    break
                if math.cos(h - yaw[ke]) > math.cos(math.radians(30.0)):
                    continue                     # the same way as the ego there: following, not a conflict
                if overlap(ego[ke], Q):
                    best = gap
                    break
        if math.isfinite(best):
            pets[oid] = round(best, 2)
    out = dict(min_vehicle_clearance_m=round(min_clear, 3) if math.isfinite(min_clear) else None,
               pet_min_s=min(pets.values()) if pets else None,
               conflicts_within_5s=sum(1 for v in pets.values() if v <= CONFLICT_PET_S),
               pet_by_car={str(k): v for k, v in sorted(pets.items(), key=lambda kv: kv[1])[:10]})
    return out


def _episodes(mask):
    """Number of separate runs of True (5 consecutive ticks = one event)."""
    m = np.asarray(mask, bool)
    return int(np.sum(np.diff(np.r_[0, m.astype(int)]) == 1))


def _split_episodes(mask, flag):
    """Runs of True in mask, split into (not flagged, flagged): a run counts as
    flagged if ANY of its ticks is (an emergency stop that began for a crossing car
    stays one junction-conflict stop after that car has left the path)."""
    m = np.r_[False, np.asarray(mask, bool), False]
    f = np.asarray(flag, bool)
    starts = np.flatnonzero(~m[:-1] & m[1:])
    ends = np.flatnonzero(m[:-1] & ~m[1:])
    flagged = sum(1 for a, b in zip(starts, ends) if f[a:b].any())
    return len(starts) - flagged, flagged


def _finite_min(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return round(float(a.min()), 2) if a.size else None


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isinf(o):
        return None
    return str(o)
