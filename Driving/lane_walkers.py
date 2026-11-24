"""
lane_walkers.py -- pedestrians for maps without CARLA pedestrian navigation (maps built from OpenDRIVE /
OpenStreetMap with --xodr / --osm, when CARLA could not build a navigation mesh for them).

CARLA's walker AI (controller.ai.walker) only works on a navigation mesh. These walkers have no AI: every
0.1 s we steer each one with carla.WalkerControl (direction + speed), so they only need ground under them.
No AI controller also means CARLA issue #7092 (the LiDAR missing AI pedestrians) does not arise.

Where they walk (worked out once, every 4 m along the ego route, from the map's lanes and
world.ground_projection):
  footpath    1.2 m (else 0.6 m) beyond the outermost driving lane, where there is ground there
              (a footpath or shoulder; agent.py adds footpaths to --osm / --xodr maps, --sidewalk-width)
  road edge   0.25 m inside the outermost lane, where there is no ground beyond it (roads without
              footpaths), but only where that is clear of the ego's path (|d| >= clear_d): in practice
              the far side of a two-way road
  junctions   nowhere (walkers turn round before a junction)
What they do:
  walk        along their line, either way, turning round at junctions and at the route's ends
  cross       a share `cross` of them cross the road when the ego is 3.5-6 s away, but never closer
              than 12 m + 1.5 s x its speed (on Indian roads people cross anywhere), to the line on the
              far side, then walk on there. Only where the far side has a line: a walker left at the
              edge of the ego's own lane would block it or be brushed by it
  wait        nobody steps into the ego's lane while the moving ego is under 3 s away or still beside
              them (until its rear is 1.5 m past), and nobody walks into the car's body, moving or stopped
  come back   a walker 30 m behind the ego starts again 70-160 m ahead of it, on a random line,
              so a long route keeps meeting pedestrians. Walkers waiting to come back stand far off
              the map with physics off (no actor is destroyed during the run).
Coordinates: d is the lateral offset from the route line, + to the right (CARLA: right of heading h is
(-sin h, cos h)).
"""
import math
import random
import re

import numpy as np


class LaneWalkers:
    def __init__(self, world, cmap, carla, bp, route, n, cross, seed, keep_clear, actors, clear_d=1.9,
                 speed=(1.0, 1.6), first=(10.0, 160.0), again=(70.0, 160.0), behind=30.0, probe_m=4.0, every=2,
                 ego_half_len=2.4, ego_half_w=1.05):
        self.world, self.cmap, self.C = world, cmap, carla
        self.rng = random.Random(seed)
        self.xy = np.asarray(route.xy, float)
        self.s = np.asarray(route.s, float)
        d = np.diff(self.xy, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        self.h = np.r_[h, h[-1]]
        self.right = np.c_[-np.sin(self.h), np.cos(self.h)]
        wps = route.wps
        self.z = (np.array([w.transform.location.z for w in wps], float) if wps is not None
                  else np.zeros(len(self.xy)))
        self.cross, self.clear_d, self.speed = float(cross), float(clear_d), speed
        self.again, self.behind, self.every = again, behind, max(1, int(every))
        self.hl, self.hw = float(ego_half_len), float(ego_half_w)
        self.tan = np.c_[np.cos(self.h), np.sin(self.h)]
        self.keep_clear = list(keep_clear)
        self.lines, self.share = self._walk_lines(wps, probe_m)
        self.park = (float(self.xy[:, 0].max()) + 2000.0, float(self.xy[:, 1].max()) + 2000.0, -100.0)
        self.t, self.n_calls, self.crossings, self.placed = 0.0, 0, 0, 0
        self.events = []                                   # crossings started, for the run's events.jsonl
        self.walkers = []
        wbps = list(bp.filter('walker.pedestrian.*'))
        for k in range(n):
            b = self.rng.choice(wbps)
            if b.has_attribute('is_invincible'):
                b.set_attribute('is_invincible', 'false')     # so that a hit registers as a collision
            park = (self.park[0] + 3.0 * k, self.park[1], self.park[2])       # each its own spot
            a = world.try_spawn_actor(b, carla.Transform(carla.Location(*park)))
            if a is None:
                continue
            actors.append(a)
            a.set_simulate_physics(False)
            w = dict(actor=a, mode='off', side=0, dir=1, i=0, v=self.rng.uniform(*speed), crosser=False,
                     trig=5.0, t_next=0.0, target=None, park=park)
            self.walkers.append(w)
            self._place_ahead(w, 0.0, first)

    # ------------------------------------------------------------ where to walk
    def _lat(self, i, loc):
        return (loc.x - self.xy[i, 0]) * self.right[i, 0] + (loc.y - self.xy[i, 1]) * self.right[i, 1]

    def _pt(self, i, d):
        return self.xy[i] + self.right[i] * d

    def _edges(self, i):
        """lateral offsets of the road's outer edges at route point i: [right (+), left (-)], or None.
        Walks sideways lane by lane with plain geometry (get_left_lane of an oncoming lane points back
        at us, and OSM maps make each direction its own road)."""
        C = self.C
        x, y = self.xy[i]
        z = float(self.z[i])
        try:
            w0 = self.cmap.get_waypoint(C.Location(x, y, z), project_to_road=True, lane_type=C.LaneType.Driving)
        except RuntimeError:
            return None
        if w0 is None:
            return None
        out = []
        for sg in (1.0, -1.0):
            d, wd = self._lat(i, w0.transform.location), float(w0.lane_width)
            for _ in range(8):
                pd = d + sg * wd
                px, py = self._pt(i, pd)
                try:
                    q = self.cmap.get_waypoint(C.Location(float(px), float(py), z), project_to_road=False,
                                               lane_type=C.LaneType.Driving)
                except RuntimeError:
                    q = None
                if q is None:
                    break
                qd = self._lat(i, q.transform.location)
                if abs(qd - d) < 0.5 * wd:                    # the same lane again
                    break
                d, wd = qd, float(q.lane_width)
            out.append(d + sg * 0.5 * wd)
        return out

    def _ground_at(self, i, d):
        x, y = self._pt(i, d)
        z = float(self.z[i])
        try:
            p = self.world.ground_projection(self.C.Location(float(x), float(y), z + 2.0), 4.0)
        except (RuntimeError, AttributeError):
            return False
        return p is not None and abs(p.location.z - z) < 0.6

    def _walk_lines(self, wps, probe_m):
        N = len(self.xy)
        probes = np.unique(np.clip(np.searchsorted(self.s, np.arange(0.0, self.s[-1] + 1e-6, probe_m)), 0, N - 1))
        vals = np.full((2, len(probes)), np.nan)
        kind = np.zeros((2, len(probes)), int)              # 0 none, 1 footpath, 2 road edge
        for k, i in enumerate(probes):
            if wps is not None and wps[i].is_junction:
                continue
            e = self._edges(i)
            if e is None:
                continue
            for j, sg in enumerate((1.0, -1.0)):            # 0 = right side, 1 = left side
                for off, kd in ((1.2, 1), (0.6, 1), (-0.25, 2)):
                    dd = e[j] + sg * off
                    if abs(dd) < self.clear_d or (kd == 1 and not self._ground_at(i, dd)):
                        continue
                    vals[j, k], kind[j, k] = dd, kd
                    break
        ps = self.s[probes]
        if len(ps) == 1:
            pick = np.zeros(N, int)
        else:
            hi = np.clip(np.searchsorted(ps, self.s), 1, len(ps) - 1)
            lo = hi - 1
            pick = np.where(np.abs(self.s - ps[lo]) <= np.abs(ps[hi] - self.s), lo, hi)
        m = max(len(probes), 1)
        share = {f"{side} {name}": float(np.sum(kind[j] == kd)) / m
                 for j, side in enumerate(('right', 'left')) for kd, name in ((1, 'footpath'), (2, 'road edge'))}
        return vals[:, pick], share

    def describe(self):
        sh = self.share
        return (f"right side: footpath {100 * sh['right footpath']:.0f}%, road edge {100 * sh['right road edge']:.0f}%; "
                f"left side: footpath {100 * sh['left footpath']:.0f}%, road edge {100 * sh['left road edge']:.0f}% "
                f"of the route")

    def has_room(self):
        return bool(np.any(~np.isnan(self.lines)))

    # ------------------------------------------------------------ actions
    def _set(self, w, x, y, z, yaw):
        C, a = self.C, w['actor']
        a.set_transform(C.Transform(C.Location(float(x), float(y), float(z)), C.Rotation(yaw=float(yaw))))
        a.set_simulate_physics(True)

    def _leave(self, w, pause):
        C, a = self.C, w['actor']
        a.apply_control(C.WalkerControl(C.Vector3D(1.0, 0.0, 0.0), 0.0))
        a.set_simulate_physics(False)
        a.set_transform(C.Transform(C.Location(*w['park'])))
        w.update(mode='off', t_next=self.t + self.rng.uniform(*pause), target=None)

    def _place_ahead(self, w, ego_s, window):
        N = len(self.s)
        for _ in range(12):
            s_t = ego_s + self.rng.uniform(*window)
            if s_t > self.s[-1] - 2.0:
                continue
            i = min(int(np.searchsorted(self.s, s_t)), N - 1)
            sides = [j for j in (0, 1) if not np.isnan(self.lines[j, i])]
            if not sides:
                continue
            j = self.rng.choice(sides)
            x, y = self._pt(i, self.lines[j, i])
            if any(math.hypot(x - c.x, y - c.y) < 8.0 for c in self.keep_clear):
                continue
            if any(o is not w and o['mode'] != 'off' and o.get('xy') is not None
                   and math.hypot(x - o['xy'][0], y - o['xy'][1]) < 2.5 for o in self.walkers):
                continue
            dr = self.rng.choice((1, -1))
            self._set(w, x, y, self.z[i] + 1.2, math.degrees(self.h[i]) + (0.0 if dr > 0 else 180.0))
            w.update(mode='walk', side=j, dir=dr, i=i, crosser=self.rng.random() < self.cross,
                     trig=self.rng.uniform(3.5, 6.0), xy=(x, y), target=None)
            self.placed += 1
            return True
        w['t_next'] = self.t + 1.0
        return False

    def _index(self, w, x, y):
        N = len(self.s)
        lo, hi = max(0, w['i'] - 60), min(N, w['i'] + 61)
        seg = self.xy[lo:hi]
        return lo + int(np.argmin((seg[:, 0] - x) ** 2 + (seg[:, 1] - y) ** 2))

    def _sd(self, i, x, y):
        """(s, d) of a point near route point i"""
        dx, dy = x - self.xy[i, 0], y - self.xy[i, 1]
        return (self.s[i] + dx * self.tan[i, 0] + dy * self.tan[i, 1],
                dx * self.right[i, 0] + dy * self.right[i, 1])

    def tick(self, dt, ego_s, ego_v, ego_xy=None):
        """call once per simulation tick, before world.tick(); ego_s = progress of the ego's centre along the
        route (m), ego_xy = its position (for its lateral offset when it steers around something)"""
        self.t += dt
        self.n_calls += 1
        if self.n_calls % self.every:
            return
        C = self.C
        ie = min(int(np.searchsorted(self.s, ego_s)), len(self.s) - 1)
        ego_d = self._sd(ie, *ego_xy)[1] if ego_xy is not None else 0.0
        # the car's body (+ room), and where it will be within 3 s
        body = lambda s_, d_: abs(s_ - ego_s) < self.hl + 0.5 and abs(d_ - ego_d) < self.hw + 0.6
        zone = lambda s_, d_: (-(self.hl + 1.5) < s_ - ego_s < self.hl + 3.0 * ego_v
                               and abs(d_ - ego_d) < self.clear_d)
        for w in self.walkers:
            a = w['actor']
            if not a.is_alive:
                continue
            if w['mode'] == 'off':
                if self.t >= w['t_next']:
                    self._place_ahead(w, ego_s, self.again)
                continue
            L = a.get_location()
            i = w['i'] = self._index(w, L.x, L.y)
            w['xy'] = (L.x, L.y)
            if L.z < self.z[i] - 5.0 or self.s[i] < ego_s - self.behind:     # fell off, or left behind
                self._leave(w, (0.5, 3.0))
                continue
            ahead = self.s[i] - ego_s - self.hl            # from the ego's front bumper
            d_now = self._lat(i, L)
            if w['mode'] == 'walk' and w['crosser']:
                far = 1 - w['side']
                d_far = self.lines[far, i]
                room = 12.0 + 1.5 * ego_v                  # never right in front of the car: it can stop gently
                t_in = max(0.0, abs(d_now) - self.clear_d) / w['v']     # walk to the ego's lane first
                if room < ahead < max(ego_v, 2.5) * (w['trig'] + t_in) + 3.0 and not np.isnan(d_far):
                    w.update(mode='cross', crosser=False, target=(i, float(d_far)), side=far)
                    self.crossings += 1
                    self.events.append(dict(walker=a.id, ahead_m=round(float(ahead), 1),
                                            ego_kmh=round(3.6 * ego_v, 1), to=('right', 'left')[far]))
            if w['mode'] == 'cross':
                ti, td = w['target']
                tx, ty = self._pt(ti, td)
                if math.hypot(tx - L.x, ty - L.y) < 0.5:
                    w.update(mode='walk', dir=self.rng.choice((1, -1)))
            if w['mode'] == 'walk':
                line = self.lines[w['side']]
                tx = ty = None
                for _ in range(2):
                    j = int(np.clip(np.searchsorted(self.s, self.s[i] + w['dir'] * 2.0), 0, len(self.s) - 1))
                    if j != i and not np.isnan(line[j]):
                        tx, ty = self._pt(j, line[j])
                        break
                    w['dir'] *= -1                         # junction or route end ahead: turn round
                if tx is None:
                    self._leave(w, (0.5, 3.0))
                    continue
            dx, dy = tx - L.x, ty - L.y
            nn = math.hypot(dx, dy)
            if nn > 1e-6:
                ux, uy = dx / nn, dy / nn
                s_c, d_c = self._sd(i, L.x, L.y)
                s_n, d_n = self._sd(i, L.x + 0.8 * ux, L.y + 0.8 * uy)
                if body(s_n, d_n) or (ego_v > 0.5 and zone(s_n, d_n) and not zone(s_c, d_c)):
                    a.apply_control(C.WalkerControl(C.Vector3D(ux, uy, 0.0), 0.0))   # wait for the car
                    continue
                a.apply_control(C.WalkerControl(C.Vector3D(ux, uy, 0.0), w['v']))

    def pop_events(self):
        out, self.events = self.events, []
        return out

    def counts(self):
        m = [w['mode'] for w in self.walkers if w['actor'].is_alive]
        return {k: m.count(k) for k in ('walk', 'cross', 'off')}


def add_sidewalks(xodr, width):
    """a footpath (OpenDRIVE lane of type sidewalk, `width` m) beside the outermost lane of each road side
    whose outermost lane is a driving lane, junction roads excepted. OpenStreetMap conversions have no
    footpaths, so CARLA builds no pedestrian surface or navigation on them. Text edit only: everything
    else in the file stays as it is, and a file that already has footpaths is left alone.
    Returns (new text, number of footpaths added)."""
    lane_re = re.compile(r'<lane\b[^>]*>')
    attr = lambda tag, k: (re.search(r'\b' + k + r'="([^"]*)"', tag) or [None, None])[1]
    added = [0]

    def side(m):
        grp, body = m.group(1), m.group(2)
        tags = lane_re.findall(body)
        ids = [(int(attr(t, 'id')), attr(t, 'type')) for t in tags if attr(t, 'id') not in (None, '')]
        if not ids:
            return m.group(0)
        out_id, out_type = (min(ids) if grp == 'right' else max(ids))
        if (out_type or '').lower() != 'driving':
            return m.group(0)
        new = out_id - 1 if grp == 'right' else out_id + 1
        added[0] += 1
        lane = (f'<lane id="{new}" type="sidewalk" level="false"><link/>'
                f'<width sOffset="0" a="{width:g}" b="0" c="0" d="0"/></lane>')
        return f'<{grp}>{body}{lane}</{grp}>'

    def road(m):
        txt = m.group(0)
        head = txt[:txt.index('>') + 1]
        if (attr(head, 'junction') or '-1') != '-1':
            return txt
        return re.sub(r'<(right|left)>(.*?)</\1>', side, txt, flags=re.S)

    out = re.sub(r'<road\b.*?</road>', road, xodr, flags=re.S)
    return out, added[0]