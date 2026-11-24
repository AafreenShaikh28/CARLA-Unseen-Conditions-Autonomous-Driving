"""
occupancy.py - class-free safety layer for agent.py (--occupancy): LiDAR points that stand up from the
road inside the car's own lane ahead become an 'unknown' obstacle, whatever they are. The detectors only
know the shapes they were trained on (an upright pedestrian, a car ...); a person lying on the road,
debris or an animal can be missed by them but still returns LiDAR points above the road surface.


How it works, per tick, on the same LiDAR points the detector gets:
  1. points ahead, inside the car's width (+ half_width_extra) along the route, up to range_m
  2. height above the road: the route's waypoint height, corrected by the road points themselves
     (30th percentile of what lies within 30 cm of the waypoint height), so slopes and small offsets cancel
  3. 0.5 m slices along the route; a slice with >= min_pts points between min_h and max_h is occupied
  4. the first group of occupied slices becomes one obstacle (its footprint, height, class 'unknown')
  5. if the detector already has a box there (within near_det_m), nothing is added: the detector's box
     has the velocity, so a car driving ahead is still followed, not treated as a wall


The decision logic then treats it like any other obstacle in the corridor (3-of-5 confirmation, stop).
"""
import math


import numpy as np


from world_model import Obstacle, footprint


DEFAULTS = dict(min_h=0.12, max_h=2.2, bin_m=0.5, min_pts=8, range_m=40.0, half_width_extra=0.2, near_det_m=1.0)
TRACK_ID = -9                     # one stable id for the occupancy obstacle




class OccupancyCheck:
    def __init__(self, route, ego_half_width, ego_front, **kw):
        self.p = dict(DEFAULTS, **{k: v for k, v in kw.items() if v is not None})
        self.route = route
        self.hw = ego_half_width + self.p['half_width_extra']
        self.front = ego_front
        self.z = (np.array([w.transform.location.z for w in route.wps], float)
                  if getattr(route, 'wps', None) else None)
        self.last = None                                     # diagnostics of the last call


    def _on_route(self, s_abs):
        r = self.route
        j = int(np.clip(np.searchsorted(r.s, s_abs) - 1, 0, len(r.seg_len) - 1))
        t = float(np.clip((s_abs - r.s[j]) / max(r.seg_len[j], 1e-9), 0.0, 1.0))
        xy = r.xy[j] + t * (r.xy[j + 1] - r.xy[j])
        return xy, float(r.heading[j])


    def __call__(self, lidar, lidar_matrix, k, detected=(), ground_z=None):
        """lidar: [N, 4] x, y-LEFT, z, i in the sensor frame (as perception_scheduler decodes it);
        lidar_matrix: 4x4 sensor -> world. Returns an Obstacle or None."""
        self.last = None
        if lidar is None or len(lidar) == 0:
            return None
        p = self.p
        x, y = lidar[:, 0], lidar[:, 1]
        m = (x > 0.5) & (x < p['range_m'] + 5.0) & (np.abs(y) < self.hw + 4.0)
        if not m.any():
            return None
        P = np.c_[lidar[m, 0], -lidar[m, 1], lidar[m, 2], np.ones(int(m.sum()))]   # CARLA y-right
        W = (P @ np.asarray(lidar_matrix, float).T)[:, :3]
        s_rel, d, s_abs, beyond = self.route.project(W[:, :2], p['range_m'], return_beyond=True)
        if self.z is not None:
            zr = np.interp(s_abs, self.route.s, self.z)
        elif ground_z is not None:
            zr = np.full(len(W), float(ground_z))
        else:
            return None
        r = W[:, 2] - zr
        band = (~beyond) & (np.abs(d) <= self.hw + 1.5) & (s_rel > self.front)
        road = band & (np.abs(r) < 0.3)
        off = float(np.percentile(r[road], 30)) if road.sum() >= 30 else 0.0
        h = r - off
        inside = ((~beyond) & (np.abs(d) <= self.hw) & (s_rel > self.front + 0.3) & (s_rel < p['range_m'])
                  & (h > p['min_h']) & (h < p['max_h']))
        n_in = int(inside.sum())
        self.last = dict(points_above_road=n_in, road_offset_m=round(off, 3))
        if n_in < p['min_pts']:
            return None
        sb = np.floor((s_rel[inside] - self.front) / p['bin_m']).astype(int)
        counts = np.bincount(sb)
        occ = np.nonzero(counts >= p['min_pts'])[0]
        if not len(occ):
            return None
        b0 = b1 = int(occ[0])
        for b in occ[1:]:
            if b == b1 + 1:
                b1 = int(b)
            else:
                break
        idx = np.nonzero(inside)[0][(sb >= b0) & (sb <= b1)]
        s0, s1 = float(s_rel[idx].min()), float(s_rel[idx].max())
        d0, d1 = float(d[idx].min()), float(d[idx].max())
        hmax = float(h[idx].max())
        (cx, cy), head = self._on_route(self.route.s_ego + (s0 + s1) / 2)
        dm = (d0 + d1) / 2
        cx, cy = cx - math.sin(head) * dm, cy + math.cos(head) * dm            # d > 0 = left of the route
        for o in detected or ():
            if math.hypot(o.center[0] - cx, o.center[1] - cy) <= 0.5 * max(o.size[0], o.size[1]) + p['near_det_m']:
                self.last.update(suppressed_by=o.cls)
                return None
        length, width = s1 - s0 + 0.2, d1 - d0 + 0.2
        zc = float(np.median(zr[idx])) + off + hmax / 2
        self.last.update(points=len(idx), gap_m=round(s0 - self.front, 2), height_m=round(hmax, 2))
        return Obstacle('unknown', 1.0, np.array([cx, cy, zc]), footprint(cx, cy, length, width, head),
                        (length, width, hmax), head, np.zeros(2), TRACK_ID, k, 'occupancy')




def selftest():
    """synthetic checks: flat road, a lying body, a car the detector already has, and noise"""
    from route import Route
    ok = True


    def check(cond, text):
        nonlocal ok
        ok &= bool(cond)
        print(('  ok    ' if cond else '  FAIL  ') + text)


    class WP:
        def __init__(self, z):
            self.transform = type('T', (), {'location': type('L', (), {'z': z})()})()
    xs = np.arange(0, 120.0, 1.0)
    route = Route(np.c_[xs, np.zeros_like(xs)], [WP(10.0) for _ in xs])
    route.update_progress((0.0, 0.0))
    rng = np.random.default_rng(0)
    # LiDAR at (0, 0, 12.4) = 2.4 m above the road at z 10, axes = world (yaw 0); y-left frame
    M = np.eye(4)
    M[:3, 3] = [0.0, 0.0, 12.4]
    road = np.c_[rng.uniform(1, 50, 20000), rng.uniform(-6, 6, 20000), np.full(20000, -2.4) + rng.normal(0, 0.01, 20000)]
    body = np.c_[rng.uniform(19.8, 20.3, 120), rng.uniform(-0.9, 0.9, 120), rng.uniform(-2.35, -2.12, 120)]
    occ = OccupancyCheck(route, 0.95, 2.3)


    def cloud(*parts):
        c = np.vstack(parts)
        c = np.c_[c[:, 0], -c[:, 1], c[:, 2], np.ones(len(c))]      # world y-right -> sensor y-left
        return c.astype(np.float32)
    check(occ(cloud(road), M, 0) is None, 'flat road only: nothing')
    ob = occ(cloud(road, body), M, 1)
    check(ob is not None and abs(ob.center[0] - 20.05) < 0.4 and ob.cls == 'unknown',
          f'lying body 20 m ahead (5-28 cm high): obstacle at x {ob.center[0] if ob is not None else None}')
    check(ob is not None and 0.1 < ob.size[2] < 0.35, 'its height is the body height')
    side = body + [0.0, 3.5, 0.0]
    check(occ(cloud(road, side), M, 2) is None, 'the same body 3.5 m to the side (next lane / footpath): nothing')
    det = Obstacle('Car', 0.9, np.array([20.0, 0.0, 10.7]), footprint(20, 0, 4.5, 1.9, 0.0), (4.5, 1.9, 1.5), 0.0)
    check(occ(cloud(road, body), M, 3, detected=[det]) is None, 'already detected there: left to the detector')
    few = body[:5]
    check(occ(cloud(road, few), M, 4) is None, '5 stray points: nothing')
    slope = road.copy()
    slope[:, 2] += 0.03 * slope[:, 0]
    occ_z = OccupancyCheck(route, 0.95, 2.3)
    occ_z.z = 10.0 + 0.03 * route.xy[:, 0]                          # the waypoints follow the slope
    check(occ_z(cloud(slope), M, 5) is None, '3% uphill road with matching waypoints: nothing')
    print('ALL OK' if ok else 'FAILURES')
    return ok




if __name__ == '__main__':
    raise SystemExit(0 if selftest() else 1)