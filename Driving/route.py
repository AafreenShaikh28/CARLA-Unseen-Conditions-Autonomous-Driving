"""
route.py -- the planned path as a polyline, and everything measured along it.

Pure numpy, no CARLA import, so it can be tested offline. agent.py builds it
from GlobalRoutePlanner.trace_route() with waypoints about 1 m apart.

Two coordinates describe any point relative to the path:
    s  = distance along the path (metres)
    d  = signed lateral offset from the path (metres)
Coordinates are CARLA world x, y. CARLA is left-handed, which only flips the
sign of d; every check in decision.py uses the d-interval, never its sign alone.
"""
import numpy as np


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Route:
    def __init__(self, xy, waypoints=None):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        # drop near-duplicate points (trace_route can emit two almost identical points where
        # one road edge meets the next): a few-mm segment at a skewed angle looks like a sharp
        # curve to speed_cap and tilts tangent_at. Keep points >= 40% of the usual spacing apart.
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        min_sep = 0.4 * float(np.median(seg[seg > 1e-3])) if np.any(seg > 1e-3) else 1e-3
        keep = [0]
        for i in range(1, len(xy)):
            if np.linalg.norm(xy[i] - xy[keep[-1]]) >= min_sep:
                keep.append(i)
        if keep[-1] != len(xy) - 1:                        # always end exactly at the goal
            if len(keep) > 1 and np.linalg.norm(xy[-1] - xy[keep[-2]]) >= min_sep:
                keep[-1] = len(xy) - 1
            elif np.linalg.norm(xy[-1] - xy[keep[-1]]) > 1e-3:
                keep.append(len(xy) - 1)
        self.xy = xy[keep]
        if len(self.xy) < 2:
            raise ValueError("route needs at least 2 distinct points")
        self.wps = None if waypoints is None else [waypoints[i] for i in keep]
        seg = np.diff(self.xy, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.r_[0.0, np.cumsum(self.seg_len)]       # s at each point
        self.length = float(self.s[-1])
        self.heading = np.arctan2(seg[:, 1], seg[:, 0])     # per segment
        self.idx = 0                                        # ego's segment (forward-only)
        self.s_ego = 0.0

    # ---------------------------------------------------------------- core
    def _nearest(self, P, lo, hi):
        """Nearest point on segments lo..hi-1 for each row of P.
        Returns segment index, fraction t along it, distance, signed d."""
        A, B = self.xy[lo:hi], self.xy[lo + 1:hi + 1]
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-9)
        AP = P[:, None, :] - A[None]
        t = np.clip((AP * AB[None]).sum(2) / L2[None], 0.0, 1.0)
        dist = np.linalg.norm(AP - t[..., None] * AB[None], axis=2)
        k = dist.argmin(1)
        n = np.arange(len(P))
        cross = AB[k, 0] * AP[n, k, 1] - AB[k, 1] * AP[n, k, 0]
        d = np.where(cross >= 0, 1.0, -1.0) * dist[n, k]
        return lo + k, t[n, k], dist[n, k], d

    def update_progress(self, ego_xy, window=60):
        """Move the ego's index forward only (no jumping to a later loop of the
        same road). Returns (s_ego, lateral deviation in metres)."""
        lo = max(self.idx - 2, 0)
        hi = min(self.idx + window, len(self.seg_len))
        j, t, dist, _ = self._nearest(np.asarray(ego_xy, float).reshape(1, 2), lo, hi)
        self.idx = int(j[0])
        self.s_ego = float(self.s[self.idx] + t[0] * self.seg_len[self.idx])
        return self.s_ego, float(dist[0])

    def project(self, P, horizon=60.0, return_beyond=False):
        """Project world points onto the path ahead of the ego.
        Returns s_rel (from the ego), d, s_abs (from the route start), and with
        return_beyond=True a mask of points past the end of the searched window
        (route end or horizon). Those snap onto the window's last point, where
        left and right corners get opposite signs: their s and d are meaningless."""
        P = np.asarray(P, float).reshape(-1, 2)
        lo = self.idx
        hi = int(np.searchsorted(self.s, self.s_ego + horizon)) + 1
        hi = min(max(hi, lo + 1), len(self.seg_len))
        j, t, _, d = self._nearest(P, lo, hi)
        s_abs = self.s[j] + t * self.seg_len[j]
        if return_beyond:
            return s_abs - self.s_ego, d, s_abs, (j == hi - 1) & (t >= 1.0)
        return s_abs - self.s_ego, d, s_abs

    def tangent_at(self, s_abs):
        j = int(np.clip(np.searchsorted(self.s, s_abs) - 1, 0, len(self.heading) - 1))
        h = self.heading[j]
        return np.array([np.cos(h), np.sin(h)])

    # ------------------------------------------------------------- helpers
    @property
    def remaining(self):
        return self.length - self.s_ego

    @property
    def completion(self):
        return self.s_ego / self.length if self.length > 0 else 1.0

    def target_index(self, lookahead):
        return min(int(np.searchsorted(self.s, self.s_ego + lookahead)), len(self.xy) - 1)

    def target_waypoint(self, lookahead):
        """carla.Waypoint about `lookahead` metres ahead (or xy if built offline)."""
        i = self.target_index(lookahead)
        return self.wps[i] if self.wps is not None else self.xy[i]

    def lookahead_point(self, ego_xy, L):
        """Pure-pursuit target: the first point ahead on the path at straight-line
        distance L from the car. It moves continuously with the car. The older
        target (s_ego + L) did not: inside a tight curve the car's projection
        hops from one segment to the next, and each hop kicked the steering PID."""
        P = np.asarray(ego_xy, float)
        for j in range(self.idx, len(self.seg_len)):
            A, B = self.xy[j], self.xy[j + 1]
            if np.linalg.norm(B - P) >= L:
                d, f = B - A, A - P
                a, b, c = d @ d, 2.0 * (f @ d), f @ f - L * L
                disc = b * b - 4 * a * c
                if disc < 0:                  # car further than L from the path: head for B
                    return B.copy()
                t = (-b + np.sqrt(disc)) / (2 * a)
                return A + float(np.clip(t, 0.0, 1.0)) * d
        return self.xy[-1].copy()

    def point_at(self, s_abs):
        """xy exactly at path position s_abs, interpolated between points. Steering at
        this instead of at the nearest waypoint removes the 1 m jumps that make the
        PID's derivative term chatter."""
        s_abs = float(np.clip(s_abs, 0.0, self.length))
        j = int(np.clip(np.searchsorted(self.s, s_abs) - 1, 0, len(self.seg_len) - 1))
        t = (s_abs - self.s[j]) / max(self.seg_len[j], 1e-9)
        return self.xy[j] + t * (self.xy[j + 1] - self.xy[j])

    def speed_cap(self, a_lat_max=2.0, b=2.0, horizon=40.0, span=5.0, v_max=50.0):
        """Fastest speed now that still lets the car slow down (at b) to every
        upcoming curve's limit sqrt(a_lat_max / curvature) before reaching it."""
        i0 = self.idx
        i1 = min(int(np.searchsorted(self.s, self.s_ego + horizon)) + 1, len(self.heading))
        h, s = self.heading[i0:i1], self.s[i0:i1]
        step = max(float(np.median(self.seg_len[i0:i1])) if i1 > i0 else 1.0, 0.1)
        k = max(1, int(round(span / step)))
        if len(h) <= k:
            return v_max
        kappa = np.abs(wrap_pi(h[k:] - h[:-k])) / np.maximum(s[k:] - s[:-k], 1e-3)
        v_curve = np.sqrt(a_lat_max / np.maximum(kappa, 1e-6))
        dist = np.maximum(s[:-k] - self.s_ego, 0.0)
        return float(min(v_max, np.min(np.sqrt(v_curve ** 2 + 2.0 * b * dist))))
