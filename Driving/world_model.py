"""
world_model.py -- detector boxes -> world-frame obstacles the planner can use.

Detector output (pp_common.decode_predictions, both LiDAR-only and fusion):
    boxes [K,7] = x, y, z_centre, l, w, h, yaw   in the TRAINED LiDAR frame
                  (x forward, y LEFT, z up; yaw CCW from +x)
    the model learned  yaw_model = YAW_SIGN * yaw_label + YAW_OFFSET
    (pp_common constants, both default to identity)

Conversion to CARLA world, per box:
    1. undo pp_common's yaw transform
    2. build footprint corners, centre and a heading tip in the y-left frame
    3. flip y  (y-left -> CARLA's y-right sensor frame)
    4. multiply by the LiDAR's world matrix AT CAPTURE TIME
       (Stamped.lidar_tf, not the ego's current pose)
    5. world yaw = direction from transformed centre to transformed tip
Steps 3-4 are the frame bugs the M1 box check (agent.py --check-boxes) exists to catch.

Velocity: each track keeps ~1 s of positions (history_s); velocity is the least-squares
slope. A two-frame difference would turn 0.1 m of box jitter into 2 m/s. Even the slope
over 0.5 s turned the ~0.15 m jitter of real detector boxes into ~0.5 m/s on a parked car
(it then counted as moving: no steering around it, creeping up to it). So the window is 1 s
and a slope smaller than vel_sig x its own standard error (from the scatter of the positions
around the fitted line) is set to 0: not distinguishable from jitter. Such a static object is
then placed at the mean of its positions over the window (its jitter shrinks ~3x).
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Obstacle:
    cls: str
    score: float
    center: np.ndarray            # world x, y, z
    corners: np.ndarray           # (4, 2) world footprint
    size: tuple                   # length, width, height
    yaw: float                    # world yaw, radians, CARLA convention
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    track_id: int = -1
    capture_k: int = 0
    source: str = ''

    def shifted(self, dxy):
        """Copy moved by dxy (prediction from capture time to now)."""
        return Obstacle(self.cls, self.score,
                        self.center + np.r_[dxy, 0.0], self.corners + dxy, self.size,
                        self.yaw, self.velocity.copy(), self.track_id,
                        self.capture_k, self.source)


def footprint(cx, cy, l, w, yaw):
    """Four footprint corners. Same formula in the y-left frame and in CARLA's
    world frame, because each frame's yaw turns +x towards its own +y."""
    c, s = math.cos(yaw), math.sin(yaw)
    lx = np.array([1, 1, -1, -1]) * l / 2
    ly = np.array([1, -1, -1, 1]) * w / 2
    return np.stack([cx + c * lx - s * ly, cy + s * lx + c * ly], 1)


def lidar_boxes_to_world(boxes, M, yaw_sign=1.0, yaw_offset=0.0):
    """boxes [K,7] in the trained y-left frame, M = 4x4 LiDAR->world matrix
    (np.array(carla.Transform.get_matrix())). Returns centers [K,3],
    corners [K,4,2], yaw_world [K]."""
    b = np.asarray(boxes, np.float64).reshape(-1, 7)
    if len(b) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4, 2)), np.zeros(0)
    yaw = (b[:, 6] - yaw_offset) / yaw_sign
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    lx = np.array([1, 1, -1, -1, 0, 1]) * 0.5 * b[:, 3:4]     # 4 corners, centre, tip
    ly = np.array([1, -1, -1, 1, 0, 0]) * 0.5 * b[:, 4:5]
    px = b[:, 0:1] + c * lx - s * ly
    py = b[:, 1:2] + s * lx + c * ly
    pz = np.repeat(b[:, 2:3], 6, axis=1)
    pts = np.stack([px, -py, pz, np.ones_like(px)], axis=-1)   # y-left -> y-right
    w = pts @ np.asarray(M, np.float64).T
    centers = w[:, 4, :3]
    yaw_w = np.arctan2(w[:, 5, 1] - centers[:, 1], w[:, 5, 0] - centers[:, 0])
    return centers, w[:, :4, :2], yaw_w


class WorldModel:
    def __init__(self, class_names, yaw_sign=1.0, yaw_offset=0.0, min_score=0.3,
                 ego_radius=2.5, match_radius=2.0, history_s=1.0, max_track_age_s=1.0,
                 dt=0.05, source='detector', vel_sig=2.0):
        self.class_names = list(class_names)
        self.yaw_sign, self.yaw_offset = yaw_sign, yaw_offset
        self.min_score, self.ego_radius = min_score, ego_radius
        self.match_radius, self.history_s = match_radius, history_s
        self.max_track_age_s, self.dt, self.source = max_track_age_s, dt, source
        self.vel_sig = vel_sig
        self.tracks = {}
        self.next_id = 0
        self.base: List[Obstacle] = []
        self.last_capture: Optional[int] = None

    def _from_detector(self, out):
        v = out.value
        boxes = np.asarray(v['boxes'], np.float64).reshape(-1, 7)
        scores = np.asarray(v.get('scores', np.ones(len(boxes))), np.float64)
        cls = v.get('cls', v.get('labels'))
        cls = np.zeros(len(boxes), int) if cls is None else np.asarray(cls).astype(int)
        keep = (scores >= self.min_score) & (np.hypot(boxes[:, 0], boxes[:, 1]) > self.ego_radius)
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        M = np.array(out.lidar_tf.get_matrix())
        centers, corners, yaw_w = lidar_boxes_to_world(boxes, M, self.yaw_sign, self.yaw_offset)
        obs = []
        for i in range(len(boxes)):
            name = self.class_names[cls[i]] if 0 <= cls[i] < len(self.class_names) else str(cls[i])
            obs.append(Obstacle(name, float(scores[i]), centers[i], corners[i],
                                tuple(boxes[i, 3:6]), float(yaw_w[i]),
                                capture_k=out.capture_k, source=out.source))
        return obs

    def _track(self, obs, t):
        used = set()
        for ob in obs:
            best, bd = None, self.match_radius
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                dd = float(np.hypot(*(ob.center[:2] - tr['hist'][-1][1])))
                if dd < bd:
                    best, bd = tid, dd
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = {'hist': deque(maxlen=64)}
            hist = self.tracks[best]['hist']
            if hist and t - hist[-1][0] > self.history_s:
                hist.clear()                      # long gap: restart the estimate
            hist.append((t, ob.center[:2].copy()))
            while hist and t - hist[0][0] > self.history_s:
                hist.popleft()
            ts = np.array([h[0] for h in hist])
            if len(hist) >= 3 and ts[-1] - ts[0] >= 0.2:
                xy = np.array([h[1] for h in hist])
                tc = ts - ts.mean()
                ss = float((tc ** 2).sum())
                vel = (tc[:, None] * (xy - xy.mean(0))).sum(0) / ss
                if self.vel_sig > 0 and len(hist) >= 4:
                    res = xy - xy.mean(0) - tc[:, None] * vel[None]
                    se = math.sqrt(float((res ** 2).sum()) / (2.0 * (len(hist) - 2)) / ss)   # per axis
                    if math.hypot(vel[0], vel[1]) < self.vel_sig * math.sqrt(2.0) * se:
                        vel = np.zeros(2)                 # not distinguishable from box jitter
                        shift = xy.mean(0) - ob.center[:2]   # static: its best position is the mean
                        ob.center = ob.center.copy()
                        ob.center[:2] += shift
                        ob.corners = ob.corners + shift[None]
                ob.velocity = vel
            used.add(best)
            ob.track_id = best
        self.tracks = {k: v for k, v in self.tracks.items()
                       if t - v['hist'][-1][0] <= self.max_track_age_s}

    def update(self, out, k_now):
        """out: the scheduler's Stamped obstacles output (or None).
        Returns (obstacles predicted to tick k_now, fresh) where fresh means a
        new perception output arrived this tick."""
        fresh = False
        if out is None:
            self.base = []
        elif out.capture_k != self.last_capture:
            self.last_capture = out.capture_k
            self.base = self._from_detector(out)
            self._track(self.base, out.capture_k * self.dt)
            fresh = True
        return self.predicted(k_now), fresh

    def predicted(self, k_now):
        res = []
        for ob in self.base:
            age_s = (k_now - ob.capture_k) * self.dt
            res.append(ob.shifted(ob.velocity * age_s) if age_s > 0 else ob)
        return res