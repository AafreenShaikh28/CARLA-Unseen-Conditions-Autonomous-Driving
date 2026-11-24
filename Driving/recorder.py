"""
recorder.py - presentation video of a closed-loop run (agent.py --record).

A camera sensor follows the ego (behind and above it, or straight down) and every simulation tick is
written to <run folder>/video.mp4 at 20 frames per second of SIMULATION time. The video therefore plays at
real speed however slowly the run went (synchronous mode waits for the models every tick, so a screen
recording of the CARLA window would come out slow and uneven).

Drawn on every frame:
  - the route ahead (green line on the road)
  - what the car perceives: boxes of the obstacles the decision logic uses
        red = the obstacle it is reacting to, orange = other obstacles in its path, cyan = the rest
  - a panel: driving state, speed and target speed, gap and time-to-collision to the obstacle ahead,
    route progress, obstacle source; "COLLISION" flashes on an impact
  - throttle / brake bars
  - picture-in-picture of the car's own camera with the same boxes (runs with a camera model:
    --obstacle-source lidar / fusion); "CAMERA LOST" when the camera is dropped (--drop-camera)

The raw file is written with OpenCV. If ffmpeg is available (pip install imageio-ffmpeg, or ffmpeg on
PATH) it is re-encoded to H.264, which PowerPoint and every player can open; otherwise the OpenCV file is
kept and the conversion command is printed.
"""
import math
import os
import queue
import shutil
import subprocess

import numpy as np
import cv2
import carla

STATE_COL = {'FOLLOW_ROUTE': (80, 200, 80), 'BLOCKED_WAIT': (0, 200, 255), 'EMERGENCY_STOP': (40, 40, 255),
             'DEGRADED': (0, 140, 255), 'ROUTE_COMPLETE': (255, 160, 60), 'STOP_FOR_LIGHT': (60, 60, 230),
             'CRASH_STOP': (30, 30, 150), 'YIELD': (0, 170, 255)}
STATE_TXT = {'FOLLOW_ROUTE': 'FOLLOWING ROUTE', 'BLOCKED_WAIT': 'STOPPED, WAITING', 'EMERGENCY_STOP': 'EMERGENCY STOP',
             'DEGRADED': 'SENSOR LOST, STOPPING', 'ROUTE_COMPLETE': 'ROUTE COMPLETE',
             'STOP_FOR_LIGHT': 'STOPPING FOR RED LIGHT', 'CRASH_STOP': 'HIT A PEDESTRIAN: STOPPED',
             'YIELD': 'YIELDING'}
MANOEUVRE_TXT = {'steer_around': 'steering around the obstacle', 'lane_change': 'changing lane',
                 'other_lane': 'in the other lane', 'change_back': 'changing back', 'yield': 'yielding at the junction',
                 'merge_wait': 'waiting to change back', 'wait_gap': 'waiting for a gap'}
LIGHT_COL = {'Red': (40, 40, 255), 'Yellow': (0, 210, 255), 'Green': (80, 220, 80)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
LEAD, IN_PATH, OTHER = (40, 40, 255), (0, 165, 255), (230, 200, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIEWS = {   # camera pose relative to the ego origin, and how it is attached
    'chase': (carla.Location(x=-7.5, z=3.4), carla.Rotation(pitch=-14.0), 'SpringArmGhost'),
    'top': (carla.Location(x=6.0, z=38.0), carla.Rotation(pitch=-90.0), 'Rigid'),
}


def box8(ob):
    """8 world corners of an Obstacle (footprint corners + bottom/top z)."""
    z0, z1 = ob.center[2] - ob.size[2] / 2, ob.center[2] + ob.size[2] / 2
    c = np.asarray(ob.corners, float)
    return np.vstack([np.c_[c, np.full(4, z0)], np.c_[c, np.full(4, z1)]])


def to_px(P, inv, f, w, h):
    """world points [N,3] -> pixels [N,2] and a mask of points in front of the camera
    (UE camera frame: x forward, y right, z up)."""
    q = np.c_[P, np.ones(len(P))] @ np.asarray(inv).T
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    ok = x > 0.3
    xs = np.where(ok, x, 1.0)
    return np.stack([w / 2 + f * y / xs, h / 2 - f * z / xs], 1), ok


def draw_boxes(img, obstacles, hits, inv, f, w, h, thick=2, labels=True, ego_xy=None, lead_gap=None):
    in_path, crossing = {}, set()
    for j, hh in enumerate(hits):
        in_path.setdefault(id(hh.obstacle), j)
        if getattr(hh, 'kind', '') == 'crossing':
            crossing.add(id(hh.obstacle))
    for ob in obstacles:
        j = in_path.get(id(ob))
        col = LEAD if j == 0 else IN_PATH if j is not None else OTHER
        px, ok = to_px(box8(ob), inv, f, w, h)
        if not ok.all() or (np.abs(px) > 4 * max(w, h)).any():
            continue
        p = px.astype(int)
        for a, b in EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), col, thick + (1 if j == 0 else 0), cv2.LINE_AA)
        if labels and ego_xy is not None:
            d = math.hypot(ob.center[0] - ego_xy[0], ob.center[1] - ego_xy[1])
            txt = (f'{ob.cls}  gap {lead_gap:.1f} m' if j == 0 and lead_gap is not None   # same as the panel
                   else f'{ob.cls} {d:.0f} m') + ('  CROSSING' if id(ob) in crossing else '')
            x0, y0 = int(p[:, 0].min()), int(p[:, 1].min()) - 6
            cv2.putText(img, txt, (x0, max(y0, 14)), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (x0, max(y0, 14)), FONT, 0.5, col, 1, cv2.LINE_AA)


def panel(img, x, y, w, h, alpha=0.55):
    roi = img[y:y + h, x:x + w]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which('ffmpeg')


class Recorder:
    def __init__(self, world, ego, bp_lib, out_path, dt, size=(1280, 720), view='chase', fov=90.0,
                 title='', pip_fov=None):
        self.out_path, self.dt, self.title = out_path, dt, title
        self.w, self.h = int(size[0]), int(size[1])
        self.f = self.w / (2 * math.tan(math.radians(fov) / 2))
        self.pip_fov = pip_fov
        loc, rot, att = VIEWS[view]
        b = bp_lib.find('sensor.camera.rgb')
        for k, v in (('image_size_x', str(self.w)), ('image_size_y', str(self.h)), ('fov', str(fov)),
                     ('sensor_tick', '0.0')):
            if b.has_attribute(k):
                b.set_attribute(k, v)
        kind = getattr(getattr(carla, 'AttachmentType', None), att, None)
        tf = carla.Transform(loc, rot)
        self.sensor = (world.spawn_actor(b, tf, attach_to=ego, attachment_type=kind) if kind is not None
                       else world.spawn_actor(b, tf, attach_to=ego))
        self.q = queue.Queue()
        self.sensor.listen(self.q.put)
        self.raw_path = os.path.splitext(out_path)[0] + '_raw.mp4'
        self.writer = None
        for cc in ('avc1', 'mp4v'):
            wr = cv2.VideoWriter(self.raw_path, cv2.VideoWriter_fourcc(*cc), 1.0 / dt, (self.w, self.h))
            if wr.isOpened():
                self.writer, self.codec = wr, cc
                break
        if self.writer is None:
            self.raw_path = os.path.splitext(out_path)[0] + '_raw.avi'
            self.writer, self.codec = cv2.VideoWriter(self.raw_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                                      1.0 / dt, (self.w, self.h)), 'MJPG'
        self.frames = self.missed = 0
        self.n_coll, self.flash = 0, 0
        print(f'  recording: {view} camera {self.w}x{self.h} at {1.0 / dt:.0f} fps of simulation time -> '
              f'{out_path}')

    def _image(self, frame, timeout=5.0):
        """the camera image of this simulation frame (older ones are dropped)"""
        while True:
            try:
                im = self.q.get(timeout=timeout)
            except queue.Empty:
                return None
            if im.frame == frame:
                return im
            if im.frame > frame:            # this frame was skipped by the server: keep the newer one
                self.q.put(im)
                return None

    def frame(self, frame, t, dec, v, control, obstacles, hits, route, s_ego, ego_xy, n_collisions,
              source='', pip=None, pip_tf=None, camera_expected=False, light=None, manoeuvre=''):
        """light = (state, gap to the stop line, its two 3D end points, obeyed) or None;
        manoeuvre = manoeuvre.py's status (+ ': why' while waiting), drawn in the panel"""
        im = self._image(frame)
        if im is None:
            self.missed += 1
            return
        img = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(self.h, self.w, 4)[:, :, :3].copy()
        inv = im.transform.get_inverse_matrix()

        # route ahead on the road
        if route.wps is not None:
            lo, hi = np.searchsorted(route.s, [s_ego, s_ego + 45.0])
            idx = np.arange(lo, min(hi + 1, len(route.xy)))
            if len(idx) >= 2:
                P = np.array([[route.xy[i, 0], route.xy[i, 1], route.wps[i].transform.location.z + 0.05]
                              for i in idx])
                px, ok = to_px(P, inv, self.f, self.w, self.h)
                for a in range(len(idx) - 1):
                    if ok[a] and ok[a + 1]:
                        cv2.line(img, tuple(px[a].astype(int)), tuple(px[a + 1].astype(int)), (60, 220, 60), 4,
                                 cv2.LINE_AA)

        # the next stop line, in the colour of its light
        if light is not None and light[2] is not None and light[1] < 45.0:
            px, ok = to_px(np.array(light[2]), inv, self.f, self.w, self.h)
            if ok.all() and (np.abs(px) < 4 * max(self.w, self.h)).all():
                cv2.line(img, tuple(px[0].astype(int)), tuple(px[1].astype(int)),
                         LIGHT_COL.get(light[0], (200, 200, 200)), 6, cv2.LINE_AA)

        draw_boxes(img, obstacles, hits, inv, self.f, self.w, self.h, ego_xy=ego_xy, lead_gap=dec.lead_gap)

        # picture-in-picture: the car's own camera with the same boxes
        if camera_expected:
            pw = self.w // 3
            if pip is not None and pip_tf is not None and self.pip_fov:
                ph_src, pw_src = pip.shape[:2]
                cam = pip.copy()
                fp = pw_src / (2 * math.tan(math.radians(self.pip_fov) / 2))
                draw_boxes(cam, obstacles, hits, pip_tf.get_inverse_matrix(), fp, pw_src, ph_src, thick=2,
                           labels=False)
                ph = int(pw * ph_src / pw_src)
                small = cv2.resize(cam, (pw, ph), interpolation=cv2.INTER_AREA)
            else:
                ph = int(pw * 0.75)
                small = np.zeros((ph, pw, 3), np.uint8)
                cv2.putText(small, 'CAMERA LOST', (pw // 2 - 95, ph // 2), FONT, 0.9, (40, 40, 255), 2, cv2.LINE_AA)
            x0, y0 = self.w - pw - 12, 12
            img[y0:y0 + ph, x0:x0 + pw] = small
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph), (255, 255, 255), 1)
            cv2.putText(img, 'car camera', (x0 + 6, y0 + ph - 8), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # info panel
        panel(img, 12, 12, 410, 168 + (26 if light is not None else 0) + (26 if manoeuvre else 0))
        st = dec.state
        cv2.rectangle(img, (12, 12), (422, 46), STATE_COL.get(st, (200, 200, 200)), -1)
        cv2.putText(img, STATE_TXT.get(st, st), (22, 38), FONT, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        lines = [f'speed {v * 3.6:5.1f} km/h   target {dec.v_target * 3.6:5.1f}',
                 ('obstacle ahead  ' + (f'{dec.lead_gap:5.1f} m' if dec.lead_gap is not None else '  none')
                  + (f'   TTC {dec.lead_ttc:4.1f} s' if dec.lead_ttc is not None and dec.lead_ttc < 20.0
                     else '')),
                 f'route {100 * route.completion:5.1f} %    t {t:6.1f} s',
                 f'obstacles: {source}']
        if light is not None:
            lines.append(f'light ahead  {light[0].upper():6s} {light[1]:5.1f} m'
                         + ('' if light[3] else '   (not obeyed)'))
        if manoeuvre:
            word, _, why = manoeuvre.partition(': ')
            text = MANOEUVRE_TXT.get(word, word) + (f': {why}' if why else '')
            lines.append(text if len(text) <= 46 else text[:45] + '.')
        for i, s in enumerate(lines):
            cv2.putText(img, s, (22, 72 + 26 * i), FONT, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        # pedals
        bx, by = 12, self.h - 64
        panel(img, bx, by, 220, 52)
        for i, (name, val, col) in enumerate((('throttle', control.throttle, (80, 200, 80)),
                                              ('brake', control.brake, (40, 40, 255)))):
            y = by + 10 + 22 * i
            cv2.putText(img, name, (bx + 8, y + 12), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(img, (bx + 80, y), (bx + 80 + 130, y + 14), (90, 90, 90), 1)
            cv2.rectangle(img, (bx + 80, y), (bx + 80 + int(130 * min(max(val, 0.0), 1.0)), y + 14), col, -1)

        if self.title:
            (tw, _), _ = cv2.getTextSize(self.title, FONT, 0.55, 1)
            panel(img, self.w - tw - 24, self.h - 36, tw + 16, 28)
            cv2.putText(img, self.title, (self.w - tw - 16, self.h - 16), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if n_collisions > self.n_coll:
            self.n_coll, self.flash = n_collisions, int(2.0 / self.dt)
        if self.flash > 0:
            self.flash -= 1
            cv2.putText(img, 'COLLISION', (self.w // 2 - 150, self.h // 2), FONT, 2.0, (0, 0, 0), 8, cv2.LINE_AA)
            cv2.putText(img, 'COLLISION', (self.w // 2 - 150, self.h // 2), FONT, 2.0, (40, 40, 255), 4, cv2.LINE_AA)

        self.writer.write(img)
        self.frames += 1

    def close(self):
        """finish the file; re-encode to H.264 if ffmpeg is available. Returns the video path."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.frames == 0:
            print('  [video] no frames were recorded')
            return None
        secs = self.frames * self.dt
        miss = f', {self.missed} ticks without a camera image' if self.missed else ''
        ff = find_ffmpeg()
        if ff:
            cmd = [ff, '-y', '-loglevel', 'error', '-i', self.raw_path, '-c:v', 'libx264', '-preset', 'veryfast',
                   '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', self.out_path]
            try:
                subprocess.run(cmd, check=True, timeout=900)
                os.remove(self.raw_path)
                print(f'  [video] {self.out_path}  ({self.frames} frames, {secs:.1f} s, H.264{miss})')
                return self.out_path
            except Exception as e:
                print(f'  [video] H.264 conversion failed ({e}); keeping the OpenCV file')
        print(f'  [video] {self.raw_path}  ({self.frames} frames, {secs:.1f} s, codec {self.codec}{miss})')
        print('          For PowerPoint, convert to H.264:  pip install imageio-ffmpeg  (then the next run does it '
              'itself), or:  ffmpeg -i "' + self.raw_path + '" -c:v libx264 -preset veryfast -crf 20 '
              '-pix_fmt yuv420p "' + self.out_path + '"')
        return self.raw_path