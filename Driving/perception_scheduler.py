"""
perception_scheduler.py

Decides which trained perception model runs on which synchronous CARLA tick,
and stamps every output with the frame it was computed from, so the decision
FSM always knows how old each piece of evidence is.

NUMBERS IT IS BUILT ON
  test1_summary_20261030_145647 (Town03, ClearNoon, 200 synced ticks,
  0 dropped, YOLO on the ONNX CPU provider):

    stream                               median    p95   ticks@50ms (p95)
    LiDAR-only PointPillars                37.7    49.7   1 (borderline)
    fusion = SegFormer 20.6 + paint 1.6
             + fusion PointPillars 35.5    57.7    71.7   2
    YOLO, ONNX CPU, incl. pre/post         85.1   101.1   3
    lane, monitor only                     19.5    34.6   1
    everything, every tick                256.7   312.5   -> 3.8 Hz

  Median/p95 only. The first LiDAR call took 3143 ms (CUDA warm-up), which
  drags its mean to 53.2 ms; warmup() takes that out of real runs.

DESIGN
  1. One obstacle source per run: --obstacle-source fusion | lidar | none.
     No LiDAR-only/fusion race. Fusion PointPillars (35.5 ms) costs about
     the same as LiDAR-only (37.7 ms); fusion's whole premium is SegFormer +
     paint (~22 ms). Running both spends ~95 ms of GPU for one answer, and
     the M8 ablation needs "fusion" and "LiDAR-only" as separate runs anyway,
     otherwise the fusion run's behaviour is partly LiDAR-only's.
  2. Rate division. Obstacles every tick (safety-critical). YOLO every 4
     ticks = 5 Hz: lights are visible tens of metres before the stop line,
     and 200 ms is ~1.7 m of travel at 30 km/h. Lane every 4 ticks (monitor
     only). YOLO runs on k%4==1 and lane on k%4==3, so they never share a
     tick with each other.
  3. Two timing modes.
     lockstep (default, the agreed plan): sync mode pauses the server while
       we compute, so slow inference costs wall-clock time, not staleness.
       An output computed on frame k is usable on frame k. Rate-divided
       streams still age (YOLO is up to 3 ticks old) and the age is tracked.
     emulated: an output computed on frame k only becomes visible at
       k + ceil(latency / 50 ms), using a FIXED latency table, with the GPU
       and the CPU each modelled as one non-preemptive queue. That is the
       closed loop the car would get on this hardware. A fixed table keeps
       runs deterministic, so seeds stay reproducible whatever else Windows
       is doing.
  4. Every output carries capture_k, capture_frame and the LiDAR world
     transform at capture (out.lidar_tf). Put boxes into the world with
     THAT transform, not the ego's current pose: a 2-tick-old box is ~1 m
     off at 36 km/h. Boxes stay in the trained y-LEFT convention.
  5. Staleness. A stream is stale when it has no output yet, or its age is
     above latency_ticks + period - 1 + slack. snapshot.healthy is False
     while any required stream (obstacles, lights) is stale; the FSM should
     answer that with a controlled stop (M6).

USAGE   (CARLA server running; run from the folder Test 1 ran from)
  python perception_scheduler.py --selftest
      schedule logic only; no CARLA, GPU or torch needed
  python perception_scheduler.py --calibrate --seg-model SEG --lane-ckpt LANE --ticks 300
      times every stream END TO END (decode/NMS/preprocessing included,
      which Test 1's timers missed) and writes latency_table_<ts>.json
  python perception_scheduler.py --obstacle-source fusion --seg-model SEG --ticks 400
  python perception_scheduler.py --obstacle-source fusion --seg-model SEG \
      --timing emulated --latency-table test1_results/latency_table_<ts>.json
  python perception_scheduler.py ... --drop-camera 200:260
      M6-style check: camera gone for ticks 200-259, watch healthy flip

IN agent.py
  streams, _ = build_streams(args, device, mods, data_collection.CFG, table)
  sched = PerceptionScheduler(streams, timing=args.timing,
                              sync_fn=lambda: sync_if_cuda(device))
  sched.warmup(first_inputs)
  while running:
      frame = world.tick()
      try:    inputs = decode_inputs(grab_synced(queues, frame, buf), frame, CFG)
      except FrameDropped: inputs = None          # ages keep growing
      snap = sched.step(inputs, frame=frame)
      if not snap.healthy: -> controlled stop
      obs = snap.get('obstacles')     # obs.value['boxes'] [K,7], obs.age_ticks,
                                      # obs.out.lidar_tf, obs.out.source
      lights = snap.get('lights')     # value: boxes / classes / scores (pixels)

The ego is on CARLA autopilot in this file's own harness ONLY to generate
frames. It is not the evaluated driving policy.
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

try:  # needed by the model runners; the scheduler core and --selftest run without them
    import cv2
    import torch
    import torch.nn.functional as F
except ImportError:
    cv2 = torch = F = None

DT = 0.05            # must equal fixed_delta_seconds (Data_collection_LIDAR CFG)
DT_MS = DT * 1000.0
CAM_OFFSET = 1.5     # carla.Location(x=1.5, ...) in setup_sensors' cam_tf
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DRIVING_YOLO_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                        5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}


# =====================================================================
# 1. Latency table
# =====================================================================
# Built-in fallback = your Test 1 run. These are forward passes (plus YOLO
# pre/post) only: PointPillars decode/NMS and SegFormer/lane preprocessing
# sat OUTSIDE Test 1's timers (its total_tick median 256.7 ms vs 200.0 ms
# for the parts), so they under-state end-to-end latency. Run --calibrate
# and pass the result with --latency-table before reporting emulated runs.
TEST1_TABLE = {
    'source': 'built-in: test1_summary_20261030_145647 (YOLO on CPU provider)',
    'streams': {
        'obstacles_fusion': {'median_ms': 57.7, 'p95_ms': 71.7},   # seg + paint + PP
        'obstacles_lidar': {'median_ms': 37.7, 'p95_ms': 49.7},
        'lights': {'median_ms': 85.1, 'p95_ms': 101.1},
        'lane': {'median_ms': 19.5, 'p95_ms': 34.6},
    },
}

_TEST1_PARTS = {'obstacles_fusion': ('segformer', 'paint', 'fusion'),
                'obstacles_lidar': ('lidar_only',),
                'lights': ('yolo',),
                'lane': ('lane',)}


def load_latency_table(path):
    """Accepts this file's own --calibrate output or a raw test1_summary_*.json."""
    if path is None:
        return TEST1_TABLE
    with open(path) as f:
        d = json.load(f)
    if 'streams' in d:
        d.setdefault('source', os.path.basename(path))
        return d
    if 'latency_ms' in d:
        L = d['latency_ms']
        streams = {}
        for name, parts in _TEST1_PARTS.items():
            if all(L.get(p) for p in parts):
                # summing per-part p95s: a conservative stand-in for the chain's p95
                streams[name] = {key: sum(L[p][key] for p in parts)
                                 for key in ('median_ms', 'p95_ms')}
        return {'source': os.path.basename(path) + ' (Test 1 format)', 'streams': streams}
    raise ValueError(f"{path}: neither a latency table nor a Test 1 summary")


def stats(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {'n': int(a.size), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


# =====================================================================
# 2. Scheduler core (stdlib + numpy only)
# =====================================================================
@dataclass
class FrameInputs:
    frame: int
    bgr: Optional[np.ndarray] = None
    rgb: Optional[np.ndarray] = None
    lidar: Optional[np.ndarray] = None   # [N,4] x-fwd, y-LEFT, z-up, intensity
    lidar_tf: Any = None                 # carla.Transform of the LiDAR at capture

    def has(self, name):
        return getattr(self, name) is not None


@dataclass
class Stamped:
    value: Any
    source: str           # 'fusion', 'fusion_unpainted', 'lidar', 'yolo', 'lane'
    capture_k: int        # scheduler tick the input was captured on
    capture_frame: int    # CARLA frame id of that input
    release_k: int        # tick the planner first saw it (== capture_k in lockstep)
    compute_ms: float     # measured wall-clock compute
    lidar_tf: Any = None


@dataclass
class StreamView:
    out: Optional[Stamped]
    age_ticks: Optional[int]
    stale: bool

    @property
    def value(self):
        return None if self.out is None else self.out.value

    @property
    def age_s(self):
        return None if self.age_ticks is None else self.age_ticks * DT


@dataclass
class Snapshot:
    k: int
    frame: Optional[int]
    views: Dict[str, StreamView]
    launched: Dict[str, float]    # stream -> compute ms, for streams run this tick
    released: List[str]
    healthy: bool

    def get(self, name):
        return self.views.get(name)


class Stream:
    """One perception job: what it runs, what it needs, how often, on what."""

    def __init__(self, name, run_fn, needs, period=1, phase=0, resource='gpu',
                 latency_ms=None, priority=0, required=False):
        self.name = name
        self.run_fn = run_fn              # FrameInputs -> (value, source)
        self.needs = tuple(needs)         # FrameInputs fields that must be present
        self.period = max(1, int(period))
        self.phase = int(phase) % self.period
        self.resource = resource          # 'gpu' or 'cpu'
        self.latency_ms = latency_ms      # from the table; used by emulated mode
        self.priority = priority          # lower runs first within a tick
        self.required = required          # stale -> snapshot unhealthy
        self.lat_ticks = 0
        self.max_age = 0
        self.latest = None
        self.pending = None
        self.counts = defaultdict(int)
        self.compute_ms = []
        self.ages = []

    def due(self, k):
        return (k - self.phase) % self.period == 0


class PerceptionScheduler:
    def __init__(self, streams, timing='lockstep', slack_ticks=2, sync_fn=None,
                 strict=True, verbose=True):
        if timing not in ('lockstep', 'emulated'):
            raise ValueError(f"timing must be 'lockstep' or 'emulated', got {timing!r}")
        self.streams = sorted(streams, key=lambda s: s.priority)   # stable sort
        self.timing = timing
        self.sync_fn = sync_fn or (lambda: None)
        self.strict = strict
        self.k = 0
        self.busy_until = defaultdict(float)   # resource -> emulated sim time, ms
        self.unhealthy_ticks = 0
        for s in self.streams:
            if timing == 'emulated':
                if s.latency_ms is None:
                    raise ValueError(f"stream '{s.name}' has no latency in the table; "
                                     f"run --calibrate and pass --latency-table")
                s.lat_ticks = max(1, math.ceil(s.latency_ms / DT_MS - 1e-9))
                if s.period < s.lat_ticks:
                    if verbose:
                        print(f"  [note] {s.name}: period {s.period} < latency {s.lat_ticks} "
                              f"ticks, raised to {s.lat_ticks} (a busy GPU cannot start "
                              f"a new job every tick)")
                    s.period = s.lat_ticks
                    s.phase %= s.period
            s.max_age = s.lat_ticks + s.period - 1 + slack_ticks

    def utilisation(self):
        """Share of each resource the table says this schedule needs (emulated mode)."""
        u = defaultdict(float)
        for s in self.streams:
            if s.latency_ms:
                u[s.resource] += s.latency_ms / (s.period * DT_MS)
        return dict(u)

    def expected_ms_per_tick(self):
        """Average compute per tick when every stream runs back to back (lockstep wall-clock)."""
        vals = [s.latency_ms / s.period for s in self.streams if s.latency_ms]
        return sum(vals) if vals else None

    def warmup(self, inputs, iters=5):
        """Run each stream a few times untimed: CUDA init, cuDNN algo picks, ORT graph setup."""
        for s in self.streams:
            if all(inputs.has(n) for n in s.needs):
                for _ in range(iters):
                    s.run_fn(inputs)
                self.sync_fn()

    def step(self, inputs, frame=None):
        """Call once per world.tick(). inputs=None means the frame was dropped."""
        k, now_ms = self.k, self.k * DT_MS
        if inputs is not None:
            frame = inputs.frame
        released, launched = [], {}

        # 1. hand over outputs whose emulated latency has elapsed
        for s in self.streams:
            if s.pending is not None and s.pending.release_k <= k:
                s.latest, s.pending = s.pending, None
                released.append(s.name)

        # 2. launch due streams, most important first
        for s in self.streams:
            if not s.due(k):
                continue
            if inputs is None or not all(inputs.has(n) for n in s.needs):
                s.counts['skipped_no_input'] += 1
                continue
            if s.pending is not None:
                s.counts['skipped_busy'] += 1
                continue
            t0 = time.perf_counter()
            try:
                value, source = s.run_fn(inputs)
                self.sync_fn()
            except Exception:
                s.counts['errors'] += 1
                if self.strict:
                    raise
                if s.counts['errors'] <= 3:
                    traceback.print_exc()
                continue
            ms = (time.perf_counter() - t0) * 1000.0
            s.counts['runs'] += 1
            s.compute_ms.append(ms)
            if self.timing == 'lockstep':
                release_k = k
            else:
                start = max(now_ms, self.busy_until[s.resource])
                finish = start + s.latency_ms
                self.busy_until[s.resource] = finish
                release_k = k + math.ceil((finish - now_ms) / DT_MS - 1e-9)
            out = Stamped(value, source, k, frame, release_k, ms, inputs.lidar_tf)
            if release_k <= k:
                s.latest = out
                released.append(s.name)
            else:
                s.pending = out
            launched[s.name] = ms

        # 3. what the planner sees this tick
        views, healthy = {}, True
        for s in self.streams:
            age = None if s.latest is None else k - s.latest.capture_k
            stale = age is None or age > s.max_age
            if age is not None:
                s.ages.append(age)
            if stale:
                s.counts['stale_ticks'] += 1
                healthy = healthy and not s.required
            views[s.name] = StreamView(s.latest, age, stale)
        if not healthy:
            self.unhealthy_ticks += 1
        self.k += 1
        return Snapshot(k, frame, views, launched, released, healthy)

    def summary(self):
        out = {'timing': self.timing, 'ticks': self.k, 'unhealthy_ticks': self.unhealthy_ticks,
               'utilisation': self.utilisation(), 'streams': {}}
        for s in self.streams:
            out['streams'][s.name] = {
                'resource': s.resource, 'period': s.period, 'phase': s.phase,
                'latency_ms_table': s.latency_ms, 'latency_ticks': s.lat_ticks,
                'stale_after_ticks': s.max_age, 'required': s.required,
                'counts': dict(s.counts), 'compute_ms': stats(s.compute_ms),
                'age_ticks': stats(s.ages),
            }
        return out


def print_plan(sched, table, qkey):
    print(f"\nschedule: {sched.timing}  (latency table: {table.get('source')}, {qkey})")
    print(f"  {'stream':17s} {'res':4s} {'every':>5s} {'phase':>5s} {'lat ms':>7s} "
          f"{'lat tk':>6s} {'stale >':>7s}  needs")
    for s in sched.streams:
        lat = '-' if s.latency_ms is None else f"{s.latency_ms:.1f}"
        print(f"  {s.name:17s} {s.resource:4s} {s.period:5d} {s.phase:5d} {lat:>7s} "
              f"{s.lat_ticks:6d} {s.max_age:7d}  {'+'.join(s.needs)}")
    est = sched.expected_ms_per_tick()
    if est:
        print(f"  expected model compute per tick, back to back: {est:.1f} ms "
              f"(~{1000.0 / est:.1f} Hz before server time and decoding)")
    if sched.timing == 'emulated':
        for res, u in sorted(sched.utilisation().items()):
            flag = ('  <-- over 1.0: jobs queue up and get skipped; raise a period'
                    if u > 1.0 else '')
            print(f"  {res} utilisation {u:.2f}{flag}")


# =====================================================================
# 3. Model runners (same preprocessing as Test 1, which matched training)
# =====================================================================
def _need_torch():
    if torch is None or cv2 is None:
        raise ImportError("torch and opencv-python are required for the model runners")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')   # no MPS: pp_common's voxelize is not verified MPS-safe


def sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _to_numpy(d):
    return {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in d.items()}


def _check_boxes(name, preds):
    b = preds['boxes']
    if not (b.ndim == 2 and (b.shape[0] == 0 or b.shape[1] == 7)):
        raise AssertionError(f"[{name}] expected boxes [K,7], got {b.shape}")


def paint_frame(lidar_xyzi_left, seg_logits, to_cam_fn, project_fn,
                fov_deg, cam_offset, native_w, native_h, device):
    """Unchanged from Test 1: live SegFormer scores -> painted points, with the
    uint8 round-trip the fusion checkpoint was trained on."""
    n_cls = seg_logits.shape[1]
    f = native_w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    cam = to_cam_fn(lidar_xyzi_left[:, :3], 'left', cam_offset)
    u, v, _, ok = project_fn(cam, f, native_w / 2.0, native_h / 2.0, native_w, native_h)
    extra = np.zeros((len(lidar_xyzi_left), n_cls + 1), dtype=np.uint8)
    if ok.any():
        gx = (u[ok] + 0.5) / native_w * 2 - 1
        gy = (v[ok] + 0.5) / native_h * 2 - 1
        grid = torch.from_numpy(np.stack([gx, gy], 1)).float().to(device).view(1, 1, -1, 2)
        samp = F.grid_sample(seg_logits, grid, mode='bilinear',
                             padding_mode='border', align_corners=False)
        probs = torch.softmax(samp[0, :, 0, :], dim=0).T.detach().cpu().numpy()
        extra[ok, :n_cls] = np.round(probs * 255).astype(np.uint8)
        extra[ok, n_cls] = 255
    painted = np.concatenate([lidar_xyzi_left, extra.astype(np.float32) / 255.0], axis=1)
    return painted.astype(np.float32)


class LidarPP:
    def __init__(self, pp_module, ckpt_path, device, score_thr):
        _need_torch()
        self.pp, self.score_thr = pp_module, score_thr
        self.model = pp_module.build_model_from_ckpt(
            torch.load(ckpt_path, map_location='cpu'), device)
        if self.model.num_pt_feats != 4:
            raise ValueError(f"{ckpt_path}: num_pt_feats={self.model.num_pt_feats}, expected 4 "
                             f"(is --lidar-ckpt actually the fusion checkpoint?)")

    def __call__(self, inp):
        with torch.no_grad():
            out = self.model([torch.from_numpy(inp.lidar)])
            preds = _to_numpy(self.pp.decode_predictions(
                out, self.model, score_thr=self.score_thr)[0])
        _check_boxes('lidar', preds)
        return preds, 'lidar'


# transformers 5.9 renamed SegformerDecodeHead.linear_c -> linear_projections (checked
# against the 4.46.3 / 5.8.1 / 5.9.0 / 5.22.1 sources). A checkpoint saved with >= 5.9
# and loaded with <= 5.8 -- e.g. 4.46.3, the last release for Python 3.8 -- gets those
# 4 decode-head layers RANDOMLY INITIALISED, with only a warning. from_pretrained in
# the CARLA env did exactly that. Both directions are mapped here.
SEG_KEY_RENAMES = (('decode_head.linear_projections.', 'decode_head.linear_c.'),
                   ('decode_head.linear_c.', 'decode_head.linear_projections.'))


def _read_state_dict(seg_dir):
    st = os.path.join(seg_dir, 'model.safetensors')
    if os.path.isfile(st):
        from safetensors.torch import load_file
        return load_file(st)
    pt = os.path.join(seg_dir, 'pytorch_model.bin')
    if os.path.isfile(pt):
        return torch.load(pt, map_location='cpu')
    raise FileNotFoundError(f"{seg_dir}: no model.safetensors or pytorch_model.bin "
                            f"(sharded checkpoints are not handled here)")


def load_segformer(seg_dir, verbose=True):
    """from_pretrained loads and, on transformers 5.x, converts the old layer names itself;
    its own loading report is the strict check. Only if that report shows missing or unused
    tensors (e.g. 4.46.3 reading a decode head saved as linear_projections) are the raw
    weights re-loaded with the rename above. Raises if anything is still missing, instead
    of running with random layers. Returns (model on CPU, report dict)."""
    _need_torch()
    import transformers
    from transformers import SegformerForSemanticSegmentation
    from transformers.utils import logging as hf_logging
    prev = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()          # its warning is replaced by the strict check below
    try:
        model, info = SegformerForSemanticSegmentation.from_pretrained(seg_dir, output_loading_info=True)
    finally:
        hf_logging.set_verbosity(prev)
    missing = [k for k in info.get('missing_keys', []) if not k.endswith('num_batches_tracked')]
    unexpected = list(info.get('unexpected_keys', []))
    mismatched = list(info.get('mismatched_keys', []))
    renamed, first_report = 0, (missing, unexpected)
    if missing or unexpected:
        want = set(model.state_dict().keys())
        fixed = {}
        for k, v in _read_state_dict(seg_dir).items():
            if k not in want:
                for a, b in SEG_KEY_RENAMES:
                    if k.startswith(a) and b + k[len(a):] in want:
                        k, renamed = b + k[len(a):], renamed + 1
                        break
            fixed[k] = v
        missing, unexpected = model.load_state_dict(fixed, strict=False)
        missing = [k for k in missing if not k.endswith('num_batches_tracked')]
        if missing or unexpected:
            missing, unexpected = first_report
    if missing or unexpected or mismatched:
        raise RuntimeError(f"SegFormer checkpoint {seg_dir} does not load fully in this transformers "
                           f"version.\n  missing: {missing}\n  unused: {unexpected}\n  wrong shape: {mismatched}")
    report = {'tensors': len(model.state_dict()), 'renamed': renamed,
              'transformers': transformers.__version__}
    if verbose:
        note = (f", {renamed} decode-head keys renamed (linear_projections <-> linear_c)"
                if renamed else "")
        print(f"  SegFormer weights: all {report['tensors']} tensors loaded strictly{note}")
    return model.eval(), report


class FusionPP:
    def __init__(self, pp_module, ckpt_path, seg_dir, seg_hw, device, score_thr,
                 projection, cam):
        _need_torch()
        self.pp, self.score_thr, self.device = pp_module, score_thr, device
        self.proj, self.cam, self.seg_hw = projection, cam, tuple(seg_hw)
        seg, self.seg_report = load_segformer(seg_dir)
        self.seg = seg.to(device).eval()
        self.n_cls = self.seg.config.num_labels
        self.model = pp_module.build_model_from_ckpt(
            torch.load(ckpt_path, map_location='cpu'), device)
        want = 4 + self.n_cls + 1
        if self.model.num_pt_feats != want:
            raise ValueError(f"fusion num_pt_feats={self.model.num_pt_feats}, but 4 + "
                             f"SegFormer classes ({self.n_cls}) + 1 = {want}: this pairing "
                             f"was not trained together")

    def __call__(self, inp):
        with torch.no_grad():
            if inp.rgb is not None:
                h, w = self.seg_hw
                x = cv2.resize(inp.rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                logits = self.seg(pixel_values=x_t).logits.float()
                painted = paint_frame(inp.lidar, logits, self.proj.to_cam, self.proj.project,
                                      self.cam['fov'], self.cam['offset'],
                                      self.cam['w'], self.cam['h'], self.device)
                seg_labels = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
                source = 'fusion'
            else:
                # Only reachable with --camera-loss unpainted. Every point gets zero
                # paint and projected-flag 0, which is what points outside the camera
                # FOV always looked like in training: graceful, not invented.
                pad = np.zeros((len(inp.lidar), self.n_cls + 1), dtype=np.float32)
                painted = np.concatenate([inp.lidar, pad], axis=1)
                seg_labels, source = None, 'fusion_unpainted'
            out = self.model([torch.from_numpy(painted)])
            preds = _to_numpy(self.pp.decode_predictions(
                out, self.model, score_thr=self.score_thr)[0])
        _check_boxes('fusion', preds)
        preds['seg_labels'] = seg_labels     # [h', w'] uint8 at SegFormer resolution
        return preds, source


def preprocess_yolo(bgr, size=800):
    """Letterbox to size x size, float32 RGB BCHW (unchanged from Test 1)."""
    h, w = bgr.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = size - nw, size - nh
    left, right = int(round(pw / 2 - 0.1)), int(round(pw / 2 + 0.1))
    top, bottom = int(round(ph / 2 - 0.1)), int(round(ph / 2 + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    t = np.ascontiguousarray(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None])
    return t, {'scale': scale, 'left': left, 'top': top, 'w': w, 'h': h}


def postprocess_yolo(output, meta, conf_thr=0.25, iou_thr=0.45):
    """Test 1's decoder, now also returning scores. Driving classes only, per-class NMS."""
    pred = np.asarray(output)[0].transpose(1, 0)          # [N, 84]
    xywh, cls_scores = pred[:, :4], pred[:, 4:]
    boxes, classes, scores = [], [], []
    for cid in sorted(DRIVING_YOLO_CLASSES):
        sc = cls_scores[:, cid]
        idx = np.where(sc >= conf_thr)[0]
        if idx.size == 0:
            continue
        b, sc = xywh[idx], sc[idx]
        x1 = np.clip((b[:, 0] - b[:, 2] / 2 - meta['left']) / meta['scale'], 0, meta['w'])
        y1 = np.clip((b[:, 1] - b[:, 3] / 2 - meta['top']) / meta['scale'], 0, meta['h'])
        x2 = np.clip((b[:, 0] + b[:, 2] / 2 - meta['left']) / meta['scale'], 0, meta['w'])
        y2 = np.clip((b[:, 1] + b[:, 3] / 2 - meta['top']) / meta['scale'], 0, meta['h'])
        ok = (x2 > x1) & (y2 > y1)
        if not ok.any():
            continue
        xyxy = np.stack([x1[ok], y1[ok], x2[ok], y2[ok]], axis=1)
        sc = sc[ok]
        keep = cv2.dnn.NMSBoxes([[float(a), float(c), float(e - a), float(g - c)]
                                 for a, c, e, g in xyxy],
                                sc.astype(float).tolist(), float(conf_thr), float(iou_thr))
        if len(keep) == 0:
            continue
        keep = np.asarray(keep).reshape(-1)
        boxes.extend(xyxy[keep].tolist())
        classes.extend([cid] * len(keep))
        scores.extend(sc[keep].tolist())
    return (np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(classes, dtype=int), np.asarray(scores, dtype=np.float32))


class YoloOnnx:
    def __init__(self, onnx_path, provider='auto', conf=0.25, iou=0.45, size=800):
        _need_torch()
        import onnxruntime as ort
        gpu = 'CUDAExecutionProvider'
        avail = ort.get_available_providers()
        if provider == 'cuda' and gpu not in avail:
            raise RuntimeError("--yolo-provider cuda, but this onnxruntime has no CUDA "
                               "provider (pip uninstall onnxruntime; pip install onnxruntime-gpu)")
        use_gpu = provider == 'cuda' or (provider == 'auto' and gpu in avail)
        providers = [gpu, 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.active = self.sess.get_providers()[0]
        if use_gpu and self.active != gpu:
            print("  [warn] CUDA provider requested but ORT fell back to CPU "
                  "(usually a CUDA/cuDNN DLL mismatch)")
        self.resource = 'gpu' if self.active == gpu else 'cpu'
        i, o = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        if list(i.shape) != [1, 3, size, size]:
            raise ValueError(f"YOLO ONNX input {i.shape}, expected [1, 3, {size}, {size}]")
        if len(o.shape) != 3 or o.shape[1] != 84:
            raise ValueError(f"YOLO ONNX output {o.shape}, expected [1, 84, N]")
        self.in_name, self.out_name = i.name, o.name
        self.conf, self.iou, self.size = conf, iou, size

    def __call__(self, inp):
        x, meta = preprocess_yolo(inp.bgr, self.size)
        raw = self.sess.run([self.out_name], {self.in_name: x})[0]
        boxes, classes, scores = postprocess_yolo(raw, meta, self.conf, self.iou)
        return {'boxes': boxes, 'classes': classes, 'scores': scores}, 'yolo'


class LaneNet:
    """DeepLabV3-MobileNetV3, 1 channel; same build as lane_validate.py / Test 1."""

    def __init__(self, ckpt_path, device):
        _need_torch()
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        ckpt = torch.load(ckpt_path, map_location='cpu')
        m = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None)
        m.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
        m.aux_classifier = None
        m.load_state_dict(ckpt['model'])
        self.model = m.to(device).eval()
        self.hw = tuple(ckpt.get('img_size', [384, 768]))
        self.device = device

    def __call__(self, inp):
        h, w = self.hw
        x = cv2.resize(inp.rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x_t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x_t)['out'])[0, 0]
        return {'prob': prob.to(torch.float16).cpu().numpy()}, 'lane'


def build_streams(args, device, mods, cfg, table):
    """mods: {'pp_lidar', 'pp_fusion', 'projection'} modules. Loads only what the run needs."""
    qkey = 'p95_ms' if args.latency_quantile == 'p95' else 'median_ms'

    def lat(name):
        return table['streams'].get(name, {}).get(qkey)

    cam = {'fov': cfg['fov'], 'offset': CAM_OFFSET, 'w': cfg['img_w'], 'h': cfg['img_h']}
    calib = getattr(args, 'calibrate', False)
    streams = []

    if calib or args.obstacle_source == 'lidar':
        fn = LidarPP(mods['pp_lidar'], args.lidar_ckpt, device, args.score_thr)
        print(f"  LiDAR-only PointPillars: {args.lidar_ckpt}")
        streams.append(Stream('obstacles_lidar' if calib else 'obstacles', fn, ('lidar',),
                              period=args.obstacle_period, phase=0, resource='gpu',
                              latency_ms=lat('obstacles_lidar'), priority=0, required=True))
    if calib or args.obstacle_source == 'fusion':
        fn = FusionPP(mods['pp_fusion'], args.fusion_ckpt, args.seg_model, args.seg_img_size,
                      device, args.score_thr, mods['projection'], cam)
        print(f"  SegFormer ({fn.n_cls} classes) + fusion PointPillars: {args.fusion_ckpt}")
        needs = ('lidar',) if args.camera_loss == 'unpainted' else ('rgb', 'lidar')
        streams.append(Stream('obstacles_fusion' if calib else 'obstacles', fn, needs,
                              period=args.obstacle_period, phase=0, resource='gpu',
                              latency_ms=lat('obstacles_fusion'), priority=0, required=True))
    if not calib and args.obstacle_source == 'none':
        print("  obstacles: NONE (route-only / perception-disabled baseline)")

    yolo = YoloOnnx(args.yolo_ckpt, args.yolo_provider, args.yolo_conf)
    print(f"  YOLO ONNX: {args.yolo_ckpt} on {yolo.active}")
    streams.append(Stream('lights', yolo, ('bgr',), period=args.lights_period, phase=1,
                          resource=yolo.resource, latency_ms=lat('lights'),
                          priority=1, required=True))
    if args.lane_ckpt:
        lane = LaneNet(args.lane_ckpt, device)
        print(f"  lane (monitor only): {args.lane_ckpt}")
        streams.append(Stream('lane', lane, ('rgb',), period=args.lane_period, phase=3,
                              resource='gpu', latency_ms=lat('lane'), priority=2))
    if calib:
        for s in streams:
            s.period, s.phase = 1, 0
    return streams, {'yolo_provider': yolo.active}


# =====================================================================
# 4. CARLA harness (sync mode, autopilot ego as a frame generator only)
# =====================================================================
def load_module(module_name, file_path):
    """Explicit-path import: both pp_common.py files share a name (see Test 1)."""
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise ImportError(f"expected '{module_name}' at {file_path}; check the --*-dir flags")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_ckpts(args):
    if args.lidar_ckpt is None:
        args.lidar_ckpt = os.path.join(args.detection_lidar_dir, 'runs_lidar', 'pp_town03', 'best.pt')
    if args.fusion_ckpt is None:
        args.fusion_ckpt = os.path.join(args.fusion_dir, 'runs_lidar', 'pp_painted', 'best.pt')
    if args.yolo_ckpt is None:
        args.yolo_ckpt = os.path.join(args.detection_yolo_dir, 'runs', 'detect', 'runs', 'detect',
                                      'carla_finetune-10', 'weights', 'best.onnx')


def decode_inputs(data, frame_id, cfg):
    h, w = cfg['img_h'], cfg['img_w']
    bgr = np.frombuffer(data['rgb'].raw_data, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()
    lidar = np.frombuffer(data['lidar'].raw_data, dtype=np.float32).reshape(-1, 4).copy()
    lidar[:, 1] = -lidar[:, 1]    # CARLA y-right -> trained y-left, same flip as data_collection
    return FrameInputs(frame=frame_id, bgr=bgr, rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                       lidar=lidar, lidar_tf=data['lidar'].transform)


def spawn_vehicles(world, bp, tm, n):
    out = []
    pts = world.get_map().get_spawn_points()
    random.shuffle(pts)
    vbps = list(bp.filter('vehicle.*'))
    for sp in pts[:n]:
        v = world.try_spawn_actor(random.choice(vbps), sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            out.append(v)
    print(f"  background traffic: {len(out)}/{n} vehicles (no walkers in this harness)")
    return out


def parse_range(text):
    if not text:
        return None
    a, b = text.split(':')
    return int(a), int(b)


def in_range(i, r):
    return r is not None and r[0] <= i < r[1]


def count_dets(value):
    if isinstance(value, dict) and value.get('boxes') is not None:
        return len(value['boxes'])
    return ''


def fmt_view(v):
    if v is None:
        return '-'
    if v.age_ticks is None:
        return 'none yet'
    s = f"age {v.age_ticks}" + (' STALE' if v.stale else '')
    n = count_dets(v.value)
    return s if n == '' else s + f" n={n}"


def print_summary(sched, wall, sim_s, wall_s, dropped, title):
    bar = '=' * 84
    print(f"\n{bar}\n{title}\n{bar}")
    if wall_s > 0:
        print(f"{sched.k} ticks = {sim_s:.1f} s sim in {wall_s:.1f} s wall -> real-time factor "
              f"{sim_s / wall_s:.2f} ({sched.k / wall_s:.1f} Hz), {dropped} frames dropped, "
              f"{sched.unhealthy_ticks} unhealthy ticks")
    for name in ('server_ms', 'grab_ms', 'perception_ms', 'total_ms'):
        st = stats(wall.get(name, []))
        if st:
            print(f"  {name:14s} median {st['median']:7.1f}   p95 {st['p95']:7.1f}   "
                  f"max {st['max']:7.1f} ms")
    print(f"\n  {'stream':17s} {'res':4s} {'per':>3s} {'runs':>5s} {'busy':>4s} {'noin':>4s} "
          f"{'err':>3s}  {'compute med/p95 ms':>18s}  {'age mean/max':>12s} {'stale':>5s}")
    for s in sched.streams:
        c, st, ag = s.counts, stats(s.compute_ms), stats(s.ages)
        comp = f"{st['median']:.1f}/{st['p95']:.1f}" if st else '-'
        age = f"{ag['mean']:.1f}/{ag['max']:.0f}" if ag else '-'
        print(f"  {s.name:17s} {s.resource:4s} {s.period:3d} {c['runs']:5d} "
              f"{c['skipped_busy']:4d} {c['skipped_no_input']:4d} {c['errors']:3d}  "
              f"{comp:>18s}  {age:>12s} {c['stale_ticks']:5d}")
    print(bar)


def check_table_vs_live(sched, qkey):
    key = 'p95' if qkey == 'p95_ms' else 'median'
    for s in sched.streams:
        st = stats(s.compute_ms)
        if not s.latency_ms or not st or st['n'] < 20:
            continue
        r = st[key] / s.latency_ms
        if r > 1.15 or r < 0.7:
            print(f"  [warn] {s.name}: live {key} {st[key]:.1f} ms vs table {s.latency_ms:.1f} ms "
                  f"(x{r:.2f}). Emulated latencies are off for this setup; re-run --calibrate.")


def write_latency_table(path, sched, meta):
    streams = {}
    for s in sched.streams:
        st = stats(s.compute_ms)
        if st:
            streams[s.name] = {'median_ms': round(st['median'], 2), 'p95_ms': round(st['p95'], 2),
                               'max_ms': round(st['max'], 2), 'n': st['n'], 'resource': s.resource}
    with open(path, 'w') as f:
        json.dump(dict(meta, format='perception_scheduler latency table v1', streams=streams),
                  f, indent=2)


def run_carla(args):
    import carla
    _need_torch()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    resolve_ckpts(args)
    device = get_device()
    print(f"device: {device}")

    print("\nloading project modules ...")
    mods = {
        'dc': load_module('dc_module', os.path.join(args.data_collection_dir, 'data_collection.py')),
        'pp_lidar': load_module('pp_common_lidar', os.path.join(args.detection_lidar_dir, 'pp_common.py')),
        'pp_fusion': load_module('pp_common_fusion', os.path.join(args.fusion_dir, 'pp_common.py')),
        'projection': load_module('check_projection_live',
                                  os.path.join(args.fusion_dir, 'check_projection.py')),
    }
    dc = mods['dc']
    dc.CFG['use_semantic_lidar'] = False      # GT-only sensor, never a model input
    cfg = dc.CFG

    table = load_latency_table(args.latency_table)
    qkey = 'p95_ms' if args.latency_quantile == 'p95' else 'median_ms'
    print("\nloading trained models ...")
    streams, info = build_streams(args, device, mods, cfg, table)
    timing = 'lockstep' if args.calibrate else args.timing
    sched = PerceptionScheduler(streams, timing=timing, slack_ticks=args.slack_ticks,
                                sync_fn=lambda: sync_if_cuda(device), strict=not args.lenient)
    print_plan(sched, table, qkey)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = 'calibrate' if args.calibrate else f"{args.obstacle_source}_{timing}"
    csv_path = os.path.join(args.out_dir, f'sched_{tag}_{stamp}.csv')
    json_path = os.path.join(args.out_dir, f'sched_{tag}_{stamp}.json')
    drop_cam, drop_lidar = parse_range(args.drop_camera), parse_range(args.drop_lidar)

    print(f"\nconnecting to CARLA at {args.carla_host}:{args.carla_port} ...")
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(args.carla_timeout)
    client.load_world(args.town)
    world = client.get_world()
    world.set_weather(carla.WeatherParameters.ClearNoon)
    bp = world.get_blueprint_library()
    tm = client.get_trafficmanager()
    tm.set_random_device_seed(args.seed)
    original_settings = world.get_settings()

    sensors, actors = {}, []
    wall = defaultdict(list)
    dropped, t_start, t_end = 0, None, None
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = ['k', 'frame', 'server_ms', 'grab_ms', 'perception_ms', 'total_ms', 'healthy',
              'launched', 'released']
    for s in sched.streams:
        header += [f'{s.name}_age', f'{s.name}_stale', f'{s.name}_compute_ms',
                   f'{s.name}_count', f'{s.name}_source']
    writer.writerow(header)

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        ego = world.spawn_actor(bp.filter('vehicle.tesla.model3')[0],
                                random.choice(world.get_map().get_spawn_points()))
        actors.append(ego)
        print("\n*** ego on CARLA autopilot ONLY to generate frames; not the evaluated policy ***\n")
        ego.set_autopilot(True, tm.get_port())
        actors += spawn_vehicles(world, bp, tm, args.n_background_vehicles)

        sensors, queues = dc.setup_sensors(world, ego, bp)
        actors += list(sensors.values())
        if args.drop_gt_cameras:
            for key in ('sem', 'depth'):
                s = sensors.pop(key, None)
                queues.pop(key, None)
                if s is not None:
                    s.stop()
                    s.destroy()
                    actors.remove(s)
            print("  removed 'sem' and 'depth' cameras (GT-only; saves server render time)")

        for _ in range(30):   # let the ego and traffic settle
            world.tick()
        dc.drain_queues(queues)
        buffers = {}

        # warm-up on one real synced frame: removes Test 1's 3143 ms first-call spike
        frame_id = world.tick()
        first = decode_inputs(dc.grab_synced(queues, frame_id, buffers), frame_id, cfg)
        t0 = time.perf_counter()
        sched.warmup(first, iters=args.warmup_iters)
        print(f"warm-up: {args.warmup_iters} untimed passes per stream, "
              f"{time.perf_counter() - t0:.1f} s")
        dc.drain_queues(queues)
        buffers = {}

        print(f"\nrunning {args.ticks} ticks ...\n")
        t_start = time.perf_counter()
        for i in range(args.ticks):
            t0 = time.perf_counter()
            frame_id = world.tick()
            t1 = time.perf_counter()
            try:
                inputs = decode_inputs(dc.grab_synced(queues, frame_id, buffers), frame_id, cfg)
            except dc.FrameDropped as e:
                dropped += 1
                inputs = None
                print(f"  [warn] {e}")
            t2 = time.perf_counter()
            if inputs is not None:
                if in_range(i, drop_cam):
                    inputs.bgr = inputs.rgb = None
                if in_range(i, drop_lidar):
                    inputs.lidar = None
            snap = sched.step(inputs, frame=frame_id)
            t3 = time.perf_counter()

            server_ms, grab_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
            perc_ms, total_ms = (t3 - t2) * 1000, (t3 - t0) * 1000
            for name, val in (('server_ms', server_ms), ('grab_ms', grab_ms),
                              ('perception_ms', perc_ms), ('total_ms', total_ms)):
                wall[name].append(val)

            row = [snap.k, frame_id, f"{server_ms:.2f}", f"{grab_ms:.2f}", f"{perc_ms:.2f}",
                   f"{total_ms:.2f}", int(snap.healthy), '+'.join(snap.launched),
                   '+'.join(snap.released)]
            for s in sched.streams:
                v = snap.get(s.name)
                row += ['' if v.age_ticks is None else v.age_ticks, int(v.stale),
                        f"{snap.launched[s.name]:.2f}" if s.name in snap.launched else '',
                        count_dets(v.value), '' if v.out is None else v.out.source]
            writer.writerow(row)

            if (i + 1) % args.print_every == 0:
                parts = ' | '.join(f"{s.name} {fmt_view(snap.get(s.name))}" for s in sched.streams)
                state = 'healthy' if snap.healthy else 'UNHEALTHY'
                print(f"k {snap.k:4d} frame {frame_id} | {parts} | perception {perc_ms:.0f} ms, "
                      f"server {server_ms:.0f} ms | {state}")
        t_end = time.perf_counter()

    except KeyboardInterrupt:
        print("\ninterrupted; cleaning up and summarising what ran ...")

    finally:
        if t_start is not None and t_end is None:
            t_end = time.perf_counter()
        csv_file.close()
        print("\ncleaning up ...")
        def _safe(what, fn):
            try:
                fn()
            except BaseException as e:          # incl. Ctrl+C pressed again: skip this step only
                print(f"  [warn] {what}: {type(e).__name__} {e}")

        for sen in sensors.values():
            _safe('could not stop a sensor', lambda sen=sen: sen.stop() if sen.is_alive else None)
        alive = []

        def _collect():
            for a in actors:
                if a.is_alive:
                    alive.append(a)
        _safe('could not list actors', _collect)
        if alive:
            _safe('could not destroy all actors',
                  lambda: client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in alive], True))
        _safe('could not set the Traffic Manager back to async', lambda: tm.set_synchronous_mode(False))
        _safe('could not fully restore world settings', lambda: world.apply_settings(original_settings))

        sim_s = sched.k * DT
        wall_s = (t_end - t_start) if t_start is not None else 0.0
        title = ('CALIBRATION (every stream, every tick, lockstep)' if args.calibrate else
                 f"SCHEDULER RUN: obstacles={args.obstacle_source}, timing={timing}")
        print_summary(sched, wall, sim_s, wall_s, dropped, title)
        if timing == 'emulated':          # lockstep never uses the table
            check_table_vs_live(sched, qkey)

        meta = {'created': stamp, 'town': args.town, 'seed': args.seed, 'device': str(device),
                'yolo_provider': info['yolo_provider'], 'ticks_run': sched.k,
                'dropped_frames': dropped, 'warmup_iters': args.warmup_iters, 'dt': DT}
        result = dict(meta, obstacle_source=args.obstacle_source, timing=timing,
                      latency_table=table.get('source'), latency_quantile=qkey,
                      real_time_factor=(sim_s / wall_s) if wall_s else None,
                      wall_ms={k: stats(v) for k, v in wall.items()},
                      scheduler=sched.summary())
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"saved: {csv_path}\nsaved: {json_path}")
        if args.calibrate and sched.k > 0:
            table_path = os.path.join(args.out_dir, f'latency_table_{stamp}.json')
            write_latency_table(table_path, sched, meta)
            print(f"saved: {table_path}\n  -> pass it to every ablation run: "
                  f"--timing emulated --latency-table {table_path}")


# =====================================================================
# 5. Self-test: scheduling logic with instant fake models, no CARLA
# =====================================================================
def selftest():
    T = TEST1_TABLE['streams']
    failures = []
    n_checks = [0]

    def expect(cond, msg):
        n_checks[0] += 1
        if not cond:
            failures.append(msg)

    def fake(tag):
        return lambda inp: ({'boxes': np.zeros((0, 7), np.float32)}, tag)

    def make(timing):
        return PerceptionScheduler([
            Stream('obstacles', fake('fusion'), ('rgb', 'lidar'), period=1, phase=0,
                   resource='gpu', latency_ms=T['obstacles_fusion']['p95_ms'],
                   priority=0, required=True),
            Stream('lights', fake('yolo'), ('bgr',), period=4, phase=1, resource='cpu',
                   latency_ms=T['lights']['p95_ms'], priority=1, required=True),
            Stream('lane', fake('lane'), ('rgb',), period=4, phase=3, resource='gpu',
                   latency_ms=T['lane']['p95_ms'], priority=2),
        ], timing=timing, slack_ticks=2)

    def inputs(k, camera=True):
        img = np.zeros((2, 2, 3), np.uint8) if camera else None
        return FrameInputs(frame=1000 + k, bgr=img, rgb=img, lidar=np.zeros((1, 4), np.float32))

    def age(snap, name):
        v = snap.get(name)
        if v.age_ticks is None:
            return '-'
        return f"{v.age_ticks}{'*' if v.stale else ''}"

    for timing, drop, n_ticks in (('lockstep', None, 12), ('emulated', (8, 14), 22)):
        print(f"\n{'#' * 84}\n{timing}" + (f", camera dropped on ticks {drop[0]}-{drop[1] - 1}"
                                            if drop else ''))
        sched = make(timing)
        print_plan(sched, TEST1_TABLE, 'p95_ms')
        print(f"\n  {'k':>3s}  {'cam':3s}  {'launched':24s} {'released':24s} "
              f"{'obs':>4s} {'lights':>6s} {'lane':>4s}  healthy   (* = stale)")
        snaps = []
        for k in range(n_ticks):
            cam = not in_range(k, drop)
            snap = sched.step(inputs(k, camera=cam))
            snaps.append(snap)
            launched = '+'.join(snap.launched) or '-'
            released = '+'.join(snap.released) or '-'
            print(f"  {k:3d}  {'on' if cam else 'OFF':3s}  {launched:24s} {released:24s} "
                  f"{age(snap, 'obstacles'):>4s} {age(snap, 'lights'):>6s} "
                  f"{age(snap, 'lane'):>4s}  {'yes' if snap.healthy else 'no'}")

        if timing == 'lockstep':
            expect(all(s.get('obstacles').age_ticks == 0 for s in snaps),
                   'lockstep: obstacles should always be age 0')
            expect(not snaps[0].healthy and all(s.healthy for s in snaps[1:]),
                   'lockstep: healthy from k=1, the first YOLO tick')
            expect(max(s.get('lights').age_ticks for s in snaps[1:]) == 3,
                   'lockstep: lights age should cycle 0..3')
        else:
            ob = [s.get('obstacles') for s in snaps]
            obs_stream = [s for s in sched.streams if s.name == 'obstacles'][0]
            expect(obs_stream.period == 2 and obs_stream.lat_ticks == 2,
                   'emulated: fusion (71.7 ms) should need 2 ticks and run every 2')
            expect(ob[1].out is None and ob[2].age_ticks == 2 and ob[2].out.capture_k == 0,
                   'emulated: first obstacle output (captured k=0) visible at k=2')
            expect(not snaps[3].healthy and snaps[4].healthy,
                   'emulated: healthy from k=4 (lights captured k=1 + 3 ticks)')
            expect(all(s.healthy for s in snaps[4:12]), 'emulated: healthy on k=4..11')
            expect(all(not snaps[k].get('obstacles').stale for k in range(8, 12)) and
                   snaps[12].get('obstacles').stale,
                   'emulated: obstacles go stale at k=12 (last capture k=6, stale after 5)')
            expect(all(not snaps[k].healthy for k in range(12, 20)),
                   'emulated: unhealthy on k=12..19')
            expect(not ob[16].stale and ob[16].out.capture_k == 14,
                   'emulated: obstacles fresh again at k=16 (camera back at k=14)')
            expect(snaps[20].healthy and snaps[21].healthy,
                   'emulated: healthy again at k=20, once YOLO (captured k=17) lands')

    # loader: your raw Test 1 summary JSON must give the same table as the built-in one
    t1 = {'latency_ms': {
        'lidar_only': {'median_ms': 37.7, 'p95_ms': 49.7},
        'yolo': {'median_ms': 85.1, 'p95_ms': 101.1},
        'segformer': {'median_ms': 20.6, 'p95_ms': 23.5},
        'paint': {'median_ms': 1.6, 'p95_ms': 2.0},
        'fusion': {'median_ms': 35.5, 'p95_ms': 46.2},
        'lane': {'median_ms': 19.5, 'p95_ms': 34.6}}}
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(t1, f)
        tmp = f.name
    try:
        tab = load_latency_table(tmp)
    finally:
        os.remove(tmp)
    for name, want in T.items():
        got = tab['streams'].get(name, {})
        expect(all(abs(got.get(q, -1) - want[q]) < 1e-6 for q in ('median_ms', 'p95_ms')),
               f"loader: {name} {got} != {want}")

    print(f"\n{'#' * 84}")
    if failures:
        for msg in failures:
            print(f"FAIL  {msg}")
        print(f"selftest: {len(failures)} of {n_checks[0]} checks failed")
        return 1
    print(f"selftest: all {n_checks[0]} checks passed")
    return 0


# =====================================================================
# 6. CLI
# =====================================================================
def add_model_args(p):
    g = p.add_argument_group('models / repo layout (same defaults as Test 1)')
    g.add_argument('--data-collection-dir', default='../Data_collection_LIDAR')
    g.add_argument('--detection-lidar-dir', default='../Detection_LIDAR')
    g.add_argument('--fusion-dir', default='../Fusion')
    g.add_argument('--detection-yolo-dir', default='../Detection_YOLO')
    g.add_argument('--lidar-ckpt', default=None)
    g.add_argument('--fusion-ckpt', default=None)
    g.add_argument('--yolo-ckpt', default=None, help='best.onnx')
    g.add_argument('--seg-model', default=None,
                   help='SegFormer dir; required for --obstacle-source fusion and --calibrate')
    g.add_argument('--seg-img-size', type=int, nargs=2, default=[384, 768], metavar=('H', 'W'))
    g.add_argument('--lane-ckpt', default=None, help='optional; lane is monitor-only')
    g.add_argument('--score-thr', type=float, default=0.1)
    g.add_argument('--yolo-conf', type=float, default=0.25)
    g.add_argument('--yolo-provider', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='auto = CUDA if onnxruntime-gpu is installed, else CPU (what Test 1 used)')


def add_scheduler_args(p):
    g = p.add_argument_group('scheduler')
    g.add_argument('--obstacle-source', choices=['fusion', 'lidar', 'none'], default='fusion')
    g.add_argument('--timing', choices=['lockstep', 'emulated'], default='lockstep')
    g.add_argument('--latency-table', default=None,
                   help='--calibrate output or a test1_summary_*.json; default: built-in Test 1')
    g.add_argument('--latency-quantile', choices=['p95', 'median'], default='p95')
    g.add_argument('--obstacle-period', type=int, default=1)
    g.add_argument('--lights-period', type=int, default=4)
    g.add_argument('--lane-period', type=int, default=4)
    g.add_argument('--slack-ticks', type=int, default=2,
                   help='extra ticks of age tolerated before a stream counts as stale')
    g.add_argument('--camera-loss', choices=['stale', 'unpainted'], default='stale',
                   help='fusion with no camera frame: go stale (safe default) or run on '
                        'unpainted points (degraded mode to study, label it as such)')
    g.add_argument('--lenient', action='store_true',
                   help='log model exceptions and let that stream go stale instead of crashing')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--selftest', action='store_true',
                   help='check the scheduling logic with fake models; no CARLA needed')
    p.add_argument('--calibrate', action='store_true',
                   help='time every stream end to end, every tick; write latency_table_<ts>.json')
    add_model_args(p)
    add_scheduler_args(p)
    g = p.add_argument_group('harness')
    g.add_argument('--carla-host', default='localhost')
    g.add_argument('--carla-port', type=int, default=2000)
    g.add_argument('--carla-timeout', type=float, default=20.0)
    g.add_argument('--town', default='Town03', help='Town03 = same as Test 1, so latencies compare')
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--ticks', type=int, default=400)
    g.add_argument('--print-every', type=int, default=20)
    g.add_argument('--n-background-vehicles', type=int, default=15)
    g.add_argument('--warmup-iters', type=int, default=5)
    g.add_argument('--drop-camera', default=None, metavar='A:B',
                   help='simulate camera loss on ticks A..B-1')
    g.add_argument('--drop-lidar', default=None, metavar='A:B',
                   help='simulate LiDAR loss on ticks A..B-1')
    g.add_argument('--drop-gt-cameras', action='store_true',
                   help="destroy the GT-only 'sem' and 'depth' cameras after setup_sensors. "
                        "Experimental: works only if data_collection.grab_synced waits on "
                        "the queues dict it is given rather than on fixed keys")
    g.add_argument('--out-dir', default='test1_results')
    args = p.parse_args()
    if not args.selftest:
        if (args.calibrate or args.obstacle_source == 'fusion') and not args.seg_model:
            p.error('--seg-model is required for --obstacle-source fusion and for --calibrate')
        if args.timing == 'emulated' and args.latency_table is None and not args.calibrate:
            print("[note] emulated timing on the built-in Test 1 table, which leaves out "
                  "decode/NMS/preprocessing. Fine for a first look; --calibrate before reporting.")
    return args


def main():
    args = parse_args()
    if args.selftest:
        sys.exit(selftest())
    run_carla(args)


if __name__ == '__main__':
    main()
