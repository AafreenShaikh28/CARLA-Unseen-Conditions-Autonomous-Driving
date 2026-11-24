"""
lidar_weather.py - what rain and fog do to a real LiDAR, added on top of CARLA's clean point cloud.

CARLA's ray-cast LiDAR only finds where each ray meets a solid surface; rain and fog exist only in
the camera image. This module post-processes each point cloud with a simple LiDAR-equation model,
so LiDAR-only and fusion both get the same degraded points:

  1. extinction  Light is lost both ways through rain / fog (Beer-Lambert): received power
                 ~ reflectivity * exp(-2 * alpha * d) / d^2. A return is lost when that falls below
                 what the weakest surface gives at the sensor range R0 in clear air, i.e. when
                 d * exp(alpha * d) > R0 * sqrt(reflectivity / REFL_MIN). CARLA has no reflectivity,
                 so it is drawn per point (REFL_MIN..1). Intensity is scaled by exp(-2 * alpha * d).
                 alpha = 0 (clear, night) keeps every point: the clear-weather cloud is unchanged.
  2. backscatter A share of the rays is reflected by droplets close to the sensor: that ray returns a
                 weak false point 1-15 m away instead of the real surface.
  3. random loss Spray and water on the sensor window (rain).
  4. range noise Small Gaussian error along the ray.

Weather -> parameters (assumptions, not calibrated to one sensor; change the table below):
  rain   from CARLA's precipitation (0-100); at 100: alpha 0.008 /m, 10% lost, 1% backscatter, 3 cm noise
  fog    from the fog density WE set (agent.py --weather Fog = 60): alpha = density / 1000 /m,
         i.e. 60 -> 0.06 /m = ~50 m visibility (3 / alpha); backscatter on alpha x 100 % of the rays
  night  no effect (a LiDAR has its own laser). The haze CARLA's presets add for the camera is ignored.

In the car:   python agent.py ... --weather Fog --lidar-weather auto
Offline:      python lidar_weather.py --src ..\\LIDAR_only\\kitti_weather --dst ..\\LIDAR_only\\kitti_weather_lidarwx
              (copies each <Weather>/training with a degraded velodyne/; everything else is hard-linked)
Check:        python lidar_weather.py --selftest
"""
import os
import json
import shutil
import zlib
import argparse
from collections import OrderedDict

import numpy as np

REFL_MIN = 0.2           # darkest surface; it is just detectable at the sensor range in clear air
RAIN_AT_100 = dict(alpha=0.008, loss=0.10, noise=0.03, clutter=0.01, clutter_rmean=3.0)
FOG_PER_DENSITY = 0.001  # alpha (1/m) per CARLA fog_density unit: 60 -> 0.06 /m, ~50 m visibility
FOG_NOISE = 0.02
CLUTTER_RMIN, CLUTTER_RMAX = 1.0, 15.0
CLUTTER_INTENSITY = (0.02, 0.20)
DEFAULT_RANGE = 60.0     # collect_kitti_lidar.py lidar_range
# CARLA preset precipitation, used only when the carla module cannot be imported (offline)
PRECIP_BY_NAME = (('HardRain', 100.0), ('MidRain', 60.0), ('SoftRain', 30.0))


def profile(precipitation=0.0, fog_density=0.0):
    """parameters for a weather; 'active' is False when nothing changes (clear, cloudy, night)"""
    p = min(max(float(precipitation or 0.0), 0.0), 100.0) / 100.0
    a_fog = max(float(fog_density or 0.0), 0.0) * FOG_PER_DENSITY
    a_rain = RAIN_AT_100['alpha'] * p
    c_rain, c_fog = RAIN_AT_100['clutter'] * p, min(a_fog, 0.5)
    rm_fog = float(np.clip(0.25 / a_fog, 2.0, 10.0)) if a_fog > 0 else 0.0
    clutter = c_rain + c_fog
    prm = OrderedDict(
        precipitation=round(p * 100.0, 1), fog_density=round(float(fog_density or 0.0), 1),
        alpha=round(a_rain + a_fog, 5), loss=round(RAIN_AT_100['loss'] * p, 4),
        noise=round(max(RAIN_AT_100['noise'] * p, FOG_NOISE if a_fog > 0 else 0.0), 4),
        clutter=round(clutter, 5),
        clutter_rmean=round((c_rain * RAIN_AT_100['clutter_rmean'] + c_fog * rm_fog) / clutter, 2) if clutter else 0.0)
    prm['active'] = bool(prm['alpha'] > 0 or prm['loss'] > 0 or prm['clutter'] > 0 or prm['noise'] > 0)
    return prm


def max_range(alpha, r0=DEFAULT_RANGE, refl=REFL_MIN):
    """farthest distance a surface of this reflectivity still returns a point (bisection)"""
    target = r0 * np.sqrt(refl / REFL_MIN)
    if alpha <= 0:
        return min(target, r0)
    lo, hi = 0.0, r0
    if hi * np.exp(alpha * hi) <= target:
        return r0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid * np.exp(alpha * mid) <= target else (lo, mid)
    return lo


def describe(prm, r0=DEFAULT_RANGE):
    if not prm['active']:
        return 'no LiDAR effect (clear / night / no rain or fog we set)'
    return (f"alpha {prm['alpha']:.3f}/m: dark surfaces lost beyond {max_range(prm['alpha'], r0):.0f} m, "
            f"bright beyond {max_range(prm['alpha'], r0, 1.0):.0f} m (clear: {r0:.0f} m); "
            f"{prm['loss'] * 100:.0f}% random loss; {prm['clutter'] * 100:.1f}% of rays false points "
            f"~{prm['clutter_rmean']:.0f} m out; range noise {prm['noise'] * 100:.0f} cm")


def degrade(points, prm, rng, r0=DEFAULT_RANGE, return_fake=False):
    """points [N, 4] x, y, z, intensity in the sensor frame (any axis convention) -> new [M, 4] float32.
    Returns the input unchanged when the weather has no LiDAR effect. return_fake: also the mask of
    the false (backscatter) points, for tests."""
    if prm is None or not prm['active'] or points is None or len(points) == 0:
        return (points, np.zeros(0 if points is None else len(points), bool)) if return_fake else points
    pts = np.asarray(points)
    xyz = pts[:, :3].astype(np.float64)
    inten = pts[:, 3].astype(np.float64)
    n = len(pts)
    d = np.maximum(np.linalg.norm(xyz, axis=1), 1e-3)
    u = xyz / d[:, None]
    a = prm['alpha']
    refl = rng.uniform(REFL_MIN, 1.0, n)
    returns = d * np.exp(a * d) <= r0 * np.sqrt(refl / REFL_MIN)        # 1. extinction
    if prm['loss'] > 0:
        returns &= rng.random(n) >= prm['loss']                         # 3. random loss
    fake = rng.random(n) < prm['clutter'] if prm['clutter'] > 0 else np.zeros(n, bool)   # 2. backscatter
    real = returns & ~fake
    dr = d[real] + (rng.normal(0.0, prm['noise'], int(real.sum())) if prm['noise'] > 0 else 0.0)   # 4.
    out_real = np.c_[u[real] * np.maximum(dr, 0.1)[:, None], inten[real] * np.exp(-2.0 * a * d[real])]
    m = int(fake.sum())
    rc = CLUTTER_RMIN + rng.exponential(max(prm['clutter_rmean'] - CLUTTER_RMIN, 0.1), m)
    rc = np.minimum(rc, np.minimum(d[fake] - 0.1, CLUTTER_RMAX))          # in front of the real surface
    keep = rc >= 0.5
    out_fake = np.c_[u[fake][keep] * rc[keep, None], rng.uniform(*CLUTTER_INTENSITY, int(keep.sum()))]
    out = np.vstack([out_real, out_fake]).astype(np.float32)
    if return_fake:
        return out, np.r_[np.zeros(len(out_real), bool), np.ones(len(out_fake), bool)]
    return out


# ------------------------------------------------------------------ offline: KITTI-layout folders
def preset_precipitation(name):
    try:
        import carla
        w = getattr(carla.WeatherParameters, name, None)
        if isinstance(w, carla.WeatherParameters):
            return float(w.precipitation), 'CARLA preset'
    except ImportError:
        pass
    for key, val in PRECIP_BY_NAME:
        if key in name:
            return val, 'from the name (carla not importable)'
    return 0.0, 'from the name (carla not importable)' if name != 'Fog' else 'Fog = ClearNoon + fog'


def link_tree(src, dst):
    """hard-link (or copy) everything except velodyne*/ folders"""
    linked = copied = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel == '.':
            dirs[:] = [x for x in dirs if not x.startswith('velodyne')]
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            a, b = os.path.join(root, f), os.path.join(dst, rel, f)
            if os.path.exists(b):
                continue
            try:
                os.link(a, b)
                linked += 1
            except OSError:
                shutil.copy2(a, b)
                copied += 1
    return linked, copied


def run_offline(args):
    names = args.only or sorted(x for x in os.listdir(args.src)
                                if os.path.isdir(os.path.join(args.src, x, 'training', 'velodyne')))
    if not names:
        raise SystemExit(f'no <Weather>/training/velodyne under {args.src}')
    summary = OrderedDict()
    for w in names:
        src, dst = os.path.join(args.src, w, 'training'), os.path.join(args.dst, w, 'training')
        precip, how = preset_precipitation(w)
        if args.precipitation is not None:
            precip, how = args.precipitation, '--precipitation'
        fog = args.fog_density if w == 'Fog' else 0.0
        prm = profile(precip, fog)
        print(f'\n{w}: precipitation {precip:.0f} ({how}), fog density {fog:.0f}\n  {describe(prm, args.range)}')
        linked, copied = link_tree(src, dst)
        vdir = os.path.join(dst, 'velodyne')
        os.makedirs(vdir, exist_ok=True)
        ids = sorted(f[:-4] for f in os.listdir(os.path.join(src, 'velodyne')) if f.endswith('.bin'))
        n_in = n_out = 0
        for i, fid in enumerate(ids):
            pts = np.fromfile(os.path.join(src, 'velodyne', fid + '.bin'), np.float32).reshape(-1, 4)
            rng = np.random.default_rng([args.seed, zlib.crc32(w.encode()), zlib.crc32(fid.encode())])
            out = degrade(pts, prm, rng, args.range)
            out.astype(np.float32).tofile(os.path.join(vdir, fid + '.bin'))
            n_in, n_out = n_in + len(pts), n_out + len(out)
            if (i + 1) % 200 == 0:
                print(f'  {i + 1}/{len(ids)}')
        st = dict(frames=len(ids), points_per_frame_in=round(n_in / max(len(ids), 1)),
                  points_per_frame_out=round(n_out / max(len(ids), 1)), files_linked=linked, files_copied=copied)
        print(f"  {len(ids)} frames: {st['points_per_frame_in']} -> {st['points_per_frame_out']} points per frame"
              f"{'' if prm['active'] else ' (unchanged)'}; other files: {linked} linked, {copied} copied")
        summary[w] = dict(params=prm, precipitation_source=how, effect=describe(prm, args.range),
                          range_m=args.range, seed=args.seed, **st)
        with open(os.path.join(args.dst, w, 'lidar_weather.json'), 'w') as fh:
            json.dump(summary[w], fh, indent=2)
    with open(os.path.join(args.dst, 'lidar_weather.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f'\nWritten {args.dst}. Fusion must paint the degraded points: run paint_points.py --data_root '
          f'{os.path.join(args.dst, "<Weather>", "training")} (it writes a new velodyne_paint there).')


# ------------------------------------------------------------------ self-test
def selftest():
    ok = True

    def check(cond, text):
        nonlocal ok
        ok &= bool(cond)
        print(('  ok    ' if cond else '  FAIL  ') + text)

    rng0 = np.random.default_rng(0)
    n = 200000
    u = rng0.normal(size=(n, 3))
    u[:, 2] = -np.abs(u[:, 2]) * 0.3
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    d = rng0.uniform(2.0, 60.0, n)
    pts = np.c_[u * d[:, None], np.exp(-0.004 * d)].astype(np.float32)        # CARLA-like intensity

    print('profiles (LiDAR range 60 m):')
    for name, pr, fg in (('ClearNoon', 0, 0), ('ClearNight', 0, 0), ('HardRainNoon', 100, 0),
                         ('HardRainNight', 100, 0), ('Fog (60)', 0, 60)):
        print(f'  {name:<14} {describe(profile(pr, fg))}')

    clear = profile(0, 0)
    check(not clear['active'] and degrade(pts, clear, np.random.default_rng(1)) is pts,
          'clear / night: cloud returned unchanged')

    fog = profile(0, 60)
    out, fake = degrade(pts, fog, np.random.default_rng(1), return_fake=True)
    check(out.dtype == np.float32 and out.shape[1] == 4, 'output is float32 [M, 4]')
    check(np.array_equal(out, degrade(pts, fog, np.random.default_rng(1))),
          'same seed -> same cloud (LiDAR-only and fusion get the same points)')
    do = np.linalg.norm(out[:, :3], axis=1)
    lim_dark, lim_bright = max_range(0.06), max_range(0.06, refl=1.0)
    check(abs(lim_dark - 18.9) < 0.6 and abs(lim_bright - 26.8) < 0.6,
          f'fog 60: dark surfaces lost beyond {lim_dark:.1f} m, bright beyond {lim_bright:.1f} m')
    check(do[~fake].max() <= lim_bright + 0.2, 'fog: no real point beyond the bright-surface limit')
    kept_near = (do[~fake] < 10).sum() / (d < 10).sum()
    check(0.9 < kept_near < 0.97, f'fog: points within 10 m kept, minus the backscatter share ({kept_near:.3f})')
    check(abs(fake.sum() / n - fog['clutter']) < 0.005, f"fog: {fake.sum() / n * 100:.1f}% of rays are false points "
          f"(set {fog['clutter'] * 100:.0f}%)")
    check(do[fake].min() >= 0.5 and do[fake].max() <= CLUTTER_RMAX + 1e-3,
          f'false points {do[fake].min():.1f}-{do[fake].max():.1f} m from the sensor, median {np.median(do[fake]):.1f} m')
    check(np.all(out[fake, 3] <= CLUTTER_INTENSITY[1]), 'false points are weak (intensity <= 0.2)')
    i_clear = np.exp(-0.004 * do[~fake])
    check(np.all(out[~fake, 3] <= i_clear * 1.02), 'fog: intensity of real points never increases')

    rain = profile(100, 0)
    out, fake = degrade(pts, rain, np.random.default_rng(2), return_fake=True)
    do = np.linalg.norm(out[:, :3], axis=1)
    kept_near = (do[~fake] < 20).sum() / (d < 20).sum()
    check(abs(kept_near - 0.89) < 0.01, f'heavy rain: ~10% random loss + 1% backscatter near the car (kept {kept_near:.3f})')
    lim = max_range(0.008)
    frac_far = (do[~fake] > lim + 0.5).sum() / (d > lim + 0.5).sum()
    check(0.2 < frac_far < 0.85, f'heavy rain: beyond {lim:.0f} m only brighter surfaces return ({frac_far:.2f} kept)')
    print('ALL OK' if ok else 'FAILURES')
    return ok


def main():
    ap = argparse.ArgumentParser(description='LiDAR rain / fog model on top of CARLA point clouds')
    ap.add_argument('--src', help='folder with <Weather>/training (e.g. ..\\LIDAR_only\\kitti_weather)')
    ap.add_argument('--dst', help='new folder for the degraded copy (e.g. ..\\LIDAR_only\\kitti_weather_lidarwx)')
    ap.add_argument('--only', nargs='*', help='weather folders to do (default: all)')
    ap.add_argument('--fog-density', type=float, default=60.0, help="the 'Fog' folder's density (collect_weather_tests)")
    ap.add_argument('--precipitation', type=float, default=None, help='override the preset precipitation (0-100)')
    ap.add_argument('--range', type=float, default=DEFAULT_RANGE, help='LiDAR range used when collecting, m')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if not args.src or not args.dst:
        ap.error('--src and --dst are needed (or --selftest)')
    if os.path.abspath(args.src) == os.path.abspath(args.dst):
        ap.error('--dst must be a new folder: the clean data must stay as it is')
    run_offline(args)


if __name__ == '__main__':
    main()