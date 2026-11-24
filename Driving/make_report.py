"""
make_report.py - ITS report from agent.py runs: perception and driving KPIs side by side, and charts
for the slides. Reads what every run already wrote (summary.json, events.jsonl, ticks.csv,
box_check/box_check.json); nothing is re-run, CARLA is not needed.

    python make_report.py --runs runs_weather runs_video runs_m4video
    python make_report.py --runs runs_weather --filter traffic30

Writes report/ (or --out):
    report.html          every chart + the KPI tables, open it in a browser
    its_kpis.csv         one row per run, every KPI (opens in Excel)
    its_kpis_by_setup.csv  mean per setup (scene + weather + source)
    *.png                1600 x 900 charts, ready for slides

KPI groups (ITS view):
    perception  M4 first seen / confirmed distance to the parked car, M1 missed % / false positives /
                centre and yaw error, and in every LiDAR / fusion run: how often a car really in the
                path was not perceived, how often a perceived car was not there, distance error
    safety      collisions, pedestrian hits, min TTC, min gap, TTC < 1.5 s events, emergency stops
    compliance  red lights run, lane invasions
    mobility    route completion, mean moving speed, trip time, distance
    comfort     RMS jerk, hard brakes, pedal flips, steering reversals
Box-check runs (--decide-on oracle) give perception numbers only: the car drove on ground truth.
"""
import os
import csv
import json
import glob
import math
import base64
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker                 # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))

# chart style: validated categorical slots (blue, orange, aqua), neutral ink, recessive grid
COL = {'lidar': '#2a78d6', 'fusion': '#eb6834', 'oracle': '#1baf7a'}
NAME = {'lidar': 'LiDAR only', 'fusion': 'Fusion (PointPainting)', 'oracle': 'Oracle (ground truth)'}
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
WEATHER_ORDER = ['ClearNoon', 'CloudySunset', 'HardRainNoon', 'ClearNight', 'HardRainNight', 'Fog']
SHORT = {'ClearNoon': 'Clear noon', 'CloudySunset': 'Cloudy sunset', 'HardRainNoon': 'Heavy rain',
         'ClearNight': 'Night', 'HardRainNight': 'Rain + night', 'Fog': 'Fog'}
SOURCES = ['oracle', 'lidar', 'fusion']
VEHICLE_CLS = ('car', 'truck', 'bus', 'van', 'vehicle', 'cyclist')

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF,
                     'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})


# ----------------------------------------------------------------------------- loading
def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def first_event(events, kind, field):
    for e in events:
        if e.get('type') == kind:
            return e.get(field)
    return None


def tick_perception(path):
    """closed-loop perception from ticks.csv (LiDAR / fusion runs): what the car believed vs CARLA truth,
    within 30 m ahead in its path. Pedestrians are left out (the true 'car ahead' is cars only)."""
    if not os.path.exists(path):
        return {}
    n_gt = n_miss = n_seen = n_phantom = 0
    err, err_at = [], []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for r in csv.DictReader(fh):
            if r.get('state') == 'ROUTE_COMPLETE':
                continue
            g = _f(r.get('gt_lead_gap'))
            if g is None:
                g = _f(r.get('gt_gap'))                       # parked car in M4
            p = _f(r.get('lead_gap'))
            cls = (r.get('lead_cls') or '').lower()
            if cls.startswith('pedestrian'):
                p = None
            if g is not None and g <= 30.0:
                n_gt += 1
                n_miss += p is None
            if p is not None and p <= 30.0:
                n_seen += 1
                if g is None or g > p + 5.0:
                    n_phantom += 1
                elif abs(p - g) <= 5.0:                       # same object: distance error
                    err.append(abs(p - g))
                    err_at.append(g)
    out = {}
    if n_gt >= 20:
        out['inpath_missed_pct'] = 100.0 * n_miss / n_gt
    if n_seen >= 20:
        out['phantom_pct'] = 100.0 * n_phantom / n_seen
    if err:
        out['gap_error_median_m'] = float(np.median(err))
        out['_err'] = (err, err_at)
    return out


def load_run(d):
    with open(os.path.join(d, 'summary.json'), encoding='utf-8', errors='replace') as fh:
        s = json.load(fh)
    m = s.get('meta', {})
    events = []
    ep = os.path.join(d, 'events.jsonl')
    if os.path.exists(ep):
        with open(ep, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    src = m.get('obstacle_source') or '?'
    gt_driven = src not in ('oracle', 'none') and m.get('decide_on') == 'oracle'
    weather = ((m.get('weather') or 'ClearNoon') + ('*' if m.get('weather_set') else '')
               + ('~' if m.get('lidar_weather') else ''))        # ~ = agent.py --lidar-weather auto
    scen = s.get('scenario', m.get('scenario'))
    scene = 'parked car (M4)' if scen == 'static' else (
        f"{m.get('traffic', 0)} cars" + (f", {m.get('walkers')} pedestrians" if m.get('walkers') else '')
        + (', red lights obeyed' if (m.get('lights') or 'off') != 'off' else ''))
    scene_tag = 'm4' if scen == 'static' else (f"traffic{m.get('traffic', 0)}"
                                                + (f"_ped{m.get('walkers')}" if m.get('walkers') else '')
                                                + ('_lights' if (m.get('lights') or 'off') != 'off' else ''))
    hz, mvm, st = m.get('hazard'), m.get('manoeuvres') or {}, m.get('scenario_traffic') or {}
    if scen == 'static' and hz:                  # agent.py --hazard: not the parked car
        scene = f"{hz.get('label', 'hazard')}"[:60]
        scene_tag = 'hazard'
    if scen == 'static' and mvm.get('steer_around'):
        scene += ', steer around' + (' (M6)' if not hz else '') + (f", {st['oncoming']} oncoming" if st.get('oncoming')
                                                                   else '')
        scene_tag = 'm6'
    elif scen == 'overtake':
        scene = 'slow car ahead, lane change' + (' on' if mvm.get('lane_change') else ' OFF') + ' (M7)'
        scene_tag = 'm7'
    elif scen == 'junction':
        scene = (f"junction, {st.get('turn')} turn, {st.get('oncoming', 0)} oncoming / {st.get('crossing', 0)} crossing, "
                 f"yield {'on' if mvm.get('yield_at_junctions') else 'OFF'} (M8)")
        scene_tag = 'm8'
    km = (s.get('route_length_m') or 0.0) * (s.get('route_completion_pct') or 0.0) / 100.0 / 1000.0
    coll = s.get('collisions') or []
    r = OrderedDict(
        run=os.path.basename(d), folder=os.path.basename(os.path.dirname(d)), scenario=scen, scene=scene,
        _scene_tag=scene_tag,
        weather=weather, source=src, timing=m.get('timing') or 'lockstep', seed=m.get('seed'),
        drove_on='ground truth (box check)' if gt_driven else ('ground truth' if src == 'oracle' else 'model'),
        passed=bool(s.get('passed')), end=s.get('end_reason'),
        # perception
        m4_first_seen_m=_f(first_event(events, 'obstacle_in_corridor', 'gt_gap')),
        m4_confirmed_m=_f(first_event(events, 'obstacle_confirmed', 'gap')) if scen == 'static' else None,
        _confirm_k=first_event(events, 'obstacle_confirmed', 'k'),
        # safety
        collisions=len(coll),
        ego_caused_collisions=sum(1 for c in coll if c.get('ego_caused')),
        pedestrian_hits=s.get('pedestrian_hits_by_ego', sum(1 for c in coll if c.get('category') == 'pedestrian'
                                                           and c.get('ego_caused'))),
        min_ttc_s=_f(s.get('min_gt_ttc_outside_junctions_s')),
        ttc_below_1_5s=s.get('ttc_below_1_5s_events_outside_junctions'),
        min_gap_m=_f(s.get('min_gt_lead_gap_m')) if scen != 'static' else _f(s.get('final_gt_gap_m')),
        emergency_stops=s.get('emergency_stops_outside_junctions'),
        emergency_stops_pedestrians=s.get('emergency_stops_for_pedestrians'),
        # compliance
        red_lights_run=s.get('red_lights_run'),
        lane_invasions=s.get('lane_invasion_events'),
        # mobility
        route_completion_pct=_f(s.get('route_completion_pct')),
        mean_speed_kmh=_f(s.get('mean_moving_speed_kmh')),
        trip_time_s=_f(s.get('sim_seconds')),
        distance_km=round(km, 3),
        collisions_per_km=(len(coll) / km) if km > 0.05 else None,
        # comfort
        rms_jerk=_f(s.get('rms_jerk')),
        hard_brakes=s.get('hard_brake_events'),
        pedal_flips=s.get('pedal_back_and_forth'),
        steer_reversals_per_min=_f(s.get('steer_reversals_per_min')),
    )
    bp = os.path.join(d, 'box_check', 'box_check.json')
    if os.path.exists(bp):
        with open(bp, encoding='utf-8', errors='replace') as fh:
            b = json.load(fh).get('summary', {})
        gt, miss = b.get('gt_within_30m'), b.get('gt_within_30m_missed')
        r.update(m1_result=(b.get('status') or '?').split(' (')[0],
                 m1_missed_pct=(100.0 * miss / gt) if gt and miss is not None else None,
                 m1_false_pos=b.get('false_positives'), m1_centre_err_m=_f(b.get('median_center_err_m')),
                 m1_yaw_err_deg=_f(b.get('median_yaw_err_deg_vehicles')))
    tp = tick_perception(os.path.join(d, 'ticks.csv')) if src in ('lidar', 'fusion') else {}
    r['inpath_missed_pct'] = tp.get('inpath_missed_pct')
    r['phantom_pct'] = tp.get('phantom_pct')
    r['gap_error_median_m'] = tp.get('gap_error_median_m')
    r['_err'] = tp.get('_err')
    r['_dir'] = d
    return r


# ----------------------------------------------------------------------------- helpers
def worder(ws):
    return sorted(set(ws), key=lambda w: (WEATHER_ORDER.index(wbase(w)) if wbase(w) in WEATHER_ORDER
                                          else 99, '~' in w, w))


def wbase(w):
    """preset name without the markers: * = --weather-set, ~ = LiDAR weather model on"""
    return w.rstrip('*~')


def wlabel(w):
    return (SHORT.get(wbase(w), wbase(w)) + (' (custom)' if '*' in w else '')
            + (' + LiDAR wx' if '~' in w else ''))


def stats(vals):
    v = [x for x in vals if isinstance(x, (int, float)) and x is not None and math.isfinite(x)]
    if not v:
        return None
    return float(np.mean(v)), float(np.min(v)), float(np.max(v)), len(v)


def grouped_bars(ax, rows, key, title, ylabel, line=None, line_label=None, fmt='{:.1f}', sources=SOURCES):
    """bars: weather on x, one bar per source (mean, whiskers min-max if several runs)"""
    ws = worder(r['weather'] for r in rows)
    srcs = [s for s in sources if any(r['source'] == s and r.get(key) is not None for r in rows)]
    if not ws or not srcs:
        ax.set_visible(False)
        return False
    width = 0.8 / len(srcs)
    for j, s in enumerate(srcs):
        labelled = False
        for i, w in enumerate(ws):
            st = stats([r.get(key) for r in rows if r['source'] == s and r['weather'] == w])
            if st is None:
                continue
            x = i - 0.4 + width * (j + 0.5)
            ax.bar(x, st[0], width, color=COL[s], edgecolor=SURF, linewidth=2,
                   label=None if labelled else NAME[s])
            labelled = True
            if st[3] > 1 and st[2] > st[1]:
                ax.plot([x, x], [st[1], st[2]], color=INK2, lw=1.5)
            ax.text(x, st[0], fmt.format(st[0]), ha='center', va='bottom', fontsize=11, color=INK2)
    if line is not None:
        ax.axhline(line, color=INK2, lw=1.2, ls='--')
        title = f'{title} (dashed: {line_label})'
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([wlabel(w).replace(' + LiDAR wx', '\n+ LiDAR wx') for w in ws], rotation=0)
    ax.set_title(title, loc='left', color=INK, fontsize=15)
    if fmt == '{:.0f}':
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, max(top, 1.0, (line or 0) * 1.3))      # bars start at zero; all-zero panels stay readable
    return True


def figure(rows, panels, title, path, note=''):
    """small multiples: one panel per metric (never two y-scales on one chart)"""
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 9))
    axes = np.atleast_1d(axes)
    shown = [grouped_bars(ax, rows, *p) for ax, p in zip(axes, panels)]
    if not any(shown):
        plt.close(fig)
        return None
    handles, labels = [], []
    for ax in axes:
        for h, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(h)
                labels.append(lb)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.015, 0.935), ncol=len(labels), fontsize=13)
    fig.suptitle(title, x=0.02, y=0.985, ha='left', fontsize=20, color=INK)
    if note:
        fig.text(0.02, 0.015, note, fontsize=11, color=INK2)
    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(20 if len(ax.get_xticklabels()) > 3 else 0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.87))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def scatter(ax, rows, xk, yk, title, xlabel, ylabel):
    pts = [(r[xk], r[yk], r) for r in rows if r.get(xk) is not None and r.get(yk) is not None]
    if len(pts) < 3:
        ax.set_visible(False)
        return False
    marker = {'oracle': 'D', 'lidar': 'o', 'fusion': 's'}        # shape as well as colour
    for s in SOURCES:
        sp = [p for p in pts if p[2]['source'] == s]
        if not sp:
            continue
        ax.scatter([p[0] for p in sp], [p[1] for p in sp], s=110 if s == 'lidar' else 80, marker=marker[s],
                   color=COL[s], edgecolor=SURF, linewidth=2, label=NAME[s], zorder=3)
        same = OrderedDict()                                     # one label for runs on the same spot
        for x, y, r in sp:
            same.setdefault((round(x, 2), round(y, 2)), []).append(wlabel(r['weather']))
        n_w = len(set(p[2]['weather'] for p in pts))
        for j, ((x, y), ws) in enumerate(same.items()):
            ax.annotate('all weathers' if len(ws) > 2 and len(ws) == n_w else ', '.join(ws), (x, y), xytext=(7, 5 if s != 'fusion' else -14), textcoords='offset points',
                        fontsize=10, color=INK2)
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    if np.ptp(x) > 0 and np.ptp(y) > 0:
        rr = float(np.corrcoef(x, y)[0, 1])
        k, c = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, k * xs + c, color=INK2, lw=1.2, ls='--', zorder=2)
        title = f'{title}\nr = {rr:+.2f}, n = {len(pts)} runs'
    ax.set_title(title, loc='left', fontsize=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return True


def m4_timeline(runs, weather, path):
    """M4, one weather: true vs perceived distance to the parked car, and speed, LiDAR vs fusion"""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    drawn = False
    for r in runs:
        tp = os.path.join(r['_dir'], 'ticks.csv')
        if not os.path.exists(tp):
            continue
        with open(tp, encoding='utf-8', errors='replace') as fh:
            rows = list(csv.DictReader(fh))
        t = np.array([_f(x.get('t')) or 0.0 for x in rows])
        gt = np.array([_f(x.get('gt_gap')) if _f(x.get('gt_gap')) is not None else np.nan for x in rows])
        pg = np.array([_f(x.get('lead_gap')) if _f(x.get('lead_gap')) is not None else np.nan for x in rows])
        v = np.array([(_f(x.get('speed')) or 0.0) * 3.6 for x in rows])
        if not drawn:
            a1.plot(t, gt, color=INK2, lw=2, label='true distance (CARLA)')
        a1.plot(t, pg, color=COL[r['source']], lw=2.5, label=f"perceived by {NAME[r['source']]}")
        ks = [int(_f(x.get('k')) or -1) for x in rows]
        if r.get('m4_confirmed_m') is not None and r.get('_confirm_k') in ks:
            tc = t[ks.index(r['_confirm_k'])]
            a1.axvline(tc, color=COL[r['source']], lw=1.5, ls=':')
            a1.annotate(f"{NAME[r['source']]} confirms at {r['m4_confirmed_m']:.1f} m", (tc, 0.92),
                        xycoords=('data', 'axes fraction'), xytext=(6, -14 * (r['source'] == 'fusion')),
                        textcoords='offset points', color=INK2, fontsize=11)
        a2.plot(t, v, color=COL[r['source']], lw=2.5, label=NAME[r['source']])
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    a1.set_ylabel('distance to parked car (m)')
    a2.set_ylabel('speed (km/h)')
    a2.set_xlabel('time (s)')
    for a in (a1, a2):
        a.grid(color=GRID, lw=0.8)
        a.legend(loc='upper right', fontsize=12)
    fig.suptitle(f'Parked-car stop (M4), {wlabel(weather)}: when each system sees the car, and how it slows',
                 x=0.02, ha='left', fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def error_vs_range(runs, path):
    """perception accuracy vs distance: |perceived - true| gap to the car ahead, binned by true distance"""
    bins = [0, 10, 20, 30]
    fig, ax = plt.subplots(figsize=(16, 9))
    drawn = False
    for s in ('lidar', 'fusion'):
        e, g = [], []
        for r in runs:
            if r['source'] == s and r.get('_err'):
                e += r['_err'][0]
                g += r['_err'][1]
        if len(e) < 30:
            continue
        e, g = np.array(e), np.array(g)
        xs, med, p90 = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (g >= lo) & (g < hi)
            if m.sum() >= 10:
                xs.append((lo + hi) / 2)
                med.append(np.median(e[m]))
                p90.append(np.percentile(e[m], 90))
        if not xs:
            continue
        ax.plot(xs, med, '-o', color=COL[s], lw=2.5, ms=9, label=f'{NAME[s]}: median')
        ax.plot(xs, p90, '--', color=COL[s], lw=1.5, label=f'{NAME[s]}: 90th percentile')
        drawn = True
    if not drawn:
        plt.close(fig)
        return None
    ax.set_xticks([5, 15, 25])
    ax.set_xticklabels(['0-10 m', '10-20 m', '20-30 m'])
    ax.set_xlabel('true distance to the car ahead')
    ax.set_ylabel('distance error (m)')
    ax.set_title('Closed-loop perception accuracy: error of the perceived gap to the car ahead, by range',
                 loc='left', fontsize=18)
    ax.grid(color=GRID, lw=0.8)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- report
KPI_TABLE = [  # (group, column, label, format)
    ('Perception', 'm4_confirmed_m', 'M4 parked car confirmed at (m)', '{:.1f}'),
    ('Perception', 'm1_missed_pct', 'M1 cars missed within 30 m (%)', '{:.1f}'),
    ('Perception', 'm1_false_pos', 'M1 false positives', '{:.0f}'),
    ('Perception', 'inpath_missed_pct', 'car in path not perceived (% of time)', '{:.1f}'),
    ('Perception', 'phantom_pct', 'perceived car not there (% of time)', '{:.1f}'),
    ('Perception', 'gap_error_median_m', 'gap error, median (m)', '{:.2f}'),
    ('Safety', 'collisions', 'collisions', '{:.1f}'),
    ('Safety', 'pedestrian_hits', 'pedestrians hit', '{:.1f}'),
    ('Safety', 'min_ttc_s', 'min time to collision (s)', '{:.2f}'),
    ('Safety', 'min_gap_m', 'min gap / M4 final gap (m)', '{:.2f}'),
    ('Safety', 'emergency_stops', 'emergency stops (car)', '{:.1f}'),
    ('Safety', 'emergency_stops_pedestrians', 'emergency stops (pedestrian)', '{:.1f}'),
    ('Compliance', 'red_lights_run', 'red lights run', '{:.1f}'),
    ('Mobility', 'route_completion_pct', 'route completed (%)', '{:.0f}'),
    ('Mobility', 'mean_speed_kmh', 'mean moving speed (km/h)', '{:.1f}'),
    ('Mobility', 'trip_time_s', 'trip time (s)', '{:.0f}'),
    ('Comfort', 'rms_jerk', 'RMS jerk (m/s^3)', '{:.2f}'),
    ('Comfort', 'hard_brakes', 'hard brakes', '{:.1f}'),
    ('Comfort', 'pedal_flips', 'pedal flips', '{:.1f}'),
]


def setup_rows(runs):
    groups = OrderedDict()
    for r in sorted(runs, key=lambda r: (r['scene'], WEATHER_ORDER.index(wbase(r['weather']))
                                         if wbase(r['weather']) in WEATHER_ORDER else 99,
                                         SOURCES.index(r['source']) if r['source'] in SOURCES else 9)):
        groups.setdefault((r['scene'], r['weather'], r['source'], r['timing'], r['drove_on']), []).append(r)
    out = []
    for (scene, w, s, tm, dr), rs in groups.items():
        row = OrderedDict(scene=scene, weather=wlabel(w), source=NAME.get(s, s), timing=tm, drove_on=dr,
                          runs=len(rs), passed=f"{sum(r['passed'] for r in rs)}/{len(rs)}")
        for _, col, _, _ in KPI_TABLE:
            st = stats([r.get(col) for r in rs])
            row[col] = None if st is None else round(st[0], 3)
        out.append(row)
    return out


def html_table(rows):
    cols = ['scene', 'weather', 'source', 'drove_on', 'passed'] + [c for _, c, _, _ in KPI_TABLE
                                                                    if any(r.get(c) is not None for r in rows)]
    lab = {c: lb for _, c, lb, _ in KPI_TABLE}
    grp = {c: g for g, c, _, _ in KPI_TABLE}
    fmt = {c: f for _, c, _, f in KPI_TABLE}
    head1 = ''.join(f'<th class="g">{grp.get(c, "")}</th>' for c in cols)
    head2 = ''.join(f'<th>{lab.get(c, c.replace("_", " "))}</th>' for c in cols)
    body = ''
    for r in rows:
        cells = ''
        for c in cols:
            v = r.get(c)
            cells += f'<td>{"-" if v is None else (fmt[c].format(v) if c in fmt else v)}</td>'
        body += f'<tr>{cells}</tr>'
    return f'<table><thead><tr>{head1}</tr><tr>{head2}</tr></thead><tbody>{body}</tbody></table>'


def write_html(path, charts, table_html, n_runs, folders):
    imgs = ''
    for title, p, text in charts:
        with open(p, 'rb') as fh:
            b64 = base64.b64encode(fh.read()).decode()
        imgs += (f'<section><h2>{title}</h2><p>{text}</p>'
                 f'<img src="data:image/png;base64,{b64}" alt="{title}"></section>')
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ITS Run Report</title>
<style>
:root{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37}}}}
:root[data-theme="dark"]{{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#3a3a37}}
body{{margin:0;padding:24px 16px;background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif}}
main{{max-width:1240px;margin:auto}} h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:32px 0 6px}}
p{{color:var(--ink2);margin:0 0 10px}} img{{width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fcfcfb}}
.wrap{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;min-width:900px}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}}
th{{color:var(--ink2);font-weight:600}} th.g{{font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
td:nth-child(-n+5),th:nth-child(-n+5){{text-align:left}}
</style></head><body><main>
<h1>ITS run report</h1><p>{n_runs} runs from {', '.join(folders)}. Means per setup; charts show the mean and,
when a setup has several runs, the min-max range. Oracle = CARLA ground truth (reference, not a perception
result). Box-check runs (--decide-on oracle) give perception numbers only.</p>
{imgs}<section><h2>KPIs per setup (ITS view)</h2><div class="wrap">{table_html}</div></section>
</main></body></html>"""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=[os.path.join(HERE, 'runs')], help='one or more run folders')
    ap.add_argument('--filter', default='', help='only runs whose folder name contains this')
    ap.add_argument('--out', default=os.path.join(HERE, 'report'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = []
    for folder in args.runs:
        for p in sorted(glob.glob(os.path.join(folder, '*', 'summary.json'))):
            d = os.path.dirname(p)
            if args.filter and args.filter not in os.path.basename(d):
                continue
            try:
                runs.append(load_run(d))
            except (OSError, ValueError, KeyError) as e:
                print(f'skipped {os.path.basename(d)}: {e}')
    if not runs:
        raise SystemExit(f'no runs found in {args.runs}')
    print(f'{len(runs)} runs loaded')

    # per-run and per-setup KPI tables
    cols = [c for c in runs[0] if not c.startswith('_')]
    allc = cols + [c for r in runs for c in r if not c.startswith('_') and c not in cols]
    allc = list(OrderedDict.fromkeys(allc))
    with open(os.path.join(args.out, 'its_kpis.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=allc, extrasaction='ignore')
        w.writeheader()
        w.writerows(runs)
    srows = setup_rows(runs)
    with open(os.path.join(args.out, 'its_kpis_by_setup.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)

    charts = []
    model = [r for r in runs if r['drove_on'] != 'ground truth (box check)']
    static = [r for r in model if r['scenario'] == 'static' and r['source'] in SOURCES]
    boxes = [r for r in runs if r.get('m1_missed_pct') is not None]

    # 1. perception by weather
    prow = static + boxes
    p = figure(prow, [('m4_confirmed_m', 'Parked car confirmed at', 'distance (m), higher is better'),
                      ('m1_missed_pct', 'Cars missed within 30 m', '% of ground-truth cars'),
                      ('m1_false_pos', 'False positives', 'boxes with no real object', None, None, '{:.0f}')],
               'Perception in seen and unseen weather: LiDAR only vs fusion',
               os.path.join(args.out, '1_perception_by_weather.png'),
               'M4 = parked-car run; M1 = box check (car driven on ground truth, model scored against CARLA).')
    if p:
        wx = any('~' in r['weather'] for r in prow)
        charts.append(('Perception by weather', p, 'How early each system confirms the parked car, how many '
                       'real cars it misses and how many phantom boxes it makes. CARLA LiDAR does not see '
                       'weather, so without "+ LiDAR wx" changes come from the camera part of fusion'
                       + ('; "+ LiDAR wx" runs add our rain / fog model to the LiDAR points (lidar_weather.py), '
                          'the same points for both systems.' if wx else '.')))

    # 2-4. driving KPIs per traffic scene
    route = [r for r in model if r['scenario'] == 'route' and r['source'] in SOURCES]
    for scene in sorted(set(r['scene'] for r in route)):
        rs = [r for r in route if r['scene'] == scene]
        tag = rs[0]['_scene_tag']
        sets = [
            ('Safety', [('min_ttc_s', 'Min time to collision', 'seconds', 1.5, 'limit 1.5 s', '{:.2f}'),
                        ('min_gap_m', 'Min gap to the car ahead', 'metres', 2.0, 'limit 2 m', '{:.2f}'),
                        ('collisions', 'Collisions', 'count', None, None, '{:.0f}')]),
            ('Comfort', [('pedal_flips', 'Pedal flips', 'throttle/brake switches', None, None, '{:.0f}'),
                         ('hard_brakes', 'Hard brakes', 'count', None, None, '{:.0f}'),
                         ('rms_jerk', 'RMS jerk', 'm/s^3', None, None, '{:.2f}')]),
            ('Mobility', [('route_completion_pct', 'Route completed', '%', None, None, '{:.0f}'),
                          ('mean_speed_kmh', 'Mean moving speed', 'km/h', None, None, '{:.1f}'),
                          ('trip_time_s', 'Trip time', 'seconds', None, None, '{:.0f}')]),
        ]
        if any(r.get('pedestrian_hits') or r.get('red_lights_run') or r.get('emergency_stops_pedestrians')
               for r in rs):
            sets.insert(1, ('Vulnerable road users and compliance',
                            [('pedestrian_hits', 'Pedestrians hit', 'count', None, None, '{:.0f}'),
                             ('emergency_stops_pedestrians', 'Emergency stops for pedestrians', 'count',
                              None, None, '{:.0f}'),
                             ('red_lights_run', 'Red lights run', 'count', None, None, '{:.0f}')]))
        for i, (name, panels) in enumerate(sets):
            p = figure(rs, panels, f'{name} by weather: {scene}',
                       os.path.join(args.out, f'2_{tag}_{i + 1}_{name.split()[0].lower()}.png'))
            if p:
                charts.append((f'{name}: {scene}', p, ''))

    # 5. perception vs driving: only the pairs that have data (>= 3 runs)
    det_route = [r for r in route if r['source'] != 'oracle']
    pairs = [(static, 'm4_confirmed_m', 'rms_jerk', 'Later detection, harsher stop?',
              'parked car confirmed at (m)', 'RMS jerk (m/s^3)'),
             (det_route, 'inpath_missed_pct', 'min_ttc_s', 'Missed cars vs safety margin',
              'car in path not perceived (% of time)', 'min TTC (s)'),
             (det_route, 'phantom_pct', 'pedal_flips', 'Phantom detections vs comfort',
              'perceived car not there (% of time)', 'pedal flips')]
    pairs = [pp for pp in pairs if sum(1 for r in pp[0] if r.get(pp[1]) is not None and r.get(pp[2]) is not None) >= 3]
    fig, axes = plt.subplots(1, max(len(pairs), 1), figsize=(16 if len(pairs) > 1 else 12, 9))
    axes = np.atleast_1d(axes)
    ok = [scatter(ax, *pp) for ax, pp in zip(axes, pairs)]
    if any(ok):
        handles, labels = [], []
        for ax in axes:
            for h, lb in zip(*ax.get_legend_handles_labels()):
                if lb not in labels:
                    handles.append(h)
                    labels.append(lb)
        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.015, 0.935), ncol=len(labels), fontsize=13)
        fig.suptitle('Does perception quality show up in driving? (one point per run)', x=0.02, y=0.985,
                     ha='left', fontsize=20)
        fig.text(0.02, 0.015, 'Dashed line = least-squares fit, r = Pearson correlation. Few runs: read as a '
                              'trend, not proof.', fontsize=11, color=INK2)
        fig.tight_layout(rect=(0, 0.04, 1, 0.87))
        p = os.path.join(args.out, '3_perception_vs_driving.png')
        fig.savefig(p, dpi=100)
        charts.append(('Perception vs driving', p, 'Each point is one run. These link a perception KPI to a '
                       'driving KPI, the ITS question: does a perception drop make the car less safe or less '
                       'smooth?'))
    plt.close(fig)

    # 6. M4 timelines, LiDAR vs fusion per weather
    for w in worder(r['weather'] for r in static):
        pair = []
        for s in ('lidar', 'fusion'):
            cand = sorted([r for r in static if r['weather'] == w and r['source'] == s], key=lambda r: r['run'])
            if cand:
                pair.append(cand[-1])
        if len(pair) == 2:
            tag = (''.join(ch for ch in w if ch.isalnum()) + ('_custom' if '*' in w else '')
                   + ('_lidarwx' if '~' in w else ''))
            p = m4_timeline(pair, w, os.path.join(args.out, f'4_m4_timeline_{tag}.png'))
            if p:
                charts.append((f'M4 timeline: {wlabel(w)}', p, 'Grey = true distance; coloured = what each '
                               'system believed (gaps = nothing perceived). Dotted line = obstacle confirmed.'))

    # 7. accuracy vs range
    p = error_vs_range([r for r in model if r['source'] in ('lidar', 'fusion')],
                       os.path.join(args.out, '5_gap_error_vs_range.png'))
    if p:
        charts.append(('Perception accuracy vs range', p, 'Error of the perceived distance to the car ahead '
                       '(same object), all LiDAR / fusion runs.'))

    write_html(os.path.join(args.out, 'report.html'), charts, html_table(srows), len(runs),
               sorted(set(r['folder'] for r in runs)))
    print(f'{len(charts)} charts, KPI tables and report.html in {args.out}')
    for _, p, _ in charts:
        print('  ', os.path.basename(p))


if __name__ == '__main__':
    main()