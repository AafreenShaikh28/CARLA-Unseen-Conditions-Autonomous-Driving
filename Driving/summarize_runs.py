"""
summarize_runs.py - one table of every agent.py run, for the report. Reads runs\\*\\summary.json (and
events.jsonl for the M4 detection distances); nothing is re-run.

    python summarize_runs.py                       # every run in runs\\
    python summarize_runs.py --filter traffic30    # only runs whose folder name contains this
    python summarize_runs.py --since 20261110      # only runs from this date on (folder timestamp)

Prints
  1. one line per run (newest last): scenario, obstacle source, timing, seed, PASS/FAIL, key numbers
  2. per setup (scenario + source + timing): passes / runs, and the mean (worst) of the key numbers
and writes the per-run table to runs\\runs_table.csv (opens in Excel).
"""
import os
import csv
import json
import glob
import argparse
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))


def first_event(path, kind, field):
    """value of `field` in the first event of this kind in events.jsonl (or None)"""
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get('type') == kind:
                return e.get(field)
    return None


def collision_text(colls):
    """'0', or e.g. '1: vehicle.audi.a2 (ego drove into it)'"""
    if not colls:
        return '0'
    parts = [f"{c.get('other_type', '?')} ({c.get('fault', '?')}{', junction' if c.get('in_junction') else ''})"
             for c in colls]
    return f'{len(colls)}: ' + '; '.join(parts)


def manoeuvre_flags(m):
    """'sa' steer around (+p: people), 'lc' lane change, 'y' yield; '-' none"""
    mv = m.get('manoeuvres') or {}
    f = [('sa' + ('p' if mv.get('people') else '')) if mv.get('steer_around') else '',
         'lc' if mv.get('lane_change') else '', 'y' if mv.get('yield_at_junctions') else '']
    return '+'.join(x for x in f if x) or '-'


def load(run_dir):
    with open(os.path.join(run_dir, 'summary.json'), encoding='utf-8', errors='replace') as f:
        s = json.load(f)
    m = s.get('meta', {})
    ev = os.path.join(run_dir, 'events.jsonl')
    ms = s.get('manoeuvre_stats') or {}
    st = m.get('scenario_traffic') or {}
    row = OrderedDict(
        run=os.path.basename(run_dir),
        scenario=s.get('scenario', m.get('scenario')),
        source=(m.get('obstacle_source') or '') + ('+occ' if m.get('occupancy') else ''),   # --occupancy
        timing=m.get('timing') or '-',
        weather=(m.get('weather') or 'ClearNoon') + ('+set' if m.get('weather_set') else '')
        + ('+lidarwx' if m.get('lidar_weather') else ''),          # agent.py --lidar-weather auto
        hazard=f"{m['hazard'].get('pose')}@{m['hazard'].get('place')}" if m.get('hazard') else '-',
        man=manoeuvre_flags(m),             # agent.py --steer-around / --lane-change / --yield
        seed=m.get('seed'),
        # M1 box-check runs: the model runs and is scored, but the car drives on ground truth
        decides='GT' if (m.get('obstacle_source') not in ('oracle', 'none')
                         and m.get('decide_on') == 'oracle') else '-',
        result='PASS' if s.get('passed') else 'FAIL',
        end=s.get('end_reason'),
        route_pct=s.get('route_completion_pct'),
        sim_s=s.get('sim_seconds'),
        collisions=collision_text(s.get('collisions')),
        # M4 (static obstacle)
        first_seen_m=first_event(ev, 'obstacle_in_corridor', 'gt_gap'),
        confirmed_m=first_event(ev, 'obstacle_confirmed', 'gap'),
        first_brake_m=first_event(ev, 'first_brake', 'gt_gap'),
        final_gap_m=s.get('final_gt_gap_m'),
        # --hazard (posed person): what the car should do, and what perception made of it
        should=s.get('hazard_expected'),
        hz_seen_m=s.get('hazard_first_detected_m'),
        hz_seen_as=s.get('hazard_first_cls'),
        hz_seen_30m_pct=s.get('hazard_detected_pct_within_30m'),
        drove_onto=s.get('hazard_run_over'),
        occ_seen_m=s.get('hazard_occupancy_first_m'),
        # step 3 (traffic)
        follow_s=s.get('lead_following_s'),
        min_gap_m=s.get('min_gt_lead_gap_m'),
        min_ttc_s=s.get('min_gt_ttc_outside_junctions_s'),
        emerg_stops=s.get('emergency_stops_outside_junctions'),
        ego_coll=s.get('ego_caused_collisions_outside_junctions'),
        # manoeuvres (M6 steer around, M7 lane change, M8 junction)
        others=' '.join(f'{k}{st[k]}' for k in ('oncoming', 'crossing', 'overtake') if st.get(k))
        + (f" {st['turn']}" if st.get('turn') else '') or '-',
        steered=ms.get('steer_arounds'), lane_chg=ms.get('lane_changes'), aborts=ms.get('aborts'),
        yields=ms.get('yields'), yield_s=ms.get('yield_s'), gap_wait_s=ms.get('waited_for_gap_s'),
        obst_clear_m=s.get('min_obstacle_clearance_m'), car_clear_m=s.get('min_vehicle_clearance_m'),
        pet_s=s.get('pet_min_s'), conflicts=s.get('conflicts_within_5s'), passed_lead=s.get('lead_passed'),
        emergency=s.get('emergency_stops'),
        # comfort
        steer_rev_min=s.get('steer_reversals_per_min'),
        pedal_flips=s.get('pedal_back_and_forth'),
        hard_brakes=s.get('hard_brake_events'),
        rms_jerk=s.get('rms_jerk'),
    )
    return row


def load_box(run_dir, row):
    """M1 box-check numbers (runs made with --check-boxes), or None"""
    p = os.path.join(run_dir, 'box_check', 'box_check.json')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8', errors='replace') as f:
        b = json.load(f).get('summary', {})
    gt, missed = b.get('gt_within_30m'), b.get('gt_within_30m_missed')
    return OrderedDict(
        weather=row['weather'], source=row['source'], timing=row['timing'], seed=row['seed'],
        result=(b.get('status') or '?').split(' (')[0],
        observations=b.get('observations'), actors=b.get('distinct_actors'),
        centre_err_m=b.get('median_center_err_m'), p90_centre_m=b.get('p90_center_err_m'),
        yaw_err_deg=b.get('median_yaw_err_deg_vehicles'), len_err_m=b.get('median_length_err_m'),
        false_pos=b.get('false_positives'),
        fp_classes=' '.join(f'{k}:{v}' for k, v in (b.get('false_positive_classes') or {}).items()) or '-',
        gt_30m=gt, missed_30m=missed,
        missed_pct=(100.0 * missed / gt) if gt and missed is not None else None,
        run=row['run'])


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.2f}'
    return str(v)


def print_table(rows, cols):
    w = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    print('  '.join(c.ljust(w[c]) for c in cols))
    print('  '.join('-' * w[c] for c in cols))
    for r in rows:
        print('  '.join(fmt(r.get(c)).ljust(w[c]) for c in cols))


# per-setup aggregate: (column, how to aggregate, 'worst' direction)
AGG = [('first_seen_m', 'mean', 'min'), ('first_brake_m', 'mean', 'min'), ('final_gap_m', 'mean', 'min'),
       ('obst_clear_m', 'mean', 'min'), ('car_clear_m', 'mean', 'min'), ('pet_s', 'mean', 'min'),
       ('min_gap_m', 'mean', 'min'), ('min_ttc_s', 'mean', 'min'), ('follow_s', 'mean', 'min'),
       ('steer_rev_min', 'mean', 'max'), ('pedal_flips', 'mean', 'max'), ('hard_brakes', 'mean', 'max'),
       ('rms_jerk', 'mean', 'max')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--filter', default='', help='only run folders whose name contains this')
    ap.add_argument('--since', default='', help='only runs with folder timestamp >= this, e.g. 20261110')
    ap.add_argument('--csv', default='', help='default: <runs>/runs_table.csv')
    args = ap.parse_args()

    rows, boxes = [], []
    for p in glob.glob(os.path.join(args.runs, '*', 'summary.json')):
        d = os.path.dirname(p)
        name = os.path.basename(d)
        if args.filter and args.filter not in name:
            continue
        stamp = name[-15:-7] if len(name) > 15 else ''
        if args.since and stamp < args.since.replace('-', ''):
            continue
        try:
            rows.append(load(d))
        except (OSError, ValueError) as e:
            print(f'skipped {name}: {e}')
            continue
        try:
            b = load_box(d, rows[-1])
            if b:
                boxes.append(b)
        except (OSError, ValueError) as e:
            print(f'skipped box check of {name}: {e}')
    if not rows:
        raise SystemExit(f'no runs found in {args.runs}' + (f' matching {args.filter!r}' if args.filter else ''))
    rows.sort(key=lambda r: (r['scenario'] or '', r['weather'], r['source'] or '', r['timing'], r['decides'],
                             r['run'][-15:]))

    manoeuvre = [r for r in rows if r['scenario'] in ('overtake', 'junction')
                 or (r['scenario'] == 'static' and r['hazard'] == '-' and r['man'].startswith('sa'))]
    static = [r for r in rows if r['scenario'] == 'static' and r['hazard'] == '-' and r not in manoeuvre]
    hazard = [r for r in rows if r['scenario'] == 'static' and r['hazard'] != '-']
    route = [r for r in rows if r['scenario'] == 'route']
    if static:
        print('\nM4 PARKED-CAR STOP (distances are real gaps to the parked car, from CARLA ground truth)')
        print_table(static, ['weather', 'source', 'timing', 'decides', 'seed', 'result', 'end', 'first_seen_m', 'confirmed_m',
                             'first_brake_m', 'final_gap_m', 'collisions', 'hard_brakes', 'rms_jerk', 'run'])
        if any(r['source'] == 'none' for r in static):
            print('(source none = causality check: PASS means it hit the parked car, as it should without perception)')
    if hazard:
        hazard.sort(key=lambda r: (r['hazard'], r['weather'], r['source'] or '', r['run'][-15:]))
        print('\nHAZARD: PERSON (lying / sitting / crouching) OR CATTLE; hz_seen_m = GT distance when the detector first '
              'had a box on it, hz_seen_as = its class')
        print_table(hazard, ['hazard', 'weather', 'source', 'timing', 'decides', 'should', 'result', 'end', 'hz_seen_m',
                             'hz_seen_as', 'hz_seen_30m_pct'] + (['occ_seen_m'] if any(
                                 r['source'].endswith('+occ') for r in hazard) else [])
                    + ['first_seen_m', 'confirmed_m', 'final_gap_m', 'drove_onto']
                    + (['man', 'steered', 'obst_clear_m'] if any(r['man'] != '-' for r in hazard) else [])
                    + ['collisions', 'run'])
        if any(r['source'].endswith('+occ') for r in hazard):
            print("(+occ = --occupancy safety layer on; hz_seen_* = the DETECTOR's boxes only, occ_seen_m = GT distance "
                  "when the occupancy check first put an 'unknown' obstacle on the person)")
    if manoeuvre:
        manoeuvre.sort(key=lambda r: (r['scenario'], r['man'], r['others'], r['weather'], r['source'] or '',
                                      r['run'][-15:]))
        print('\nMANOEUVRES: M6 steer around (static), M7 lane change (overtake), M8 junction. man = sa steer around '
              '(p: people too), lc lane change, y yield; clearances and PET (post-encroachment time) from CARLA '
              'ground truth')
        print_table(manoeuvre, ['scenario', 'man', 'others', 'weather', 'source', 'timing', 'result', 'end',
                                'steered', 'lane_chg', 'aborts', 'yields', 'yield_s', 'gap_wait_s', 'obst_clear_m',
                                'car_clear_m', 'pet_s', 'conflicts', 'passed_lead', 'emergency', 'collisions', 'run'])
        if any(r['scenario'] == 'junction' and r['man'] == '-' for r in manoeuvre):
            print('(junction runs with man - are the baseline without --yield)')
    if route:
        print('\nROUTE / STEP 3 TRAFFIC FOLLOWING')
        print_table(route, ['weather', 'source', 'timing', 'decides'] + (['man'] if any(r['man'] != '-' for r in route)
                                                                      else []) + ['seed', 'result', 'end', 'route_pct', 'follow_s', 'min_gap_m',
                            'min_ttc_s', 'emerg_stops', 'ego_coll', 'collisions', 'steer_rev_min', 'pedal_flips',
                            'hard_brakes', 'run'])

    if boxes:
        boxes.sort(key=lambda b: (b['weather'], b['source'] or '', b['timing'], b['run'][-15:]))
        print('\nM1 BOX CHECK (detector boxes vs CARLA ground truth; false_pos and missed_30m are counts over the run)')
        print_table(boxes, list(boxes[0].keys()))

    print('\nPER SETUP: passes / runs, then mean (worst) of each number')
    groups = OrderedDict()
    for r in rows:
        groups.setdefault((r['scenario'], r['hazard'], r['man'], r['weather'], r['source'], r['timing'],
                           r['decides']), []).append(r)
    agg_rows = []
    for (sc, hzd, man, wx, src, tm, dec), rs in groups.items():
        a = OrderedDict(scenario=sc, hazard=hzd, man=man, weather=wx, source=src, timing=tm, decides=dec,
                        passed=f"{sum(r['result'] == 'PASS' for r in rs)}/{len(rs)}",
                        seeds=','.join(str(r['seed']) for r in rs))
        for col, _, worst in AGG:
            vals = [r[col] for r in rs if isinstance(r[col], (int, float))]
            if vals:
                wv = min(vals) if worst == 'min' else max(vals)
                a[col] = f'{sum(vals) / len(vals):.2f} ({wv:.2f})' if len(vals) > 1 else f'{vals[0]:.2f}'
            else:
                a[col] = None
        agg_rows.append(a)
    cols = (['scenario'] + (['hazard'] if any(a['hazard'] != '-' for a in agg_rows) else [])
            + (['man'] if any(a['man'] != '-' for a in agg_rows) else [])
            + ['weather', 'source', 'timing', 'decides', 'passed', 'seeds']) + [c for c, _, _ in AGG
                                                                  if any(a[c] is not None for a in agg_rows)]
    print_table(agg_rows, cols)
    if any(r['decides'] == 'GT' for r in rows):
        print('(decides GT = M1 box-check run (--decide-on oracle): the model is scored, the car drives on '
              'ground truth, so its driving numbers are NOT a model result)')

    out = args.csv or os.path.join(args.runs, 'runs_table.csv')
    try:
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'\n{len(rows)} runs. Per-run table saved to {out}')
    except OSError as e:     # e.g. the csv is open in Excel
        print(f'\n{len(rows)} runs. Could not write {out} ({e}); close it in Excel and run again')


if __name__ == '__main__':
    main()