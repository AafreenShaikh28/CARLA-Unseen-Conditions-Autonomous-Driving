"""
collect_weather_tests.py - offline test sets for unseen weather: the same Town05 drive recorded once
per weather, each into its own folder, with the same seed, so the traffic is (near) identical and
only the weather changes. Uses collect_kitti_lidar.py unchanged; put this file next to it (LIDAR_only).


    python collect_weather_tests.py                          # 5 weathers x 500 frames
    python collect_weather_tests.py --frames 300 --weathers HardRainNight Fog
    python collect_weather_tests.py --quick                  # 40 frames each, smoke test


Writes kitti_weather/<Weather>/training/... in the same KITTI layout as kitti_carla (velodyne, image_2,
calib, labels_3d, semantic, ...). These frames are never used for training. Then, from the Fusion
folder: paint_points.py, check_paint.py and Lidar_val.py --split all on each folder (see compare_offline.py).


'Fog' is not a CARLA preset: it is ClearNoon + fog, the same as agent.py --weather Fog (density 60 from
10 m). --fog-density changes it.
"""
import os
import sys
import types
import argparse


import collect_kitti_lidar as cc


DEFAULT_WEATHERS = ['ClearNoon', 'HardRainNoon', 'ClearNight', 'HardRainNight', 'Fog']




def install_fog(fog_density):
    """let the collector's getattr(carla.WeatherParameters, 'Fog') work, without touching its code"""
    real = cc.carla
    custom = {'Fog': ('ClearNoon', dict(fog_density=float(fog_density), fog_distance=10.0, fog_falloff=1.0))}


    class Weather:
        def __getattr__(self, name):
            if name not in custom:
                return getattr(real.WeatherParameters, name)
            base, extra = custom[name]
            preset = getattr(real.WeatherParameters, base)
            w = real.WeatherParameters()
            for n in dir(preset):
                v = getattr(preset, n, None)
                if n.startswith('_') or isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                try:
                    setattr(w, n, float(extra.get(n, v)))
                except AttributeError:
                    pass
            return w


        def __call__(self, *a, **kw):
            return real.WeatherParameters(*a, **kw)


    proxy = types.ModuleType('carla')
    proxy.__dict__.update(real.__dict__)
    proxy.WeatherParameters = Weather()
    cc.carla = proxy




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weathers', nargs='+', default=DEFAULT_WEATHERS)
    ap.add_argument('--frames', type=int, default=500, help='saved frames per weather')
    ap.add_argument('--town', default='Town05', help='Town05 = the unseen test town')
    ap.add_argument('--out', default='kitti_weather')
    ap.add_argument('--fog-density', type=float, default=60.0)
    ap.add_argument('--quick', action='store_true', help='40 frames per weather (smoke test)')
    args = ap.parse_args()


    install_fog(args.fog_density)
    for w in args.weathers:
        if w != 'Fog' and not hasattr(cc.carla.WeatherParameters, w):
            raise SystemExit(f'{w}: not a CARLA weather preset')
    for w in args.weathers:
        cc.CFG['out'] = os.path.join(args.out, w)
        cc.CFG['runs'] = [dict(name=f'{args.town.lower()}_{w.lower()}', town=args.town, weather=w,
                               frames=args.frames, split='val')]
        print(f'\n######## {w} -> {cc.CFG["out"]}')
        sys.argv = ['collect_kitti_lidar.py'] + (['--quick'] if args.quick else [])
        cc.main()               # seeds random / numpy again: every weather starts from the same state
    print('\ndone. Folders: ' + ', '.join(os.path.join(args.out, w) for w in args.weathers))




if __name__ == '__main__':
    main()