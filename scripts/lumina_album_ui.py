"""Album Plaque Maker - a tiny desktop front-end for the Stack5 album pipelines.

    cd ~/Lumina-Layers && .venv/bin/python scripts/lumina_album_ui.py

Pick an image (or several), choose a width, let the script pick the best 5 of your
12 spools (or tick exactly 5 yourself), press Generate.  The 3MF, a preview PNG,
the AMS slot list and the recipe land in the output folder (default
~/Downloads/BambuAlbums/stack5/<image name>/).  Same inputs -> byte-identical 3MF.

Print style:
* Flat plaque  - flat Stack5, printed face DOWN (core.stack5.pipeline).
* Relief       - Relief Stack5, printed face UP: the same Stack5 colours on a 3D
                 relief, 0.8 mm base + up to 4 mm relief + 0.4 mm colour shell
                 (core.relief.pipeline).  The depth comes from Depth Anything V2
                 (estimated from the cover itself, default) or from a grey depth
                 map (black = low, white = high) chosen explicitly for a single
                 image / found as <image name>_depth.png next to each image.
                 'Art-direct relief...' opens the editor (scripts/relief_editor.py):
                 select part of the cover (SAM click or brush) and dome / extrude /
                 raise it; the edits change the geometry only and are replayed at export.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))   # scripts/relief_editor.py
os.chdir(REPO)  # Lumina writes its intermediate files relative to the cwd

FILAMENTS_JSON = os.path.join(REPO, 'assets', 'filaments_user_measured.json')
DEFAULT_OUT = os.path.expanduser('~/Downloads/BambuAlbums/stack5')
DEFAULT_OUT_RELIEF = os.path.expanduser('~/Downloads/BambuAlbums/relief')
DEPTH_SUFFIXES = ('_depth', '-depth', '.depth', '_relief', '_height')
DEPTH_EXTS = ('.png', '.tif', '.tiff', '.jpg', '.jpeg', '.webp', '.npy')
DEPTH_TYPES = [('Depth maps', '*.png *.tif *.tiff *.jpg *.jpeg *.webp *.npy *.npz'), ('All files', '*')]
DEPTH_SOURCES = {'Estimate from the cover (Depth Anything V2 AI)': 'ai',
                 'Grey depth map file (black = low, white = high)': 'file'}
ADVISORY = {'None': None, 'Bottom right': 'br', 'Bottom left': 'bl', 'Bottom centre': 'bc'}
SPOOLS = {'Only the colours the art needs (2-5)': (2, 5), 'Always 5 spools': (5, 5)}
METRICS = {'Hue-first (right hue, shade may drift)': 'hue',
           'Lumina classic (right shade, hue may drift)': 'lumina',
           'True Lab (balanced)': 'lab'}
IMAGE_TYPES = [('Images', '*.jpg *.jpeg *.png *.webp *.gif *.tif *.tiff *.bmp *.avif'), ('All files', '*')]


def find_depth_map(image_path):
    """<dir>/<stem>_depth.png (or -depth / .depth / _relief / _height, any depth
    extension) next to the image, else None."""
    d, base = os.path.split(image_path)
    stem = os.path.splitext(base)[0]
    for suf in DEPTH_SUFFIXES:
        for ext in DEPTH_EXTS:
            cand = os.path.join(d, stem + suf + ext)
            if os.path.isfile(cand):
                return cand
    return None


class QueueWriter:
    """stdout/stderr replacement: lines go to the UI queue and to a log file."""

    def __init__(self, q, log_path):
        self.q, self.buf = q, ''
        self.fh = open(log_path, 'a', encoding='utf-8')

    def write(self, s):
        self.fh.write(s)
        self.buf += s
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            self.q.put(('log', line))

    def flush(self):
        self.fh.flush()

    def isatty(self):
        return False

    def close(self):
        self.fh.close()


def _print_relief_summary(res, params):
    print('-' * 70)
    print(f"[RELIEF] palette (slot order): {', '.join(res['palette'])}; backing / relief body: {res['backing']}")
    for line in res['ams']:
        print(f'[RELIEF] {line}')
    d, st = res['dims'], res['stats']
    print(f"[RELIEF] thickness: base {d['base_mm']:g} mm + relief 0..{st['relief_layers_max'] * d['layer_h_mm']:.2f} mm "
          f"+ shell {d['shell_mm']:g} mm = {st['total_height_mm']:.2f} mm ({st['total_print_layers']} layers of 0.08), face UP")
    gr = res['geometry_report']
    for w in gr.get('warnings', []):
        print(f"[RELIEF] warning: {w}")
    print(f"[RELIEF] relief levels used: {gr['stats']['levels_used']}; max neighbour step "
          f"{gr['stats']['neighbor_steps'].get('max_step_layers', 0)} layers"
          + (f" (limit {params['max_step']})" if params.get('max_step') else ''))
    v = res.get('verification')
    print(f"[RELIEF] voxel check {'ok' if res['voxel_check']['ok'] else 'FAILED'}; 3MF verification "
          f"{'ok' if v and v['ok'] else ('skipped' if v is None else 'FAILED')}")
    tw = res['tower']
    print(f"[RELIEF] prime tower {tw['width']:g} x {tw['depth_modelled']:.0f} mm ({'fits' if tw['fits'] else 'DOES NOT FIT'}); "
          f"tool changes {st['tool_changes_planned']}; purge scale {res['flush']['flush_scale']:g}")
    ad = res.get('art_direction')
    if ad:
        print(f"[RELIEF] art direction: {ad['n_ops']} edit(s), {ad['pixels_changed']} px reshaped "
              f"({100 * ad['share_changed']:.1f} % of the plaque); relief now {ad['relief_mm_after'][0]:.2f}.."
              f"{ad['relief_mm_after'][1]:.2f} mm; edits saved beside the 3MF: {os.path.basename(res['edits_json'])}")
        for line in ad.get('summary', []):
            print(f'[RELIEF]   {line}')
    print(f"[RELIEF] triangles {st['triangles_total']:,}; runtime {res['timing']['total_s']:.0f} s")
    print(f"[RELIEF] 3mf: {res['threemf']}")


def run_jobs(params, q):
    """Worker thread: run convert_album_stack5 / convert_album_relief for every chosen image."""
    from core.stack5.pipeline import convert_album_stack5
    relief = params.get('style') == 'relief'
    if relief:
        from core.relief.depth_processing import DepthParams
        from core.relief.pipeline import convert_album_relief
        from core.relief.relief_stack import ReliefDims
    os.makedirs(params['out_dir'], exist_ok=True)
    writer = QueueWriter(q, os.path.join(params['out_dir'], 'last_run.log'))
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = writer
    try:
        for img in params['images']:
            q.put(('status', f'Working on {os.path.basename(img)} ...'))
            print('=' * 70)
            print(f'[UI] {time.strftime("%Y-%m-%d %H:%M:%S")}  {img}')
            if params.get('style') == 'region-band':
                from core.band.regions import convert_region_band
                res = convert_region_band(img, params['out_dir'], width_mm=params['width'],
                                          palette=params['palette'], filaments_json=FILAMENTS_JSON,
                                          max_spools=5 if params['spools'] == (5, 5) else 4)
                st = res['stats']
                tried = ', '.join(f"{t['regions']}: {t['J']:.0f}" for t in st['regions_tried'])
                print(f"[REGION LAYERS] regions tried (score, lower is better) - {tried}")
                print(f"[REGION LAYERS] {st['region_count']} region(s); "
                      + ('your spools, all used: ' if st['palette_mode'] == 'custom' else 'auto-picked spools: ')
                      + ', '.join(st['filaments']))
                if st['accents']:
                    print('[REGION LAYERS] accents (small areas that printed in the wrong colour family): '
                          + ', '.join(f"{a['area_mm2']:.0f} mm2" for a in st['accents']))
                for r in st['regions']:
                    print(f"[REGION LAYERS]   region {r['region']}{' (accent)' if r.get('accent') else ''} "
                          f"({r['area_share']*100:.1f}% of the plaque): "
                          f"{' -> '.join(r['spools'])}"
                          + (f"  swaps at {', '.join(f'{z:g}' for z in r['swaps_mm'])} mm" if r['swaps_mm'] else ''))
                if st['region_count'] == 1:
                    print('[REGION LAYERS] one region fits best: this cover prints like Reference layers '
                          '(plate-wide swaps, no prime tower)')
                else:
                    print(f"[REGION LAYERS] {st['mixed_layers']} layers hold two spools: {st['tool_changes_planned']} "
                          f"tool changes, ~{st['purge_mm3_planned'] / 1000:.1f} cm3 purged on the prime tower")
                print(f"[REGION LAYERS] predicted colour error dE {st['mean_dE']:.1f} (p90 {st['p90_dE']:.1f}); "
                      f"region map: {res['regions_png']}")
                print(f"[REGION LAYERS] 3mf: {res['threemf']}")
                q.put(('done', res))
                continue
            if params.get('style') == 'reference-band':
                from core.band.reference import convert_reference_band
                # auto: at most one AMS (4) unless "Always 5 spools" is chosen
                lo, hi = (5, 5) if params['spools'] == (5, 5) else (2, 4)
                res = convert_reference_band(img, params['out_dir'], width_mm=params['width'],
                                             palette=params['palette'], filaments_json=FILAMENTS_JSON,
                                             min_spools=lo, max_spools=hi)
                st = res['stats']
                lv = st['tone']['brightness_levels']
                if lv:
                    print(f"[REFERENCE BAND] contrast: cover brightness {lv[0]:.2f}..{lv[1]:.2f} stretched over the "
                          f"whole relief (contrast {st['tone']['contrast']:g}); colours fitted to that "
                          f"(_target.png)")
                if st['palette_mode'] in ('auto', 'custom'):
                    fit = st['colour_fit']
                    if st['palette_mode'] == 'auto':
                        print(f"[REFERENCE BAND] auto-picked from all your spools ({fit['palettes_searched']} "
                              f"combinations, {fit['plans_searched']} swap plans):")
                        for size, b in fit['by_size'].items():
                            print(f"[REFERENCE BAND]   best {size}: {' -> '.join(b['filaments'])}  dE {b['mean_dE']:.1f}")
                    print(f"[REFERENCE BAND] spools, bottom -> top: {' -> '.join(st['filaments'])}")
                    print('[REFERENCE BAND] swaps: ' + ', '.join(f"{z:g} mm -> {n}" for (z, _, _), n
                                                                  in zip(st['swap_entries'], st['filaments'][1:])))
                    print(f"[REFERENCE BAND] predicted colour error dE {fit['mean_dE']:.1f} "
                          f"(p90 {fit['p90_dE']:.1f}); preview shows predicted colours")
                else:
                    print('[REFERENCE BAND] Graduation colours: Black -> Purple -> Pink/Red -> White; three swaps. '
                          'Height preview only.')
                lw = st['thin_feature_lowered']
                print(f"[REFERENCE BAND] removed features thinner than {st['min_feature_mm']:g} mm "
                      f"({lw['pixel_fraction']*100:.0f}% of pixels lowered, mean {lw['mean_mm']:.3f} mm)")
                print(f"[REFERENCE BAND] 3mf: {res['threemf']}")
                q.put(('done', res))
                continue
            kw = dict(width_mm=params['width'], palette=params['palette'], backing=params['backing'],
                      advisory=params['advisory'], out_dir=params['out_dir'], seed=params['seed'],
                      prefer=params['prefer'], flush_scale=params['flush_scale'],
                      quantize_colors=params['quantize'], smooth_sigma=params['smooth'],
                      metric=params['metric'], min_spools=params['spools'][0], max_spools=params['spools'][1],
                      tower_fit='auto')
            if relief:
                if params['depth_source'] == 'ai':
                    print('[UI] depth: estimated from the cover with Depth Anything V2 (CPU, deterministic)')
                    kw.update(geometry='depth-anything',
                              provider_options={'device': 'cpu', 'allow_download': False})
                else:
                    dm = params['depth_maps'][img]
                    print(f'[UI] depth map: {dm}')
                    kw.update(depth_map=dm)
                if params.get('height_edits') is not None:
                    print(f"[UI] art direction: {len(params['height_edits'])} edit(s) replayed on the relief "
                          "(geometry only; colours from the cover)")
                    kw['height_edits'] = params['height_edits']
                    if params.get('edit_geom') is not None:
                        print('[UI] depth: reusing the estimate made in the editor')
                        kw['geometry_result'] = params['edit_geom']
                kw.update(verify='auto',
                          dims=ReliefDims(base_mm=params['base_mm'], relief_mm=params['relief_mm']),
                          depth_params=DepthParams(invert=params['invert'], max_neighbor_step=params['max_step'],
                                                   flatten_background=params['flatten'],
                                                   smoothing_px=params['depth_smoothing']))
                convert = convert_album_relief
            else:
                kw.update(orientation='face-up' if params.get('style') == 'flat-up' else 'face-down',
                          layer_h=params.get('layer_h', 0.08), early_stop=params.get('early_stop', True))
                convert = convert_album_stack5
            try:
                res = convert(img, **kw)
            except ValueError as exc:
                if 'prime tower does not fit' in str(exc) and params['auto_purge'] and kw['flush_scale'] > 0.5:
                    print('[UI] prime tower does not fit beside a plaque this wide - retrying with purge volume 0.5')
                    kw['flush_scale'] = 0.5
                    res = convert(img, **kw)
                else:
                    raise
            if relief:
                _print_relief_summary(res, params)
                q.put(('done', res))
                continue
            print('-' * 70)
            print(f"[STACK5] palette (slot order): {', '.join(res['palette'])}")
            print(f"[STACK5] backing: {res['backing']}")
            hp_ = (res.get('palette_report') or {}).get('hue_plan')
            if hp_:
                print("[STACK5] art hues -> spools: " + ('; '.join(
                    f"{c['name']} {100 * c['share']:.0f}% -> {c['pick'] or 'mix'}" for c in hp_['clusters'] if c['needed'])
                    or 'none (neutral art)') + f"; light neutrals {100 * hp_['light_neutral_share']:.0f}%, dark {100 * hp_['dark_share']:.0f}%")
            print(f"[STACK5] colour metric: {params['metric']}; {len(res['palette'])} spools"
                  + (f"; not needed: {', '.join((res.get('palette_report') or {}).get('excluded_spools') or [])}"
                     if (res.get('palette_report') or {}).get('excluded_spools') else ''))
            for line in res['ams']:
                print(f'[STACK5] {line}')
            st = res['stats']
            stop = st.get('early_stopping', {})
            if stop.get('enabled'):
                print(f"[STACK5] early stopping: {100 * stop['optical_material_saved_fraction']:.1f}% fewer optical voxels; "
                      f"pixels at 0..5 layers: {stop['pixels_by_colour_layer_count']}")
            print(f"[STACK5] tool changes (est.): {st['tool_changes_est']} over {st['total_print_layers']} layers; "
                  f"mean colour error dE {st['mean_dE_predicted_palette']:.1f}")
            tw = st['tower']
            print(f"[STACK5] prime tower {tw['width']:g} x {tw['depth_modelled']:.0f} mm "
                  f"({'fits' if tw['fits'] else 'DOES NOT FIT'}); purge scale {res['flush']['flush_scale']:g}")
            print(f"[STACK5] runtime {res['timing']['total_s']:.0f} s")
            print(f"[STACK5] 3mf: {res['threemf']}")
            q.put(('done', res))
    except Exception as exc:  # noqa: BLE001 - shown to the user
        traceback.print_exc()
        q.put(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        writer.close()
        q.put(('finished', None))


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Album Plaque Maker (Stack5 / Relief)')
        self.minsize(820, 760)
        self.depth_map = None
        self.edit_script = None       # core.relief.art_direct.EditScript from the editor / a file
        self.edit_geom = None         # GeometryResult cached by the editor (same image + depth source)
        self.edit_key = None
        with open(FILAMENTS_JSON, encoding='utf-8') as fh:
            self.fils = json.load(fh)
        self.names = [f['name'] for f in self.fils]
        self.images: list[str] = []
        self.q: queue.Queue = queue.Queue()
        self.worker = None
        self.t0 = 0.0
        self.results: list[dict] = []
        self._thumb = None
        self._build()
        # Tk windows launched from a .command open behind Terminal on macOS: pull it to the front once
        self.lift()
        self.attributes('-topmost', True)
        self.after(800, lambda: self.attributes('-topmost', False))
        self.after(200, self._poll)

    # ---------------------------------------------------------------- layout
    def _build(self):
        pad = {'padx': 6, 'pady': 3}
        f = ttk.Frame(self, padding=10)
        f.grid(sticky='nsew')
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        # image
        ttk.Button(f, text='Choose image(s)...', command=self.pick_images).grid(row=0, column=0, sticky='w', **pad)
        self.img_lbl = ttk.Label(f, text='no image chosen', wraplength=380)
        self.img_lbl.grid(row=0, column=1, columnspan=2, sticky='w', **pad)
        self.thumb = ttk.Label(f, relief='groove', anchor='center', width=30)
        self.thumb.grid(row=0, column=3, rowspan=9, sticky='ne', padx=10, pady=3)

        # width
        ttk.Label(f, text='Width (mm)').grid(row=1, column=0, sticky='w', **pad)
        self.width = tk.StringVar(value='150')
        ttk.Entry(f, textvariable=self.width, width=7).grid(row=1, column=1, sticky='w', **pad)
        ttk.Label(f, text='150 fits with the normal purge; 200 needs purge 0.5 (done automatically)',
                  foreground='#666').grid(row=1, column=2, sticky='w', **pad)

        # print style: flat (face down) or relief (face up)
        style_row = ttk.Frame(f)
        style_row.grid(row=2, column=0, columnspan=3, sticky='w', **pad)
        ttk.Label(style_row, text='Print style').grid(row=0, column=0, sticky='w')
        self.style = tk.StringVar(value='flat-up')
        ttk.Radiobutton(style_row, text='Flat plaque (Stack5, prints face down)', variable=self.style,
                        value='flat').grid(row=0, column=1, sticky='w', padx=(8, 14))
        ttk.Radiobutton(style_row, text='Relief (Stack5 colours on a 3D relief, prints face up)', variable=self.style,
                        value='relief').grid(row=0, column=2, sticky='w')
        ttk.Radiobutton(style_row, text='Flat plaque (Stack5, prints face up)', variable=self.style,
                        value='flat-up').grid(row=1, column=1, sticky='w', padx=(8, 14))
        ttk.Radiobutton(style_row, text='Reference layers (face up, Graduation style): Auto picks your spools, or tick 2-5',
                        variable=self.style, value='reference-band').grid(row=2, column=1, columnspan=2, sticky='w', padx=(8, 14))
        ttk.Radiobutton(style_row, text='Region layers (face up): Reference relief, its own spool sequence per colour region',
                        variable=self.style, value='region-band').grid(row=3, column=1, columnspan=2, sticky='w', padx=(8, 14))
        layer_row = ttk.Frame(style_row)
        layer_row.grid(row=1, column=2, sticky='w')
        ttk.Label(layer_row, text='Flat colour layers (first layer 0.20 mm): ').pack(side='left')
        self.colour_layer = ttk.Combobox(layer_row, values=['0.08', '0.04 experimental'],
                                         state='readonly', width=18)
        self.colour_layer.set('0.08')
        self.colour_layer.pack(side='left')
        self.early_stop = tk.BooleanVar(value=True)
        ttk.Checkbutton(style_row, text='Face up: stop each area at its best colour match',
                        variable=self.early_stop).grid(row=4, column=1, columnspan=2, sticky='w', padx=8)
        self.style.trace_add('write', lambda *_: self._on_style())

        self.relief_frame = ttk.LabelFrame(f, text='Relief', padding=(8, 4))
        self.relief_frame.grid(row=3, column=0, columnspan=3, sticky='ew', **pad)
        rf = self.relief_frame
        ttk.Label(rf, text='Depth from').grid(row=0, column=0, sticky='w')
        self.depth_source = ttk.Combobox(rf, values=list(DEPTH_SOURCES), state='readonly', width=44)
        self.depth_source.set(next(iter(DEPTH_SOURCES)))
        self.depth_source.grid(row=0, column=1, columnspan=5, sticky='w', padx=6)
        self.depth_source.bind('<<ComboboxSelected>>', lambda *_: self._on_depth_source())
        self.depth_status = ttk.Label(rf, text='', foreground='#666', wraplength=640)
        self.depth_status.grid(row=1, column=0, columnspan=8, sticky='w', pady=(2, 0))
        self.depth_btn = ttk.Button(rf, text='Choose depth map...', command=self.pick_depth)
        self.depth_btn.grid(row=2, column=0, sticky='w', pady=(4, 0))
        self.depth_lbl = ttk.Label(rf, text='auto: <image name>_depth.png next to each image (black = low, white = high)',
                                   wraplength=520)
        self.depth_lbl.grid(row=2, column=1, columnspan=6, sticky='w', padx=6, pady=(4, 0))
        self.depth_clear = ttk.Button(rf, text='Clear', width=6, command=self.clear_depth)
        self.depth_clear.grid(row=2, column=7, sticky='w', pady=(4, 0))
        ttk.Label(rf, text='Relief height (mm)').grid(row=3, column=0, sticky='w', pady=(6, 0))
        self.relief_mm = tk.StringVar(value='4.0')
        ttk.Entry(rf, textvariable=self.relief_mm, width=6).grid(row=3, column=1, sticky='w', padx=(4, 12), pady=(6, 0))
        ttk.Label(rf, text='Base (mm)').grid(row=3, column=2, sticky='w', pady=(6, 0))
        self.base_mm = tk.StringVar(value='0.8')
        ttk.Entry(rf, textvariable=self.base_mm, width=6).grid(row=3, column=3, sticky='w', padx=(4, 12), pady=(6, 0))
        ttk.Label(rf, text='Max step (layers)').grid(row=3, column=4, sticky='w', pady=(6, 0))
        self.max_step = tk.StringVar(value='')
        ttk.Entry(rf, textvariable=self.max_step, width=6).grid(row=3, column=5, sticky='w', padx=(4, 12), pady=(6, 0))
        ttk.Label(rf, text='Depth smoothing (px)').grid(row=3, column=6, sticky='w', pady=(6, 0))
        self.depth_smoothing = tk.StringVar(value='3')
        ttk.Entry(rf, textvariable=self.depth_smoothing, width=6).grid(row=3, column=7, sticky='w', padx=(4, 0), pady=(6, 0))
        self.invert = tk.BooleanVar(value=False)
        ttk.Checkbutton(rf, text='Invert depth (white = low)', variable=self.invert).grid(row=4, column=0, columnspan=3, sticky='w', pady=(6, 0))
        self.flatten = tk.BooleanVar(value=False)
        ttk.Checkbutton(rf, text='Flatten background (needs a transparent background in the depth map)',
                        variable=self.flatten).grid(row=4, column=3, columnspan=5, sticky='w', pady=(6, 0))
        ttk.Label(rf, text='0.08 mm layers; total thickness base + relief + 0.4 mm colour shell (5.2 mm max). '
                           'Blank max step = unlimited; 8 = cliffs become ramps of <= 0.64 mm per 0.1 mm.',
                  foreground='#666', wraplength=640).grid(row=5, column=0, columnspan=8, sticky='w', pady=(4, 0))
        art = ttk.Frame(rf)
        art.grid(row=6, column=0, columnspan=8, sticky='w', pady=(6, 0))
        ttk.Button(art, text='Art-direct relief...', command=self.open_editor).pack(side='left')
        self.edits_lbl = ttk.Label(art, text='no shape edits (select an object on the cover and dome / extrude / raise it)',
                                   wraplength=430)
        self.edits_lbl.pack(side='left', padx=8)
        ttk.Button(art, text='Load edits...', command=self.load_edits).pack(side='left')
        ttk.Button(art, text='Clear', width=6, command=self.clear_edits).pack(side='left', padx=(4, 0))
        self._on_depth_source()
        self.relief_frame.grid_remove()

        # palette mode
        self.mode = tk.StringVar(value='auto')
        ttk.Radiobutton(f, text='Auto: best 5 of your 12 spools (uses a real spool when the colour is on the cover, mixes layers otherwise)',
                        variable=self.mode, value='auto').grid(row=4, column=0, columnspan=3, sticky='w', **pad)
        ttk.Radiobutton(f, text='Pick exactly 5 spools (order = AMS slots 1..5):',
                        variable=self.mode, value='manual').grid(row=5, column=0, columnspan=3, sticky='w', **pad)
        box = ttk.Frame(f)
        box.grid(row=6, column=0, columnspan=3, sticky='w', padx=24)
        self.checks: dict[str, tk.BooleanVar] = {}
        for i, fil in enumerate(self.fils):
            var = tk.BooleanVar(value=False)
            self.checks[fil['name']] = var
            r, c = divmod(i, 4)
            sw = tk.Label(box, bg=fil['hex'], width=2, relief='solid', bd=1)
            sw.grid(row=r, column=2 * c, padx=(6, 2), pady=1)
            ttk.Checkbutton(box, text=fil['name'], variable=var).grid(row=r, column=2 * c + 1, sticky='w', padx=(0, 10))

        # options row
        opts = ttk.Frame(f)
        opts.grid(row=7, column=0, columnspan=3, sticky='w', **pad)
        ttk.Label(opts, text='Prefer spool').grid(row=0, column=0, sticky='w')
        self.prefer = ttk.Combobox(opts, values=['(none)'] + self.names, state='readonly', width=15)
        self.prefer.set('(none)')
        self.prefer.grid(row=0, column=1, padx=(4, 14))
        ttk.Label(opts, text='Backing').grid(row=0, column=2, sticky='w')
        self.backing = ttk.Combobox(opts, values=['Auto'] + self.names, state='readonly', width=15)
        self.backing.set('Auto')
        self.backing.grid(row=0, column=3, padx=(4, 14))
        ttk.Label(opts, text='Advisory label').grid(row=0, column=4, sticky='w')
        self.advisory = ttk.Combobox(opts, values=list(ADVISORY), state='readonly', width=13)
        self.advisory.set('None')
        self.advisory.grid(row=0, column=5, padx=(4, 0))
        ttk.Label(opts, text='Colour matching').grid(row=1, column=0, sticky='w', pady=(6, 0))
        self.metric = ttk.Combobox(opts, values=list(METRICS), state='readonly', width=40)
        self.metric.set(next(iter(METRICS)))
        self.metric.grid(row=1, column=1, columnspan=3, sticky='w', padx=(4, 0), pady=(6, 0))
        ttk.Label(opts, text='Spools').grid(row=1, column=4, sticky='w', pady=(6, 0))
        self.spools = ttk.Combobox(opts, values=list(SPOOLS), state='readonly', width=30)
        self.spools.set(next(iter(SPOOLS)))
        self.spools.grid(row=1, column=5, sticky='w', padx=(4, 0), pady=(6, 0))

        adv = ttk.Frame(f)
        adv.grid(row=8, column=0, columnspan=3, sticky='w', **pad)
        ttk.Label(adv, text='Purge volume').grid(row=0, column=0)
        self.flush = tk.StringVar(value='1.0')
        ttk.Entry(adv, textvariable=self.flush, width=5).grid(row=0, column=1, padx=(4, 4))
        self.auto_purge = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv, text='auto-reduce if the prime tower does not fit', variable=self.auto_purge).grid(row=0, column=2, padx=(0, 14))
        ttk.Label(adv, text='Colours').grid(row=0, column=3)
        self.quant = tk.StringVar(value='96')
        ttk.Entry(adv, textvariable=self.quant, width=5).grid(row=0, column=4, padx=(4, 14))
        ttk.Label(adv, text='Smoothing').grid(row=0, column=5)
        self.smooth = tk.StringVar(value='10')
        ttk.Entry(adv, textvariable=self.smooth, width=5).grid(row=0, column=6, padx=(4, 14))
        ttk.Label(adv, text='Seed').grid(row=0, column=7)
        self.seed = tk.StringVar(value='0')
        ttk.Entry(adv, textvariable=self.seed, width=5).grid(row=0, column=8, padx=(4, 0))

        # output dir
        ttk.Label(f, text='Output folder').grid(row=9, column=0, sticky='w', **pad)
        self.out_dir = tk.StringVar(value=DEFAULT_OUT)
        ttk.Entry(f, textvariable=self.out_dir).grid(row=9, column=1, columnspan=2, sticky='ew', **pad)
        ttk.Button(f, text='Change...', command=self.pick_out).grid(row=9, column=3, sticky='w', **pad)

        # go
        go = ttk.Frame(f)
        go.grid(row=10, column=0, columnspan=4, sticky='ew', **pad)
        self.go_btn = ttk.Button(go, text='Generate 3MF', command=self.generate)
        self.go_btn.pack(side='left')
        self.status = tk.StringVar(value='Choose an image to start.')
        ttk.Label(go, textvariable=self.status).pack(side='left', padx=12)

        # log
        self.log = tk.Text(f, height=12, wrap='word', font=('Menlo', 11))
        self.log.grid(row=11, column=0, columnspan=4, sticky='nsew', **pad)
        f.rowconfigure(11, weight=1)
        sb = ttk.Scrollbar(f, command=self.log.yview)
        sb.grid(row=11, column=4, sticky='ns')
        self.log.configure(yscrollcommand=sb.set, state='disabled')

        # result buttons
        rb = ttk.Frame(f)
        rb.grid(row=12, column=0, columnspan=4, sticky='w', **pad)
        self.finder_btn = ttk.Button(rb, text='Show in Finder', command=self.show_in_finder, state='disabled')
        self.finder_btn.pack(side='left')
        self.bambu_btn = ttk.Button(rb, text='Open in Bambu Studio', command=self.open_in_bambu, state='disabled')
        self.bambu_btn.pack(side='left', padx=8)

    # --------------------------------------------------------------- actions
    def _on_style(self):
        relief = self.style.get() == 'relief'
        if relief:
            self.relief_frame.grid()
            if os.path.expanduser(self.out_dir.get().strip()) in ('', DEFAULT_OUT):
                self.out_dir.set(DEFAULT_OUT_RELIEF)
            self.go_btn.configure(text='Generate relief 3MF')
        else:
            self.relief_frame.grid_remove()
            if os.path.expanduser(self.out_dir.get().strip()) in ('', DEFAULT_OUT_RELIEF):
                self.out_dir.set(DEFAULT_OUT)
            self.go_btn.configure(text='Generate 3MF')
        self._update_depth_label()

    def _on_depth_source(self):
        ai = DEPTH_SOURCES[self.depth_source.get()] == 'ai'
        for w in (self.depth_btn, self.depth_lbl, self.depth_clear):
            if ai:
                w.grid_remove()
            else:
                w.grid()
        if ai:
            try:
                from core.relief.depth_anything import DEFAULT_MODEL_ID, dependency_status, weights_cached
                st = dependency_status()
                if not all(st.values()):
                    txt = ('Depth Anything is not installed: run  .venv/bin/pip install torch torchvision transformers  in ~/Lumina-Layers, '
                           'or switch to a depth map file.')
                elif not weights_cached(DEFAULT_MODEL_ID):
                    txt = (f'Model weights not downloaded yet: run  .venv/bin/python scripts/convert_relief.py --fetch-depth-model '
                           f' once (about 100 MB), or switch to a depth map file.')
                else:
                    txt = ('The relief is estimated from the cover itself (faces and objects come forward, the background '
                           'sinks). Nothing else to supply.')
            except Exception as exc:  # noqa: BLE001
                txt = f'(could not check the AI depth model: {exc})'
            self.depth_status.configure(text=txt)
        else:
            self.depth_status.configure(text='')

    # --------------------------------------------------------------- art direction
    def _ai_depth_check(self):
        from core.relief.depth_anything import DEFAULT_MODEL_ID, dependency_status, weights_cached
        if not all(dependency_status().values()):
            raise ValueError('Depth Anything is not installed. In Terminal:\n\n'
                             'cd ~/Lumina-Layers && .venv/bin/pip install torch torchvision transformers\n\n'
                             'or choose "Grey depth map file" as the depth source.')
        if not weights_cached(DEFAULT_MODEL_ID):
            raise ValueError('The Depth Anything model is not downloaded yet. In Terminal:\n\n'
                             'cd ~/Lumina-Layers && .venv/bin/python scripts/convert_relief.py --fetch-depth-model\n\n'
                             '(about 100 MB, once) or choose "Grey depth map file" as the depth source.')

    def _relief_settings(self):
        """Relief controls -> (dims, params, source, depth map for the single image)."""
        from core.relief.depth_processing import DepthParams
        from core.relief.relief_stack import ReliefDims
        relief_mm = float(self.relief_mm.get())
        base_mm = float(self.base_mm.get())
        if not 0 <= relief_mm <= 10:
            raise ValueError('Relief height must be between 0 and 10 mm (4 is the standard).')
        if not 0.08 <= base_mm <= 5:
            raise ValueError('Base must be between 0.08 and 5 mm (0.8 is the standard).')
        ms = self.max_step.get().strip()
        max_step = None
        if ms:
            max_step = int(ms)
            if max_step < 1:
                raise ValueError('Max step must be a whole number of layers >= 1, or blank for unlimited.')
        smoothing = float(self.depth_smoothing.get() or 0)
        if smoothing < 0:
            raise ValueError('Depth smoothing must be >= 0 px.')
        source = DEPTH_SOURCES[self.depth_source.get()]
        dm = None
        if source == 'ai':
            self._ai_depth_check()
        elif self.images:
            dm = self.depth_map if (self.depth_map and len(self.images) == 1) else find_depth_map(self.images[0])
        dims = ReliefDims(base_mm=base_mm, relief_mm=relief_mm)
        params = DepthParams(invert=bool(self.invert.get()), max_neighbor_step=max_step,
                             flatten_background=bool(self.flatten.get()), smoothing_px=smoothing)
        return dims, params, source, dm

    def open_editor(self):
        try:
            if self.style.get() != 'relief':
                raise ValueError('Switch Print style to Relief first.')
            if len(self.images) != 1:
                raise ValueError('Choose exactly one image to art-direct its relief.')
            width = float(self.width.get())
            if not 20 <= width <= 250:
                raise ValueError('Width must be between 20 and 250 mm.')
            dims, params, source, dm = self._relief_settings()
            if source == 'file' and not dm:
                raise ValueError('No depth map for this image: pick one with "Choose depth map..." or switch the depth '
                                 'source to the AI estimate.')
        except ValueError as exc:
            messagebox.showwarning('Album Plaque Maker', str(exc))
            return
        from relief_editor import ReliefEditor
        img = self.images[0]
        key = (os.path.abspath(img), source, dm)
        prev = self.edit_script if (self.edit_script is not None and self.edit_key and self.edit_key[0] == key[0]) else None
        ReliefEditor(self, img, width, dims=dims, params=params, depth_source=source, depth_map=dm,
                     provider_options={'device': 'cpu', 'allow_download': False}, on_done=self._edits_done,
                     initial_script=prev)

    def _edits_done(self, script, geom):
        self.edit_script = script if len(script) else None
        img = self.images[0] if self.images else script.meta.get('image')
        source = DEPTH_SOURCES[self.depth_source.get()]
        dm = None if source == 'ai' else (self.depth_map or (find_depth_map(img) if img else None))
        self.edit_geom = geom
        self.edit_key = (os.path.abspath(img) if img else None, source, dm)
        self._update_edits_label()

    def load_edits(self):
        path = filedialog.askopenfilename(title='Load relief edits', filetypes=[('Relief edits', '*.json'), ('All files', '*')])
        if not path:
            return
        try:
            from core.relief.art_direct import EditScript
            script = EditScript.load(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror('Album Plaque Maker', f'Could not load {path}:\n{exc}')
            return
        self.edit_script = script if len(script) else None
        self.edit_geom = None
        self.edit_key = (os.path.abspath(script.meta['image']) if script.meta.get('image') else None, None, None)
        if self.style.get() != 'relief':
            self.style.set('relief')
        self._update_edits_label(path)

    def clear_edits(self):
        self.edit_script = None
        self.edit_geom = None
        self.edit_key = None
        self._update_edits_label()

    def _update_edits_label(self, path=None):
        if self.edit_script is None:
            self.edits_lbl.configure(text='no shape edits (select an object on the cover and dome / extrude / raise it)')
            return
        img = self.edit_script.meta.get('image')
        n = len(self.edit_script)
        txt = f"{n} shape edit{'s' if n != 1 else ''}" + (f' for {os.path.basename(img)}' if img else '') + ': ' \
            + '; '.join(self.edit_script.summary())[:160]
        if path:
            txt += f'  (from {os.path.basename(path)})'
        self.edits_lbl.configure(text=txt)

    def pick_depth(self):
        path = filedialog.askopenfilename(title='Choose the depth map (black = low, white = high)', filetypes=DEPTH_TYPES)
        if path:
            self.depth_map = path
            self._update_depth_label()

    def clear_depth(self):
        self.depth_map = None
        self._update_depth_label()

    def _update_depth_label(self):
        if self.depth_map:
            self.depth_lbl.configure(text=self.depth_map)
            return
        if self.images:
            found = [find_depth_map(p) for p in self.images]
            n = sum(1 for d in found if d)
            if len(self.images) == 1:
                txt = f'auto: {found[0]}' if found[0] else ('auto: no <name>_depth.png found next to the image - '
                                                            'choose a depth map')
            else:
                txt = f'auto: {n} of {len(self.images)} images have a <name>_depth.png next to them'
        else:
            txt = 'auto: <image name>_depth.png next to each image (black = low, white = high)'
        self.depth_lbl.configure(text=txt)

    def pick_images(self):
        paths = filedialog.askopenfilenames(title='Choose album art', filetypes=IMAGE_TYPES)
        if paths:
            self.set_images(paths)

    def set_images(self, paths):
        self.images = list(paths)
        if len(paths) == 1:
            self.img_lbl.configure(text=paths[0])
        else:
            self.img_lbl.configure(text=f'{len(paths)} images: ' + ', '.join(os.path.basename(p) for p in paths))
        self._show_thumb(paths[0])
        self._update_depth_label()
        self.status.set('Ready. Press Generate 3MF.')

    def pick_out(self):
        d = filedialog.askdirectory(title='Output folder', initialdir=self.out_dir.get())
        if d:
            self.out_dir.set(d)

    def _show_thumb(self, path):
        try:
            from PIL import Image, ImageTk
            im = Image.open(path).convert('RGB')
            im.thumbnail((230, 230))
            self._thumb = ImageTk.PhotoImage(im)
            self.thumb.configure(image=self._thumb, text='', width=0)
        except Exception as exc:  # noqa: BLE001
            self.thumb.configure(image='', text=f'(no preview: {exc})')

    def _log(self, line):
        self.log.configure(state='normal')
        self.log.insert('end', line + '\n')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _params(self):
        if not self.images:
            raise ValueError('Choose an image first.')
        width = float(self.width.get())
        if not 20 <= width <= 250:
            raise ValueError('Width must be between 20 and 250 mm.')
        style = self.style.get()
        palette = None
        if self.mode.get() == 'manual':
            palette = [n for n in self.names if self.checks[n].get()]
            if style in ('reference-band', 'region-band'):
                # any 2-5 spools; they are stacked dark -> light and the swap layers fitted to the image
                if not 2 <= len(palette) <= 5:
                    raise ValueError(f'Tick 2 to 5 spools for Reference / Region layers (you ticked {len(palette)}).')
            elif len(palette) != 5:
                raise ValueError(f'Tick exactly 5 spools (you ticked {len(palette)}).')
        backing = None if self.backing.get() == 'Auto' else self.backing.get()
        if style not in ('reference-band', 'region-band') and palette and backing and backing not in palette:
            raise ValueError('The backing spool must be one of the 5 ticked spools.')
        prefer = () if self.prefer.get() == '(none)' else (self.prefer.get(),)
        params = {
            'style': style,
            'layer_h': float(self.colour_layer.get().split()[0]),
            'early_stop': self.early_stop.get(),
            'images': list(self.images), 'width': width, 'palette': palette, 'backing': backing,
            'prefer': prefer, 'advisory': ADVISORY[self.advisory.get()],
            'flush_scale': float(self.flush.get()), 'auto_purge': bool(self.auto_purge.get()),
            'quantize': int(self.quant.get()), 'smooth': float(self.smooth.get()),
            'seed': int(self.seed.get()),
            'out_dir': os.path.expanduser(self.out_dir.get().strip() or (DEFAULT_OUT_RELIEF if style == 'relief' else DEFAULT_OUT)),
            'metric': METRICS[self.metric.get()], 'spools': SPOOLS[self.spools.get()],
        }
        if style == 'relief':
            relief_mm = float(self.relief_mm.get())
            base_mm = float(self.base_mm.get())
            if not 0 <= relief_mm <= 10:
                raise ValueError('Relief height must be between 0 and 10 mm (4 is the standard).')
            if not 0.08 <= base_mm <= 5:
                raise ValueError('Base must be between 0.08 and 5 mm (0.8 is the standard).')
            ms = self.max_step.get().strip()
            max_step = None
            if ms:
                max_step = int(ms)
                if max_step < 1:
                    raise ValueError('Max step must be a whole number of layers >= 1, or blank for unlimited.')
            smoothing = float(self.depth_smoothing.get() or 0)
            if smoothing < 0:
                raise ValueError('Depth smoothing must be >= 0 px.')
            source = DEPTH_SOURCES[self.depth_source.get()]
            depth_maps = {}
            if source == 'ai':
                from core.relief.depth_anything import DEFAULT_MODEL_ID, dependency_status, weights_cached
                if not all(dependency_status().values()):
                    raise ValueError('Depth Anything is not installed. In Terminal:\n\n'
                                     'cd ~/Lumina-Layers && .venv/bin/pip install torch torchvision transformers\n\n'
                                     'or choose "Grey depth map file" as the depth source.')
                if not weights_cached(DEFAULT_MODEL_ID):
                    raise ValueError('The Depth Anything model is not downloaded yet. In Terminal:\n\n'
                                     'cd ~/Lumina-Layers && .venv/bin/python scripts/convert_relief.py --fetch-depth-model\n\n'
                                     '(about 100 MB, once) or choose "Grey depth map file" as the depth source.')
            else:
                missing = []
                for img in self.images:
                    dm = self.depth_map if (self.depth_map and len(self.images) == 1) else find_depth_map(img)
                    if dm:
                        depth_maps[img] = dm
                    else:
                        missing.append(os.path.basename(img))
                if missing:
                    raise ValueError('No depth map for: ' + ', '.join(missing) + '.\n\nPut a grey depth image named '
                                     '<image name>_depth.png next to each image (black = low, white = high), choose '
                                     'a single image and pick its depth map with "Choose depth map...", or switch the '
                                     'depth source to the AI estimate.')
            params.update(relief_mm=relief_mm, base_mm=base_mm, max_step=max_step, depth_smoothing=smoothing,
                          invert=bool(self.invert.get()), flatten=bool(self.flatten.get()), depth_maps=depth_maps,
                          depth_source=source)
            if self.edit_script is not None:
                if len(self.images) != 1:
                    raise ValueError('The shape edits belong to one image: choose a single image, or Clear the edits '
                                     'to run a batch.')
                img0 = os.path.abspath(self.images[0])
                simg = self.edit_script.meta.get('image')
                if simg and os.path.abspath(simg) != img0:
                    raise ValueError(f'The loaded shape edits were made for {os.path.basename(simg)}, not '
                                     f'{os.path.basename(img0)}. Clear them or open the editor for this image.')
                params['height_edits'] = self.edit_script
                key = (img0, source, depth_maps.get(self.images[0]) if source == 'file' else None)
                params['edit_geom'] = self.edit_geom if (self.edit_geom is not None and self.edit_key == key) else None
        return params

    def generate(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            params = self._params()
        except ValueError as exc:
            messagebox.showwarning('Album Plaque Maker', str(exc))
            return
        self.results = []
        self.finder_btn.configure(state='disabled')
        self.bambu_btn.configure(state='disabled')
        self.go_btn.configure(state='disabled')
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.configure(state='disabled')
        self.t0 = time.time()
        self.worker = threading.Thread(target=run_jobs, args=(params, self.q), daemon=True)
        self.worker.start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == 'log':
                    if payload.startswith(('[STACK5]', '[RELIEF]', '[UI]')):
                        self._log(payload)
                elif kind == 'status':
                    self._cur = payload
                elif kind == 'done':
                    self.results.append(payload)
                    self._show_thumb(payload['preview_png'])
                elif kind == 'error':
                    self._log('FAILED: ' + payload)
                    messagebox.showerror('Album Plaque Maker', payload + '\n\nFull log: ' + os.path.join(self.out_dir.get(), 'last_run.log'))
                elif kind == 'finished':
                    self.go_btn.configure(state='normal')
                    if self.results:
                        self.finder_btn.configure(state='normal')
                        self.bambu_btn.configure(state='normal')
                        n = len(self.results)
                        self.status.set(f'Done: {n} file{"s" if n > 1 else ""} in {time.time() - self.t0:.0f} s -> {self.results[-1]["threemf"]}')
                    else:
                        self.status.set('Failed - see log.')
        except queue.Empty:
            pass
        if self.worker and self.worker.is_alive():
            hint = '1-3 min per album' if self.style.get() == 'relief' else '30-90 s per album'
            self.status.set(f'{getattr(self, "_cur", "Working")}  {time.time() - self.t0:.0f} s  ({hint})')
        self.after(200, self._poll)

    def show_in_finder(self):
        if self.results:
            subprocess.Popen(['open', '-R', self.results[-1]['threemf']])

    def open_in_bambu(self):
        for r in self.results:
            subprocess.Popen(['open', r['threemf']])


def main():
    app = App()
    argv = sys.argv[1:]
    if '--image' in argv:  # optional pre-selected image(s): --image a.jpg [b.png ...]
        i = argv.index('--image') + 1
        imgs = []
        while i < len(argv) and not argv[i].startswith('--'):
            imgs.append(os.path.abspath(argv[i]))
            i += 1
        if imgs:
            app.set_images(imgs)
    if '--relief' in argv:
        app.style.set('relief')
    if '--depth-map' in argv:
        app.depth_map = os.path.abspath(argv[argv.index('--depth-map') + 1])
        app.depth_source.set(list(DEPTH_SOURCES)[1])
        app._on_depth_source()
        app._update_depth_label()
    if '--edits' in argv:
        from core.relief.art_direct import EditScript
        app.edit_script = EditScript.load(argv[argv.index('--edits') + 1])
        app.edit_key = (os.path.abspath(app.edit_script.meta['image']) if app.edit_script.meta.get('image') else None, None, None)
        app.style.set('relief')
        app._update_edits_label()
    if '--selftest' in argv:
        app.update()
        app.style.set('relief')
        app.update()
        app.style.set('flat')
        app.update()
        app.destroy()
        print('selftest ok')
        return
    if '--screenshot' in argv:  # capture the window to a PNG and quit (used for QA)
        path = argv[argv.index('--screenshot') + 1]

        def shot():
            app.attributes('-topmost', True)
            app.update()
            region = f'{app.winfo_rootx()},{app.winfo_rooty()},{app.winfo_width()},{app.winfo_height()}'
            subprocess.run(['screencapture', '-x', '-R', region, path], check=False)
            app.destroy()
        app.after(900, shot)
        app.mainloop()
        print('screenshot', path)
        return
    app.mainloop()


if __name__ == '__main__':
    main()
