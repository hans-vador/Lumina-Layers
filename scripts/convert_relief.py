#!/usr/bin/env python
"""Relief Stack5 CLI: album cover + depth -> face-up full-colour relief 3MF (Bambu X2D).

Colour comes from the ORIGINAL image through the unchanged Stack5 matcher;
geometry comes from a depth provider.  Defaults: 0.8 mm base, up to 4.0 mm relief,
0.08 mm layers, five colour layers (max total 5.2 mm = 65 layers).

Examples
  # grey depth map (black = low, white = high), 150 mm wide plaque, auto palette
  .venv/bin/python scripts/convert_relief.py album.png --depth-map album_depth.png --output album_relief.3mf

  # fixed palette, gentler relief, limit cliffs to 8 layers (0.64 mm) between neighbours
  .venv/bin/python scripts/convert_relief.py album.png --depth-map d.png --relief-mm 3 \\
      --max-neighbor-step 8 --palette 'Black,White,Red,Oak,Beige'

  # Depth Anything V2 (optional: pip install torch torchvision transformers; weights must be local
  # unless --allow-download is given)
  .venv/bin/python scripts/convert_relief.py album.png --geometry depth-anything --allow-download

  # Wonder3D reconstruction: orthographic front depth rendered from its mesh
  .venv/bin/python scripts/convert_relief.py album.png --geometry wonder3d --mesh recon.obj \\
      --foreground-mask album_mask.png

  # art direction: replay the editor's mask edits (scripts/relief_editor.py or the Album Plaque Maker window)
  .venv/bin/python scripts/convert_relief.py album.jpg --geometry depth-anything --edits album_edits.json

  # 60 x 60 mm calibration relief (dome + 0..4 mm step band) for the first physical test
  .venv/bin/python scripts/convert_relief.py --calibration --out output/relief
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

setattr(np, "asscalar", lambda a: a.item())                 # colormath shim (main.py:24-28)
os.environ.setdefault("LUMINA_COLOR_RECIPE_POLICY", "off")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _split(s):
    return [x.strip() for x in (s or '').split(',') if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image', nargs='?', help='original album image (any raster format)')
    ap.add_argument('--output', default=None, help='output .3mf path (side files go next to it)')
    ap.add_argument('--out', default=os.path.join(REPO, 'output', 'relief'),
                    help='output root when --output is not given (default output/relief/<image>/)')
    ap.add_argument('--title', default=None, help='plate / object name (default: image name)')
    ap.add_argument('--calibration', action='store_true',
                    help='generate and convert the 60x60 mm calibration relief instead of an image')
    ap.add_argument('--calibration-mm', type=float, default=60.0, help='calibration plaque size (default 60)')
    ap.add_argument('--check-deps', action='store_true', help='report optional AI dependencies and exit')
    ap.add_argument('--fetch-depth-model', action='store_true',
                    help='download the Depth Anything V2 Small checkpoint (~100 MB) into the Hugging Face cache and exit')
    ap.add_argument('--json', action='store_true', help='also print the result as JSON')
    ap.add_argument('--fetch-sam-model', action='store_true',
                    help='download the SAM checkpoint for click-to-select (about 375 MB, or --sam-weights) and exit')
    ap.add_argument('--sam-weights', default=None, help='SAM checkpoint: HF model id or local directory')

    a = ap.add_argument_group('art direction (geometry only; colours stay the original image)')
    a.add_argument('--edits', default=None,
                   help='relief edit script (<image>_edits.json from the editor): mask operations replayed on the relief')
    a.add_argument('--edit-clean-px', type=int, default=None,
                   help='min-feature kernel applied to the edited region (default: the script value, 4 = 0.4 mm)')

    g = ap.add_argument_group('geometry')
    g.add_argument('--geometry', choices=('imported-depth', 'depth-anything', 'wonder3d'), default='imported-depth')
    g.add_argument('--depth-map', default=None, help='grey depth image (black = low, white = high) or .npy')
    g.add_argument('--mesh', default=None, help='wonder3d: reconstructed mesh (.obj/.ply/.glb/.stl)')
    g.add_argument('--normals', default=None, help='RGB normal map (x right, y down, z out) for fusion')
    g.add_argument('--normals-y-up', action='store_true', help='normal map uses OpenGL y-up')
    g.add_argument('--normal-weight', type=float, default=0.0, help='fuse normals into the depth (0 = off)')
    g.add_argument('--depth-weight', type=float, default=1.0)
    g.add_argument('--smoothness-weight', type=float, default=0.0)
    g.add_argument('--foreground-mask', default=None, help='foreground mask image (white / opaque = object)')
    g.add_argument('--weights', default=None, help='depth-anything: local checkpoint dir or HF model id')
    g.add_argument('--allow-download', action='store_true', help='depth-anything: allow fetching weights')
    g.add_argument('--device', default='auto', help='depth-anything: cpu / cuda / mps / auto')
    g.add_argument('--front-axis', default='+z', help='wonder3d: mesh axis pointing at the viewer (default +z)')
    g.add_argument('--up-axis', default='+y', help='wonder3d: mesh up axis (default +y)')
    g.add_argument('--render-px', type=int, default=768, help='wonder3d: depth render size, long side')
    g.add_argument('--fit', choices=('mask', 'image'), default=None,
                   help="wonder3d: align the mesh bbox to the mask bbox or the whole image")
    g.add_argument('--depth-map-convention', choices=('higher-is-closer', 'higher-is-farther'),
                   default='higher-is-closer', help='wonder3d --depth-map: z-buffer maps are higher-is-farther')

    r = ap.add_argument_group('relief')
    r.add_argument('--relief-mm', type=float, default=4.0, help='maximum relief above the base (default 4.0)')
    r.add_argument('--base-mm', type=float, default=0.8, help='flat base thickness (default 0.8)')
    r.add_argument('--layer-height', type=float, default=0.08, help='layer height (default 0.08)')
    r.add_argument('--color-layers', type=int, default=5, help='Stack5 colour layers (5)')
    r.add_argument('--invert-depth', action='store_true', help='treat the depth as white = low')
    r.add_argument('--depth-normalize', choices=('percentile', 'none'), default='percentile',
                   help="'none' takes the map as already 0..1")
    r.add_argument('--depth-low-percentile', type=float, default=1.0)
    r.add_argument('--depth-high-percentile', type=float, default=99.0)
    r.add_argument('--depth-smoothing', type=float, default=3.0,
                   help='bilateral sigma_space in grid pixels (0.1 mm each); 0 = off (default 3)')
    r.add_argument('--depth-smoothing-range', type=float, default=0.08,
                   help='bilateral sigma_color in normalised depth (default 0.08)')
    r.add_argument('--median-px', type=int, default=0, help='odd median kernel before smoothing (0 = off)')
    r.add_argument('--max-neighbor-step', type=int, default=None,
                   help='max height difference between neighbouring pixels, in layers (default: unlimited)')
    r.add_argument('--neighbor-connectivity', type=int, choices=(4, 8), default=4)
    r.add_argument('--no-protect-silhouette', action='store_true',
                   help='also ramp the foreground/background boundary (default keeps it as a vertical wall)')
    r.add_argument('--min-feature-px', type=int, default=4,
                   help='remove relief features thinner than this many pixels (default 4 = 0.4 mm)')
    r.add_argument('--step-layers', type=int, default=1, help='relief granularity in layers (default 1)')
    r.add_argument('--flatten-background', action='store_true', help='set the background (mask) flat')
    r.add_argument('--background-level', type=float, default=0.0, help='normalised height of the flat background')

    c = ap.add_argument_group('colour (Stack5, unchanged)')
    c.add_argument('--width', type=float, default=150.0, help='plaque width in mm (default 150)')
    c.add_argument('--palette', default='auto', help="'auto' or a comma list of 2..5 filament names in AMS order")
    c.add_argument('--backing', default=None, help='backing filament (default: from the palette search)')
    c.add_argument('--must-include', default='')
    c.add_argument('--prefer', default='')
    c.add_argument('--quantize', type=int, default=96)
    c.add_argument('--smooth-sigma', type=float, default=10.0)
    c.add_argument('--metric', choices=('hue', 'lumina', 'lab'), default='hue')
    c.add_argument('--hue-params', default='')
    c.add_argument('--min-region-px', type=int, default=16)
    c.add_argument('--advisory', choices=('br', 'bl', 'bc', 'none'), default='none')
    c.add_argument('--filaments', default=None, help='filament library JSON (default: measured library)')
    c.add_argument('--max-spools', type=int, default=5)
    c.add_argument('--min-spools', type=int, default=None)
    c.add_argument('--seed', type=int, default=0)

    p = ap.add_argument_group('print')
    p.add_argument('--flush-scale', type=float, default=1.0)
    p.add_argument('--min-flush', type=float, default=None)
    p.add_argument('--tower-fit', choices=('error', 'warn', 'auto'), default='auto')
    p.add_argument('--verify', choices=('auto', 'yes', 'no'), default='auto',
                   help='re-rasterise the written 3MF and compare with the voxel model')
    p.add_argument('--keep-intermediate', action='store_true')
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.check_deps:
        from core.relief.depth_anything import DEFAULT_MODEL_ID, dependency_status, weights_cached
        from core.relief.segment import DEFAULT_SAM_MODEL_ID, sam_status
        st = dependency_status()
        cached = weights_cached(args.weights or DEFAULT_MODEL_ID) if all(st.values()) else False
        sam_ok, sam_msg = sam_status(args.sam_weights or DEFAULT_SAM_MODEL_ID)
        print(json.dumps({'depth_anything': st, 'default_model': DEFAULT_MODEL_ID, 'weights_cached': cached,
                          'ready': all(st.values()) and cached,
                          'hint': (None if all(st.values()) and cached else
                                   'pip install torch torchvision transformers' if not all(st.values()) else
                                   'scripts/convert_relief.py --fetch-depth-model'),
                          'sam': {'model': args.sam_weights or DEFAULT_SAM_MODEL_ID, 'ready': sam_ok, 'status': sam_msg,
                                  'hint': None if sam_ok else 'scripts/convert_relief.py --fetch-sam-model'}}, indent=2))
        return 0
    if args.fetch_sam_model:
        from core.relief.provider import MissingDependencyError
        from core.relief.segment import DEFAULT_SAM_MODEL_ID, fetch_sam_weights
        try:
            mid = fetch_sam_weights(args.sam_weights or DEFAULT_SAM_MODEL_ID)
        except MissingDependencyError as e:
            print(f"[RELIEF] {e}", file=sys.stderr)
            return 3
        print(f"[RELIEF] SAM weights ready: {mid}")
        return 0
    if args.fetch_depth_model:
        from core.relief.depth_anything import DEFAULT_MODEL_ID, fetch_weights
        from core.relief.provider import MissingDependencyError
        try:
            mid = fetch_weights(args.weights or DEFAULT_MODEL_ID)
        except MissingDependencyError as e:
            print(f"[RELIEF] {e}", file=sys.stderr)
            return 3
        print(f"[RELIEF] Depth Anything weights ready: {mid}")
        return 0

    from core.relief.depth_processing import DepthParams
    from core.relief.pipeline import convert_album_relief
    from core.relief.provider import GeometryError, MissingDependencyError
    from core.relief.relief_stack import ReliefDims
    from core.stack5.metric import parse_hue_params

    if int(args.color_layers) != 5:
        ap.error("--color-layers must be 5: Stack5 recipes are five-layer stacks (5^5 LUT)")
    if abs(float(args.layer_height) - 0.08) > 1e-9:
        ap.error("--layer-height must be 0.08: the Stack5 LUT, the measured filament TDs and the "
                 "'Stack5 0.08mm @BBL X2D' process preset are all calibrated for 0.08 mm layers")
    if float(args.width) <= 0 or float(args.calibration_mm) <= 0:
        ap.error("--width and --calibration-mm must be > 0")
    if float(args.relief_mm) < 0 or float(args.base_mm) <= 0:
        ap.error("--relief-mm must be >= 0 and --base-mm > 0")
    if args.geometry == 'depth-anything' and args.depth_map:
        print("[RELIEF] warning: --depth-map is ignored with --geometry depth-anything (the model predicts depth)")
    try:
        dims = ReliefDims(base_mm=args.base_mm, relief_mm=args.relief_mm, layer_h=args.layer_height,
                          color_layers=args.color_layers)
        params = DepthParams(invert=args.invert_depth, normalize=args.depth_normalize,
                             low_percentile=args.depth_low_percentile, high_percentile=args.depth_high_percentile,
                             smoothing_px=args.depth_smoothing, smoothing_range=args.depth_smoothing_range,
                             median_px=args.median_px, flatten_background=args.flatten_background,
                             background_level=args.background_level, max_neighbor_step=args.max_neighbor_step,
                             neighbor_connectivity=args.neighbor_connectivity,
                             protect_silhouette=not args.no_protect_silhouette,
                             min_feature_px=args.min_feature_px, step_layers=args.step_layers)
    except ValueError as e:
        ap.error(str(e))

    image = args.image
    depth_map = args.depth_map
    edits = None
    if args.edits:
        from core.relief.art_direct import EditScript
        try:
            edits = EditScript.load(args.edits)
        except (OSError, ValueError, KeyError) as e:
            ap.error(f"--edits {args.edits}: {e}")
        if not image and edits.meta.get('image') and os.path.isfile(edits.meta['image']):
            image = edits.meta['image']
            print(f"[RELIEF] image taken from the edit script: {image}")
        print(f"[RELIEF] art direction: {len(edits)} edit(s) from {os.path.abspath(args.edits)}")
        for line in edits.summary():
            print(f"[RELIEF]   {line}")
    palette = None if args.palette == 'auto' else _split(args.palette)
    width = args.width
    if args.calibration:
        from core.relief.calibration import CALIB_PALETTE, make_calibration_assets
        cal_dir = os.path.dirname(os.path.abspath(args.output)) if args.output else os.path.join(args.out, 'relief_calibration')
        assets = make_calibration_assets(cal_dir, size_px=int(round(args.calibration_mm * 10)),
                                         relief_mm=args.relief_mm, base_mm=args.base_mm,
                                         layer_h=args.layer_height, color_layers=args.color_layers)
        image, depth_map = assets['image'], assets['depth_map']
        width = args.calibration_mm
        if palette is None:
            palette = list(CALIB_PALETTE)
        if args.geometry != 'imported-depth':
            print("[RELIEF] calibration uses its own depth map (imported-depth)")
        args.geometry = 'imported-depth'
        print(f"[RELIEF] calibration assets: {assets['image']}  {assets['depth_map']}")
        for n in assets['notes']:
            print(f"[RELIEF]   {n}")
    if not image:
        ap.error("an image is required (or --calibration)")
    if args.geometry == 'imported-depth' and not depth_map:
        ap.error("--geometry imported-depth needs --depth-map")

    provider_options = {}
    if args.geometry == 'depth-anything':
        provider_options = {'weights': args.weights, 'allow_download': args.allow_download, 'device': args.device}
    elif args.geometry == 'wonder3d':
        provider_options = {'front_axis': args.front_axis, 'up_axis': args.up_axis, 'render_px': args.render_px,
                            'fit': args.fit, 'depth_map_convention': args.depth_map_convention}

    colour_kw = dict(palette=palette, backing=args.backing, must_include=_split(args.must_include),
                     prefer=_split(args.prefer), quantize_colors=args.quantize, smooth_sigma=args.smooth_sigma,
                     metric=args.metric, hue_params=parse_hue_params(args.hue_params) if args.hue_params else None,
                     min_region_px=args.min_region_px, advisory=None if args.advisory == 'none' else args.advisory,
                     filaments_json=args.filaments, max_spools=args.max_spools, min_spools=args.min_spools)
    try:
        res = convert_album_relief(image, width_mm=width, geometry=args.geometry, depth_map=depth_map,
                                   mesh=args.mesh, normals=args.normals, normals_y_up=args.normals_y_up,
                                   foreground_mask=args.foreground_mask, provider_options=provider_options,
                                   dims=dims, depth_params=params, depth_weight=args.depth_weight,
                                   normal_weight=args.normal_weight, smoothness_weight=args.smoothness_weight,
                                   out_dir=args.out, output=args.output, title=args.title, verify=args.verify,
                                   keep_intermediate=args.keep_intermediate, flush_scale=args.flush_scale,
                                   min_flush=args.min_flush, tower_fit=args.tower_fit, seed=args.seed,
                                   height_edits=edits, edit_clean_px=args.edit_clean_px, **colour_kw)
    except MissingDependencyError as e:
        print(f"[RELIEF] missing optional dependency:\n{e}", file=sys.stderr)
        return 3
    except GeometryError as e:
        print(f"[RELIEF] geometry error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"[RELIEF] input file not found: {e}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, RuntimeError) as e:
        print(f"[RELIEF] error: {e}", file=sys.stderr)
        return 2

    d = res['dims']
    st = res['stats']
    print(f"[RELIEF] palette ({len(res['palette'])} spools): {', '.join(res['palette'])}  backing/body: {res['backing']}")
    for line in res['ams']:
        print(f"[RELIEF]   {line}")
    print(f"[RELIEF] thickness: base {d['base_mm']:g} mm ({d['base_layers']} layers) + relief 0..{st['relief_layers_max'] * d['layer_h_mm']:.2f} mm "
          f"({st['relief_layers_max']} of {d['relief_layers_max']} layers) + shell {d['shell_mm']:g} mm -> total "
          f"{st['total_height_mm']:.2f} mm ({st['total_print_layers']} layers, budget {d['total_mm_max']:g})")
    gr = res['geometry_report']
    for w in gr.get('warnings', []):
        print(f"[RELIEF] warning: {w}")
    print(f"[RELIEF] relief levels used: {gr['stats']['levels_used']}, max neighbour step "
          f"{gr['stats']['neighbor_steps'].get('max_step_layers', 0)} layers")
    ad = res.get('art_direction')
    if ad:
        print(f"[RELIEF] art direction: {ad['n_ops']} edit(s), {ad['pixels_changed']} px changed "
              f"({100 * ad['share_changed']:.1f} %), relief {ad['relief_mm_after'][0]:.2f}..{ad['relief_mm_after'][1]:.2f} mm; "
              f"edits saved: {res['edits_json']}")
    print(f"[RELIEF] voxel check: {'ok' if res['voxel_check']['ok'] else 'FAILED'}; 3MF verification: "
          f"{'ok' if res['verification'] and res['verification']['ok'] else ('skipped' if res['verification'] is None else 'FAILED')}")
    tw = res['tower']
    print(f"[RELIEF] prime tower: {tw['width']:g} x {tw['depth_modelled']:.0f} mm ({'fits' if tw['fits'] else 'DOES NOT FIT'}), "
          f"tool changes {st['tool_changes_planned']}, purge {st['purge_total_mm3']} mm3")
    print(f"[RELIEF] triangles: {st['triangles_total']:,} in {st['n_objects']} objects")
    print(f"[RELIEF] 3mf     : {res['threemf']}")
    print(f"[RELIEF] preview : {res['preview_png']}")
    print(f"[RELIEF] depth   : {res['depth_png']}")
    print(f"[RELIEF] recipe  : {res['recipe_json']}")
    if args.json:
        out = {k: v for k, v in res.items() if k not in ('steps', 'voxels')}
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
