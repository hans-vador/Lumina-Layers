#!/usr/bin/env python
"""Headless CLI: album cover(s) -> Stack5 3MF (Lumina native 5-layer stacks, 5 of
the user's 12 spools per album, Bambu Lab X2D project settings).

Examples
  .venv/bin/python scripts/lumina_album.py --image art/graduation.jpg --width 150 --advisory br
  .venv/bin/python scripts/lumina_album.py --image art/graduation.jpg --palette 'Black,White,Pink,Red,Klein Blue'
  .venv/bin/python scripts/lumina_album.py --dir art/ --must-include Pink --structure double
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

# ---- environment shims (same as main.py:24-28) -----------------------------
setattr(np, "asscalar", lambda a: a.item())
os.environ.setdefault("LUMINA_COLOR_RECIPE_POLICY", "off")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.heic')


def _collect_images(args) -> list[str]:
    if args.image:
        return [args.image]
    files = [p for p in sorted(glob.glob(os.path.join(args.dir, '*'))) if p.lower().endswith(IMAGE_EXTS)]
    if not files:
        sys.exit(f"no images found in {args.dir}")
    return files


def _split(s: str | None) -> list[str]:
    return [x.strip() for x in (s or '').split(',') if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', help='single image file')
    src.add_argument('--dir', help='directory of images (batch)')
    ap.add_argument('--width', type=float, default=150.0, help='plaque width in mm (default 150)')
    ap.add_argument('--palette', default='auto',
                    help="'auto' (exhaustive search over all 5-subsets of the library) or a comma list of "
                         "up to 5 filament names in AMS slot order, e.g. 'Black,White,Pink,Red,Klein Blue'")
    ap.add_argument('--must-include', default='',
                    help="comma list of filaments the auto search must keep, e.g. 'Pink'")
    ap.add_argument('--prefer', default='',
                    help="comma list of filaments to prefer (soft must-include): used when the best subset "
                         "containing them exceeds the optimum's cost by at most --prefer-tol x the optimum's "
                         "weighted dE, e.g. 'Pink'")
    ap.add_argument('--prefer-tol', type=float, default=0.10,
                    help='tolerance for --prefer as a fraction of the optimum\'s weighted dE (default 0.10)')
    ap.add_argument('--backing', default=None,
                    help='backing filament name (default: searched - every palette member is scored as the '
                         'backing and the lowest-cost configuration wins)')
    ap.add_argument('--quantize', type=int, default=96, help='K-Means colours for Lumina HiFi (default 96)')
    ap.add_argument('--smooth-sigma', type=float, default=10.0, help='bilateral sigma (default 10)')
    ap.add_argument('--structure', choices=('single', 'double'), default='single')
    ap.add_argument('--spacer', type=float, default=1.0, help='backing thickness in mm (default 1.0; face-up snaps to complete layers)')
    ap.add_argument('--orientation', choices=['face-down', 'face-up'], default='face-down',
                    help='face-up puts the thick first layer in the backing')
    ap.add_argument('--layer-height', type=float, choices=[0.08, 0.04], default=0.08,
                    help='colour thickness; 0.04 is experimental and below the X2D profile minimum')
    ap.add_argument('--no-early-stop', action='store_true',
                    help='keep all five colour layers in face-up jobs (default: stop at best match)')
    ap.add_argument('--first-layer', type=float, default=0.20,
                    help='first layer in mm (default 0.20): backing when face-up, viewing layer when face-down')
    ap.add_argument('--advisory', choices=('br', 'bl', 'bc', 'none'), default='none',
                    help='stamp a PARENTAL ADVISORY label into a bottom corner before processing')
    ap.add_argument('--out', default=os.path.join(REPO, 'output', 'stack5'), help='output root directory')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed for cv2/numpy (default 0)')
    ap.add_argument('--filaments', default=None,
                    help='filament library JSON (default: assets/filaments_user_measured.json if present, '
                         'else assets/filaments_user.json)')
    ap.add_argument('--metric', choices=('hue', 'lumina', 'lab'), default='hue',
                    help="colour-matching metric for BOTH the palette search and the per-pixel LUT match: "
                         "'hue' (default) = hue-first LCh (right hue even if the shade is off; see --hue-params); "
                         "'lumina' = Lumina's OpenCV 8-bit Lab Euclidean (lightness weighted 6.5x); "
                         "'lab' = true CIELAB weighted by --wL")
    ap.add_argument('--hue-params', default='',
                    help="tune --metric hue, e.g. 'wL=0.6,wH=2,wC=0.6,wN=1.5,h0=8,h1=60' (wL = lightness weight for "
                         "chromatic colours, wLn = for neutrals, wH = hue-shift weight with ramp h0..h1 degrees, "
                         "wC = chroma weight, wN = colour-loss (going grey) weight, knee = C* where a colour counts "
                         "as fully chromatic, c_vis/c_keep = chroma thresholds)")
    ap.add_argument('--wL', type=float, default=1.0,
                    help="lightness weight in the palette-search dE, only for --metric lab (default 1)")
    ap.add_argument('--min-region-px', type=int, default=16,
                    help='remove per-layer material islands smaller than this many pixels (0.1 mm/px; '
                         'default 16 = 0.4x0.4 mm, the smallest island a 0.4 nozzle prints; 0 = off)')
    ap.add_argument('--keep-dilation', action='store_true',
                    help="keep Lumina's 1-px mask dilation before meshing (parts then overlap by 0.1 mm and "
                         "Bambu Studio carves them in extruder order; default off)")
    ap.add_argument('--flush-scale', type=float, default=1.0,
                    help='multiply the Bambu-formula flush volumes written to the 3MF (default 1.0)')
    ap.add_argument('--min-flush', type=float, default=None,
                    help='per-nozzle minimum purge in mm3 added to every flush volume (default: from the X2D '
                         'template, nozzle_volume minus the long-retraction volume = 48)')
    ap.add_argument('--max-spools', type=int, default=5, help='largest palette the search may use (default 5)')
    ap.add_argument('--min-spools', type=int, default=None,
                    help='smallest palette the search may use (default 2; use 5 to force exactly five spools)')
    ap.add_argument('--spool-penalty', type=float, default=0.75,
                    help='cost every extra spool must earn back in weighted colour error (default 0.75)')
    ap.add_argument('--need-share', type=float, default=0.02,
                    help="a hue in the art needs this chroma-weighted pixel share to earn its nearest spool (default 0.02)")
    ap.add_argument('--backing-layer', type=float, default=0.2,
                    help='layer height for the backing layers via a Bambu height-range modifier '
                         '(default 0.2; colour layers stay 0.08; 0 = everything at 0.08)')
    ap.add_argument('--tower-fit', choices=('error', 'warn', 'auto'), default='auto',
                    help='when the modelled prime tower does not fit beside the plaque: fail (default) or warn')
    ap.add_argument('--hist-k', type=int, default=64, help='histogram bins for the palette search (default 64)')
    ap.add_argument('--chroma-weight', type=float, default=0.0,
                    help='bin weight multiplier 1 + c*C*/50 in the palette search (default 0 = off)')
    ap.add_argument('--spool-bonus', type=float, default=10.0,
                    help='bonus (dE per unit weight) for dominant colours within --spool-de of a PURE stack, '
                         'i.e. an exact spool colour (default 10; 0 = off)')
    ap.add_argument('--spool-de', type=float, default=10.0, help='dE radius of an exact spool match (default 10)')
    ap.add_argument('--dominant-w', type=float, default=0.01,
                    help='histogram weight above which a colour counts as dominant (default 0.01)')
    ap.add_argument('--k-opaque', type=float, default=None, help='Beer-Lambert steepness (default optics.DEFAULT_K_OPAQUE)')
    ap.add_argument('--td-scale', type=float, default=1.0,
                    help='multiply every library TD (optical calibration; library values are HueForge-style '
                         'guesses, Lumina\'s measured RYBW LUT implies far more opaque pigmented spools)')
    ap.add_argument('--td', default='',
                    help="per-filament TD overrides in mm, e.g. 'Pink=1.5,White=8' (applied after --td-scale)")
    ap.add_argument('--lut', default=None,
                    help="measured .npz LUT {rgb, stacks} for the given --palette (slot order of the printed "
                         "board) to use instead of the synthetic Beer-Lambert LUT")
    ap.add_argument('--title', default=None, help='plate name (default: image name)')
    ap.add_argument('--keep-lumina-output', action='store_true',
                    help="keep Lumina's original 3mf in output/ (default: removed after post-processing)")
    ap.add_argument('--json', action='store_true', help='also print the result dict as JSON')
    args = ap.parse_args(argv)

    from core.stack5.pipeline import convert_album_stack5
    from core.stack5.metric import parse_hue_params
    hue_params = parse_hue_params(args.hue_params) if args.hue_params else None

    images = _collect_images(args)
    palette = None if args.palette.strip().lower() == 'auto' else _split(args.palette)
    must = _split(args.must_include)
    prefer = _split(args.prefer)
    td_overrides = {}
    for item in _split(args.td):
        if '=' not in item:
            sys.exit(f"--td expects NAME=mm entries, got {item!r}")
        k, v = item.split('=', 1)
        td_overrides[k.strip()] = float(v)
    failures = 0
    for img in images:
        print("=" * 78)
        print(f"[STACK5] {img}")
        try:
            res = convert_album_stack5(
                img, width_mm=args.width, filaments_json=args.filaments, palette=palette,
                backing=args.backing, quantize_colors=args.quantize, smooth_sigma=args.smooth_sigma,
                structure=args.structure, spacer_mm=args.spacer, first_layer_mm=args.first_layer,
                orientation=args.orientation, layer_h=args.layer_height, early_stop=not args.no_early_stop,
                advisory=None if args.advisory == 'none' else args.advisory, out_dir=args.out,
                title=args.title, seed=args.seed, must_include=must,
                wL=args.wL, hist_k=args.hist_k, k_opaque=args.k_opaque,
                keep_lumina_output=args.keep_lumina_output, td_scale=args.td_scale,
                td_overrides=td_overrides, lut_npz=args.lut, chroma_weight=args.chroma_weight,
                spool_bonus=args.spool_bonus, spool_de=args.spool_de, dominant_w=args.dominant_w,
                prefer=prefer, prefer_tol=args.prefer_tol, metric=args.metric, hue_params=hue_params,
                max_spools=args.max_spools, min_spools=args.min_spools,
                spool_penalty=args.spool_penalty, need_share=args.need_share,
                min_region_px=args.min_region_px, dilate=args.keep_dilation,
                flush_scale=args.flush_scale, min_flush=args.min_flush, tower_fit=args.tower_fit,
                backing_layer_h=(args.backing_layer if args.backing_layer and args.backing_layer > 0 else None))
        except Exception as exc:  # keep batch going
            failures += 1
            print(f"[STACK5] FAILED: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
            continue
        st = res['stats']
        print("-" * 78)
        print(f"[STACK5] palette (slot order): {', '.join(res['palette'])}  ({len(res['palette'])} spools)")
        pr_ = res.get('palette_report') or {}
        hp_ = pr_.get('hue_plan')
        if hp_:
            parts = [f"{c['name']} h{c['hue']:.0f} {100 * c['share']:.0f}% -> {c['pick'] or 'mix'}"
                     for c in hp_['clusters'] if c['needed']]
            print("[STACK5] art hues -> spools: " + ('; '.join(parts) or 'none (neutral art)')
                  + f"; light neutrals {100 * hp_['light_neutral_share']:.0f}%, dark {100 * hp_['dark_share']:.0f}%"
                  + f" -> required {pr_.get('required_spools')}")
        if pr_.get('excluded_spools') is not None:
            sh = (hp_ or {}).get('hue_present_share') or {}
            print("[STACK5] spools whose colour is not in this art (pixel share within 25 deg of their hue): "
                  + (', '.join(f"{n_} {100 * sh.get(n_, 0.0):.1f}%" for n_ in pr_['excluded_spools']) or 'none'))
            if pr_.get('sizes_evaluated'):
                print("[STACK5] palette sizes tried: " + ', '.join(f"{k} spools x{v}" for k, v in pr_['sizes_evaluated'].items()))
        print(f"[STACK5] backing: {res['backing']} (slot {res['backing_slot']})")
        print(f"[STACK5] colour metric: {res.get('pixel_matcher') or res['palette_report'].get('metric')}")
        pr = (res.get('palette_report') or {}).get('prefer') or {}
        if pr.get('requested'):
            ex = pr.get('excess_dE')
            ex_txt = 'n/a' if ex is None else '%.3f dE = %.1f%%' % (ex, 100 * pr['excess_ratio'])
            print("[STACK5] prefer %s: %s (rank %s, cost excess %s, tolerance %.0f%% = %.3f dE)" % (
                pr['requested'], 'applied' if pr.get('applied') else 'NOT applied', pr.get('rank'), ex_txt,
                100 * pr.get('tolerance', 0), pr.get('tolerance_dE', 0)))
        bu = (res.get('palette_report') or {}).get('best_unconstrained')
        if bu and bu.get('filament_names') != res['palette']:
            print(f"[STACK5] unconstrained optimum: {', '.join(bu['filament_names'])} (cost {bu['cost']:.3f} vs {st['palette_cost']:.3f})")
        for line in res['ams']:
            print(f"[STACK5] {line}")
        print(f"[STACK5] unique stacks used: {st['n_unique_stacks_used']} "
              f"(pure-stack pixels {100 * st['pure_stack_pixel_share']:.1f}%)")
        print(f"[STACK5] per-slot layer share: " + ', '.join(f"{k} {100 * v:.1f}%" for k, v in st['per_slot_layer_share'].items()))
        print(f"[STACK5] palette cost {st['palette_cost']:.3f} (weighted dE {st['palette_cost_dE']:.3f}; dominant pixels on exact spool colour {100 * st['dominant_exact_spool_share']:.1f}%)")
        print(f"[STACK5] mean dE predicted {st['mean_dE_predicted_palette']:.2f}"
              + (f", matched {st['mean_dE_matched_quantized_vs_lut']:.2f}" if st.get('mean_dE_matched_quantized_vs_lut') is not None else ''))
        print(f"[STACK5] objects: {st['n_objects']}, triangles: {st['triangles_total']:,} {st['triangles_per_object']}")
        stop = st.get('early_stopping', {})
        if stop.get('enabled'):
            print(f"[STACK5] early stopping: {100 * stop['optical_material_saved_fraction']:.1f}% fewer optical voxels; "
                  f"pixels at 0..5 layers: {stop['pixels_by_colour_layer_count']}")
        print(f"[STACK5] tool changes (est.): {st['tool_changes_est']} "
              f"(+layer transitions: {st['tool_changes_est_with_layer_transitions']}) over {st['total_print_layers']} layers")
        tw = st['tower']
        print(f"[STACK5] purge per layer (Bambu formula, min {res['flush']['flush_min_mm3']:g} mm3, "
              f"scale {res['flush']['flush_scale']:g}): worst {max(st['purge_per_layer_mm3'])} mm3, "
              f"total {st['purge_total_mm3']} mm3 over {st['tool_changes_planned']} changes")
        print(f"[STACK5] prime tower: {tw['width']:g} x {tw['depth_modelled']:.1f} mm at ({tw['x']:g}, {tw['y']:g}), "
              f"{tw['depth_available']:.1f} mm available ({'fits' if tw['fits'] else 'DOES NOT FIT'}, "
              f"margin {tw['margin_mm']:.1f} mm); plaque at {res['layout']['plaque_xy']}")
        if st.get('islands_per_optical_layer') is not None:
            print(f"[STACK5] islands per optical layer: {st['islands_per_optical_layer']} "
                  f"(before cleanup {st['islands_per_optical_layer_before_cleanup']}); sub-0.4 mm share "
                  + ', '.join(f"{100 * v:.1f}%" for v in st['sub_nozzle_share_per_optical_layer'])
                  + f"; {st['cleanup_reassigned_px']} px reassigned")
        print(f"[STACK5] runtime: {res['timing']['total_s']:.1f} s "
              f"(palette {res['timing']['palette_s']:.1f}, lumina {res['timing']['lumina_s']:.1f}, post {res['timing']['postprocess_s']:.1f})")
        print(f"[STACK5] 3mf:     {res['threemf']}")
        print(f"[STACK5] preview: {res['preview_png']}")
        print(f"[STACK5] recipe:  {res['recipe_json']}")
        print(f"[STACK5] ams:     {res['ams_txt']}")
        print(f"[STACK5] lut:     {res['lut_npz']}")
        if args.json:
            print(json.dumps(res, indent=2, default=str))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
