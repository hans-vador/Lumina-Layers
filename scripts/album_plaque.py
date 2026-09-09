#!/usr/bin/env python
"""Headless CLI: album cover(s) -> HueForge-style Band-mode plaque 3MF.

Examples
  .venv/bin/python scripts/album_plaque.py --image art/astroworld.jpg --width 133.33 \
      --palette 'Black,Red,Sunny Orange,White' --order fixed
  .venv/bin/python scripts/album_plaque.py --dir art/ --palette auto --advisory br
  .venv/bin/python scripts/album_plaque.py --dir art/ --palette global   # one AMS order for all
"""

from __future__ import annotations

import argparse
import glob
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
    files = []
    for p in sorted(glob.glob(os.path.join(args.dir, '*'))):
        if p.lower().endswith(IMAGE_EXTS):
            files.append(p)
    if not files:
        sys.exit(f"no images found in {args.dir}")
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', help='single image file')
    src.add_argument('--dir', help='directory of images (batch)')
    ap.add_argument('--width', type=float, default=150.0, help='plaque width in mm (default 150)')
    ap.add_argument('--palette', default='auto',
                    help="'auto' (per album), 'global' (one order for all images), or a comma list "
                         "of filament names bottom->top, e.g. 'Black,Red,Sunny Orange,White'")
    ap.add_argument('--order', choices=('auto', 'fixed'), default='auto',
                    help='auto = sort a given palette by luminance (darkest base); fixed = keep as given')
    ap.add_argument('--layers', default=None,
                    help="fixed per-band layer counts (base first) matching --palette, e.g. '9,4,6,8'; "
                         "skips the thickness optimiser (reference Astroworld = 9,4,6,8)")
    ap.add_argument('--pitch', type=float, default=0.15, help='mesh XY pitch in mm (default 0.15)')
    ap.add_argument('--no-engrave', action='store_true', help='disable base-band relief engraving')
    ap.add_argument('--advisory', choices=('br', 'bl', 'bc', 'none'), default='none',
                    help='stamp a binary PARENTAL ADVISORY label in a bottom corner')
    ap.add_argument('--out', default=os.path.join(REPO, 'output'), help='output directory')
    ap.add_argument('--filaments', default=os.path.join(REPO, 'assets', 'filaments_user.json'))
    ap.add_argument('--n-filaments', default='4,5',
                    help="palette sizes the auto optimiser may choose, e.g. '4,5' (default; picks 4 when "
                         "cost4 <= 1.05*cost5) or '5' to force five filaments")
    ap.add_argument('--n0', type=int, default=9, help='base band layers (default 9 -> first swap 0.88 mm)')
    ap.add_argument('--max-layers', type=int, default=27)
    ap.add_argument('--quantize', type=int, default=256, help='K-Means colours (default 256)')
    ap.add_argument('--smooth-sigma', type=float, default=10.0, help='bilateral sigma (default 10)')
    ap.add_argument('--no-thumbnail', action='store_true', help='do not embed the preview as plate thumbnail')
    ap.add_argument('--ramp-dir', default=None,
                    help='where to save the ramp LUT .npz (default lut-npy预设/Custom so Lumina lists it)')
    ap.add_argument('--luma-weight', type=float, default=5.0,
                    help='matcher lightness weight (1 = plain Lab distance; default 5 = luminance-dominant, '
                         'HueForge-like tonal matching)')
    ap.add_argument('--k-opaque', type=float, default=None,
                    help='Beer-Lambert steepness o=1-exp(-k*t/TD) (default optics.DEFAULT_K_OPAQUE=7: '
                         '99.9%% opaque at t=TD)')
    ap.add_argument('--height-mode', choices=('luminance', 'match'), default='luminance',
                    help="'luminance' (default): HueForge-style tone mapping L* -> height (reproduces the "
                         "reference surface distribution); 'match': Lumina nearest-ramp-colour matcher")
    ap.add_argument('--tone', default='reference',
                    help="tone curve for --height-mode luminance: 'reference' (fitted to the HueForge "
                         "reference, default), 'linear' (L* 0 -> base, 100 -> top) or 'q_lo,q_hi,gamma'")
    ap.add_argument('--engrave-floor', type=float, default=None,
                    help='relief floor in mm (default 0.521 = reference relief minimum; contract value 0.48)')
    args = ap.parse_args(argv)

    from core.band.optics import BeerLambertModel, load_filament_library
    from core.band.pipeline import _assignment_kwargs, convert_album_to_band_3mf
    from core.band.tone import ToneCurve

    images = _collect_images(args)
    library = load_filament_library(args.filaments)
    n_filaments = tuple(int(b) for b in str(args.n_filaments).split(',') if b.strip())
    if not n_filaments or any(b < 1 or b > 5 for b in n_filaments):
        sys.exit("--n-filaments must list palette sizes between 1 and 5, e.g. '4,5' or '5'")
    if args.n0 < 1 or args.n0 >= args.max_layers:
        sys.exit("--n0 must be >= 1 and < --max-layers")
    try:
        tone = ToneCurve.from_any(args.tone)
    except Exception as exc:
        sys.exit(f"--tone: {exc}")
    palette_arg = args.palette.strip()
    palette_names = None
    global_order = None
    if palette_arg.lower() == 'global':
        from core.band import palette as pal
        hists = [pal.image_hist_lab(p) for p in images]
        assign = _assignment_kwargs(args.height_mode, args.luma_weight, tone)
        global_order = tuple(pal.select_global_order(hists, library, BeerLambertModel(args.k_opaque),
                                                     n_filaments=n_filaments,
                                                     n0=args.n0, max_layers=args.max_layers, **assign))
        print(f"[BAND] global filament order (bottom->top): {', '.join(global_order)}")
    elif palette_arg.lower() != 'auto':
        palette_names = [p.strip() for p in palette_arg.split(',') if p.strip()]

    fixed_schedule = None
    if args.layers:
        from core.band.schedule import BandSchedule
        counts = tuple(int(c) for c in args.layers.split(',') if c.strip())
        names = tuple(global_order) if global_order is not None else tuple(palette_names or ())
        if not names or len(names) != len(counts):
            sys.exit("--layers needs an explicit --palette (or 'global') with the same number of entries")
        missing = [n for n in names if n not in library]
        if missing:
            sys.exit(f"unknown filament(s) {missing}; available: {sorted(library)}")
        if args.order != 'fixed' and global_order is None:
            # sort filament/count PAIRS by luminance so each count stays with its filament
            pairs = sorted(zip(names, counts), key=lambda nc: library[nc[0]].luminance)
            names, counts = tuple(n for n, _ in pairs), tuple(c for _, c in pairs)
        fixed_schedule = BandSchedule(names, counts, base_min_layers=min(5, counts[0]))
        print(f"[BAND] fixed schedule: {fixed_schedule}")

    failures = 0
    for img in images:
        print("=" * 78)
        print(f"[BAND] {img}")
        common = dict(
            width_mm=args.width, filaments_json=args.filaments, schedule=fixed_schedule,
            mesh_pitch_mm=args.pitch, engrave=not args.no_engrave, out_dir=args.out,
            advisory=None if args.advisory == 'none' else args.advisory,
            quantize_colors=args.quantize, smooth_sigma=args.smooth_sigma,
            embed_thumbnail=not args.no_thumbnail, ramp_dir=args.ramp_dir,
            luma_weight=args.luma_weight, k_opaque=args.k_opaque,
            n0=args.n0, max_layers=args.max_layers, n_filaments=n_filaments,
            height_mode=args.height_mode, tone=tone, engrave_floor_mm=args.engrave_floor,
        )
        try:
            if global_order is not None:
                res = convert_album_to_band_3mf(img, palette=list(global_order), order='fixed', **common)
            else:
                res = convert_album_to_band_3mf(img, palette=palette_names, order=args.order, **common)
        except Exception as exc:  # keep batch going
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"[BAND] FAILED: {img}: {exc}")
            continue

        st = res['stats']
        print(f"[BAND] 3MF:      {res['threemf']}")
        print(f"[BAND] preview:  {res['preview_png']}")
        print(f"[BAND] schedule: {res['schedule']['filament_names']} layers {res['schedule']['layer_counts']} "
              f"(K={res['schedule']['n_layers']}, {st.get('tool_changes')} changes, "
              f"mean dE {st.get('mean_dE', 0):.2f}, mean |dL*| {st.get('mean_dL', 0):.2f}, "
              f"optical {st.get('optical_model')}, schedule search {st.get('schedule_seconds', 0):.1f}s)")
        print(f"[BAND] height mode: {st.get('height_mode')} tone {st.get('tone_curve')}; "
              f"heights {st.get('height_min_mm', 0):.3f}..{st.get('height_max_mm', 0):.3f} mm, "
              f"columns on a slice plane: {st.get('slice_plane_hits')}")
        per_fil = st.get('top_surface_fraction_per_filament', {})
        if per_fil:
            print("[BAND] top-surface share per filament: " +
                  ", ".join(f"{k} {v * 100:.1f}%" for k, v in per_fil.items()))
        per_layer = st.get('top_surface_fraction_per_layer', {})
        if per_layer:
            print("[BAND] top-surface share per layer: " +
                  " ".join(f"L{int(n)} {float(v) * 100:.1f}" for n, v in sorted(per_layer.items(), key=lambda kv: int(kv[0]))
                           if float(v) > 0))
        print(f"[BAND] mesh: {st.get('mesh_vertices')} verts / {st.get('mesh_faces')} faces, "
              f"watertight {st.get('mesh_watertight')}, extent {st.get('mesh_extent_mm')}")
        rep = st.get('palette_report')
        if rep:
            print(f"[BAND] palette optimiser: {rep.get('search')}; predicted weighted mean dE "
                  f"{rep.get('mean_dE', 0):.2f} (wL {rep.get('wL')}), cost {rep.get('cost', 0):.2f}, "
                  f"band pixel shares " + ", ".join(f"{n} {s * 100:.1f}%" for n, s in
                                                    zip(rep.get('filament_names', []), rep.get('band_shares', [])))
                  + (f", {rep.get('stage2_evaluations')} stage-2 evaluations" if rep.get('stage2_evaluations') else '')
                  + (f", {rep.get('seconds', 0):.1f}s" if rep.get('seconds') else ''))
            for B, alt in (rep.get('alternatives') or {}).items():
                print(f"[BAND]   {B}-filament best: {alt['filament_names']} layers {alt['layer_counts']} "
                      f"cost {alt['cost']:.2f} mean dE {alt['mean_dE']:.2f}")
        print("[BAND] swap instructions:")
        for line in res['swap_instructions'].splitlines():
            print("    " + line)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
