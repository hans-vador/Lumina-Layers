#!/usr/bin/env python
"""TD wedge boards: measure transmission distance on your own spools.

  # 1. print this (Black base + up to 4 filaments, one AMS slot each)
  .venv/bin/python scripts/td_wedge.py board --filaments 'White,Lavender Purple,Beige,Klein Blue'

  # 2. photograph it flat, evenly lit, then sample the patches into a CSV
  #    (rows = filaments in the order above, cols = 0..12 layers, 'R G B' per cell)
  .venv/bin/python scripts/td_wedge.py fit --patches patches.csv \\
      --filaments 'White,Lavender Purple,Beige,Klein Blue' --apply

Column 0 of every row is the bare base - it is the reference the fit needs, so
sample it too.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

setattr(np, "asscalar", lambda a: a.item())
os.environ.setdefault("LUMINA_COLOR_RECIPE_POLICY", "off")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_LIB = os.path.join(REPO, 'assets', 'filaments_user_measured.json')


def _split(s):
    return [x.strip() for x in str(s).split(',') if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    b = sub.add_parser('board', help='generate a printable wedge board')
    b.add_argument('--filaments', required=True, help='up to 4 names, comma separated')
    b.add_argument('--base', default='Black', help='opaque base filament (default Black)')
    b.add_argument('--max-layers', type=int, default=12)
    b.add_argument('--block', type=float, default=6.0, help='patch size mm (default 6)')
    b.add_argument('--library', default=DEFAULT_LIB)
    b.add_argument('--out', default=None)

    f = sub.add_parser('fit', help='fit TDs from sampled patch colours')
    f.add_argument('--patches', required=True, help='CSV of sampled sRGB, one row per filament')
    f.add_argument('--filaments', required=True)
    f.add_argument('--base', default='Black')
    f.add_argument('--max-layers', type=int, default=12)
    f.add_argument('--layer-h', type=float, default=0.08)
    f.add_argument('--library', default=DEFAULT_LIB)
    f.add_argument('--apply', action='store_true', help='write fitted TDs into the library')
    f.add_argument('--out-library', default=None)

    args = ap.parse_args(argv)
    from core.band.calibration import apply_fits, fit_board, generate_td_wedge_board
    from core.band.optics import load_filament_library
    lib = load_filament_library(args.library)

    if args.cmd == 'board':
        r = generate_td_wedge_board(_split(args.filaments), lib, base_name=args.base,
                                    max_layers=args.max_layers, block_mm=args.block,
                                    out_path=args.out)
        png = os.path.splitext(r['threemf'])[0] + '_layout.png'
        r['preview'].save(png)
        print(f"3MF     : {r['threemf']}")
        print(f"layout  : {png}")
        print(f"board   : {r['board_mm'][0]:.1f} x {r['board_mm'][1]:.1f} mm")
        print(f"grid    : {r['rows']} rows x {r['cols']} cols (col 0 = bare {r['base']})")
        print(f"AMS     : slot 1 = {r['base']} (base), then "
              + ', '.join(f"slot {i + 2} = {n}" for i, n in enumerate(r['filaments'])))
        print("\nPrint face-down at 0.08 mm, then photograph it flat and evenly lit.")
        return 0

    names = _split(args.filaments)
    rows = [ln for ln in open(args.patches, encoding='utf-8').read().splitlines() if ln.strip()]
    patches = np.array([[[float(v) for v in cell.split()] for cell in ln.split(',')]
                        for ln in rows], dtype=float)
    board = {'rows': len(names), 'cols': args.max_layers + 1, 'filaments': names,
             'base': args.base, 'layer_h': args.layer_h}
    fits = fit_board(patches, board)

    for name, fit in fits.items():
        old = lib[name].td_mm if name in lib else float('nan')
        flag = '  [low confidence]' if fit.get('confidence') == 'low' else ''
        print(f"{name:18s} TD {fit['td_mm']:>7.3f} mm   (library said {old:g})"
              f"   {fit['n_points']} pts{flag}")
        if fit.get('note'):
            print(f"{'':20s}{fit['note']}")
    if args.apply:
        p = apply_fits(args.library, fits, args.out_library)
        print(f"\nwritten: {p}")
    else:
        print(f"\n{json.dumps({k: v['td_mm'] for k, v in fits.items()}, indent=2)}")
        print("(re-run with --apply to write these into the library)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
