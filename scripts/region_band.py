#!/usr/bin/env python3
"""Region layers: Reference layers relief with its own spool sequence per colour region."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.band.regions import MAX_REGIONS, MAX_SPOOLS, convert_region_band

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=float, default=200)
    ap.add_argument('--pitch', type=float, default=.2, help='mesh spacing mm (default .2)')
    ap.add_argument('--palette', default='auto',
                    help="auto (default) = pick from the library; or 2-5 library filaments to choose from, "
                         "e.g. 'Black,Klein Blue,Pink,Beige'")
    ap.add_argument('--filaments', default=None, help='filament library JSON (default assets/filaments_user_measured.json)')
    ap.add_argument('--max-regions', type=int, default=MAX_REGIONS,
                    help=f'most colour regions to try (default {MAX_REGIONS}; 1 = Reference layers)')
    ap.add_argument('--max-spools', type=int, default=MAX_SPOOLS, help=f'auto palette size limit (default {MAX_SPOOLS})')
    ap.add_argument('--contrast', type=float, default=0., help='as Reference layers (default 0 = levels only)')
    ap.add_argument('--min-feature', type=float, default=.45, help='as Reference layers (default .45 mm)')
    args = ap.parse_args()
    palette = args.palette.strip()
    if palette != 'auto':
        palette = [n.strip() for n in palette.split(',') if n.strip()]
    res = convert_region_band(args.image, args.out, args.width, args.pitch, palette=palette,
                              filaments_json=args.filaments, max_regions=args.max_regions,
                              max_spools=args.max_spools, contrast=args.contrast, min_feature_mm=args.min_feature)
    print(json.dumps(res, indent=2, default=str))
