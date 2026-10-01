#!/usr/bin/env python3
"""Generate deterministic single-colour-per-layer Graduation-reference relief."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.band.reference import AUTO_MAX_SPOOLS, DEFAULT_CONTRAST, convert_reference_band

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--width',type=float,default=200)
    ap.add_argument('--pitch',type=float,default=.2,help='mesh spacing mm (default .2)')
    ap.add_argument('--backing',type=float,choices=[0,1],default=1,
                    help='1 mm backing or 0 to retain reference thin floor')
    ap.add_argument('--palette',default='auto',
                    help="auto (default) = pick the spools from the library; 'graduation' = the reference's "
                         "own colours and swaps; or 2-5 library filaments, e.g. 'Black,Klein Blue,White'")
    ap.add_argument('--filaments',default=None,help='filament library JSON (default assets/filaments_user_measured.json)')
    ap.add_argument('--order',choices=['auto','fixed'],default='auto',
                    help='auto stacks the palette dark -> light; fixed keeps the order given (bottom -> top)')
    ap.add_argument('--min-spools',type=int,default=2,help='auto palette: fewest spools (default 2)')
    ap.add_argument('--max-spools',type=int,default=AUTO_MAX_SPOOLS,
                    help=f'auto palette: most spools (default {AUTO_MAX_SPOOLS}, one AMS)')
    ap.add_argument('--tone',choices=['auto','reference'],default=None,
                    help="auto = stretch the cover's brightness over the whole relief (default for spools); "
                         "reference = the raw Graduation rule (default for --palette graduation)")
    ap.add_argument('--contrast',type=float,default=DEFAULT_CONTRAST,
                    help=f'auto tone: 0 = levels only, 1 = full histogram equalisation (default {DEFAULT_CONTRAST})')
    ap.add_argument('--min-feature',type=float,default=.45,
                    help='remove raised features thinner than this, mm (0 = raw tone rule; default .45)')
    args=ap.parse_args()
    palette=args.palette.strip()
    if palette not in ('auto','graduation'):
        palette=[n.strip() for n in palette.split(',') if n.strip()]
    res=convert_reference_band(args.image,args.out,args.width,args.backing,args.pitch,
                               palette=palette,filaments_json=args.filaments,order=args.order,
                               min_feature_mm=args.min_feature,tone=args.tone,contrast=args.contrast,
                               min_spools=args.min_spools,max_spools=args.max_spools)
    print(json.dumps(res,indent=2,default=str))
