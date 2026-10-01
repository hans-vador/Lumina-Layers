#!/usr/bin/env python
"""Launch Band Studio - the drag-and-drop filament-painting editor.

  .venv/bin/python scripts/band_studio.py --image art/cover.jpg
  .venv/bin/python scripts/band_studio.py --port 7871 --library assets/filaments_user.json

Drag filament swatches onto the vertical layer bar (bottom = printed first),
type TDs in place, then "Auto-divide layers" to let the optimiser spend the
layer budget where the image has detail.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

setattr(np, "asscalar", lambda a: a.item())
os.environ.setdefault("LUMINA_COLOR_RECIPE_POLICY", "off")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=7871)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--image', default=None, help='open this image on start')
    ap.add_argument('--width', type=float, default=150.0, help='plaque width mm')
    ap.add_argument('--library', default=None)
    args = ap.parse_args(argv)

    import uvicorn
    from core.band import studio_api

    if args.library:
        studio_api.STUDIO = studio_api.Studio(args.library)
    if args.image:
        studio_api.STUDIO.open_image(os.path.abspath(args.image), args.width)
        print(f"[studio] opened {args.image}")

    print(f"[studio] http://{args.host}:{args.port}")
    uvicorn.run(studio_api.create_app(), host=args.host, port=args.port, log_level='warning')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
