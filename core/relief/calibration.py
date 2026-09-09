"""First physical test: a small calibration relief (default 60 x 60 mm).

Generates a synthetic RGB subject and a matching depth map so the printer,
filaments and settings can be checked before a full album cover:

* background     : Beige, relief 0 (base + shell only = 1.2 mm)
* bottom band    : eleven grey steps, relief 0 -> 4.0 mm left to right
                   (0, 5, 10, ... 50 layers) - measure them with calipers
* face disc      : Oak, a dome rising to the full relief in the centre
* eyes           : Black, slightly raised above the dome
* mouth          : Red arc, slightly recessed into the dome
* highlight      : White dot on the forehead

The RGB image only uses flat spool colours so the Stack5 recipe is mostly pure
stacks (few tool changes); use ``--palette Black,White,Red,Oak,Beige``.
Depth is authored black = low, white = high (imported-depth convention).
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

CALIB_COLOURS = {
    'Beige': (0xE8, 0xDC, 0xC5), 'Oak': (0xB0, 0x8A, 0x5C), 'Black': (0x00, 0x00, 0x00),
    'Red': (0xC1, 0x2E, 0x1F), 'White': (0xFF, 0xFF, 0xFF),
}
CALIB_PALETTE = ['Black', 'White', 'Red', 'Oak', 'Beige']


def make_calibration_assets(out_dir: str, size_px: int = 600, stem: str = 'relief_calibration',
                            relief_mm: float = 4.0, base_mm: float = 0.8, layer_h: float = 0.08,
                            color_layers: int = 5) -> dict:
    """Write <stem>.png (RGB subject) and <stem>_depth.png (16-bit grey) into
    out_dir.  ``size_px`` = 10 px/mm x plaque size (600 -> 60 mm).  The notes are
    computed for the given relief / base / layer height."""
    n = int(size_px)
    if n < 20:
        raise ValueError("size_px must be >= 20 (2 mm)")
    os.makedirs(out_dir, exist_ok=True)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    u, v = xx / (n - 1), yy / (n - 1)                     # 0..1, v down
    rgb = np.empty((n, n, 3), np.uint8)
    rgb[:] = CALIB_COLOURS['Beige']
    depth = np.zeros((n, n), np.float64)

    # bottom band: 11 steps of relief (0 .. 1) in equal columns
    band = v >= 0.82
    steps = np.clip(np.floor(u * 11), 0, 10) / 10.0
    depth[band] = steps[band]
    grey = np.round(40 + 175 * steps).astype(np.uint8)   # visible bands in the RGB too
    rgb[band] = np.stack([grey, grey, grey], -1)[band]

    # face: disc centred at (0.5, 0.42), radius 0.30 -> dome
    cx, cy, r = 0.5, 0.42, 0.30
    d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2) / r
    face = d <= 1.0
    dome = np.sqrt(np.clip(1.0 - d ** 2, 0.0, 1.0))
    rgb[face] = CALIB_COLOURS['Oak']
    depth[face] = dome[face]

    # eyes: two discs, +6 % relief above the dome
    for ex in (0.40, 0.60):
        eye = np.sqrt((u - ex) ** 2 + (v - 0.36) ** 2) <= 0.045
        rgb[eye] = CALIB_COLOURS['Black']
        depth[eye] = np.clip(depth[eye] + 0.06, 0, 1)
    # mouth: arc (ring segment), -8 % into the dome
    md = np.sqrt((u - cx) ** 2 + (v - 0.44) ** 2)
    mouth = (md >= 0.14) & (md <= 0.175) & (v > 0.50)
    rgb[mouth] = CALIB_COLOURS['Red']
    depth[mouth] = np.clip(depth[mouth] - 0.08, 0, 1)
    # highlight dot
    hl = np.sqrt((u - 0.5) ** 2 + (v - 0.20) ** 2) <= 0.03
    rgb[hl] = CALIB_COLOURS['White']

    img_path = os.path.join(out_dir, f"{stem}.png")
    dep_path = os.path.join(out_dir, f"{stem}_depth.png")
    Image.fromarray(rgb, 'RGB').save(img_path)
    Image.fromarray(np.round(depth * 65535).astype(np.uint16)).save(dep_path)
    lh = float(layer_h)
    base = round(float(base_mm) / lh) * lh
    relief = round(float(relief_mm) / lh) * lh
    shell = int(color_layers) * lh
    return {'image': img_path, 'depth_map': dep_path, 'palette': list(CALIB_PALETTE),
            'size_px': n, 'width_mm': n / 10.0,
            'notes': [f"bottom band: 11 relief steps 0..{relief:g} mm left to right ({relief / 10:.2f} mm apart)",
                      f"face dome reaches the full relief in the centre; eyes +{0.06 * relief:.2f} mm, "
                      f"mouth -{0.08 * relief:.2f} mm",
                      f"expected thickness: {base + shell:.2f} mm background, {base + relief + shell:.2f} mm at the dome centre"]}
