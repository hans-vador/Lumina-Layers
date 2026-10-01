"""TD wedge boards: measure transmission distance on the spools you actually own.

Every TD in ``assets/filaments_user*.json`` is inherited from Lumina's bundled
Bambu LUTs or estimated from them, and ``docs/band/PLAN.md`` says outright that
those presets are not transferable.  A wedge board closes that gap: it prints a
staircase of each filament over an opaque base, you photograph it once, and
:func:`fit_td` turns the measured patch colours into a ``td_mm`` per spool.

Why a *dark* base: hiding power is only visible against something the filament
has to hide.  White over white measures nothing; white over black measures
everything.  The board therefore defaults to a Black base.

Layout (face-down - the squished first layer on the plate is the viewing face):

    row = one filament, column m = m layers of it over the base
    column 0 = base only (the reference for the fit)

This is PLAN.md's C2 ("General TD wedge boards"), built on the same voxel ->
mesh -> 3MF path as the boards in ``core/calibration.py``.
"""
from __future__ import annotations

import os
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import trimesh
from PIL import Image

from config import ColorSystem, OUTPUT_DIR, PrinterConfig
from core.band.optics import (DEFAULT_K_OPAQUE, Filament, hex_to_rgb01, srgb_to_linear)
from core.calibration import _generate_voxel_mesh
from utils.bambu_3mf_writer import export_scene_with_bambu_metadata

DEFAULT_MAX_LAYERS = 12
DEFAULT_BASE_MM = 0.8


# --------------------------------------------------------------------------- #
# board generation
# --------------------------------------------------------------------------- #
def wedge_stacks(n_filaments: int, max_layers: int) -> np.ndarray:
    """(rows, cols) layer counts: column 0 is the bare base, column m is m layers."""
    return np.tile(np.arange(0, max_layers + 1, dtype=int), (n_filaments, 1))


def generate_td_wedge_board(filament_names: Sequence[str], library: Mapping[str, Filament],
                            base_name: str = 'Black', max_layers: int = DEFAULT_MAX_LAYERS,
                            base_mm: float = DEFAULT_BASE_MM, block_mm: float = 6.0,
                            gap_mm: float = 1.0, layer_h: float = PrinterConfig.LAYER_HEIGHT,
                            out_path: Optional[str] = None) -> dict:
    """Print-ready wedge board for up to 4 filaments over one opaque base.

    Returns {'threemf', 'preview', 'rows', 'cols', 'filaments', 'base',
             'layer_h', 'max_layers'} - keep ``rows``/``cols`` for :func:`fit_td`.
    """
    names = [n for n in filament_names if n != base_name]
    if not names:
        raise ValueError("need at least one filament besides the base")
    if base_name not in library:
        raise KeyError(f"base {base_name!r} is not in the library")
    missing = [n for n in names if n not in library]
    if missing:
        raise KeyError(f"not in the library: {missing}")
    if len(names) > 4:
        raise ValueError("at most 4 filaments per board (5 AMS slots with the base)")

    rows, cols = len(names), max_layers + 1
    base_layers = max(1, int(round(base_mm / layer_h)))
    total_layers = base_layers + max_layers

    px = max(1, int(round(block_mm / PrinterConfig.NOZZLE_WIDTH)))
    gp = max(1, int(round(gap_mm / PrinterConfig.NOZZLE_WIDTH)))
    pad = 1                                     # one-block border, like the other boards
    grid_r, grid_c = rows + 2 * pad, cols + 2 * pad
    vh, vw = grid_r * (px + gp), grid_c * (px + gp)

    slot_names = [base_name] + names             # material id 0 = base
    matrix = np.zeros((total_layers, vh, vw), dtype=int)   # everything base by default

    for r, name in enumerate(names):
        mat = r + 1
        for c in range(cols):
            m = c                                # column 0 = bare base
            if m == 0:
                continue
            y = (r + pad) * (px + gp)
            x = (c + pad) * (px + gp)
            # face-down: Z=0 is the viewing surface, so the wedge sits on top of it
            matrix[0:m, y:y + px, x:x + px] = mat

    scene = trimesh.Scene()
    preview_colors, used = {}, []
    for mat, name in enumerate(slot_names):
        rgb = np.clip(np.round(hex_to_rgb01(library[name].hex) * 255), 0, 255).astype(int)
        preview_colors[mat] = [int(rgb[0]), int(rgb[1]), int(rgb[2]), 255]
        mesh = _generate_voxel_mesh(matrix, mat, vh, vw)
        if mesh is not None:
            mesh.visual.face_colors = preview_colors[mat]
            mesh.metadata['name'] = name
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            used.append(name)

    out_path = out_path or os.path.join(
        OUTPUT_DIR, f"TD_wedge_{'_'.join(n.replace(' ', '') for n in names)}_over_{base_name}.3mf")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # export_scene_with_bambu_metadata resolves each mesh's colour by looking the
    # slot name up in ColorSystem.get(color_mode)['slots'] and indexing
    # preview_colors by THAT system's material ids.  Passing a stock mode would
    # match our names against RYBW/CMYW and silently fall back to grey, so
    # register a colour system whose ids are ours (same trick as
    # core.band.processor.band_mode_conf).
    mode_key = 'TDWedge:' + '-'.join(n.replace(' ', '') for n in slot_names)
    ColorSystem.register_dynamic(mode_key, {
        'name': 'TDWedge', 'slots': list(slot_names), 'preview': dict(preview_colors),
        'map': {n: i for i, n in enumerate(slot_names)},
        'layer_count': int(max_layers), 'base': len(slot_names),
    })

    export_scene_with_bambu_metadata(
        scene=scene, output_path=out_path, slot_names=used, preview_colors=preview_colors,
        color_mode=mode_key,
        settings={
            'layer_height': str(layer_h),
            'initial_layer_print_height': str(layer_h),   # the real Bambu key
            'initial_layer_height': str(layer_h),         # legacy alias, harmless
            'wall_loops': '1', 'top_shell_layers': '0', 'bottom_shell_layers': '0',
            'sparse_infill_density': '100%', 'sparse_infill_pattern': 'zig-zag',
        })

    top = matrix[0].astype(np.uint8)
    prev = np.zeros((vh, vw, 3), dtype=np.uint8)
    for mat, rgba in preview_colors.items():
        prev[top == mat] = rgba[:3]

    return {
        'threemf': out_path, 'preview': Image.fromarray(prev),
        'rows': rows, 'cols': cols, 'filaments': names, 'base': base_name,
        'layer_h': float(layer_h), 'max_layers': int(max_layers),
        'board_mm': (grid_c * block_mm + (grid_c - 1) * gap_mm,
                     grid_r * block_mm + (grid_r - 1) * gap_mm),
    }


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_td(colors_over_base: Sequence, full, base, layer_h: float = PrinterConfig.LAYER_HEIGHT,
           k_opaque: float = DEFAULT_K_OPAQUE, lo: float = 0.05, hi: float = 0.95) -> dict:
    """TD (mm) from one measured wedge row.

    ``colors_over_base[m-1]`` is the patch with m layers over the base; ``full``
    is the filament's own colour at full thickness and ``base`` the bare base.
    Each patch gives an opacity by projecting it onto the base->full axis
    (PLAN.md C2):

        o_m = <c_m - base, d> / |d|^2,   d = full - base

    The TD is then the one that best reproduces the whole row at once,

        TD = argmin  SUM_m [ o_m - (1 - exp(-k * m * lh / TD)) ]^2

    rather than the average of the per-patch inversions
    ``-k*m*lh / ln(1 - o_m)``.  Inverting each patch separately looks simpler
    but is badly behaved: ``ln(1 - o)`` explodes as a patch approaches opacity,
    so for an opaque filament - which saturates after two or three layers, and
    therefore has only two or three usable patches - a couple of camera counts
    of noise on the near-saturated patch swings the answer by over 100%.
    Fitting in the opacity domain keeps every residual bounded, so saturated
    patches contribute what they actually know ("already opaque by here")
    instead of dominating.

    ``lo``/``hi`` still bracket the *informative* band, and are used to report
    a bound when no patch lands inside it.
    """
    def lin(x):
        x = np.asarray(x, dtype=np.float64)
        if x.max() > 1.0 + 1e-9:
            x = x / 255.0
        return srgb_to_linear(x)

    base_l, full_l = lin(base), lin(full)
    d = full_l - base_l
    denom = float(d @ d)
    if denom < 1e-9:
        raise ValueError("base and full colours are identical - nothing to fit")

    per_layer, thick, obs = [], [], []
    for i, c in enumerate(colors_over_base):
        m = i + 1
        o = float((lin(c) - base_l) @ d) / denom
        per_layer.append({'layers': m, 'thickness_mm': round(m * layer_h, 4),
                          'opacity': round(o, 4)})
        thick.append(m * layer_h)
        obs.append(min(max(o, 0.0), 1.0))

    usable = [p for p in per_layer if lo < p['opacity'] < hi]
    if usable:
        t = np.asarray(thick, dtype=np.float64)
        y = np.asarray(obs, dtype=np.float64)

        def sse(td: float) -> float:
            return float(np.sum((y - (1.0 - np.exp(-k_opaque * t / td))) ** 2))

        # coarse log sweep, then golden-section refine - deterministic, no scipy
        grid = np.geomspace(0.02, 50.0, 400)
        i0 = int(np.argmin([sse(g) for g in grid]))
        a, b = grid[max(i0 - 1, 0)], grid[min(i0 + 1, len(grid) - 1)]
        phi = (np.sqrt(5.0) - 1.0) / 2.0
        c1, c2 = b - phi * (b - a), a + phi * (b - a)
        f1, f2 = sse(c1), sse(c2)
        for _ in range(80):
            if f1 < f2:
                b, c2, f2 = c2, c1, f1
                c1 = b - phi * (b - a)
                f1 = sse(c1)
            else:
                a, c1, f1 = c1, c2, f2
                c2 = a + phi * (b - a)
                f2 = sse(c2)
        td = float((a + b) / 2.0)
        resid = float(np.sqrt(sse(td) / len(y)))
        out = {'td_mm': round(td, 3), 'td_source': 'fitted', 'n_points': len(usable),
               'rms_opacity_residual': round(resid, 4), 'per_layer': per_layer}
        if len(usable) < 3:
            # A very opaque filament saturates within a layer or two, so the
            # wedge only ever sees one or two informative steps and the fit is
            # poorly constrained no matter how it is done.  Say so rather than
            # implying three decimal places of confidence.
            out['confidence'] = 'low'
            out['note'] = (f"only {len(usable)} patch(es) landed between {lo} and {hi} opacity - "
                           f"this filament is near-opaque at {layer_h} mm, so treat the value as "
                           f"'about {td:.1f} mm or less'.")
        return out

    tds = []
    if not tds:
        # everything saturated (opaque) or nothing moved (transparent): bound it
        finished = [p for p in per_layer if p['opacity'] >= hi]
        if finished:
            t = min(p['thickness_mm'] for p in finished)
            return {'td_mm': round(-k_opaque * t / np.log(1.0 - hi), 3),
                    'td_source': 'fitted_upper_bound', 'n_points': 0,
                    'note': f"opaque by {t} mm - TD is at most this; print thinner steps",
                    'per_layer': per_layer}
        t = max(p['thickness_mm'] for p in per_layer)
        return {'td_mm': round(-k_opaque * t / np.log(1.0 - lo), 3),
                'td_source': 'fitted_lower_bound', 'n_points': 0,
                'note': f"still translucent at {t} mm - TD is at least this; print thicker steps",
                'per_layer': per_layer}

    td = float(np.mean(tds))
    return {'td_mm': round(td, 3), 'td_source': 'fitted', 'n_points': len(tds),
            'td_spread_mm': (round(float(np.min(tds)), 3), round(float(np.max(tds)), 3)),
            'per_layer': per_layer}


def fit_board(patches: np.ndarray, board: Mapping, **kw) -> Dict[str, dict]:
    """Fit every row of a photographed board.

    ``patches`` is (rows, cols, 3) sampled sRGB, matching the board's layout:
    column 0 is the bare base and column m is m layers.
    """
    p = np.asarray(patches, dtype=np.float64)
    rows, cols = int(board['rows']), int(board['cols'])
    if p.shape[:2] != (rows, cols):
        raise ValueError(f"expected patches {(rows, cols, 3)}, got {p.shape}")
    out = {}
    for r, name in enumerate(board['filaments']):
        out[name] = fit_td(p[r, 1:], full=p[r, -1], base=p[r, 0],
                           layer_h=float(board['layer_h']), **kw)
    return out


def apply_fits(library_path: str, fits: Mapping[str, Mapping], out_path: Optional[str] = None) -> str:
    """Write fitted TDs back into a filament library JSON."""
    from core.band.optics import load_filament_library, save_filament_library
    lib = load_filament_library(library_path)
    for name, fit in fits.items():
        if name in lib:
            f = lib[name]
            lib[name] = Filament.from_hex(f.name, f.hex, float(fit['td_mm']),
                                          str(fit.get('td_source', 'fitted')),
                                          f.settings_id, f.filament_id)
    return save_filament_library(out_path or library_path, lib)
