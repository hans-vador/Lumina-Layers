"""Bambu Studio flush volumes, tool ordering and prime-tower sizing for Stack5.

Everything here is a port of the Bambu Studio logic the produced 3MF is sliced
with (sources: src/libslic3r/FlushVolCalc.cpp, GUI/Plater.cpp
get_min_flush_volumes, libslic3r/ToolOrdering.cpp "Auto For Flush",
libslic3r/GCode/WipeTower.cpp plan_toolchange/plan_tower_new, GUI/PartPlate.cpp
set_default_wipe_tower_pos_for_plate):

* ``bambu_flush_volume``  - FlushVolCalculator::calc_flush_vol_rgb (HSV distance +
  luminance term, third edge of a 120 deg triangle, floor 60) + the per-nozzle
  minimum (nozzle_volume minus the long-retraction-when-cut volume), capped at
  g_max_flush_volume = 900.  Bambu's X2D profile normally consults a data-driven
  predictor first; that model is not public, the RGB formula is its documented
  fallback and reproduces the 500-600 mm3 dark->light purges it is known for.
* ``plan_layer_orders``   - per printed layer, the filament order that minimises
  the flush given the previous layer's last filament (what "Auto For Flush"
  does for <= 5 filaments), i.e. the purge the wipe tower must absorb per layer.
* ``tower_depth``         - WipeTower::plan_toolchange: every purge is laid down
  as lines of width w_perimeter = nozzle * 1.25 across the tower's inner width;
  depth = sum(ceil(length / inner_width) * gap) (+ one perimeter width), with the
  rib wall OFF so the tower is a plain width x depth rectangle and
  prime_tower_infill_gap 100 % (gap = line width).
* ``plan_layout``         - plaque at the left bed edge (centred in Y), tower in
  the right strip inside Bambu's WIPE_TOWER_MARGIN clamp; fails loudly when the
  modelled depth does not fit.

Numbers are deterministic functions of the palette hex colours and the
per-layer material sets, so they are recorded in the recipe.
"""
from __future__ import annotations

import colorsys
import itertools
import math
from typing import Sequence

import numpy as np

# --- Bambu Studio constants ------------------------------------------------- #
G_MAX_FLUSH_VOLUME = 900            # FlushVolCalc.cpp g_max_flush_volume
FLUSH_FLOOR = 60.0                  # calc_flush_vol_rgb: std::max(flush_volume, 60.f)
WIDTH_TO_NOZZLE_RATIO = 1.25        # WipeTower: m_perimeter_width = nozzle_diameter * ratio
BBS_WIPE_TOWER_MARGIN = 15.0        # PartPlate.hpp WIPE_TOWER_MARGIN: x in [margin, bed - w - margin - brim]
FILAMENT_DIAMETER = 1.75

# --- Stack5 plate layout ---------------------------------------------------- #
BED_MM = 256.0                      # X2D printable_area 0x0..256x256
PLAQUE_EDGE_MM = 3.0                # plaque min corner from the bed edge (skirt_distance 2 + 0.5 line)
SKIRT_OUT_MM = 2.5                  # skirt outer edge beyond the plaque
TOWER_CLEARANCE_MM = 1.0            # between the skirt and the tower (brim)
TOWER_BRIM_MM = 0.0                 # prime_tower_brim_width: a 2 mm tall slab needs none
TOWER_INFILL_GAP_PCT = 100          # prime_tower_infill_gap: solid purge lines
TOWER_MIN_WIDTH_MM = 20.0
TOWER_SAFETY_MM = 2.0               # modelled depth must leave this much before the bed edge


def hex_to_rgb8(h: str) -> tuple[int, int, int]:
    s = str(h).strip().lstrip('#')
    if len(s) != 6:
        raise ValueError(f"bad hex colour {h!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _luminance(r: float, g: float, b: float) -> float:
    return r * 0.3 + g * 0.59 + b * 0.11


def _delta_hs(h1, s1, v1, h2, s2, v2) -> float:
    a1, a2 = math.radians(h1), math.radians(h2)
    dx = math.cos(a1) * s1 * v1 - math.cos(a2) * s2 * v2
    dy = math.sin(a1) * s1 * v1 - math.sin(a2) * s2 * v2
    return min(1.2, math.hypot(dx, dy))


def bambu_flush_rgb(src: Sequence[int], dst: Sequence[int]) -> float:
    """FlushVolCalculator::calc_flush_vol_rgb (mm3, before minimum and cap)."""
    sr, sg, sb = (c / 255.0 for c in src)
    dr, dg, db = (c / 255.0 for c in dst)
    fh, fs, fv = colorsys.rgb_to_hsv(sr, sg, sb)
    th, ts, tv = colorsys.rgb_to_hsv(dr, dg, db)
    hs = _delta_hs(fh * 360.0, fs, fv, th * 360.0, ts, tv)
    fl, tl = _luminance(sr, sg, sb), _luminance(dr, dg, db)
    if tl >= fl:
        lumi = (tl - fl) ** 0.7 * 560.0
    else:
        lumi = (fl - tl) * 80.0
        hs = min(0.67 * tv + 0.33 * fv, hs)
    hs_flush = 230.0 * hs
    third = math.sqrt(max(hs_flush ** 2 + lumi ** 2 - 2 * hs_flush * lumi * math.cos(math.radians(120.0)), 0.0))
    return max(third, FLUSH_FLOOR)


def bambu_flush_volume(src_hex: str, dst_hex: str, min_flush: float = 0.0, scale: float = 1.0,
                       max_flush: int = G_MAX_FLUSH_VOLUME) -> int:
    """Purge volume src -> dst in mm3: (rgb formula + min_flush) * scale, capped."""
    if src_hex.upper() == dst_hex.upper():
        return 0
    v = (bambu_flush_rgb(hex_to_rgb8(src_hex), hex_to_rgb8(dst_hex)) + float(min_flush)) * float(scale)
    return int(min(int(v), int(max_flush)))


def min_flush_from_config(cfg: dict) -> float:
    """Plater.cpp get_min_flush_volumes for nozzle 0 / filament 0 of a project
    config: nozzle_volume minus the long-retraction-when-cut filament volume
    (pi d^2/4 * retract) when the machine level enables it per filament."""
    def first(key, default):
        v = cfg.get(key, default)
        if isinstance(v, list):
            v = v[0] if v else default
        return v
    nozzle_volume = float(first('nozzle_volume', 0) or 0)
    level = int(float(first('enable_long_retraction_when_cut', 0) or 0))
    fil_on = int(float(first('filament_long_retractions_when_cut', 0) or 0))
    machine_on = int(float(first('long_retractions_when_cut', 0) or 0))
    retract = 0.0
    if fil_on == 1 and level == 2:                          # EnableFilament
        retract = float(first('filament_retraction_distances_when_cut', 18) or 18)
    elif fil_on and level and machine_on:
        retract = float(first('retraction_distances_when_cut', 18) or 18)
    extra = nozzle_volume - math.pi * FILAMENT_DIAMETER ** 2 / 4.0 * retract
    return float(int(max(extra, 0.0)))                      # Bambu keeps it as int


def flush_matrix(hexes: Sequence[str], min_flush: float = 0.0, scale: float = 1.0,
                 max_flush: int = G_MAX_FLUSH_VOLUME) -> np.ndarray:
    """(F,F) int matrix, [from, to]."""
    F = len(hexes)
    M = np.zeros((F, F), dtype=np.int64)
    for i in range(F):
        for j in range(F):
            if i != j:
                M[i, j] = bambu_flush_volume(hexes[i], hexes[j], min_flush, scale, max_flush)
    return M


def flatten_matrix(M: np.ndarray, n_blocks: int = 2) -> list[str]:
    """Bambu flush_volumes_matrix: n_blocks copies of the row-major F*F matrix."""
    row = [str(int(v)) for v in np.asarray(M).reshape(-1)]
    return row * int(n_blocks)


# --------------------------------------------------------------------------- #
# Tool ordering ("Auto For Flush") and per-layer purge
# --------------------------------------------------------------------------- #
def _best_order(mats: Sequence[int], M: np.ndarray, start: int | None) -> tuple[list[int], list[int]]:
    """Order of ``mats`` minimising the flush from ``start`` (None = free start).
    Exhaustive for <= 7 materials, greedy nearest-neighbour beyond."""
    mats = sorted(int(m) for m in mats)
    if not mats:
        return [], []
    if start is not None and start in mats and len(mats) == 1:
        return mats, [0]
    best = None
    if len(mats) <= 7:
        for perm in itertools.permutations(mats):
            purges = []
            prev = start
            for m in perm:
                purges.append(int(M[prev, m]) if (prev is not None and prev != m) else 0)
                prev = m
            tot = sum(purges)
            key = (tot, perm)
            if best is None or key < best[0]:
                best = (key, list(perm), purges)
        return best[1], best[2]
    order, purges, prev, left = [], [], start, list(mats)
    while left:
        if prev is None:
            m = left[0]
        else:
            m = min(left, key=lambda x: (int(M[prev, x]) if x != prev else 0, x))
        purges.append(int(M[prev, m]) if (prev is not None and prev != m) else 0)
        order.append(m)
        left.remove(m)
        prev = m
    return order, purges


def plan_layer_orders(layer_sets: Sequence[Sequence[int]], M: np.ndarray) -> list[dict]:
    """Per printed layer: {'materials', 'order', 'purges', 'total_mm3'}; the first
    filament of a layer is reached from the previous layer's last one."""
    plan = []
    last = None
    for j, mats in enumerate(layer_sets):
        mats = sorted(set(int(m) for m in mats))
        if not mats:
            plan.append({'layer': j, 'materials': [], 'order': [], 'purges': [], 'total_mm3': 0})
            continue
        order, purges = _best_order(mats, M, last)
        n_changes = sum(1 for k, m in enumerate(order) if k > 0 or (last is not None and m != last))
        plan.append({'layer': j, 'materials': mats, 'order': order, 'purges': purges,
                     'total_mm3': int(sum(purges)), 'n_changes': int(n_changes)})
        last = order[-1]
    return plan


# --------------------------------------------------------------------------- #
# Tower geometry
# --------------------------------------------------------------------------- #
def tower_depth(purges_per_layer: Sequence[Sequence[float]], width: float, layer_h: float,
                nozzle: float = 0.4, infill_gap_pct: float = TOWER_INFILL_GAP_PCT) -> dict:
    """WipeTower depth for a rectangular (rib wall off) tower of ``width``."""
    pw = float(nozzle) * WIDTH_TO_NOZZLE_RATIO
    inner = float(width) - 2.0 * pw
    if inner <= 0:
        raise ValueError("tower width too small")
    gap = pw * float(infill_gap_pct) / 100.0
    per_layer = []
    for purges in purges_per_layer:
        d = 0.0
        for v in purges:
            if v <= 0:
                continue
            length = float(v) / (pw * float(layer_h))          # volume_to_length
            d += math.ceil(length / inner) * gap
        per_layer.append(d)
    depth = (max(per_layer) if per_layer else 0.0) + pw
    return {'depth_mm': depth, 'per_layer_depth_mm': per_layer, 'perimeter_width_mm': pw,
            'inner_width_mm': inner, 'line_gap_mm': gap}


def plan_layout(plaque_w: float, plaque_h: float, bed: float = BED_MM, brim: float = TOWER_BRIM_MM,
                plaque_edge: float = PLAQUE_EDGE_MM) -> dict:
    """Plaque at the left edge (Y centred), tower in the right strip.  All values
    in plate coordinates (mm); the tower position is its front-left corner."""
    tx = float(plaque_edge)
    ty = max(float(plaque_edge), (float(bed) - float(plaque_h)) / 2.0)
    if tx + plaque_w + SKIRT_OUT_MM > bed or ty + plaque_h + SKIRT_OUT_MM > bed:
        raise ValueError(f"plaque {plaque_w:g}x{plaque_h:g} mm (+skirt) does not fit the {bed:g} mm bed")
    tower_x = tx + float(plaque_w) + SKIRT_OUT_MM + float(brim) + TOWER_CLEARANCE_MM
    tower_x = math.ceil(tower_x * 2) / 2.0
    tower_x = max(tower_x, BBS_WIPE_TOWER_MARGIN)
    tower_w = math.floor((bed - BBS_WIPE_TOWER_MARGIN - float(brim) - tower_x) * 2) / 2.0
    tower_y = max(BBS_WIPE_TOWER_MARGIN, float(brim))
    depth_avail = bed - tower_y - float(brim) - TOWER_SAFETY_MM
    return {'bed_mm': float(bed), 'plaque_xy': [round(tx, 4), round(ty, 4)],
            'plaque_size': [float(plaque_w), float(plaque_h)],
            'tower_x': tower_x, 'tower_y': tower_y, 'tower_width': tower_w,
            'tower_depth_available': depth_avail, 'tower_brim': float(brim),
            'bbs_margin': BBS_WIPE_TOWER_MARGIN}


def plan_tower(hexes: Sequence[str], layer_sets: Sequence[Sequence[int]], plaque_w: float,
               plaque_h: float, layer_h: float, min_flush: float = 0.0, scale: float = 1.0,
               nozzle: float = 0.4, bed: float = BED_MM, brim: float = TOWER_BRIM_MM,
               infill_gap_pct: float = TOWER_INFILL_GAP_PCT) -> dict:
    """Flush matrix + per-layer purge plan + tower footprint + fit verdict."""
    M = flush_matrix(hexes, min_flush, scale)
    orders = plan_layer_orders(layer_sets, M)
    lay = plan_layout(plaque_w, plaque_h, bed, brim)
    if lay['tower_width'] < TOWER_MIN_WIDTH_MM:
        raise ValueError(f"only {lay['tower_width']:g} mm left for the prime tower beside a "
                         f"{plaque_w:g} mm plaque on the {bed:g} mm bed")
    td = tower_depth([o['purges'] for o in orders], lay['tower_width'], layer_h, nozzle, infill_gap_pct)
    fits = td['depth_mm'] <= lay['tower_depth_available']
    per_layer_total = [o['total_mm3'] for o in orders]
    worst = int(np.argmax(per_layer_total)) if per_layer_total else 0
    capacity = (lay['tower_depth_available'] - td['perimeter_width_mm']) * td['inner_width_mm'] * float(layer_h) \
        / (float(infill_gap_pct) / 100.0)
    return {
        'flush_matrix': M.tolist(),
        'flush_min_mm3': float(min_flush),
        'flush_scale': float(scale),
        'flush_max_mm3': G_MAX_FLUSH_VOLUME,
        'flush_formula': 'BambuStudio FlushVolCalc::calc_flush_vol_rgb + per-nozzle minimum, cap 900',
        'ordering': 'per layer: permutation minimising flush from the previous layer\'s last filament (Auto For Flush)',
        'layers': orders,
        'purge_per_layer_mm3': per_layer_total,
        'purge_total_mm3': int(sum(per_layer_total)),
        'worst_layer': worst,
        'tool_changes': int(sum(o['n_changes'] for o in orders)),
        'tower': {
            'x': lay['tower_x'], 'y': lay['tower_y'], 'width': lay['tower_width'],
            'depth_modelled': round(td['depth_mm'], 3),
            'depth_available': round(lay['tower_depth_available'], 3),
            'margin_mm': round(lay['tower_depth_available'] - td['depth_mm'], 3),
            'capacity_mm3_per_layer': round(capacity, 1),
            'brim': float(brim), 'infill_gap_pct': float(infill_gap_pct), 'rib_wall': False,
            'perimeter_width': td['perimeter_width_mm'], 'inner_width': td['inner_width_mm'],
            'per_layer_depth': [round(d, 3) for d in td['per_layer_depth_mm']],
            'fits': bool(fits),
        },
        'layout': lay,
    }


def tower_config_keys(plan: dict) -> dict:
    """project_settings.config overrides for the planned tower."""
    t = plan['tower']
    return {
        'enable_prime_tower': '1',
        'prime_tower_rib_wall': '0',
        'prime_tower_infill_gap': f"{int(round(t['infill_gap_pct']))}%",
        'prime_tower_width': f"{t['width']:g}",
        'prime_tower_brim_width': f"{t['brim']:g}",
        'wipe_tower_x': [f"{t['x']:g}", f"{t['x']:g}"],
        'wipe_tower_y': [f"{t['y']:g}", f"{t['y']:g}"],
        'wipe_tower_rotation_angle': '0',
        'flush_volumes_matrix': flatten_matrix(np.array(plan['flush_matrix']), 2),
    }
