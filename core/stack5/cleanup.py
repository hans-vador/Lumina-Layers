"""Minimum-region cleanup and printability statistics for Stack5 material matrices.

Lumina's own ``isolated_pixel_cleanup`` only removes single pixels whose whole
stack differs from all 8 neighbours.  At 0.1 mm/px a 0.4 mm nozzle cannot
reproduce material islands smaller than ~4x4 px, and every island start costs a
retraction / z-hop / seam, so :func:`min_region_cleanup` removes, per optical
layer and per material, every connected component (4-connectivity: corner
touching pixels slice into separate islands) smaller than ``min_px`` and
refills it from the nearest surviving pixel of that layer (Euclidean distance
transform - deterministic, no RNG).  Repeats until stable (bounded).

:func:`region_stats` reports island counts, the sub-threshold share and the
share removed by a square morphological opening (features thinner than the
nozzle) so the user can trade detail against printability.
"""
from __future__ import annotations

import numpy as np


def _components(binary: np.ndarray, connectivity: int = 4):
    import cv2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=connectivity)
    return n, labels, stats[:, cv2.CC_STAT_AREA]


def region_stats(material_matrix: np.ndarray, mask_solid: np.ndarray, min_px: int = 16,
                 opening_px: int = 4, connectivity: int = 4) -> dict:
    """Per optical layer: islands, pixels in islands < min_px, pixels removed by
    an opening_px square opening (all as counts and shares of the solid area)."""
    import cv2
    mm = np.asarray(material_matrix)
    mask = np.asarray(mask_solid, bool)
    H, W, L = mm.shape
    solid = int(mask.sum()) or 1
    kernel = np.ones((int(opening_px), int(opening_px)), np.uint8)
    layers = []
    for j in range(L):
        img = mm[:, :, j]
        mats = np.unique(img[mask])
        mats = mats[mats >= 0]
        n_isl = 0
        small_px = 0
        thin_px = 0
        per_mat = {}
        for m in mats:
            b = (img == m) & mask
            n, labels, areas = _components(b, connectivity)
            areas = areas[1:]                                   # drop background
            n_isl += int(areas.size)
            s = int(areas[areas < int(min_px)].sum())
            small_px += s
            opened = cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            t = int(b.sum() - opened.sum())
            thin_px += t
            per_mat[int(m)] = {'islands': int(areas.size), 'small_px': s, 'thin_px': t, 'area_px': int(b.sum())}
        layers.append({'layer': j, 'islands': n_isl, 'small_px': small_px, 'small_share': small_px / solid,
                       'thin_px': thin_px, 'thin_share': thin_px / solid, 'per_material': per_mat})
    return {'min_px': int(min_px), 'opening_px': int(opening_px), 'connectivity': int(connectivity),
            'solid_px': solid, 'layers': layers,
            'islands_per_layer': [l['islands'] for l in layers],
            'small_share_per_layer': [l['small_share'] for l in layers],
            'thin_share_per_layer': [l['thin_share'] for l in layers]}


def min_region_cleanup(material_matrix: np.ndarray, mask_solid: np.ndarray, min_px: int = 16,
                       connectivity: int = 4, max_iter: int = 6, protected_mask=None) -> tuple[np.ndarray, dict]:
    """Return (cleaned material matrix (H,W,L), stats).  Pixels outside
    ``mask_solid`` are untouched (they stay whatever they were, normally -1)."""
    from scipy import ndimage
    mm = np.asarray(material_matrix).copy()
    mask = np.asarray(mask_solid, bool)
    protected = np.zeros(mask.shape, bool) if protected_mask is None else np.asarray(protected_mask, bool)
    if protected.shape != mask.shape:
        raise ValueError('protected mask must match the solid mask shape')
    H, W, L = mm.shape
    min_px = int(min_px)
    stats = {'min_px': min_px, 'connectivity': int(connectivity), 'layers': []}
    if min_px <= 1:
        stats['reassigned_px_total'] = 0
        return mm, stats
    total = 0
    for j in range(L):
        img = mm[:, :, j]
        reassigned = 0
        iters = 0
        for it in range(int(max_iter)):
            iters = it + 1
            mats = np.unique(img[mask])
            mats = mats[mats >= 0]
            marked = np.zeros((H, W), bool)
            for m in mats:
                b = (img == m) & mask
                n, labels, areas = _components(b, connectivity)
                small = np.where(areas < min_px)[0]
                small = small[small != 0]
                if small.size:
                    marked |= np.isin(labels, small)
            marked &= ~protected
            if not marked.any():
                break
            src = mask & ~marked
            if not src.any():
                break
            _, (iy, ix) = ndimage.distance_transform_edt(~src, return_indices=True)
            new = img.copy()
            new[marked] = img[iy[marked], ix[marked]]
            changed = int(np.count_nonzero(new[marked] != img[marked]))
            reassigned += int(marked.sum())
            img = new
            if changed == 0:
                break
        mm[:, :, j] = img
        total += reassigned
        stats['layers'].append({'layer': j, 'reassigned_px': reassigned, 'iterations': iters})
    stats['reassigned_px_total'] = int(total)
    stats['reassigned_share'] = float(total / max(int(mask.sum()) * L, 1))
    return mm, stats


def resync_matched_rgb(material_matrix: np.ndarray, mask_solid: np.ndarray, matched_rgb: np.ndarray,
                       lut_rgb: np.ndarray, ref_stacks: np.ndarray) -> np.ndarray:
    """Recompute matched_rgb from the LUT for every solid pixel (stack -> LUT row)."""
    mm = np.asarray(material_matrix)
    mask = np.asarray(mask_solid, bool)
    out = np.asarray(matched_rgb).copy()
    stacks = np.asarray(ref_stacks, dtype=np.int64)
    L = stacks.shape[1]
    base = int(stacks.max()) + 1
    weights = base ** np.arange(L - 1, -1, -1, dtype=np.int64)
    lut_code = stacks @ weights
    order = np.argsort(lut_code, kind='stable')
    codes_sorted = lut_code[order]
    px = mm[mask].astype(np.int64) @ weights
    pos = np.searchsorted(codes_sorted, px)
    pos = np.clip(pos, 0, len(codes_sorted) - 1)
    hit = codes_sorted[pos] == px
    rows = order[pos]
    vals = out[mask]
    vals[hit] = np.asarray(lut_rgb)[rows[hit]]
    out[mask] = vals
    return out
