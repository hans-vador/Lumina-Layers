"""Heightfield -> watertight trimesh (band mode geometry).

Vectorised port of ``band_part`` from Downloads/BambuAlbums/tools/rebuild3.py
(L204-230) with ``lo = 0``: a single continuous solid whose top surface is the
height field, whose bottom is flat at z = 0 and whose boundary is closed by
vertical walls.  Vertex layout (image convention -> print convention):

    x = c * pitch
    y = (H - 1 - r) * pitch      # image row 0 is at max y => reads correctly face-up
    z = height_mm[r, c]          # top grid;  z = 0 for the bottom grid

Top: 2 triangles per cell (normals +z).  Bottom: 2 triangles per cell
(normals -z).  Walls: 2 triangles per boundary edge (normals outward).
The result is watertight with consistent outward winding (positive volume).
"""
from __future__ import annotations

import numpy as np
import trimesh


def heightfield_to_trimesh(height_mm: np.ndarray, pitch_mm: float,
                           mask: np.ndarray | None = None) -> trimesh.Trimesh:
    """Convert a (H, W) height field in mm to a closed heightfield solid.

    Args:
        height_mm: (H, W) float array of top-surface heights (mm).  Every
            height used by an emitted cell must be > 0 (the bottom is z = 0).
        pitch_mm: XY spacing between adjacent samples (mm).
        mask: optional (H, W) bool array.  A cell (r, c) is emitted only if all
            four of its corner samples are True.  None => full rectangle.

    Returns:
        trimesh.Trimesh (process=False) that is watertight, winding-consistent
        and has outward normals; ``vertices[:, 2].min() == 0``.
    """
    h = np.asarray(height_mm, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 2:
        raise ValueError(f"height_mm must be (H>=2, W>=2), got {h.shape}")
    pitch = float(pitch_mm)
    if not pitch > 0:
        raise ValueError("pitch_mm must be > 0")
    H, W = h.shape
    B = H * W  # offset of the bottom vertex grid

    # --- cell inclusion ----------------------------------------------------
    if mask is None:
        inc = np.ones((H - 1, W - 1), dtype=bool)
    else:
        m = np.asarray(mask, dtype=bool)
        if m.shape != h.shape:
            raise ValueError("mask shape must match height_mm")
        inc = m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]
    if not inc.any():
        raise ValueError("mask excludes every cell; nothing to mesh")
    used_px = np.zeros((H, W), dtype=bool)
    used_px[:-1, :-1] |= inc
    used_px[:-1, 1:] |= inc
    used_px[1:, :-1] |= inc
    used_px[1:, 1:] |= inc
    if not np.all(h[used_px] > 0):
        raise ValueError("every emitted top height must be > 0 (bottom is z = 0)")

    # --- vertices ------------------------------------------------------------
    xs = np.arange(W, dtype=np.float64) * pitch
    ys = (H - 1 - np.arange(H, dtype=np.float64)) * pitch
    X, Y = np.meshgrid(xs, ys)                      # (H, W): X varies along columns
    top = np.stack([X.ravel(), Y.ravel(), h.ravel()], axis=1)
    bottom = np.stack([X.ravel(), Y.ravel(), np.zeros(B)], axis=1)
    verts = np.concatenate([top, bottom], axis=0)

    # --- top / bottom triangles ----------------------------------------------
    idx = np.arange(B, dtype=np.int64).reshape(H, W)
    a, b = idx[:-1, :-1], idx[:-1, 1:]              # upper-left, upper-right (max y)
    c, d = idx[1:, :-1], idx[1:, 1:]                # lower-left, lower-right (min y)
    sel = inc.ravel()
    a, b, c, d = a.ravel()[sel], b.ravel()[sel], c.ravel()[sel], d.ravel()[sel]

    faces = [
        np.stack([a, c, b], axis=1),                # top, CCW seen from +z
        np.stack([b, c, d], axis=1),
        np.stack([a, b, c], axis=1) + B,            # bottom, CCW seen from -z
        np.stack([b, d, c], axis=1) + B,
    ]

    # --- boundary walls --------------------------------------------------------
    inc2 = np.zeros((H + 1, W + 1), dtype=bool)
    inc2[1:H, 1:W] = inc
    # (dr, dc, v1(rr, cc), v2(rr, cc)) chosen so [v1, v2, v2+B], [v1, v2+B, v1+B]
    # have outward normals (north +y, south -y, west -x, east +x).
    directions = (
        (-1, 0, lambda r, cc: idx[r, cc],         lambda r, cc: idx[r, cc + 1]),
        (1, 0,  lambda r, cc: idx[r + 1, cc + 1], lambda r, cc: idx[r + 1, cc]),
        (0, -1, lambda r, cc: idx[r + 1, cc],     lambda r, cc: idx[r, cc]),
        (0, 1,  lambda r, cc: idx[r, cc + 1],     lambda r, cc: idx[r + 1, cc + 1]),
    )
    for dr, dc, n1, n2 in directions:
        neighbour = inc2[1 + dr:H + dr, 1 + dc:W + dc]
        rr, cc = np.nonzero(inc & ~neighbour)
        if len(rr):
            v1 = n1(rr, cc)
            v2 = n2(rr, cc)
            faces.append(np.stack([v1, v2, v2 + B], axis=1))
            faces.append(np.stack([v1, v2 + B, v1 + B], axis=1))

    F = np.concatenate(faces, axis=0)

    # --- drop unreferenced vertices (only when a mask excluded cells) ----------
    if mask is not None:
        used = np.unique(F)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        verts = verts[used]
        F = remap[F]

    mesh = trimesh.Trimesh(vertices=verts, faces=F, process=False)
    return mesh
