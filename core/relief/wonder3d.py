"""Wonder3D adapter (Milestone 3): geometry from a Wonder3D reconstruction.

Wonder3D (Long et al. 2023) turns one image into six consistent RGB views and
normal maps and reconstructs a textured mesh (instant-NSR / NeuS).  Relief
Stack5 uses NOTHING of its colour: the reconstructed geometry is rendered as an
orthographic depth map from the ORIGINAL front view, aligned with the original
artwork and handed to the common relief processing.  The Stack5 colour still
comes from the original image only.

Accepted inputs (Wonder3D itself does not have to be installed):

* ``mesh``      - a mesh file trimesh can read: .obj, .ply, .glb/.gltf, .stl,
                  .off (scenes are flattened to one mesh).
* ``depth_map`` - a front-facing depth image already rendered from that mesh
                  (any renderer).  Loaded like imported-depth; pass
                  ``depth_map_convention='higher-is-farther'`` for z-buffer
                  distances.  Alpha < 10 = background.

Coordinate system / camera (for ``mesh``)
----------------------------------------
Wonder3D reconstructs in a normalised object frame (object centred at the
origin, roughly unit size).  Its input view is the *front* view: an
orthographic camera on the +Z side looking down -Z with +Y up (the camera
convention of its NeuS/instant-nsr training data) - so, by default,
``front_axis='+z'`` (toward the viewer) and ``up_axis='+y'``.  Both can be
changed if a reconstruction pipeline exported another frame.  The mesh is
rotated so that the front axis becomes +Z and the up axis +Y, then projected
orthographically: image x = mesh X, image y = -mesh Y (rows grow downward),
depth = mesh Z (larger = closer -> 'higher-is-closer').  Back faces are hidden
by a max z-buffer; the background gets the far value and mask=False.

Alignment with the original image
---------------------------------
Wonder3D crops / recentres the object before reconstruction, so the mesh's
projected bounding box corresponds to the object's bounding box in the input,
not to the whole picture.  ``fit='mask'`` (default when a foreground mask is
given) maps the mesh's XY bounding box onto the bounding box of the mask (with
``fit_margin`` extra), ``fit='image'`` maps it onto the whole image frame - use
this when the album cover *is* the square Wonder3D input.  Rendering happens at
``render_px`` on the long side of the image; the pipeline resamples onto the
Stack5 grid.  Where there is no mask the background is flat at the far value.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from core.relief.provider import (HIGHER_IS_CLOSER, HIGHER_IS_FARTHER, GeometryError, GeometryResult,
                                  load_depth_file, load_mask_file)

AXES = {'+x': (0, 1.0), '-x': (0, -1.0), '+y': (1, 1.0), '-y': (1, -1.0), '+z': (2, 1.0), '-z': (2, -1.0)}
MESH_EXTS = ('.obj', '.ply', '.glb', '.gltf', '.stl', '.off', '.3mf')


def _axis_vec(name: str) -> np.ndarray:
    key = name.strip().lower()
    if key not in AXES:
        raise GeometryError(f"axis must be one of {sorted(AXES)}, got {name!r}")
    i, s = AXES[key]
    v = np.zeros(3)
    v[i] = s
    return v


def frame_rotation(front_axis: str, up_axis: str) -> np.ndarray:
    """3x3 rotation taking mesh ``front_axis`` -> +Z and ``up_axis`` -> +Y."""
    f = _axis_vec(front_axis)
    u = _axis_vec(up_axis)
    if abs(float(f @ u)) > 1e-9:
        raise GeometryError("front_axis and up_axis must be perpendicular")
    r = np.cross(u, f)                     # right = up x front  (right-handed: x = y cross z)
    R = np.stack([r, u, f], axis=0)        # rows: new x, y, z expressed in old coordinates
    return R


def load_mesh_vertices_faces(path: str) -> tuple[np.ndarray, np.ndarray]:
    import trimesh
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    obj = trimesh.load(path, force='mesh')
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise GeometryError(f"{path}: no triangle geometry")
        obj = trimesh.util.concatenate(geoms)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise GeometryError(f"{path}: no triangle geometry")
    return np.asarray(obj.vertices, dtype=np.float64), np.asarray(obj.faces, dtype=np.int64)


def rasterize_depth(vertices: np.ndarray, faces: np.ndarray, out_hw: tuple[int, int], x_range: tuple[float, float],
                    y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic max z-buffer.  Pixel (r, c) centre samples
    x = x0 + (c + 0.5) * dx, y = y1 - (r + 0.5) * dy (rows grow downward).
    Returns (depth (H, W) float32 with -inf where nothing was hit, hit mask)."""
    H, W = int(out_hw[0]), int(out_hw[1])
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    sx = W / (x1 - x0)
    sy = H / (y1 - y0)
    V = np.asarray(vertices, dtype=np.float64)
    P = np.empty_like(V)
    P[:, 0] = (V[:, 0] - x0) * sx - 0.5              # pixel-centre coordinates
    P[:, 1] = (y1 - V[:, 1]) * sy - 0.5
    P[:, 2] = V[:, 2]
    zbuf = np.full((H, W), -np.inf, dtype=np.float64)
    T = np.asarray(faces, dtype=np.int64)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    xmin = np.floor(np.minimum(np.minimum(A[:, 0], B[:, 0]), C[:, 0])).astype(np.int64)
    xmax = np.ceil(np.maximum(np.maximum(A[:, 0], B[:, 0]), C[:, 0])).astype(np.int64)
    ymin = np.floor(np.minimum(np.minimum(A[:, 1], B[:, 1]), C[:, 1])).astype(np.int64)
    ymax = np.ceil(np.maximum(np.maximum(A[:, 1], B[:, 1]), C[:, 1])).astype(np.int64)
    keep = (xmax >= 0) & (xmin <= W - 1) & (ymax >= 0) & (ymin <= H - 1)
    xmin, xmax = np.clip(xmin, 0, W - 1), np.clip(xmax, 0, W - 1)
    ymin, ymax = np.clip(ymin, 0, H - 1), np.clip(ymax, 0, H - 1)
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (C[:, 0] - A[:, 0]) * (B[:, 1] - A[:, 1])
    keep &= np.abs(area) > 1e-12
    idx = np.nonzero(keep)[0]
    # triangles grouped by bounding-box size (only the size classes present): small
    # boxes run vectorised per class, the few large ones one by one
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    if idx.size:
        key = bw[idx] * (int(bh[idx].max()) + 1) + bh[idx]
        order = np.argsort(key, kind='stable')
        keys_sorted = key[order]
        bounds = np.flatnonzero(np.diff(keys_sorted)) + 1
        for grp in np.split(idx[order], bounds):
            w, h = int(bw[grp[0]]), int(bh[grp[0]])
            if w * h <= 64:
                _raster_batch(grp, A, B, C, area, xmin, ymin, w, h, zbuf)
            else:
                for t in grp:
                    _raster_one(t, A, B, C, area, xmin, xmax, ymin, ymax, zbuf)
    hit = np.isfinite(zbuf)
    return zbuf.astype(np.float32), hit


def _bary_depth(px, py, a, b, c, area):
    w0 = ((b[..., 0] - px) * (c[..., 1] - py) - (c[..., 0] - px) * (b[..., 1] - py)) / area
    w1 = ((c[..., 0] - px) * (a[..., 1] - py) - (a[..., 0] - px) * (c[..., 1] - py)) / area
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    z = w0 * a[..., 2] + w1 * b[..., 2] + w2 * c[..., 2]
    return inside, z


def _raster_batch(sel, A, B, C, area, xmin, ymin, w, h, zbuf):
    n = sel.size
    oy, ox = np.mgrid[0:h, 0:w]
    px = (xmin[sel][:, None, None] + ox[None]).astype(np.float64)
    py = (ymin[sel][:, None, None] + oy[None]).astype(np.float64)
    a = A[sel][:, None, None, :]
    b = B[sel][:, None, None, :]
    c = C[sel][:, None, None, :]
    inside, z = _bary_depth(px, py, a, b, c, area[sel][:, None, None])
    ii = px.astype(np.int64)[inside]
    jj = py.astype(np.int64)[inside]
    ok = (ii >= 0) & (ii < zbuf.shape[1]) & (jj >= 0) & (jj < zbuf.shape[0])
    np.maximum.at(zbuf, (jj[ok], ii[ok]), z[inside][ok])


def _raster_one(t, A, B, C, area, xmin, xmax, ymin, ymax, zbuf):
    ys, xs = np.mgrid[ymin[t]:ymax[t] + 1, xmin[t]:xmax[t] + 1]
    inside, z = _bary_depth(xs.astype(np.float64), ys.astype(np.float64), A[t], B[t], C[t], area[t])
    sub = zbuf[ymin[t]:ymax[t] + 1, xmin[t]:xmax[t] + 1]
    np.maximum(sub, np.where(inside, z, -np.inf), out=sub)


class Wonder3DProvider:
    name = 'wonder3d'

    def __init__(self, mesh: Optional[str] = None, depth_map: Optional[str] = None, mask: Optional[str] = None,
                 front_axis: str = '+z', up_axis: str = '+y', render_px: int = 768, fit: Optional[str] = None,
                 fit_margin: float = 0.0, depth_map_convention: str = HIGHER_IS_CLOSER, **_ignored):
        if not mesh and not depth_map:
            raise GeometryError("wonder3d provider needs a reconstructed mesh (--mesh) or a rendered "
                                "front-view depth image (--depth-map)")
        if fit not in (None, 'mask', 'image'):
            raise GeometryError("fit must be 'mask' or 'image'")
        self.mesh_path = mesh
        self.depth_map = depth_map
        self.mask_path = mask
        self.front_axis = front_axis
        self.up_axis = up_axis
        self.render_px = int(render_px)
        self.fit = fit
        self.fit_margin = float(fit_margin)
        if depth_map_convention not in (HIGHER_IS_CLOSER, HIGHER_IS_FARTHER):
            raise GeometryError("depth_map_convention must be 'higher-is-closer' or 'higher-is-farther'")
        self.depth_map_convention = depth_map_convention

    def preflight(self) -> None:
        """Cheap input checks before the (slow) colour stage."""
        for label, path in (('mesh', self.mesh_path), ('depth_map', self.depth_map), ('mask', self.mask_path)):
            if path and not os.path.isfile(path):
                raise FileNotFoundError(f"wonder3d {label} file not found: {path}")

    def _image_size(self, image_path: str) -> tuple[int, int]:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size            # (W, H)

    def generate(self, image_path: str, target_hw: Optional[tuple[int, int]] = None) -> GeometryResult:
        mask = load_mask_file(self.mask_path) if self.mask_path else None
        if self.depth_map:
            depth, alpha_mask = load_depth_file(self.depth_map)
            m = mask if mask is not None else alpha_mask
            if m is not None and m.shape != depth.shape:
                import cv2
                m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            if m is not None and m.any() and not m.all():
                # background gets the farthest foreground value so it never dominates the range
                far = depth[m].min() if self.depth_map_convention == HIGHER_IS_CLOSER else depth[m].max()
                depth = depth.copy()
                depth[~m] = far
            return GeometryResult(depth=depth, mask=m, depth_convention=self.depth_map_convention, source=self.name,
                                  meta={'input': 'rendered front depth', 'depth_map': os.path.abspath(self.depth_map)})

        # mesh -> orthographic front depth
        V, F = load_mesh_vertices_faces(self.mesh_path)
        R = frame_rotation(self.front_axis, self.up_axis)
        V = V @ R.T
        W_img, H_img = self._image_size(image_path)
        long_side = max(W_img, H_img)
        scale = self.render_px / long_side
        rw, rh = max(2, int(round(W_img * scale))), max(2, int(round(H_img * scale)))
        bx0, by0 = V[:, 0].min(), V[:, 1].min()
        bx1, by1 = V[:, 0].max(), V[:, 1].max()
        fit = self.fit or ('mask' if mask is not None else 'image')
        if fit == 'mask':
            if mask is None:
                raise GeometryError("fit='mask' needs a foreground mask of the original image (--foreground-mask)")
            rows = np.nonzero(mask.any(axis=1))[0]
            cols = np.nonzero(mask.any(axis=0))[0]
            if rows.size == 0:
                raise GeometryError("foreground mask is empty")
            mh, mw = mask.shape
            # mask bbox in normalised image coordinates (0..1, y down)
            u0, u1 = cols[0] / mw, (cols[-1] + 1) / mw
            v0, v1 = rows[0] / mh, (rows[-1] + 1) / mh
        else:
            u0, u1, v0, v1 = 0.0, 1.0, 0.0, 1.0
        mrg = self.fit_margin
        u0, u1 = u0 - mrg * (u1 - u0), u1 + mrg * (u1 - u0)
        v0, v1 = v0 - mrg * (v1 - v0), v1 + mrg * (v1 - v0)
        # mesh bbox -> [u0, u1] x [v0, v1] of the image; image frame -> mesh units
        su = (bx1 - bx0) / max(u1 - u0, 1e-9)
        sv = (by1 - by0) / max(v1 - v0, 1e-9)
        x_range = (bx0 - u0 * su, bx0 + (1.0 - u0) * su)
        y_top = by1 + v0 * sv                              # image top (v = 0) in mesh y
        y_range = (y_top - sv, y_top)
        depth, hit = rasterize_depth(V, F, (rh, rw), x_range, y_range)
        if not hit.any():
            raise GeometryError("the mesh projects outside the image frame; check front_axis / up_axis / fit")
        far = float(depth[hit].min())
        depth = np.where(hit, depth, far).astype(np.float32)
        meta = {'input': 'mesh', 'mesh': os.path.abspath(self.mesh_path), 'front_axis': self.front_axis,
                'up_axis': self.up_axis, 'fit': fit, 'fit_margin': mrg, 'render_hw': [rh, rw],
                'triangles': int(F.shape[0]), 'hit_share': float(hit.mean()),
                'mesh_bbox_xy': [float(bx0), float(by0), float(bx1), float(by1)],
                'camera': 'orthographic, looking along -Z after rotation, +Y up, max z-buffer'}
        return GeometryResult(depth=depth, mask=hit, depth_convention=HIGHER_IS_CLOSER, source=self.name, meta=meta)
