"""Extension point: AI object-geometry providers for art-directed relief.

A future provider (e.g. Stable Fast 3D) receives the selected object as an RGBA
crop + mask (:class:`ObjectCrop`) and returns either a front depth map or a mesh
(:class:`ObjectGeometry`).  :func:`object_profile` turns that into a 0..1 relief
profile on the full grid and :func:`composite_object` writes it into the mm
height map through the same primitive the dome uses (``heightfield.op_profile``):
apex at ``height_mm``, rim on the continued background, feathered edge.

Only geometry flows through here.  The provider never sees or changes colour.

Stable Fast 3D is NOT integrated: :class:`StableFast3DProvider` is a stub whose
``available()`` explains that, so the GUI / CLI can list it without loading
anything.  Register real providers in :data:`OBJECT_PROVIDERS`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from core.relief.heightfield import as_bool_mask, op_profile
from core.relief.provider import HIGHER_IS_CLOSER, HIGHER_IS_FARTHER, GeometryError, MissingDependencyError


@dataclass
class ObjectCrop:
    """The selected object, cut from the relief grid image."""
    rgba: np.ndarray                      # (h, w, 4) uint8, alpha = mask
    mask: np.ndarray                      # (h, w) bool
    bbox: tuple[int, int, int, int]       # (x0, y0, x1, y1) on the grid, x1/y1 exclusive
    grid_hw: tuple[int, int]
    pixel_mm: float
    source_image: str = ''

    @property
    def size_mm(self) -> tuple[float, float]:
        return ((self.bbox[2] - self.bbox[0]) * self.pixel_mm, (self.bbox[3] - self.bbox[1]) * self.pixel_mm)


@dataclass
class ObjectGeometry:
    """What a provider returns: a front depth map (crop-sized, larger = closer by
    default) and/or a mesh (vertices (V, 3), faces (F, 3)) whose +Z faces the viewer."""
    depth: Optional[np.ndarray] = None    # (h, w) float32
    hit: Optional[np.ndarray] = None      # (h, w) bool, where depth is valid
    mesh: Optional[tuple[np.ndarray, np.ndarray]] = None
    depth_convention: str = HIGHER_IS_CLOSER
    front_axis: str = '+z'
    up_axis: str = '+y'
    source: str = ''
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.depth is None and self.mesh is None:
            raise GeometryError("ObjectGeometry needs a depth map or a mesh")
        if self.depth_convention not in (HIGHER_IS_CLOSER, HIGHER_IS_FARTHER):
            raise GeometryError("depth_convention must be 'higher-is-closer' or 'higher-is-farther'")


@runtime_checkable
class ObjectGeometryProvider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...
    def generate(self, crop: ObjectCrop) -> ObjectGeometry: ...


class StableFast3DProvider:
    """Placeholder for Stable Fast 3D (not integrated yet)."""
    name = 'stable-fast-3d'

    def __init__(self, **_options):
        self.options = dict(_options)

    def available(self) -> tuple[bool, str]:
        return False, ("Stable Fast 3D is not integrated yet: this is the provider interface only. "
                       "Use Dome / Extrude on the selection, or plug a provider into "
                       "core.relief.object_provider.OBJECT_PROVIDERS.")

    def generate(self, crop: ObjectCrop) -> ObjectGeometry:
        raise MissingDependencyError(self.available()[1])


OBJECT_PROVIDERS = {'stable-fast-3d': StableFast3DProvider}


def get_object_provider(name: str, **options) -> ObjectGeometryProvider:
    key = (name or '').strip().lower()
    if key not in OBJECT_PROVIDERS:
        raise ValueError(f"unknown object provider {name!r}; choose from {sorted(OBJECT_PROVIDERS)}")
    return OBJECT_PROVIDERS[key](**options)


# --------------------------------------------------------------------------- crop / composite
def crop_object(rgb_grid: np.ndarray, mask: np.ndarray, pixel_mm: float, margin_px: int = 8,
                source_image: str = '') -> ObjectCrop:
    """RGBA crop of the masked object (alpha = mask) with a margin, bbox on the grid."""
    rgb = np.asarray(rgb_grid)
    m = as_bool_mask(mask, rgb.shape[:2])
    if not m.any():
        raise GeometryError("cannot crop an empty mask")
    ys, xs = np.nonzero(m)
    H, W = m.shape
    x0, x1 = max(int(xs.min()) - margin_px, 0), min(int(xs.max()) + 1 + margin_px, W)
    y0, y1 = max(int(ys.min()) - margin_px, 0), min(int(ys.max()) + 1 + margin_px, H)
    sub = m[y0:y1, x0:x1]
    rgba = np.zeros(sub.shape + (4,), np.uint8)
    rgba[..., :3] = rgb[y0:y1, x0:x1, :3]
    rgba[..., 3] = np.where(sub, 255, 0).astype(np.uint8)
    return ObjectCrop(rgba=rgba, mask=sub.copy(), bbox=(x0, y0, x1, y1), grid_hw=(H, W), pixel_mm=float(pixel_mm),
                      source_image=source_image)


def front_depth(geom: ObjectGeometry, crop: ObjectCrop) -> tuple[np.ndarray, np.ndarray]:
    """Crop-sized (h, w) front depth as 'larger = raised' + validity mask.  A mesh
    is rendered orthographically with the Wonder3D rasteriser, its XY bounding
    box fitted to the crop's mask bounding box."""
    h, w = crop.mask.shape
    if geom.depth is not None:
        d = np.asarray(geom.depth, dtype=np.float32)
        hit = np.ones(d.shape, bool) if geom.hit is None else np.asarray(geom.hit).astype(bool)
        if geom.depth_convention == HIGHER_IS_FARTHER:
            d = -d
        if d.shape != (h, w):
            import cv2
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            hit = cv2.resize(hit.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return d.astype(np.float32), hit
    from core.relief.wonder3d import frame_rotation, rasterize_depth
    V, F = geom.mesh
    V = np.asarray(V, dtype=np.float64) @ frame_rotation(geom.front_axis, geom.up_axis).T
    ys, xs = np.nonzero(crop.mask)
    bx0, bx1 = float(xs.min()), float(xs.max() + 1)
    by0, by1 = float(ys.min()), float(ys.max() + 1)
    mx0, mx1 = float(V[:, 0].min()), float(V[:, 0].max())
    my0, my1 = float(V[:, 1].min()), float(V[:, 1].max())
    if mx1 - mx0 <= 0 or my1 - my0 <= 0:
        raise GeometryError("mesh has no XY extent")
    # map the mesh bbox onto the mask bbox (pixel units), rows grow downward in the crop
    sx, sy = (bx1 - bx0) / (mx1 - mx0), (by1 - by0) / (my1 - my0)
    P = np.empty_like(V)
    P[:, 0] = bx0 + (V[:, 0] - mx0) * sx
    P[:, 1] = by1 - (V[:, 1] - my0) * sy          # y up in the mesh -> row down in the crop
    P[:, 2] = V[:, 2]
    # rasterize_depth samples y = y1 - (r + 0.5) * dy: feed it a y range that makes row r = P_y
    depth, hit = rasterize_depth(np.stack([P[:, 0], h - P[:, 1], P[:, 2]], axis=1), np.asarray(F), (h, w), (0.0, float(w)),
                                 (0.0, float(h)))
    depth = np.where(hit, depth, 0.0).astype(np.float32)
    return depth, hit


def object_profile(depth: np.ndarray, hit: np.ndarray, crop: ObjectCrop, low_pct: float = 1.0,
                   high_pct: float = 99.0) -> np.ndarray:
    """Full-grid 0..1 profile: the object's front depth normalised (robust
    percentiles over hit & mask) and placed in the crop's bbox; 0 elsewhere."""
    H, W = crop.grid_hw
    x0, y0, x1, y1 = crop.bbox
    d = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(hit).astype(bool) & crop.mask
    prof = np.zeros((H, W), np.float32)
    if not valid.any():
        return prof
    lo, hi = (float(v) for v in np.percentile(d[valid], [low_pct, high_pct]))
    if hi - lo <= 1e-12:
        sub = np.where(valid, 1.0, 0.0).astype(np.float32)
    else:
        sub = np.where(valid, np.clip((d - lo) / (hi - lo), 0.0, 1.0), 0.0).astype(np.float32)
    prof[y0:y1, x0:x1] = sub
    return prof


def composite_object(h_mm: np.ndarray, mask: np.ndarray, crop: ObjectCrop, geom: ObjectGeometry, height_mm: float,
                     feather_px: float = 0.0, relief_max: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Write a provider's geometry into the mm height map inside ``mask``.
    Returns (new height map, the 0..1 profile used)."""
    depth, hit = front_depth(geom, crop)
    prof = object_profile(depth, hit, crop)
    m = as_bool_mask(mask, np.asarray(h_mm).shape)
    return op_profile(h_mm, m, prof, height_mm, feather_px, relief_max), prof
