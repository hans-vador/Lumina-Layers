"""Art-directed relief: height maps in millimetres and mask-driven shape edits.

The normal Relief Stack5 pipeline turns a depth map into whole-layer relief
steps (:mod:`core.relief.depth_processing`).  Art direction works on top of that
result as a **float32 height map in millimetres** (0 = the flat base, up to the
relief budget, 4.0 mm by default): the user selects part of the cover with a
mask and replaces or offsets the relief inside it.  The edited map is quantised
back to whole layers by :func:`quantize_mm` before it reaches the voxel builder,
so everything downstream (colour shell, meshing, 3MF) is unchanged.

Only geometry is edited.  The Stack5 colour recipe still comes from the original
RGB pixels; no operation here ever touches colour.

Operations (all take a bool mask on the relief grid; an empty mask is a no-op):

* ``dome``     - a smooth elliptical hemisphere inside the mask.  The apex is at
                 ``height_mm`` above the flat base; the rim meets the surrounding
                 relief (the background continued under the object).  The
                 profile is ``(1 - u**n) ** (1/n)`` with ``n = 2 / roundness``
                 (roundness 1 = hemisphere, < 1 = flatter top with steeper
                 sides, > 1 = pointier), ``u`` the normalised radius from the
                 mask centroid to the outline in that direction (exact
                 ellipsoid for an elliptical mask).
* ``extrude``  - the whole mask becomes a flat plateau at ``height_mm``.
* ``offset``   - raise / lower the mask by a signed ``delta_mm``.
* ``feather``  - blend the relief across the mask outline (Gaussian, band of
                 ``width_mm`` centred on the outline); pixels farther than half
                 the band from the outline are untouched.
* ``reset``    - restore the original (Depth Anything) values inside the mask.
* ``object``   - composite a provider's front-depth profile (0..1) inside the
                 mask, apex at ``height_mm`` (see core.relief.object_provider).

``feather_mm`` on dome / extrude / offset blends the new shape into the current
relief over a band INSIDE the mask, so nothing outside the selection changes.
Every result is clamped to ``0 .. relief_max_mm``.  Everything is deterministic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

OPS = ('dome', 'extrude', 'offset', 'feather', 'reset', 'object')
DEFAULT_ROUNDNESS = 1.0
ROUNDNESS_RANGE = (0.2, 3.0)
MIN_RADIAL_BINS, MAX_RADIAL_BINS = 36, 720


def _cv2():
    import cv2
    return cv2


# --------------------------------------------------------------------------- masks and weights
def as_bool_mask(mask, shape: Optional[tuple[int, int]] = None) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    if shape is not None and m.shape != tuple(shape):
        raise ValueError(f"mask shape {m.shape} does not match the height map {tuple(shape)}")
    return m


def smoothstep(t) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def inside_distance(mask: np.ndarray) -> np.ndarray:
    """Distance (px) from every inside pixel to the nearest outside pixel; 0 outside.
    Pixels on the outline get 1."""
    m = as_bool_mask(mask)
    if not m.any():
        return np.zeros(m.shape, np.float32)
    if m.all():
        return np.full(m.shape, float(max(m.shape)), np.float32)
    from scipy import ndimage
    return ndimage.distance_transform_edt(m).astype(np.float32)


def outside_distance(mask: np.ndarray) -> np.ndarray:
    return inside_distance(~as_bool_mask(mask))


def feather_weight(mask: np.ndarray, feather_px: float) -> np.ndarray:
    """1 deep inside the mask, smoothly down to ~0 on the outline over a band of
    ``feather_px`` INSIDE the mask, exactly 0 outside."""
    m = as_bool_mask(mask)
    if float(feather_px) <= 0:
        return m.astype(np.float32)
    d = inside_distance(m)
    w = smoothstep((d - 0.5) / float(feather_px))
    w[~m] = 0.0
    return w


def band_weight(mask: np.ndarray, width_px: float) -> np.ndarray:
    """1 on the mask outline, smoothly 0 at ``width_px / 2`` on either side."""
    m = as_bool_mask(mask)
    if float(width_px) <= 0 or not m.any() or m.all():
        return np.zeros(m.shape, np.float32)
    d = np.where(m, inside_distance(m), outside_distance(m)) - 0.5      # 0.5 on both outline rows
    half = max(float(width_px) / 2.0, 0.5)
    return (1.0 - smoothstep(d / half)).astype(np.float32)


def radial_coordinate(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Normalised radius ``u`` for every mask pixel: 0 at the mask centroid, 1 on
    the outline in that direction (the outline radius is sampled per angular
    bin, so an ellipse gives the exact elliptical radius).  Pixels outside the
    mask get 1."""
    m = as_bool_mask(mask)
    H, W = m.shape
    u = np.ones((H, W), np.float32)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return u, {'empty': True}
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = xs - cx, ys - cy
    r = np.hypot(dx, dy)
    r_max = float(r.max())
    if r_max <= 0:
        u[ys, xs] = 0.0
        return u, {'centroid_xy': [cx, cy], 'radius_px': [0.5, 0.5]}
    nb = int(np.clip(2.0 * np.pi * r_max / 2.0, MIN_RADIAL_BINS, MAX_RADIAL_BINS))   # ~1 bin per 2 outline px
    b = (((np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)) * nb).astype(np.int64) % nb
    R = np.zeros(nb, np.float64)
    np.maximum.at(R, b, r + 0.5)
    empty = R <= 0
    if empty.any():
        idx = np.arange(nb)
        good = idx[~empty]
        gg = np.concatenate([good - nb, good, good + nb])
        pos = np.searchsorted(gg, idx)
        lo = gg[np.clip(pos - 1, 0, len(gg) - 1)]
        hi = gg[np.clip(pos, 0, len(gg) - 1)]
        near = np.where(np.abs(idx - lo) <= np.abs(hi - idx), lo, hi) % nb
        R[empty] = R[near[empty]]
    u[ys, xs] = np.clip(r / R[b], 0.0, 1.0).astype(np.float32)
    return u, {'centroid_xy': [cx, cy], 'radius_px': [float(R.min()), float(R.max())],
               'radius_mean_px': float(R.mean()), 'bins': int(nb)}


def dome_profile(u: np.ndarray, roundness: float = DEFAULT_ROUNDNESS) -> np.ndarray:
    """``(1 - u**n) ** (1/n)`` with ``n = 2 / roundness``: 1 at u = 0, 0 at u = 1.
    roundness 1 -> hemisphere, 0.5 -> flat top / steep sides, 2 -> cone."""
    rr = float(np.clip(float(roundness), *ROUNDNESS_RANGE))
    n = 2.0 / rr
    uu = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    return np.power(np.clip(1.0 - np.power(uu, n), 0.0, 1.0), 1.0 / n).astype(np.float32)


def inpaint_base(h: np.ndarray, mask: np.ndarray, sigma_px: Optional[float] = None) -> np.ndarray:
    """The relief with the masked object 'removed': inside the mask every pixel
    takes the nearest outside value, blurred (sigma ~ a quarter of the mask's
    equivalent radius) so the background continues smoothly under the object."""
    from core.relief.depth_processing import nearest_fill
    hh = np.asarray(h, dtype=np.float32)
    m = as_bool_mask(mask, hh.shape)
    if not m.any() or m.all():
        return hh.copy()
    fill = nearest_fill(hh, ~m)
    if sigma_px is None:
        sigma_px = max(1.0, 0.25 * float(np.sqrt(m.sum() / np.pi)))
    cv2 = _cv2()
    blurred = cv2.GaussianBlur(np.ascontiguousarray(fill, dtype=np.float32), (0, 0), float(sigma_px),
                               borderType=cv2.BORDER_REPLICATE)
    out = hh.copy()
    out[m] = blurred[m]
    return out


# --------------------------------------------------------------------------- operations
def _blend(h: np.ndarray, mask: np.ndarray, target: np.ndarray, feather_px: float, relief_max: float) -> np.ndarray:
    w = feather_weight(mask, feather_px)
    out = h + w * (target.astype(np.float32) - h)
    return np.clip(out, 0.0, float(relief_max)).astype(np.float32)


def op_profile(h: np.ndarray, mask: np.ndarray, profile: np.ndarray, height_mm: float, feather_px: float,
               relief_max: float, base: Optional[np.ndarray] = None) -> np.ndarray:
    """Inside the mask: ``base + (height_mm - base) * profile`` (profile 1 = apex
    at height_mm, 0 = the continued background), feathered into the current relief."""
    hh = np.asarray(h, dtype=np.float32)
    m = as_bool_mask(mask, hh.shape)
    if not m.any():
        return hh.copy()
    p = np.clip(np.asarray(profile, dtype=np.float32), 0.0, 1.0)
    if p.shape != hh.shape:
        raise ValueError("profile must have the height map's shape")
    if base is None:
        base = inpaint_base(hh, m)
    target = base + (float(height_mm) - base) * p
    return _blend(hh, m, target, feather_px, relief_max)


def op_dome(h: np.ndarray, mask: np.ndarray, height_mm: float, roundness: float = DEFAULT_ROUNDNESS,
            feather_px: float = 0.0, relief_max: float = 4.0) -> tuple[np.ndarray, dict]:
    hh = np.asarray(h, dtype=np.float32)
    m = as_bool_mask(mask, hh.shape)
    if not m.any():
        return hh.copy(), {'empty': True}
    u, info = radial_coordinate(m)
    p = dome_profile(u, roundness)
    p[~m] = 0.0
    return op_profile(hh, m, p, height_mm, feather_px, relief_max), info


def op_extrude(h: np.ndarray, mask: np.ndarray, height_mm: float, feather_px: float = 0.0,
               relief_max: float = 4.0) -> np.ndarray:
    hh = np.asarray(h, dtype=np.float32)
    m = as_bool_mask(mask, hh.shape)
    if not m.any():
        return hh.copy()
    target = np.full(hh.shape, float(np.clip(height_mm, 0.0, relief_max)), np.float32)
    return _blend(hh, m, target, feather_px, relief_max)


def op_offset(h: np.ndarray, mask: np.ndarray, delta_mm: float, feather_px: float = 0.0,
              relief_max: float = 4.0) -> np.ndarray:
    hh = np.asarray(h, dtype=np.float32)
    m = as_bool_mask(mask, hh.shape)
    if not m.any():
        return hh.copy()
    return _blend(hh, m, hh + np.float32(delta_mm), feather_px, relief_max)


def op_feather(h: np.ndarray, mask: np.ndarray, width_px: float, relief_max: float = 4.0) -> np.ndarray:
    """Smooth the relief across the mask outline: within a band of ``width_px``
    centred on the outline the map is blended with its Gaussian blur
    (sigma = width / 3).  Pixels farther than width / 2 from the outline are unchanged."""
    hh = np.asarray(h, dtype=np.float32)
    m = as_bool_mask(mask, hh.shape)
    if float(width_px) <= 0 or not m.any() or m.all():
        return hh.copy()
    w = band_weight(m, width_px)
    cv2 = _cv2()
    sigma = max(float(width_px) / 3.0, 0.5)
    blurred = cv2.GaussianBlur(np.ascontiguousarray(hh), (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
    out = hh + w * (blurred - hh)
    return np.clip(out, 0.0, float(relief_max)).astype(np.float32)


def op_reset(h: np.ndarray, original: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    hh = np.asarray(h, dtype=np.float32)
    orig = np.asarray(original, dtype=np.float32)
    if orig.shape != hh.shape:
        raise ValueError("original must have the height map's shape")
    if mask is None:
        return orig.copy()
    m = as_bool_mask(mask, hh.shape)
    out = hh.copy()
    out[m] = orig[m]
    return out


def quantize_mm(h: np.ndarray, layer_h: float, max_layers: int) -> np.ndarray:
    """mm -> whole layers, round half up, clamped to 0..max_layers (int16).
    ``k * layer_h`` (float32) always maps back to ``k``."""
    q = np.floor(np.asarray(h, dtype=np.float64) / float(layer_h) + 0.5)
    return np.clip(q, 0, int(max_layers)).astype(np.int16)


def _range_mm(a: np.ndarray, m: np.ndarray) -> list[float]:
    v = a[m]
    if v.size == 0:
        return [0.0, 0.0]
    return [round(float(v.min()), 4), round(float(v.max()), 4)]


def apply_op(h: np.ndarray, original: np.ndarray, op: dict, pixel_mm: float, relief_max: float) -> tuple[np.ndarray, dict]:
    """Apply one edit dict to the mm height map.  Keys: ``op`` (one of OPS),
    ``mask`` (bool (H, W)), ``height_mm``, ``roundness``, ``feather_mm``,
    ``delta_mm``, ``width_mm`` (feather op), ``profile`` (object op).  Geometric
    sizes are in mm and converted with ``pixel_mm`` so a script replays on any grid."""
    hh = np.asarray(h, dtype=np.float32)
    kind = str(op.get('op', '')).lower()
    if kind not in OPS:
        raise ValueError(f"unknown edit op {kind!r}; choose from {OPS}")
    mask = op.get('mask')
    m = None if mask is None else as_bool_mask(mask, hh.shape)
    info: dict = {'op': kind}
    if m is None or not m.any():
        info.update(noop=True, reason='empty mask', pixels_changed=0, mask_pixels=0)
        return hh.copy(), info
    px = float(pixel_mm)
    if px <= 0:
        raise ValueError("pixel_mm must be > 0")
    feather_px = max(0.0, float(op.get('feather_mm', 0.0) or 0.0)) / px
    extra: dict = {}
    if kind == 'dome':
        out, extra = op_dome(hh, m, float(op['height_mm']), float(op.get('roundness', DEFAULT_ROUNDNESS)), feather_px,
                             relief_max)
    elif kind == 'extrude':
        out = op_extrude(hh, m, float(op['height_mm']), feather_px, relief_max)
    elif kind == 'offset':
        out = op_offset(hh, m, float(op['delta_mm']), feather_px, relief_max)
    elif kind == 'feather':
        width_mm = op.get('width_mm', op.get('feather_mm', 1.0))
        out = op_feather(hh, m, float(width_mm or 0.0) / px, relief_max)
    elif kind == 'reset':
        out = op_reset(hh, original, m)
    else:  # object
        out = op_profile(hh, m, np.asarray(op['profile'], dtype=np.float32), float(op['height_mm']), feather_px,
                         relief_max)
    changed = out != hh
    info.update(noop=not bool(changed.any()), pixels_changed=int(changed.sum()), mask_pixels=int(m.sum()),
                mm_before=_range_mm(hh, m), mm_after=_range_mm(out, m),
                params={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                        for k, v in op.items() if k not in ('mask', 'profile', 'op')})
    if extra:
        info['dome'] = extra
    return out, info
