"""Depth cleanup and quantisation: provider depth -> whole-layer relief field.

Order of operations (:func:`process_depth`):

 1. take the provider depth as 'larger = raised' (:meth:`GeometryResult.raised`)
 2. resample depth (and mask) onto the exact Stack5 output grid (H, W)
 3. normalise with configurable low / high percentiles (over the foreground
    when a mask exists), clamp to 0..1
 4. optionally invert (1 - d)
 5. optionally flatten the background to ``background_level`` using the mask
 6. edge-preserving smoothing (bilateral: sigma_space = ``smoothing_px`` px,
    sigma_color = ``smoothing_range`` in normalised units), optional median
 7. quantise:  relief_steps = round(d * relief_layers / step_layers) * step_layers,
    clamped to 0..relief_layers  (default relief_layers = round(4.0 / 0.08) = 50)
 8. remove features thinner than the nozzle (grey-level opening + closing with
    a ``min_feature_px`` square) so no 1-px spike or slot survives
 9. limit the height difference between 4- (or 8-) neighbours to
    ``max_neighbor_step`` layers with an exact lower-envelope clamp
    (h <- min(h, min_neighbour + step) until stable).  Pixels outside the
    protected region (foreground mask, and always the plaque cut-out) do not
    constrain it, so object silhouettes stay as sharp, fully supported
    vertical walls while accidental cliffs inside a region are turned into
    printable ramps.
10. re-apply the flattened background.

Everything is deterministic (no RNG).  Every output height is an integer
number of layers; the pipeline turns it into mm with ``steps * layer_h``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from core.relief.provider import GeometryResult
from core.relief.relief_stack import ReliefDims

NORMALIZE_MODES = ('percentile', 'none')
RESIZE_MODES = ('auto', 'linear', 'area', 'cubic', 'nearest')


@dataclass
class DepthParams:
    invert: bool = False
    normalize: str = 'percentile'           # 'percentile' | 'none' (input already 0..1)
    low_percentile: float = 1.0
    high_percentile: float = 99.0
    smoothing_px: float = 3.0               # bilateral sigma_space (px on the Stack5 grid); 0 = off
    smoothing_range: float = 0.08           # bilateral sigma_color in normalised depth units
    median_px: int = 0                      # odd kernel; 0 = off
    flatten_background: bool = False
    background_level: float = 0.0           # normalised height of the flattened background
    max_neighbor_step: Optional[int] = None  # layers; None / 0 = off
    neighbor_connectivity: int = 4          # 4 or 8
    protect_silhouette: bool = True         # foreground/background boundary may stay a cliff
    min_feature_px: int = 4                 # grey opening+closing kernel (0.4 mm at 0.1 mm/px); <=1 = off
    step_layers: int = 1                    # relief granularity in layers (1 = every 0.08 mm)
    resize: str = 'auto'                    # 'auto' (area when shrinking, linear when growing)

    def __post_init__(self):
        if self.normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}")
        if not (0.0 <= float(self.low_percentile) < float(self.high_percentile) <= 100.0):
            raise ValueError("need 0 <= low_percentile < high_percentile <= 100")
        if float(self.smoothing_px) < 0 or float(self.smoothing_range) <= 0:
            raise ValueError("smoothing_px must be >= 0 and smoothing_range > 0")
        if int(self.median_px) < 0 or (int(self.median_px) > 0 and int(self.median_px) % 2 == 0):
            raise ValueError("median_px must be 0 or an odd kernel size")
        if not (0.0 <= float(self.background_level) <= 1.0):
            raise ValueError("background_level must be within 0..1")
        if self.max_neighbor_step is not None and int(self.max_neighbor_step) < 0:
            raise ValueError("max_neighbor_step must be >= 0 (0 = off)")
        if int(self.neighbor_connectivity) not in (4, 8):
            raise ValueError("neighbor_connectivity must be 4 or 8")
        if int(self.min_feature_px) < 0:
            raise ValueError("min_feature_px must be >= 0")
        if int(self.step_layers) < 1:
            raise ValueError("step_layers must be >= 1")
        if self.resize not in RESIZE_MODES:
            raise ValueError(f"resize must be one of {RESIZE_MODES}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReliefField:
    """Quantised relief on the Stack5 grid."""
    steps: np.ndarray                       # (H, W) int16 relief layers, 0..relief_layers
    normalized: np.ndarray                  # (H, W) float32 normalised depth after cleanup (0..1)
    mask: Optional[np.ndarray]              # (H, W) bool foreground mask on the grid (or None)
    dims: ReliefDims
    report: dict = field(default_factory=dict)

    def height_mm(self) -> np.ndarray:
        return self.steps.astype(np.float64) * float(self.dims.layer_h)


# --------------------------------------------------------------------------- helpers
def _cv2():
    import cv2
    return cv2


def _interp(mode: str, src_hw, dst_hw):
    cv2 = _cv2()
    if mode == 'auto':
        shrinking = dst_hw[0] * dst_hw[1] < src_hw[0] * src_hw[1]
        return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return {'linear': cv2.INTER_LINEAR, 'area': cv2.INTER_AREA, 'cubic': cv2.INTER_CUBIC,
            'nearest': cv2.INTER_NEAREST}[mode]


def align_depth(depth: np.ndarray, target_hw: tuple[int, int], resize: str = 'auto') -> tuple[np.ndarray, dict]:
    """Resample (H0, W0) depth onto (H, W).  The map is stretched to the grid;
    an aspect-ratio deviation > 2 % is reported as a warning."""
    cv2 = _cv2()
    d = np.asarray(depth, dtype=np.float32)
    H, W = int(target_hw[0]), int(target_hw[1])
    H0, W0 = d.shape
    info = {'source_hw': [H0, W0], 'target_hw': [H, W], 'warnings': []}
    if H0 == H and W0 == W:
        info['resampled'] = False
        return d.copy(), info
    r_src, r_dst = W0 / H0, W / H
    dev = abs(r_src - r_dst) / r_dst
    info['aspect_deviation'] = float(dev)
    if dev > 0.02:
        info['warnings'].append(f"depth map aspect {W0}x{H0} differs from the image grid {W}x{H} by "
                                f"{dev:.0%}; it is stretched to fit")
    out = cv2.resize(d, (W, H), interpolation=_interp(resize, (H0, W0), (H, W)))
    info['resampled'] = True
    return out.astype(np.float32), info


def align_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    cv2 = _cv2()
    m = np.asarray(mask).astype(np.uint8)
    H, W = int(target_hw[0]), int(target_hw[1])
    if m.shape == (H, W):
        return m.astype(bool)
    return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def nearest_fill(a: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Copy of ``a`` where pixels outside ``mask`` take the value of the nearest
    inside pixel (Euclidean distance transform) - a Neumann-like padding so
    filters and morphology near the mask boundary see no artificial cliff."""
    a = np.asarray(a)
    if mask is None:
        return a.copy()
    mask = np.asarray(mask).astype(bool)
    if mask.all() or not mask.any():
        return a.copy()
    from scipy import ndimage
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return a[iy, ix]


def normalize_depth(depth: np.ndarray, params: DepthParams, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    d = np.asarray(depth, dtype=np.float64)
    if mask is not None and np.asarray(mask).any():
        vals = d[np.asarray(mask).astype(bool)]
    else:
        vals = d.ravel()
    info: dict = {'mode': params.normalize, 'flat': False}
    if params.normalize == 'none':
        lo, hi = 0.0, 1.0
    else:
        lo, hi = (float(v) for v in np.percentile(vals, [float(params.low_percentile), float(params.high_percentile)]))
    info['low'], info['high'] = lo, hi
    if hi - lo <= 1e-12:
        info['flat'] = True
        info['warning'] = ("depth map is flat (no range between the low and high percentiles); relief set to 0 "
                           "everywhere")
        return np.zeros(d.shape, dtype=np.float32), info
    norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    info['clipped_low_share'] = float(np.mean(d < lo - 1e-9))
    info['clipped_high_share'] = float(np.mean(d > hi + 1e-9))
    if params.normalize == 'none' and info['clipped_low_share'] + info['clipped_high_share'] > 0.01:
        info['warning'] = (f"normalize='none' expects a 0..1 map but {info['clipped_low_share'] + info['clipped_high_share']:.0%} "
                           f"of the pixels lie outside (observed range {float(d.min()):g}..{float(d.max()):g}); they are clipped")
    return norm.astype(np.float32), info


def smooth_depth(norm: np.ndarray, params: DepthParams, mask: Optional[np.ndarray] = None) -> np.ndarray:
    cv2 = _cv2()
    out = nearest_fill(np.asarray(norm, dtype=np.float32), mask)
    if int(params.median_px) > 1:
        k = int(params.median_px)
        # cv2.medianBlur only takes float32 for k in {3, 5}; use a rank filter otherwise
        if k <= 5:
            out = cv2.medianBlur(out, k)
        else:
            from scipy import ndimage
            out = ndimage.median_filter(out, size=k, mode='nearest').astype(np.float32)
    if float(params.smoothing_px) > 0:
        # replicate-pad so a ramp keeps its extremes at the image border (cv2's
        # default reflect border would pull them toward the interior)
        pad = int(np.ceil(1.5 * float(params.smoothing_px))) + 2
        padded = cv2.copyMakeBorder(out, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        padded = cv2.bilateralFilter(padded, 0, float(params.smoothing_range), float(params.smoothing_px))
        out = padded[pad:-pad, pad:-pad]
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    if mask is not None:
        out[~np.asarray(mask).astype(bool)] = np.asarray(norm, dtype=np.float32)[~np.asarray(mask).astype(bool)]
    return out


def quantize_steps(norm: np.ndarray, dims: ReliefDims, step_layers: int = 1) -> np.ndarray:
    """relief_steps = round(norm * relief_layers / step_layers) * step_layers, clamped."""
    n_max = int(dims.relief_layers)
    s = max(1, int(step_layers))
    top = (n_max // s) * s                      # stay a multiple of step_layers (48 for 50 / 3)
    q = np.floor(np.asarray(norm, dtype=np.float64) * n_max / s + 0.5) * s    # round half up
    q = np.clip(q, 0, top)
    return q.astype(np.int16)


def _open_close(f: np.ndarray, k: int, close_first: bool = True) -> np.ndarray:
    """Grey opening / closing with EXACT reflected anchors, so even kernels do not
    shift the result by a pixel (cv2's default anchor would)."""
    cv2 = _cv2()
    kernel = np.ones((k, k), np.uint8)
    a = k // 2
    an, ar = (a, a), (k - 1 - a, k - 1 - a)
    big = float(np.abs(f).max() + 1.0) if f.size else 1.0

    def erode(x, anchor):
        return cv2.erode(x, kernel, anchor=anchor, borderType=cv2.BORDER_CONSTANT, borderValue=big)

    def dilate(x, anchor):
        return cv2.dilate(x, kernel, anchor=anchor, borderType=cv2.BORDER_CONSTANT, borderValue=-big)

    def opening(x):
        return dilate(erode(x, an), ar)

    def closing(x):
        return erode(dilate(x, an), ar)

    # replicate-pad so a ramp running into the image border keeps its end value
    # (on the bare finite domain an opening clips it by ~k/2 pixels)
    padded = cv2.copyMakeBorder(np.ascontiguousarray(f, dtype=np.float32), k, k, k, k, cv2.BORDER_REPLICATE)
    out = opening(closing(padded)) if close_first else closing(opening(padded))
    return out[k:-k, k:-k]


def morphological_clean(steps: np.ndarray, min_feature_px: int, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Grey-level closing (fills slots / pits narrower than the kernel) then
    opening (removes ridges / spikes narrower than the kernel) on the step
    map, i.e. no feature thinner than ``min_feature_px`` survives in either
    direction.  Values stay whole layers (min/max only) and nothing shifts.
    Outside-mask pixels are padded with their nearest inside neighbour and
    restored afterwards; image borders never constrain."""
    k = int(min_feature_px)
    if k <= 1:
        return np.asarray(steps).copy()
    src = np.asarray(steps)
    f = nearest_fill(src.astype(np.float32), mask)
    f = _open_close(f, k, close_first=True)
    out = np.rint(f).astype(src.dtype)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        out[~m] = src[~m]
    return out


def limit_neighbor_step(steps: np.ndarray, max_step: Optional[int], mask: Optional[np.ndarray] = None,
                        connectivity: int = 4, max_iter: int = 4096) -> tuple[np.ndarray, dict]:
    """Clamp |h(p) - h(q)| <= max_step for neighbouring pixels p, q inside ``mask``
    by lowering the higher side: h <- min(h, min_neighbour(h) + max_step),
    iterated to the fixed point (the exact lower envelope, i.e. the tallest
    surface under ``steps`` whose slope never exceeds max_step per pixel).
    Pixels outside ``mask`` are neither changed nor used as constraints, so a
    silhouette against the background may remain a vertical wall."""
    src = np.asarray(steps)
    if max_step is None or int(max_step) <= 0:
        return src.copy(), {'applied': False}
    cv2 = _cv2()
    k = float(int(max_step))
    big = np.float32(1e6)
    work = src.astype(np.float32)
    m = None if mask is None else np.asarray(mask).astype(bool)
    if m is not None:
        work[~m] = big
    kernel = (cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) if int(connectivity) == 4
              else np.ones((3, 3), np.uint8))
    iters = 0
    lowered_total = 0
    for iters in range(1, int(max_iter) + 1):
        low = cv2.erode(work, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=float(big))
        new = np.minimum(work, low + k)
        if m is not None:
            new[~m] = big
        changed = new != work
        if not changed.any():
            break
        lowered_total += int(changed.sum())
        work = new
    out = np.rint(work).astype(src.dtype)
    if m is not None:
        out[~m] = src[~m]
    return out, {'applied': True, 'max_step_layers': int(max_step), 'connectivity': int(connectivity),
                 'iterations': int(iters), 'pixels_lowered': int(np.count_nonzero(out != src)),
                 'max_lowering_layers': int((src.astype(np.int64) - out.astype(np.int64)).max())}


def neighbor_step_stats(steps: np.ndarray, mask: Optional[np.ndarray] = None) -> dict:
    s = np.asarray(steps).astype(np.int64)
    m = np.ones(s.shape, bool) if mask is None else np.asarray(mask).astype(bool)
    dx = np.abs(np.diff(s, axis=1))
    dy = np.abs(np.diff(s, axis=0))
    vx = m[:, 1:] & m[:, :-1]
    vy = m[1:, :] & m[:-1, :]
    diffs = np.concatenate([dx[vx], dy[vy]]) if (vx.any() or vy.any()) else np.zeros(0, np.int64)
    if diffs.size == 0:
        return {'edges': 0}
    return {'edges': int(diffs.size), 'max_step_layers': int(diffs.max()),
            'mean_step_layers': float(diffs.mean()),
            'share_steps_over_5_layers': float(np.mean(diffs > 5)),
            'share_steps_over_12_layers': float(np.mean(diffs > 12))}


def relief_stats(steps: np.ndarray, dims: ReliefDims, mask: Optional[np.ndarray] = None) -> dict:
    s = np.asarray(steps).astype(np.int64)
    m = np.ones(s.shape, bool) if mask is None else np.asarray(mask).astype(bool)
    vals = s[m] if m.any() else s.ravel()
    lh = float(dims.layer_h)
    hist = np.bincount(vals, minlength=dims.relief_layers + 1)[:dims.relief_layers + 1]
    return {
        'relief_layers_min': int(vals.min()), 'relief_layers_max': int(vals.max()),
        'relief_mm_min': round(float(vals.min()) * lh, 6), 'relief_mm_max': round(float(vals.max()) * lh, 6),
        'relief_mm_mean': round(float(vals.mean()) * lh, 6),
        'levels_used': int(np.count_nonzero(hist)),
        'histogram_layers': [int(v) for v in hist],
        'share_at_max_budget': float(np.mean(vals == dims.relief_layers)),
        'total_thickness_mm_min': round((dims.base_layers + int(vals.min()) + dims.shell_layers) * lh, 6),
        'total_thickness_mm_max': round((dims.base_layers + int(vals.max()) + dims.shell_layers) * lh, 6),
        'neighbor_steps': neighbor_step_stats(s, m),
    }


# --------------------------------------------------------------------------- main entry
def process_depth(geom: GeometryResult, target_hw: tuple[int, int], dims: ReliefDims,
                  params: Optional[DepthParams] = None, foreground_mask: Optional[np.ndarray] = None,
                  plaque_mask: Optional[np.ndarray] = None) -> ReliefField:
    """Provider geometry -> quantised relief on the (H, W) Stack5 grid.

    foreground_mask: overrides ``geom.mask`` (any resolution, resampled NEAREST).
    plaque_mask:     Stack5 ``mask_solid`` (H, W); pixels outside it are air and
                     never constrain the relief.
    """
    params = params or DepthParams()
    H, W = int(target_hw[0]), int(target_hw[1])
    report: dict = {'params': params.to_dict(), 'provider': geom.describe(), 'warnings': []}

    depth = geom.raised()
    # foreground mask at provider resolution
    fg_src = None
    if foreground_mask is not None:
        fg_src = align_mask(np.asarray(foreground_mask), depth.shape)
        report['mask_source'] = 'foreground_mask'
    elif geom.mask is not None:
        fg_src = np.asarray(geom.mask).astype(bool)
        report['mask_source'] = 'provider'
    else:
        report['mask_source'] = None
    if fg_src is not None and (not fg_src.any() or fg_src.all()):
        if not fg_src.any():
            report['warnings'].append("foreground mask is empty; ignoring it")
        fg_src = None
    # neither side must bleed into the other at the silhouette when the grid is
    # resampled: resample the object padded with its own nearest values and the
    # background padded with ITS nearest values, then pick per pixel with the
    # NEAREST-resampled mask (keeps the background depth intact for normalisation)
    if fg_src is not None:
        depth_fg, align_info = align_depth(nearest_fill(depth, fg_src), (H, W), params.resize)
        depth_bg, _ = align_depth(nearest_fill(depth, ~fg_src), (H, W), params.resize)
        fg = align_mask(fg_src, (H, W))
        depth_g = np.where(fg, depth_fg, depth_bg).astype(np.float32)
    else:
        depth_g, align_info = align_depth(depth, (H, W), params.resize)
        fg = None
    report['align'] = align_info
    report['warnings'] += align_info.get('warnings', [])

    plaque = None if plaque_mask is None else np.asarray(plaque_mask).astype(bool)
    if plaque is not None and plaque.shape != (H, W):
        raise ValueError("plaque_mask must be (H, W)")
    if fg is not None and plaque is not None:
        fg = fg & plaque
    if fg is not None and not fg.any():
        report['warnings'].append("foreground mask is empty inside the plaque; ignoring it")
        fg = None
    if params.flatten_background and fg is None:
        report['warnings'].append("flatten_background requested but there is no foreground mask "
                                  "(--foreground-mask, depth alpha or provider mask): nothing flattened")
    if (fg is not None and not params.flatten_background and report['mask_source'] == 'provider'
            and geom.meta.get('mask_from_alpha')):
        report['warnings'].append("the depth file's alpha marks a background whose depth values are undefined; "
                                  "consider --flatten-background")

    norm, ninfo = normalize_depth(depth_g, params, mask=fg if fg is not None else plaque)
    if ninfo.get('flat') and fg is not None:
        # a constant foreground (e.g. a cut-out object) still has a height relative
        # to its background: normalise over the whole plaque instead
        norm, ninfo = normalize_depth(depth_g, params, mask=plaque)
        ninfo['fallback'] = 'foreground depth is constant; normalised over the whole plaque'
    report['normalize'] = ninfo
    flat = bool(ninfo.get('flat'))
    if ninfo.get('warning'):
        report['warnings'].append(ninfo['warning'])
    if flat:
        report['warnings'].append("flat depth: invert, background flattening, smoothing and slope limiting are skipped")
    if not flat and params.invert:
        norm = (1.0 - norm).astype(np.float32)
    flatten = bool(params.flatten_background and fg is not None and not flat)
    if flatten:
        norm[~fg] = np.float32(params.background_level)
    report['background_flattened'] = flatten

    # region whose pixels constrain each other; the cut-out is always excluded
    region = plaque if plaque is not None else np.ones((H, W), bool)
    protect_fg = bool(params.protect_silhouette and fg is not None)
    protect = (fg & region) if protect_fg else region
    smooth_region = protect if flatten else region
    smooth_mask = smooth_region if not smooth_region.all() else None
    if not flat:
        sel = norm[smooth_region] if smooth_region.any() else norm.ravel()
        lo0, hi0 = float(sel.min()), float(sel.max())
        norm = smooth_depth(norm, params, mask=smooth_mask)
        # smoothing pulls extremes inward (a ramp loses a layer at each end): map the
        # smoothed range back onto the PRE-smoothing range, so the relief budget is
        # really used but an authored map (normalize='none') keeps its headroom
        if float(params.smoothing_px) > 0 or int(params.median_px) > 1:
            sel = norm[smooth_region] if smooth_region.any() else norm.ravel()
            lo, hi = float(sel.min()), float(sel.max())
            if hi - lo > 1e-6 and hi0 - lo0 > 1e-6:
                stretched = np.clip(lo0 + (norm - lo) / (hi - lo) * (hi0 - lo0), 0.0, 1.0).astype(np.float32)
                if flatten:
                    stretched[~fg] = norm[~fg]
                norm = stretched
                report['restretched_after_smoothing'] = {'smoothed': [lo, hi], 'restored_to': [lo0, hi0]}

    steps = quantize_steps(norm, dims, params.step_layers)
    if not flat:
        steps = morphological_clean(steps, params.min_feature_px, mask=smooth_mask)
        # slope limiting: with a protected silhouette the foreground and the background
        # are limited separately (neither side constrains the other across the outline)
        if protect_fg:
            steps, info_fg = limit_neighbor_step(steps, params.max_neighbor_step, mask=fg & region,
                                                 connectivity=params.neighbor_connectivity)
            bg = region & ~fg
            if not flatten and bg.any():
                steps, info_bg = limit_neighbor_step(steps, params.max_neighbor_step, mask=bg,
                                                     connectivity=params.neighbor_connectivity)
            else:
                info_bg = {'applied': False, 'reason': 'background flattened' if flatten else 'no background'}
            report['slope_limit'] = {'foreground': info_fg, 'background': info_bg,
                                     'applied': bool(info_fg.get('applied')), 'silhouette_protected': True}
        else:
            steps, info = limit_neighbor_step(steps, params.max_neighbor_step,
                                              mask=region if not region.all() else None,
                                              connectivity=params.neighbor_connectivity)
            report['slope_limit'] = dict(info, silhouette_protected=False)
        if flatten and protect_fg:
            # the background was excluded from cleanup / limiting: restore its flat level
            bg_steps = quantize_steps(np.full((1, 1), params.background_level, np.float32), dims, params.step_layers)[0, 0]
            steps[~fg] = bg_steps
            report['background_steps'] = int(bg_steps)
    else:
        report['slope_limit'] = {'applied': False, 'reason': 'flat depth'}
    if plaque is not None:
        steps[~plaque] = 0
    steps = np.clip(steps, 0, dims.relief_layers).astype(np.int16)
    report['stats'] = relief_stats(steps, dims, mask=plaque)
    report['dims'] = dims.summary()
    report['warnings'] += dims.rounding_warnings()
    return ReliefField(steps=steps, normalized=norm.astype(np.float32), mask=fg, dims=dims, report=report)
