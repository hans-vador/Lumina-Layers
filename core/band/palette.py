"""Palette / thickness optimiser for Band mode.

Cost of a schedule S against an image histogram (Lab centres c_j, weights w_j):

    cost(S) = sum_j w_j * min_k dE(c_j, r_k)  +  lambda_K * K
              + band_penalty * #{bands with pixel share < min_band_share}

with dE = sqrt(wL*dL^2 + da^2 + db^2) and r_k the ramp colours of S.

Search (vectorised numpy):
  Stage 1 - every subset of the library (C(12,5)=792 and C(12,4)=495), ordered
            by luminance ascending (darkest = base), scored on 3 canonical
            thickness vectors; keep the best ``stage1_top`` subsets per size.
  Stage 2 - for each kept subset, all thickness vectors n in [lo..hi]^(B-1)
            with n0 + sum(n) <= max_layers.
  Pick 4 filaments when cost4 <= 1.05 * cost5 (one fewer swap).

Ramp composition is vectorised over (schedule, k, band) so the whole run takes
a few seconds; only analytic optical models (with ``.opacity``) are supported.
"""
from __future__ import annotations

import itertools
import os
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from core.band.optics import (Filament, composite_bands, linear_rgb_to_lab_d65,
                              luminance_y, srgb_to_lab_d65)
from core.band.schedule import BandSchedule
from core.band.tone import ToneCurve

Hist = tuple[np.ndarray, np.ndarray]     # (lab (k,3), weights (k,))
ALPHA_TRANSPARENT = 10                   # same threshold as core/image_processing.py (alpha < 10)

# Canonical thickness vectors for stage 1, keyed by number of bands above the base.
# All sum to <= 18 so that n0=9 + sum <= 27 layers.
_CANONICAL = {
    1: [(4,), (8,), (12,)],
    2: [(6, 8), (4, 8), (8, 8)],
    3: [(4, 6, 8), (5, 5, 6), (3, 5, 8)],
    4: [(3, 4, 5, 6), (4, 4, 4, 6), (2, 3, 5, 8)],
}


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #
def image_hist_lab(image_path: str, k: int = 64, size: int = 256,
                   center_weight: float = 0.0) -> Hist:
    """Resize to size x size (LANCZOS), true CIELAB, k-means -> (lab (k,3), w (k,)).

    center_weight > 0 multiplies pixel weights by a Gaussian window centred on
    the image (1 at the centre, ~exp(-center_weight) at the corners).
    Transparent pixels (alpha < 10, the processor's threshold) get zero weight
    so a transparent background never biases the palette.
    """
    img = Image.open(image_path).convert('RGBA').resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    rgb = arr[..., :3] / 255.0
    alpha = arr[..., 3]
    lab = srgb_to_lab_d65(rgb.reshape(-1, 3))
    if center_weight > 0:
        yy, xx = np.mgrid[0:size, 0:size]
        r2 = ((yy - (size - 1) / 2) ** 2 + (xx - (size - 1) / 2) ** 2) / (2 * ((size - 1) / 2) ** 2)
        pix_w = np.exp(-center_weight * r2).reshape(-1)
    else:
        pix_w = np.ones(lab.shape[0])
    pix_w = pix_w * (alpha.reshape(-1) >= ALPHA_TRANSPARENT)
    if pix_w.sum() <= 0:
        raise ValueError(f"{image_path}: every pixel is transparent")
    return kmeans_hist(lab, pix_w, k)


def kmeans_hist(lab: np.ndarray, pix_w: np.ndarray | None = None, k: int = 64) -> Hist:
    """K-means (cv2 if available) of Lab samples -> weighted centres."""
    lab = np.asarray(lab, dtype=np.float64).reshape(-1, 3)
    n = lab.shape[0]
    if pix_w is None:
        pix_w = np.ones(n)
    uniq = np.unique(np.round(lab, 3), axis=0)
    k_eff = int(max(1, min(k, uniq.shape[0])))
    if k_eff >= uniq.shape[0]:
        # few distinct colours: exact histogram
        keys = np.round(lab, 3)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        counts = np.bincount(inv, weights=pix_w, minlength=uniq.shape[0])
        centres = np.zeros((uniq.shape[0], 3))
        for c in range(3):
            centres[:, c] = np.bincount(inv, weights=lab[:, c] * pix_w, minlength=uniq.shape[0])
        centres /= np.maximum(counts, 1e-12)[:, None]
        labels = None
    else:
        try:
            import cv2
            data = lab.astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.25)
            _, labels, centres = cv2.kmeans(data, k_eff, None, criteria, 1, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape(-1)
            centres = centres.astype(np.float64)
        except Exception:
            labels, centres = _numpy_kmeans(lab, k_eff)
        counts = np.bincount(labels, weights=pix_w, minlength=centres.shape[0])
    keep = counts > 0
    centres, counts = centres[keep], counts[keep]
    w = counts / counts.sum()
    order = np.argsort(-w)
    return centres[order], w[order]


def _numpy_kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = x[rng.choice(x.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        d = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for c in range(k):
            m = labels == c
            if m.any():
                centres[c] = x[m].mean(0)
    return labels, centres


def merge_hists(hists: Sequence[Hist]) -> Hist:
    """Concatenate per-image histograms with equal total weight per image."""
    labs = [np.asarray(h[0], dtype=np.float64) for h in hists]
    ws = [np.asarray(h[1], dtype=np.float64) / (np.asarray(h[1]).sum() * len(hists)) for h in hists]
    return np.concatenate(labs, 0), np.concatenate(ws, 0)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
# Palette scoring: 'vivid' (LCH with chroma/hue terms, chroma-weighted pixels) or
# 'plain' (weighted Lab dE). Override with env BAND_SCORE_MODE.
SCORE_MODE = os.environ.get('BAND_SCORE_MODE', 'vivid')
VIVID_WC = float(os.environ.get('BAND_VIVID_WC', '1.5'))
VIVID_WH = float(os.environ.get('BAND_VIVID_WH', '1.0'))
VIVID_MU = float(os.environ.get('BAND_VIVID_MU', '0.8'))
# artist heuristics: darkest filament as base when the art has real shadows,
# brightest as top when it has highlights (HueForge convention)
DARK_FRAC_FOR_BLACK_BASE = 0.05     # weight of pixels with L* < 20
BRIGHT_FRAC_FOR_WHITE_TOP = 0.03    # weight of pixels with L* > 85


def _weighted_de2(ramp_lab: np.ndarray, lab: np.ndarray, wL: float) -> np.ndarray:
    """ramp_lab (..., K', 3), lab (k,3) -> squared dE (..., K', k)."""
    diff = ramp_lab[..., :, None, :] - lab[None, :, :]
    return wL * diff[..., 0] ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2


def score(lab: np.ndarray, w: np.ndarray, ramp_lab: np.ndarray, wL: float = 1.5):
    """sum_j w_j * min_k dE(c_j, r_k). ramp_lab (K',3) -> float; (M,K',3) -> (M,)."""
    lab = np.asarray(lab, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ramp_lab = np.asarray(ramp_lab, dtype=np.float64)
    squeeze = ramp_lab.ndim == 2
    if squeeze:
        ramp_lab = ramp_lab[None]
    d2 = _weighted_de2(ramp_lab, lab, wL)          # (M, K', k)
    s = (np.sqrt(d2.min(axis=1)) * w).sum(-1)
    return float(s[0]) if squeeze else s


def _opacity_fn(model):
    fn = getattr(model, 'opacity', None)
    if not callable(fn):
        raise TypeError(
            f"palette search needs an analytic optical model with .opacity(t, td) "
            f"(BeerLambertModel / LinearTDModel), got {type(model).__name__}")
    return fn


def thickness_vectors(n_above: int, lo: int, hi: int, max_sum: int) -> np.ndarray:
    """All integer vectors in [lo..hi]^n_above with sum <= max_sum, shape (M, n_above)."""
    if n_above == 0:
        return np.zeros((1, 0), dtype=np.int64)
    axes = [np.arange(lo, hi + 1)] * n_above
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, n_above)
    return grid[grid.sum(1) <= max_sum].astype(np.int64)


def ramp_lab_batch(rgb_lin: np.ndarray, td: np.ndarray, nvec: np.ndarray, n0: int,
                   lh: float, model, max_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ramp colours.

    rgb_lin (M,B,3), td (M,B), nvec (M,B-1) -> (lab (M,K',3), K_total (M,))
    with K' = max_layers - n0 + 1 rows for k = n0..max_layers.  Rows with
    k > K_total(m) repeat the top colour (harmless for a min over k).
    """
    opacity = _opacity_fn(model)
    rgb_lin = np.asarray(rgb_lin, dtype=np.float64)
    td = np.asarray(td, dtype=np.float64)
    nvec = np.asarray(nvec, dtype=np.int64)
    M, B = rgb_lin.shape[0], rgb_lin.shape[1]
    kp = max_layers - n0 + 1
    k = n0 + np.arange(kp)                                                  # (K',)
    if B == 1:
        lab = linear_rgb_to_lab_d65(np.broadcast_to(rgb_lin[:, None, 0, :], (M, kp, 3)))
        return lab, np.full(M, n0)
    cum_prev = n0 + np.concatenate([np.zeros((M, 1), dtype=np.int64),
                                    np.cumsum(nvec[:, :-1], axis=1)], axis=1)   # (M,B-1)
    layers = np.clip(k[None, :, None] - cum_prev[:, None, :], 0, nvec[:, None, :])  # (M,K',B-1)
    t = layers.astype(np.float64) * lh
    o = opacity(t, td[:, None, 1:])
    c = composite_bands(rgb_lin[:, None, 0, :], rgb_lin[:, None, 1:, :], o)     # (M,K',3)
    return linear_rgb_to_lab_d65(c), n0 + nvec.sum(1)


def matcher_wL(luma_weight: float) -> float:
    """True-Lab squared lightness weight equivalent to BandProcessor's matcher.

    The matcher works in OpenCV 8-bit Lab (L* scaled by 2.55, a*/b* unscaled)
    with the L axis multiplied by ``luma_weight``, so its squared distance is
    (2.55*luma_weight)^2 * dL*^2 + da*^2 + db*^2.
    """
    return float((2.55 * float(luma_weight)) ** 2)


def tone_assignment(lab_L: np.ndarray, K: np.ndarray, tone: ToneCurve, n0: int, lh: float,
                    first_layer_mm: float, base_min_layers: int) -> np.ndarray:
    """Row index (k - n0) that the luminance-mapping processor gives each
    histogram colour, for schedules with K (m,) layers -> (m, k) int.

    Mirrors BandProcessor(height_mode='luminance'): h = z_lo + (z_hi - z_lo) *
    tone(L*), z_lo = engraving floor, z_hi = top_z(K); slice-plane rule; rows
    clipped to n0..K.
    """
    K = np.asarray(K, dtype=np.float64).reshape(-1)
    t = tone.t(np.asarray(lab_L, dtype=np.float64))                       # (k,)
    z_lo = first_layer_mm + lh * (base_min_layers - 1) + lh / 2.0 + 1e-3
    z_hi = first_layer_mm + lh * (K - 1.0)                                # (m,)
    z = z_lo + (z_hi[:, None] - z_lo) * t[None, :]                        # (m,k)
    n = np.floor((z - first_layer_mm) / lh + 1.5)
    n = np.clip(n, n0, K[:, None])
    return (n - n0).astype(np.int64)


def evaluate_schedules(hist: Hist, rgb_lin: np.ndarray, td: np.ndarray, nvec: np.ndarray,
                       n0: int, lh: float, model, max_layers: int, wL: float = 1.5,
                       lambda_K: float = 0.02, min_band_share: float = 0.005,
                       band_penalty: float = 5.0, chunk: int = 2048,
                       wL_match: float | None = None, tone: ToneCurve | None = None,
                       first_layer_mm: float = 0.16, base_min_layers: int = 5) -> dict:
    """Cost of M schedules (shared shapes with ramp_lab_batch). Returns arrays (M,...).

    Assignment of histogram colours to ramp rows (must mirror the processor):
    * tone given (height_mode='luminance'): the tone curve maps each colour's
      L* to a layer, whatever the ramp looks like; the perceptual error (wL) of
      the ramp colour at that layer is scored.  This is the default pipeline.
    * wL_match given (height_mode='match'): nearest row under the pixel
      matcher's metric (see :func:`matcher_wL`), scored with ``wL``.
    * neither: assign and score with the same ``wL`` (pure nearest colour).
    Predicting the assignment matters: a luminance-driven assignment puts every
    pixel of a given lightness on the same row whatever its hue, so a palette
    that looks good under nearest-colour scoring can render a gold mid-tone
    green if a green band happens to own that lightness range.
    """
    lab, w = np.asarray(hist[0], dtype=np.float64), np.asarray(hist[1], dtype=np.float64)
    M, B = rgb_lin.shape[0], rgb_lin.shape[1]
    costs = np.empty(M)
    scores = np.empty(M)
    shares = np.empty((M, B))
    ks = np.empty(M, dtype=np.int64)
    cums_all = n0 + np.concatenate([np.zeros((M, 1), dtype=np.int64),
                                    np.cumsum(nvec, axis=1)], axis=1)      # (M,B): cum(b)
    band_ids = np.arange(B)
    same_metric = wL_match is None or abs(float(wL_match) - float(wL)) < 1e-12
    base_min_layers = int(min(base_min_layers, n0))
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        ramp, K = ramp_lab_batch(rgb_lin[s:e], td[s:e], nvec[s:e], n0, lh, model, max_layers)
        diff = ramp[:, :, None, :] - lab[None, None, :, :]  # (m,K',k,3)
        dL2 = diff[..., 0] ** 2
        dab2 = diff[..., 1] ** 2 + diff[..., 2] ** 2
        d2 = wL * dL2 + dab2                                 # (m,K',k) perceptual
        if tone is not None:
            idx = tone_assignment(lab[:, 0], K, tone, n0, lh, first_layer_mm, base_min_layers)
        elif same_metric:
            idx = d2.argmin(axis=1)                          # (m,k)
        else:
            idx = (float(wL_match) * dL2 + dab2).argmin(axis=1)   # matcher's assignment
        if SCORE_MODE == 'vivid':
            # LCH error: a neutral ramp is charged for the chroma it discards and
            # vivid pixels weigh more, so saturated art gets a saturated palette
            # (a hue the palette cannot reach costs at most ~2*min(C), not dE).
            rsel = np.take_along_axis(ramp, idx[:, :, None], axis=1)       # (m,k,3)
            c_r = np.hypot(rsel[..., 1], rsel[..., 2])
            c_p = np.hypot(lab[:, 1], lab[:, 2])[None, :]
            dh = np.arctan2(rsel[..., 2], rsel[..., 1]) - np.arctan2(lab[:, 2], lab[:, 1])[None, :]
            dH2 = (2.0 * np.sqrt(c_r * c_p) * np.sin(dh / 2.0)) ** 2
            dLs2 = (rsel[..., 0] - lab[:, 0][None, :]) ** 2
            dmin = np.sqrt(wL * dLs2 + VIVID_WC * (c_r - c_p) ** 2 + VIVID_WH * dH2)
            wv = w * (0.4 + c_p[0] / 50.0)
            sc = (dmin * wv).sum(1) / wv.sum() * w.sum()
        else:
            dmin = np.sqrt(np.take_along_axis(d2, idx[:, None, :], axis=1)[:, 0, :])
            sc = (dmin * w).sum(1)
        ksel = np.minimum(n0 + idx, K[:, None])              # (m,k) layers of matched row
        band = np.minimum((ksel[:, :, None] > cums_all[s:e][:, None, :]).sum(-1), B - 1)
        onehot = (band[:, :, None] == band_ids[None, None, :]).astype(np.float64)
        sh = (onehot * w[None, :, None]).sum(1)              # (m,B)
        pen = band_penalty * (sh < min_band_share).sum(1)
        if SCORE_MODE == 'vivid':
            # vividness prior: a ramp whose usage-weighted chroma falls below the
            # image's chroma is charged for the saturation it throws away
            fil_lab = linear_rgb_to_lab_d65(rgb_lin[s:e])                    # (m,B,3)
            c_fil = np.hypot(fil_lab[..., 1], fil_lab[..., 2])                # (m,B)
            c_img = float((np.hypot(lab[:, 1], lab[:, 2]) * w).sum() / max(w.sum(), 1e-9))
            c_ramp = (sh * c_fil).sum(1) / np.maximum(sh.sum(1), 1e-9)
            pen = pen + VIVID_MU * np.clip(c_img - c_ramp, 0.0, None) * w.sum()
        costs[s:e] = sc + lambda_K * K + pen
        scores[s:e] = sc
        shares[s:e] = sh
        ks[s:e] = K
    return {'cost': costs, 'score': scores, 'shares': shares, 'K': ks}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def order_by_luminance(names: Iterable[str], library: Mapping[str, Filament]) -> tuple[str, ...]:
    return tuple(sorted(names, key=lambda n: library[n].luminance))


def _fil_arrays(orders: Sequence[Sequence[str]], library: Mapping[str, Filament]):
    rgb = np.array([[library[n].rgb_lin for n in o] for o in orders], dtype=np.float64)
    td = np.array([[library[n].td_mm for n in o] for o in orders], dtype=np.float64)
    return rgb, td


def _orders_for_subset(subset: Sequence[str], library, orders: str) -> list[tuple[str, ...]]:
    lum = order_by_luminance(subset, library)
    if orders == 'luminance':
        return [lum]
    if orders == 'all':
        return [tuple(p) for p in itertools.permutations(lum)]
    if orders == 'ends':
        B = len(lum)
        if B < 3:
            return [lum]
        out = set()
        for base in lum[:2]:
            for top in lum[-2:]:
                if top == base:
                    continue
                mid = [n for n in lum if n not in (base, top)]
                for p in itertools.permutations(mid):
                    out.add((base,) + tuple(p) + (top,))
        return sorted(out)
    raise ValueError(f"orders must be 'luminance', 'ends' or 'all', got {orders!r}")


def _report(hist, names, counts, res, i, n0, fl, lh, model, wL, wL_match=None, tone=None) -> dict:
    return {
        'filament_names': list(names),
        'layer_counts': [int(n0)] + [int(c) for c in counts],
        'K': int(res['K'][i]),
        'cost': float(res['cost'][i]),
        'mean_dE': float(res['score'][i]),
        'band_shares': [float(x) for x in res['shares'][i]],
        'n_changes': len(names) - 1,
        'model': getattr(model, 'name', type(model).__name__),
        'model_repr': repr(model),
        'wL': wL,
        'wL_match': wL_match,
        'assignment': 'luminance' if tone is not None else ('matcher' if wL_match is not None else 'nearest'),
        'tone_curve': tone.to_dict() if tone is not None else None,
    }


# --------------------------------------------------------------------------- #
# Public optimisers
# --------------------------------------------------------------------------- #
def optimise_thickness_detailed(hist: Hist, filament_names: Sequence[str],
                                library: Mapping[str, Filament], model, n0: int = 9,
                                max_layers: int = 27, thickness_range=(2, 8),
                                first_layer_mm: float = 0.16, layer_h: float = 0.08,
                                base_min_layers: int = 5, wL: float = 1.5,
                                lambda_K: float = 0.02, min_band_share: float = 0.005,
                                band_penalty: float = 5.0,
                                wL_match: float | None = None,
                                tone: ToneCurve | None = None) -> tuple[BandSchedule, dict]:
    """Fixed filament order -> best band thicknesses (exhaustive, vectorised)."""
    names = tuple(filament_names)
    if n0 < 1:
        raise ValueError("n0 must be >= 1")
    base_min_layers = int(min(base_min_layers, n0))       # BandSchedule needs base_min_layers <= n0
    nvec = thickness_vectors(len(names) - 1, thickness_range[0], thickness_range[1], max_layers - n0)
    if nvec.shape[0] == 0:
        raise ValueError("no feasible thickness vector: relax thickness_range or max_layers")
    rgb1, td1 = _fil_arrays([names], library)
    rgb = np.repeat(rgb1, nvec.shape[0], axis=0)
    td = np.repeat(td1, nvec.shape[0], axis=0)
    res = evaluate_schedules(hist, rgb, td, nvec, n0, layer_h, model, max_layers, wL,
                             lambda_K, min_band_share, band_penalty, wL_match=wL_match, tone=tone,
                             first_layer_mm=first_layer_mm, base_min_layers=base_min_layers)
    i = int(np.argmin(res['cost']))
    counts = (n0,) + tuple(int(c) for c in nvec[i])
    sched = BandSchedule(names, counts, first_layer_mm, layer_h, base_min_layers)
    return sched, _report(hist, names, nvec[i], res, i, n0, first_layer_mm, layer_h, model, wL,
                          wL_match, tone)


def optimise_thickness(hist: Hist, filament_names: Sequence[str], library: Mapping[str, Filament],
                       model, n0: int = 9, max_layers: int = 27, thickness_range=(2, 8),
                       **kw) -> BandSchedule:
    return optimise_thickness_detailed(hist, filament_names, library, model, n0, max_layers,
                                       thickness_range, **kw)[0]


def _search_size(hist: Hist, library: Mapping[str, Filament], model, B: int, n0: int,
                 max_layers: int, thickness_range, stage1_top: int, orders: str,
                 first_layer_mm: float, layer_h: float, base_min_layers: int, wL: float,
                 lambda_K: float, min_band_share: float, band_penalty: float,
                 candidates: Sequence[str] | None = None,
                 wL_match: float | None = None,
                 tone: ToneCurve | None = None) -> tuple[BandSchedule, dict]:
    """Two-stage search over all B-subsets of the library."""
    pool = list(candidates) if candidates is not None else list(library.keys())
    if B < 1:
        raise ValueError("need at least one filament")
    if len(pool) < B:
        raise ValueError(f"library has {len(pool)} filaments, need {B}")
    base_min_layers = int(min(base_min_layers, n0))
    common = dict(wL_match=wL_match, tone=tone, first_layer_mm=first_layer_mm,
                  base_min_layers=base_min_layers)
    subsets = [order_by_luminance(c, library) for c in itertools.combinations(pool, B)]
    if SCORE_MODE == 'vivid' and B >= 3:
        lab_h, w_h = np.asarray(hist[0], float), np.asarray(hist[1], float)
        wsum = max(float(w_h.sum()), 1e-9)
        dark = float(w_h[lab_h[:, 0] < 20].sum()) / wsum
        bright = float(w_h[lab_h[:, 0] > 85].sum()) / wsum
        darkest = min(pool, key=lambda n: library[n].luminance)
        brightest = max(pool, key=lambda n: library[n].luminance)
        want = []
        if dark >= DARK_FRAC_FOR_BLACK_BASE:
            want.append(darkest)
        if bright >= BRIGHT_FRAC_FOR_WHITE_TOP:
            want.append(brightest)
        filt = [sub for sub in subsets if all(n in sub for n in want)]
        if filt:
            subsets = filt
    lo, hi = thickness_range

    if B == 1:
        # single filament: base only (K = n0, no swaps); pick the best base colour
        rgb_s, td_s = _fil_arrays(subsets, library)                 # (S,1,3),(S,1)
        nvec = np.zeros((len(subsets), 0), dtype=np.int64)
        res = evaluate_schedules(hist, rgb_s, td_s, nvec, n0, layer_h, model, max_layers, wL,
                                 lambda_K, min_band_share, band_penalty, **common)
        i = int(np.argmin(res['cost']))
        names = subsets[i]
        sched = BandSchedule(names, (n0,), first_layer_mm, layer_h, base_min_layers)
        rep = _report(hist, names, (), res, i, n0, first_layer_mm, layer_h, model, wL, wL_match, tone)
        rep['stage1_subsets'] = [list(s) for s in subsets]
        rep['stage2_evaluations'] = int(len(subsets))
        return sched, rep

    canon = np.array(_CANONICAL.get(B - 1) or [tuple([max(1, (max_layers - n0) // (B - 1))] * (B - 1))],
                     dtype=np.int64)
    canon = canon[canon.sum(1) <= max_layers - n0]
    canon = np.clip(canon, lo, hi)

    # Stage 1
    rgb_s, td_s = _fil_arrays(subsets, library)                     # (S,B,3),(S,B)
    S, C = len(subsets), canon.shape[0]
    rgb = np.repeat(rgb_s, C, axis=0)
    td = np.repeat(td_s, C, axis=0)
    nvec = np.tile(canon, (S, 1))
    res1 = evaluate_schedules(hist, rgb, td, nvec, n0, layer_h, model, max_layers, wL,
                              lambda_K, min_band_share, band_penalty, **common)
    cost_subset = res1['cost'].reshape(S, C).min(1)
    keep = np.argsort(cost_subset)[:max(1, min(stage1_top, S))]

    # Stage 2
    full = thickness_vectors(B - 1, lo, hi, max_layers - n0)
    orders_list: list[tuple[str, ...]] = []
    for si in keep:
        orders_list.extend(_orders_for_subset(subsets[si], library, orders))
    rgb_o, td_o = _fil_arrays(orders_list, library)                 # (O,B,3)
    O, Mv = len(orders_list), full.shape[0]
    rgb = np.repeat(rgb_o, Mv, axis=0)
    td = np.repeat(td_o, Mv, axis=0)
    nvec = np.tile(full, (O, 1))
    res2 = evaluate_schedules(hist, rgb, td, nvec, n0, layer_h, model, max_layers, wL,
                              lambda_K, min_band_share, band_penalty, **common)
    i = int(np.argmin(res2['cost']))
    names = orders_list[i // Mv]
    counts = (n0,) + tuple(int(c) for c in nvec[i])
    sched = BandSchedule(names, counts, first_layer_mm, layer_h, base_min_layers)
    rep = _report(hist, names, nvec[i], res2, i, n0, first_layer_mm, layer_h, model, wL, wL_match, tone)
    rep['stage1_subsets'] = [list(subsets[s]) for s in keep]
    rep['stage2_evaluations'] = int(O * Mv)
    return sched, rep


def select_palette_detailed(hist: Hist, library: Mapping[str, Filament], model,
                            n_filaments=(4, 5), n0: int = 9, max_layers: int = 27,
                            stage1_top: int = 12, thickness_range=(2, 8),
                            orders: str = 'luminance', prefer_fewer_ratio: float = 1.05,
                            first_layer_mm: float = 0.16, layer_h: float = 0.08,
                            base_min_layers: int = 5, wL: float = 1.5, lambda_K: float = 0.02,
                            min_band_share: float = 0.005, band_penalty: float = 5.0,
                            candidates: Sequence[str] | None = None,
                            wL_match: float | None = None,
                            tone: ToneCurve | None = None) -> tuple[BandSchedule, dict]:
    """Pick filaments (subset + order) and thicknesses for one histogram.

    Evaluates each size in ``n_filaments``; a smaller palette wins when its cost
    is <= prefer_fewer_ratio * (cost of the next larger one).
    wL_match / tone: how histogram colours are assigned to ramp rows - must
    mirror the processor's height_mode (see evaluate_schedules).
    """
    if isinstance(n_filaments, int):
        n_filaments = (n_filaments,)
    sizes = sorted(set(int(b) for b in n_filaments))
    if not sizes or sizes[0] < 1 or sizes[-1] > 5:
        raise ValueError("n_filaments must be within 1..5")
    t0 = time.perf_counter()
    results = {}
    for B in sizes:
        results[B] = _search_size(hist, library, model, B, n0, max_layers, thickness_range,
                                  stage1_top, orders, first_layer_mm, layer_h, base_min_layers,
                                  wL, lambda_K, min_band_share, band_penalty, candidates,
                                  wL_match=wL_match, tone=tone)
    # largest palette is the reference; step down while the smaller one is close enough
    best_B = sizes[-1]
    for B in reversed(sizes[:-1]):
        if results[B][1]['cost'] <= prefer_fewer_ratio * results[best_B][1]['cost']:
            best_B = B
    sched, rep = results[best_B]
    rep = dict(rep)
    rep['alternatives'] = {str(B): {'filament_names': r[1]['filament_names'],
                                    'layer_counts': r[1]['layer_counts'],
                                    'cost': r[1]['cost'], 'mean_dE': r[1]['mean_dE']}
                           for B, r in results.items()}
    rep['seconds'] = time.perf_counter() - t0
    return sched, rep


def select_palette(hist: Hist, library: Mapping[str, Filament], model, n_filaments=(4, 5),
                   n0: int = 9, max_layers: int = 27, stage1_top: int = 12, **kw) -> BandSchedule:
    """See :func:`select_palette_detailed`; returns only the BandSchedule."""
    return select_palette_detailed(hist, library, model, n_filaments, n0, max_layers,
                                   stage1_top, **kw)[0]


def select_global_order(hists: Sequence[Hist], library: Mapping[str, Filament], model,
                        n_filaments=(4, 5), n0: int = 9, max_layers: int = 27,
                        stage1_top: int = 12, **kw) -> tuple[str, ...]:
    """Filament subset + order that best serves all histograms together.

    Thicknesses are per-print (optimise_thickness with this fixed order); the
    order is global so the AMS slots never need re-spooling.
    """
    merged = merge_hists(list(hists))
    sched = select_palette(merged, library, model, n_filaments, n0, max_layers, stage1_top, **kw)
    return sched.filament_names


def write_schedule_json(schedule: BandSchedule, report: dict | None, path: str) -> str:
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {'schedule': schedule.to_dict(), 'report': report or {}}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path
