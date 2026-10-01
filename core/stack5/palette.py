"""Palette search for Stack5 mode.

Lumina matches every (quantised) image colour to the NEAREST LUT colour, so the
predicted error of a configuration (5-filament subset P, backing b) against an
image histogram (true-CIELAB centres c_j, weights w_j) is

    cost_dE(P, b) = sum_j w_j * dE76(c_j, lut_{P,b}[argmin_j])

where lut_{P,b} is the synthetic LUT of P composited over backing b (all 3125
stacks) and argmin_j is the entry LUMINA WOULD PICK.

Matching metric (``metric``):

* ``'hue'`` (default) - hue-first LCh metric (core.stack5.metric.hue_dist2, true
  CIELAB): lightness error is down-weighted for chromatic colours and hue error
  is up-weighted, so a brown is matched to a lighter brown rather than to a dark
  olive of the same lightness.  Parameters via ``hue_params``.
* ``'lumina'`` - the argmin is computed exactly the way
  ``core.image_processing.LuminaImageProcessor`` does it: LUT colours rounded to
  uint8 sRGB, converted with ``cv2.cvtColor(..., COLOR_RGB2Lab)`` (8-bit Lab:
  L* scaled by 255/100, a*/b* offset by 128 and rounded) and matched by plain
  Euclidean distance (its KDTree).  That metric weights lightness 6.5x more than
  true CIELAB (2.55^2), so a search in true Lab ranks palettes differently from
  what Lumina then actually prints.
* ``'lab'`` - legacy weighted true-Lab argmin, dE^2 = wL*dL^2 + da^2 + db^2.

Whatever the metric, the REPORTED numbers (cost_dE, mean_dE, p90, max) are the
true CIELAB dE76 between the bin and the entry the argmin selected, so they stay
interpretable and the ``prefer`` tolerance stays in dE units.

Backing search: for every subset all n members are tried as the opaque backing
(S*n LUTs) unless ``backing`` fixes it; the (subset, backing) with the lowest
cost wins.  This is what lets a translucent spool's real colour (the pure stack
of the backing filament is the only stack that shows a spool colour exactly)
enter the optimisation - e.g. the Pink spool for pink art.

Scoring refinements (all off when their weight is 0):

* ``chroma_weight`` c: bin weights become w_j * (1 + c * C_j / 50).
* ``spool_bonus`` b: every DOMINANT bin (w_j >= dominant_w) within ``spool_de``
  (true dE) of a pure stack [s]*L of the configuration earns
  b * w_j * (1 - dE_pure_j / spool_de) off the cost.
* ``prefer`` (soft must-include): after ranking subsets (each with its best
  backing), the best subset containing all preferred spools replaces the
  optimum when its cost exceeds the optimum's by at most ``prefer_tol`` x the
  optimum's weighted dE.

Determinism: no RNG here; ``image_hist`` seeds OpenCV (cv2.setRNGSeed) before
``core.band.palette.image_hist_lab`` runs its k-means.
"""
from __future__ import annotations

import itertools
import time
from typing import Mapping, Sequence

import numpy as np

from core.band.optics import (BeerLambertModel, Filament, lab_to_linear_rgb_d65, linear_to_srgb,
                              srgb_to_lab_d65)
from core.band.palette import Hist, image_hist_lab
from core.stack5.lut import (LAYER_H, N_LAYERS, enumerate_stacks, is_pure, synth_lut_linear)
from core.stack5.metric import DEFAULT_HUE_PARAMS, hue_dist2, resolve_hue_params

DEFAULT_SPOOL_BONUS = 10.0      # dE-equivalents per unit weight for an exact spool match
DEFAULT_SPOOL_DE = 10.0         # a dominant bin within this dE of a pure stack is "exact"
DEFAULT_DOMINANT_W = 0.01       # bins holding >= 1 % of the pixels are dominant
DEFAULT_CHROMA_WEIGHT = 0.0
DEFAULT_PREFER_TOL = 0.10
DEFAULT_METRIC = 'hue'
METRICS = ('hue', 'lumina', 'lab')
DEFAULT_N_MIN = 2               # the search may use as few as this many spools ...
DEFAULT_SPOOL_PENALTY = 0.75    # ... each spool must lower the weighted cost by this much to earn its slot
DEFAULT_NEED_SHARE = 0.02       # a hue cluster needs this chroma-weighted pixel share to earn a spool
DEFAULT_CLUSTER_DH = 25.0       # image colours within this many degrees of hue form one cluster
DEFAULT_NEED_DH = 75.0          # a spool may serve a hue cluster within this many degrees (a slate blue is still blue)
NEED_SPOOL_CHROMA = 12.0        # spools below this C* are neutral (Black, White); Beige (C* 12.8) must earn its hue
NEED_BIN_CHROMA = 8.0           # image colours below this C* (pastel pink is ~11) or ...
NEED_BIN_L = 12.0               # ... darker than this L* are not visibly chromatic
LUMINA_L_SCALE = 255.0 / 100.0  # OpenCV 8-bit Lab: L* * 2.55, a*/b* + 128
LUMINA_WL_EQUIV = LUMINA_L_SCALE ** 2   # 6.5025 - the effective wL of Lumina's KDTree metric
BACKING_RULE_SEARCH = 'searched: every member of every subset, lowest cost wins'


def image_hist(image_path: str, k: int = 64, seed: int = 0, size: int = 256) -> Hist:
    """Deterministic image histogram: cv2.setRNGSeed(seed) then image_hist_lab."""
    try:
        import cv2
        cv2.setRNGSeed(int(seed))
    except Exception:
        pass
    np.random.seed(int(seed))
    return image_hist_lab(image_path, k=k, size=size)


# --------------------------------------------------------------------------- #
# Lumina's matching space
# --------------------------------------------------------------------------- #
def linear_to_uint8(lin: np.ndarray) -> np.ndarray:
    """Linear RGB (...,3) -> the uint8 sRGB Lumina's LUT file holds (synth_lut rounding)."""
    return np.clip(np.round(linear_to_srgb(lin) * 255.0), 0, 255).astype(np.uint8)


def lumina_lab_from_uint8(rgb8: np.ndarray) -> np.ndarray:
    """uint8 sRGB (...,3) -> OpenCV 8-bit Lab as float64 (...,3): exactly
    ``LuminaImageProcessor._rgb_to_lab`` (RGB->BGR->Lab; the channel swap is a
    no-op for the result)."""
    import cv2
    rgb8 = np.ascontiguousarray(np.asarray(rgb8, dtype=np.uint8))
    flat = rgb8.reshape(-1, 1, 3)
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2Lab)
    return lab.reshape(rgb8.shape).astype(np.float64)


def lab_true_to_uint8(lab: np.ndarray) -> np.ndarray:
    """True CIELAB (k,3) -> uint8 sRGB (clipped) - what a histogram centre looks
    like as an image colour."""
    return linear_to_uint8(lab_to_linear_rgb_d65(np.asarray(lab, dtype=np.float64)))


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (..., N, 3) x (k, 3) -> (..., N, k) via matmul."""
    a2 = (a * a).sum(-1)[..., :, None]
    b2 = (b * b).sum(-1)[None, :]
    d2 = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


class _Bins:
    """Histogram bins in every space the search needs."""

    def __init__(self, hist: Hist, chroma_weight: float, metric: str, wL: float,
                 hue_params: Mapping[str, float] | None = None):
        lab = np.asarray(hist[0], dtype=np.float64).reshape(-1, 3)
        w_raw = np.asarray(hist[1], dtype=np.float64).reshape(-1)
        if lab.shape[0] != w_raw.shape[0]:
            raise ValueError("histogram lab/weights length mismatch")
        w = w_raw.copy()
        if chroma_weight > 0:
            C = np.hypot(lab[:, 1], lab[:, 2])
            w = w * (1.0 + float(chroma_weight) * C / 50.0)
        self.lab = lab                       # true Lab (for reporting + bonus)
        self.w = w
        self.w_raw = w_raw
        self.metric = metric
        self.wL = float(wL)
        self.hue_params = resolve_hue_params(hue_params)
        if metric == 'hue':
            self.match = lab                                                  # true Lab, LCh weights applied in hue_dist2
        elif metric == 'lumina':
            self.match = lumina_lab_from_uint8(lab_true_to_uint8(lab))       # (k,3) cv2 Lab
        elif metric == 'lab':
            self.match = lab * np.array([np.sqrt(self.wL), 1.0, 1.0])       # weighted true Lab
        else:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")

    def lut_match_space(self, lin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """LUT linear RGB (...,N,3) -> (colours in the matching space, true Lab of
        the uint8-rounded LUT colour) - both (...,N,3)."""
        rgb8 = linear_to_uint8(lin)
        lab_true = srgb_to_lab_d65(rgb8.astype(np.float64) / 255.0)
        if self.metric == 'hue':
            return lab_true, lab_true
        if self.metric == 'lumina':
            return lumina_lab_from_uint8(rgb8), lab_true
        return lab_true * np.array([np.sqrt(self.wL), 1.0, 1.0]), lab_true


def _score(lin: np.ndarray, bins: _Bins, pure: np.ndarray, spool_bonus: float, spool_de: float,
           dominant_w: float):
    """lin (m,N,3) linear LUT colours -> dict of per-config arrays:
    cost (m,), cost_dE (m,) true dE76 of the picked entries, cost_fit (m,) the
    same in the MATCHING metric's own distance (== cost_dE for lumina/lab; the
    hue-first distance for 'hue' so the palette is chosen by what the pixels
    will be matched with), bonus (m,), idx (m,k) argmin entry per bin, dmin (m,k)
    true dE76 of that entry, dpure (m,k) true dE to the nearest pure stack."""
    match, lab_true = bins.lut_match_space(lin)                 # (m,N,3) each
    if bins.metric == 'hue':
        d2 = hue_dist2(bins.lab, lab_true, bins.hue_params)     # (m,N,k) hue-first LCh
    else:
        d2 = _sqdist(match, bins.match)                         # (m,N,k) in Lumina's / weighted space
    idx = d2.argmin(axis=1)                                     # (m,k)
    m = lin.shape[0]
    chosen = np.take_along_axis(lab_true, idx[..., None], axis=1)   # (m,k,3)
    dmin = np.linalg.norm(chosen - bins.lab[None, :, :], axis=-1)   # (m,k) true dE76
    dpure = np.sqrt(_sqdist(lab_true[:, pure, :], bins.lab).min(axis=1))   # (m,k) true dE
    if bins.metric == 'hue':
        d2min = np.take_along_axis(d2, idx[:, None, :], axis=1)[:, 0, :]
        fit = np.sqrt(np.maximum(d2min, 0.0)).astype(np.float64)               # (m,k) hue-first distance
        dpure_fit = np.sqrt(np.maximum(d2[:, pure, :].min(axis=1), 0.0)).astype(np.float64)
    else:
        fit, dpure_fit = dmin, dpure
    cost_de = (dmin * bins.w[None, :]).sum(axis=1)
    cost_fit = (fit * bins.w[None, :]).sum(axis=1)
    bonus = np.zeros(m, dtype=np.float64)
    if spool_bonus > 0 and spool_de > 0:
        dom = (bins.w_raw >= float(dominant_w))[None, :]
        gain = np.clip(1.0 - dpure_fit / float(spool_de), 0.0, 1.0) * dom
        bonus = float(spool_bonus) * (gain * bins.w[None, :]).sum(axis=1)
    return {'cost': cost_fit - bonus, 'cost_dE': cost_de, 'cost_fit': cost_fit, 'bonus': bonus, 'idx': idx,
            'dmin': dmin, 'dmin_fit': fit, 'dpure': dpure, 'lab_true': lab_true}

def score_configs(hist: Hist, combos: np.ndarray, backing_pos: np.ndarray, rgb_all: np.ndarray,
                  td_all: np.ndarray, model, stacks: np.ndarray, layer_h: float,
                  metric: str = DEFAULT_METRIC, wL: float = 1.0, chunk: int = 32,
                  chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
                  spool_bonus: float = DEFAULT_SPOOL_BONUS, spool_de: float = DEFAULT_SPOOL_DE,
                  dominant_w: float = DEFAULT_DOMINANT_W,
                  hue_params: Mapping[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cost (S,), cost_dE (S,)) for configurations combos (S,n) of library indices
    with backing position backing_pos (S,) inside each combo (lower is better)."""
    bins = _Bins(hist, chroma_weight, metric, wL, hue_params)
    pure = is_pure(stacks)
    S = combos.shape[0]
    costs = np.empty(S, dtype=np.float64)
    costs_de = np.empty(S, dtype=np.float64)
    for s in range(0, S, chunk):
        e = min(S, s + chunk)
        rgb = rgb_all[combos[s:e]]                           # (m,n,3)
        td = td_all[combos[s:e]]                             # (m,n)
        lin = synth_lut_linear(rgb, td, backing_pos[s:e], model, stacks, layer_h)   # (m,N,3)
        sc = _score(lin, bins, pure, spool_bonus, spool_de, dominant_w)
        costs[s:e] = sc['cost']
        costs_de[s:e] = sc['cost_dE']
    return costs, costs_de


# kept for API compatibility: one backing per combo
def score_subsets(hist: Hist, combos: np.ndarray, rgb_all: np.ndarray, td_all: np.ndarray,
                  backing_pos: np.ndarray, model, stacks: np.ndarray, layer_h: float,
                  wL: float = 1.0, chunk: int = 32, chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
                  spool_bonus: float = DEFAULT_SPOOL_BONUS, spool_de: float = DEFAULT_SPOOL_DE,
                  dominant_w: float = DEFAULT_DOMINANT_W, metric: str = DEFAULT_METRIC) -> np.ndarray:
    return score_configs(hist, combos, backing_pos, rgb_all, td_all, model, stacks, layer_h,
                         metric=metric, wL=wL, chunk=chunk, chroma_weight=chroma_weight,
                         spool_bonus=spool_bonus, spool_de=spool_de, dominant_w=dominant_w)[0]


def _config_detail(hist: Hist, names: Sequence[str], fils: Sequence[Filament], backing_pos: int,
                   model, stacks: np.ndarray, layer_h: float, metric: str, wL: float,
                   chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
                   spool_bonus: float = DEFAULT_SPOOL_BONUS, spool_de: float = DEFAULT_SPOOL_DE,
                   dominant_w: float = DEFAULT_DOMINANT_W,
                   hue_params: Mapping[str, float] | None = None) -> dict:
    bins = _Bins(hist, chroma_weight, metric, wL, hue_params)
    rgb = np.array([f.rgb_lin for f in fils])[None]
    td = np.array([f.td_mm for f in fils])[None]
    lin = synth_lut_linear(rgb, td, np.array([int(backing_pos)]), model, stacks, layer_h)
    pure = is_pure(stacks)
    sc = _score(lin, bins, pure, spool_bonus, spool_de, dominant_w)
    idx = sc['idx'][0]
    dmin = sc['dmin'][0]
    dpure = sc['dpure'][0]
    w, w_raw = bins.w, bins.w_raw
    wsum = float(w.sum()) or 1.0
    wsum_raw = float(w_raw.sum()) or 1.0
    pure_hit = pure[idx]
    dom = w_raw >= float(dominant_w)
    exact = dom & (dpure <= float(spool_de))
    # dominant bins whose nearest pure stack is THIS slot (true Lab)
    d2p = _sqdist(sc['lab_true'][0][pure, :], bins.lab)                 # (n,k) slot order
    near_slot = d2p.argmin(axis=0)
    pure_by, surf_by, use_by, exact_by = {}, {}, {}, {}
    for i, nme in enumerate(names):
        pure_by[nme] = float(w_raw[pure_hit & (stacks[idx, 0] == i)].sum() / wsum_raw)
        surf_by[nme] = float(w_raw[stacks[idx, 0] == i].sum() / wsum_raw)
        use_by[nme] = float((w_raw[:, None] * (stacks[idx] == i)).sum() / (wsum_raw * stacks.shape[1]))
        exact_by[nme] = float(w_raw[exact & (near_slot == i)].sum() / wsum_raw)
    return {
        'filament_names': list(names),
        'backing': names[int(backing_pos)],
        'backing_slot': int(backing_pos),
        'cost': float(sc['cost'][0]),
        'cost_dE': float(sc['cost_dE'][0]),
        'spool_bonus': float(sc['bonus'][0]),
        'mean_dE': float((dmin * w_raw).sum() / wsum_raw),
        'mean_dE_weighted': float((dmin * w).sum() / wsum),
        'max_dE': float(dmin.max()),
        'p90_dE': float(np.quantile(np.repeat(dmin, np.maximum(1, np.round(w_raw * 1000).astype(int))), 0.9)),
        'pure_stack_share': float(w_raw[pure_hit].sum() / wsum_raw),
        'pure_stack_share_by_filament': pure_by,
        'viewing_surface_share_by_filament': surf_by,
        'layer_usage_share_by_filament': use_by,
        'dominant_share': float(w_raw[dom].sum() / wsum_raw),
        'dominant_exact_spool_share': float(w_raw[exact].sum() / wsum_raw),
        'dominant_exact_spool_share_by_filament': exact_by,
        'n_distinct_stacks_used': int(np.unique(idx).size),
        'metric': metric,
        'n_slots': len(names),
        'cost_fit': float(sc['cost_fit'][0]),
        'mean_fit': float((sc['dmin_fit'][0] * w_raw).sum() / wsum_raw),
    }


def _scoring_dict(metric, wL, chroma_weight, spool_bonus, spool_de, dominant_w, prefer, prefer_tol,
                  hue_params=None) -> dict:
    hp = resolve_hue_params(hue_params)
    if metric == 'hue':
        argmin = ("hue-first colour-family metric (true CIELAB): d^2 = (wL(C)*dL)^2 + (wC*dC)^2 + "
                  "(wH*C_src*ramp(dh))^2 + (wN*C_src*colour_loss)^2; wL %.2f (neutral) -> %.2f (chromatic, knee C*=%g), "
                  "wC %.2f, wH %.2f (ramp %g-%g deg), wN %.2f"
                  % (hp['wL_neutral'], hp['wL_chroma'], hp['knee'], hp['wC'], hp['wH'], hp['h0'], hp['h1'], hp['wN']))
    elif metric == 'lumina':
        argmin = "OpenCV 8-bit Lab Euclidean (LuminaImageProcessor KDTree; effective wL %.4f)" % LUMINA_WL_EQUIV
    else:
        argmin = "true CIELAB, dE^2 = wL*dL^2 + da^2 + db^2"
    return {'metric': metric,
            'argmin': argmin,
            'hue_params': hp if metric == 'hue' else None,
            'report': 'true CIELAB dE76 of the entry the argmin selects',
            'cost': 'sum_j w_j * dE76_j - spool_bonus * sum_{dominant j} w_j * max(0, 1 - dE_pure_j / spool_de); '
                    'w_j *= 1 + chroma_weight * C_j / 50',
            'wL': float(wL), 'chroma_weight': float(chroma_weight), 'spool_bonus': float(spool_bonus),
            'spool_de': float(spool_de), 'dominant_w': float(dominant_w),
            'prefer': list(prefer), 'prefer_tol': float(prefer_tol),
            'cost_fit': ('hue-first distance of the picked entry' if metric == 'hue'
                         else 'true dE76 of the picked entry (same as cost_dE)')}


def spool_lab(f: Filament) -> np.ndarray:
    """Spool display colour as true CIELAB (from its linear RGB)."""
    return srgb_to_lab_d65(linear_to_srgb(np.asarray(f.rgb_lin, dtype=np.float64))[None])[0]


def spool_need_shares(hist: Hist, library: Mapping[str, Filament], need_dh: float = DEFAULT_NEED_DH,
                      bin_chroma: float = NEED_BIN_CHROMA, bin_L: float = NEED_BIN_L,
                      spool_chroma: float = NEED_SPOOL_CHROMA) -> dict:
    """For every spool: the pixel share whose colour is visibly chromatic
    (C* >= bin_chroma and L* >= bin_L) and within ``need_dh`` degrees of the
    spool's hue - how much of the picture actually HAS that colour.  None for
    neutral spools (C* < spool_chroma: Black, White, Beige), which are always
    allowed because shade is their job."""
    lab = np.asarray(hist[0], dtype=np.float64).reshape(-1, 3)
    w = np.asarray(hist[1], dtype=np.float64).reshape(-1)
    Cj = np.hypot(lab[:, 1], lab[:, 2])
    hj = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    vis = (Cj >= float(bin_chroma)) & (lab[:, 0] >= float(bin_L))
    out = {}
    for name, f in library.items():
        sl = spool_lab(f)
        Cs = float(np.hypot(sl[1], sl[2]))
        if Cs < float(spool_chroma):
            out[name] = None
            continue
        hs = float(np.degrees(np.arctan2(sl[2], sl[1])))
        dh = np.abs((hj - hs + 180.0) % 360.0 - 180.0)
        out[name] = float(w[vis & (dh <= float(need_dh))].sum())
    return out


def hue_name(h_deg: float) -> str:
    h = float(h_deg) % 360.0
    for lo, hi, nm in ((0, 20, 'red/pink'), (20, 45, 'red-orange'), (45, 75, 'brown/orange'), (75, 110, 'yellow'),
                       (110, 170, 'green'), (170, 220, 'cyan'), (220, 275, 'blue'), (275, 335, 'purple'),
                       (335, 360, 'magenta/pink')):
        if lo <= h < hi:
            return nm
    return 'red/pink'


def _circ_diff(a, b):
    return np.abs((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 180.0) % 360.0 - 180.0)


def hue_plan(hist: Hist, library: Mapping[str, Filament], pick_dh: float = DEFAULT_NEED_DH,
             cluster_dh: float = DEFAULT_CLUSTER_DH, cluster_min: float = DEFAULT_NEED_SHARE,
             neutral_min: float = 0.02, dark_min: float = 0.05, extra_dh: float = 25.0, extra_share: float = 0.03,
             family_dh: float = 20.0,
             hue_params: Mapping[str, float] | None = None,
             spool_chroma: float = NEED_SPOOL_CHROMA, bin_chroma: float = NEED_BIN_CHROMA,
             bin_L: float = NEED_BIN_L, model=None, layers: int = N_LAYERS, layer_h: float = LAYER_H) -> dict:
    """Colour-theory plan: WHICH spools the picture needs.

    1. Visibly chromatic histogram bins (C* >= bin_chroma, L* >= bin_L) are
       merged into hue clusters (neighbours within ``cluster_dh`` degrees).
    2. A cluster is *needed* when its chroma-weighted pixel share
       (sum w * min(C*/30, 1)) is >= ``cluster_min`` - a faint purple tint on
       a black-and-white photo does not count, brown skin does.
    3. Every needed cluster gets the chromatic spool within ``pick_dh``
       degrees of its hue that RENDERS it best: each candidate is judged by the
       hue-first error of the cluster's pixels against the stacks it can form
       together with the darkest and lightest neutral spools (so Red + White
       can serve pink, Vivid Yellow + Black can serve dark yellow, but pastel
       Pink cannot serve a saturated red).
    4. The lightest neutral spool is required when >= ``neutral_min`` of the
       pixels are light neutrals (C* < 15, L* >= 55) - white cannot be mixed
       from colours; the darkest when >= ``dark_min`` are dark (L* < 25) -
       near-black CAN be stacked from dark colours, so Black competes on cost
       for nearly-all-bright art.  When more colours are needed than there are
       slots, the cost search picks the best n of them.
    ``required`` = those picks; ``allowed`` = required + all neutral spools +
    every chromatic spool whose hue is actually present (>= ``extra_share`` of
    the pixels visibly chromatic within ``extra_dh`` degrees of it); everything
    else is excluded - no spool enters as a mere mixing ingredient.  ``fallback_order`` ranks the
    excluded spools by hue distance to the art, for callers that must fill 5."""
    lab = np.asarray(hist[0], dtype=np.float64).reshape(-1, 3)
    w = np.asarray(hist[1], dtype=np.float64).reshape(-1)
    Lj = lab[:, 0]
    Cj = np.hypot(lab[:, 1], lab[:, 2])
    hj = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360.0
    vis = (Cj >= float(bin_chroma)) & (Lj >= float(bin_L))
    names = list(library.keys())
    slab = {nm: spool_lab(library[nm]) for nm in names}
    sC = {nm: float(np.hypot(l[1], l[2])) for nm, l in slab.items()}
    sh = {nm: float(np.degrees(np.arctan2(l[2], l[1])) % 360.0) for nm, l in slab.items()}
    chromatic = [nm for nm in names if sC[nm] >= float(spool_chroma)]
    neutral = [nm for nm in names if sC[nm] < float(spool_chroma)]
    darkest = min(neutral, key=lambda nm: slab[nm][0]) if neutral else None
    lightest = max(neutral, key=lambda nm: slab[nm][0]) if neutral else None
    if model is None:
        model = BeerLambertModel()
    helpers = [nm for nm in (darkest, lightest) if nm]

    def render_fit(bins_idx, spool):
        """Weighted hue-first error of the bins against every stack of
        {darkest, lightest, spool} (all backings tried) - lower is better."""
        fl = [library[x] for x in helpers + [spool]]
        rgb = np.array([f.rgb_lin for f in fl], dtype=np.float64)[None]
        td = np.array([f.td_mm for f in fl], dtype=np.float64)[None]
        st = enumerate_stacks(len(fl), layers)
        wb = w[bins_idx] / max(float(w[bins_idx].sum()), 1e-12)
        best = np.inf
        for b in range(len(fl)):
            lin = synth_lut_linear(rgb, td, np.array([b]), model, st, layer_h)
            lab_true = srgb_to_lab_d65(linear_to_uint8(lin).astype(np.float64) / 255.0)     # (1,N,3)
            d2 = hue_dist2(lab[bins_idx], lab_true, hue_params)[0]                          # (N,kc)
            best = min(best, float((np.sqrt(np.maximum(d2.min(axis=0), 0.0)) * wb).sum()))
        return best

    # --- hue clusters: greedy merge along the hue circle, span-limited so red
    #     cannot chain into gold through orange
    idx = [int(i) for i in np.where(vis)[0]]
    idx.sort(key=lambda i: hj[i])
    groups: list[list[int]] = []
    for i in idx:
        if groups:
            g = groups[-1]
            wg = w[g]
            mean_h = float(np.degrees(np.arctan2((wg * np.sin(np.radians(hj[g]))).sum(),
                                                 (wg * np.cos(np.radians(hj[g]))).sum())) % 360.0)
            if (_circ_diff(hj[i], mean_h) <= float(cluster_dh)
                    and _circ_diff(hj[i], hj[g[0]]) <= 1.8 * float(cluster_dh)):
                g.append(i)
                continue
        groups.append([i])
    if len(groups) > 1:                                   # wrap-around merge (350 deg and 10 deg are neighbours)
        first, last = groups[0], groups[-1]
        if (_circ_diff(hj[first[0]], hj[last[-1]]) <= float(cluster_dh)
                and _circ_diff(hj[first[-1]], hj[last[0]]) <= 1.8 * float(cluster_dh)):
            groups[0] = last + first
            groups.pop()
    clusters = []
    for g in groups:
        wg = w[g]
        share = float(wg.sum())
        if share <= 0:
            continue
        weight = float((wg * np.minimum(Cj[g] / 30.0, 1.0)).sum())
        mean_h = float(np.degrees(np.arctan2((wg * np.sin(np.radians(hj[g]))).sum(),
                                             (wg * np.cos(np.radians(hj[g]))).sum())) % 360.0)
        mean_L = float((wg * Lj[g]).sum() / share)
        mean_C = float((wg * Cj[g]).sum() / share)
        centre = np.array([mean_L, mean_C * np.cos(np.radians(mean_h)), mean_C * np.sin(np.radians(mean_h))])
        cands = [nm for nm in chromatic if _circ_diff(sh[nm], mean_h) <= float(pick_dh)]
        needed_c = weight >= float(cluster_min)
        if needed_c and cands:
            fit = {nm: render_fit(g, nm) for nm in cands}
        else:   # minor clusters: cheap spool-colour distance is enough for the report
            fit = {nm: float(np.sqrt(hue_dist2(centre[None], slab[nm][None], hue_params)[0, 0])) for nm in cands}
        cands.sort(key=lambda nm: (fit[nm], names.index(nm)))
        clusters.append({'hue': round(mean_h, 1), 'name': hue_name(mean_h), 'L': round(mean_L, 1),
                         'C': round(mean_C, 1), 'share': share, 'weight': weight,
                         'needed': needed_c, 'candidates': cands, 'candidate_fit': fit,
                         'pick': cands[0] if cands else None})
    clusters.sort(key=lambda c: -c['weight'])
    needed = [c for c in clusters if c['needed']]

    # --- neutrals
    light_share = float(w[(Cj < 15.0) & (Lj >= 55.0)].sum())
    bright_share = float(w[(Cj < 15.0) & (Lj >= 80.0)].sum())       # true whites/highlights: unmixable
    dark_share = float(w[Lj < 25.0].sum())
    required: list[str] = []
    if darkest and dark_share >= float(dark_min):
        required.append(darkest)
    if lightest and (light_share >= float(neutral_min) or bright_share >= 0.005) and lightest not in required:
        required.append(lightest)
    covered_h: list[float] = []                       # one spool per colour family (biggest cluster wins)
    for c in needed:
        pk = c['pick']
        if not pk or pk in required:
            continue
        if any(_circ_diff(sh[pk], h) <= float(family_dh) for h in covered_h):
            c['pick_note'] = 'family already covered by a bigger cluster'
            continue
        required.append(pk)
        covered_h.append(sh[pk])
    # extras: chromatic spools whose hue is visibly present in the picture
    present = {}
    for nm in chromatic:
        dh = _circ_diff(hj, sh[nm])
        present[nm] = float(w[vis & (dh <= float(extra_dh))].sum())
    extras = [nm for nm in chromatic if present[nm] >= float(extra_share)]
    allowed = list(dict.fromkeys(required + neutral + extras))
    allowed = [nm for nm in names if nm in allowed]                       # library order
    excluded = [nm for nm in names if nm not in allowed]

    def art_dist(nm):
        if not needed:
            return 999.0
        return min(float(_circ_diff(sh[nm], c['hue'])) for c in needed)
    fallback = sorted(excluded, key=lambda nm: (art_dist(nm), names.index(nm)))
    return {'clusters': clusters, 'needed': [c for c in needed], 'required': required, 'allowed': allowed,
            'excluded': excluded, 'fallback_order': fallback, 'hue_present_share': present,
            'light_neutral_share': light_share, 'bright_share': bright_share, 'dark_share': dark_share,
            'darkest_spool': darkest, 'lightest_spool': lightest,
            'params': {'pick_dh': float(pick_dh), 'cluster_dh': float(cluster_dh), 'cluster_min': float(cluster_min),
                       'neutral_min': float(neutral_min), 'dark_min': float(dark_min),
                       'extra_dh': float(extra_dh), 'extra_share': float(extra_share), 'family_dh': float(family_dh)}}


def select_palette(hist: Hist, library: Mapping[str, Filament], model=None, n: int = 5,
                   must_include: Sequence[str] = (), backing: str | None = None,
                   wL: float = 1.0, chunk: int = 32, layers: int = N_LAYERS,
                   layer_h: float = LAYER_H, top_k: int = 10,
                   chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
                   spool_bonus: float = DEFAULT_SPOOL_BONUS, spool_de: float = DEFAULT_SPOOL_DE,
                   dominant_w: float = DEFAULT_DOMINANT_W,
                   prefer: Sequence[str] = (), prefer_tol: float = DEFAULT_PREFER_TOL,
                   metric: str = DEFAULT_METRIC,
                   hue_params: Mapping[str, float] | None = None,
                   n_min: int | None = None, spool_penalty: float = DEFAULT_SPOOL_PENALTY,
                   need_share: float = DEFAULT_NEED_SHARE, need_dh: float = DEFAULT_NEED_DH
                   ) -> tuple[list[str], dict]:
    """Exhaustive search over subsets of ``library`` with n_min..n spools x all
    backings.  "Stick to the colours the picture needs":

    * a chromatic spool may only enter a palette when >= ``need_share`` of the
      pixels are visibly chromatic and within ``need_dh`` degrees of its hue
      (spool_need_shares); neutral spools are always allowed; must_include /
      backing / prefer spools bypass the filter;
    * every spool adds ``spool_penalty`` to the cost, so a 4th or 5th spool has
      to buy at least that much weighted colour error to be used - a purple
      tint on a black-and-white photo does not earn a purple spool;
    * the cost is the MATCHING metric's own distance (hue-first for 'hue'), so
      the palette is chosen by the same rule the pixels are then matched with.

    Returns (names in library order, report).  report['best'] is the winning
    configuration (its 'backing' is the backing to print with, 'n_slots' the
    palette size); report['top'] lists the best ``top_k`` subsets of any size,
    each with its best backing and 'backing_costs'.  ``must_include`` is hard;
    ``prefer`` is soft (report['prefer'] records rank, excess and decision).
    ``metric``: 'hue' (default), 'lumina' (Lumina's OpenCV 8-bit Lab KDTree
    metric) or 'lab' (weighted true Lab with ``wL``).  n_min=None -> DEFAULT_N_MIN;
    pass n_min=n to force exactly n spools.
    """
    if model is None:
        model = BeerLambertModel()
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    t0 = time.perf_counter()
    names = list(library.keys())
    M = len(names)
    n_max = int(n)
    n_lo = DEFAULT_N_MIN if n_min is None else int(n_min)
    if not (1 <= n_lo <= n_max <= M):
        raise ValueError(f"need 1 <= n_min ({n_lo}) <= n ({n_max}) <= {M}")
    if spool_penalty < 0 or need_share < 0 or need_dh <= 0:
        raise ValueError("spool_penalty and need_share must be >= 0, need_dh > 0")
    must = [m for m in (must_include or ()) if m]
    pref = [p for p in (prefer or ()) if p]
    missing = [m for m in must if m not in library]
    if missing:
        raise KeyError(f"must_include filaments not in library: {missing}; available: {names}")
    missing = [p for p in pref if p not in library]
    if missing:
        raise KeyError(f"prefer filaments not in library: {missing}; available: {names}")
    if backing is not None and backing not in library:
        raise KeyError(f"backing {backing!r} not in library; available: {names}")
    if prefer_tol < 0:
        raise ValueError("prefer_tol must be >= 0")
    need = set(names.index(m) for m in must)
    if backing is not None:
        need.add(names.index(backing))
    pset = set(names.index(p) for p in pref)
    if len(need) > n_max:
        raise ValueError("more required filaments than palette slots")
    if len(need | pset) > n_max:
        raise ValueError("more required + preferred filaments than palette slots")

    # colour-theory plan: which spools the picture needs / may use
    shares = spool_need_shares(hist, library, need_dh=need_dh)
    plan = hue_plan(hist, library, pick_dh=need_dh, cluster_min=need_share, hue_params=hue_params,
                    model=model, layers=layers, layer_h=layer_h)
    req_idx = set(names.index(nm) for nm in plan['required'])
    hard = set(need) | req_idx
    allowed_set = set(names.index(nm) for nm in plan['allowed']) | need      # prefer spools join via the second search
    if len(hard) > n_max:
        # more needed colours than slots: the search picks the best n_max of them
        allowed_set = hard | pset
        hard = set(need)
        n_lo = n_max
    if n_lo == n_max and len(allowed_set) < n_max:
        # caller insists on exactly n spools: top up with the spools nearest to the art's hues
        for nm in plan['fallback_order']:
            if len(allowed_set) >= n_max:
                break
            allowed_set.add(names.index(nm))
    allowed = sorted(allowed_set)
    excluded = [nm for i, nm in enumerate(names) if i not in allowed_set]
    n_lo = max(n_lo, len(hard))

    fils = [library[k] for k in names]
    rgb_all = np.array([f.rgb_lin for f in fils], dtype=np.float64)
    td_all = np.array([f.td_mm for f in fils], dtype=np.float64)
    kw = dict(chroma_weight=float(chroma_weight), spool_bonus=float(spool_bonus),
              spool_de=float(spool_de), dominant_w=float(dominant_w), hue_params=hue_params)

    def search(hard_set, allowed_list, k_lo, k_hi):
        """One group of configurations per palette size k: (subset, backing position)."""
        out = []
        for k in range(k_lo, k_hi + 1):
            if k < len(hard_set) or k > len(allowed_list):
                continue
            combos_k = np.array([c for c in itertools.combinations(allowed_list, k) if hard_set.issubset(c)],
                                dtype=np.int64)
            if combos_k.size == 0:
                continue
            S_k = combos_k.shape[0]
            if backing is not None:
                ib = names.index(backing)
                cfg_subset = np.arange(S_k, dtype=np.int64)
                cfg_bpos = np.argmax(combos_k == ib, axis=1).astype(np.int64)
                n_b = 1
            else:
                cfg_subset = np.repeat(np.arange(S_k, dtype=np.int64), k)
                cfg_bpos = np.tile(np.arange(k, dtype=np.int64), S_k)
                n_b = k
            stacks_k = enumerate_stacks(k, layers)
            costs_cfg, _ = score_configs(hist, combos_k[cfg_subset], cfg_bpos, rgb_all, td_all, model,
                                         stacks_k, layer_h, metric=metric, wL=float(wL), chunk=chunk, **kw)
            cc = costs_cfg.reshape(S_k, n_b)
            best_b = cc.argmin(axis=1)                       # ties -> lower backing position
            rows = np.arange(S_k)
            out.append({'k': k, 'combos': combos_k, 'stacks': stacks_k, 'cc': cc, 'n_b': n_b,
                        'cost_raw': cc[rows, best_b], 'cost': cc[rows, best_b] + float(spool_penalty) * k,
                        'bpos': cfg_bpos.reshape(S_k, n_b)[rows, best_b], 'n_cfg': int(costs_cfg.size)})
        return out

    groups = search(hard, allowed, n_lo, n_max)
    if not groups:
        raise ValueError("no feasible subset")
    n_plan_groups = len(groups)
    # soft preference: also evaluate palettes built around the preferred spools (they may
    # displace a required pick) so the tolerance rule has something to compare
    if pset and not pset.issubset(hard):
        neutral_req = {names.index(nm) for nm in plan['required']
                       if nm in (plan.get('darkest_spool'), plan.get('lightest_spool'))}
        hard_b = set(need) | pset | neutral_req
        allowed_b = sorted(allowed_set | pset)
        groups += search(hard_b, allowed_b, max(DEFAULT_N_MIN if n_min is None else int(n_min), len(hard_b)), n_max)
    backing_rule = f"fixed: {backing}" if backing is not None else BACKING_RULE_SEARCH

    all_cost = np.concatenate([g['cost'] for g in groups])
    gidx = np.concatenate([np.full(len(g['cost']), gi, dtype=np.int64) for gi, g in enumerate(groups)])
    rows_all = np.concatenate([np.arange(len(g['cost']), dtype=np.int64) for g in groups])
    order = np.lexsort((rows_all, gidx, all_cost))      # stable: ties -> fewer spools, then lower combo index
    order_plan = order[gidx[order] < n_plan_groups]     # the plan's own configurations define best/top

    def combo_of(e: int):
        g = groups[gidx[e]]
        return g, g['combos'][rows_all[e]], int(g['bpos'][rows_all[e]])

    def detail(e: int) -> dict:
        g, combo, bpos = combo_of(e)
        sub_names = [names[i] for i in combo]
        sub_fils = [library[k] for k in sub_names]
        det = _config_detail(hist, sub_names, sub_fils, bpos, model, g['stacks'], layer_h, metric, float(wL), **kw)
        det['backing_costs'] = {names[combo[b]]: float(g['cc'][rows_all[e], b]) for b in range(g['n_b'])}
        det['n_slots'] = int(g['k'])
        det['spool_penalty_total'] = float(spool_penalty) * int(g['k'])
        det['cost_total'] = float(all_cost[e])
        return det

    top = []
    for e in order_plan[:max(1, min(top_k, len(order_plan)))]:
        det = detail(int(e))
        det['rank'] = len(top) + 1
        top.append(det)
    best_unconstrained = top[0]
    best = best_unconstrained
    scale = max(float(best_unconstrained['cost_dE']), 1e-9)
    prefer_info = {'requested': pref, 'applied': False, 'rank': None, 'excess_dE': None,
                   'excess_ratio': None, 'tolerance': float(prefer_tol),
                   'tolerance_dE': float(prefer_tol) * scale}
    if pref:
        for rank_i, e in enumerate(order):
            g, combo, bpos = combo_of(int(e))
            if pset.issubset(combo):
                excess = float(all_cost[e] - all_cost[order_plan[0]])
                prefer_info.update({'rank': rank_i + 1, 'excess_dE': excess,
                                    'excess_ratio': excess / scale, 'cost': float(all_cost[e]),
                                    'filament_names': [names[i] for i in combo],
                                    'backing': names[combo[bpos]]})
                if excess <= float(prefer_tol) * scale + 1e-12:
                    prefer_info['applied'] = True
                    best = dict(detail(int(e)), rank=rank_i + 1)
                break
    report = {
        'search': f"exhaustive subsets of {n_lo}..{n_max} of {len(allowed)} allowed spools"
                  + (f" containing {sorted(must)}" if must else '')
                  + (" x all backings" if backing is None else f" with backing {backing}"),
        'n_subsets_evaluated': int(sum(len(g['cost']) for g in groups)),
        'n_configs_evaluated': int(sum(g['n_cfg'] for g in groups)),
        'sizes_evaluated': {int(g['k']): int(len(g['cost'])) for g in groups},
        'n_stacks_per_subset': {int(g['k']): int(g['stacks'].shape[0]) for g in groups},
        'n_slots': int(best['n_slots']),
        'n_min': int(n_lo), 'n_max': int(n_max),
        'spool_penalty': float(spool_penalty),
        'need_share': float(need_share), 'need_dh': float(need_dh),
        'spool_need_share': shares,
        'hue_plan': plan,
        'required_spools': list(plan['required']),
        'hard_spools': [names[i] for i in sorted(hard)],
        'allowed_spools': [names[i] for i in allowed],
        'excluded_spools': excluded,
        'hist_bins': int(np.asarray(hist[0]).shape[0]),
        'metric': metric,
        'wL': float(wL),
        'scoring': _scoring_dict(metric, wL, chroma_weight, spool_bonus, spool_de, dominant_w, pref, prefer_tol,
                                 hue_params),
        'model': getattr(model, 'name', type(model).__name__),
        'model_repr': repr(model),
        'layers': int(layers),
        'layer_h': (float(layer_h) if np.ndim(layer_h) == 0
                    else [float(t) for t in np.asarray(layer_h).ravel()]),
        'backing_rule': backing_rule,
        'best': best,
        'best_unconstrained': best_unconstrained,
        'prefer': prefer_info,
        'top': top,
        'seconds': time.perf_counter() - t0,
    }
    return list(best['filament_names']), report

def evaluate_palette(hist: Hist, names: Sequence[str], library: Mapping[str, Filament],
                     backing: str | None = None, model=None, wL: float = 1.0,
                     layers: int = N_LAYERS, layer_h: float = LAYER_H,
                     chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
                     spool_bonus: float = DEFAULT_SPOOL_BONUS, spool_de: float = DEFAULT_SPOOL_DE,
                     dominant_w: float = DEFAULT_DOMINANT_W, metric: str = DEFAULT_METRIC,
                     hue_params: Mapping[str, float] | None = None) -> dict:
    """Detail report for one explicit palette (same fields as report['top'][i]).
    ``backing`` None -> every member is tried and the lowest-cost backing is
    chosen (detail['backing'], detail['backing_costs'])."""
    if model is None:
        model = BeerLambertModel()
    names = list(names)
    fils = [library[k] for k in names]
    stacks = enumerate_stacks(len(fils), layers)
    kw = dict(chroma_weight=float(chroma_weight), spool_bonus=float(spool_bonus),
              spool_de=float(spool_de), dominant_w=float(dominant_w), hue_params=hue_params)
    if backing is not None:
        if backing not in names:
            raise KeyError(f"backing {backing!r} not in palette {names}")
        cands = [names.index(backing)]
    else:
        cands = list(range(len(fils)))
    rgb = np.array([f.rgb_lin for f in fils])[None].repeat(len(cands), axis=0)
    td = np.array([f.td_mm for f in fils])[None].repeat(len(cands), axis=0)
    bins = _Bins(hist, float(chroma_weight), metric, float(wL), hue_params)
    lin = synth_lut_linear(rgb, td, np.array(cands), model, stacks, layer_h)
    sc = _score(lin, bins, is_pure(stacks), float(spool_bonus), float(spool_de), float(dominant_w))
    b = int(cands[int(np.argmin(sc['cost']))])
    det = _config_detail(hist, names, fils, b, model, stacks, layer_h, metric, float(wL), **kw)
    det['backing_costs'] = {names[c]: float(sc['cost'][i]) for i, c in enumerate(cands)}
    det['rank'] = None
    return det
