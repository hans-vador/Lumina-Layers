"""Hue-first ("colour family") metric for Stack5 - shared by the palette search
and the per-pixel LUT matching so both agree on what "closest colour" means.

Why it exists: Lumina matches in OpenCV 8-bit Lab (L* scaled by 2.55), so a
lightness error costs 6.5x a chroma error and dark brown skin (L19 C33 h51) is
matched to a near-black olive stack (L15 C13 h103) - the hue is thrown away to
keep the shade.  Plain CIELAB / CIEDE2000 have the opposite blind spot: a light
blue sky (L63 C22) is "closer" to grey than to a darker, purpler blue, because
desaturating costs only dC while a 40 degree hue shift costs a lot.  Hans wants
the RIGHT COLOUR FAMILY even when the shade is off: blue sky must be blue,
brown skin brown, and a colour must never collapse to grey.

    d^2 = (wL(C_src) * dL)^2 + (wC * dC)^2 + (wH * C_src * s(dh))^2 + (wN * C_src * n)^2

    dL, dC       lightness / chroma differences (candidate - source)
    s(dh)        hue-shift ramp: 0 up to h0 degrees (same family), 1 from h1 on;
                 only defined when the candidate is chromatic (C_cand >= c_hue_min)
    n            colour loss = max(0, 1 - C_cand / min(C_src, c_keep)): a grey
                 candidate for a chromatic source pays like a wrong hue, a less
                 vivid but still clearly coloured one does not
    C_src        in the hue and loss terms is the chroma ABOVE c_vis (12): faint
                 tints do not demand a spool
    wL(C_src) = wL_neutral + (wL_chroma - wL_neutral) * min(C_src / knee, 1)

Neutrals (black, white, grey: hue meaningless) keep the full lightness weight;
the more chromatic the source colour, the more lightness it may trade for the
right family.  Defaults (``DEFAULT_HUE_PARAMS``): wL 1.0 -> 0.6, wC 0.6,
wH 2.0, wN 1.5, knee 40, h0 8 deg, h1 60 deg, c_hue_min 8, c_vis 12, c_keep 25.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from core.band.optics import srgb_to_lab_d65

DEFAULT_HUE_PARAMS = {'wL_neutral': 1.0, 'wL_chroma': 0.6, 'wC': 0.6, 'wH': 2.0, 'wN': 1.5,
                      'knee': 40.0, 'h0': 8.0, 'h1': 60.0, 'c_hue_min': 8.0, 'c_vis': 12.0, 'c_keep': 25.0}
_ALIASES = {'wl': 'wL_chroma', 'wl_chroma': 'wL_chroma', 'wln': 'wL_neutral', 'wl_neutral': 'wL_neutral',
            'wc': 'wC', 'wh': 'wH', 'wn': 'wN', 'knee': 'knee', 'h0': 'h0', 'h1': 'h1', 'c_hue_min': 'c_hue_min',
            'c_vis': 'c_vis', 'c_keep': 'c_keep'}


def resolve_hue_params(params: Mapping[str, float] | None = None) -> dict:
    """Merge user overrides into the defaults and validate."""
    out = dict(DEFAULT_HUE_PARAMS)
    for k, v in (params or {}).items():
        key = _ALIASES.get(str(k).lower().replace('-', '_'))
        if key is None:
            raise ValueError(f"unknown hue-metric parameter {k!r}; use one of {sorted(set(_ALIASES.values()))}")
        out[key] = float(v)
    for k in ('wL_neutral', 'wL_chroma', 'wC', 'wH', 'wN', 'h0', 'c_hue_min', 'c_vis', 'c_keep'):
        if out[k] < 0:
            raise ValueError(f"{k} must be >= 0, got {out[k]}")
    if out['knee'] <= 0 or out['h1'] <= out['h0']:
        raise ValueError("knee must be > 0 and h1 > h0")
    return out


def parse_hue_params(text: str) -> dict:
    """'wL=0.4,wH=2.5' -> {'wL_chroma': 0.4, 'wH': 2.5} (validated, defaults filled)."""
    raw = {}
    for item in (text or '').split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(f"hue params expect NAME=value entries, got {item!r}")
        k, v = item.split('=', 1)
        raw[k.strip()] = float(v)
    return resolve_hue_params(raw)


def lab_to_lch(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True CIELAB (...,3) -> (L*, C*, h in radians) each (...)."""
    lab = np.asarray(lab, dtype=np.float32)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    return L, np.hypot(a, b), np.arctan2(b, a)


def hue_dist2(src_lab: np.ndarray, cand_lab: np.ndarray, params: Mapping[str, float] | None = None) -> np.ndarray:
    """Squared colour-family distance between k source colours (k,3) and
    candidate colours (...,N,3), both true CIELAB -> (...,N,k) float32.  See the
    module docstring for the formula."""
    p = resolve_hue_params(params)
    Ls, Cs, hs = lab_to_lch(np.asarray(src_lab, dtype=np.float32).reshape(-1, 3))       # (k,)
    Lc, Cc, hc = lab_to_lch(cand_lab)                                                   # (...,N)
    wL = p['wL_neutral'] + (p['wL_chroma'] - p['wL_neutral']) * np.clip(Cs / p['knee'], 0.0, 1.0)   # (k,)
    dL = (Lc[..., None] - Ls) * wL
    dC = (Cc[..., None] - Cs) * np.float32(p['wC'])
    dh = np.abs(hc[..., None] - hs)
    dh = np.degrees(np.minimum(dh, 2.0 * np.pi - dh))                                  # 0..180
    ramp = np.clip((dh - p['h0']) / max(p['h1'] - p['h0'], 1e-6), 0.0, 1.0)
    chromatic_cand = (Cc[..., None] >= p['c_hue_min']).astype(np.float32)
    # only chroma ABOVE the visibility threshold counts as "colour" to defend:
    # a faint tint (C* 16 on a black-and-white photo) neither demands its hue
    # nor its saturation, skin at C* 37 does
    Ceff = np.maximum(Cs - p['c_vis'], 0.0)
    dH = np.float32(p['wH']) * Ceff * ramp * chromatic_cand
    # colour loss: the candidate dropping below the visible-chroma level of the
    # source (grey for a blue sky), not merely being less vivid (Lavender for a
    # saturated purple is still purple)
    c_floor = np.minimum(Cs, p['c_keep'])
    loss = np.clip(1.0 - Cc[..., None] / np.maximum(c_floor, 1e-6), 0.0, 1.0)
    dN = np.float32(p['wN']) * Ceff * loss
    return dL * dL + dC * dC + dH * dH + dN * dN


def weighted_lab_dist2(src_lab: np.ndarray, cand_lab: np.ndarray, wL: float = 1.0) -> np.ndarray:
    """Legacy 'lab' metric: (...,N,3) x (k,3) -> (...,N,k) with dE^2 = wL*dL^2 + da^2 + db^2."""
    w = np.array([np.sqrt(float(wL)), 1.0, 1.0], dtype=np.float64)
    a = np.asarray(cand_lab, dtype=np.float64) * w
    b = np.asarray(src_lab, dtype=np.float64).reshape(-1, 3) * w
    a2 = (a * a).sum(-1)[..., :, None]
    b2 = (b * b).sum(-1)[None, :]
    return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)


class Stack5Matcher:
    """Drop-in replacement for ``LuminaImageProcessor.hue_matcher``: Lumina calls
    ``match_colors_batch(unique_rgb8, k)`` and expects one LUT index per colour.
    metric 'hue' -> hue_dist2 (true Lab); 'lab' -> weighted true Lab.  ('lumina'
    keeps Lumina's own KDTree, so no matcher is installed for it.)"""

    def __init__(self, lut_rgb8: np.ndarray, metric: str = 'hue', wL: float = 1.0,
                 hue_params: Mapping[str, float] | None = None):
        if metric not in ('hue', 'lab'):
            raise ValueError(f"Stack5Matcher supports 'hue' and 'lab', got {metric!r}")
        self.lut_rgb = np.ascontiguousarray(np.asarray(lut_rgb8, dtype=np.uint8).reshape(-1, 3))
        self.lut_lab = srgb_to_lab_d65(self.lut_rgb.astype(np.float64) / 255.0)
        self.metric = metric
        self.wL = float(wL)
        self.params = resolve_hue_params(hue_params)
        self.n_colors = int(self.lut_rgb.shape[0])

    def match_colors_batch(self, input_rgb: np.ndarray, k: int | None = None) -> np.ndarray:
        rgb8 = np.asarray(input_rgb, dtype=np.uint8).reshape(-1, 3)
        lab = srgb_to_lab_d65(rgb8.astype(np.float64) / 255.0)
        if self.metric == 'hue':
            d2 = hue_dist2(lab, self.lut_lab, self.params)          # (N,n)
        else:
            d2 = weighted_lab_dist2(lab, self.lut_lab, self.wL)      # (N,n)
        return np.asarray(d2.argmin(axis=0), dtype=np.intp)

    def describe(self) -> str:
        if self.metric == 'hue':
            p = self.params
            return ("hue-first LCh: wL %.2f (neutral) -> %.2f (chromatic, knee C*=%g), wC %.2f, wH %.2f "
                    "(ramp %g-%g deg), colour-loss wN %.2f"
                    % (p['wL_neutral'], p['wL_chroma'], p['knee'], p['wC'], p['wH'], p['h0'], p['h1'], p['wN']))
        return f"true CIELAB, dE^2 = {self.wL:g}*dL^2 + da^2 + db^2"


def make_matcher(metric: str, lut_rgb8: np.ndarray, wL: float = 1.0,
                 hue_params: Mapping[str, float] | None = None):
    """None for 'lumina' (Lumina's own 8-bit Lab KDTree), a Stack5Matcher otherwise."""
    if metric == 'lumina':
        return None
    return Stack5Matcher(lut_rgb8, metric=metric, wL=wL, hue_params=hue_params)
