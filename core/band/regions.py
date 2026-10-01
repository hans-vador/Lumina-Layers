"""Region layers: Reference layers geometry, a separate colour ramp per region.

The relief is exactly Reference layers' (brightness -> height, same tone,
same thin-feature cleanup, same mesh).  Reference prints one spool per layer
across the whole plate, so equal heights always print equal colours: the pink
title on Graduation's equally bright purple sky vanishes.  Here the cover is
split into a few large regions and each region gets its own spool sequence
(its own swap heights), chosen from one shared palette:

1. Regions start from k-means on the cover's hue (CIELAB a*, b*).
2. Fit: for every spool combination of the library (dark -> light) and every
   swap plan, the least-squares colour error of each region, all at once
   (same Beer-Lambert stack colours as Reference).  The palette is the <= 4
   spools (one AMS) whose best per-region plans fit best (ticked spools: all of
   them, each in at least one region).  Every region shares the darkest spool,
   which also prints the 1 mm backing.
3. Reassign: every pixel moves to the region whose ramp predicts its colour
   best (error maps smoothed over ~1 mm, then a majority filter and a minimum
   region area, so regions stay large and printable).  Repeat.
4. More regions are only kept when they pay for themselves: the mean squared
   colour error, times (1 + MIX_COST per layer that holds two spools +
   REGION_COST per extra region), must go down.  One region = Reference.
5. Accents: small, clearly coloured areas the ramps still print in the wrong
   colour family (Blond's green hair, 0.8% of the plaque) get a region of their
   own, scored colour-first (add_accents).

Printing: one object, the base spool, plus one modifier volume per other spool
(named after it) that sets that spool inside its regions between two layer
boundaries.  Modifiers never overlap and end just above the relief, so the
plaque moves as one piece.  Layers where regions use different spools need tool changes, so a
prime tower is planned with Stack5's flush model (core.stack5.flush).
"""
from pathlib import Path
import hashlib
import io
import itertools
import json
import re
import uuid
import zipfile
from xml.sax.saxutils import quoteattr

import numpy as np
import shapely
import shapely.affinity
import trimesh
from PIL import Image, ImageOps
from scipy import ndimage
from shapely.geometry import box as shapely_box

from core.band import writer3mf
from core.band.heightfield_mesh import heightfield_to_trimesh
from core.band.optics import (DEFAULT_K_OPAQUE, linear_rgb_to_lab_d65, linear_to_srgb,
                              load_filament_library)
from core.band.reference import (AMPLITUDE, DEFAULT_CONTRAST, DEFAULT_FILAMENTS_JSON, FLOOR, LAYER_H,
                                 MIN_FEATURE_MM, _stack_colour, auto_tone, column_layers,
                                 complete_two_nozzle_process, height_from_tone, palette_filaments,
                                 remove_thin_features, toned_target)
from core.band.writer3mf import _write_triangles_bytes, _write_vertices_bytes, write_band_3mf
from core.stack5 import flush as flushmod
from core.stack5.metric import lab_to_lch, resolve_hue_params

REPO = Path(__file__).resolve().parents[2]
BASE_TOP = 1.          # five .2 mm backing layers, then .08 mm colour layers (as Reference)
SHIFT = .56            # Reference's art/swap translation for the 1 mm backing
MAX_REGIONS = 3
MAX_SPOOLS = 4         # one AMS on nozzle 1
# A layer holding two spools costs a tool change and a purge on the tower; an
# extra region adds seams.  Relative costs on the mean squared colour error.
MIX_COST = .005
REGION_COST = .05
# Segmentation grid and cleanup, fine enough for lettering: Graduation's pink
# title (strokes ~1.5 mm on an equally bright purple sky) becomes its own region
# at these values and vanished at 0.5 / 1 / 2.5 / 6.  Tool changes depend on
# the layers regions share, not on how many islands there are.
WORK_MM = .35          # segmentation grid (the final fit runs on the full mesh grid)
SMOOTH_MM = .5         # error maps are blurred this much before reassigning pixels
MAJORITY_MM = 1.       # majority-filter window
MIN_REGION_MM2 = 2.    # smaller islands join their neighbour (2 mm2 ~ a 1.4 mm square)
ITERATIONS = 5
SWITCH_MARGIN = .02
# region map colours (previews only)
REGION_MAP_RGB = np.array([(96, 125, 190), (232, 138, 60), (92, 176, 120), (196, 88, 160)], np.uint8)
SEED = 0
# Colour scoring.  Main regions use plain CIELAB least squares.  Accent regions
# (below) are scored colour-first: Stack5's hue-first metric (core.stack5.metric)
# for clearly coloured targets, blended from plain CIELAB between C* 10 and 20.
# Plain CIELAB ranks solid Coffee Brown above a thin veil of Green for Blond's
# dark olive hair (28 vs 31) because it matches the darkness.  Hue-first alone,
# on the other hand, lets greys take any colour of the right lightness (Rodeo's
# grey-green background went Pink, to_hell_with_it's shadows Red), hence the
# blend and hence main regions stay CIELAB.  The parameters weigh hue harder
# than Stack5's defaults, which count only chroma above C* 12 as colour and
# make the hair (C* ~21) a faint tint.
REGION_HUE_PARAMS = {'c_vis': 4., 'wL_chroma': .3, 'knee': 20., 'wH': 2.5}
# Being MORE vivid than a clearly coloured target costs this fraction of the
# normal chroma weight: green hair printed in a vivid green still reads as green
# hair (at full weight the darker hair pixels scored closer to Black).
CHROMA_EXCESS_WEIGHT = .2
HUE_BLEND = (10., 20.)     # source C* where accent scoring moves from CIELAB to hue-first
_HUE = resolve_hue_params(REGION_HUE_PARAMS)
# Accents: after the regions are chosen, areas the main ramps print in the wrong
# colour family get a ramp of their own - Blond's green hair is 0.8% of the
# plaque, far too little to move a whole-plaque score, yet obvious to the eye.
MAX_TOTAL_REGIONS = 6      # extra regions add no modifiers (one per spool), only tool changes
ACCENT_MIN_CHROMA = 15.    # C* of the cover pixel: only clearly coloured areas
ACCENT_DE = 15.            # hue-first distance counted as "wrong colour"
ACCENT_MIN_MM2 = 30.       # smallest accent kept (connected area)
ACCENT_MAX_SHARE = .04     # an accent is small: larger areas are the main regions' job
                           # (uncapped, In Rainbows' text rows became a 24% 'accent', 29 tool changes)
ACCENT_GAIN_DE = 4.        # the accent's own pixels must get at least this much closer
# Neutral accents: grey / black-and-white areas the ramps print in a colour.
# Blond's barcode and advisory label: their thin white gaps and letters are too
# narrow to print as raised lines, get lowered to mid height by the thin-feature
# cleanup, and mid height is Coffee Brown on the main ramp.  Such areas get a
# ramp from the neutral spools only (Black, White, greys), scored so that adding
# colour to a grey costs NEUTRAL_CHROMA_W times a lightness error.
NEUTRAL_MAX_C = 8.         # cover pixel counts as neutral below this C*
NEUTRAL_PRINT_C = 12.      # ... and as wrongly coloured when its print is above this C*
NEUTRAL_SPOOL_C = 10.      # spools below this C* are neutral
NEUTRAL_CHROMA_W = 3.
NEUTRAL_GROW_MM = 2.       # a neutral accent takes the neutral pixels this close to the
                           # wrongly coloured ones: the whole advisory label, not its specks
TOWER_SCALES = (1., .8, .65, .5)   # purge scale steps if the tower does not fit (as Stack5)


# --------------------------------------------------------------------------- fitting
def _region_stats(lab, n, labels, R, top):
    """Per region and layer count: pixel count W, summed CIELAB WM; per region
    the summed |Lab|^2 SS, so plan costs are the total per-pixel squared error."""
    W = np.zeros((R, top + 1))
    WM = np.zeros((R, top + 1, 3))
    SS = np.zeros(R)
    for r in range(R):
        m = labels == r
        W[r] = np.bincount(n[m], minlength=top + 1)
        for c in range(3):
            WM[r, :, c] = np.bincount(n[m], weights=lab[m, c], minlength=top + 1)
        SS[r] = float((lab[m] ** 2).sum())
    return W, WM, SS


def _hue_d2(src, cand):
    """Stack5's hue-first squared distance (core.stack5.metric.hue_dist2),
    element-wise with broadcasting (src, cand (..., 3) true CIELAB), with the
    Region-layers parameters and a cheaper chroma excess for coloured targets."""
    p = _HUE
    Ls, Cs, hs = lab_to_lch(src)
    Lc, Cc, hc = lab_to_lch(cand)
    wL = p['wL_neutral'] + (p['wL_chroma'] - p['wL_neutral']) * np.clip(Cs / p['knee'], 0., 1.)
    dL = (Lc - Ls) * wL
    colourful = np.clip(Cs / p['knee'], 0., 1.)
    over = np.where(Cc > Cs, 1. - (1. - CHROMA_EXCESS_WEIGHT) * colourful, 1.)
    dC = (Cc - Cs) * p['wC'] * over
    dh = np.abs(hc - hs)
    dh = np.degrees(np.minimum(dh, 2 * np.pi - dh))
    ramp = np.clip((dh - p['h0']) / max(p['h1'] - p['h0'], 1e-6), 0., 1.)
    ceff = np.maximum(Cs - p['c_vis'], 0.)
    dH = p['wH'] * ceff * ramp * (Cc >= p['c_hue_min'])
    loss = np.clip(1. - Cc / np.maximum(np.minimum(Cs, p['c_keep']), 1e-6), 0., 1.)
    dN = p['wN'] * ceff * loss
    return (dL * dL + dC * dC + dH * dH + dN * dN).astype(np.float64)


def _d2(a, b, metric):
    """'lab': squared CIELAB distance.  'neutral': chroma errors weighted
    NEUTRAL_CHROMA_W (neutral accents).  'hue': plain CIELAB for greys and dull
    colours (C* <= HUE_BLEND[0]), hue-first for clearly coloured targets
    (C* >= HUE_BLEND[1]), blended in between.  Hue-first on its own lets a grey
    take any colour of the right lightness - Rodeo's grey-green background went
    Pink - while plain CIELAB turns Blond's green hair Coffee Brown."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if metric == 'neutral':
        return diff[..., 0] ** 2 + NEUTRAL_CHROMA_W ** 2 * (diff[..., 1] ** 2 + diff[..., 2] ** 2)
    lab_d2 = (diff ** 2).sum(-1)
    if metric == 'lab':
        return lab_d2
    _, cs, _ = lab_to_lch(a)
    w = np.clip((cs - HUE_BLEND[0]) / (HUE_BLEND[1] - HUE_BLEND[0]), 0., 1.)
    return (1. - w) * lab_d2 + w * _hue_d2(a, b)


def _plan_costs(fils, W, WM, SS, top, k, metrics):
    """Best swap plan of this dark -> light spool list for every region at once.
    metrics[r]: 'lab' is exact (per-pixel squared error from per-layer sums);
    'hue' (accents) scores each layer's mean colour.  Returns (cost per region,
    starts per region)."""
    special = {m: [r for r, mm in enumerate(metrics) if mm == m] for m in ('hue', 'neutral')}
    B = len(fils)
    layers = np.arange(top + 1)
    R = W.shape[0]
    best_c, best_p = np.full(R, np.inf), [None] * R
    if B == 1:
        chunks = [np.zeros((1, 0), np.int64)]
    else:
        plans = np.array(list(itertools.combinations(range(1, top + 1), B - 1)), np.int64)
        chunks = np.array_split(plans, max(1, len(plans) // 4096))
    for ch in chunks:
        if B == 1:
            pred = np.broadcast_to(linear_rgb_to_lab_d65(fils[0].rgb_lin), (1, top + 1, 3))
        else:
            pred = linear_rgb_to_lab_d65(_stack_colour(layers, ch, fils, k))
        cost = (pred ** 2).sum(-1) @ W.T - 2 * np.einsum('mlc,rlc->mr', pred, WM) + SS       # (M, R)
        for m, rs in special.items():
            if rs:
                mu = WM[rs] / np.maximum(W[rs], 1)[..., None]                            # (h, L, 3)
                cost[:, rs] = (_d2(mu[None], pred[:, None], m) * W[rs][None]).sum(-1)
        i = cost.argmin(0)
        for r in range(R):
            if cost[i[r], r] < best_c[r]:
                best_c[r], best_p[r] = float(cost[i[r], r]), ch[i[r]].copy()
    return best_c, best_p


def _chroma(fil):
    lab = linear_rgb_to_lab_d65(fil.rgb_lin)
    return float(np.hypot(lab[1], lab[2]))


def _layer_spools(S, starts, top):
    seq = np.full(top + 1, S[0])
    for b, s in enumerate(starts, 1):
        seq[s:] = S[b]
    return seq


def _fit(lab, n, labels, R, pool, max_spools, k, restrict=None, use_all=False, metrics=None):
    """Shared palette P and one plan per region (a dark -> light subset of P that
    starts with P's darkest spool).  restrict: only subsets of these pool indices.
    use_all: P is the whole pool (the spools you ticked) and every one of them is
    printed by at least one region - regions still choose their own sequence.
    metrics: per region 'lab' (default), 'hue' (colour accents) or 'neutral'
    (neutral accents, which print only neutral spools above the shared base)."""
    metrics = list(metrics) if metrics is not None else ['lab'] * R
    neutral_ok = {i for i, f in enumerate(pool) if _chroma(f) < NEUTRAL_SPOOL_C}

    def allowed(S, r):
        return metrics[r] != 'neutral' or all(i in neutral_ok for i in S[1:])
    top = int(n.max())
    W, WM, SS = _region_stats(lab, n, labels, R, top)
    present = np.array([[bool(W[r, j:].sum()) for j in range(top + 1)] for r in range(R)])
    universe = sorted(restrict) if restrict is not None else list(range(len(pool)))
    table = {}
    for size in range(1, min(max_spools, len(universe)) + 1):
        for S in itertools.combinations(universe, size):
            table[S] = _plan_costs([pool[i] for i in S], W, WM, SS, top, k, metrics)
    npx = float(len(lab))

    def score(P, choice):
        total = sum(table[S][0][r] for r, (S, _) in enumerate(choice))
        seqs = np.array([_layer_spools(S, st, top) for S, st in choice])
        mixed = int(sum(len(set(seqs[present[:, j], j])) > 1 for j in range(top + 1)))
        distinct = len({(S, tuple(int(v) for v in st)) for S, st in choice})
        mse = total / npx
        return {'J': mse * (1 + MIX_COST * mixed + REGION_COST * (distinct - 1)), 'mse': mse, 'P': P,
                'choice': choice, 'mixed': mixed, 'distinct': distinct, 'top': top, 'metrics': metrics}

    best = None
    if use_all:
        # Every ticked spool must be printed by some region.  Dynamic programming
        # over which spools are covered so far (<= 2^5 states): each region picks
        # its sequence, the cheapest assignment that covers them all wins.  (Trying
        # every combination is 16^regions - with six regions it ran for minutes.)
        P = tuple(universe)
        bit = {i: 1 << j for j, i in enumerate(P)}
        full = (1 << len(P)) - 1
        subs = [S for s in range(1, len(P) + 1) for S in itertools.combinations(P, s) if S[0] == P[0]]
        states = {0: (0., [])}
        for r in range(R):
            nxt = {}
            for mask, (cost, picks) in states.items():
                for S in subs:
                    if not allowed(S, r):
                        continue
                    m2 = mask | sum(bit[i] for i in S)
                    c2 = cost + table[S][0][r]
                    if m2 not in nxt or c2 < nxt[m2][0] - 1e-9:
                        nxt[m2] = (c2, picks + [S])
            states = nxt
        if full not in states:
            raise ValueError(f'the cover spans too few layers to print all {len(P)} spools')
        combo = states[full][1]
        return score(P, [(S, table[S][1][r]) for r, S in enumerate(combo)])
    for size in range(1, min(max_spools, len(universe)) + 1):
        for P in itertools.combinations(universe, size):
            subs = [S for s in range(1, size + 1) for S in itertools.combinations(P, s) if S[0] == P[0]]
            choice = []
            for r in range(R):
                S = min((S for S in subs if allowed(S, r)), key=lambda S: (table[S][0][r], len(S), S))
                choice.append((S, table[S][1][r]))
            if len({i for S, _ in choice for i in S}) < size:
                continue                        # a spool nobody uses: the smaller palette covers it
            cand = score(P, choice)
            if best is None or cand['J'] < best['J'] - 1e-9:
                best = cand
    return best


def _ramp(pool, S, starts, top, k):
    fils = [pool[i] for i in S]
    if len(S) == 1:
        return np.broadcast_to(fils[0].rgb_lin, (top + 1, 3)).copy()
    return _stack_colour(np.arange(top + 1), np.asarray(starts)[None], fils, k)[0]


# --------------------------------------------------------------------------- regions
def _kmeans_hue(lab_img, R, px_mm, seed=SEED):
    """Deterministic k-means++ on the blurred a*, b* of the (re-lit) cover."""
    ab = np.stack([ndimage.gaussian_filter(lab_img[..., c], SMOOTH_MM / px_mm).ravel() for c in (1, 2)], 1)
    rng = np.random.default_rng(seed)
    X = ab[rng.choice(len(ab), min(20000, len(ab)), replace=False)]
    C = [X[rng.integers(len(X))]]
    for _ in range(1, R):
        d = np.min([((X - c) ** 2).sum(1) for c in C], 0)
        C.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(C)
    for _ in range(30):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[a == r].mean(0) if (a == r).any() else C[r] for r in range(R)])
    return ((ab[:, None] - C[None]) ** 2).sum(-1).argmin(1).reshape(lab_img.shape[:2])


def clean_regions(labels, R, px_mm):
    """Majority filter, then islands under MIN_REGION_MM2 join the neighbour they
    touch most, then no two cells of a region touch only at a corner."""
    size = max(1, int(round(MAJORITY_MM / px_mm)) | 1)
    votes = np.stack([ndimage.uniform_filter((labels == r).astype(float), size, mode='nearest')
                      for r in range(R)])
    out = votes.argmax(0)
    min_px = MIN_REGION_MM2 / px_mm ** 2
    for _ in range(6):
        changed = False
        for r in range(R):
            comp, count = ndimage.label(out == r)
            if not count:
                continue
            sizes = np.bincount(comp.ravel())
            for ci in np.nonzero(sizes < min_px)[0]:
                if ci == 0:
                    continue
                m = comp == ci
                ring = ndimage.binary_dilation(m) & ~m
                nb = np.bincount(out[ring], minlength=R)
                nb[r] = 0
                if nb.sum():
                    out[m] = int(nb.argmax())
                    changed = True
        if not changed:
            break
    for _ in range(20):                          # corner-only contacts -> fill one cell
        a, b, c, d = out[:-1, :-1], out[:-1, 1:], out[1:, :-1], out[1:, 1:]
        bad = (a == d) & (b == c) & (a != b)
        if not bad.any():
            break
        rr, cc = np.nonzero(bad)
        out[rr, cc + 1] = out[rr, cc]
    return out


def _relabel(labels):
    """Region ids in order of decreasing area (ties by id): region 0 is the largest."""
    ids, counts = np.unique(labels, return_counts=True)
    order = [int(i) for i, _ in sorted(zip(ids, counts), key=lambda t: (-t[1], t[0]))]
    lut = np.zeros(int(labels.max()) + 1, np.int64)
    for new, old in enumerate(order):
        lut[old] = new
    return lut[labels], len(order)


def pixel_d2(lab, n, labels, fit, pool, k):
    """Per-pixel squared colour distance of the print to the target (fit's metric)."""
    out = np.zeros(len(lab))
    for r, (S, st) in enumerate(fit['choice']):
        m = labels == r
        if m.any():
            ramp = linear_rgb_to_lab_d65(_ramp(pool, S, st, fit['top'], k))
            out[m] = _d2(lab[m], ramp[n[m]], fit['metrics'][r])
    return out


def _exact_cost(fit, d2):
    """Replace the fit's (per-layer mean) score by the exact per-pixel one."""
    mse = float(d2.mean())
    return dict(fit, mse=mse, J=mse * (1 + MIX_COST * fit['mixed'] + REGION_COST * (fit['distinct'] - 1)))


def segment(lab_img, n_img, R, pool, max_spools, k, px_mm, restrict=None, use_all=False):
    """Regions for R ramps on a working grid: k-means start, then alternate the
    palette/plan fit and the per-pixel reassignment.  Returns (labels, fit)."""
    N = lab_img.shape[0]
    lab = lab_img.reshape(-1, 3)
    n = n_img.ravel()
    labels = clean_regions(_kmeans_hue(lab_img, R, px_mm), R, px_mm)
    labels, R = _relabel(labels)
    fit = _fit(lab, n, labels.ravel(), R, pool, max_spools, k, restrict, use_all)
    for _ in range(ITERATIONS):
        ramps = [linear_rgb_to_lab_d65(_ramp(pool, S, st, fit['top'], k)) for S, st in fit['choice']]
        err = np.stack([ndimage.gaussian_filter(_d2(lab, rp[n], 'lab').reshape(N, N), SMOOTH_MM / px_mm)
                        for rp in ramps])
        # a pixel only moves when another ramp is clearly better (ties - e.g. white
        # on top of both ramps - keep the current region instead of flickering)
        cur = np.take_along_axis(err, labels[None], 0)[0]
        choice = np.where(err.min(0) < cur * (1 - SWITCH_MARGIN), err.argmin(0), labels)
        new, R2 = _relabel(clean_regions(choice, R, px_mm))
        if R2 == R and (new != labels).mean() < .005:
            labels = new
            break
        labels, R = new, R2
        fit = _fit(lab, n, labels.ravel(), R, pool, max_spools, k, fit['P'], use_all)   # palette fixed
    fit = _fit(lab, n, labels.ravel(), R, pool, max_spools, k, restrict, use_all)     # palette free again
    return labels, _exact_cost(fit, pixel_d2(lab, n, labels.ravel(), fit, pool, k))


ACCENT_KINDS = ('lab', 'hue', 'neutral')     # kind map: 0 main region, 1 colour accent, 2 neutral accent


def region_metrics(labels, kind, R):
    """Each region's scoring from the accent kind most of its pixels carry."""
    return [ACCENT_KINDS[int(np.bincount(kind[labels == r], minlength=3).argmax())] for r in range(R)]


def add_accents(lab_img, n_img, labels, fit, pool, max_spools, k, px_mm, use_all=False,
                max_total=MAX_TOTAL_REGIONS):
    """Give small areas that print in the wrong colour family their own ramp.

    The main regions keep plain CIELAB scoring.  Two kinds of candidate:
    * colour: C* >= ACCENT_MIN_CHROMA and colour-first distance (_d2 'hue') to
      the print >= ACCENT_DE - Blond's green hair printed brown;
    * neutral: C* <= NEUTRAL_MAX_C but printed with C* >= NEUTRAL_PRINT_C -
      Blond's barcode printed brown - plus the neutral pixels within
      NEUTRAL_GROW_MM of them.  These may only use neutral spools.
    Candidates are cleaned (1 mm open/close) and kept as connected areas of
    ACCENT_MIN_MM2 .. ACCENT_MAX_SHARE of the plaque; colour areas are grouped by
    mean hue, neutral areas form one group.  Neutral groups go first, then the
    group with the most excess error; it becomes a new region, the palette and every plan refitted together;
    it is kept when its own pixels get ACCENT_GAIN_DE closer (in its own
    scoring) and the rest of the cover gets no more than 1% worse.  Accent pixels
    are never split by a later accent.  Repeats up to max_total regions.
    Returns (labels, fit, accents, kind) - kind: 0 main, 1 colour, 2 neutral."""
    N = lab_img.shape[0]
    lab = lab_img.reshape(-1, 3)
    n = n_img.ravel()
    _, C, _ = lab_to_lch(lab)
    se = np.ones((max(1, int(round(1. / px_mm))),) * 2, bool)
    max_mm2 = ACCENT_MAX_SHARE * N * N * px_mm ** 2
    accents = []
    tried = set()
    kind = np.zeros(N * N, np.uint8)
    while fit['distinct'] < max_total and labels.max() + 1 < max_total:
        pred = _pred_lab(lab, n, labels.ravel(), fit, pool, k)
        _, Cp, _ = lab_to_lch(pred)
        err = {1: _d2(lab, pred, 'hue'), 2: _d2(lab, pred, 'neutral')}
        neutral = C <= NEUTRAL_MAX_C
        # small wrongly coloured neutral areas (big ones - Blond's grey tiles - are
        # the main regions' job and would swallow the barcode when grown), each
        # grown into the neutral pixels around it so a label is one area
        grow_se = np.ones((2 * int(round(NEUTRAL_GROW_MM / px_mm)) + 1,) * 2, bool)
        wrong = ndimage.binary_closing((neutral & (Cp >= NEUTRAL_PRINT_C) & (kind == 0)).reshape(N, N), grow_se)
        wcomp, _ = ndimage.label(wrong)
        wsize = np.bincount(wcomp.ravel()) * px_mm ** 2
        small = np.isin(wcomp, np.nonzero(wsize <= max_mm2)[0][1:])
        grow = ndimage.binary_dilation(small, grow_se)
        cands = {1: (C >= ACCENT_MIN_CHROMA) & (np.sqrt(err[1]) >= ACCENT_DE),
                 2: grow.ravel() & neutral}
        groups = []                                                    # (excess error, kind, mask)
        for kd, cand in cands.items():
            cand = ndimage.binary_closing(ndimage.binary_opening((cand & (kind == 0)).reshape(N, N), se), se)
            comp, count = ndimage.label(cand)
            if not count:
                continue
            sizes = np.bincount(comp.ravel()) * px_mm ** 2
            big = np.nonzero((sizes >= ACCENT_MIN_MM2) & (sizes <= max_mm2))[0]
            big = big[big > 0]
            if not len(big):
                continue
            flat = comp.ravel()
            if kd == 1 and len(big) > 1:      # a hairdo stays one area; similar hues group together
                means = np.array([lab[flat == ci, 1:].mean(0) for ci in big])
                comp_group = _kmeans_points(means, min(3, len(big)))
            else:
                comp_group = np.zeros(len(big), np.int64)
            for g in sorted(set(comp_group.tolist())):
                mask = np.isin(flat, big[comp_group == g])
                if ACCENT_MIN_MM2 <= mask.sum() * px_mm ** 2 <= max_mm2:
                    groups.append((float(err[kd][mask].sum()), kd, mask))
        if not groups:
            break
        rest_before = _d2(lab, pred, 'lab')
        accepted = False
        # neutral areas first (black-and-white text, labels and barcodes printed in a
        # colour stand out most), then by excess error
        for _, kd, mask in sorted(groups, key=lambda t: (t[1] != 2, -t[0])):
            key = (kd, int(mask.sum()), int(np.flatnonzero(mask)[0]))
            if key in tried:
                continue
            tried.add(key)
            new = labels.ravel().copy()
            new[mask] = labels.max() + 1
            new, R2 = _relabel(new.reshape(N, N))
            kind_try = kind.copy()
            kind_try[mask] = kd
            cand_fit = _fit(lab, n, new.ravel(), R2, pool, max_spools, k, None, use_all,
                            region_metrics(new.ravel(), kind_try, R2))
            pred2 = _pred_lab(lab, n, new.ravel(), cand_fit, pool, k)
            metric = ACCENT_KINDS[kd]
            before = float(np.sqrt(err[kd][mask]).mean())
            after = float(np.sqrt(_d2(lab[mask], pred2[mask], metric)).mean())
            rest = kind_try == 0
            rest_after = _d2(lab[rest], pred2[rest], 'lab')
            if before - after >= ACCENT_GAIN_DE and rest_after.mean() <= rest_before[rest].mean() * 1.01:
                S, _ = cand_fit['choice'][int(new.ravel()[mask][0])]
                accents.append({'kind': 'neutral' if kd == 2 else 'colour',
                                'area_mm2': round(float(mask.sum() * px_mm ** 2), 1),
                                'mean_dE_before': round(before, 2), 'mean_dE_after': round(after, 2),
                                'spools': [pool[i].name for i in S]})
                labels, fit, kind, accepted = new, cand_fit, kind_try, True
                break
        if not accepted:
            break
    return labels, fit, accents, kind.reshape(N, N)


def _pred_lab(lab, n, labels, fit, pool, k):
    out = np.zeros_like(lab)
    for r, (S, st) in enumerate(fit['choice']):
        m = labels == r
        out[m] = linear_rgb_to_lab_d65(_ramp(pool, S, st, fit['top'], k))[n[m]]
    return out


def _kmeans_points(X, K, seed=SEED):
    """Deterministic k-means++ on rows of X; returns a label per row."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, K):
        d = np.min([((X - c) ** 2).sum(1) for c in C], 0)
        if d.sum() <= 0:
            break
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(30):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(len(C))])
    return ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)


def _merge_identical(labels, fit):
    """Regions that ended with the same plan are one region."""
    keys, remap = [], []
    for S, st in fit['choice']:
        key = (S, tuple(int(v) for v in st))
        if key not in keys:
            keys.append(key)
        remap.append(keys.index(key))
    return np.asarray(remap)[labels], len(keys)


# --------------------------------------------------------------------------- geometry
def region_polygon(cell_mask, pitch):
    """Exact outline (with holes) of a cell mask, in mesh coordinates: cell (i, j)
    spans x [j, j+1] * pitch, y [H-2-i, H-1-i] * pitch (heightfield_to_trimesh)."""
    Hc = cell_mask.shape[0]
    boxes = []
    for i in range(Hc):                         # integer cell units: the union is exact
        row = np.concatenate([[False], cell_mask[i], [False]])
        edges = np.flatnonzero(row[1:] != row[:-1])
        for j0, j1 in zip(edges[::2], edges[1::2]):
            boxes.append(shapely_box(int(j0), Hc - 1 - i, int(j1), Hc - i))
    if not boxes:
        return shapely.Polygon()
    merged = shapely.union_all(boxes).simplify(0)       # drop collinear staircase vertices
    return shapely.affinity.scale(merged, pitch, pitch, origin=(0, 0))


def _prism(poly, z0, z1):
    parts = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    meshes = []
    for p in parts:
        m = trimesh.creation.extrude_polygon(p, z1 - z0)
        m.apply_translation([0, 0, z0])
        meshes.append(m)
    return trimesh.util.concatenate(meshes)


def band_modifiers(labels_cells, fit, pool, pool_index, pitch, top_z):
    """One modifier volume per spool other than the base: the union of every
    (region, band) printed in that spool, as exact prisms (region outline x the
    band's layer range).  The object itself is the base spool, so nothing else
    is needed; regions are disjoint and a spool is one band per region, so the
    modifiers never overlap and the result does not depend on their order.
    Returns [(mesh, extruder, spool name)] in extruder order."""
    per_spool = {}
    for r, (S, starts) in enumerate(fit['choice']):
        if len(S) == 1:
            continue
        poly = region_polygon(labels_cells == r, pitch)
        if poly.is_empty:
            continue
        lo = [1] + [int(v) for v in starts]            # band b covers colour layers lo[b] .. lo[b+1]-1
        for b in range(1, len(S)):
            z0 = BASE_TOP + (lo[b] - 1) * LAYER_H     # bottom of the band's first layer
            z1 = top_z if b == len(S) - 1 else BASE_TOP + (lo[b + 1] - 1) * LAYER_H
            if z1 > z0:
                per_spool.setdefault(S[b], []).append(_prism(poly, round(z0, 6), round(z1, 6)))
    return [(trimesh.util.concatenate(ms), pool_index[i] + 1, pool[i].name)
            for i, ms in sorted(per_spool.items(), key=lambda kv: pool_index[kv[0]])]


# --------------------------------------------------------------------------- 3MF
def _add_modifiers(members, mods, plaque_xy):
    obj = members['3D/Objects/object_1.model'].decode()
    extra = []
    for k, (mesh, ext, name) in enumerate(mods):
        oid = 3 + k
        vb, tb = io.BytesIO(), io.BytesIO()
        _write_vertices_bytes(vb, mesh.vertices, '%.4f')
        _write_triangles_bytes(tb, mesh.faces)
        extra.append(f'  <object id="{oid}" type="model">\n   <mesh>\n    <vertices>\n{vb.getvalue().decode()}'
                     f'    </vertices>\n    <triangles>\n{tb.getvalue().decode()}    </triangles>\n   </mesh>\n  </object>\n')
    members['3D/Objects/object_1.model'] = obj.replace(' </resources>', ''.join(extra) + ' </resources>').encode()
    main = members['3D/3dmodel.model'].decode()
    comps = ''.join(f'    <component p:path="/3D/Objects/object_1.model" objectid="{3 + k}" '
                    f'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n' for k in range(len(mods)))
    main = main.replace('   </components>', comps + '   </components>')
    ms = members['Metadata/model_settings.config'].decode()
    parts = ''.join(
        f'    <part id="{3 + k}" subtype="modifier_part">\n'
        f'      <metadata key="name" value={quoteattr(name)}/>\n'
        f'      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n'
        f'      <metadata key="extruder" value="{ext}"/>\n'
        f'    </part>\n' for k, (_, ext, name) in enumerate(mods))
    ms = ms.replace('  </object>', parts + '  </object>', 1)
    if plaque_xy is not None:
        t = writer3mf._fmt_transform(plaque_xy[0], plaque_xy[1], 0.0)
        main = re.sub(r'(<item objectid="2"[^>]* transform=")[^"]*"', lambda m: m.group(1) + t + '"', main)
        ms = re.sub(r'(<assemble_item object_id="2"[^>]* transform=")[^"]*"', lambda m: m.group(1) + t + '"', ms)
    members['3D/3dmodel.model'] = main.encode()
    members['Metadata/model_settings.config'] = ms.encode()


def _plan_tower(fils, fit, pool_index, width_mm):
    """Stack5's flush model and tower layout for the spools each layer prints."""
    top = fit['top']
    seqs = [_layer_spools(S, st, top) for S, st in fit['choice']]
    present = fit['present']
    layer_sets = []
    for j in range(1, top + 1):
        used = sorted({pool_index[int(seqs[r][j])] for r in range(len(seqs)) if present[r][j]})
        layer_sets.append(used)
    template = writer3mf.load_template(writer3mf.DEFAULT_TEMPLATE_SOURCES)
    min_flush = flushmod.min_flush_from_config(template['base'])
    for scale in TOWER_SCALES:
        plan = flushmod.plan_tower([f.hex for f in fils], layer_sets, float(width_mm), float(width_mm),
                                   LAYER_H, min_flush=float(min_flush), scale=scale)
        if plan['tower']['fits']:
            return plan
    raise ValueError(f"the prime tower does not fit beside a {width_mm:g} mm plaque even at purge scale "
                     f"{TOWER_SCALES[-1]:g}; use a smaller width or fewer regions")


# --------------------------------------------------------------------------- entry point
def convert_region_band(image_path, out_dir, width_mm=200., pitch_mm=.2, palette='auto',
                        filaments_json=None, max_regions=MAX_REGIONS, max_spools=MAX_SPOOLS,
                        contrast=DEFAULT_CONTRAST, min_feature_mm=MIN_FEATURE_MM, k_opaque=None,
                        accents=True):
    """Reference layers relief with a separate spool ramp per region (see module doc).

    palette 'auto' picks up to max_spools spools from the library; a list of
    2-5 library names prints exactly those spools (each one in at least one
    region), as Reference layers does with ticked spools.  max_regions 1 gives
    Reference layers' single ramp (printed with plate-wide swaps, no tower).
    accents adds regions for small areas printed in the wrong colour family
    (add_accents), up to MAX_TOTAL_REGIONS.
    """
    if not 10 <= width_mm <= 240 or not .08 <= pitch_mm <= .5:
        raise ValueError('width must be 10..240 mm and pitch .08..0.5 mm')
    if not 1 <= int(max_regions) <= 4:
        raise ValueError('max_regions must be 1..4')
    k = DEFAULT_K_OPAQUE if k_opaque is None else float(k_opaque)
    library = load_filament_library(str(filaments_json or DEFAULT_FILAMENTS_JSON))
    if palette is None or palette == 'auto':
        pool = sorted(library.values(), key=lambda f: (f.luminance, f.name))
        palette_mode = 'auto'
    else:
        pool = palette_filaments(palette, library)              # validates, sorts dark -> light
        max_spools = len(pool)
        palette_mode = 'custom'
    if not 1 <= int(max_spools) <= 5:
        raise ValueError('max_spools must be 1..5')
    use_all = palette_mode == 'custom'

    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    if image.width != image.height:
        raise ValueError('Region layers requires a square image; crop explicitly first')

    # Geometry: identical to Reference layers (1 mm backing).
    n_px = int(round(width_mm / pitch_mm)) + 1
    pitch = width_mm / (n_px - 1)
    rgb = np.asarray(image.resize((n_px, n_px), Image.Resampling.BILINEAR))
    t, levels = auto_tone(rgb, contrast)
    target = toned_target(rgb, t)
    raw = height_from_tone(t) + SHIFT
    h, window_px = remove_thin_features(raw, pitch, min_feature_mm)
    n_full = column_layers(h, BASE_TOP)
    lab_full = linear_rgb_to_lab_d65(target.reshape(-1, 3))

    # Regions on a coarser working grid, R = 1..max_regions, keep the best J.
    n_w = int(round(width_mm / WORK_MM)) + 1
    px_w = width_mm / (n_w - 1)
    rgb_w = np.asarray(image.resize((n_w, n_w), Image.Resampling.BILINEAR))
    t_w = auto_tone(rgb_w, contrast)[0]
    lab_w = linear_rgb_to_lab_d65(toned_target(rgb_w, t_w))
    h_w, _ = remove_thin_features(height_from_tone(t_w) + SHIFT, px_w, min_feature_mm)
    n_w_img = column_layers(h_w, BASE_TOP)
    tried = []
    best_labels, best_fit = np.zeros((n_w, n_w), np.int64), None
    for R in range(1, int(max_regions) + 1):
        if R == 1:
            labels_w = np.zeros((n_w, n_w), np.int64)
            fit_w = _fit(lab_w.reshape(-1, 3), n_w_img.ravel(), labels_w.ravel(), 1, pool, max_spools, k,
                         use_all=use_all)
            fit_w = _exact_cost(fit_w, pixel_d2(lab_w.reshape(-1, 3), n_w_img.ravel(), labels_w.ravel(),
                                                fit_w, pool, k))
        else:
            labels_w, fit_w = segment(lab_w, n_w_img, R, pool, max_spools, k, px_w, use_all=use_all)
        tried.append({'regions_asked': R, 'regions': int(labels_w.max()) + 1, 'J': round(fit_w['J'], 3),
                      'mse': round(fit_w['mse'], 3), 'mixed_layers': fit_w['mixed']})
        if best_fit is None or fit_w['J'] < best_fit['J'] - 1e-9:
            best_labels, best_fit = labels_w, fit_w
    accent_log, accent_w = [], np.zeros((n_w, n_w), np.uint8)
    if accents:
        best_labels, best_fit, accent_log, accent_w = add_accents(lab_w, n_w_img, best_labels, best_fit,
                                                                  pool, max_spools, k, px_w, use_all)

    # Final fit on the full grid.  Cells get the working label under their centre.
    cells = np.asarray(Image.fromarray(best_labels.astype(np.uint8)).resize((n_px - 1, n_px - 1),
                                                                             Image.Resampling.NEAREST)).astype(np.int64)
    labels_px = np.pad(cells, ((0, 1), (0, 1)), mode='edge')
    accent_cells = np.asarray(Image.fromarray(accent_w.astype(np.uint8)).resize((n_px - 1, n_px - 1),
                                                                                Image.Resampling.NEAREST))
    accent_px = np.pad(accent_cells, ((0, 1), (0, 1)), mode='edge').ravel()
    R = int(labels_px.max()) + 1
    fit = _fit(lab_full, n_full.ravel(), labels_px.ravel(), R, pool, max_spools, k, use_all=use_all,
               metrics=region_metrics(labels_px.ravel(), accent_px, R))
    merged, R2 = _merge_identical(labels_px, fit)
    if R2 < R:
        cells = np.asarray(_merge_identical(cells, fit)[0])
        labels_px, R = merged, R2
        fit = _fit(lab_full, n_full.ravel(), labels_px.ravel(), R, pool, max_spools, k, use_all=use_all,
                   metrics=region_metrics(labels_px.ravel(), accent_px, R))
    fit = _exact_cost(fit, pixel_d2(lab_full, n_full.ravel(), labels_px.ravel(), fit, pool, k))
    top = fit['top']
    fit['present'] = [[bool((n_full[labels_px == r] >= j).any()) for j in range(top + 1)] for r in range(R)]

    # Palette = the spools the plans use, dark -> light; extruder = position + 1.
    used = sorted({i for S, _ in fit['choice'] for i in S})
    fils = [pool[i] for i in used]
    pool_index = {i: used.index(i) for i in used}

    # Predicted print colours, target, height and region previews.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem + '_region_band'
    pred = np.zeros((n_px * n_px, 3))
    flat_labels = labels_px.ravel()
    flat_n = n_full.ravel()
    for r, (S, st) in enumerate(fit['choice']):
        ramp = _ramp(pool, S, st, top, k)
        m = flat_labels == r
        pred[m] = ramp[flat_n[m]]
    de = np.linalg.norm(lab_full - linear_rgb_to_lab_d65(pred), axis=1)
    predicted_png = out / (stem + '_predicted.png')
    Image.fromarray(np.uint8(np.round(linear_to_srgb(pred.reshape(n_px, n_px, 3)) * 255))).save(predicted_png)
    Image.fromarray(np.uint8(np.round(linear_to_srgb(target) * 255))).save(out / (stem + '_target.png'))
    Image.fromarray(np.uint8(np.clip((h - SHIFT - FLOOR) / AMPLITUDE, 0, 1) * 255)).save(out / (stem + '_height.png'))
    # region map: the predicted print, each region washed with its own colour
    shade = linear_to_srgb(pred.reshape(n_px, n_px, 3))
    rmap = np.uint8(np.round((.45 * shade + .55 * REGION_MAP_RGB[labels_px % len(REGION_MAP_RGB)] / 255) * 255))
    Image.fromarray(rmap).save(out / (stem + '_regions.png'))

    # Mesh (Reference's), plate layout and modifiers.
    mesh = heightfield_to_trimesh(h, pitch)
    top_z = round(float(h.max()) + .1, 3)      # modifiers end just above the relief
    scripts = json.loads((REPO / 'assets/x2d_machine_gcodes.json').read_text())['config']
    overrides = dict(scripts, initial_layer_print_height='0.2', layer_height='0.08',
                     print_settings_id='Region Band 0.08mm X2D',
                     top_surface_pattern='monotonic', bottom_surface_pattern='monotonic',
                     internal_solid_infill_pattern='monotonic', brim_type='no_brim',
                     nozzle_temperature=['220'] * len(fils),
                     nozzle_temperature_initial_layer=['220'] * len(fils))
    single = R == 1
    tower = None
    if single:
        S, st = fit['choice'][0]
        swaps = [(round(BASE_TOP + int(v) * LAYER_H, 6), pool_index[S[b]] + 1, pool[S[b]].hex)
                 for b, v in enumerate(st, 1)]
        mods, plaque_xy = [], None
    else:
        swaps = []
        tower = _plan_tower(fils, fit, pool_index, width_mm)
        overrides.update(flushmod.tower_config_keys(tower))
        overrides.update({'flush_into_infill': '0', 'flush_into_objects': '0', 'flush_into_support': '1'})
        mods = band_modifiers(cells, fit, pool, pool_index, pitch, top_z)
        plaque_xy = tower['layout']['plaque_xy']
    target_path = out / (stem + '.3mf')
    write_band_3mf(mesh, str(target_path), fils, swaps, title=stem,
                   description='Region layers: Reference relief, one spool ramp per region',
                   size_mm=width_mm, print_overrides=overrides, thumbnail_png=str(predicted_png))
    with zipfile.ZipFile(target_path) as z:
        members = {name: z.read(name) for name in z.namelist()}
    if mods:
        _add_modifiers(members, mods, plaque_xy)
    cfg = json.loads(members['Metadata/project_settings.config'])
    complete_two_nozzle_process(cfg)
    cfg['inherits_group'][0] = ''
    cfg['different_settings_to_system'][0] = ';'.join(sorted(
        set(cfg['different_settings_to_system'][0].split(';')) | (set(overrides) - set(scripts))))
    cfg['different_settings_to_system'][-1] = ';'.join(sorted(set(scripts)))
    members['Metadata/project_settings.config'] = json.dumps(cfg, indent=2).encode()
    members['Metadata/layer_config_ranges.xml'] = (
        b'<?xml version="1.0"?><objects><object id="1"><range min_z="0" max_z="1">'
        b'<option opt_key="layer_height">0.2</option></range></object></objects>')
    # Repeatable bytes: UUIDs derived from the geometry, no creation date.
    seed_hex = hashlib.sha256(h.tobytes() + cells.tobytes()).hexdigest()
    ids = {}

    def stable_id(match):
        old = match.group().lower()
        if old not in ids:
            ids[old] = str(uuid.uuid5(uuid.NAMESPACE_URL, seed_hex + ':' + str(len(ids)))).encode()
        return ids[old]
    for name, data in list(members.items()):
        if name.endswith(('.model', '.config', '.xml')):
            data = re.sub(rb'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', stable_id, data)
            data = re.sub(rb' <metadata name="CreationDate">[^<]*</metadata>\n', b'', data)
            members[name] = data
    with zipfile.ZipFile(target_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    np.save(out / (stem + '_height.npy'), h)
    np.save(out / (stem + '_regions.npy'), labels_px.astype(np.uint8))

    regions = []
    for r, (S, st) in enumerate(fit['choice']):
        m = flat_labels == r
        regions.append({'region': r + 1, 'area_share': round(float(m.mean()), 4),
                        'accent': {'lab': None, 'hue': 'colour', 'neutral': 'neutral'}[fit['metrics'][r]],
                        'spools': [pool[i].name for i in S],
                        'swaps_mm': [round(BASE_TOP + int(v) * LAYER_H, 6) for v in st],
                        'mean_dE': round(float(de[m].mean()), 3)})
    report = {
        'engine': 'region-layers-v1', 'source': str(Path(image_path).resolve()),
        'source_sha256': hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        'height_sha256': hashlib.sha256(h.tobytes()).hexdigest(),
        'geometry': 'Reference layers relief (same tone, mesh and thin-feature cleanup)',
        'tone': {'contrast': float(contrast), 'brightness_levels': levels},
        'pitch_mm': pitch, 'width_mm': float(width_mm), 'min_feature_mm': float(min_feature_mm or 0),
        'thin_feature_window_px': window_px,
        'palette_mode': palette_mode, 'filaments': [f.name for f in fils],
        'filament_td_mm': {f.name: float(f.td_mm) for f in fils},
        'regions': regions, 'region_count': R,
        'regions_tried': tried, 'accents': accent_log,
        'scoring': 'plain CIELAB; accent regions colour-first (hue-first above C* 20, blended from C* 10)',
        'mixed_layers': fit['mixed'],
        'cost': {'J': round(fit['J'], 3), 'mse': round(fit['mse'], 3), 'mix_cost': MIX_COST,
                 'region_cost': REGION_COST},
        'mean_dE': round(float(de.mean()), 3), 'p90_dE': round(float(np.percentile(de, 90)), 3),
        'print': ('plate-wide swaps (one region: same as Reference layers)' if single else
                  f'{len(mods)} modifier volumes; prime tower'),
        'swap_entries': swaps,
        'tool_changes_planned': None if tower is None else tower['tool_changes'],
        'purge_mm3_planned': None if tower is None else tower['purge_total_mm3'],
        'tower': None if tower is None else tower['tower'],
        'plaque_xy': plaque_xy,
        'watertight': bool(mesh.is_watertight),
        'preview_kind': 'predicted print colour from spool TDs (Beer-Lambert); regions map saved too',
    }
    recipe = out / (stem + '.json')
    recipe.write_text(json.dumps(report, indent=2, default=float) + '\n')
    return {'threemf': str(target_path), 'preview_png': str(predicted_png),
            'regions_png': str(out / (stem + '_regions.png')), 'recipe_json': str(recipe), 'stats': report}
