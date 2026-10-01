"""Image-to-height policies, independent of the physical filament prediction.

Color Match has its own reference core. Changing the print filaments must not
change geometry unless the reference core is explicitly copied again.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from core.band.optics import Filament, BeerLambertModel, srgb_to_lab_d65, linear_to_srgb
from core.band.pipeline import composite_heights
from core.band.schedule import BandSchedule

MODES = ('luminance', 'combo', 'max_channel', 'color_match', 'color_aware', 'color_pop')
METRICS = ('rgb', 'cielab', 'hsl', 'dot')


def reference_ramp(stops: list[dict], first_layer: float, layer_h: float, k_opaque: float):
    if not 1 <= len(stops) <= 32:
        raise ValueError('Mesh Core needs 1–32 color stops')
    ends = [int(s['layer']) for s in stops]
    if ends[0] < 1 or any(b <= a for a, b in zip(ends, ends[1:])):
        raise ValueError('Mesh Core end layers must increase from bottom to top')
    if ends[-1] > 1000:
        raise ValueError('Mesh Core must fit within 1000 layers')
    names = [f'reference-{i}' for i in range(len(stops))]
    counts = np.diff([0] + ends).tolist()
    schedule = BandSchedule(tuple(names), tuple(counts), first_layer, layer_h,
                            min(5, counts[0]))
    filaments = {n: Filament.from_hex(n, s['hex'], s.get('td_mm', 1))
                 for n, s in zip(names, stops)}
    heights = schedule.heights()
    colors = linear_to_srgb(composite_heights(heights, schedule, filaments,
                                             BeerLambertModel(k_opaque)))
    return heights, colors


def features(rgb: np.ndarray, metric: str) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if metric == 'rgb':
        return rgb
    if metric == 'cielab':
        return srgb_to_lab_d65(rgb)
    if metric == 'dot':
        # Include magnitude so black and neutral grays remain distinguishable.
        norm = np.linalg.norm(rgb, axis=-1, keepdims=True)
        return np.concatenate([rgb / np.maximum(norm, 1e-12), norm], axis=-1)
    if metric == 'hsl':
        import cv2
        hls = cv2.cvtColor(rgb.astype(np.float32).reshape(-1, 1, 3), cv2.COLOR_RGB2HLS)[:, 0]
        angle = np.deg2rad(hls[:, 0]); sat = hls[:, 2]
        return np.stack([sat*np.cos(angle), sat*np.sin(angle), hls[:, 1]], axis=-1).reshape(rgb.shape[:-1]+(3,))
    raise ValueError(f'Unknown matching metric: {metric}')


def match_heights(rgb: np.ndarray, heights: np.ndarray, colors: np.ndarray,
                  metric: str = 'cielab') -> np.ndarray:
    # Drop duplicate colors before building the tree: exact ties choose the
    # lowest layer deterministically instead of the KD-tree's arbitrary row.
    reference = features(colors, metric)
    _, first = np.unique(np.round(reference, 9), axis=0, return_index=True)
    first.sort()
    tree = cKDTree(reference[first])
    target = features(rgb, metric).reshape(-1, reference.shape[-1])
    index = tree.query(target)[1]
    return heights[first[index]].reshape(rgb.shape[:2])


def mapped_fraction(rgb: np.ndarray, luminance: np.ndarray, mode: str, tone,
                    combo: float = 1, tolerance: float = 8, split: float = .5,
                    reverse: bool = False, channel_order: str = 'BGR') -> np.ndarray:
    q = luminance
    if mode in ('combo', 'max_channel'):
        mix = 0 if mode == 'max_channel' else combo
        q = mix * q + (1-mix) * np.max(rgb, axis=-1)*100
    t = tone.t(q)
    if mode == 'color_pop':
        gray = (rgb.max(-1)-rgb.min(-1))*255 <= tolerance
        upper = ~gray if not reverse else gray
        t = np.where(upper, split+(1-split)*t, split*t)
    elif mode == 'color_aware':
        if sorted(channel_order) != ['B', 'G', 'R']:
            raise ValueError('Color Aware order must contain R, G and B once each')
        channel = np.argmax(rgb, axis=-1)
        # Equal neutral channels default to blue, matching the documented rule.
        gray = (rgb.max(-1)-rgb.min(-1))*255 <= tolerance
        channel = np.where(gray, 2, channel)
        ranks = np.array([channel_order.index(c) for c in 'RGB'])
        rank = ranks[channel]
        if reverse:
            rank = 2-rank
        t = (rank+t)/3
    return t
