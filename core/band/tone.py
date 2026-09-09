"""Tone curve: image lightness -> relief height (HueForge-style luminance mapping).

Dissecting the professional reference (Astroworld borderless, HueForge export;
scratchpad ref_tone_curve.py / ref_lum_fit.py) showed that its heightfield is
NOT a nearest-colour match against the filament ramp: registered to the source
image, the printed height is a smooth monotone function of image lightness,

    z(L*) = z_lo + (z_hi - z_lo) * clip((L*/100 - q_lo) / (q_hi - q_lo), 0, 1) ** gamma

with an RMS residual of 0.036 mm (0.45 layers) over 1e6 vertices.  Every one of
the 22 tonal layers then carries 1-10 % of the surface because the layer
distribution is simply the image's lightness histogram binned every ~4.5 L*.
Nearest-colour matching against a physically saturating ramp cannot do that:
it collapses whole lightness ranges onto the few rows where the ramp changes
colour (e.g. the last four orange layers of the reference schedule differ by
< 1 L* each) and moves band boundaries to the L* midpoints between rows.

The default parameters are the least-squares fit to the reference with the
processor's endpoints (z_lo = engraving floor 0.521 mm, z_hi = top of the last
layer); q_lo < 0 reproduces the reference's black floor (pure black sits at
~0.58 mm, inside layer 6, not on the 0.52 floor).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from core.band.schedule import BandSchedule

# Fitted to REF_ASTRO (see module docstring): q_lo -0.135, q_hi 1.061, gamma 1.568.
REFERENCE_Q_LO = -0.13
REFERENCE_Q_HI = 1.06
REFERENCE_GAMMA = 1.55


@dataclass(frozen=True)
class ToneCurve:
    """Monotone map from lightness q = L*/100 to a relief fraction t in [0, 1]."""
    q_lo: float = REFERENCE_Q_LO
    q_hi: float = REFERENCE_Q_HI
    gamma: float = REFERENCE_GAMMA

    def __post_init__(self):
        if not (self.q_hi > self.q_lo):
            raise ValueError("tone curve needs q_hi > q_lo")
        if self.gamma <= 0:
            raise ValueError("tone curve gamma must be > 0")

    @classmethod
    def reference(cls) -> 'ToneCurve':
        return cls()

    @classmethod
    def linear(cls) -> 'ToneCurve':
        """HueForge's untouched default: L* 0 -> base, L* 100 -> top."""
        return cls(0.0, 1.0, 1.0)

    def t(self, L_star) -> np.ndarray:
        """Relief fraction for CIELAB L* (0..100)."""
        q = np.asarray(L_star, dtype=np.float64) / 100.0
        u = np.clip((q - self.q_lo) / (self.q_hi - self.q_lo), 0.0, 1.0)
        return u ** self.gamma

    def height(self, L_star, z_lo: float, z_hi: float) -> np.ndarray:
        """Continuous column height in mm."""
        return z_lo + (z_hi - z_lo) * self.t(L_star)

    def layer(self, L_star, schedule: 'BandSchedule', z_lo: float, z_hi: float | None = None) -> np.ndarray:
        """Printed-layer count of the column (slice-plane rule), clipped to n0..K."""
        z_hi = schedule.top_z(schedule.n_layers) if z_hi is None else z_hi
        n = schedule.layer_of_height(self.height(L_star, z_lo, z_hi))
        return np.clip(n, schedule.n0, schedule.n_layers)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_any(cls, v) -> 'ToneCurve':
        if v is None:
            return cls()
        if isinstance(v, ToneCurve):
            return v
        if isinstance(v, dict):
            return cls(**v)
        if isinstance(v, str):
            key = v.strip().lower()
            if key in ('reference', 'ref', 'hueforge', 'default'):
                return cls()
            if key == 'linear':
                return cls.linear()
            parts = [float(x) for x in v.split(',')]
            return cls(*parts)
        return cls(*[float(x) for x in v])


def engrave_floor(schedule: 'BandSchedule') -> float:
    """Reference relief minimum: the slice plane between layers base_min_layers
    and base_min_layers+1 plus 1e-3 mm, so the darkest pixels are unambiguously
    inside layer base_min_layers+1 (0.521 mm for FL 0.16 / lh 0.08 / 5)."""
    return float(schedule.top_z(int(schedule.base_min_layers)) + schedule.layer_h / 2.0 + 1e-3)
