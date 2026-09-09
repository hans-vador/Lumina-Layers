"""Colour science and optical (translucency) models for Band mode.

All colour maths is done in *linear* sRGB (D65).  ``srgb_to_lab_d65`` returns
true CIELAB (L* 0..100), NOT OpenCV's 8-bit scaled Lab.

Optical models turn a :class:`~core.band.schedule.BandSchedule` into the ramp of
colours seen from above for a column of k layers, k = n0..K:

* :class:`BeerLambertModel` - o(t, TD) = 1 - exp(-k_opaque * t / TD)
* :class:`LinearTDModel`    - o(t, TD) = min(1, t / TD)
* :class:`MeasuredRampModel` - measured sRGB rows loaded from a ramp-board .npz
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from core.band.schedule import BandSchedule

DEFAULT_SETTINGS_ID = 'Bambu PLA Basic @BBL X2D 0.4 nozzle'
DEFAULT_FILAMENT_ID = 'GFA00'

# Beer-Lambert steepness: o(t) = 1 - exp(-k_opaque * t / TD).
# The filament library seeds TD from HueForge-style values, where TD is the
# thickness at which light "no longer passes" (visually fully opaque), so the
# model must be ~fully opaque at t = TD: k_opaque = 7 -> o(TD) = 1 - e^-7 = 99.9%.
# (The earlier k_opaque = 3, o(TD) = 95%, left every band under-saturated: with
# the seeds, 4 layers of Red over Black reached only L* 36 (pure Red: 45), the
# 8-layer White band spanned L* 63..80, and Lumina's face-down wedges
# "white over red ~45% at 4 layers" implied TD_W = 1.6 vs the 4.0 seed; at
# k_opaque = 7 that same measurement gives TD_W = 3.75, consistent with the seed.)
DEFAULT_K_OPAQUE = 7.0

# sRGB (D65) -> XYZ, IEC 61966-2-1
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])


# --------------------------------------------------------------------------- #
# Transfer functions
# --------------------------------------------------------------------------- #
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Piecewise sRGB decoding, 0..1 float in -> 0..1 float out."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Piecewise sRGB encoding, 0..1 float in -> 0..1 float out (clipped)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def hex_to_rgb01(hex_str: str) -> np.ndarray:
    """'#RRGGBB' -> sRGB-encoded floats (3,) in 0..1."""
    h = hex_str.strip().lstrip('#')
    if len(h) != 6:
        raise ValueError(f"bad hex colour {hex_str!r}")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


def rgb01_to_hex(rgb01: np.ndarray) -> str:
    v = np.clip(np.round(np.asarray(rgb01, dtype=np.float64) * 255.0), 0, 255).astype(int)
    return '#%02X%02X%02X' % tuple(int(c) for c in v)


def luminance_y(rgb_lin: np.ndarray) -> np.ndarray:
    """Relative luminance Y from linear sRGB (last axis = 3)."""
    return np.asarray(rgb_lin, dtype=np.float64) @ _M_RGB2XYZ[1]


# --------------------------------------------------------------------------- #
# CIELAB
# --------------------------------------------------------------------------- #
def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)


def linear_rgb_to_lab_d65(rgb_lin: np.ndarray) -> np.ndarray:
    """Linear sRGB (..., 3) -> true CIELAB D65 (..., 3)."""
    rgb_lin = np.asarray(rgb_lin, dtype=np.float64)
    xyz = rgb_lin @ _M_RGB2XYZ.T
    f = _lab_f(xyz / _WHITE_D65)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def srgb_to_lab_d65(rgb01: np.ndarray) -> np.ndarray:
    """sRGB-encoded floats (N,3) 0..1 -> true CIELAB (N,3), white = (100,0,0)."""
    return linear_rgb_to_lab_d65(srgb_to_linear(rgb01))


def lab_to_linear_rgb_d65(lab: np.ndarray) -> np.ndarray:
    """Inverse of :func:`linear_rgb_to_lab_d65` (unclipped)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    delta = 6.0 / 29.0

    def finv(f):
        return np.where(f > delta, f ** 3, 3 * delta ** 2 * (f - 4.0 / 29.0))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * _WHITE_D65
    return xyz @ np.linalg.inv(_M_RGB2XYZ).T


# --------------------------------------------------------------------------- #
# Filaments
# --------------------------------------------------------------------------- #
@dataclass
class Filament:
    """One spool: display colour, transmission distance and Bambu preset ids."""
    name: str
    hex: str
    rgb_lin: np.ndarray
    td_mm: float
    td_source: str = 'guess'          # 'guess' | 'hueforge_lib' | 'fitted' | 'measured'
    settings_id: str = DEFAULT_SETTINGS_ID
    filament_id: str = DEFAULT_FILAMENT_ID

    def __post_init__(self):
        self.hex = '#' + self.hex.strip().lstrip('#').upper()
        self.rgb_lin = np.asarray(self.rgb_lin, dtype=np.float64).reshape(3)
        self.td_mm = float(self.td_mm)
        if self.td_mm <= 0:
            raise ValueError(f"{self.name}: td_mm must be > 0")

    @classmethod
    def from_hex(cls, name: str, hex_str: str, td_mm: float, td_source: str = 'guess',
                 settings_id: str = DEFAULT_SETTINGS_ID,
                 filament_id: str = DEFAULT_FILAMENT_ID) -> 'Filament':
        return cls(name=name, hex=hex_str, rgb_lin=srgb_to_linear(hex_to_rgb01(hex_str)),
                   td_mm=td_mm, td_source=td_source, settings_id=settings_id,
                   filament_id=filament_id)

    @property
    def luminance(self) -> float:
        return float(luminance_y(self.rgb_lin))

    @property
    def rgb8(self) -> tuple[int, int, int]:
        v = np.clip(np.round(linear_to_srgb(self.rgb_lin) * 255), 0, 255).astype(int)
        return int(v[0]), int(v[1]), int(v[2])

    def to_dict(self) -> dict:
        return {'name': self.name, 'hex': self.hex, 'td_mm': self.td_mm,
                'td_source': self.td_source, 'settings_id': self.settings_id,
                'filament_id': self.filament_id}


def load_filament_library(path: str) -> dict[str, Filament]:
    """Load a JSON list of {name, hex, td_mm, td_source[, settings_id, filament_id]}.

    Returns an insertion-ordered dict keyed by filament name.
    """
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if isinstance(data, dict) and 'filaments' in data:
        data = data['filaments']
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of filaments")
    lib: dict[str, Filament] = {}
    for entry in data:
        f = Filament.from_hex(
            name=str(entry['name']),
            hex_str=str(entry['hex']),
            td_mm=float(entry['td_mm']),
            td_source=str(entry.get('td_source', 'guess')),
            settings_id=str(entry.get('settings_id', DEFAULT_SETTINGS_ID)),
            filament_id=str(entry.get('filament_id', DEFAULT_FILAMENT_ID)),
        )
        if f.name in lib:
            raise ValueError(f"{path}: duplicate filament name {f.name!r}")
        lib[f.name] = f
    return lib


def save_filament_library(path: str, library: dict[str, Filament]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump([f.to_dict() for f in library.values()], fh, indent=2, ensure_ascii=False)
        fh.write('\n')
    return path


# --------------------------------------------------------------------------- #
# Optical models
# --------------------------------------------------------------------------- #
class OpticalModel(Protocol):
    name: str

    def ramp(self, schedule: 'BandSchedule', filaments: Sequence[Filament]) -> np.ndarray:
        """Return linear-RGB colours (K',3) for k in schedule.k_rows()."""
        ...


def composite_bands(base_rgb_lin: np.ndarray, band_rgb_lin: np.ndarray,
                    opacity: np.ndarray) -> np.ndarray:
    """Bottom-up compositing (rebuild3.comp recurrence).

    base_rgb_lin: (..., 3) colour of the opaque base.
    band_rgb_lin: (..., B1, 3) colours of bands 1..B-1 (bottom -> top).
    opacity:      (..., B1)    opacity of each band at its local thickness.
    Returns (..., 3) linear RGB.
    """
    base_rgb_lin = np.asarray(base_rgb_lin, dtype=np.float64)
    band_rgb_lin = np.asarray(band_rgb_lin, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    out = np.broadcast_to(base_rgb_lin, np.broadcast_shapes(
        base_rgb_lin.shape, opacity.shape[:-1] + (3,), band_rgb_lin.shape[:-2] + (3,))).copy()
    n_bands = opacity.shape[-1]
    for b in range(n_bands):
        o = opacity[..., b][..., None]
        out = out * (1.0 - o) + band_rgb_lin[..., b, :] * o
    return out


class _AnalyticModel:
    """Shared machinery for models defined by an opacity function o(t, TD)."""
    name = 'analytic'

    def opacity(self, thickness_mm: np.ndarray, td_mm: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def ramp(self, schedule: 'BandSchedule', filaments: Sequence[Filament]) -> np.ndarray:
        filaments = list(filaments)
        if len(filaments) != schedule.n_bands:
            raise ValueError(f"schedule has {schedule.n_bands} bands, got {len(filaments)} filaments")
        stacks = schedule.stacks()[:, :schedule.n_bands]            # (K', B)
        base = filaments[0].rgb_lin
        if schedule.n_bands == 1:
            return np.broadcast_to(base, (stacks.shape[0], 3)).copy()
        band_rgb = np.stack([f.rgb_lin for f in filaments[1:]], axis=0)   # (B-1, 3)
        td = np.array([f.td_mm for f in filaments[1:]], dtype=np.float64)  # (B-1,)
        t = stacks[:, 1:].astype(np.float64) * schedule.layer_h              # (K', B-1)
        o = self.opacity(t, td[None, :])
        return composite_bands(base, band_rgb[None, :, :], o)


class BeerLambertModel(_AnalyticModel):
    """o(t) = 1 - exp(-k_opaque * t / TD).

    Default k_opaque = DEFAULT_K_OPAQUE (7 -> 99.9% opaque at t = TD, the
    HueForge meaning of TD).  k_opaque = 3 gives 95% at TD.
    """
    name = 'beer-lambert'

    def __init__(self, k_opaque: float | None = None):
        self.k_opaque = float(DEFAULT_K_OPAQUE if k_opaque is None else k_opaque)
        if self.k_opaque <= 0:
            raise ValueError("k_opaque must be > 0")

    def opacity(self, thickness_mm: np.ndarray, td_mm: np.ndarray) -> np.ndarray:
        t = np.asarray(thickness_mm, dtype=np.float64)
        td = np.asarray(td_mm, dtype=np.float64)
        return 1.0 - np.exp(-self.k_opaque * t / td)

    def __repr__(self):
        return f"BeerLambertModel(k_opaque={self.k_opaque})"


class LinearTDModel(_AnalyticModel):
    """o(t) = min(1, t / TD) - the HueForge-style linear ramp."""
    name = 'linear-td'

    def opacity(self, thickness_mm: np.ndarray, td_mm: np.ndarray) -> np.ndarray:
        t = np.asarray(thickness_mm, dtype=np.float64)
        td = np.asarray(td_mm, dtype=np.float64)
        return np.clip(t / td, 0.0, 1.0)

    def __repr__(self):
        return "LinearTDModel()"


class MeasuredRampModel:
    """Measured ramp colours (uint8 sRGB) for one exact schedule (calibration board)."""
    name = 'measured'

    def __init__(self, rgb_u8: np.ndarray, stacks: np.ndarray | None = None,
                 source: str | None = None):
        self.rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8).reshape(-1, 3)
        self.stacks = None if stacks is None else np.asarray(stacks, dtype=np.int32)
        self.source = source

    @classmethod
    def from_npz(cls, path: str) -> 'MeasuredRampModel':
        data = np.load(path)
        rgb = data['rgb']
        stacks = data['stacks'] if 'stacks' in data.files else None
        return cls(rgb, stacks, source=path)

    def ramp(self, schedule: 'BandSchedule', filaments: Sequence[Filament]) -> np.ndarray:
        expected = schedule.stacks()
        if self.rgb_u8.shape[0] != expected.shape[0]:
            raise ValueError(
                f"measured ramp has {self.rgb_u8.shape[0]} rows, schedule needs {expected.shape[0]}")
        if self.stacks is not None:
            w = min(self.stacks.shape[1], expected.shape[1])
            if not np.array_equal(self.stacks[:, :w], expected[:, :w]):
                raise ValueError("measured ramp stacks do not match the schedule")
        return srgb_to_linear(self.rgb_u8.astype(np.float64) / 255.0)

    def __repr__(self):
        return f"MeasuredRampModel(rows={self.rgb_u8.shape[0]}, source={self.source!r})"


def ramp_to_uint8(rgb_lin: np.ndarray) -> np.ndarray:
    """Linear RGB (...,3) -> sRGB uint8."""
    return np.clip(np.round(linear_to_srgb(rgb_lin) * 255.0), 0, 255).astype(np.uint8)
