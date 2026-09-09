"""Ramp LUT: the Band-mode colour table in Lumina's .npz {rgb, stacks} format.

Row i corresponds to a column of k = n0 + i layers (k in schedule.k_rows()).
``stacks`` rows are per-band layer counts, so LuminaImageProcessor's
material_matrix.sum(-1) recovers k for every pixel.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from core.band.optics import Filament, OpticalModel, ramp_to_uint8
from core.band.schedule import BandSchedule

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUSTOM_LUT_DIR = os.path.join(_REPO, 'lut-npy预设', 'Custom')


@dataclass
class RampLUT:
    rgb: np.ndarray            # (K',3) uint8 sRGB
    stacks: np.ndarray         # (K',5) int32
    k_values: np.ndarray       # (K',) int
    heights: np.ndarray        # (K',) float top z of a k-layer column
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.rgb = np.asarray(self.rgb, dtype=np.uint8).reshape(-1, 3)
        self.stacks = np.asarray(self.stacks, dtype=np.int32)
        self.k_values = np.asarray(self.k_values, dtype=np.int64).reshape(-1)
        self.heights = np.asarray(self.heights, dtype=np.float64).reshape(-1)
        n = self.rgb.shape[0]
        if self.stacks.shape[0] != n or self.k_values.shape[0] != n or self.heights.shape[0] != n:
            raise ValueError("RampLUT arrays must have the same number of rows")

    @property
    def n_rows(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def n0(self) -> int:
        return int(self.k_values[0])

    def save(self, path_npz: str) -> str:
        """np.savez(path, rgb, stacks[, k_values, heights]) + '<path>.json' sidecar."""
        if not path_npz.endswith('.npz'):
            path_npz = path_npz + '.npz'
        os.makedirs(os.path.dirname(os.path.abspath(path_npz)), exist_ok=True)
        np.savez(path_npz, rgb=self.rgb, stacks=self.stacks,
                 k_values=self.k_values, heights=self.heights)
        side = dict(self.meta)
        side.setdefault('format', 'lumina-band-ramp-lut/1')
        side['k_values'] = [int(k) for k in self.k_values]
        side['heights'] = [round(float(h), 6) for h in self.heights]
        side['rgb_hex'] = ['#%02X%02X%02X' % tuple(int(c) for c in row) for row in self.rgb]
        with open(path_npz + '.json', 'w', encoding='utf-8') as fh:
            json.dump(side, fh, indent=2, ensure_ascii=False)
        return path_npz

    @classmethod
    def load(cls, path_npz: str) -> 'RampLUT':
        data = np.load(path_npz)
        rgb = data['rgb']
        stacks = data['stacks']
        meta: dict = {}
        side = path_npz + '.json'
        if os.path.exists(side):
            with open(side, 'r', encoding='utf-8') as fh:
                meta = json.load(fh)
        if 'k_values' in data.files:
            k_values = data['k_values']
        elif 'k_values' in meta:
            k_values = np.asarray(meta['k_values'])
        else:
            k_values = stacks.sum(axis=1)
        if 'heights' in data.files:
            heights = data['heights']
        elif 'heights' in meta:
            heights = np.asarray(meta['heights'], dtype=np.float64)
        else:
            sch = meta.get('schedule')
            if sch:
                heights = BandSchedule.from_dict(sch).top_z(np.asarray(k_values))
            else:
                heights = 0.16 + 0.08 * (np.asarray(k_values, dtype=np.float64) - 1)
        return cls(rgb=rgb, stacks=stacks, k_values=k_values, heights=heights, meta=meta)

    def schedule(self) -> BandSchedule | None:
        sch = self.meta.get('schedule')
        return BandSchedule.from_dict(sch) if sch else None


def build_ramp_lut(schedule: BandSchedule, filaments: Mapping[str, Filament],
                   model: OpticalModel) -> RampLUT:
    """Compose the ramp colours for ``schedule`` with ``model`` and pack them."""
    fils = [filaments[n] for n in schedule.filament_names]
    rgb_lin = np.asarray(model.ramp(schedule, fils), dtype=np.float64)
    k_values = np.arange(schedule.n0, schedule.n_layers + 1)
    if rgb_lin.shape != (k_values.size, 3):
        raise ValueError(f"model.ramp returned {rgb_lin.shape}, expected {(k_values.size, 3)}")
    meta = {
        'schedule': schedule.to_dict(),
        'filaments': [{'name': f.name, 'hex': f.hex, 'td_mm': f.td_mm,
                       'td_source': f.td_source, 'settings_id': f.settings_id,
                       'filament_id': f.filament_id} for f in fils],
        'model': getattr(model, 'name', type(model).__name__),
        'model_repr': repr(model),
    }
    return RampLUT(rgb=ramp_to_uint8(rgb_lin), stacks=schedule.stacks(),
                   k_values=k_values, heights=schedule.heights(), meta=meta)


def mode_key_for(schedule: BandSchedule) -> str:
    return f"Band:{schedule.key()}"


def model_tag(model) -> str:
    """Short filename-safe tag identifying the optical model *and* its constants,
    so ramp files for different k_opaque values never overwrite each other
    (schedule.key() hashes names/counts/FL/lh only)."""
    if getattr(model, 'name', '') == 'measured':
        return 'measured'
    k = getattr(model, 'k_opaque', None)
    if k is not None:
        return f"bl{float(k):g}".replace('.', 'p')
    name = str(getattr(model, 'name', type(model).__name__))
    return ''.join(ch for ch in name.lower() if ch.isalnum()) or 'model'


def ramp_filename(schedule: BandSchedule, model=None) -> str:
    """Band_<schedule key>_<model tag>.npz (``Band_<key>_measured.npz`` for measured ramps)."""
    tag = model_tag(model) if model is not None else 'measured'
    return f"Band_{schedule.key()}_{tag}.npz"


def default_ramp_path(schedule: BandSchedule, measured: bool = False, model=None,
                      directory: str | None = None) -> str:
    """Default ramp location.  Measured ramps live in lut-npy预设/Custom (the
    calibration step writes them there); synthetic ramps default to the same
    directory unless ``directory`` is given (the pipeline passes its output
    directory so the legacy LUT manager does not list them as 'Merged' LUTs)."""
    base = directory or CUSTOM_LUT_DIR
    if measured:
        return os.path.join(base, f"Band_{schedule.key()}_measured.npz")
    return os.path.join(base, ramp_filename(schedule, model))


def register_color_mode(schedule: BandSchedule, filaments: Mapping[str, Filament]) -> str:
    """Register a dynamic 5-slot ColorSystem for this schedule; returns the mode key.

    Uses config.ColorSystem.register_dynamic when available (added by the
    config.py builder); otherwise returns the key without registering.
    """
    mode_key = mode_key_for(schedule)
    names = list(schedule.filament_names) + [''] * (5 - schedule.n_bands)
    preview = {}
    for i, n in enumerate(names):
        if n:
            r, g, b = filaments[n].rgb8
            preview[i] = [r, g, b, 255]
        else:
            preview[i] = [128, 128, 128, 255]
    conf = {
        'name': 'Band',
        'slots': names,
        'preview': preview,
        'map': {n: i for i, n in enumerate(names) if n},
        'layer_count': 5,
        'schedule': schedule.to_dict(),
    }
    try:
        from config import ColorSystem  # type: ignore
        reg = getattr(ColorSystem, 'register_dynamic', None)
        if callable(reg):
            reg(mode_key, conf)
    except Exception:
        pass
    return mode_key
