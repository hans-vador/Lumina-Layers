"""Synthetic Stack5 LUT: every 5-layer stack of 5 filaments, composited face-down.

Lumina conventions (core/image_processing.py, core/converter.py):
* ``stacks[i, 0]`` is the VIEWING surface (printed first, at Z=0, face-down on the
  bed); ``stacks[i, L-1]`` is the layer next to the backing.
* The LUT file is ``.npz`` with ``rgb`` (N,3) uint8 sRGB and ``stacks`` (N,L) int32
  material slot ids; ``LuminaImageProcessor`` loads it directly, takes
  ``layer_count = stacks.shape[1]`` and builds a KDTree in OpenCV Lab.
* Row order: row i = base-n digits of i, most significant digit first, so row 1 is
  ``[0,0,0,0,1]`` - the same rule Lumina uses for its 4-colour .npy LUTs.

Compositing (linear light): light enters at ``stacks[:, 0]``; the colour is the
fold from the opaque backing filament upward through ``stacks[:, L-1] ...
stacks[:, 0]`` with per-layer opacity ``o = model.opacity(layer_h, td)``:
``out = out * (1 - o) + c * o``.  For a Beer-Lambert model this equals the
band-mode result for n equal layers (``(1-o)^n = 1 - o(n*t)``).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Sequence

import numpy as np

from core.band.optics import (BeerLambertModel, Filament, linear_to_srgb, luminance_y,
                              srgb_to_linear)

LAYER_H = 0.08
N_SLOTS = 5
N_LAYERS = 5
MODE_PREFIX = 'Stack5:'
MODE_NAME = 'Stack5'
# Substrings that ColorSystem.get / converter / naming test for; a Stack5 mode key
# must never contain one of them (checked in register_stack5_mode).
FORBIDDEN_KEY_SUBSTRINGS = ('CMYW', 'RYBW', '4-Color', '6-Color', '8-Color', 'BW',
                            'Merged', 'Band', '5-Color')


# --------------------------------------------------------------------------- #
# Stacks
# --------------------------------------------------------------------------- #
def enumerate_stacks(n_slots: int = N_SLOTS, layers: int = N_LAYERS) -> np.ndarray:
    """All n^L stacks, shape (n^L, L) int32; column 0 = viewing surface.

    Row i holds the base-n digits of i, most significant first (row 1 = [0,..,0,1]).
    """
    n, L = int(n_slots), int(layers)
    if n < 1 or L < 1:
        raise ValueError("n_slots and layers must be >= 1")
    idx = np.arange(n ** L, dtype=np.int64)
    out = np.empty((n ** L, L), dtype=np.int32)
    for j in range(L - 1, -1, -1):
        out[:, j] = idx % n
        idx //= n
    return out


def pure_stack_indices(n_slots: int = N_SLOTS, layers: int = N_LAYERS) -> np.ndarray:
    """Row index of the pure stack [i]*L for every slot i: i * (n^L - 1) / (n - 1)."""
    n, L = int(n_slots), int(layers)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    return np.arange(n, dtype=np.int64) * ((n ** L - 1) // (n - 1))


def is_pure(stacks: np.ndarray) -> np.ndarray:
    """(N,) bool: every layer of the stack is the same slot."""
    s = np.asarray(stacks)
    return np.all(s == s[:, :1], axis=1)


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #
def _opacity_fn(model):
    fn = getattr(model, 'opacity', None)
    if not callable(fn):
        raise TypeError(f"need an analytic optical model with .opacity(t, td), got {type(model).__name__}")
    return fn


def layer_thicknesses(first_layer_mm: float | None, layer_h: float = LAYER_H,
                      layers: int = N_LAYERS) -> np.ndarray:
    """Per-layer thickness vector, col 0 = viewing surface (the slicer's first layer).

    Face-down, the viewing layer is printed first, so a thicker slicer first
    layer makes the *top* colour layer thicker.  None -> uniform ``layer_h``.
    """
    t = np.full(int(layers), float(layer_h), dtype=np.float64)
    if first_layer_mm is not None:
        t[0] = float(first_layer_mm)
    return t


def _thickness_vector(layer_h, L: int) -> np.ndarray:
    """Accept a scalar or an (L,) sequence of per-layer thicknesses."""
    t = np.asarray(layer_h, dtype=np.float64)
    if t.ndim == 0:
        return np.full(L, float(t))
    if t.shape != (L,):
        raise ValueError(f"layer_h must be a scalar or shape ({L},), got {t.shape}")
    return t


def synth_lut_linear(rgb_lin: np.ndarray, td: np.ndarray, backing_idx: np.ndarray, model,
                     stacks: np.ndarray, layer_h=LAYER_H) -> np.ndarray:
    """Vectorised face-down compositing.

    rgb_lin (S,n,3) linear filament colours, td (S,n) mm, backing_idx (S,) slot of
    the opaque backing, stacks (N,L) slot ids (col 0 = viewing surface) ->
    linear RGB (S,N,3).  ``layer_h`` is a scalar or an (L,) per-layer thickness
    vector (see :func:`layer_thicknesses`) so a thick first layer is modelled.
    """
    opacity = _opacity_fn(model)
    rgb_lin = np.asarray(rgb_lin, dtype=np.float64)
    td = np.asarray(td, dtype=np.float64)
    backing_idx = np.asarray(backing_idx, dtype=np.int64).reshape(-1)
    stacks = np.asarray(stacks, dtype=np.int64)
    S, n = rgb_lin.shape[0], rgb_lin.shape[1]
    N, L = stacks.shape
    if stacks.min() < 0 or stacks.max() >= n:
        raise ValueError("stack slot ids out of range for the filament set")
    thick = _thickness_vector(layer_h, L)                                          # (L,)
    # opacity of every filament at every layer's thickness: (L,S,n)
    op = np.stack([np.clip(opacity(np.full_like(td, float(t)), td), 0.0, 1.0) for t in thick])
    base = rgb_lin[np.arange(S), backing_idx]                                      # (S,3)
    out = np.broadcast_to(base[:, None, :], (S, N, 3)).copy()
    for j in range(L - 1, -1, -1):
        slot = stacks[:, j]                                                        # (N,)
        o = op[j][:, slot][..., None]                                              # (S,N,1)
        c = rgb_lin[:, slot, :]                                                    # (S,N,3)
        out = out * (1.0 - o) + c * o
    return out


def synth_lut(filaments: Sequence[Filament], backing_slot: int, model=None,
              layers: int = N_LAYERS, layer_h: float = LAYER_H,
              first_layer_mm: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """LUT for ``filaments`` (slot order 0..n-1) over the opaque backing filament
    ``filaments[backing_slot]`` -> (rgb uint8 (N,3), stacks int32 (N,L)).

    N = n^layers; the n pure stacks [i]*L are rows ``pure_stack_indices()``.
    """
    fils = list(filaments)
    n = len(fils)
    if n < 1:
        raise ValueError("need at least one filament")
    if not (0 <= int(backing_slot) < n):
        raise ValueError(f"backing_slot {backing_slot} outside 0..{n - 1}")
    if model is None:
        model = BeerLambertModel()
    stacks = enumerate_stacks(n, layers)
    rgb_lin = np.array([f.rgb_lin for f in fils], dtype=np.float64)[None]          # (1,n,3)
    td = np.array([f.td_mm for f in fils], dtype=np.float64)[None]                  # (1,n)
    thick = layer_thicknesses(first_layer_mm, layer_h, layers)
    lin = synth_lut_linear(rgb_lin, td, np.array([int(backing_slot)]), model, stacks, thick)[0]
    rgb = np.clip(np.round(linear_to_srgb(lin) * 255.0), 0, 255).astype(np.uint8)
    return rgb, stacks


def choose_backing(filaments: Sequence[Filament], backing: str | None = None) -> int:
    """Slot index of the backing: ``backing`` by name if given, else 'White' when
    present, else the lightest (highest luminance) member."""
    fils = list(filaments)
    names = [f.name for f in fils]
    if backing:
        if backing not in names:
            raise KeyError(f"backing {backing!r} not in palette {names}")
        return names.index(backing)
    if 'White' in names:
        return names.index('White')
    return int(np.argmax([f.luminance for f in fils]))


# --------------------------------------------------------------------------- #
# Files / registration
# --------------------------------------------------------------------------- #
def save_lut_npz(path: str, rgb: np.ndarray, stacks: np.ndarray, meta: dict | None = None) -> str:
    """np.savez(path, rgb=uint8 (N,3), stacks=int32 (N,L)) - the layout
    ``core.lut_merger.LUTMerger.save_merged_lut`` writes and
    ``LuminaImageProcessor._load_lut`` reads - plus a ``<path>.json`` sidecar."""
    if not path.endswith('.npz'):
        path = path + '.npz'
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    stacks = np.asarray(stacks, dtype=np.int32)
    if rgb.shape[0] != stacks.shape[0]:
        raise ValueError("rgb and stacks must have the same number of rows")
    np.savez(path, rgb=rgb, stacks=stacks)
    side = dict(meta or {})
    side.setdefault('format', 'lumina-stack5-lut/1')
    side['n_colors'] = int(rgb.shape[0])
    side['layer_count'] = int(stacks.shape[1])
    side['pure_stack_rows'] = [int(i) for i in pure_stack_indices(int(stacks.max()) + 1, stacks.shape[1])]
    side['pure_stack_hex'] = ['#%02X%02X%02X' % tuple(int(c) for c in rgb[i]) for i in side['pure_stack_rows']]
    with open(path + '.json', 'w', encoding='utf-8') as fh:
        json.dump(side, fh, indent=2, ensure_ascii=False)
    return path


def lut_meta(filaments: Sequence[Filament], backing_slot: int, model, layers: int = N_LAYERS,
             layer_h: float = LAYER_H, first_layer_mm: float | None = None) -> dict:
    fils = list(filaments)
    thick = layer_thicknesses(first_layer_mm, layer_h, layers)
    return {
        'first_layer_mm': float(thick[0]),
        'layer_thicknesses_mm': [float(t) for t in thick],
        'palette': [{'slot': i, 'name': f.name, 'hex': f.hex, 'td_mm': f.td_mm,
                     'td_source': f.td_source} for i, f in enumerate(fils)],
        'backing': {'slot': int(backing_slot), 'name': fils[int(backing_slot)].name},
        'model': getattr(model, 'name', type(model).__name__),
        'model_repr': repr(model),
        'layers': int(layers),
        'layer_h': float(layer_h),
        'convention': 'stacks[:,0] = viewing surface; backing behind stacks[:,L-1]; print orientation is stored in the recipe',
    }


def stack5_key(filaments: Sequence[Filament]) -> str:
    """8-hex digest of (name, hex, td) in slot order."""
    h = hashlib.sha1()
    for f in filaments:
        h.update(f"{f.name}|{f.hex}|{f.td_mm:g};".encode('utf-8'))
    return h.hexdigest()[:8]


def register_stack5_mode(filaments: Sequence[Filament], mode_key: str | None = None) -> str:
    """Register a dynamic ColorSystem for this palette; returns 'Stack5:<key>'.

    conf = {'name': 'Stack5', 'slots': names, 'preview': {i: [r,g,b,255]},
            'map': {name: i}, 'layer_count': 5}
    """
    from config import ColorSystem
    fils = list(filaments)
    names = [f.name for f in fils]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate filament names in palette: {names}")
    key = mode_key or f"{MODE_PREFIX}{stack5_key(fils)}"
    bad = [s for s in FORBIDDEN_KEY_SUBSTRINGS if s in key]
    if bad:
        raise ValueError(f"mode key {key!r} contains reserved substring(s) {bad}")
    preview = {}
    for i, f in enumerate(fils):
        r, g, b = f.rgb8
        preview[i] = [int(r), int(g), int(b), 255]
    conf = {
        'name': MODE_NAME,
        'slots': names,
        'preview': preview,
        'map': {n: i for i, n in enumerate(names)},
        'layer_count': N_LAYERS,
        'filament_hex': [f.hex for f in fils],
        'filament_td_mm': [f.td_mm for f in fils],
    }
    ColorSystem.register_dynamic(key, conf)
    got = ColorSystem.get(key)
    if got is not conf:
        raise RuntimeError(f"ColorSystem.get({key!r}) did not return the registered config")
    return key


def filament_from_hex(name: str, hex_str: str, td_mm: float) -> Filament:
    """Convenience for tests / ad-hoc palettes."""
    return Filament.from_hex(name, hex_str, td_mm)
