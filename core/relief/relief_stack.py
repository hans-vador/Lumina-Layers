"""Face-up Relief Stack5 voxel construction.

Flat Stack5 (core/stack5) prints FACE DOWN: voxel z = 0 is the viewing surface
``stack[0]`` and the backing is on top.  A relief has to print FACE UP, so the
physical Z order is reversed here while the recipe itself is untouched:

    z (layers, bottom = print bed)
    0 .. base_layers-1                        flat base           (backing filament)
    base_layers .. surface-1                  relief body/support (backing filament)
    surface + 0                               stack[4]  (deepest optical layer)
    surface + 1                               stack[3]
    surface + 2                               stack[2]
    surface + 3                               stack[1]
    surface + 4                               stack[0]  (visible top)          <- viewer

with ``surface[y, x] = base_layers + relief_steps[y, x]``.  Every solid pixel
therefore gets exactly ``color_layers`` optical layers of 1 layer height each,
never stretched with depth, and everything below them is solid backing.  The
backing filament is the palette slot the LUT was synthesised over (Stack5's
``backing_slot``) - the colour of every stack depends on it, so it cannot be a
different material.

Thickness budget (defaults): base 0.8 mm = 10 layers, relief 0..4.0 mm = 0..50
layers, shell 5 x 0.08 = 0.40 mm, total 1.2 .. 5.2 mm = 15 .. 65 layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_BASE_MM = 0.8
DEFAULT_RELIEF_MM = 4.0
DEFAULT_LAYER_H = 0.08
DEFAULT_COLOR_LAYERS = 5
AIR = -1


def _layers(mm: float, layer_h: float) -> int:
    return int(np.floor(float(mm) / float(layer_h) + 0.5))


@dataclass(frozen=True)
class ReliefDims:
    """Vertical budget of a relief plaque.  All heights are whole layers."""
    base_mm: float = DEFAULT_BASE_MM
    relief_mm: float = DEFAULT_RELIEF_MM
    layer_h: float = DEFAULT_LAYER_H
    color_layers: int = DEFAULT_COLOR_LAYERS

    def __post_init__(self):
        if not (0 < float(self.layer_h) <= 1.0):
            raise ValueError(f"layer_h must be within (0, 1] mm, got {self.layer_h}")
        if float(self.base_mm) < 0 or float(self.relief_mm) < 0:
            raise ValueError("base_mm and relief_mm must be >= 0")
        if int(self.color_layers) < 1:
            raise ValueError("color_layers must be >= 1")
        if self.base_layers < 1:
            raise ValueError(f"base_mm {self.base_mm} is thinner than one layer ({self.layer_h}): the colour "
                             "shell needs backing filament under it for the LUT colours to hold")
        if self.total_max_layers > 30000:
            raise ValueError(f"{self.total_max_layers} layers exceed the int16 step budget; use a coarser layer_h "
                             "or a smaller relief_mm / base_mm")

    @property
    def base_layers(self) -> int:
        return _layers(self.base_mm, self.layer_h)

    @property
    def relief_layers(self) -> int:
        """Maximum relief above the base, in layers (50 by default)."""
        return _layers(self.relief_mm, self.layer_h)

    @property
    def shell_layers(self) -> int:
        return int(self.color_layers)

    @property
    def total_max_layers(self) -> int:
        return self.base_layers + self.relief_layers + self.shell_layers

    @property
    def total_min_layers(self) -> int:
        return self.base_layers + self.shell_layers

    def mm(self, layers: int) -> float:
        return float(layers) * float(self.layer_h)

    def summary(self) -> dict:
        lh = float(self.layer_h)
        return {
            'layer_h_mm': lh,
            'base_layers': self.base_layers, 'base_mm': round(self.base_layers * lh, 6),
            'base_mm_requested': float(self.base_mm),
            'relief_layers_max': self.relief_layers, 'relief_mm_max': round(self.relief_layers * lh, 6),
            'relief_mm_requested': float(self.relief_mm),
            'shell_layers': self.shell_layers, 'shell_mm': round(self.shell_layers * lh, 6),
            'total_layers_min': self.total_min_layers, 'total_mm_min': round(self.total_min_layers * lh, 6),
            'total_layers_max': self.total_max_layers, 'total_mm_max': round(self.total_max_layers * lh, 6),
            'rounding_warnings': self.rounding_warnings(),
        }

    def rounding_warnings(self) -> list[str]:
        out = []
        lh = float(self.layer_h)
        for label, mm, n in (('base_mm', self.base_mm, self.base_layers),
                             ('relief_mm', self.relief_mm, self.relief_layers)):
            if abs(float(mm) - n * lh) > 1e-6:
                out.append(f"{label} {float(mm):g} is not a multiple of the layer height {lh:g}; "
                           f"using {n} layers = {n * lh:g} mm")
        return out


def build_relief_voxels(material_matrix: np.ndarray, mask_solid: np.ndarray, relief_steps: np.ndarray,
                        dims: ReliefDims, backing_slot: int, trim: bool = True,
                        n_slots: Optional[int] = None) -> tuple[np.ndarray, dict]:
    """Face-up voxel matrix (Z, H, W) of palette slot ids (-1 = air).

    material_matrix: (H, W, L) Stack5 recipe, ``[..., 0]`` = viewing surface.
    mask_solid:      (H, W) bool, printed pixels.
    relief_steps:    (H, W) int relief layers 0..dims.relief_layers.
    trim:            Z = base + max(present steps) + L (True) or the full budget.
    n_slots:         palette size (validates slot ids); inferred from the data when None.
    """
    mm = np.asarray(material_matrix)
    mask = np.asarray(mask_solid).astype(bool)
    steps = np.asarray(relief_steps)
    if mm.ndim != 3:
        raise ValueError(f"material_matrix must be (H, W, L), got {mm.shape}")
    H, W, L = mm.shape
    if mask.shape != (H, W) or steps.shape != (H, W):
        raise ValueError("mask_solid and relief_steps must be (H, W) like the material matrix")
    if L != dims.shell_layers:
        raise ValueError(f"material matrix has {L} optical layers but dims.color_layers = {dims.shell_layers}")
    if not np.issubdtype(steps.dtype, np.integer):
        if not np.allclose(steps, np.round(steps)):
            raise ValueError("relief_steps must be whole layers")
        steps = np.round(steps).astype(np.int64)
    steps = steps.astype(np.int64)
    if mask.any():
        smin, smax = int(steps[mask].min()), int(steps[mask].max())
        if smin < 0 or smax > dims.relief_layers:
            raise ValueError(f"relief_steps outside 0..{dims.relief_layers}: {smin}..{smax}")
        stacks = mm[mask]
        if stacks.min() < 0:
            raise ValueError("material matrix has air (-1) inside mask_solid")
        used_max = int(stacks.max()) + 1
    else:
        smax = 0
        used_max = 1
    if n_slots is None:
        n_slots = max(used_max, int(backing_slot) + 1)
    n_slots = int(n_slots)
    if used_max > n_slots:
        raise ValueError(f"material matrix uses slot {used_max - 1} but the palette has {n_slots} slots")
    if not (0 <= int(backing_slot) < n_slots) or n_slots > 120:
        raise ValueError(f"backing_slot {backing_slot} invalid for {n_slots} slots")

    base = dims.base_layers
    Z = base + (smax if trim else dims.relief_layers) + L
    vox = np.full((Z, H, W), AIR, dtype=np.int8)
    surface = base + steps                                   # first optical layer index per pixel
    # backing: base + relief body, every z below the surface of a solid pixel
    zz = np.arange(Z, dtype=np.int64)[:, None, None]
    below = (zz < surface[None, :, :]) & mask[None, :, :]
    vox[below] = np.int8(backing_slot)
    # optical shell, reversed: stack[L-1] at the surface, stack[0] on top
    yy, xx = np.nonzero(mask)
    surf = surface[yy, xx]
    for i in range(L):
        vox[surf + i, yy, xx] = mm[yy, xx, L - 1 - i].astype(np.int8)

    per_layer_sets = printed_layer_sets(vox)
    meta = {
        'shape_zhw': [int(Z), int(H), int(W)],
        'base_layers': int(base),
        'relief_layers_present_max': int(smax),
        'relief_layers_budget': int(dims.relief_layers),
        'shell_layers': int(L),
        'total_layers': int(Z),
        'total_mm': round(Z * float(dims.layer_h), 6),
        'backing_slot': int(backing_slot),
        'backing_voxels': int(np.count_nonzero(below)),
        'shell_voxels': int(len(yy) * L),
        'solid_pixels': int(len(yy)),
        'materials_per_printed_layer': [len(s) for s in per_layer_sets],
        'printed_layer_material_sets': per_layer_sets,
        'convention': 'face-up: z=0 on the bed; stack[0] is the topmost voxel of every column',
    }
    return vox, meta


def printed_layer_sets(vox: np.ndarray) -> list[list[int]]:
    """Palette slot ids present in every printed layer (bottom -> top)."""
    out = []
    for z in range(vox.shape[0]):
        ids = np.unique(vox[z])
        out.append([int(i) for i in ids if i >= 0])
    return out


def check_relief_voxels(vox: np.ndarray, material_matrix: np.ndarray, mask_solid: np.ndarray,
                        relief_steps: np.ndarray, dims: ReliefDims, backing_slot: int) -> dict:
    """Exhaustive structural check of a face-up voxel matrix.  Returns a dict of
    booleans (all True when the structure is sound) plus counts; raises nothing.
    Used by the tests and by ``--verify``."""
    mm = np.asarray(material_matrix)
    mask = np.asarray(mask_solid).astype(bool)
    steps = np.asarray(relief_steps).astype(np.int64)
    Z, H, W = vox.shape
    L = mm.shape[2]
    base = dims.base_layers
    surface = base + steps
    zz = np.arange(Z)[:, None, None]
    solid = vox >= 0
    report: dict = {}
    # columns outside the mask are air
    report['air_outside_mask'] = bool(not solid[:, ~mask].any()) if (~mask).any() else True
    # inside the mask: solid from 0 up to surface+L-1 and air above
    expect_solid = (zz < (surface + L)[None]) & mask[None]
    report['columns_filled_exactly_to_shell_top'] = bool(np.array_equal(solid, expect_solid))
    # backing below the surface
    below = (zz < surface[None]) & mask[None]
    report['backing_fills_below_shell'] = bool(np.all(vox[below] == backing_slot)) if below.any() else True
    # shell order: z = surface + i holds stack[L-1-i]
    yy, xx = np.nonzero(mask)
    ok_shell = True
    for i in range(L):
        got = vox[surface[yy, xx] + i, yy, xx]
        if not np.array_equal(got, mm[yy, xx, L - 1 - i]):
            ok_shell = False
            break
    report['shell_order_stack4_bottom_stack0_top'] = bool(ok_shell)
    # exactly L colour voxels per solid pixel = solid count minus backing count
    n_solid_per_col = solid.sum(axis=0)
    n_back_per_col = (vox == backing_slot).sum(axis=0) if mask.any() else np.zeros((H, W), int)
    # (a shell layer may itself be the backing slot; count by position instead)
    n_colour = np.zeros((H, W), dtype=np.int64)
    n_colour[mask] = L
    report['five_colour_layers_per_pixel'] = bool(np.all((n_solid_per_col[mask] - surface[mask]) == L))
    report['top_voxel_is_stack0'] = bool(np.array_equal(vox[surface[yy, xx] + L - 1, yy, xx], mm[yy, xx, 0]))
    report['lowest_shell_voxel_is_stack_last'] = bool(np.array_equal(vox[surface[yy, xx], yy, xx], mm[yy, xx, L - 1]))
    report['flat_bottom_all_backing'] = bool(np.all(vox[0][mask] == backing_slot)) if mask.any() else True
    report['base_layers'] = int(base)
    report['total_layers'] = int(Z)
    report['total_mm'] = round(Z * dims.layer_h, 6)
    report['max_total_layers_budget'] = int(dims.total_max_layers)
    report['within_budget'] = bool(Z <= dims.total_max_layers)
    st = np.asarray(relief_steps)
    report['all_heights_whole_layers'] = bool(np.issubdtype(st.dtype, np.integer) or np.allclose(st, np.round(st)))
    report['total_height_is_whole_layers'] = bool(abs(Z * dims.layer_h / dims.layer_h - round(Z * dims.layer_h / dims.layer_h)) < 1e-9)
    report['ok'] = all(v for k, v in report.items() if isinstance(v, bool))
    del n_back_per_col, n_colour
    return report
