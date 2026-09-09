"""Independent verification of a written Relief Stack5 3MF.

The object model is parsed back and every part is rasterised layer by layer
(scan-line fill over its X-perpendicular faces at mid layer height, the same
technique the flat Stack5 tests use), then compared with the voxel matrix the
meshes were built from.  This proves, on the FILE that will be sliced:

* every voxel of every material is covered by exactly one part (no gaps, no
  overlapping material volumes),
* nothing is covered where the voxel matrix has air (no floating material),
* the five colour layers and the backing sit where the voxel builder put them,
* no geometry lies outside the voxel grid (misplaced or above-budget material).

Everything is accumulated per layer, so the peak memory is a few (H, W) arrays.
"""
from __future__ import annotations

import re
import zipfile
from typing import Optional, Sequence

import numpy as np

_VERTEX_RE = re.compile(rb'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"')
_TRI_RE = re.compile(rb'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"')
EPS_MM = 1e-6


def read_object_model(threemf_path: str, name: str = '3D/Objects/object_1.model') -> bytes:
    with zipfile.ZipFile(threemf_path) as zf:
        return zf.read(name)


def parse_object_meshes(object_model: bytes) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(vertices (V,3) float64, faces (F,3) int64)] in file order."""
    starts = [m.start() for m in re.finditer(rb'<object ', object_model)] + [len(object_model)]
    out = []
    for oi in range(len(starts) - 1):
        chunk = object_model[starts[oi]:starts[oi + 1]]
        V = np.array(_VERTEX_RE.findall(chunk), dtype=np.float64).reshape(-1, 3)
        T = np.array(_TRI_RE.findall(chunk), dtype=np.int64).reshape(-1, 3)
        out.append((V, T))
    return out


def geometry_bounds_check(meshes: Sequence[tuple[np.ndarray, np.ndarray]], n_layers: int, shape_hw: tuple[int, int],
                          pixel_mm: float, layer_h: float) -> dict:
    H, W = int(shape_hw[0]), int(shape_hw[1])
    xmax, ymax, zmax = W * pixel_mm, H * pixel_mm, n_layers * layer_h
    outside = 0
    for V, T in meshes:
        if len(V) == 0:
            continue
        bad = ((V[:, 0] < -EPS_MM) | (V[:, 0] > xmax + EPS_MM) | (V[:, 1] < -EPS_MM) | (V[:, 1] > ymax + EPS_MM)
               | (V[:, 2] < -EPS_MM) | (V[:, 2] > zmax + EPS_MM))
        outside += int(bad.sum())
    return {'vertices_outside_grid': outside, 'grid_mm': [xmax, ymax, zmax]}


def _layer_iter(meshes, n_layers, shape_hw, pixel_mm, layer_h):
    """Yield (layer index, (H, W) uint16 bitmask of the parts covering each pixel)
    in the MESH frame (row 0 = min y)."""
    H, W = int(shape_hw[0]), int(shape_hw[1])
    prepared = []
    for oi, (V, T) in enumerate(meshes):
        if len(T) == 0:
            prepared.append(None)
            continue
        P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        xperp = (P0[:, 0] == P1[:, 0]) & (P1[:, 0] == P2[:, 0])
        P0, P1, P2 = P0[xperp], P1[xperp], P2[xperp]
        zmin = np.minimum(np.minimum(P0[:, 2], P1[:, 2]), P2[:, 2])
        zmax = np.maximum(np.maximum(P0[:, 2], P1[:, 2]), P2[:, 2])
        nx = np.cross(P1 - P0, P2 - P0)[:, 0]
        ymin = np.minimum(np.minimum(P0[:, 1], P1[:, 1]), P2[:, 1])
        ymax = np.maximum(np.maximum(P0[:, 1], P1[:, 1]), P2[:, 1])
        prepared.append((np.rint(P0[:, 0] / pixel_mm).astype(np.int64), zmin, zmax,
                         np.rint(ymin / pixel_mm).astype(np.int64), np.rint(ymax / pixel_mm).astype(np.int64),
                         np.where(nx < 0, 1, -1).astype(np.int32)))
    for k in range(int(n_layers)):
        occ = np.zeros((H, W), np.uint16)
        zm = (k + 0.5) * float(layer_h)
        for oi, prep in enumerate(prepared):
            if prep is None:
                continue
            x, zmin, zmax, y0, y1, s = prep
            sel = (zmin < zm) & (zmax > zm)
            if not sel.any():
                continue
            E = np.zeros((H + 1, W + 1), np.int32)
            xs = np.clip(x[sel], 0, W)
            np.add.at(E, (np.clip(y0[sel], 0, H), xs), s[sel])
            np.add.at(E, (np.clip(y1[sel], 0, H), xs), -s[sel])
            filled = np.cumsum(np.cumsum(E, axis=0)[:H], axis=1)[:, :W] >= 1
            occ[filled] |= np.uint16(1 << oi)
        yield k, occ


def rasterize_meshes(meshes: Sequence[tuple[np.ndarray, np.ndarray]], n_layers: int, shape_hw: tuple[int, int],
                     pixel_mm: float, layer_h: float) -> np.ndarray:
    """(n_layers, H, W) uint16 bitmask: bit i set where part i covers the voxel
    (mesh frame: row 0 = min y).  Convenience for small models / tests."""
    H, W = int(shape_hw[0]), int(shape_hw[1])
    occ = np.zeros((int(n_layers), H, W), np.uint16)
    for k, layer in _layer_iter(meshes, n_layers, shape_hw, pixel_mm, layer_h):
        occ[k] = layer
    return occ


def verify_relief_3mf(threemf_path: str, vox: np.ndarray, part_slots: Sequence[int], pixel_mm: float,
                      layer_h: float, backing_slot: Optional[int] = None) -> dict:
    """Compare the parts of ``threemf_path`` (part i is palette slot part_slots[i])
    with the face-up voxel matrix ``vox`` (Z, H, W) - see the module docstring."""
    Z, H, W = vox.shape
    meshes = parse_object_meshes(read_object_model(threemf_path))
    if len(meshes) != len(part_slots):
        raise ValueError(f"3MF has {len(meshes)} parts but {len(part_slots)} slots were given")
    bounds = geometry_bounds_check(meshes, Z, (H, W), pixel_mm, layer_h)
    n_parts = len(part_slots)
    popcount = np.array([bin(v).count('1') for v in range(1 << n_parts)], dtype=np.uint8)
    slot_bits = {int(slot): np.uint16(1 << i) for i, slot in enumerate(part_slots)}
    overlap = gaps = floating = wrong = mismatch_total = 0
    layers_bad = 0
    bottom_ok = None
    for k, occ_mesh in _layer_iter(meshes, Z, (H, W), pixel_mm, layer_h):
        occ = occ_mesh[::-1, :]                                  # image row 0 is at max y in the mesh
        vz = vox[k]
        expected = np.zeros((H, W), np.uint16)
        for slot, bit in slot_bits.items():
            expected[vz == slot] |= bit
        counts = popcount[occ]
        solid = vz >= 0
        mism = occ != expected
        overlap += int(np.count_nonzero(counts > 1))
        gaps += int(np.count_nonzero(solid & (occ == 0)))
        floating += int(np.count_nonzero(~solid & (occ != 0)))
        wrong += int(np.count_nonzero(mism & solid & (occ != 0) & (counts == 1)))
        n_mis = int(mism.sum())
        mismatch_total += n_mis
        layers_bad += int(n_mis > 0)
        if k == 0 and backing_slot is not None and int(backing_slot) in slot_bits:
            bb = slot_bits[int(backing_slot)]
            bottom_ok = bool(np.all((occ == 0) | (occ == bb)))
    rep = {
        'parts': n_parts, 'layers': int(Z), 'voxels_solid': int(np.count_nonzero(vox >= 0)),
        'overlapping_voxels': overlap, 'uncovered_solid_voxels': gaps, 'covered_air_voxels': floating,
        'wrong_material_voxels': wrong, 'mismatched_voxels_total': mismatch_total,
        'layers_with_mismatch': layers_bad, 'vertices_outside_grid': bounds['vertices_outside_grid'],
        'ok': bool(mismatch_total == 0 and bounds['vertices_outside_grid'] == 0),
    }
    if bottom_ok is not None:
        rep['bottom_layer_all_backing'] = bottom_ok
    return rep
