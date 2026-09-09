"""Voxel matrix -> one closed box-union mesh per palette slot.

Reuses the flat Stack5 meshing pieces unchanged:

* :func:`core.stack5.pipeline._merge_layers_no_dilation` - consecutive Z layers
  with an identical material mask become one group (no 1-px dilation: parts
  must tile every layer exactly, or Bambu Studio would carve the overlaps);
* :meth:`core.mesh_generators.HighFidelityMesher._greedy_rect_merge` - maximal
  axis-aligned rectangles covering a layer group's mask.

On top of that, for a relief:

* rectangles that repeat in the next layer group are extended upward into one
  taller box (fewer triangles, no coincident interior faces between stacked
  shell / body boxes);
* optionally (``cancel_interior``) exactly coincident, oppositely wound triangle
  pairs are removed so hand-built boxes sharing a whole face fuse.  The greedy
  rectangles + vertical merge never produce such a pair on pipeline input, so
  the pipeline leaves it off (it costs ~4x in boxes_to_mesh at plaque scale);
* the signed mesh volume is compared with the voxel count (closed, outward
  oriented boxes have exactly that volume) - reported as ``volume_ok``.

Like flat Stack5's HighFidelityMesher output, every material mesh is a union of
closed axis-aligned boxes: exact volume, consistent outward winding, no gaps or
overlaps between parts (verified by rasterisation in validate.py), but NOT
trimesh-"watertight" in the edge-manifold sense because neighbouring boxes of
different heights meet in T-junctions.  Bambu Studio slices such unions the same
way it slices the flat Stack5 plaques.

Vertex layout (same as Lumina's converter, so the plaque reads correctly when
viewed from +Z, face up): x = column * pixel_mm, y = (H - row) * pixel_mm,
z = layer * layer_h.  Mesh coordinates are local; placement on the bed is the
build-item transform written by postprocess_3mf.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import trimesh

from core.mesh_generators import HighFidelityMesher
from core.stack5.pipeline import _merge_layers_no_dilation

# box corner template: (x, y, z) in {0,1}; faces wound outward (from core/mesh_generators.py)
_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                     [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.int64)
# Every face is split along the diagonal from its min corner to its max corner, so the
# +x face of one box and the -x face of its neighbour (same extent) produce the same two
# triangles with opposite winding - which lets cancel_interior remove them.
_BOX_FACES = np.array([[0, 2, 1], [0, 3, 2],        # bottom (-z)  diagonal 0-2
                       [4, 5, 6], [4, 6, 7],        # top (+z)     diagonal 4-6
                       [0, 1, 5], [0, 5, 4],        # -y           diagonal 0-5
                       [1, 2, 6], [1, 6, 5],        # +x           diagonal 1-6
                       [3, 6, 2], [3, 7, 6],        # +y           diagonal 3-6
                       [0, 7, 3], [0, 4, 7]], dtype=np.int64)  # -x  diagonal 0-7


class NoDilationMesher(HighFidelityMesher):
    """Lumina's high-fidelity mesher with the 3x3 mask dilation disabled
    (per-material parts must not overlap)."""
    _merge_layers_with_dilation = _merge_layers_no_dilation


def layer_groups(vox: np.ndarray, mat_id: int) -> list[tuple[int, int, np.ndarray]]:
    """[(z_start, z_end_inclusive, mask (H, W) bool)] for one material."""
    return _merge_layers_no_dilation(None, vox, mat_id)


def greedy_rectangles(mask: np.ndarray) -> np.ndarray:
    """(N, 4) int64 [x0, y0, x1, y1) covering ``mask`` exactly (Lumina's greedy merge)."""
    rects = HighFidelityMesher()._greedy_rect_merge(np.asarray(mask, dtype=bool), mask.shape[0])
    if not rects:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(rects, dtype=np.float64).astype(np.int64)


def boxes_for_material(vox: np.ndarray, mat_id: int) -> np.ndarray:
    """(N, 6) int64 boxes [x0, y0, x1, y1, z0, z1) (half-open, voxel units) that
    partition every voxel == mat_id.  Rectangles repeating in the directly
    following layer group are merged upward."""
    groups = layer_groups(vox, int(mat_id))
    boxes: list[list[int]] = []
    open_boxes: dict[tuple[int, int, int, int], int] = {}   # rect -> index of a box ending at prev_end
    prev_end = None
    for z0, z1, mask in groups:
        z1 += 1                                          # half-open
        rects = greedy_rectangles(mask)
        new_open: dict[tuple[int, int, int, int], int] = {}
        for r in rects:
            key = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))
            if prev_end == z0 and key in open_boxes:
                idx = open_boxes[key]
                boxes[idx][5] = z1
                new_open[key] = idx
            else:
                boxes.append([key[0], key[1], key[2], key[3], int(z0), int(z1)])
                new_open[key] = len(boxes) - 1
        open_boxes = new_open
        prev_end = z1
    if not boxes:
        return np.zeros((0, 6), dtype=np.int64)
    return np.asarray(boxes, dtype=np.int64)


def boxes_to_mesh(boxes: np.ndarray, height_px: int, pixel_mm: float, layer_h: float,
                  cancel_interior: bool = True) -> trimesh.Trimesh:
    """Boxes (voxel units) -> trimesh in mm.  Coincident opposite triangles
    (a face shared by two boxes of the same material) are removed."""
    b = np.asarray(boxes, dtype=np.int64)
    if b.size == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False)
    H = int(height_px)
    n = b.shape[0]
    # integer world coordinates: X = x, Y = H - y (image row 0 at max y), Z = z
    x = np.stack([b[:, 0], b[:, 2]], axis=1)                     # (n, 2) x0, x1
    y = np.stack([H - b[:, 3], H - b[:, 1]], axis=1)             # (n, 2) y_low, y_high
    z = np.stack([b[:, 4], b[:, 5]], axis=1)                     # (n, 2)
    cx = x[:, _CORNERS[:, 0]]                                    # (n, 8)
    cy = y[:, _CORNERS[:, 1]]
    cz = z[:, _CORNERS[:, 2]]
    corners = np.stack([cx, cy, cz], axis=2).reshape(-1, 3)      # (n*8, 3)
    faces = (_BOX_FACES[None, :, :] + (np.arange(n, dtype=np.int64) * 8)[:, None, None]).reshape(-1, 3)
    # unique integer vertex ids
    W1 = int(corners[:, 0].max()) + 1
    H1 = int(corners[:, 1].max()) + 1
    ids = (corners[:, 2] * H1 + corners[:, 1]) * W1 + corners[:, 0]
    uniq, inverse = np.unique(ids, return_inverse=True)
    faces = inverse[faces]
    if cancel_interior and len(faces):
        a, bb, c = faces[:, 0], faces[:, 1], faces[:, 2]
        parity = ((a > bb).astype(np.int64) + (a > c) + (bb > c)) & 1      # 1 = odd permutation
        key = np.sort(faces, axis=1)
        _, grp, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        grp = grp.reshape(-1)
        signed = np.bincount(grp, weights=(parity * 2 - 1).astype(np.float64), minlength=counts.size)
        drop = (counts[grp] == 2) & (signed[grp] == 0)
        faces = faces[~drop]
    used = np.unique(faces) if len(faces) else np.zeros(0, np.int64)
    remap = np.full(uniq.size, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    vid = uniq[used]
    vx = vid % W1
    vy = (vid // W1) % H1
    vz = vid // (W1 * H1)
    verts = np.stack([vx * float(pixel_mm), vy * float(pixel_mm), vz * float(layer_h)], axis=1)
    return trimesh.Trimesh(vertices=verts, faces=remap[faces], process=False)


def mesh_relief_voxels(vox: np.ndarray, slot_names: Sequence[str], pixel_mm: float, layer_h: float,
                       cancel_interior: bool = False, check_volume: bool = True) -> list[dict]:
    """One mesh per palette slot present in ``vox`` (slot order).  Each entry:
    {slot, name, mesh, n_boxes, n_voxels, volume_mm3, expected_volume_mm3, volume_ok}."""
    Z, H, W = vox.shape
    cell = float(pixel_mm) ** 2 * float(layer_h)
    out = []
    for slot, name in enumerate(slot_names):
        n_vox = int(np.count_nonzero(vox == slot))
        if n_vox == 0:
            continue
        boxes = boxes_for_material(vox, slot)
        box_vox = int(np.sum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4])))
        if box_vox != n_vox:
            raise RuntimeError(f"{name}: boxes cover {box_vox} voxels but the material has {n_vox}")
        mesh = boxes_to_mesh(boxes, H, pixel_mm, layer_h, cancel_interior=cancel_interior)
        entry = {'slot': slot, 'name': name, 'mesh': mesh, 'n_boxes': int(boxes.shape[0]),
                 'n_voxels': n_vox, 'n_triangles': int(len(mesh.faces)), 'n_vertices': int(len(mesh.vertices)),
                 'expected_volume_mm3': n_vox * cell}
        if check_volume:
            vol = float(mesh.volume)
            entry['volume_mm3'] = vol
            # half a voxel: float64 error on a 1500x1500x65 plaque is ~1e-9 mm3
            entry['volume_ok'] = bool(abs(vol - n_vox * cell) <= 0.5 * cell)
            entry['winding_consistent'] = bool(mesh.is_winding_consistent)
        out.append(entry)
    return out


def scene_from_meshes(meshes: Sequence[dict]) -> trimesh.Scene:
    scene = trimesh.Scene()
    for e in meshes:
        m = e['mesh']
        m.metadata['name'] = e['name']
        scene.add_geometry(m, node_name=e['name'], geom_name=e['name'])
    return scene
