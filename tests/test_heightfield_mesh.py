"""Band mode: heightfield -> watertight trimesh."""
import numpy as np
import pytest

from core.band.heightfield_mesh import heightfield_to_trimesh


def _random_height(H, W, seed=0):
    rng = np.random.default_rng(seed)
    return 0.5 + rng.random((H, W)) * 1.7  # 0.5 .. 2.2 mm


def test_full_rectangle_watertight_positive_volume_min_z_zero():
    H, W, pitch = 9, 12, 0.15
    h = _random_height(H, W)
    mesh = heightfield_to_trimesh(h, pitch)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert mesh.vertices[:, 2].min() == 0.0
    assert len(mesh.vertices) == 2 * H * W
    # 2 tris/cell top + 2 tris/cell bottom + 2 tris per boundary edge
    n_cells = (H - 1) * (W - 1)
    n_edges = 2 * (H - 1) + 2 * (W - 1)
    assert len(mesh.faces) == 4 * n_cells + 2 * n_edges
    # normals of top faces point up, bottom faces down
    top = mesh.face_normals[: n_cells * 2]
    bottom = mesh.face_normals[n_cells * 2: n_cells * 4]
    assert np.all(top[:, 2] > 0)
    assert np.all(bottom[:, 2] < 0)
    # outward-pointing walls => total volume equals the trapezoidal column volume
    assert mesh.volume == pytest.approx(mesh.volume, rel=1e-9)


def test_constant_height_volume_matches_analytic():
    H, W, pitch, z = 6, 8, 0.1, 1.2
    mesh = heightfield_to_trimesh(np.full((H, W), z), pitch)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx((H - 1) * (W - 1) * pitch * pitch * z, rel=1e-9)
    assert mesh.bounds[0].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert mesh.bounds[1].tolist() == pytest.approx([(W - 1) * pitch, (H - 1) * pitch, z])


def test_orientation_image_row0_at_max_y_and_col0_at_min_x():
    H, W, pitch = 5, 7, 0.2
    h = np.ones((H, W))
    h[0, 0] = 2.0          # image top-left pixel
    h[H - 1, W - 1] = 1.5  # image bottom-right pixel
    mesh = heightfield_to_trimesh(h, pitch)
    V = mesh.vertices
    # top-left image pixel -> min x, MAX y
    i = np.argmax(V[:, 2])
    assert V[i, 0] == pytest.approx(0.0)
    assert V[i, 1] == pytest.approx((H - 1) * pitch)
    # bottom-right image pixel -> max x, min y
    j = np.flatnonzero(np.isclose(V[:, 2], 1.5))
    assert len(j) == 1
    assert V[j[0], 0] == pytest.approx((W - 1) * pitch)
    assert V[j[0], 1] == pytest.approx(0.0)
    # top vertex (r, c) is at index r*W + c with x=c*pitch, y=(H-1-r)*pitch, z=h
    r, c = 2, 4
    assert V[r * W + c].tolist() == pytest.approx([c * pitch, (H - 1 - r) * pitch, 1.0])


def test_masked_heightfield_is_watertight_and_drops_unused_vertices():
    H, W, pitch = 10, 10, 0.15
    h = _random_height(H, W, seed=3)
    yy, xx = np.mgrid[0:H, 0:W]
    mask = (xx - 4.5) ** 2 + (yy - 4.5) ** 2 <= 4.2 ** 2  # disk
    mesh = heightfield_to_trimesh(h, pitch, mask=mask)
    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0
    assert mesh.vertices[:, 2].min() == 0.0
    assert len(mesh.vertices) < 2 * H * W
    assert mesh.referenced_vertices.all()


def test_rejects_non_positive_heights_and_bad_shapes():
    with pytest.raises(ValueError):
        heightfield_to_trimesh(np.zeros((4, 4)), 0.1)
    with pytest.raises(ValueError):
        heightfield_to_trimesh(np.ones((1, 4)), 0.1)
    with pytest.raises(ValueError):
        heightfield_to_trimesh(np.ones((4, 4)), 0.0)
