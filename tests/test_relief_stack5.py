"""Relief Stack5 (core/relief): depth processing, face-up voxel stack, meshing,
3MF export/verification, provider errors, recipe invariance vs flat Stack5."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.relief.depth_processing import (DepthParams, limit_neighbor_step, morphological_clean,  # noqa: E402
                                          normalize_depth, process_depth, quantize_steps)
from core.relief.mesh import boxes_for_material, boxes_to_mesh, mesh_relief_voxels  # noqa: E402
from core.relief.provider import (GeometryError, GeometryResult, MissingDependencyError,  # noqa: E402
                                  get_provider, load_depth_file)
from core.relief.relief_stack import ReliefDims, build_relief_voxels, check_relief_voxels  # noqa: E402
from core.relief.validate import parse_object_meshes, rasterize_meshes, read_object_model  # noqa: E402

PALETTE = ['Black', 'White', 'Pink', 'Red', 'Klein Blue']
LH = 0.08


# --------------------------------------------------------------------------- helpers
def _gradient_image(path, n=48):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    rgb = np.stack([0.9 * xx + 0.05, 0.6 * yy + 0.1, 0.9 * (1 - xx) * yy + 0.05], -1)
    rgb[: n // 4, : n // 4] = 0.0
    rgb[-n // 4:, -n // 4:] = 1.0
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
    return str(path)


def _ramp_depth(path, n=48, axis=1, bits=8):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    g = xx if axis == 1 else yy
    if bits == 16:
        Image.fromarray(np.round(g * 65535).astype(np.uint16)).save(path)
    else:
        Image.fromarray(np.round(g * 255).astype(np.uint8)).save(path)
    return str(path)


def _random_recipe(H, W, L=5, n=5, seed=0, hole=True):
    rng = np.random.default_rng(seed)
    mm = rng.integers(0, n, size=(H, W, L)).astype(np.int64)
    mask = np.ones((H, W), bool)
    if hole:
        mask[:3, :4] = False
        mask[H // 2, W // 2] = False
    mm[~mask] = -1
    return mm, mask


def _relief_run(tmp_path, img, dep, sub='out', **kw):
    from core.relief.pipeline import convert_album_relief
    kw.setdefault('palette', PALETTE)
    kw.setdefault('quantize_colors', 12)
    kw.setdefault('verify', 'yes')
    return convert_album_relief(img, width_mm=4.8, depth_map=dep, out_dir=str(tmp_path / sub), seed=0, **kw)


# --------------------------------------------------------------------------- dims
def test_dims_defaults_budget_and_validation():
    d = ReliefDims()
    assert (d.base_layers, d.relief_layers, d.shell_layers) == (10, 50, 5)
    assert d.total_max_layers == 65 and d.summary()['total_mm_max'] == pytest.approx(5.2)
    assert d.summary()['base_mm'] == pytest.approx(0.8) and d.summary()['shell_mm'] == pytest.approx(0.4)
    assert d.rounding_warnings() == []
    assert ReliefDims(base_mm=0.83).rounding_warnings()      # not a multiple of 0.08 -> warned, rounded to 10
    assert ReliefDims(base_mm=0.83).base_layers == 10
    with pytest.raises(ValueError):
        ReliefDims(base_mm=0.0)
    with pytest.raises(ValueError):
        ReliefDims(layer_h=0)


# --------------------------------------------------------------------------- depth processing
def test_flat_depth_gives_uniform_relief():
    dims = ReliefDims()
    g = GeometryResult(depth=np.full((20, 30), 0.37, np.float32))
    f = process_depth(g, (20, 30), dims, DepthParams())
    assert np.unique(f.steps).tolist() == [0]
    assert any('flat' in w for w in f.report['warnings'])
    # 'none' normalisation keeps an authored constant level
    f2 = process_depth(GeometryResult(depth=np.full((20, 30), 0.5, np.float32)), (20, 30), dims,
                       DepthParams(normalize='none'))
    assert np.unique(f2.steps).tolist() == [25]
    # a flat map stays 0 even with invert / flatten_background
    f3 = process_depth(GeometryResult(depth=np.full((20, 30), 0.37, np.float32), mask=np.eye(20, 30, dtype=bool)),
                       (20, 30), dims, DepthParams(invert=True, flatten_background=True, background_level=0.5))
    assert np.unique(f3.steps).tolist() == [0]
    # normalize='none' keeps authored headroom even with smoothing on (default)
    d = np.linspace(0.2, 0.6, 64)[None, :].repeat(40, 0).astype(np.float32)
    f4 = process_depth(GeometryResult(depth=d), (40, 64), dims, DepthParams(normalize='none'))
    assert int(f4.steps.min()) == 10 and int(f4.steps.max()) == 30
    f5 = process_depth(GeometryResult(depth=d * 300.0), (40, 64), dims, DepthParams(normalize='none', smoothing_px=0))
    assert any("normalize='none'" in w for w in f5.report['warnings']) and np.unique(f5.steps).tolist() == [50]


def test_gradient_is_monotonic_and_spans_0_to_50():
    dims = ReliefDims()
    d = np.linspace(0, 1, 64)[None, :].repeat(40, 0).astype(np.float32)
    for params in (DepthParams(smoothing_px=0, min_feature_px=0), DepthParams()):
        f = process_depth(GeometryResult(depth=d), (40, 64), dims, params)
        assert np.all(np.diff(f.steps, axis=1) >= 0)
        assert int(f.steps.min()) == 0 and int(f.steps.max()) == 50
        assert f.report['stats']['levels_used'] >= 40
    # resampled onto a different grid (aspect warning recorded)
    f3 = process_depth(GeometryResult(depth=d), (30, 90), dims, DepthParams(smoothing_px=0, min_feature_px=0))
    assert f3.steps.shape == (30, 90) and np.all(np.diff(f3.steps, axis=1) >= 0)
    assert any('aspect' in w for w in f3.report['warnings'])


def test_heights_are_whole_layers_within_budget():
    dims = ReliefDims()
    rng = np.random.default_rng(1)
    d = rng.random((33, 47)).astype(np.float32) * 123.0 - 7.0
    f = process_depth(GeometryResult(depth=d), (33, 47), dims, DepthParams())
    assert f.steps.dtype == np.int16
    assert f.steps.min() >= 0 and f.steps.max() <= 50
    h = f.height_mm()
    assert np.allclose(h / LH, np.round(h / LH), atol=1e-9)
    assert h.max() <= 4.0 + 1e-9
    f2 = process_depth(GeometryResult(depth=d), (33, 47), ReliefDims(relief_mm=2.0), DepthParams(step_layers=5))
    assert f2.steps.max() <= 25 and set(np.unique(f2.steps) % 5) == {0}


def test_invert_percentiles_and_quantize_rule():
    dims = ReliefDims()
    d = np.linspace(0, 1, 50)[None, :].repeat(10, 0).astype(np.float32)
    p0 = DepthParams(smoothing_px=0, min_feature_px=0)
    a = process_depth(GeometryResult(depth=d), (10, 50), dims, p0).steps
    b = process_depth(GeometryResult(depth=d), (10, 50), dims, DepthParams(invert=True, smoothing_px=0, min_feature_px=0)).steps
    assert np.array_equal(a[:, ::-1], b) or np.all(np.diff(b, axis=1) <= 0)
    assert b[0, 0] == 50 and b[0, -1] == 0
    # a single spike is ignored by the 99th percentile and clamps to the top
    spiky = np.full((20, 20), 0.2, np.float32)
    spiky[5, 5] = 1000.0
    spiky[:, 10:] = 0.4
    norm, info = normalize_depth(spiky, DepthParams())
    assert norm[5, 5] == 1.0 and info['high'] < 1.0
    assert norm[0, 0] == 0.0 and norm[0, 15] == 1.0
    # round(norm * 4.0 / 0.08) half-up: exact ties 0.25 -> 12.5 -> 13, 0.75 -> 37.5 -> 38
    q = quantize_steps(np.array([[0.0, 0.5, 0.995, 0.02, 1.0, 0.25, 0.75]], np.float64), dims)
    assert q.tolist() == [[0, 25, 50, 1, 50, 13, 38]]
    # step_layers stays a multiple: 50 // 3 * 3 = 48 is the top
    q3 = quantize_steps(np.array([[1.0, 0.98, 0.5]], np.float64), dims, step_layers=3)
    assert q3.tolist() == [[48, 48, 24]]
    # 'higher-is-farther' providers are negated
    far = GeometryResult(depth=d, depth_convention='higher-is-farther')
    c = process_depth(far, (10, 50), dims, p0).steps
    assert c[0, 0] == 50 and c[0, -1] == 0


def test_flatten_background_and_slope_limits():
    dims = ReliefDims()
    H, W = 40, 40
    d = np.zeros((H, W), np.float32)
    d[10:30, 10:30] = 1.0                      # a raised square object
    mask = d > 0.5
    g = GeometryResult(depth=d, mask=mask)
    f = process_depth(g, (H, W), dims, DepthParams(flatten_background=True, background_level=0.0,
                                                    smoothing_px=0, min_feature_px=0))
    assert np.all(f.steps[~mask] == 0) and np.all(f.steps[mask] == 50)
    # protected silhouette: the object edge stays a vertical wall even with a step limit
    f2 = process_depth(g, (H, W), dims, DepthParams(flatten_background=True, max_neighbor_step=5,
                                                     smoothing_px=0, min_feature_px=0))
    assert np.all(f2.steps[mask] == 50) and np.all(f2.steps[~mask] == 0)
    assert f2.report['slope_limit']['applied'] is True
    # unprotected: the cliff becomes a ramp of <= 5 layers per pixel everywhere
    f3 = process_depth(g, (H, W), dims, DepthParams(flatten_background=True, max_neighbor_step=5,
                                                     protect_silhouette=False, smoothing_px=0, min_feature_px=0))
    s = f3.steps.astype(int)
    assert np.abs(np.diff(s, axis=0)).max() <= 5 and np.abs(np.diff(s, axis=1)).max() <= 5
    assert s.max() == 50 and s[~mask].min() == 0
    # direct limiter: a lone pillar is cut to max_step above its neighbours
    st = np.zeros((9, 9), np.int16)
    st[4, 4] = 50
    lim, info = limit_neighbor_step(st, 10)
    assert lim[4, 4] == 10 and lim.sum() == 10 and info['pixels_lowered'] == 1
    # protected silhouette, background NOT flattened: a cliff inside the background is
    # still limited (both regions are limited separately), the outline stays a cliff
    d2 = np.zeros((H, W), np.float32)
    d2[10:30, 10:30] = 1.0
    d2[:, 35:] = 0.8                        # background step
    f4 = process_depth(GeometryResult(depth=d2, mask=mask), (H, W), dims,
                       DepthParams(max_neighbor_step=5, smoothing_px=0, min_feature_px=0))
    s4 = f4.steps.astype(int)
    bg = ~mask
    assert np.abs(np.diff(np.where(bg, s4, 0), axis=1))[:, 33:].max() <= 5     # background cliff ramped
    assert s4[15, 9] == 0 and s4[15, 10] == 50                                   # outline untouched
    assert f4.report['slope_limit']['background']['applied'] is True
    # protect off + flattened background: the limiter's result is final (no cliff re-created)
    f5 = process_depth(g, (H, W), dims, DepthParams(flatten_background=True, background_level=1.0,
                                                     protect_silhouette=False, max_neighbor_step=5,
                                                     smoothing_px=0, min_feature_px=0))
    s5 = f5.steps.astype(int)
    assert np.abs(np.diff(s5, axis=0)).max() <= 5 and np.abs(np.diff(s5, axis=1)).max() <= 5
    # rim bleed: a coarse object on a low background resampled onto a finer grid keeps its
    # rim at the object level (nearest fill before resampling)
    d3 = np.zeros((20, 20), np.float32)
    d3[5:15, 5:15] = 1.0
    m3 = d3 > 0.5
    f6 = process_depth(GeometryResult(depth=d3, mask=m3), (40, 40), dims,
                       DepthParams(flatten_background=True, smoothing_px=0, min_feature_px=0))
    assert np.all(f6.steps[f6.mask] == 50) and np.all(f6.steps[~f6.mask] == 0)


def test_min_feature_cleanup_removes_spikes_and_pits_without_shift():
    sp = np.zeros((24, 24), np.int16)
    sp[12, 12] = 40                # 1-px spike
    sp[2:10, 2:10] = 30            # 8x8 plateau
    sp[5, 5] = 0                   # 1-px pit in it
    sp[14:16, 14:22] = 25          # 2-px ridge
    out = morphological_clean(sp, 4)
    assert out[12, 12] == 0 and out[14, 18] == 0
    assert np.all(out[2:10, 2:10] == 30)
    q = np.zeros((20, 20), np.int16)
    q[7:13, 7:13] = 20
    for k in (3, 4, 5):
        assert np.array_equal(morphological_clean(q, k), q)          # nothing shifts, nothing shrinks
    m = np.zeros((20, 20), bool)
    m[:, :13] = True
    assert np.array_equal(morphological_clean(q, 4, mask=m), q)      # mask edge does not erode the plateau
    assert set(np.unique(out)) <= {0, 30}                            # whole layers only


# --------------------------------------------------------------------------- face-up voxels
def test_voxel_stack_face_up_structure_and_thickness():
    dims = ReliefDims()
    H, W, L = 14, 18, 5
    mm, mask = _random_recipe(H, W)
    steps = np.round(np.linspace(0, 50, W))[None, :].repeat(H, 0).astype(np.int16)
    vox, meta = build_relief_voxels(mm, mask, steps, dims, backing_slot=1)
    assert vox.shape == (65, H, W) and vox.dtype == np.int8
    assert meta['total_layers'] == 65 and meta['total_mm'] == pytest.approx(5.2)
    assert meta['base_layers'] == 10
    rep = check_relief_voxels(vox, mm, mask, steps, dims, 1)
    assert rep['ok'], rep
    yy, xx = np.nonzero(mask)
    surf = 10 + steps[yy, xx]
    for i in range(L):                                     # stack[4] at the surface ... stack[0] on top
        assert np.array_equal(vox[surf + i, yy, xx], mm[yy, xx, L - 1 - i])
    assert np.array_equal(vox[surf + L - 1, yy, xx], mm[yy, xx, 0])          # stack[0] uppermost
    top_air = surf + L
    ok = top_air < vox.shape[0]
    assert np.all(vox[top_air[ok], yy[ok], xx[ok]] == -1)                    # nothing above the shell
    for z in range(10):
        assert np.all(vox[z][mask] == 1) and np.all(vox[z][~mask] == -1)     # flat base, backing only
    zz = np.arange(65)[:, None, None]
    below = (zz < surf.max()) & mask[None]
    body = (zz < (10 + steps)[None]) & mask[None]
    assert np.all(vox[body] == 1)                                             # solid backing under every shell
    n_solid = (vox >= 0).sum(axis=0)
    assert np.all(n_solid[mask] == 10 + steps[mask] + L)                     # exactly 5 colour voxels per pixel
    assert not (vox[:, ~mask] >= 0).any()
    # zero relief everywhere: 15 layers = 1.2 mm
    vox0, meta0 = build_relief_voxels(mm, mask, np.zeros((H, W), np.int16), dims, 1)
    assert vox0.shape[0] == 15 and meta0['total_mm'] == pytest.approx(1.2)
    assert meta['printed_layer_material_sets'][0] == [1]
    assert len(meta['printed_layer_material_sets']) == 65


def test_voxel_stack_rejects_bad_inputs():
    dims = ReliefDims()
    mm, mask = _random_recipe(8, 8)
    with pytest.raises(ValueError):
        build_relief_voxels(mm, mask, np.full((8, 8), 51, np.int16), dims, 1)
    with pytest.raises(ValueError):
        build_relief_voxels(mm[:, :, :4], mask, np.zeros((8, 8), np.int16), dims, 1)
    bad = mm.copy()
    bad[4, 5, 2] = -1
    with pytest.raises(ValueError):
        build_relief_voxels(bad, mask, np.zeros((8, 8), np.int16), dims, 1)
    with pytest.raises(ValueError):
        build_relief_voxels(mm, mask, np.zeros((8, 8), np.int16), dims, 7, n_slots=5)
    with pytest.raises(ValueError):
        build_relief_voxels(mm, mask, np.zeros((8, 8), np.int16), dims, 1, n_slots=3)


# --------------------------------------------------------------------------- meshing
def test_meshes_are_closed_boxes_with_exact_volume_and_no_overlap():
    dims = ReliefDims()
    H, W = 12, 16
    mm, mask = _random_recipe(H, W, seed=3)
    steps = np.round(np.linspace(0, 50, W))[None, :].repeat(H, 0).astype(np.int16)
    vox, _ = build_relief_voxels(mm, mask, steps, dims, backing_slot=1)
    meshes = mesh_relief_voxels(vox, PALETTE, 0.1, LH)
    assert [e['slot'] for e in meshes] == [0, 1, 2, 3, 4]
    for e in meshes:
        assert e['volume_ok'] and e['winding_consistent'], e['name']
        assert e['expected_volume_mm3'] == pytest.approx(e['n_voxels'] * 0.1 * 0.1 * LH)
        assert e['mesh'].vertices[:, 2].min() >= 0
    # rasterising the meshes reproduces the voxel matrix exactly: no gap, no overlap
    occ = rasterize_meshes([(e['mesh'].vertices, e['mesh'].faces) for e in meshes], vox.shape[0], (H, W), 0.1, LH)
    occ = occ[:, ::-1, :]
    expected = np.zeros_like(occ)
    for i, e in enumerate(meshes):
        expected[vox == e['slot']] |= np.uint16(1 << i)
    assert np.array_equal(occ, expected)
    # boxes sharing a whole face (stacked, x-adjacent, y-adjacent) fuse into one watertight solid
    for b in (np.array([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 1, 3]]), np.array([[0, 0, 2, 2, 0, 3], [2, 0, 4, 2, 0, 3]]),
              np.array([[0, 0, 2, 2, 0, 3], [0, 2, 2, 4, 0, 3]])):
        m = boxes_to_mesh(b, 4, 1.0, 1.0, cancel_interior=True)
        assert len(m.faces) == 24 - 4 and m.is_watertight and m.is_winding_consistent
        assert m.volume == pytest.approx(boxes_to_mesh(b, 4, 1.0, 1.0, cancel_interior=False).volume)
    m = boxes_to_mesh(np.array([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 1, 3]]), 2, 1.0, 1.0, cancel_interior=True)
    assert m.volume == pytest.approx(4 + 8)
    # a neighbour whose side face is larger (T-junction) stays a closed box union with exact volume
    b3 = np.array([[0, 0, 2, 2, 0, 1], [0, 0, 2, 2, 1, 3], [2, 0, 3, 2, 0, 3]])
    m3 = boxes_to_mesh(b3, 2, 1.0, 1.0)
    assert m3.volume == pytest.approx(4 + 8 + 6) and len(m3.faces) == 36 - 4 and m3.is_winding_consistent
    # a body with two heights: boxes partition the voxels, volume exact
    v = np.full((6, 4, 6), -1, np.int8)
    v[:3, :, :3] = 0
    v[:5, :, 3:] = 0
    bx = boxes_for_material(v, 0)
    assert bx.shape[0] == 2
    assert boxes_to_mesh(bx, 4, 1.0, 1.0).volume == pytest.approx(4 * 3 * 3 + 4 * 3 * 5)


# --------------------------------------------------------------------------- pipeline
def test_pipeline_imported_depth_end_to_end(tmp_path):
    img = _gradient_image(tmp_path / 'grad48.png')
    dep = _ramp_depth(tmp_path / 'ramp.png', bits=16)
    res = _relief_run(tmp_path, img, dep)
    assert os.path.isfile(res['threemf']) and os.path.isfile(res['preview_png']) and os.path.isfile(res['depth_png'])
    assert res['voxel_check']['ok'] and res['verification']['ok']
    assert res['verification']['overlapping_voxels'] == 0 and res['verification']['uncovered_solid_voxels'] == 0
    assert res['verification']['covered_air_voxels'] == 0 and res['verification']['bottom_layer_all_backing']
    st = res['stats']
    assert st['total_print_layers'] <= 65 and st['total_height_mm'] <= 5.2 + 1e-9
    assert st['relief_layers_max'] == 50 and st['total_print_layers'] == 65
    assert int(res['steps'].min()) == 0 and int(res['steps'].max()) == 50
    assert res['dims']['base_layers'] == 10 and res['dims']['total_layers_max'] == 65
    rec = json.loads(open(res['recipe_json']).read())
    assert rec['mode'] == 'relief-stack5' and rec['colour']['depth_independent'] is True
    assert rec['thickness']['total_mm_max'] == pytest.approx(5.2) and rec['thickness']['shell_mm'] == pytest.approx(0.4)
    assert rec['settings']['depth_params']['max_neighbor_step'] is None
    assert [p['name'] for p in rec['palette']] == PALETTE
    assert rec['backing']['name'] == res['backing']
    assert rec['stats']['material_matrix_sha256'] == res['material_matrix_sha256']
    assert rec['geometry']['report']['stats']['relief_mm_max'] == pytest.approx(4.0)
    # 3MF contents
    with zipfile.ZipFile(res['threemf']) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        assert {'3D/3dmodel.model', '3D/Objects/object_1.model', 'Metadata/model_settings.config',
                'Metadata/project_settings.config', '[Content_Types].xml', '_rels/.rels'} <= names
        cfg = json.loads(zf.read('Metadata/project_settings.config'))
        ms = zf.read('Metadata/model_settings.config').decode()
        main = zf.read('3D/3dmodel.model').decode()
        obj = zf.read('3D/Objects/object_1.model')
    F = len(res['parts'])
    assert F == ms.count('<part ') == obj.count(b'<object ') == len(cfg['filament_colour']) >= 2
    exts = [int(x) for x in __import__('re').findall(r'key="extruder" value="(\d+)"', ms)]
    assert exts == list(range(1, F + 1))
    lib = {p['name']: p['hex'] for p in rec['palette']}
    assert cfg['filament_colour'] == [lib[p['name']] for p in res['parts']]
    assert cfg['layer_height'] == cfg['initial_layer_print_height'] == '0.08'
    assert cfg['print_settings_id'] == 'Stack5 0.08mm @BBL X2D' and cfg['printer_settings_id'] == 'Bambu Lab X2D 0.4 nozzle'
    assert cfg['enable_prime_tower'] == '1' and len(cfg['flush_volumes_matrix']) == 2 * F * F
    mtx = np.array(cfg['flush_volumes_matrix'][:F * F], dtype=int).reshape(F, F)
    assert (np.diagonal(mtx) == 0).all() and (mtx[~np.eye(F, dtype=bool)] > 0).all()
    assert mtx.tolist() == res['flush']['flush_matrix']
    assert cfg['wall_loops'] == '1' and cfg['sparse_infill_density'] == '100%' and cfg['brim_type'] == 'no_brim'
    tf = rec['threemf_check']['build_transform']
    assert f'transform="{tf}" printable="1"' in main and 'plater_name" value="grad48"' in ms
    assert res['tower']['fits'] and res['stats']['tool_changes_planned'] > 0
    # relief prints face up with colour at varying heights: no backing height-range modifier
    with zipfile.ZipFile(res['threemf']) as zf:
        assert 'Metadata/layer_config_ranges.xml' not in zf.namelist()
    # every printed layer of the recipe has the same number of material sets as layers
    assert len(rec['geometry']['voxels']['printed_layer_material_sets']) == 65
    # the mesh z range is the model height, face up (min z = 0 reached by the backing part)
    parsed = parse_object_meshes(obj)
    zmin = min(V[:, 2].min() for V, _ in parsed)
    zmax = max(V[:, 2].max() for V, _ in parsed)
    assert zmin == pytest.approx(0.0) and zmax == pytest.approx(5.2)
    backing_part = [p for p in res['parts'] if p['slot'] == res['backing_slot']][0]
    assert parsed[backing_part['part_id'] - 1][0][:, 2].min() == pytest.approx(0.0)
    assert all(T.max() < len(V) for V, T in parsed)


def test_depth_changes_geometry_but_not_the_stack5_recipe(tmp_path):
    from core.stack5.pipeline import convert_album_stack5
    img = _gradient_image(tmp_path / 'grad48.png')
    dep_x = _ramp_depth(tmp_path / 'ramp_x.png', axis=1)
    dep_y = _ramp_depth(tmp_path / 'ramp_y.png', axis=0)
    a = _relief_run(tmp_path, img, dep_x, 'a', verify='no')
    b = _relief_run(tmp_path, img, dep_y, 'b', verify='no')
    assert a['material_matrix_sha256'] == b['material_matrix_sha256']
    assert not np.array_equal(a['steps'], b['steps'])
    assert a['palette'] == b['palette'] and a['backing'] == b['backing']
    ra = json.loads(open(a['recipe_json']).read())
    rb = json.loads(open(b['recipe_json']).read())
    assert ra['colour']['palette_report']['best']['cost'] == rb['colour']['palette_report']['best']['cost']
    # ... and it is exactly the flat (face-down) Stack5 recipe for the same image + palette.
    # The recipe depends on the first-layer height (the LUT models the viewing layer at its
    # real thickness), and relief still models a 0.08 mm first layer, so compare at 0.08 -
    # flat Stack5's own default is 0.16.  If relief ever moves to 0.16 this pins it.
    flat = convert_album_stack5(img, width_mm=4.8, palette=PALETTE, quantize_colors=12,
                                out_dir=str(tmp_path / 'flat'), seed=0, first_layer_mm=0.08)
    assert flat['stats']['material_matrix_sha256'] == a['material_matrix_sha256']
    assert flat['palette'] == a['palette'] and flat['backing'] == a['backing']


def test_relief_pipeline_is_deterministic(tmp_path):
    img = _gradient_image(tmp_path / 'grad48.png')
    dep = _ramp_depth(tmp_path / 'ramp.png')
    a = _relief_run(tmp_path, img, dep, 'a', verify='no')
    b = _relief_run(tmp_path, img, dep, 'b', verify='no')
    ha = hashlib.sha256(open(a['threemf'], 'rb').read()).hexdigest()
    hb = hashlib.sha256(open(b['threemf'], 'rb').read()).hexdigest()
    assert ha == hb
    ra = json.loads(open(a['recipe_json']).read())
    rb = json.loads(open(b['recipe_json']).read())
    for r in (ra, rb):
        r.pop('timing')
        r.pop('outputs')
        r['colour']['palette_report'].pop('seconds', None)
    assert ra == rb


def test_tiny_3mf_is_valid_xml_package(tmp_path):
    img = _gradient_image(tmp_path / 'g16.png', n=16)
    dep = _ramp_depth(tmp_path / 'r16.png', n=16)
    res = _relief_run(tmp_path, img, dep, palette=['Black', 'White'], quantize_colors=4)
    with zipfile.ZipFile(res['threemf']) as zf:
        assert zf.testzip() is None
        ct = ET.fromstring(zf.read('[Content_Types].xml'))
        assert ct.tag.endswith('Types')
        rels = ET.fromstring(zf.read('_rels/.rels'))
        targets = {r.get('Target') for r in rels}
        assert '/3D/3dmodel.model' in targets
        main = ET.fromstring(zf.read('3D/3dmodel.model'))
        ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
        assert main.get('unit') == 'millimeter'
        items = main.findall('.//m:build/m:item', ns)
        assert len(items) == 1 and items[0].get('printable') == '1'
        obj = ET.fromstring(zf.read('3D/Objects/object_1.model'))
        objects = obj.findall('.//m:object', ns)
        assert len(objects) == len(res['parts']) >= 1
        for o in objects:
            verts = o.findall('.//m:vertices/m:vertex', ns)
            tris = o.findall('.//m:triangles/m:triangle', ns)
            assert len(verts) >= 8 and len(tris) >= 12
            idx = np.array([[int(t.get('v1')), int(t.get('v2')), int(t.get('v3'))] for t in tris])
            assert idx.min() >= 0 and idx.max() < len(verts)
        ms = ET.fromstring(zf.read('Metadata/model_settings.config'))
        assert ms.find('plate') is not None
        json.loads(zf.read('Metadata/project_settings.config'))


# --------------------------------------------------------------------------- providers
def test_provider_registry_and_helpful_errors(tmp_path):
    with pytest.raises(ValueError):
        get_provider('no-such-provider')
    with pytest.raises(GeometryError):
        get_provider('imported-depth')
    with pytest.raises(GeometryError, match='--mesh'):
        get_provider('wonder3d')
    prov = get_provider('imported-depth', depth_map=str(tmp_path / 'missing.png'))
    with pytest.raises(FileNotFoundError):
        prov.generate(str(tmp_path / 'x.png'))
    dep = _ramp_depth(tmp_path / 'r.png', n=20, bits=16)
    g = get_provider('imported-depth', depth_map=dep).generate('unused.png')
    assert g.shape == (20, 20) and g.depth_convention == 'higher-is-closer'
    assert g.depth.min() == pytest.approx(0.0) and g.depth.max() == pytest.approx(1.0)
    # alpha becomes the mask; 8-bit grey loads as 0..1
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[2:8, 2:8, 3] = 255
    Image.fromarray(rgba, 'RGBA').save(tmp_path / 'a.png')
    d, m = load_depth_file(str(tmp_path / 'a.png'))
    assert m.sum() == 36 and d.max() <= 1.0
    with pytest.raises(GeometryError):
        GeometryResult(depth=np.zeros((1, 5)))
    with pytest.raises(GeometryError):
        GeometryResult(depth=np.full((4, 4), np.nan))
    # .npz: 'mask' is never taken as the depth; .npy (H, W, 1) is squeezed
    np.savez(tmp_path / 'd.npz', mask=np.ones((6, 6), bool), heights=np.arange(36, dtype=np.float32).reshape(6, 6))
    d, m = load_depth_file(str(tmp_path / 'd.npz'))
    assert d[5, 5] == 35.0 and m.all()
    np.savez(tmp_path / 'm.npz', mask=np.ones((6, 6), bool))
    with pytest.raises(GeometryError):
        load_depth_file(str(tmp_path / 'm.npz'))
    np.save(tmp_path / 'h.npy', np.ones((5, 7, 1), np.float32))
    assert load_depth_file(str(tmp_path / 'h.npy'))[0].shape == (5, 7)
    # 16-bit RGB PNG keeps its precision (cv2 path)
    import cv2
    ramp16 = np.round(np.linspace(0, 2000, 64)).astype(np.uint16)[None, :].repeat(8, 0)
    cv2.imwrite(str(tmp_path / 'rgb16.png'), np.stack([ramp16] * 3, axis=-1))
    d16, _ = load_depth_file(str(tmp_path / 'rgb16.png'))
    assert d16.dtype == np.float32 and len(np.unique(d16)) >= 60 and d16.max() == pytest.approx(2000 / 65535, rel=1e-3)
    # Depth Anything metric checkpoints are flagged 'higher-is-farther'
    from core.relief.depth_anything import DepthAnythingProvider
    assert DepthAnythingProvider().depth_convention == 'higher-is-closer'
    assert DepthAnythingProvider(weights='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf').depth_convention == 'higher-is-farther'
    # missing depth map fails in preflight (before any colour work)
    with pytest.raises(FileNotFoundError):
        get_provider('imported-depth', depth_map=str(tmp_path / 'nope.png')).preflight()


def test_missing_optional_ai_dependencies_are_reported(tmp_path):
    from core.relief.depth_anything import dependency_status
    st = dependency_status()
    img = _gradient_image(tmp_path / 'g.png', n=8)
    prov = get_provider('depth-anything')
    if not all(st.values()):
        with pytest.raises(MissingDependencyError) as ei:
            prov.generate(img)
        assert 'pip install' in str(ei.value) and 'transformers' in str(ei.value)
    else:
        # packages present: weights must still not be downloaded silently
        prov = get_provider('depth-anything', weights=str(tmp_path / 'no-weights-here'))
        with pytest.raises(MissingDependencyError):
            prov.generate(img)
    # the CLI reports the dependency status without needing them
    r = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'convert_relief.py'), '--check-deps'],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and 'depth_anything' in r.stdout


def test_depth_anything_provider_when_installed(tmp_path):
    from core.relief.depth_anything import DEFAULT_MODEL_ID, DepthAnythingProvider, dependency_status, weights_cached
    if not all(dependency_status().values()) or not weights_cached(DEFAULT_MODEL_ID):
        pytest.skip('Depth Anything V2 packages or weights not installed')
    n = 64
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    rgb = np.full((n, n, 3), 40, np.uint8)
    disc = (xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.08
    rgb[disc] = (230, 200, 160)
    img = str(tmp_path / 'disc.png')
    Image.fromarray(rgb).save(img)
    prov = DepthAnythingProvider(device='cpu', allow_download=False)
    g = prov.generate(img)
    assert g.shape == (n, n) and g.depth_convention == 'higher-is-closer' and g.depth.dtype == np.float32
    assert np.isfinite(g.depth).all() and g.depth.max() > g.depth.min()
    g2 = prov.generate(img)
    assert np.array_equal(g.depth, g2.depth)                      # CPU inference is deterministic
    f = process_depth(g, (n, n), ReliefDims(), DepthParams())
    assert f.steps.min() == 0 and f.steps.max() == 50 and f.report['stats']['levels_used'] > 5


def test_wonder3d_orthographic_front_depth_from_mesh(tmp_path):
    import trimesh
    from core.relief.wonder3d import Wonder3DProvider, frame_rotation
    assert np.allclose(frame_rotation('+z', '+y'), np.eye(3))
    R = frame_rotation('-y', '+z')
    assert np.allclose(R @ np.array([0, -1.0, 0]), [0, 0, 1]) and np.allclose(R @ np.array([0, 0, 1.0]), [0, 1, 0])
    with pytest.raises(GeometryError):
        frame_rotation('+z', '+z')
    sph = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh_path = str(tmp_path / 'sphere.obj')
    sph.export(mesh_path)
    img = str(tmp_path / 'img.png')
    Image.new('RGB', (64, 64), (255, 255, 255)).save(img)
    g = Wonder3DProvider(mesh=mesh_path, render_px=64, fit='image').generate(img)
    assert g.shape == (64, 64) and g.depth_convention == 'higher-is-closer'
    assert abs(float(g.mask.mean()) - np.pi / 4) < 0.03
    assert g.depth[32, 32] > 0.95 and not g.mask[1, 1] and g.depth[1, 1] == pytest.approx(float(g.depth[g.mask].min()))
    # mask fit: mesh bbox lands on the mask bbox
    m = np.zeros((64, 64), bool)
    m[16:48, 32:64] = True
    Image.fromarray((m * 255).astype(np.uint8)).save(tmp_path / 'mask.png')
    g2 = Wonder3DProvider(mesh=mesh_path, render_px=64, mask=str(tmp_path / 'mask.png')).generate(img)
    rows, cols = np.nonzero(g2.mask)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (16, 47, 32, 63)
    # runs through the common relief processing: dome -> 0..50 with the centre highest
    f = process_depth(g, (64, 64), ReliefDims(), DepthParams(flatten_background=True, smoothing_px=0, min_feature_px=0))
    assert f.steps[32, 32] == 50 and f.steps[1, 1] == 0


def test_normal_integration_and_fusion():
    from core.relief.normal_integration import fuse_depth_normals, integrate_normals, normals_from_height
    yy, xx = np.mgrid[0:40, 0:56].astype(float)
    plane = 0.3 * xx - 0.2 * yy
    dome = 8 * np.exp(-((xx - 28) ** 2 + (yy - 20) ** 2) / 120.0)
    for z in (plane, dome, plane + dome):
        zi = integrate_normals(normals_from_height(z))
        err = (zi - zi.mean()) - (z - z.mean())
        assert np.sqrt((err ** 2).mean()) < 0.02
    fused, info = fuse_depth_normals(dome * 3, normals_from_height(dome), depth_weight=1.0, normal_weight=0.0)
    assert np.allclose(fused, dome * 3, atol=1e-4)
    fused2, info2 = fuse_depth_normals(dome * 3, normals_from_height(dome), depth_weight=0.1, normal_weight=1.0)
    assert info2['normal_scale'] == pytest.approx(3.0, rel=0.05)
    assert np.sqrt(((fused2 - 3 * dome) ** 2).mean()) < 0.1
    with pytest.raises(ValueError):
        fuse_depth_normals(dome, normals_from_height(dome), depth_weight=0.0, normal_weight=0.0)


# --------------------------------------------------------------------------- calibration + CLI
def test_calibration_assets_and_relief(tmp_path):
    from core.relief.calibration import CALIB_PALETTE, make_calibration_assets
    assets = make_calibration_assets(str(tmp_path / 'cal'), size_px=60)
    assert os.path.isfile(assets['image']) and os.path.isfile(assets['depth_map'])
    d, _ = load_depth_file(assets['depth_map'])
    assert d.shape == (60, 60) and d.max() == pytest.approx(1.0, abs=1e-3)
    from core.relief.pipeline import convert_album_relief
    res = convert_album_relief(assets['image'], width_mm=6.0, depth_map=assets['depth_map'], palette=CALIB_PALETTE,
                               quantize_colors=8, out_dir=str(tmp_path / 'out'), seed=0, verify='yes',
                               depth_params=DepthParams(normalize='none', smoothing_px=0, min_feature_px=0))
    s = res['steps']
    assert res['verification']['ok'] and s.max() == 50
    band = s[-6:, :]
    assert np.all(np.diff(band[-1], axis=0) >= 0) and band[-1, 0] == 0 and band[-1, -1] == 50
    assert s[int(0.42 * 59), 30] >= 45                            # dome centre near the full relief


def test_cli_smoke(tmp_path):
    img = _gradient_image(tmp_path / 'grad32.png', n=32)
    dep = _ramp_depth(tmp_path / 'ramp32.png', n=32)
    out = tmp_path / 'cli' / 'grad32_relief.3mf'
    cmd = [sys.executable, os.path.join(REPO, 'scripts', 'convert_relief.py'), img, '--depth-map', dep,
           '--width', '3.2', '--palette', 'Black,White,Red', '--quantize', '8', '--output', str(out),
           '--relief-mm', '2.4', '--base-mm', '0.8', '--max-neighbor-step', '6', '--verify', 'yes']
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert out.is_file() and (tmp_path / 'cli' / 'grad32_relief_recipe.json').is_file()
    assert '3MF verification: ok' in r.stdout and 'thickness: base 0.8 mm (10 layers)' in r.stdout
    rec = json.loads((tmp_path / 'cli' / 'grad32_relief_recipe.json').read_text())
    assert rec['thickness']['relief_layers_max'] == 30 and rec['stats']['total_print_layers'] <= 45
    assert rec['settings']['depth_params']['max_neighbor_step'] == 6
    r2 = subprocess.run(cmd + ['--color-layers', '4'], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r2.returncode != 0
    r3 = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'convert_relief.py'), img, '--width', '3.2'],
                        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r3.returncode != 0 and 'depth-map' in (r3.stderr + r3.stdout)
    r4 = subprocess.run(cmd + ['--layer-height', '0.1'], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r4.returncode != 0 and '0.08' in (r4.stderr + r4.stdout)
    # bad palette name: a clean error line and exit 2, not a traceback
    r5 = subprocess.run(cmd[:-8] + ['--palette', 'Black,Nope', '--output', str(tmp_path / 'cli' / 'x.3mf')],
                        cwd=REPO, capture_output=True, text=True, timeout=300)
    assert r5.returncode == 2 and '[RELIEF] error' in r5.stderr and 'Traceback' not in r5.stderr
    # a missing depth file is caught in preflight before the colour stage (fast, exit 2)
    r6 = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'convert_relief.py'), img, '--depth-map',
                         str(tmp_path / 'missing.png'), '--width', '3.2', '--palette', 'Black,White',
                         '--output', str(tmp_path / 'cli' / 'y.3mf')], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r6.returncode == 2 and 'not found' in r6.stderr and not (tmp_path / 'cli' / 'y_lut.npz').exists()


def test_cli_calibration_branch(tmp_path):
    out = tmp_path / 'cal' / 'sub' / 'plaque.3mf'
    cmd = [sys.executable, os.path.join(REPO, 'scripts', 'convert_relief.py'), '--calibration', '--calibration-mm', '6',
           '--quantize', '8', '--output', str(out), '--verify', 'yes', '--relief-mm', '2.4']
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert out.is_file() and (out.parent / 'relief_calibration.png').is_file()
    assert (out.parent / 'relief_calibration_depth.png').is_file()
    assert 'backing/body: Beige' in r.stdout and '0..2.4 mm' in r.stdout and '0.24 mm apart' in r.stdout
    assert '3MF verification: ok' in r.stdout
    rec = json.loads((out.parent / 'plaque_recipe.json').read_text())
    assert [p['name'] for p in rec['palette']] == ['Black', 'White', 'Red', 'Oak', 'Beige']
    assert rec['thickness']['relief_layers_max'] == 30 and rec['stats']['total_print_layers'] == 45
    r2 = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'convert_relief.py'), '--calibration',
                         '--calibration-mm', '6', '--quantize', '8', '--out', str(tmp_path / 'outdir')],
                        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0 and (tmp_path / 'outdir' / 'relief_calibration' / 'relief_calibration_relief.3mf').is_file()
