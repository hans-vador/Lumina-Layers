"""Art-directed relief (core/relief/heightfield, art_direct, segment, object_provider):
mask operations on the mm height map, edit scripts, the pipeline hook, SAM / brush
masks, the object-provider stub and the editor selftest."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.relief.art_direct import (EditScript, EditSession, apply_edit_script, decode_mask_png,  # noqa: E402
                                    encode_mask_png, grid_to_image_xy, image_to_grid_xy, original_mm_from_field,
                                    relief_for_editor, replay_ops, resample_mask)
from core.relief.depth_processing import DepthParams, process_depth  # noqa: E402
from core.relief.heightfield import (apply_op, band_weight, dome_profile, feather_weight, inside_distance,  # noqa: E402
                                     op_dome, op_extrude, op_feather, op_offset, op_reset, quantize_mm,
                                     radial_coordinate)
from core.relief.provider import GeometryResult, MissingDependencyError  # noqa: E402
from core.relief.relief_stack import ReliefDims  # noqa: E402

PALETTE = ['Black', 'White', 'Pink', 'Red', 'Klein Blue']
R_MAX = 4.0


# --------------------------------------------------------------------------- helpers
def _ellipse(H, W, cx, cy, a, b):
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1.0


def _rim(mask):
    return mask & ~ndimage.binary_erosion(mask)


def _gradient_image(path, n=48):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    rgb = np.stack([0.9 * xx + 0.05, 0.6 * yy + 0.1, 0.9 * (1 - xx) * yy + 0.05], -1)
    rgb[: n // 4, : n // 4] = 0.0
    rgb[-n // 4:, -n // 4:] = 1.0
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
    return str(path)


def _ramp_depth(path, n=48):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    Image.fromarray(np.round(xx * 65535).astype(np.uint16)).save(path)
    return str(path)


def _sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


# --------------------------------------------------------------------------- dome
def test_dome_reaches_height_at_centre_and_meets_background_at_edge():
    H, W = 200, 240
    h = np.full((H, W), 0.5, np.float32)
    ell = _ellipse(H, W, 120, 100, 60, 40)
    out, info = op_dome(h, ell, 3.0, 1.0, 0.0, R_MAX)
    assert out.dtype == np.float32
    assert out[100, 120] == pytest.approx(3.0, abs=1e-5)                 # apex = requested height
    assert float(out.max()) <= 3.0 + 1e-5
    rim = _rim(ell)
    assert float(np.mean(out[rim] - 0.5)) < 0.2 * 2.5                      # rim close to the background
    assert float(np.max(out[rim] - 0.5)) < 0.3 * 2.5
    assert np.array_equal(out[~ell], h[~ell])                              # nothing outside the mask moves
    assert np.all(np.diff(out[100, 60:121]) >= -1e-6) and np.all(np.diff(out[60:101, 120]) >= -1e-6)
    # hemisphere: h(u) = bg + (H - bg) * sqrt(1 - u^2) along the major axis (u = 0.5 at x = 150)
    assert out[100, 150] == pytest.approx(0.5 + 2.5 * np.sqrt(0.75), abs=0.03)
    assert info['centroid_xy'] == pytest.approx([120.0, 100.0], abs=0.6)
    # roundness: < 1 flatter (higher at u = 0.5), > 1 pointier
    flat, _ = op_dome(h, ell, 3.0, 0.5, 0.0, R_MAX)
    point, _ = op_dome(h, ell, 3.0, 2.0, 0.0, R_MAX)
    assert flat[100, 150] > out[100, 150] > point[100, 150]
    assert flat[100, 120] == pytest.approx(3.0, abs=1e-5) and point[100, 120] == pytest.approx(3.0, abs=1e-5)
    # sloped background: the rim follows the local background, the apex is still the requested height
    yy = np.mgrid[0:H, 0:W][0]
    hs = (0.5 + 1.5 * yy / (H - 1)).astype(np.float32)
    outs, _ = op_dome(hs, ell, 3.5, 1.0, 0.0, R_MAX)
    assert outs[100, 120] == pytest.approx(3.5, abs=1e-5)
    # the hemisphere rim is steep (vertical tangent), so the last pixel ring sits a little above the
    # local background: within a quarter of the dome's height above it
    assert float(np.mean(np.abs(outs[rim] - hs[rim]))) < 0.25 * float(np.mean(3.5 - hs[rim]))
    assert np.array_equal(outs[~ell], hs[~ell])
    # requested height above the budget is clamped
    hi, _ = op_dome(h, ell, 9.0, 1.0, 0.0, R_MAX)
    assert float(hi.max()) == pytest.approx(R_MAX)
    # edge feather keeps the rim at the current relief and blends inward
    fe, _ = op_dome(h, ell, 3.0, 1.0, 10.0, R_MAX)
    assert np.allclose(fe[rim], h[rim], atol=0.06) and fe[100, 120] == pytest.approx(3.0, abs=1e-5)


def test_dome_profile_and_radial_coordinate():
    u = np.linspace(0, 1, 11)
    p = dome_profile(u, 1.0)
    assert p[0] == pytest.approx(1.0) and p[-1] == pytest.approx(0.0)
    assert np.allclose(p, np.sqrt(1 - u ** 2), atol=1e-6)
    assert np.all(dome_profile(u, 0.5) >= p - 1e-6) and np.all(dome_profile(u, 2.0) <= p + 1e-6)
    ell = _ellipse(120, 160, 80, 60, 50, 30)
    uu, info = radial_coordinate(ell)
    assert uu.shape == ell.shape and float(uu[60, 80]) == pytest.approx(0.0, abs=0.02)
    assert float(uu[60, 105]) == pytest.approx(0.5, abs=0.03)      # x offset 25 on a = 50
    assert float(uu[45, 80]) == pytest.approx(0.5, abs=0.03)       # y offset 15 on b = 30
    assert np.all(uu[~ell] == 1.0) and float(uu[ell].max()) <= 1.0
    assert info['radius_px'][0] == pytest.approx(30.5, abs=1.5) and info['radius_px'][1] == pytest.approx(50.5, abs=1.5)


# --------------------------------------------------------------------------- extrude / offset / feather
def test_extrude_is_constant_inside_mask():
    H, W = 120, 160
    yy = np.mgrid[0:H, 0:W][0]
    h = (2.0 * yy / (H - 1)).astype(np.float32)
    ell = _ellipse(H, W, 80, 60, 40, 30)
    out = op_extrude(h, ell, 2.5, 0.0, R_MAX)
    assert np.unique(out[ell]).tolist() == [pytest.approx(2.5)]
    assert np.array_equal(out[~ell], h[~ell])
    assert float(op_extrude(h, ell, 7.0, 0.0, R_MAX)[ell].max()) == pytest.approx(R_MAX)
    # feathered: constant beyond the inner band, unchanged outside
    fe = op_extrude(h, ell, 2.5, 8.0, R_MAX)
    deep = inside_distance(ell) > 9.0
    assert np.allclose(fe[deep], 2.5, atol=1e-5) and np.array_equal(fe[~ell], h[~ell])
    assert not np.allclose(fe[_rim(ell)], 2.5)


def test_raise_lower_and_clipping():
    H, W = 100, 100
    h = np.full((H, W), 1.0, np.float32)
    box = np.zeros((H, W), bool)
    box[30:70, 20:80] = True
    up = op_offset(h, box, 1.25, 0.0, R_MAX)
    assert np.allclose(up[box], 2.25) and np.array_equal(up[~box], h[~box])
    assert np.allclose(op_offset(h, box, 5.0, 0.0, R_MAX)[box], R_MAX)          # clipped at the budget
    assert np.allclose(op_offset(h, box, -2.0, 0.0, R_MAX)[box], 0.0)           # clipped at the base
    fe = op_offset(h, box, 1.0, 6.0, R_MAX)
    assert np.allclose(fe[inside_distance(box) > 7.0], 2.0) and np.array_equal(fe[~box], h[~box])
    assert 1.0 < float(fe[_rim(box)].mean()) < 1.5
    # apply_op records the clamp in mm
    out, info = apply_op(h, h, {'op': 'offset', 'mask': box, 'delta_mm': 5.0}, 0.1, R_MAX)
    assert info['mm_after'] == [R_MAX, R_MAX] and info['pixels_changed'] == int(box.sum())


def test_feather_does_not_touch_pixels_far_from_the_outline():
    H, W = 140, 160
    h = np.full((H, W), 0.4, np.float32)
    ell = _ellipse(H, W, 80, 70, 45, 35)
    plateau = op_extrude(h, ell, 3.0, 0.0, R_MAX)
    width_px = 20.0
    fe = op_feather(plateau, ell, width_px, R_MAX)
    dist = np.where(ell, inside_distance(ell), inside_distance(~ell)) - 0.5
    far = dist > width_px / 2 + 1e-6
    assert np.array_equal(fe[far], plateau[far])
    assert np.count_nonzero(fe != plateau) > 0.5 * np.count_nonzero(~far)
    # the step across the outline is smaller after feathering
    row = 70
    assert float(np.abs(np.diff(fe[row])).max()) < float(np.abs(np.diff(plateau[row])).max())
    assert 0.4 - 1e-6 <= float(fe.min()) and float(fe.max()) <= 3.0 + 1e-6
    w = band_weight(ell, width_px)
    assert float(w[_rim(ell)].min()) > 0.9 and np.all(w[far] == 0)
    assert np.array_equal(op_feather(plateau, ell, 0.0, R_MAX), plateau)


def test_empty_mask_does_nothing():
    H, W = 60, 80
    h = (np.random.default_rng(0).random((H, W)) * 3).astype(np.float32)
    empty = np.zeros((H, W), bool)
    for op in ({'op': 'dome', 'height_mm': 3.0}, {'op': 'extrude', 'height_mm': 2.0}, {'op': 'offset', 'delta_mm': 1.0},
               {'op': 'feather', 'width_mm': 1.0}, {'op': 'reset'}):
        out, info = apply_op(h, h, dict(op, mask=empty), 0.1, R_MAX)
        assert np.array_equal(out, h) and info['noop'] and info['pixels_changed'] == 0
        out2, info2 = apply_op(h, h, dict(op, mask=None), 0.1, R_MAX)
        assert np.array_equal(out2, h) and info2['noop']
    assert feather_weight(empty, 5.0).sum() == 0 and band_weight(empty, 5.0).sum() == 0
    sess = EditSession(h, 0.1, R_MAX)
    info = sess.apply({'op': 'extrude', 'mask': empty, 'height_mm': 2.0})
    assert info['noop'] and sess.ops == [] and np.array_equal(sess.current, h)
    with pytest.raises(ValueError):
        apply_op(h, h, {'op': 'bogus', 'mask': empty}, 0.1, R_MAX)


def test_reset_restores_original_and_session_undo():
    H, W = 90, 110
    h = (np.linspace(0, 2, W)[None, :].repeat(H, 0)).astype(np.float32)
    ell = _ellipse(H, W, 55, 45, 30, 20)
    sess = EditSession(h, 0.1, R_MAX)
    sess.apply({'op': 'dome', 'mask': ell, 'height_mm': 4.0})
    sess.apply({'op': 'offset', 'mask': ell, 'delta_mm': -0.5})
    assert len(sess.ops) == 2 and not np.array_equal(sess.current, h)
    inner = _ellipse(H, W, 55, 45, 15, 10)
    sess.apply({'op': 'reset', 'mask': inner})
    assert np.array_equal(sess.current[inner], h[inner]) and not np.array_equal(sess.current[ell & ~inner], h[ell & ~inner])
    assert sess.undo() and len(sess.ops) == 2 and not np.array_equal(sess.current[inner], h[inner])
    assert sess.undo() and sess.undo() and not sess.undo()
    assert np.array_equal(sess.current, h)
    sess.apply({'op': 'extrude', 'mask': ell, 'height_mm': 1.0})
    sess.reset()
    assert sess.ops == [] and np.array_equal(sess.current, sess.original)
    assert np.array_equal(op_reset(sess.current + 1, h), h)


# --------------------------------------------------------------------------- quantisation / scripts
def test_quantize_roundtrip_and_script_json(tmp_path):
    steps = np.arange(51, dtype=np.int16)
    mm = (steps.astype(np.float64) * 0.08).astype(np.float32)
    assert np.array_equal(quantize_mm(mm, 0.08, 50), steps)
    assert quantize_mm(np.array([0.039, 0.0401, 0.13, 4.5, -1.0], np.float32), 0.08, 50).tolist() == [0, 1, 2, 50, 0]
    H, W = 64, 80
    ell = _ellipse(H, W, 40, 32, 20, 14)
    assert np.array_equal(decode_mask_png(encode_mask_png(ell)), ell)
    h = np.full((H, W), 0.3, np.float32)
    sess = EditSession(h, 0.1, R_MAX)
    sess.apply({'op': 'dome', 'mask': ell, 'height_mm': 3.0, 'roundness': 0.8, 'feather_mm': 0.4, 'label': 'ball'})
    sess.apply({'op': 'feather', 'mask': ell, 'width_mm': 1.0})
    script = sess.to_script(image='cover.jpg', width_mm=8.0, base_mm=0.8, depth={'source': 'depth-anything'})
    d = script.to_dict()
    assert d['format'] == 'relief-edits/1' and d['grid_hw'] == [H, W] and len(d['ops']) == 2
    assert d['ops'][0]['mask_png'] and 'mask' not in d['ops'][0] and d['summary'][0].startswith('dome to 3 mm')
    path = script.save(str(tmp_path / 'e.json'))
    back = EditScript.load(path)
    assert len(back) == 2 and back.grid_hw == (H, W) and back.meta['width_mm'] == 8.0
    assert np.array_equal(back.ops[0]['mask'], ell) and back.ops[0]['label'] == 'ball'
    h2, infos = replay_ops(h, back.ops, 0.1, R_MAX)
    assert np.array_equal(h2, sess.current) and len(infos) == 2
    json.loads(json.dumps(d))                                            # plain JSON
    # replay on another grid: masks are resampled NEAREST, mm sizes stay
    ops2 = back.ops_resampled((32, 40))
    assert ops2[0]['mask'].shape == (32, 40) and 0.15 < ops2[0]['mask'].mean() < 0.4
    assert ops2[0]['feather_mm'] == 0.4
    assert EditScript.coerce(path).summary() == back.summary() and EditScript.coerce(d).grid_hw == (H, W)
    with pytest.raises(ValueError):
        EditScript.from_dict({'format': 'nope'})
    with pytest.raises(FileNotFoundError):
        EditScript.coerce(str(tmp_path / 'missing.json'))
    with pytest.raises(TypeError):
        EditScript.coerce(42)
    assert np.array_equal(resample_mask(ell, (H, W)), ell)
    ix, iy = grid_to_image_xy(10, 20, (100, 200), (50, 100))
    gx, gy = image_to_grid_xy(ix, iy, (100, 200), (50, 100))
    assert (gx, gy) == pytest.approx((10, 20))


# --------------------------------------------------------------------------- pipeline hook
def _ramp_field(H=60, W=90, dims=None, params=None):
    d = np.linspace(0, 1, W)[None, :].repeat(H, 0).astype(np.float32)
    dims = dims or ReliefDims()
    plaque = np.ones((H, W), bool)
    plaque[:5, :5] = False
    return process_depth(GeometryResult(depth=d), (H, W), dims, params or DepthParams(), plaque_mask=plaque), plaque


def test_apply_edit_script_keeps_unedited_pixels():
    field, plaque = _ramp_field()
    H, W = field.steps.shape
    # an empty script or no-op ops leave the steps identical
    empty = EditScript([], grid_hw=[H, W])
    f0, rep0 = apply_edit_script(field, empty, 0.1, plaque)
    assert np.array_equal(f0.steps, field.steps) and rep0['pixels_changed'] == 0 and rep0['n_ops'] == 0
    f1, rep1 = apply_edit_script(field, EditScript([{'op': 'offset', 'mask': np.zeros((H, W), bool), 'delta_mm': 1}]),
                                 0.1, plaque)
    assert np.array_equal(f1.steps, field.steps) and rep1['ops'][0]['noop']
    # a dome: only the (dilated) mask region changes; whole layers within the budget; cut-out stays 0
    ell = _ellipse(H, W, 45, 30, 18, 14)
    script = EditScript([{'op': 'dome', 'mask': ell, 'height_mm': 4.0}], grid_hw=[H, W], relief_mm=4.0)
    f2, rep2 = apply_edit_script(field, script, 0.1, plaque)
    changed = f2.steps != field.steps
    assert changed.any() and rep2['pixels_changed'] == int(changed.sum())
    region = ndimage.binary_dilation(ell, iterations=4)
    assert not changed[~region].any()
    assert f2.steps.dtype == np.int16 and 0 <= f2.steps.min() and f2.steps.max() <= 50
    assert int(f2.steps[30, 45]) == 50 and not f2.steps[~plaque].any()
    assert rep2['relief_mm_after'][1] == pytest.approx(4.0) and rep2['colour_untouched'] is True
    assert 'art_direction' in f2.report and f2.report['stats']['relief_layers_max'] == 50
    assert f2.mask is field.mask and f2.dims == field.dims
    # a different authoring grid is resampled with a warning; a different budget warns too
    f3, rep3 = apply_edit_script(field, EditScript([{'op': 'extrude', 'mask': _ellipse(30, 45, 22, 15, 9, 7),
                                                     'height_mm': 2.0}], grid_hw=[30, 45], relief_mm=6.0), 0.1, plaque)
    assert any('resampled' in w for w in rep3['warnings']) and any('budget' in w for w in rep3['warnings'])
    assert np.unique(f3.steps[_ellipse(H, W, 45, 30, 12, 9)]).tolist() == [25]
    # cleanup: a 1-px sliver in the mask does not survive the min-feature rule (but does with clean_px = 0)
    sliver = np.zeros((H, W), bool)
    sliver[30, 10:40] = True
    f4, rep4 = apply_edit_script(field, EditScript([{'op': 'extrude', 'mask': sliver, 'height_mm': 4.0}]), 0.1, plaque)
    assert rep4['cleanup_pixels_changed'] > 0 and int(f4.steps[30, 20]) < 50
    f5, _ = apply_edit_script(field, EditScript([{'op': 'extrude', 'mask': sliver, 'height_mm': 4.0}]), 0.1, plaque,
                              clean_px=0)
    assert int(f5.steps[30, 20]) == 50
    with pytest.raises(ValueError):
        apply_edit_script(field, script, 0.1, np.ones((3, 3), bool))


def test_pipeline_with_and_without_edits(tmp_path):
    from core.relief.pipeline import convert_album_relief
    img = _gradient_image(tmp_path / 'g48.png')
    dep = _ramp_depth(tmp_path / 'ramp.png')
    kw = dict(width_mm=4.8, depth_map=dep, palette=PALETTE, quantize_colors=12, seed=0, verify='no')
    plain = convert_album_relief(img, out_dir=str(tmp_path / 'plain'), **kw)
    empty = convert_album_relief(img, out_dir=str(tmp_path / 'empty'), height_edits=EditScript([]), **kw)
    assert _sha(plain['threemf']) == _sha(empty['threemf'])                      # unedited relief unchanged
    assert np.array_equal(plain['steps'], empty['steps']) and empty['art_direction'] is None
    assert empty['edits_json'] is None and not os.path.exists(os.path.join(tmp_path, 'empty', 'g48', 'g48_relief_edits.json'))
    H, W = plain['steps'].shape
    ell = _ellipse(H, W, W // 2, H // 2, W // 4, H // 5)
    script = EditScript([{'op': 'dome', 'mask': ell, 'height_mm': 4.0, 'feather_mm': 0.1, 'label': 'ball'}],
                        grid_hw=[H, W], image=img, width_mm=4.8, relief_mm=4.0)
    ed = convert_album_relief(img, out_dir=str(tmp_path / 'edit'), height_edits=script, verify='yes',
                              **{k: v for k, v in kw.items() if k != 'verify'})
    assert _sha(ed['threemf']) != _sha(plain['threemf'])
    assert ed['material_matrix_sha256'] == plain['material_matrix_sha256']      # colours untouched
    assert ed['palette'] == plain['palette'] and ed['backing'] == plain['backing']
    changed = ed['steps'] != plain['steps']
    assert changed.any() and not changed[~ndimage.binary_dilation(ell, iterations=4)].any()
    # apex 50 layers; on this 48 px grid the 50-layer plateau is ~5 x 3 px, so the min-feature
    # rule (4 px) may take one layer off it
    assert int(ed['steps'][H // 2, W // 2]) in (49, 50) and ed['art_direction']['cleanup_px'] == 4
    assert ed['voxel_check']['ok'] and ed['verification']['ok']
    assert ed['stats']['total_print_layers'] <= 65
    assert ed['art_direction']['n_ops'] == 1 and ed['art_direction']['pixels_changed'] == int(changed.sum())
    assert os.path.isfile(ed['edits_json'])
    rec = json.load(open(ed['recipe_json']))
    assert rec['art_direction']['summary'][0].startswith('dome to 4 mm') and rec['settings']['height_edits'] == ed['edits_json']
    assert rec['colour']['depth_independent'] is True
    # the saved script replays to the same 3MF (dict / path forms)
    again = convert_album_relief(img, out_dir=str(tmp_path / 'again'), height_edits=ed['edits_json'], **kw)
    assert np.array_equal(again['steps'], ed['steps'])
    # the flat Stack5 pipeline is not involved at all: same recipe as flat for this image
    from core.stack5.pipeline import convert_album_stack5
    flat = convert_album_stack5(img, width_mm=4.8, palette=PALETTE, quantize_colors=12, out_dir=str(tmp_path / 'flat'), seed=0)
    assert flat['stats']['material_matrix_sha256'] == ed['material_matrix_sha256']


def test_relief_for_editor_matches_pipeline_original(tmp_path):
    from core.relief.pipeline import convert_album_relief
    img = _gradient_image(tmp_path / 'g48.png')
    dep = _ramp_depth(tmp_path / 'ramp.png')
    base = relief_for_editor(img, 4.8, geometry='imported-depth', depth_map=dep)
    res = convert_album_relief(img, width_mm=4.8, depth_map=dep, palette=PALETTE, quantize_colors=12, seed=0, verify='no',
                               out_dir=str(tmp_path / 'p'))
    assert base['grid_hw'] == res['steps'].shape and base['pixel_mm'] == pytest.approx(0.1)
    assert np.array_equal(quantize_mm(base['original_mm'], 0.08, 50), res['steps'])
    assert np.array_equal(base['original_mm'], original_mm_from_field(base['field']))
    assert base['rgb_grid'].shape == res['steps'].shape + (3,) and base['plaque_mask'].all()
    assert base['relief_max_mm'] == pytest.approx(4.0) and base['image_hw'] == (48, 48)
    # the cached geometry can be handed back to the pipeline
    res2 = convert_album_relief(img, width_mm=4.8, geometry_result=base['geom'], palette=PALETTE, quantize_colors=12,
                                seed=0, verify='no', out_dir=str(tmp_path / 'p2'))
    assert np.array_equal(res2['steps'], res['steps'])


# --------------------------------------------------------------------------- masks: SAM + brush
def test_sam_provider_reports_missing_weights_without_downloading(tmp_path):
    from core.relief.segment import SamMaskProvider, sam_dependency_status, sam_status
    prov = SamMaskProvider(weights=str(tmp_path / 'no-sam-here'), allow_download=False)
    ok, msg = prov.available()
    assert not ok and 'fetch-sam-model' in msg and not prov.loaded
    with pytest.raises(MissingDependencyError):
        prov.load()
    with pytest.raises(RuntimeError):
        prov.predict([(1, 1)], [1])
    ok2, msg2 = sam_status('nobody/no-such-sam-checkpoint')
    assert not ok2 and ('fetch-sam-model' in msg2 or 'pip install' in msg2)
    assert set(sam_dependency_status()) == {'torch', 'transformers'}


def _tiny_sam():
    try:
        import torch
        from transformers import (SamConfig, SamImageProcessor, SamMaskDecoderConfig, SamModel, SamProcessor,
                                  SamPromptEncoderConfig, SamVisionConfig)
    except Exception:  # noqa: BLE001
        pytest.skip('torch / transformers not installed')
    hid = 32
    cfg = SamConfig(vision_config=SamVisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=64,
                                                  patch_size=16, output_channels=hid, global_attn_indexes=[0], window_size=2,
                                                  mlp_dim=32, num_pos_feats=hid // 2),
                    prompt_encoder_config=SamPromptEncoderConfig(hidden_size=hid, image_size=64, patch_size=16,
                                                                 mask_input_channels=4),
                    mask_decoder_config=SamMaskDecoderConfig(hidden_size=hid, num_hidden_layers=1, num_attention_heads=2,
                                                             mlp_dim=32, iou_head_depth=1, iou_head_hidden_dim=16))
    torch.manual_seed(0)
    model = SamModel(cfg).eval()
    proc = SamProcessor(image_processor=SamImageProcessor(size={'longest_edge': 64}, pad_size={'height': 64, 'width': 64}))
    return model, proc


def test_sam_provider_api_with_tiny_random_model(tmp_path):
    from core.relief.segment import MaskProposal, SamMaskProvider
    model, proc = _tiny_sam()
    prov = SamMaskProvider(model=model, processor=proc, device='cpu')
    assert prov.available()[0] and prov.loaded and not prov.image_set
    rgb = np.zeros((50, 40, 3), np.uint8)
    rgb[10:30, 5:25] = 200
    prov.set_image(rgb)
    assert prov.image_set and prov.image_hw == (50, 40)
    prop = prov.predict([(10, 20), (35, 45)], [1, 0])
    assert isinstance(prop, MaskProposal) and prop.masks.shape == (3, 50, 40) and prop.masks.dtype == bool
    assert prop.scores.shape == (3,) and prop.points == [(10.0, 20.0), (35.0, 45.0)] and prop.labels == [1, 0]
    best = prop.choose('best')
    assert best.shape == (50, 40) and np.array_equal(best, prop.masks[prop.best])
    order = prop.by_size()
    assert prop.masks[order[0]].sum() <= prop.masks[order[-1]].sum()
    assert np.array_equal(prop.choose('large'), prop.masks[order[-1]]) and prop.choose(1).shape == (50, 40)
    with pytest.raises(ValueError):
        prop.choose('huge')
    grid = resample_mask(best, (100, 80))                                    # onto a relief grid
    assert grid.shape == (100, 80)
    # the same model saved to a directory loads through the normal (offline) path
    d = str(tmp_path / 'tiny-sam')
    model.save_pretrained(d)
    proc.save_pretrained(d)
    p2 = SamMaskProvider(weights=d, allow_download=False, device='cpu')
    assert p2.available()[0]
    p2.set_image(rgb)
    prop2 = p2.predict([(10, 20)], [1])
    assert prop2.masks.shape == (3, 50, 40)


def test_brush_helpers():
    from core.relief.segment import component_containing, fill_holes, paint_disc, paint_stroke
    m = np.zeros((60, 80), bool)
    paint_disc(m, 20, 30, 5)
    assert m[30, 20] and m[30, 25] and not m[30, 27] and int(m.sum()) == pytest.approx(np.pi * 25, rel=0.25)
    paint_stroke(m, (40, 10), (70, 10), 3, True)
    assert m[10, 55] and m[13, 55] and not m[15, 55]
    paint_disc(m, 20, 30, 2, False)
    assert not m[30, 20] and m[30, 24]
    c = component_containing(m, 55, 10)
    assert c[10, 55] and not c[30, 24]
    assert np.array_equal(component_containing(m, 0, 0), m)
    ring = np.zeros((30, 30), bool)
    ring[5:25, 5:25] = True
    ring[10:20, 10:20] = False
    assert fill_holes(ring)[15, 15]
    with pytest.raises(ValueError):
        paint_disc(np.zeros((4, 4), np.uint8), 1, 1, 1)


# --------------------------------------------------------------------------- object provider stub
def test_object_provider_stub_and_composite():
    from core.relief.object_provider import (OBJECT_PROVIDERS, ObjectCrop, ObjectGeometry, StableFast3DProvider,
                                             composite_object, crop_object, front_depth, get_object_provider,
                                             object_profile)
    prov = StableFast3DProvider()
    ok, msg = prov.available()
    assert not ok and 'not integrated' in msg and 'stable-fast-3d' in OBJECT_PROVIDERS
    assert isinstance(get_object_provider('stable-fast-3d'), StableFast3DProvider)
    with pytest.raises(ValueError):
        get_object_provider('nope')
    H, W = 80, 100
    rgb = np.full((H, W, 3), 90, np.uint8)
    ell = _ellipse(H, W, 50, 40, 20, 15)
    crop = crop_object(rgb, ell, 0.1, margin_px=4, source_image='x.jpg')
    assert isinstance(crop, ObjectCrop) and crop.rgba.shape[2] == 4 and crop.bbox == (26, 21, 75, 60)
    assert crop.rgba[..., 3].astype(bool).sum() == ell.sum() and crop.size_mm == pytest.approx((4.9, 3.9))
    with pytest.raises(MissingDependencyError):
        prov.generate(crop)
    # a depth-map geometry (synthetic hemisphere, larger = closer) composites like a dome
    h_, w_ = crop.mask.shape
    yy, xx = np.mgrid[0:h_, 0:w_]
    dome = np.sqrt(np.clip(1 - ((xx - (50 - 26)) / 20.0) ** 2 - ((yy - (40 - 21)) / 15.0) ** 2, 0, 1)).astype(np.float32)
    geom = ObjectGeometry(depth=dome, hit=crop.mask, source='synthetic')
    h = np.full((H, W), 0.5, np.float32)
    out, prof = composite_object(h, ell, crop, geom, 3.0, 0.0, R_MAX)
    assert prof.shape == (H, W) and float(prof.max()) == pytest.approx(1.0) and np.all(prof[~ell] == 0)
    assert out[40, 50] == pytest.approx(3.0, abs=0.05) and np.array_equal(out[~ell], h[~ell])
    assert float(out[ell].min()) >= 0.5 - 1e-5
    # 'higher-is-farther' depth is flipped
    geom2 = ObjectGeometry(depth=-dome, hit=crop.mask, depth_convention='higher-is-farther')
    d2, hit2 = front_depth(geom2, crop)
    assert np.allclose(d2, dome) and hit2.shape == crop.mask.shape
    # a mesh is rasterised into the crop (icosphere -> hemisphere-like front depth)
    trimesh = pytest.importorskip('trimesh')
    sph = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    geom3 = ObjectGeometry(mesh=(np.asarray(sph.vertices), np.asarray(sph.faces)), source='mesh')
    d3, hit3 = front_depth(geom3, crop)
    assert hit3.shape == crop.mask.shape and hit3[40 - 21, 50 - 26] and float(d3[hit3].max()) == pytest.approx(1.0, abs=0.05)
    assert hit3.sum() == pytest.approx(ell.sum(), rel=0.2)
    prof3 = object_profile(d3, hit3, crop)
    assert prof3[40, 50] == pytest.approx(1.0, abs=0.05) and prof3[~ell].max() == 0
    with pytest.raises(Exception):
        ObjectGeometry()
    # the 'object' op in a script carries the profile
    sess = EditSession(h, 0.1, R_MAX)
    info = sess.apply({'op': 'object', 'mask': ell, 'profile': prof, 'height_mm': 2.0})
    assert not info['noop'] and sess.current[40, 50] == pytest.approx(2.0, abs=0.05)
    d = sess.to_script().to_dict()
    assert d['ops'][0]['profile_png'] and np.allclose(EditScript.from_dict(d).ops[0]['profile'], prof, atol=2e-5)


# --------------------------------------------------------------------------- CLI + editor
def test_cli_edits_and_check_deps(tmp_path):
    cli = os.path.join(REPO, 'scripts', 'convert_relief.py')
    r = subprocess.run([sys.executable, cli, '--check-deps'], cwd=REPO, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0
    st = json.loads(r.stdout)
    assert 'sam' in st and set(st['sam']) >= {'model', 'ready', 'status'}
    img = _gradient_image(tmp_path / 'g16.png', n=16)
    dep = _ramp_depth(tmp_path / 'r16.png', n=16)
    ell = _ellipse(16, 16, 8, 8, 5, 4)
    script = EditScript([{'op': 'extrude', 'mask': ell, 'height_mm': 2.0}], grid_hw=[16, 16], image=img, width_mm=1.6)
    ep = script.save(str(tmp_path / 'g16_edits.json'))
    out = str(tmp_path / 'g16_relief.3mf')
    r = subprocess.run([sys.executable, cli, img, '--depth-map', dep, '--width', '1.6', '--palette', 'Black,White',
                        '--quantize', '4', '--edits', ep, '--output', out, '--verify', 'yes'],
                       cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'art direction: 1 edit(s)' in r.stdout and os.path.isfile(out)
    rec = json.load(open(str(tmp_path / 'g16_relief_recipe.json')))
    assert rec['art_direction']['n_ops'] == 1 and os.path.isfile(str(tmp_path / 'g16_relief_edits.json'))
    # the image may come from the script
    r2 = subprocess.run([sys.executable, cli, '--depth-map', dep, '--width', '1.6', '--palette', 'Black,White',
                         '--quantize', '4', '--edits', ep, '--output', str(tmp_path / 'b' / 'g16.3mf'), '--verify', 'no'],
                        cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r2.returncode == 0 and 'image taken from the edit script' in r2.stdout
    r3 = subprocess.run([sys.executable, cli, img, '--depth-map', dep, '--edits', str(tmp_path / 'nope.json')],
                        cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r3.returncode == 2 and '--edits' in r3.stderr


def test_relief_editor_selftest():
    r = subprocess.run([sys.executable, os.path.join(REPO, 'scripts', 'relief_editor.py'), '--selftest'], cwd=REPO,
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and ('TclError' in r.stderr or 'no display' in r.stderr.lower()):
        pytest.skip('no display for Tk')
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'selftest ok' in r.stdout
