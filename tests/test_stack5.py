"""Stack5 mode: LUT synthesis, mode registration, palette search (Lumina metric +
backing search), flush/tower planning, min-region cleanup, CLI smoke."""
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.band.optics import (BeerLambertModel, Filament, hex_to_rgb01,  # noqa: E402
                              load_filament_library, srgb_to_lab_d65)
from core.stack5.lut import (FORBIDDEN_KEY_SUBSTRINGS, choose_backing, enumerate_stacks,  # noqa: E402
                             is_pure, pure_stack_indices, register_stack5_mode, save_lut_npz,
                             synth_lut, synth_lut_linear)
from core.stack5.palette import (_Bins, _score, evaluate_palette, image_hist,  # noqa: E402
                                 lab_true_to_uint8, select_palette)

LIB_JSON = os.path.join(REPO, 'assets', 'filaments_user.json')
MEASURED_JSON = os.path.join(REPO, 'assets', 'filaments_user_measured.json')
ALBUM_DIR = '/Users/hans_vador/Downloads/BambuAlbums/tools/albumart'


@pytest.fixture(scope='module')
def library():
    return load_filament_library(LIB_JSON)


@pytest.fixture(scope='module')
def measured():
    return load_filament_library(MEASURED_JSON if os.path.isfile(MEASURED_JSON) else LIB_JSON)


def _lab(hex_str):
    return srgb_to_lab_d65(hex_to_rgb01(hex_str)[None])[0]


# --------------------------------------------------------------------------- stacks
def test_enumerate_stacks_shape_and_pure():
    st = enumerate_stacks(5, 5)
    assert st.shape == (3125, 5) and st.dtype == np.int32
    assert np.unique(st, axis=0).shape[0] == 3125
    assert st[0].tolist() == [0, 0, 0, 0, 0]
    assert st[1].tolist() == [0, 0, 0, 0, 1]        # Lumina's MSB-first digit rule
    rows = pure_stack_indices(5, 5)
    assert rows.tolist() == [0, 781, 1562, 2343, 3124]
    for i, r in enumerate(rows):
        assert st[r].tolist() == [i] * 5
    assert is_pure(st).sum() == 5
    assert enumerate_stacks(3, 4).shape == (81, 4)


# --------------------------------------------------------------------------- synth LUT
def test_synth_lut_pure_stack_colours(library):
    fils = [library[n] for n in ('Black', 'White', 'Pink', 'Red', 'Klein Blue')]
    b = choose_backing(fils)
    assert fils[b].name == 'White'
    rgb, stacks = synth_lut(fils, b, BeerLambertModel())
    assert rgb.shape == (3125, 3) and rgb.dtype == np.uint8
    assert stacks.shape == (3125, 5) and stacks.dtype == np.int32
    rows = pure_stack_indices(5, 5)
    d_black = np.linalg.norm(srgb_to_lab_d65(rgb[rows[0]:rows[0] + 1] / 255.0)[0] - _lab('#000000'))
    assert d_black < 6.0
    assert rgb[rows[1]].tolist() == [255, 255, 255]
    L_pink = srgb_to_lab_d65(rgb[rows[2]:rows[2] + 1] / 255.0)[0, 0]
    assert L_pink > _lab('#F5A3B7')[0]
    L_red = srgb_to_lab_d65(rgb[rows[3]:rows[3] + 1] / 255.0)[0, 0]
    assert L_red > _lab('#C12E1F')[0]
    top_black = stacks[:, 0] == 0
    L_top_black = srgb_to_lab_d65(rgb[top_black] / 255.0)[:, 0]
    assert L_top_black.max() < 30.0
    assert L_top_black.min() == 0.0


def test_synth_lut_backing_matters(library):
    fils = [library[n] for n in ('Black', 'White', 'Pink', 'Red', 'Klein Blue')]
    rgb_w, _ = synth_lut(fils, 1, BeerLambertModel())
    rgb_k, _ = synth_lut(fils, 0, BeerLambertModel())
    rows = pure_stack_indices(5, 5)
    assert rgb_k[rows[2]].astype(int).sum() < rgb_w[rows[2]].astype(int).sum()
    assert rgb_k[rows[0]].tolist() == [0, 0, 0]
    # pure Pink over a Pink backing IS the spool colour
    rgb_p, _ = synth_lut(fils, 2, BeerLambertModel())
    assert rgb_p[rows[2]].tolist() == [0xF5, 0xA3, 0xB7]


# --------------------------------------------------------------------------- mode + processor
def test_register_mode_and_processor_loads(library, tmp_path):
    from config import ColorSystem
    from core.image_processing import LuminaImageProcessor
    fils = [library[n] for n in ('Black', 'White', 'Pink', 'Red', 'Klein Blue')]
    key = register_stack5_mode(fils)
    assert key.startswith('Stack5:') and len(key) == len('Stack5:') + 8
    assert not any(s in key for s in FORBIDDEN_KEY_SUBSTRINGS)
    conf = ColorSystem.get(key)
    assert conf['name'] == 'Stack5' and conf['layer_count'] == 5
    assert conf['slots'] == ['Black', 'White', 'Pink', 'Red', 'Klein Blue']
    assert conf['map'] == {'Black': 0, 'White': 1, 'Pink': 2, 'Red': 3, 'Klein Blue': 4}
    assert conf['preview'][2] == [0xF5, 0xA3, 0xB7, 255]
    assert ColorSystem.get('RYBW') is ColorSystem.RYBW
    assert ColorSystem.get('6-Color (Smart 1296)') is ColorSystem.SIX_COLOR

    rgb, stacks = synth_lut(fils, 1, BeerLambertModel())
    lut = save_lut_npz(str(tmp_path / 'lut.npz'), rgb, stacks, {'palette': [f.name for f in fils]})
    assert os.path.isfile(lut) and os.path.isfile(lut + '.json')
    proc = LuminaImageProcessor(lut, key)
    assert proc.layer_count == 5
    assert proc.ref_stacks.shape == (3125, 5)
    assert proc.lut_rgb.shape == (3125, 3)
    assert proc.kdtree is not None and proc.lut_lab.shape == (3125, 3)
    from core.converter import _get_actual_lut_slot_colors, detect_lut_color_mode
    cols = _get_actual_lut_slot_colors(proc)
    assert set(cols) == {0, 1, 2, 3, 4} and cols[0] == (0, 0, 0) and cols[1] == (255, 255, 255)
    assert detect_lut_color_mode(lut) == 'Merged'


# --------------------------------------------------------------------------- matching metric
def test_lumina_metric_reproduces_processor_argmin(measured, tmp_path):
    """The palette search's nearest-LUT pick must be the entry Lumina's KDTree
    (OpenCV 8-bit Lab, L* x 2.55) picks - the true-Lab metric is not."""
    from core.image_processing import LuminaImageProcessor
    fils = [measured[n] for n in ('Black', 'White', 'Klein Blue', 'Red', 'Vivid Yellow')]
    rgb, stacks = synth_lut(fils, 1, BeerLambertModel())
    lut = save_lut_npz(str(tmp_path / 'l.npz'), rgb, stacks)
    proc = LuminaImageProcessor(lut, register_stack5_mode(fils))
    cols = np.random.default_rng(1).integers(0, 256, (300, 3)).astype(np.uint8)
    _, idx_lumina = proc.kdtree.query(proc._rgb_to_lab(cols))
    lab_true = srgb_to_lab_d65(cols / 255.0)
    assert (lab_true_to_uint8(lab_true) == cols).all()          # bins round-trip exactly
    hist = (lab_true, np.ones(len(cols)) / len(cols))
    lin = synth_lut_linear(np.array([f.rgb_lin for f in fils])[None], np.array([f.td_mm for f in fils])[None],
                           np.array([1]), BeerLambertModel(), stacks, 0.08)
    idx = _score(lin, _Bins(hist, 0.0, 'lumina', 1.0), is_pure(stacks), 0, 0, 0)['idx'][0]
    agree = (idx == idx_lumina).mean()
    assert agree > 0.75
    diff = idx != idx_lumina                                    # only exact ties may differ
    q = proc._rgb_to_lab(cols)
    d_ours = np.linalg.norm(proc.lut_lab[idx[diff]] - q[diff], axis=1)
    d_lum = np.linalg.norm(proc.lut_lab[idx_lumina[diff]] - q[diff], axis=1)
    assert np.allclose(d_ours, d_lum, atol=1e-9)
    idx_lab = _score(lin, _Bins(hist, 0.0, 'lab', 1.0), is_pure(stacks), 0, 0, 0)['idx'][0]
    assert (idx_lab == idx_lumina).mean() < agree - 0.3
    with pytest.raises(ValueError):
        _Bins(hist, 0.0, 'nope', 1.0)


# --------------------------------------------------------------------------- palette search
def _pink_black_white_image(path, n=128, pink=(0xF5, 0xA3, 0xB7)):
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[:] = pink
    img[: n // 3, :] = 0
    img[-n // 4:, :] = 255
    Image.fromarray(img).save(path)
    return str(path)


def test_select_palette_includes_pink_and_is_deterministic(library, tmp_path):
    from core.stack5.pipeline import apply_td_overrides
    lib_a = apply_td_overrides(library, td_overrides={'Pink': 1.5})
    p = _pink_black_white_image(tmp_path / 'pbw.png')
    hist = image_hist(p, k=16)
    names, rep = select_palette(hist, lib_a, BeerLambertModel())
    assert 2 <= len(names) <= 5 and len(set(names)) == len(names) and rep['n_slots'] == len(names)
    assert 'Pink' in names
    assert rep['n_subsets_evaluated'] == sum(rep['sizes_evaluated'].values()) > 0
    assert rep['n_configs_evaluated'] > rep['n_subsets_evaluated']      # every member tried as the backing
    assert set(rep['excluded_spools']).isdisjoint(names)
    assert rep['seconds'] < 120
    assert rep['best']['backing'] in names
    assert rep['metric'] == 'hue' and rep['best']['metric'] == 'hue'      # hue-first is the default
    assert set(rep['best']['backing_costs']) == set(names)
    assert rep['best']['cost'] == min(rep['best']['backing_costs'].values())
    names2, rep2 = select_palette(image_hist(p, k=16), lib_a, BeerLambertModel())
    assert names2 == names
    assert rep2['best']['cost'] == rep['best']['cost'] and rep2['best']['backing'] == rep['best']['backing']
    assert [t['filament_names'] for t in rep2['top']] == [t['filament_names'] for t in rep['top']]
    # library TDs, image drawn in the printable pure-Pink-over-White colour
    fils = [library[n] for n in ('Black', 'White', 'Pink')]
    rgb, _ = synth_lut(fils, 1, BeerLambertModel())
    printable_pink = tuple(int(c) for c in rgb[pure_stack_indices(3, 5)[2]])
    assert printable_pink[0] > 240 and printable_pink[1] > 200
    p2 = _pink_black_white_image(tmp_path / 'pbw2.png', pink=printable_pink)
    names_b, rep_b = select_palette(image_hist(p2, k=16), library, BeerLambertModel())
    assert 'Pink' in names_b
    assert rep_b['best']['pure_stack_share_by_filament']['Pink'] > 0.3
    # must_include / fixed backing constraints
    n3, r3 = select_palette(hist, library, BeerLambertModel(), must_include=['Oak'], backing='Beige')
    assert 'Oak' in n3 and 'Beige' in n3 and r3['best']['backing'] == 'Beige'
    assert r3['backing_rule'] == 'fixed: Beige' and r3['n_configs_evaluated'] == r3['n_subsets_evaluated']
    with pytest.raises(KeyError):
        select_palette(hist, library, BeerLambertModel(), must_include=['Nope'])
    with pytest.raises(ValueError):
        select_palette(hist, library, BeerLambertModel(), metric='hsv')


def _pink50_image(path, n=128):
    """50 % exact Pink spool colour, 25 % black, 25 % white (finding 2)."""
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[:] = (0xF5, 0xA3, 0xB7)
    img[: n // 4] = 0
    img[-n // 4:] = 255
    Image.fromarray(img).save(path)
    return str(path)


def test_backing_is_searched_for_exact_spool_colour(measured, tmp_path):
    """Only the backing filament's pure stack shows a spool colour exactly: with
    the backing part of the search, exact-pink art gets a Pink backing."""
    hist = image_hist(_pink50_image(tmp_path / 'pink50.png'), k=16)
    names, rep = select_palette(hist, measured, BeerLambertModel(), top_k=3)
    best = rep['best']
    assert 'Pink' in names and best['backing'] == 'Pink'
    assert best['pure_stack_share_by_filament']['Pink'] > 0.4
    assert best['dominant_exact_spool_share'] > 0.6
    assert best['backing_costs']['Pink'] < min(v for k, v in best['backing_costs'].items() if k != 'Pink')
    assert rep['backing_rule'].startswith('searched')
    # the same subset scored with White forced as backing is strictly worse
    ev_w = evaluate_palette(hist, names, measured, backing='White', model=BeerLambertModel())
    assert ev_w['backing'] == 'White' and ev_w['cost'] > best['cost']
    ev = evaluate_palette(hist, names, measured, model=BeerLambertModel())
    assert ev['backing'] == 'Pink' and ev['cost'] == pytest.approx(best['cost'])
    assert set(ev['backing_costs']) == set(names)
    with pytest.raises(KeyError):
        evaluate_palette(hist, names, measured, backing='Oak', model=BeerLambertModel())


# --------------------------------------------------------------------------- flush / tower
def test_bambu_flush_formula_and_tower_plan():
    from core.band.writer3mf import load_template
    from core.stack5 import flush as fl
    # FlushVolCalc::calc_flush_vol_rgb reference values (no minimum)
    assert fl.bambu_flush_volume('#000000', '#FFFFFF') == 559
    assert fl.bambu_flush_volume('#000000', '#F4E62A') == 606
    assert fl.bambu_flush_volume('#1E44BE', '#F4E62A') == 565
    assert fl.bambu_flush_volume('#C12E1F', '#FFFFFF') == 504
    assert fl.bambu_flush_volume('#FFFFFF', '#000000') == 79
    assert fl.bambu_flush_volume('#FFFFFF', '#FFFFFF') == 0
    assert fl.bambu_flush_volume('#000000', '#FFFFFF', min_flush=48) == 607
    assert fl.bambu_flush_volume('#000000', '#FFFFFF', min_flush=48, scale=2.0) == 900      # cap
    tpl = load_template(None)['base']
    assert fl.min_flush_from_config(tpl) == 48.0        # nozzle_volume 92 - pi*1.75^2/4*18
    hexes = ['#000000', '#FFFFFF', '#1E44BE', '#C12E1F', '#F4E62A']
    M = fl.flush_matrix(hexes, 48)
    assert M.shape == (5, 5) and (np.diagonal(M) == 0).all() and M[0, 1] == 607 and M[1, 0] == 127
    assert len(fl.flatten_matrix(M)) == 50
    # ordering: light -> dark inside a layer, one dark -> light transition per layer
    sets = [[0, 1, 2, 3, 4]] * 5 + [[1]] * 20
    orders = fl.plan_layer_orders(sets, M)
    assert orders[0]['order'] == [1, 4, 3, 2, 0] and orders[0]['total_mm3'] == 792
    assert orders[1]['order'][0] == 0 and orders[1]['n_changes'] == 4 and orders[1]['purges'][0] == 0
    assert orders[5]['order'] == [1] and orders[5]['n_changes'] == 1 and orders[5]['total_mm3'] == M[orders[4]['order'][-1], 1]
    assert orders[6]['total_mm3'] == 0 and orders[6]['n_changes'] == 0
    plan = fl.plan_tower(hexes, sets, 150, 150, 0.08, min_flush=48)
    t = plan['tower']
    assert t['rib_wall'] is False and t['infill_gap_pct'] == 100 and t['brim'] == 0
    assert t['x'] == 156.5 and t['y'] == 15 and t['width'] == 84.5
    assert t['x'] + t['width'] + fl.BBS_WIPE_TOWER_MARGIN <= fl.BED_MM
    assert t['x'] >= 3 + 150 + fl.SKIRT_OUT_MM + fl.TOWER_CLEARANCE_MM
    assert t['fits'] and 150 < t['depth_modelled'] < t['depth_available'] <= 241
    assert max(plan['purge_per_layer_mm3']) == 1281 and plan['tool_changes'] == 21
    assert plan['layout']['plaque_xy'] == [3.0, 53.0]
    keys = fl.tower_config_keys(plan)
    assert keys['prime_tower_rib_wall'] == '0' and keys['prime_tower_infill_gap'] == '100%'
    assert keys['prime_tower_width'] == '84.5' and keys['wipe_tower_x'] == ['156.5', '156.5']
    assert keys['flush_volumes_matrix'][1] == '607' and len(keys['flush_volumes_matrix']) == 50
    # depth model: rectangle, one line per inner width
    td = fl.tower_depth([[1000.0]], 84.5, 0.08)
    assert td['inner_width_mm'] == 83.5 and td['perimeter_width_mm'] == 0.5
    assert td['depth_mm'] == pytest.approx(np.ceil(1000 / (0.5 * 0.08) / 83.5) * 0.5 + 0.5)
    # does not fit -> reported; too wide a plaque -> no room for a tower
    big = fl.plan_tower(hexes, sets, 150, 150, 0.08, min_flush=48, scale=2.0)
    assert big['tower']['fits'] is False
    with pytest.raises(ValueError):
        fl.plan_tower(hexes, sets, 225, 150, 0.08)
    # small plaque: tower falls back to the full-width bar at the BBS margin
    small = fl.plan_tower(hexes, sets, 6.4, 6.4, 0.08)
    assert small['tower']['x'] == 15 and small['tower']['width'] == 226


def test_first_layer_must_equal_layer_height(tmp_path):
    from core.stack5.pipeline import convert_album_stack5, stack5_print_overrides
    ov = stack5_print_overrides(0.08, 0.08)
    assert ov['initial_layer_print_height'] == ov['layer_height'] == '0.08'
    with pytest.raises(ValueError):
        stack5_print_overrides(0.16, 0.08)
    img = _gradient_image(tmp_path / 'g.png', n=16)
    with pytest.raises(ValueError):
        convert_album_stack5(img, width_mm=1.6, palette=['Black', 'White'], first_layer_mm=0.16,
                             out_dir=str(tmp_path / 'o'))
    assert not (tmp_path / 'o').exists()                        # rejected before any work


# --------------------------------------------------------------------------- cleanup / dilation
def test_min_region_cleanup_and_resync():
    from core.stack5.cleanup import min_region_cleanup, region_stats, resync_matched_rgb
    rng = np.random.default_rng(0)
    H = W = 200
    mm = np.zeros((H, W, 5), int)
    mm[:, :100] = 1
    mm[100:, :] += 2
    noise = rng.random((H, W)) < 0.01
    mm[noise, 0] = 4                     # 1-px islands (layer 0)
    mm[10:13, 10:13, 2] = 3              # 9 px island (< 16) on layer 2
    mm[50:56, 50:56, 2] = 3              # 36 px island (kept)
    mask = np.ones((H, W), bool)
    mask[:5, :5] = False
    mm[~mask] = -1
    before = region_stats(mm, mask, 16)
    assert before['islands_per_layer'][0] > 300 and before['small_share_per_layer'][0] > 0.005
    new, st = min_region_cleanup(mm, mask, 16)
    after = region_stats(new, mask, 16)
    assert after['islands_per_layer'] == [4, 4, 5, 4, 4] and max(after['small_share_per_layer']) == 0
    assert (new[:, :, 0] != 4).all() and (new[10:13, 10:13, 2] != 3).all() and (new[50:56, 50:56, 2] == 3).all()
    assert (new[~mask] == -1).all()
    assert st['reassigned_px_total'] == int(noise[mask].sum()) + 9
    same, st0 = min_region_cleanup(mm, mask, 0)
    assert (same == mm).all() and st0['reassigned_px_total'] == 0
    stacks = enumerate_stacks(5, 5)
    lut = (np.arange(3125)[:, None] * np.array([1, 3, 7]) % 256).astype(np.uint8)
    out = resync_matched_rgb(new, mask, np.zeros((H, W, 3), np.uint8), lut, stacks)
    idx = new[mask].astype(np.int64) @ (5 ** np.arange(4, -1, -1))
    assert (out[mask] == lut[idx]).all() and (out[~mask] == 0).all()


def test_merge_layers_without_dilation_matches_masks():
    import cv2
    from core.mesh_generators import HighFidelityMesher
    from core.stack5.pipeline import _merge_layers_no_dilation
    vox = np.full((6, 20, 20), -1, int)
    vox[0, 5:10, 5:10] = 2
    vox[1, 5:10, 5:10] = 2
    vox[2, 5:12, 5:10] = 2
    vox[4:, :, :] = 1
    groups = _merge_layers_no_dilation(None, vox, 2)
    assert [(a, b) for a, b, _ in groups] == [(0, 1), (2, 2)]
    assert (groups[0][2] == (vox[0] == 2)).all() and (groups[1][2] == (vox[2] == 2)).all()
    dil = HighFidelityMesher()._merge_layers_with_dilation(vox, 2)
    assert dil[0][2].sum() == cv2.dilate((vox[0] == 2).astype(np.uint8), np.ones((3, 3), np.uint8)).sum() > groups[0][2].sum()
    g1 = _merge_layers_no_dilation(None, vox, 1)
    assert [(a, b) for a, b, _ in g1] == [(4, 5)] and g1[0][2].all()
    assert _merge_layers_no_dilation(None, vox, 3) == []


# --------------------------------------------------------------------------- CLI smoke
def _gradient_image(path, n=64):
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    rgb = np.stack([0.9 * xx + 0.05, 0.6 * yy + 0.1, 0.9 * (1 - xx) * yy + 0.05], -1)
    rgb[: n // 4, : n // 4] = 0.0
    rgb[-n // 4:, -n // 4:] = 1.0
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
    return str(path)


def _rasterize_parts(object_model: bytes, n_layers: int, px=0.1, lh=0.08):
    """Per printed layer, bit i set where part i covers the pixel (scanline over
    the X-perpendicular faces at mid layer height)."""
    starts = [m.start() for m in re.finditer(rb'<object ', object_model)] + [len(object_model)]
    occ = None
    for oi in range(len(starts) - 1):
        chunk = object_model[starts[oi]:starts[oi + 1]]
        vb, ve = chunk.find(b'<vertices>'), chunk.find(b'</vertices>')
        tb, te = chunk.find(b'<triangles>'), chunk.find(b'</triangles>')
        V = np.fromstring(re.sub(rb'[^0-9.\-]+', b' ', chunk[vb + 10:ve]), dtype=np.float64, sep=' ').reshape(-1, 3)
        T = np.fromstring(re.sub(rb'[^0-9]+', b' ', chunk[tb + 11:te]), dtype=np.int64, sep=' ').reshape(-1, 6)[:, [1, 3, 5]]
        if occ is None:
            W = int(round(V[:, 0].max() / px)); H = int(round(V[:, 1].max() / px))
            occ = np.zeros((n_layers, H, W), np.uint8)
        P0, P1, P2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        zmin = np.minimum(np.minimum(P0[:, 2], P1[:, 2]), P2[:, 2])
        zmax = np.maximum(np.maximum(P0[:, 2], P1[:, 2]), P2[:, 2])
        xperp = (P0[:, 0] == P1[:, 0]) & (P1[:, 0] == P2[:, 0])
        nx = np.cross(P1 - P0, P2 - P0)[:, 0]
        ymin = np.minimum(np.minimum(P0[:, 1], P1[:, 1]), P2[:, 1])
        ymax = np.maximum(np.maximum(P0[:, 1], P1[:, 1]), P2[:, 1])
        H, W = occ.shape[1:]
        for k in range(n_layers):
            zm = (k + 0.5) * lh
            sel = xperp & (zmin < zm) & (zmax > zm)
            if not sel.any():
                continue
            x = np.round(P0[sel, 0] / px).astype(int)
            y0 = np.round(ymin[sel] / px).astype(int)
            y1 = np.round(ymax[sel] / px).astype(int)
            s = np.where(nx[sel] < 0, 1, -1).astype(np.int32)
            E = np.zeros((H + 1, W + 1), np.int32)
            np.add.at(E, (y0, x), s)
            np.add.at(E, (y1, x), -s)
            filled = np.cumsum(np.cumsum(E, axis=0)[:H], axis=1)[:, :W] >= 1
            occ[k][filled] |= np.uint8(1 << oi)
    return occ


def test_cli_smoke_64px(tmp_path):
    from core.band.writer3mf import build_project_settings, classify_filament_keys, load_template
    img = _gradient_image(tmp_path / 'grad64.png')
    out = tmp_path / 'out'
    cmd = [sys.executable, os.path.join(REPO, 'scripts', 'lumina_album.py'), '--image', img,
           '--width', '6.4', '--palette', 'Black,White,Pink,Red,Klein Blue', '--quantize', '16',
           '--advisory', 'br', '--out', str(out), '--seed', '0']
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    assert '[STACK5] prime tower:' in r.stdout and 'fits' in r.stdout
    # the first-layer flag is gone: the viewing layer must be one colour layer thick
    r2 = subprocess.run(cmd + ['--first-layer', '0.16'], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert r2.returncode != 0
    job = out / 'grad64'
    threemf = job / 'grad64_stack5.3mf'
    recipe_p = job / 'grad64_recipe.json'
    assert threemf.is_file() and recipe_p.is_file()
    assert (job / 'grad64_preview.png').is_file() and (job / 'grad64_ams.txt').is_file()
    assert (job / 'grad64_stack5_lut.npz').is_file()
    recipe = json.loads(recipe_p.read_text())
    assert [p['name'] for p in recipe['palette']] == ['Black', 'White', 'Pink', 'Red', 'Klein Blue']
    assert recipe['backing']['name'] == recipe['palette_report']['best']['backing']
    assert recipe['stats']['resolution_px'] == [64, 64]
    assert recipe['settings']['advisory'] == 'br'
    assert recipe['settings']['metric'] == 'hue' and recipe['settings']['min_region_px'] == 16
    assert recipe['settings']['hue_params']['wL_chroma'] == 0.6 and 'hue-first' in recipe['settings']['pixel_matcher']
    assert recipe['settings']['dilate'] is False and recipe['stats']['mesh_dilated'] is False
    assert recipe['settings']['filaments_json'] == os.path.abspath(
        MEASURED_JSON if os.path.isfile(MEASURED_JSON) else LIB_JSON)
    assert recipe['settings']['scoring'] == {'chroma_weight': 0.0, 'spool_bonus': 10.0, 'spool_de': 10.0,
                                             'dominant_w': 0.01}
    assert [p['td_mm'] for p in recipe['palette']][:2] == [0.15, 5.0]
    assert recipe['flush']['flush_min_mm3'] == 48.0 and recipe['tower']['fits'] is True
    assert recipe['layout']['plaque_xy'] == [3.0, 124.8] and recipe['tower']['x'] == 15.0
    assert len(recipe['stats']['islands_per_optical_layer']) == 5
    with zipfile.ZipFile(threemf) as zf:
        names = set(zf.namelist())
        assert {'Metadata/model_settings.config', 'Metadata/project_settings.config',
                '3D/Objects/object_1.model', '3D/3dmodel.model'} <= names
        ms = zf.read('Metadata/model_settings.config').decode()
        cfg = json.loads(zf.read('Metadata/project_settings.config'))
        obj = zf.read('3D/Objects/object_1.model')
        main = zf.read('3D/3dmodel.model').decode()
    F = ms.count('<part ')
    assert F == recipe['stats']['n_objects'] == len(recipe['parts']) >= 2
    assert obj.count(b'<object ') == F
    assert obj.count(b'<triangle ') == recipe['stats']['triangles_total'] > 0
    assert len(cfg['filament_colour']) == F
    assert cfg['filament_colour'] == [p['hex'] for p in (
        [{'hex': dict((q['name'], q['hex']) for q in recipe['palette'])[pp['name']]} for pp in recipe['parts']])]
    assert cfg['printer_settings_id'] == 'Bambu Lab X2D 0.4 nozzle'
    assert cfg['print_settings_id'] == 'Stack5 0.08mm @BBL X2D'          # self-contained project preset
    assert cfg['inherits_group'][0] == '' and cfg['print_compatible_printers'] == ['Bambu Lab X2D 0.4 nozzle']
    assert cfg['print_extruder_id'] == ['1', '1', '1', '2', '2', '2']    # X2D dual-extruder variant lists
    assert cfg['filament_settings_id'] == ['Bambu PLA Basic @BBL X2D 0.4 nozzle'] * F
    assert cfg['layer_height'] == cfg['initial_layer_print_height'] == '0.08'
    assert cfg['wall_loops'] == '1' and cfg['top_shell_layers'] == '0' and cfg['bottom_shell_layers'] == '0'
    assert cfg['sparse_infill_density'] == '100%' and cfg['sparse_infill_pattern'] == 'zig-zag'
    # prime tower sized from the modelled purge, rectangular, dense
    assert cfg['enable_prime_tower'] == '1' and cfg['prime_tower_rib_wall'] == '0'
    assert cfg['prime_tower_infill_gap'] == '100%' and cfg['prime_tower_brim_width'] == '0'
    assert float(cfg['prime_tower_width']) == recipe['tower']['width']
    assert cfg['wipe_tower_x'] == [str(int(recipe['tower']['x']))] * 2 and cfg['wipe_tower_y'] == ['15', '15']
    # flush matrix: Bambu formula per pair, not flat
    assert len(cfg['flush_volumes_matrix']) == 2 * F * F and len(cfg['flush_volumes_vector']) == 2 * F
    mtx = np.array(cfg['flush_volumes_matrix'][:F * F], dtype=int).reshape(F, F)
    assert (np.diagonal(mtx) == 0).all() and (mtx[~np.eye(F, dtype=bool)] > 0).all()
    assert mtx.tolist() == recipe['flush']['flush_matrix']
    assert mtx[0, 1] == 607 and mtx[1, 0] == 127 and len(set(mtx[~np.eye(F, dtype=bool)].tolist())) > 2
    assert cfg['filament_self_index'] == [str(i + 1) for i in range(F)]
    # plaque placed by the build item transform (Bambu applies it as the instance offset) + assemble
    tf = recipe['threemf_check']['build_transform']
    assert tf == '1 0 0 0 1 0 0 0 1 3 124.8 0'
    assert f'<item objectid="{F + 1}" transform="{tf}" printable="1"/>' in main
    assert f'<assemble_item object_id="{F + 1}" instance_id="0" transform="{tf}" offset="0 0 0"/>' in ms
    tpl = load_template(None)
    fils = [Filament.from_hex(n, h, 1.0) for n, h in (('a', '#000000'), ('b', '#FFFFFF'), ('c', '#FF0000'))]
    refs = [build_project_settings(tpl, fils[:2]), build_project_settings(tpl, fils[:3])]
    mult = classify_filament_keys(refs + [cfg])
    assert mult == tpl['per_filament_multiplier']
    for k, m in mult.items():
        assert len(cfg[k]) == m * F, k
    dsts = cfg['different_settings_to_system'][0].split(';')
    assert dsts == sorted(dsts) and {'initial_layer_print_height', 'prime_tower_brim_width', 'prime_tower_width',
                                    'wall_loops', 'skirt_loops', 'bottom_shell_layers'} <= set(dsts)
    assert cfg['brim_type'] == 'no_brim' and cfg['prime_tower_brim_width'] == '0'
    assert len(cfg['different_settings_to_system']) == F + 2
    # first-layer adhesion (Sep 5 2026): slow, unfanned, 65 C bed, no elephant-foot compensation,
    # first layer still one 0.08 mm colour layer; filament keys listed per filament so Studio keeps them
    assert cfg['initial_layer_speed'] == ['18'] * 6 and cfg['initial_layer_infill_speed'] == ['20'] * 6
    assert cfg['elefant_foot_compensation'] == '0' and cfg['initial_layer_acceleration'] == ['500'] * 6
    assert {'initial_layer_speed', 'initial_layer_infill_speed', 'elefant_foot_compensation'} <= set(dsts)
    assert cfg['hot_plate_temp_initial_layer'] == ['65'] * F and cfg['textured_plate_temp_initial_layer'] == ['65'] * F
    assert cfg['hot_plate_temp'] == ['65'] * F and cfg['close_fan_the_first_x_layers'] == ['1'] * F
    assert cfg['first_x_layer_fan_speed'] == ['0'] * F and cfg['nozzle_temperature_initial_layer'] == ['225'] * F
    for i in range(1, F + 1):
        fd = set(cfg['different_settings_to_system'][i].split(';'))
        assert {'hot_plate_temp_initial_layer', 'close_fan_the_first_x_layers', 'nozzle_temperature_initial_layer'} <= fd
    assert cfg['different_settings_to_system'][F + 1] == ''
    assert recipe['threemf_check']['project_settings']['initial_layer_speed'] == ['18'] * 6
    # backing printed at 0.2 mm through a height-range modifier: colour 0..0.4 stays 0.08,
    # backing 0.4..2.0 (20 x 0.08 = 1.6 mm) becomes 8 layers of 0.2
    with zipfile.ZipFile(threemf) as zf:
        lcr = zf.read('Metadata/layer_config_ranges.xml').decode()
    assert '<object id="1">' in lcr and '<range min_z="0.4" max_z="2">' in lcr
    assert '<option opt_key="layer_height">0.2</option>' in lcr
    assert recipe['stats']['backing_layer_ranges'] == [[0.4, 2.0, 0.2]] and recipe['stats']['sliced_layers_est'] == 13
    assert recipe['settings']['backing_layer_h'] == 0.2
    exts = [int(x) for x in re.findall(r'key="extruder" value="(\d+)"', ms)]
    assert exts == list(range(1, F + 1))
    assert 'plater_name" value="grad64"' in ms
    # parts tile every layer exactly (no dilation): no pixel in two parts, none uncovered
    n_layers = recipe['stats']['total_print_layers']
    occ = _rasterize_parts(obj, n_layers)
    popcount = np.array([bin(v).count('1') for v in range(256)])
    for k in range(n_layers):
        assert (popcount[occ[k]] == 1).all(), f"layer {k}: overlap/gaps"
    shares = recipe['stats']['per_slot_viewing_surface_share']
    for i, part in enumerate(recipe['parts']):
        area = int(((occ[0] >> i) & 1).sum())
        assert area == round(shares[part['name']] * 64 * 64)


def test_backing_layer_ranges_rules():
    from core.stack5.pipeline import backing_layer_ranges, layer_config_ranges_xml
    assert backing_layer_ranges(25, 'single') == [(0.4, 2.0, 0.2)]
    assert backing_layer_ranges(30, 'double') == [(0.4, 2.0, 0.2)]          # colour on both faces
    assert backing_layer_ranges(25, 'single', backing_layer_h=None) == []
    assert backing_layer_ranges(25, 'single', backing_layer_h=0.08) == []   # no thicker than layer_h
    assert backing_layer_ranges(6, 'single') == []                           # backing thinner than 0.2
    xml = layer_config_ranges_xml([(0.4, 2.0, 0.2)])
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    obj = root.find('object')
    assert obj.get('id') == '1'
    rng = obj.find('range')
    assert (rng.get('min_z'), rng.get('max_z')) == ('0.4', '2')
    assert rng.find('option').get('opt_key') == 'layer_height' and rng.find('option').text == '0.2'


def test_pipeline_deterministic(tmp_path):
    from core.stack5.pipeline import convert_album_stack5
    img = _gradient_image(tmp_path / 'grad48.png', n=48)
    pal = ['Black', 'White', 'Pink', 'Red', 'Klein Blue']
    a = convert_album_stack5(img, width_mm=4.8, palette=pal, quantize_colors=12,
                             out_dir=str(tmp_path / 'a'), seed=0)
    b = convert_album_stack5(img, width_mm=4.8, palette=pal, quantize_colors=12,
                             out_dir=str(tmp_path / 'b'), seed=0)
    ra = json.loads(open(a['recipe_json']).read())
    rb = json.loads(open(b['recipe_json']).read())
    for r in (ra, rb):
        r.pop('timing'); r.pop('outputs'); r['stats'].pop('lumina_status')
        r['palette_report'].pop('seconds', None)
    assert ra['stats']['material_matrix_sha256'] == rb['stats']['material_matrix_sha256']
    assert ra == rb
    assert a['stats']['tool_changes_est'] >= 0
    assert a['stats']['n_unique_stacks_used'] >= 1
    assert a['tower']['fits'] and a['flush']['purge_total_mm3'] > 0


# --------------------------------------------------------------------------- default library / scoring
def test_default_library_is_measured_when_present():
    from core.stack5 import pipeline
    d = pipeline.default_filaments_json()
    if os.path.isfile(MEASURED_JSON):
        assert d == MEASURED_JSON
        lib = load_filament_library(d)
        assert list(lib) == list(load_filament_library(LIB_JSON))
        assert lib['Black'].td_mm == 0.15 and lib['White'].td_mm == 5.0 and lib['Pink'].td_mm == 4.5
    else:
        assert d == LIB_JSON
    assert pipeline.DEFAULT_FILAMENTS_JSON == d


def _spool_image(path, n=96):
    """Exact spool colours in large areas: Green (40 %), White (30 %), Black (30 %)."""
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[: int(0.4 * n), :] = (0x00, 0xAE, 0x42)
    img[int(0.4 * n): int(0.7 * n), :] = 255
    Image.fromarray(img).save(path)
    return str(path)


def test_spool_bonus_prefers_exact_spool_colours(measured, tmp_path):
    hist = image_hist(_spool_image(tmp_path / 'spools.png'), k=8)
    names, rep = select_palette(hist, measured, BeerLambertModel(), spool_bonus=10.0)
    best = rep['best']
    assert {'Green', 'White', 'Black'} <= set(names)
    assert best['backing'] == 'White'
    assert best['dominant_exact_spool_share'] > 0.9
    assert best['dominant_exact_spool_share_by_filament']['Green'] > 0.3
    assert abs(best['cost'] - (best['cost_fit'] - best['spool_bonus'])) < 1e-9
    assert best['spool_bonus'] > 0
    names0, rep0 = select_palette(hist, measured, BeerLambertModel(), spool_bonus=0.0)
    assert rep0['best']['spool_bonus'] == 0.0 and rep0['best']['cost'] == rep0['best']['cost_fit']
    assert rep0['scoring']['spool_bonus'] == 0.0
    names_c, rep_c = select_palette(hist, measured, BeerLambertModel(), chroma_weight=1.0)
    assert rep_c['best']['mean_dE_weighted'] > 0 and rep_c['scoring']['chroma_weight'] == 1.0


def test_prefer_is_soft_and_bounded(measured, tmp_path):
    hist = image_hist(_gradient_image(tmp_path / 'grad96.png', n=96), k=32)
    base, rep = select_palette(hist, measured, BeerLambertModel(), top_k=3)
    other = next(n for n in ('Oak', 'Pink', 'Beige', 'Coffee Brown', 'Lavender Purple') if n not in base)
    n0, r0 = select_palette(hist, measured, BeerLambertModel(), prefer=[other], prefer_tol=0.0, top_k=3)
    assert r0['best_unconstrained']['filament_names'] == base            # the plan itself never moves
    if r0['prefer']['applied']:                                          # only when the preferred palette is no worse
        assert other in n0 and r0['prefer']['excess_dE'] <= 1e-9
    else:
        assert n0 == base and r0['prefer']['rank'] > 1 and r0['prefer']['excess_dE'] > 0
        assert r0['best'] is r0['best_unconstrained']
    n1, r1 = select_palette(hist, measured, BeerLambertModel(), prefer=[other], prefer_tol=100.0, top_k=3)
    assert other in n1 and r1['prefer']['applied'] is True
    assert r1['best_unconstrained']['filament_names'] == base
    assert r1['prefer']['rank'] == r1['best']['rank']
    assert r1['prefer']['tolerance_dE'] == pytest.approx(100.0 * r1['best_unconstrained']['cost_dE'])
    assert r1['prefer']['backing'] == r1['best']['backing']
    tol = r0['prefer']['excess_ratio']
    if tol > 0:                                                          # the tolerance is a sharp boundary
        n2, r2 = select_palette(hist, measured, BeerLambertModel(), prefer=[other], prefer_tol=tol * 0.99, top_k=3)
        n3, r3 = select_palette(hist, measured, BeerLambertModel(), prefer=[other], prefer_tol=tol * 1.01, top_k=3)
        assert r2['prefer']['applied'] is False and r3['prefer']['applied'] is True and other in n3
    n2, r2 = select_palette(hist, measured, BeerLambertModel(), must_include=[other], top_k=3)
    assert other in n2
    with pytest.raises(KeyError):
        select_palette(hist, measured, BeerLambertModel(), prefer=['Nope'])
    with pytest.raises(ValueError):
        select_palette(hist, measured, BeerLambertModel(), must_include=['Black', 'White', 'Red'],
                       prefer=['Oak', 'Pink', 'Beige'])


@pytest.mark.skipif(not os.path.isfile(os.path.join(ALBUM_DIR, 'graduation.jpg')),
                    reason='album art not available')
def test_graduation_palette_regression():
    """Graduation through the colour plan: the sky is purple/magenta (Lavender
    Purple / Pink), the bear is white with black outlines, there is red and
    yellow, and no green worth a spool.  Deterministic; when Pink is a real
    pick --prefer Pink is a no-op, otherwise it is a soft request."""
    lib = load_filament_library(MEASURED_JSON)
    hist = image_hist(os.path.join(ALBUM_DIR, 'graduation.jpg'), k=64)
    names, rep = select_palette(hist, lib, BeerLambertModel(), top_k=5)
    plan = rep['hue_plan']
    assert len(names) == 5 and 'Green' not in names
    assert 'Lavender Purple' in names and 'White' in names
    assert plan['light_neutral_share'] > 0.05 and plan['dark_share'] > 0.04
    needed = [c['name'] for c in plan['clusters'] if c['needed']]
    assert any(k.startswith('purple') for k in needed) and any(('pink' in k) or ('red' in k) for k in needed)
    assert rep['best']['backing'] in names and rep['backing_rule'].startswith('searched')
    assert rep['n_slots'] == 5 and rep['sizes_evaluated']
    names2, rep2 = select_palette(hist, lib, BeerLambertModel(), top_k=5)
    assert names2 == names and rep2['best']['cost_total'] == rep['best']['cost_total']
    names_p, rep_p = select_palette(hist, lib, BeerLambertModel(), prefer=['Pink'], top_k=5)
    assert rep_p['prefer']['requested'] == ['Pink'] and rep_p['prefer']['applied'] == ('Pink' in names_p)
    assert rep_p['best_unconstrained']['filament_names'] == names          # the plan itself is unchanged

def test_hue_metric_prefers_hue_over_shade(measured):
    """Dark brown skin (#502107: L19 C33 h51) with Black/White/Coffee Brown/
    Lavender/Yellow: Lumina's 8-bit Lab metric picks a near-black olive stack
    (hue thrown away to keep the shade); the hue-first metric picks a brown
    stack (right hue, lighter shade).  Neutrals keep their lightness, the
    per-pixel matcher and the palette scorer agree, and params parse."""
    from core.stack5.metric import Stack5Matcher, parse_hue_params, resolve_hue_params, make_matcher
    fils = [measured[n] for n in ('Black', 'White', 'Coffee Brown', 'Lavender Purple', 'Vivid Yellow')]
    rgb, stacks = synth_lut(fils, 4, BeerLambertModel())            # backing Vivid Yellow
    lab_lut = srgb_to_lab_d65(rgb.astype(np.float64) / 255.0)
    lin = synth_lut_linear(np.array([f.rgb_lin for f in fils])[None], np.array([f.td_mm for f in fils])[None],
                           np.array([4]), BeerLambertModel(), stacks, 0.08)
    skin = np.array([[0x50, 0x21, 0x07]], np.uint8)
    hist_skin = (srgb_to_lab_d65(skin / 255.0), np.ones(1))

    def hue_deg(lab):
        return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360)

    i_hue = int(Stack5Matcher(rgb, 'hue').match_colors_batch(skin)[0])
    i_lum = int(_score(lin, _Bins(hist_skin, 0.0, 'lumina', 1.0), is_pure(stacks), 0, 0, 0)['idx'][0][0])
    i_hue_pal = int(_score(lin, _Bins(hist_skin, 0.0, 'hue', 1.0), is_pure(stacks), 0, 0, 0)['idx'][0][0])
    assert i_hue_pal == i_hue                                       # scorer and pixel matcher agree
    Lh, ah, bh = lab_lut[i_hue]
    assert 30 < hue_deg(lab_lut[i_hue]) < 80 and ah > 5 and np.hypot(ah, bh) > 12   # brown, chromatic
    assert stacks[i_hue][0] == 2                                    # Coffee Brown on the viewing surface
    assert lab_lut[i_lum][1] < 5                                    # Lumina's pick has lost the red/brown hue
    # neutrals keep their lightness (grey shirt #D4D7D6, L86)
    grey = np.array([[0xD4, 0xD7, 0xD6]], np.uint8)
    g = lab_lut[int(Stack5Matcher(rgb, 'hue').match_colors_batch(grey)[0])]
    assert abs(g[0] - 86) < 12 and np.hypot(g[1], g[2]) < 20
    # random colours: palette scorer argmin == pixel matcher argmin (ties aside)
    cols = np.random.default_rng(3).integers(0, 256, (200, 3)).astype(np.uint8)
    hist = (srgb_to_lab_d65(cols / 255.0), np.ones(len(cols)) / len(cols))
    idx_pal = _score(lin, _Bins(hist, 0.0, 'hue', 1.0), is_pure(stacks), 0, 0, 0)['idx'][0]
    idx_pix = Stack5Matcher(rgb, 'hue').match_colors_batch(cols)
    assert (idx_pal == idx_pix).mean() > 0.97
    # params
    assert parse_hue_params('wL=0.5,wH=2')['wL_chroma'] == 0.5 and parse_hue_params('wL=0.5,wH=2')['wH'] == 2.0
    assert resolve_hue_params(None)['wL_neutral'] == 1.0
    with pytest.raises(ValueError):
        parse_hue_params('foo=1')
    assert make_matcher('lumina', rgb) is None and make_matcher('lab', rgb, wL=2.0).metric == 'lab'
