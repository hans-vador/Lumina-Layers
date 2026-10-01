import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from scipy import ndimage
from core.band.optics import load_filament_library
from core.band.optics import linear_rgb_to_lab_d65, srgb_to_linear
from core.band.reference import (tone_height, convert_reference_band, column_layers, fit_swaps, _stack_colour,
                                 palette_filaments, remove_thin_features, complete_two_nozzle_process,
                                 auto_tone, toned_target, height_from_tone, choose_palette,
                                 DEFAULT_FILAMENTS_JSON, SYSTEM_PROCESS_JSON)


def test_tone_preserves_boundary_without_quantization():
    rgb=np.zeros((4,8,3),np.uint8);rgb[:,4:]=255
    h=tone_height(rgb)
    assert np.all(h[:,:4]==h[0,0]) and np.all(h[:,4:]==h[0,7])
    assert h[0,7]-h[0,0]>1.8
    ramp=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(2,1,3))
    assert np.all(np.diff(tone_height(ramp)[0])>0)


def test_repeatable_geometry_and_three_global_swaps(tmp_path):
    image=tmp_path/'input.png';Image.fromarray(np.tile(np.arange(32,dtype=np.uint8)[None,:,None]*8,(32,1,3))).save(image)
    # min_feature_mm=0: this test pins the raw tone rule; thin-feature removal is tested below
    results=[convert_reference_band(str(image),str(tmp_path/str(i)),width_mm=10,pitch_mm=.5,min_feature_mm=0,
                                    palette='graduation')
             for i in range(2)]
    reports=[json.load(open(r['recipe_json'])) for r in results]
    assert reports[0]['height_sha256']==reports[1]['height_sha256']
    assert Path(results[0]['threemf']).read_bytes()==Path(results[1]['threemf']).read_bytes()
    assert reports[0]['watertight'] and reports[0]['winding_consistent']
    assert reports[0]['border_mm'] == 0
    assert reports[0]['art_height_range_mm'][1] < 3
    heights = np.load(Path(results[0]['threemf']).with_name('input_reference_band_height.npy'))
    expected = tone_height(np.asarray(Image.open(image).resize((21,21),Image.Resampling.BILINEAR))) + .56
    assert np.array_equal(heights, expected)  # perimeter is artwork too
    assert [x[0] for x in reports[0]['swap_entries']]==[1.48,1.64,1.88]
    assert reports[0]['art_height_range_mm'][0]>1
    for r in results:
        with zipfile.ZipFile(r['threemf']) as z:
            cfg=json.loads(z.read('Metadata/project_settings.config'))
            assert float(cfg['initial_layer_print_height'])==.2
            assert float(cfg['layer_height'])==.08
            assert 'max_z="1"' in z.read('Metadata/layer_config_ranges.xml').decode()
            assert z.read('Metadata/custom_gcode_per_layer.xml').count(b'type="2"')==3
            assert len(cfg['filament_colour'])==4
            assert 'M104' in cfg['machine_start_gcode'] and 'machine: A1' not in cfg['machine_start_gcode']


def _two_tone(path, n=48):
    """Black field with a white square - the simplest image with one obvious swap."""
    a = np.zeros((n, n, 3), np.uint8); a[n//4:3*n//4, n//4:3*n//4] = 255
    Image.fromarray(a).save(path); return str(path)


def test_palette_order_and_validation():
    lib = load_filament_library(str(DEFAULT_FILAMENTS_JSON))
    fils = palette_filaments(['White', 'Klein Blue', 'Black'], lib)
    assert [f.name for f in fils] == ['Black', 'Klein Blue', 'White']        # auto: dark -> light
    assert [f.name for f in palette_filaments(['White', 'Black'], lib, 'fixed')] == ['White', 'Black']
    for bad in (['Black'], ['Black'] * 2, ['Black', 'Unobtainium'], ['Black', 'White', 'Red', 'Green', 'Pink', 'Oak']):
        with pytest.raises((ValueError, KeyError)):
            palette_filaments(bad, lib)


def test_fit_swaps_puts_white_exactly_where_the_image_is_white():
    lib = load_filament_library(str(DEFAULT_FILAMENTS_JSON))
    fils = palette_filaments(['Black', 'White'], lib)
    rgb = np.zeros((8, 8, 3), np.uint8); rgb[:, 4:] = 255
    h = tone_height(rgb) + .56
    n = column_layers(h, 1.)
    fit = fit_swaps(rgb, h, fils, 1.)
    # black columns (1 layer) must not reach White; white columns get as much White as possible
    assert fit['starts'] == [int(n[:, :4].max()) + 1]
    assert fit['surface_share']['White'] == .5 and fit['surface_share']['Black'] == .5


def test_custom_palette_export(tmp_path):
    img = _two_tone(tmp_path / 'two.png')
    runs = [convert_reference_band(img, str(tmp_path / str(i)), width_mm=12, pitch_mm=.25,
                                   palette=['White', 'Lavender Purple', 'Black']) for i in range(2)]
    assert Path(runs[0]['threemf']).read_bytes() == Path(runs[1]['threemf']).read_bytes()   # deterministic
    st = runs[0]['stats']
    assert st['palette_mode'] == 'custom' and st['filaments'] == ['Black', 'Lavender Purple', 'White']
    zs = [z for z, _, _ in st['swap_entries']]
    assert len(zs) == 2 and zs == sorted(zs) and len(set(zs)) == 2
    assert all(abs((z - 1.) / .08 - round((z - 1.) / .08)) < 1e-9 for z in zs)          # on real layer tops
    assert [slot for _, slot, _ in st['swap_entries']] == [2, 3]
    assert Path(runs[0]['preview_png']).name.endswith('_predicted.png')
    with zipfile.ZipFile(runs[0]['threemf']) as z:
        cfg = json.loads(z.read('Metadata/project_settings.config'))
        assert len(cfg['filament_colour']) == 3
        assert z.read('Metadata/custom_gcode_per_layer.xml').count(b'type="2"') == 2
        assert len(cfg['nozzle_temperature']) == 3


def test_thin_features_are_removed_at_every_layer():
    h = np.full((40, 40), 1.1); h[20, 5:35] = 2.8          # a one-sample-wide ridge
    h[10:18, 10:18] = 2.0                                     # a printable 8-sample plateau
    out, k = remove_thin_features(h, .15)
    assert k == 4 and np.all(out <= h)                        # only ever lowers
    assert out[20, 5:35].max() <= 1.1 + 1e-9                  # hairline ridge flattened
    assert np.allclose(out[10:18, 10:18], 2.0)                # real feature kept
    for t in np.unique(out)[1:]:                              # every cross-section survives a k x k opening
        layer = out >= t
        assert np.array_equal(ndimage.binary_opening(layer, np.ones((k, k))), layer)
    assert remove_thin_features(h, .15, 0) == (h, 0)


def test_process_covers_both_x2d_nozzles(tmp_path):
    """A 5th spool is fed to the X2D's second (Bowden) nozzle; our preset is used
    verbatim, so every process list needs an entry for every nozzle variant or
    Bambu Studio fails with a bare 'Slicing failed'."""
    img = _two_tone(tmp_path / 'five.png')
    r = convert_reference_band(img, str(tmp_path / 'o'), width_mm=12, pitch_mm=.25,
                               palette=['Black', 'Klein Blue', 'Oak', 'Lavender Purple', 'White'])
    sysp = json.loads(SYSTEM_PROCESS_JSON.read_text())['config']
    with zipfile.ZipFile(r['threemf']) as z:
        cfg = json.loads(z.read('Metadata/project_settings.config'))
    for k, v in sysp.items():
        assert k in cfg, k
        if isinstance(v, list):
            assert len(cfg[k]) == len(v), k
    assert cfg['print_extruder_id'] == ['1', '1', '1', '2', '2', '2']
    assert cfg['wall_loops'] == '2' and cfg['layer_height'] == '0.08' and cfg['top_shell_layers'] == '9999'


def test_two_nozzle_rule_keeps_ours_and_bambus():
    system = {'config': {'outer_wall_speed': ['200', '200', '200', '120', '120', '120'],
                         'inner_wall_speed': ['300', '300', '300', '150', '150', '150'],
                         'print_extruder_id': ['1', '1', '1', '2', '2', '2'],
                         'only_one_wall_top': '1'}}
    cfg = {'outer_wall_speed': ['90'], 'inner_wall_speed': ['300'], 'print_extruder_id': ['1']}
    complete_two_nozzle_process(cfg, system)
    assert cfg['outer_wall_speed'] == ['90'] * 6                              # our change -> every nozzle
    assert cfg['inner_wall_speed'] == ['300', '300', '300', '150', '150', '150']   # untouched -> Bambu's per-nozzle
    assert cfg['print_extruder_id'] == ['1', '1', '1', '2', '2', '2'] and cfg['only_one_wall_top'] == '1'


def _dark_cover(n=64):
    """A night-time cover: mostly near-black, a few dim highlights (like to_hell_with_it)."""
    rng = np.random.default_rng(0)
    q = np.clip(rng.gamma(1.2, .04, (n, n)), 0, .4)
    q[n//3:n//2, n//4:3*n//4] = np.linspace(.15, .37, n//2)[None, :]
    return np.uint8(np.round(np.stack([q * .8, q * .9, q * 1.3], -1).clip(0, 1) * 255))


def test_auto_tone_spreads_a_dark_cover_over_the_relief():
    rgb = _dark_cover()
    raw = column_layers(tone_height(rgb) + .56, 1.)
    t, (lo, hi) = auto_tone(rgb, .5)
    auto = column_layers(height_from_tone(t) + .56, 1.)
    share = lambda n: np.bincount(n.ravel(), minlength=30)[:5].sum() / n.size
    assert share(raw) > .85 and share(auto) < .55                    # no longer nearly all the lowest layers
    assert (np.bincount(auto.ravel()) / auto.size > .01).sum() >= 15  # uses most of the relief
    assert t.min() == 0 and t.max() == 1 and lo < hi
    q = rgb / 255 @ np.array([.299, .587, .114])
    order = np.argsort(q.ravel(), kind='stable')
    assert np.all(np.diff(t.ravel()[order]) >= -1e-12)               # brighter is never lower
    assert np.array_equal(auto_tone(rgb, .5)[0], t)                  # deterministic
    s0, _ = auto_tone(rgb)
    assert np.allclose(s0, np.clip((q - lo) / (hi - lo), 0, 1))     # default: levels only
    levels = column_layers(height_from_tone(s0) + .56, 1.)
    assert share(levels) < share(raw) - .2                           # levels alone already spread it


def test_default_tone_keeps_a_big_flat_background_where_its_brightness_puts_it():
    """Rodeo: 55% flat dark background.  Equalisation lifted it two layers (into
    the next spool's band); the default levels-only tone must not."""
    rng = np.random.default_rng(1)
    q = np.clip(rng.normal(.18, .01, (64, 64)), 0, 1)                # background
    q[:, 20:36] = np.linspace(0, .15, 16)[None, :]                   # subject: mostly darker than it (hair,
    q[:, 36:44] = np.linspace(.25, 1, 8)[None, :]                    # shadows), some highlights - like Rodeo
    rgb = np.uint8(np.round(np.repeat(q[..., None], 3, -1) * 255))
    bg = (slice(None), slice(0, 20))
    raw_layers = np.median(column_layers(tone_height(rgb) + .56, 1.)[bg])
    lv_layers = np.median(column_layers(height_from_tone(auto_tone(rgb)[0]) + .56, 1.)[bg])
    eq_layers = np.median(column_layers(height_from_tone(auto_tone(rgb, .5)[0]) + .56, 1.)[bg])
    assert lv_layers == raw_layers and eq_layers >= raw_layers + 2


def test_toned_target_keeps_hue_and_sets_brightness():
    rgb = np.array([[[10, 20, 60], [40, 40, 40], [250, 250, 250]]], np.uint8)
    t = np.array([[.25, .5, .9]])
    out = toned_target(rgb, t)
    y = out @ np.array([.2126729, .7151522, .0721750])
    assert np.allclose(y, srgb_to_linear(t), atol=1e-9)               # luminance of a grey of that tone
    lin = srgb_to_linear(rgb[0, 0] / 255.)
    assert np.allclose(out[0, 0] / out[0, 0].sum(), lin / lin.sum())  # navy stays the same blue
    assert np.allclose(out[0, 1], out[0, 1, 0])                       # grey stays grey
    bright = toned_target(np.array([[[200, 20, 20]]], np.uint8), np.array([[.95]]))
    assert bright.max() <= 1 and np.isclose(bright @ np.array([.2126729, .7151522, .0721750]),
                                             srgb_to_linear(.95)).all()


def test_choose_palette_finds_the_spools_the_art_is_made_of():
    lib = load_filament_library(str(DEFAULT_FILAMENTS_JSON))
    fils = palette_filaments(['Black', 'Lavender Purple', 'Beige'], lib)
    ramp = np.linspace(0, 1, 64)
    q = np.tile(ramp[None, :], (16, 1))
    h = height_from_tone(q) + .56
    n = column_layers(h, 1.)
    starts = [4, 12]                                  # render a known plan, then ask for it back
    col = _stack_colour(np.arange(n.max() + 1), np.array([starts]), fils, 7.)[0]
    lab = linear_rgb_to_lab_d65(col[n]).reshape(-1, 3)
    fit = choose_palette(None, h, lib, 1., target_lab=lab)
    assert [f.name for f in fit['filaments']] == ['Black', 'Lavender Purple', 'Beige']
    assert fit['starts'] == starts and fit['mean_dE'] < 1e-6
    assert set(fit['by_size']) == {2, 3, 4} and fit['palettes_searched'] == 66 + 220 + 495


def test_auto_palette_export(tmp_path):
    img = tmp_path / 'dark.png'; Image.fromarray(_dark_cover(48)).save(img)
    runs = [convert_reference_band(str(img), str(tmp_path / str(i)), width_mm=12, pitch_mm=.25) for i in range(2)]
    assert Path(runs[0]['threemf']).read_bytes() == Path(runs[1]['threemf']).read_bytes()   # deterministic
    st = runs[0]['stats']
    assert st['palette_mode'] == 'auto' and st['tone']['mode'] == 'auto'
    assert 2 <= len(st['filaments']) <= 4 and len(st['swap_entries']) == len(st['filaments']) - 1
    lib = load_filament_library(str(DEFAULT_FILAMENTS_JSON))
    lum = [lib[name].luminance for name in st['filaments']]
    assert lum == sorted(lum)                                           # dark -> light
    assert Path(runs[0]['threemf']).with_name('dark_reference_band_target.png').exists()
    assert st['colour_fit']['against'].startswith('contrast-stretched')
    with zipfile.ZipFile(runs[0]['threemf']) as z:
        cfg = json.loads(z.read('Metadata/project_settings.config'))
        assert len(cfg['filament_colour']) == len(st['filaments'])
