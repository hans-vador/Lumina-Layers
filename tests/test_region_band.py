import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from core.band.reference import convert_reference_band
from core.band.regions import band_modifiers, clean_regions, convert_region_band, region_polygon


def _two_hue(path, n=64):
    """Left half pink, right half blue, the SAME brightness ramp top -> bottom:
    Reference layers prints both halves the same colour at every height."""
    ramp = np.linspace(0, 1, n)[:, None, None]
    img = np.zeros((n, n, 3))
    img[:, :n // 2] = ramp * np.array([245, 163, 183]) / 255
    img[:, n // 2:] = np.clip(ramp * np.array([30, 68, 190]) / 255 * 1.6, 0, 1)
    Image.fromarray(np.uint8(np.round(img * 255))).save(path)
    return str(path)


@pytest.fixture(scope='module')
def two_hue_run(tmp_path_factory):
    d = tmp_path_factory.mktemp('rb')
    img = _two_hue(d / 'two.png')
    return img, d, [convert_region_band(img, str(d / str(i)), width_mm=30, pitch_mm=.25) for i in range(2)]


def test_two_hues_at_one_brightness_get_two_ramps(two_hue_run):
    _, _, runs = two_hue_run
    st = runs[0]['stats']
    assert st['region_count'] == 2
    tried = {t['regions_asked']: t['J'] for t in st['regions_tried']}
    assert tried[2] < .5 * tried[1]                                   # regions pay for themselves here
    spools = [set(r['spools']) for r in st['regions']]
    assert spools[0] != spools[1]
    assert all(r['spools'][0] == st['filaments'][0] for r in st['regions'])   # shared base = backing
    labels = np.load(Path(runs[0]['threemf']).with_name('two_region_band_regions.npy'))
    mid = labels.shape[0] // 2                                          # the coloured middle rows split left / right
    rows = slice(mid - 8, mid + 8)
    left, right = labels[rows, : labels.shape[1] // 2 - 4], labels[rows, labels.shape[1] // 2 + 4:]
    assert np.bincount(left.ravel()).argmax() != np.bincount(right.ravel()).argmax()


def test_region_export_is_deterministic_and_has_modifiers(two_hue_run):
    _, _, runs = two_hue_run
    assert Path(runs[0]['threemf']).read_bytes() == Path(runs[1]['threemf']).read_bytes()
    st = runs[0]['stats']
    with zipfile.ZipFile(runs[0]['threemf']) as z:
        ms = z.read('Metadata/model_settings.config').decode()
        cfg = json.loads(z.read('Metadata/project_settings.config'))
        gc = z.read('Metadata/custom_gcode_per_layer.xml').decode()
        assert 'max_z="1"' in z.read('Metadata/layer_config_ranges.xml').decode()
    assert ms.count('subtype="modifier_part"') == int(st['print'].split()[0]) > 0
    assert 'type="2"' not in gc                                          # no plate-wide swaps
    assert cfg['enable_prime_tower'] == '1' and len(cfg['filament_colour']) == len(st['filaments'])
    assert st['watertight'] and st['tool_changes_planned'] >= 1


def test_one_modifier_per_spool_inside_the_plaque():
    """The object is the base spool; every other spool is ONE modifier - the union
    of that spool's bands in every region - so the file is one tidy object."""
    from core.band.optics import Filament
    pool = [Filament.from_hex(nm, hx, 1.) for nm, hx in
            (('Black', '#000000'), ('Brown', '#6E4F3A'), ('Pink', '#F5A3B7'), ('White', '#FFFFFF'))]
    cells = np.zeros((8, 8), np.int64)
    cells[:, 4:] = 1
    fit = {'choice': [((0, 1, 3), np.array([3, 6])), ((0, 2, 3), np.array([2, 6]))]}
    mods = band_modifiers(cells, fit, pool, {0: 0, 1: 1, 2: 2, 3: 3}, .5, 3.)
    assert [(ext, name) for _, ext, name in mods] == [(2, 'Brown'), (3, 'Pink'), (4, 'White')]
    brown, pink, white = (m for m, *_ in mods)
    assert brown.bounds.tolist() == [[0, 0, 1.16], [2., 4., 1.4]]                  # left half, layers 3..5
    assert pink.bounds.tolist() == [[2., 0, 1.08], [4., 4., 1.4]]                  # right half, layers 2..5
    assert white.bounds.tolist() == [[0, 0, 1.4], [4., 4., 3.]]                    # both halves, layer 6 up
    assert all(m.is_watertight for m, *_ in mods)
    assert abs(white.volume - 16 * 1.6) < 1e-6                                     # two prisms, no overlap


def test_region_polygon_is_exact_with_holes():
    mask = np.ones((10, 10), bool)
    mask[3:6, 3:6] = False
    poly = region_polygon(mask, .2)
    assert poly.geom_type == 'Polygon' and len(poly.interiors) == 1
    assert abs(poly.area - (100 - 9) * .04) < 1e-9


def test_clean_regions_drops_specks_and_corner_contacts():
    lab = np.zeros((40, 40), np.int64)
    lab[20:, :] = 1
    lab[5, 5] = 1                                                       # a one-cell speck
    lab[10, 30] = lab[11, 31] = 1                                       # diagonal pair
    out = clean_regions(lab, 2, .35)
    assert out[5, 5] == 0 and out[10, 30] == 0 and out[11, 31] == 0
    a, b, c, d = out[:-1, :-1], out[:-1, 1:], out[1:, :-1], out[1:, 1:]
    assert not ((a == d) & (b == c) & (a != b)).any()
    for r in (0, 1):
        assert ndimage.label(out == r)[1] == 1


def test_one_region_prints_like_reference(tmp_path):
    img = tmp_path / 'grey.png'
    g = np.tile(np.linspace(0, 255, 48, dtype=np.uint8)[None, :, None], (48, 1, 3))
    Image.fromarray(g).save(img)
    res = convert_region_band(str(img), str(tmp_path / 'r'), width_mm=12, pitch_mm=.25, max_regions=1)
    ref = convert_reference_band(str(img), str(tmp_path / 'f'), width_mm=12, pitch_mm=.25)
    assert res['stats']['region_count'] == 1 and res['stats']['tower'] is None
    assert np.array_equal(np.load(Path(res['threemf']).with_name('grey_region_band_height.npy')),
                          np.load(Path(ref['threemf']).with_name('grey_reference_band_height.npy')))
    with zipfile.ZipFile(res['threemf']) as z:
        assert 'modifier_part' not in z.read('Metadata/model_settings.config').decode()
        assert z.read('Metadata/custom_gcode_per_layer.xml').count(b'type="2"') == len(res['stats']['swap_entries'])


def test_ticked_spools_are_all_used(tmp_path):
    """Picking spools means printing exactly those, as in Reference layers - not a
    shortlist to choose from (Blond: 5 ticked, Green was silently dropped)."""
    img = _two_hue(tmp_path / 'two.png', 48)
    picks = ['Black', 'Coffee Brown', 'Oak', 'Green', 'White']
    st = convert_region_band(img, str(tmp_path / 'o'), width_mm=24, pitch_mm=.25, palette=picks)['stats']
    assert st['palette_mode'] == 'custom' and sorted(st['filaments']) == sorted(picks)
    assert set().union(*(set(r['spools']) for r in st['regions'])) == set(picks)


def test_scoring_is_plain_for_greys_and_colour_first_for_colours():
    from core.band.regions import _d2
    from core.band.optics import load_filament_library, linear_rgb_to_lab_d65
    from core.band.reference import _stack_colour
    grey = np.array([30., -4., 2.])                                   # Rodeo's grey-green background
    pink = np.array([35., 30., 5.])
    assert _d2(grey, pink, 'hue') == pytest.approx(_d2(grey, pink, 'lab'))   # greys: plain CIELAB
    lib = load_filament_library('assets/filaments_user_measured.json')
    hair = np.array([38., -18., 16.])                                 # Blond's olive hair
    brown = linear_rgb_to_lab_d65(lib['Coffee Brown'].rgb_lin)
    veil = linear_rgb_to_lab_d65(_stack_colour(np.array([3]), np.array([[1, 3]]),
                                               [lib['Black'], lib['Coffee Brown'], lib['Green']], 7.)[0, 0])
    assert _d2(hair, brown, 'lab') < _d2(hair, veil, 'lab')          # plain CIELAB: brown wins
    assert _d2(hair, veil, 'hue') < _d2(hair, brown, 'hue')          # colour-first: the green veil wins


def test_small_coloured_patch_gets_its_own_spool():
    """A green patch (~1% of the plaque) as bright as the brown around it: a main
    ramp without Green prints it brown; the accent pass gives it Green."""
    import core.band.regions as rg
    from core.band.optics import load_filament_library, linear_rgb_to_lab_d65
    from core.band.reference import (auto_tone, column_layers, height_from_tone, palette_filaments,
                                     toned_target)
    n = 172                                                            # 60 mm at the .35 mm working grid
    q = np.tile(np.linspace(.15, .6, n)[None, :], (n, 1))
    img = np.stack([q * 1.15, q * .9, q * .7], -1)                     # warm brown ramp
    patch = (slice(72, 90), slice(72, 90))
    img[patch] = np.array([.25, .75, .3]) * q[patch][..., None] / (.299 * .25 + .587 * .75 + .114 * .3)
    rgb = np.uint8(np.round(np.clip(img, 0, 1) * 255))
    t, _ = auto_tone(rgb)
    lab_img = linear_rgb_to_lab_d65(toned_target(rgb, t))
    n_img = column_layers(height_from_tone(t) + .56, 1.)
    lib = load_filament_library('assets/filaments_user_measured.json')
    pool = palette_filaments(['Black', 'Coffee Brown', 'Oak', 'Green'], lib)      # dark -> light
    assert [f.name for f in pool][3] == 'Green'
    labels = np.zeros((n, n), np.int64)
    fit = rg._fit(lab_img.reshape(-1, 3), n_img.ravel(), labels.ravel(), 1, pool, 4, 7., restrict=[0, 1, 2])
    fit = rg._exact_cost(fit, rg.pixel_d2(lab_img.reshape(-1, 3), n_img.ravel(), labels.ravel(), fit, pool, 7.))
    labels2, fit2, accents, taken = rg.add_accents(lab_img, n_img, labels, fit, pool, 4, 7., 60 / (n - 1))
    assert len(accents) == 1 and 'Green' in accents[0]['spools']
    assert accents[0]['mean_dE_before'] - accents[0]['mean_dE_after'] >= rg.ACCENT_GAIN_DE
    inside = taken[patch].mean()
    assert inside > .7 and taken.sum() < 1.5 * taken[patch].size         # the accent is the patch
    assert fit2['metrics'].count('hue') == 1


def test_grey_patch_printed_brown_becomes_black_and_white():
    """Blond's barcode: a neutral area the main ramp prints brown gets a ramp
    of neutral spools only."""
    import core.band.regions as rg
    from core.band.optics import load_filament_library, linear_rgb_to_lab_d65
    from core.band.reference import (auto_tone, column_layers, height_from_tone, palette_filaments,
                                     toned_target)
    n = 172
    q = np.tile(np.linspace(.15, .7, n)[None, :], (n, 1))
    img = np.stack([q * 1.15, q * .9, q * .7], -1)                     # warm brown ramp
    patch = (slice(72, 92), slice(60, 90))
    img[patch] = q[patch][..., None]                                   # a grey label, same brightness
    rgb = np.uint8(np.round(np.clip(img, 0, 1) * 255))
    t, _ = auto_tone(rgb)
    lab_img = linear_rgb_to_lab_d65(toned_target(rgb, t))
    n_img = column_layers(height_from_tone(t) + .56, 1.)
    lib = load_filament_library('assets/filaments_user_measured.json')
    pool = palette_filaments(['Black', 'Coffee Brown', 'White'], lib)
    labels = np.zeros((n, n), np.int64)
    fit = rg._fit(lab_img.reshape(-1, 3), n_img.ravel(), labels.ravel(), 1, pool, 3, 7., restrict=[0, 1])
    fit = rg._exact_cost(fit, rg.pixel_d2(lab_img.reshape(-1, 3), n_img.ravel(), labels.ravel(), fit, pool, 7.))
    _, fit2, accents, kind = rg.add_accents(lab_img, n_img, labels, fit, pool, 3, 7., 60 / (n - 1))
    neutral = [a for a in accents if a['kind'] == 'neutral']
    assert neutral and set(neutral[0]['spools']) <= {'Black', 'White'}
    assert (kind[patch] == 2).mean() > .7 and 'neutral' in fit2['metrics']
