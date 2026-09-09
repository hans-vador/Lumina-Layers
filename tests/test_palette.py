"""Palette optimiser: synthetic images and runtime budget."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PIL import Image

from core.band.optics import BeerLambertModel, load_filament_library
from core.band.palette import (image_hist_lab, optimise_thickness, ramp_lab_batch, score,
                               select_global_order, select_palette, select_palette_detailed,
                               thickness_vectors)
from core.band.schedule import BandSchedule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = load_filament_library(os.path.join(REPO, 'assets', 'filaments_user.json'))
MODEL = BeerLambertModel(3.0)


def _two_tone(path, dark=(20, 20, 20), light=(235, 235, 235), size=256):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, : size // 2] = dark
    arr[:, size // 2:] = light
    Image.fromarray(arr).save(path)
    return str(path)


def test_hist_two_tone(tmp_path):
    p = _two_tone(tmp_path / 'two.png')
    lab, w = image_hist_lab(p, k=64)
    assert lab.shape[1] == 3 and w.sum() == pytest.approx(1.0)
    assert lab.shape[0] <= 64
    # two dominant tones with ~50% each
    top = np.sort(w)[-2:]
    assert top.min() > 0.4
    assert lab[:, 0].min() < 15 and lab[:, 0].max() > 90


def test_score_shapes():
    lab = np.array([[50.0, 0, 0], [80.0, 0, 0]])
    w = np.array([0.5, 0.5])
    ramp = np.array([[50.0, 0, 0], [80.0, 0, 0], [20.0, 0, 0]])
    assert score(lab, w, ramp) == pytest.approx(0.0)
    batch = np.stack([ramp, ramp + [10, 0, 0]])
    s = score(lab, w, batch, wL=1.0)
    assert s.shape == (2,) and s[0] == pytest.approx(0.0) and s[1] == pytest.approx(10.0)


def test_thickness_vectors():
    import itertools
    for nb in (3, 4):
        v = thickness_vectors(nb, 1, 8, 18)
        expected = sum(1 for t in itertools.product(range(1, 9), repeat=nb) if sum(t) <= 18)
        assert v.shape == (expected, nb)
        assert v.min() == 1 and v.max() == 8 and v.sum(1).max() <= 18
        assert len({tuple(r) for r in v.tolist()}) == expected


def test_ramp_lab_batch_matches_model():
    sched = BandSchedule(('Black', 'Red', 'Sunny Orange', 'White'), (9, 4, 6, 8))
    fils = [LIB[n] for n in sched.filament_names]
    from core.band.optics import linear_rgb_to_lab_d65
    ref = linear_rgb_to_lab_d65(MODEL.ramp(sched, fils))
    rgb = np.array([[f.rgb_lin for f in fils]])
    td = np.array([[f.td_mm for f in fils]])
    lab, K = ramp_lab_batch(rgb, td, np.array([[4, 6, 8]]), 9, 0.08, MODEL, 27)
    assert K[0] == 27 and lab.shape == (1, 19, 3)
    np.testing.assert_allclose(lab[0], ref, atol=1e-9)


def test_two_tone_picks_dark_base_light_top(tmp_path):
    p = _two_tone(tmp_path / 'two.png')
    hist = image_hist_lab(p, k=64)
    t0 = time.perf_counter()
    sched, rep = select_palette_detailed(hist, LIB, MODEL, n_filaments=(4, 5), n0=9, max_layers=27)
    dt = time.perf_counter() - t0
    assert dt < 90, f"select_palette took {dt:.1f}s"
    lum = [LIB[n].luminance for n in sched.filament_names]
    assert lum == sorted(lum), "band order must be luminance ascending"
    assert lum[0] < 0.1, f"base should be dark, got {sched.filament_names[0]}"
    assert lum[-1] > 0.6, f"top should be light, got {sched.filament_names[-1]}"
    assert sched.layer_counts[0] == 9 and sched.n_layers <= 27
    assert sched.n_bands in (4, 5)
    assert rep['mean_dE'] < 12
    # the two tones must be served by the base and by the top band
    assert rep['band_shares'][0] > 0.3 and rep['band_shares'][-1] > 0.3


def test_optimise_thickness_fixed_order(tmp_path):
    p = _two_tone(tmp_path / 'two.png', dark=(0, 0, 0), light=(255, 255, 255))
    hist = image_hist_lab(p, k=16)
    sched = optimise_thickness(hist, ('Black', 'Red', 'Sunny Orange', 'White'), LIB, MODEL)
    assert sched.filament_names == ('Black', 'Red', 'Sunny Orange', 'White')
    assert sched.layer_counts[0] == 9 and sched.n_layers <= 27
    # pure white needs the thickest possible white band
    assert sched.layer_counts[-1] >= 6


def test_hist_ignores_transparent_pixels(tmp_path):
    arr = np.zeros((64, 64, 4), dtype=np.uint8)
    arr[:, :32, :3] = 0            # black under a transparent background
    arr[:, :32, 3] = 0
    arr[:, 32:, :3] = 235          # opaque light grey
    arr[:, 32:, 3] = 255
    p = tmp_path / 'rgba.png'
    Image.fromarray(arr, 'RGBA').save(p)
    lab, w = image_hist_lab(str(p), k=16)
    assert w.sum() == pytest.approx(1.0)
    assert lab[:, 0].min() > 80, "transparent black pixels must carry no weight"
    arr[:, :32, 3] = 255
    Image.fromarray(arr, 'RGBA').save(p)
    lab2, w2 = image_hist_lab(str(p), k=16)
    assert lab2[:, 0].min() < 5 and w2[np.argmin(lab2[:, 0])] == pytest.approx(0.5, abs=0.02)


def test_single_filament_and_small_n0(tmp_path):
    p = _two_tone(tmp_path / 'two.png')
    hist = image_hist_lab(p, k=16)
    sched, rep = select_palette_detailed(hist, LIB, MODEL, n_filaments=(1,), n0=9, max_layers=27)
    assert sched.n_bands == 1 and sched.layer_counts == (9,) and rep['n_changes'] == 0
    assert rep['band_shares'] == pytest.approx([1.0])
    # n0 below the default base_min_layers (5) must not crash the optimiser
    s4 = optimise_thickness(hist, ('Black', 'Red', 'Sunny Orange', 'White'), LIB, MODEL, n0=4, max_layers=22)
    assert s4.layer_counts[0] == 4 and s4.base_min_layers == 4 and s4.n_layers <= 22
    s3, _ = select_palette_detailed(hist, LIB, MODEL, n_filaments=(2,), n0=3, max_layers=12, stage1_top=3)
    assert s3.n0 == 3 and s3.base_min_layers == 3


def test_luminance_assignment_mirrors_processor():
    """With a tone curve the optimiser assigns histogram colours to the layer the
    luminance processor would print them on, whatever the ramp colours are."""
    from core.band.palette import evaluate_schedules, tone_assignment
    from core.band.tone import ToneCurve, engrave_floor
    tone = ToneCurve.reference()
    sched = BandSchedule(('Black', 'Red', 'Sunny Orange', 'White'), (9, 4, 6, 8))
    Ls = np.array([0.0, 10.0, 30.0, 50.0, 72.0, 90.0, 100.0])
    idx = tone_assignment(Ls, np.array([27]), tone, 9, 0.08, 0.16, 5)[0]
    expect = tone.layer(Ls, sched, engrave_floor(sched)) - 9
    np.testing.assert_array_equal(idx, expect)
    # black -> base row; L* 100 -> layer 25/26 (reference curve, q_hi 1.06); monotone
    assert idx[0] == 0 and 16 <= idx[-1] <= 17 and np.all(np.diff(idx) >= 0)
    # L* 72 (the Astroworld sky) -> layer 18/19 = orange band, as in the reference, not white
    k72 = 9 + int(tone_assignment(np.array([72.0]), np.array([27]), tone, 9, 0.08, 0.16, 5)[0, 0])
    assert k72 in (18, 19) and int(sched.band_of_layer(np.array([k72]))[0]) == 2
    hist = (np.array([[30.0, 40.0, 30.0], [72.0, -5.0, -20.0]]), np.array([0.5, 0.5]))
    fils = [LIB[n] for n in sched.filament_names]
    rgb = np.array([[f.rgb_lin for f in fils]])
    td = np.array([[f.td_mm for f in fils]])
    res = evaluate_schedules(hist, rgb, td, np.array([[4, 6, 8]]), 9, 0.08, MODEL, 27, tone=tone)
    assert res['shares'].shape == (1, 4) and res['shares'][0].sum() == pytest.approx(1.0)
    assert res['shares'][0][2] == pytest.approx(0.5)      # the L* 72 colour is owned by the orange band
    # shorter schedule (K=17): the same L* maps to a proportionally lower layer
    idx17 = tone_assignment(Ls, np.array([17]), tone, 9, 0.08, 0.16, 5)[0]
    assert idx17.max() <= 8 and np.all(idx17 <= idx)


def test_global_order_and_runtime_on_12_library(tmp_path):
    a = _two_tone(tmp_path / 'a.png', dark=(40, 10, 10), light=(250, 200, 60))
    b = _two_tone(tmp_path / 'b.png', dark=(10, 10, 40), light=(240, 240, 240))
    hists = [image_hist_lab(a), image_hist_lab(b)]
    t0 = time.perf_counter()
    names = select_global_order(hists, LIB, MODEL)
    assert time.perf_counter() - t0 < 90
    assert 4 <= len(names) <= 5 and len(set(names)) == len(names)
    assert all(n in LIB for n in names)
    lum = [LIB[n].luminance for n in names]
    assert lum == sorted(lum)
    # per-album thicknesses with the fixed global order
    s = optimise_thickness(hists[0], names, LIB, MODEL)
    assert s.filament_names == tuple(names)
