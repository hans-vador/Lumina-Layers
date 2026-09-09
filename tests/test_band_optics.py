"""Colour maths and optical models for Band mode."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from core.band.optics import (BeerLambertModel, Filament, LinearTDModel, MeasuredRampModel,
                              linear_to_srgb, load_filament_library, luminance_y,
                              srgb_to_lab_d65, srgb_to_linear)
from core.band.ramp_lut import RampLUT, build_ramp_lut
from core.band.schedule import BandSchedule

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_PATH = os.path.join(REPO, 'assets', 'filaments_user.json')


def test_srgb_linear_roundtrip():
    x = np.linspace(0, 1, 1001)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-9)
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(x)), x, atol=1e-9)
    assert srgb_to_linear(np.array(0.5)) == pytest.approx(0.2140, abs=1e-3)


def test_lab_white_black_and_red():
    lab = srgb_to_lab_d65(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    np.testing.assert_allclose(lab[0], [100.0, 0.0, 0.0], atol=0.02)
    np.testing.assert_allclose(lab[1], [0.0, 0.0, 0.0], atol=1e-9)
    # sRGB red is L* ~53.2, a* ~80.1, b* ~67.2 (true CIELAB, not OpenCV 8-bit)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.2)


def test_filament_library_loads_12():
    lib = load_filament_library(LIB_PATH)
    assert len(lib) == 12
    assert lib['Black'].td_mm == 0.2 and lib['White'].hex == '#FFFFFF'
    assert lib['Klein Blue'].hex == '#1E44BE'
    assert all(f.settings_id == 'Bambu PLA Basic @BBL X2D 0.4 nozzle' for f in lib.values())
    assert lib['Black'].luminance < lib['Coffee Brown'].luminance < lib['White'].luminance
    assert lib['White'].rgb8 == (255, 255, 255)


def _bw():
    return {'Black': Filament.from_hex('Black', '#000000', 0.2),
            'White': Filament.from_hex('White', '#FFFFFF', 4.0)}


def test_black_white_ramp_monotonic_beer_lambert():
    lib = _bw()
    sched = BandSchedule(('Black', 'White'), (9, 18))
    ramp = BeerLambertModel(3.0).ramp(sched, [lib['Black'], lib['White']])
    assert ramp.shape == (19, 3)
    y = luminance_y(ramp)
    assert np.all(np.diff(y) > 0), "luminance must strictly increase with white thickness"
    assert y[0] == pytest.approx(0.0)
    # 18 layers = 1.44 mm of TD 4 white over black: o = 1-exp(-3*1.44/4) ~ 0.66
    assert y[-1] == pytest.approx(1 - np.exp(-3 * 1.44 / 4.0), abs=1e-6)
    # neutral: r == g == b
    np.testing.assert_allclose(ramp[:, 0], ramp[:, 1])
    np.testing.assert_allclose(ramp[:, 0], ramp[:, 2])


def test_linear_model_saturates():
    lib = _bw()
    sched = BandSchedule(('Black', 'White'), (9, 18))
    lib['White'].td_mm = 0.4   # 5 layers to opaque
    y = luminance_y(LinearTDModel().ramp(sched, [lib['Black'], lib['White']]))
    assert np.all(np.diff(y) >= 0)
    assert y[5] == pytest.approx(1.0) and y[-1] == pytest.approx(1.0)
    assert y[2] == pytest.approx(0.4)


def test_reference_schedule_ramp_and_lut_roundtrip(tmp_path):
    lib = load_filament_library(LIB_PATH)
    sched = BandSchedule(('Black', 'Red', 'Sunny Orange', 'White'), (9, 4, 6, 8))
    lut = build_ramp_lut(sched, lib, BeerLambertModel())
    assert lut.rgb.shape == (19, 3) and lut.rgb.dtype == np.uint8
    assert lut.stacks.shape == (19, 5)
    np.testing.assert_array_equal(lut.k_values, np.arange(9, 28))
    np.testing.assert_array_equal(lut.rgb[0], [0, 0, 0])            # base row is black
    assert lut.rgb[4].argmax() == 0                                  # 4 layers red: red dominant
    # Lumina .npz format + sidecar
    p = lut.save(str(tmp_path / 'Band_test.npz'))
    data = np.load(p)
    assert set(['rgb', 'stacks']) <= set(data.files)
    assert data['rgb'].dtype == np.uint8 and data['stacks'].dtype == np.int32
    side = json.load(open(p + '.json'))
    assert side['schedule']['layer_counts'] == [9, 4, 6, 8]
    assert side['model'] == 'beer-lambert'
    back = RampLUT.load(p)
    np.testing.assert_array_equal(back.rgb, lut.rgb)
    np.testing.assert_array_equal(back.stacks, lut.stacks)
    np.testing.assert_allclose(back.heights, sched.heights())
    assert back.schedule() == sched
    # measured model replays the same ramp
    meas = MeasuredRampModel.from_npz(p)
    ramp_lin = meas.ramp(sched, [lib[n] for n in sched.filament_names])
    np.testing.assert_array_equal(np.clip(np.round(linear_to_srgb(ramp_lin) * 255), 0, 255), lut.rgb)
    with pytest.raises(ValueError):
        meas.ramp(BandSchedule(('Black', 'White'), (9, 18)), [lib['Black'], lib['White']])
