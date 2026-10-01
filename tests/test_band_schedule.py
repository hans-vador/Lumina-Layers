"""BandSchedule: reference parity with the Astroworld 3mf and basic invariants."""
from __future__ import annotations

import numpy as np
import pytest

from core.band.optics import Filament
from core.band.schedule import BandSchedule

LIB = {
    'Black': Filament.from_hex('Black', '#000000', 0.2),
    'Red': Filament.from_hex('Red', '#C12E1F', 0.9),
    'Sunny Orange': Filament.from_hex('Sunny Orange', '#FF9016', 1.2),
    'White': Filament.from_hex('White', '#FFFFFF', 4.0),
}
ASTRO = BandSchedule(('Black', 'Red', 'Sunny Orange', 'White'), (9, 4, 6, 8), 0.16, 0.08)


def test_reference_swap_parity():
    entries = ASTRO.swap_entries(LIB)
    assert [e[1] for e in entries] == [2, 3, 4]
    assert [e[2] for e in entries] == ['#C12E1F', '#FF9016', '#FFFFFF']
    np.testing.assert_allclose([e[0] for e in entries], [0.88, 1.20, 1.68], atol=1e-6)
    # every swap top_z must be a layer top FL + lh*(n-1) with integer n >= 2
    for z, _, _ in entries:
        n = (z - 0.16) / 0.08 + 1
        assert abs(n - round(n)) < 1e-6 and round(n) >= 2


def test_layer_counts_and_heights():
    assert ASTRO.n_layers == 27
    assert ASTRO.n_layers <= 27
    assert abs(ASTRO.total_height_mm - 2.24) < 1e-9
    assert ASTRO.top_z(1) == pytest.approx(0.16)
    assert ASTRO.top_z(10) == pytest.approx(0.88)
    assert list(ASTRO.k_rows()) == list(range(9, 28))
    assert ASTRO.band_first_layer(0) == 1
    assert ASTRO.band_first_layer(1) == 10
    assert ASTRO.band_first_layer(3) == 20
    assert ASTRO.cum(-1) == 0 and ASTRO.cum(0) == 9 and ASTRO.cum(3) == 27


def test_stacks_rows_sum_to_k():
    st = ASTRO.stacks()
    assert st.shape == (19, 5) and st.dtype == np.int32
    k = np.arange(9, 28)
    np.testing.assert_array_equal(st.sum(1), k)
    np.testing.assert_array_equal(st[:, 0], 9)
    np.testing.assert_array_equal(st[:, 4], 0)
    # first row = base only, last row = full schedule
    np.testing.assert_array_equal(st[0], [9, 0, 0, 0, 0])
    np.testing.assert_array_equal(st[-1], [9, 4, 6, 8, 0])
    np.testing.assert_allclose(ASTRO.heights(), 0.16 + 0.08 * (k - 1))


def test_band_of_layer_and_slice_rule():
    np.testing.assert_array_equal(ASTRO.band_of_layer(np.array([1, 9, 10, 13, 14, 19, 20, 27])),
                                  [0, 0, 1, 1, 2, 2, 3, 3])
    # slice-plane rule: h just above top_z(n) - lh/2 contains layer n
    h = np.array([0.0, 0.05, 0.16, 0.20 - 1e-6, 0.20 + 1e-6, 0.88, 2.24])
    np.testing.assert_array_equal(ASTRO.layer_of_height(h), [0, 0, 1, 1, 2, 10, 27])


def test_key_stable_and_roundtrip():
    k1 = ASTRO.key()
    same = BandSchedule(('Black', 'Red', 'Sunny Orange', 'White'), (9, 4, 6, 8))
    assert k1 == same.key() and len(k1) == 10
    assert BandSchedule(('Black', 'Red', 'Sunny Orange', 'White'), (9, 4, 6, 9)).key() != k1
    assert BandSchedule.from_dict(ASTRO.to_dict()) == ASTRO
    text = ASTRO.swap_instructions(LIB)
    assert 'layer 10 (0.88 mm)' in text and 'Red' in text and '3 filament change' in text


def test_validation():
    with pytest.raises(ValueError):
        BandSchedule(('Black', 'White'), (9,))
    with pytest.raises(ValueError):
        BandSchedule(('Black', 'White'), (9, 0))
    with pytest.raises(ValueError):
        BandSchedule(('Black', 'White'), (3, 4), base_min_layers=5)
    with pytest.raises(ValueError):
        BandSchedule(tuple(str(i) for i in range(33)), (9,)+(1,)*32)
