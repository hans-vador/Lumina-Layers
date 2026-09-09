"""Smoke test for Band mode processing pipeline (processor + pipeline + CLI seams).

Skipped until the sibling modules (optics/schedule/ramp_lut/heightfield_mesh/
writer3mf, built concurrently) are importable.
"""

import json
import os
import sys
import zipfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

optics = pytest.importorskip("core.band.optics")
schedule_mod = pytest.importorskip("core.band.schedule")
pytest.importorskip("core.band.ramp_lut")
pytest.importorskip("core.band.heightfield_mesh")
pytest.importorskip("core.band.writer3mf")

from PIL import Image  # noqa: E402

LIB = [
    {"name": "Black", "hex": "#000000", "td_mm": 0.2, "td_source": "hueforge_lib"},
    {"name": "White", "hex": "#FFFFFF", "td_mm": 4.0, "td_source": "guess"},
    {"name": "Red", "hex": "#C12E1F", "td_mm": 0.9, "td_source": "guess"},
    {"name": "Sunny Orange", "hex": "#FF9016", "td_mm": 1.2, "td_source": "guess"},
    {"name": "Klein Blue", "hex": "#1E44BE", "td_mm": 0.8, "td_source": "guess"},
]


@pytest.fixture()
def filaments_json(tmp_path):
    p = tmp_path / "filaments.json"
    p.write_text(json.dumps(LIB))
    return str(p)


@pytest.fixture()
def gradient_image(tmp_path):
    """Colourful diagonal gradient with a dark corner so the base tone and engraving are exercised."""
    n = 96
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    r = np.clip(0.1 + 0.9 * xx, 0, 1)
    g = np.clip(0.05 + 0.5 * xx * yy, 0, 1)
    b = np.clip(0.1 + 0.9 * yy, 0, 1)
    rgb = np.stack([r, g, b], -1)
    rgb[: n // 4, : n // 4] *= 0.15  # near-black patch
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    p = tmp_path / "gradient.png"
    img.save(p)
    return str(p)


def _schedule():
    return schedule_mod.BandSchedule(("Black", "Red", "Sunny Orange", "White"), (9, 4, 6, 8))


def test_dynamic_color_system_registration():
    from config import ColorSystem
    conf = {"name": "Band", "slots": ["a", "b", "c", "d", "e"], "preview": {}, "map": {}, "layer_count": 5}
    ColorSystem.register_dynamic("Band:test", conf)
    assert ColorSystem.get("Band:test") is conf
    # built-in heuristics untouched
    assert ColorSystem.get("RYBW") is ColorSystem.RYBW
    assert ColorSystem.get("Band:unknown-key") is ColorSystem.RYBW  # fallback unchanged


def test_converter_guard_returns_empty_for_band():
    from core.converter import _get_actual_lut_slot_colors

    class P:
        color_mode = "Band:abc"
        ref_stacks = np.full((5, 5), 3, dtype=np.int32)
        lut_rgb = np.zeros((5, 3), dtype=np.uint8)

    assert _get_actual_lut_slot_colors(P()) == {}


def test_process_image_resample_kwarg_default():
    import inspect
    from core.image_processing import LuminaImageProcessor
    sig = inspect.signature(LuminaImageProcessor.process_image)
    assert "resample" in sig.parameters
    assert sig.parameters["resample"].default == Image.Resampling.NEAREST


def _processor(tmp_path, filaments_json):
    from core.band.optics import BeerLambertModel, load_filament_library
    from core.band.processor import BandProcessor
    from core.band.ramp_lut import build_ramp_lut

    lib = load_filament_library(filaments_json)
    sch = _schedule()
    ramp = build_ramp_lut(sch, lib, BeerLambertModel())
    npz = str(tmp_path / f"Band_{sch.key()}.npz")
    ramp.save(npz)
    return BandProcessor(ramp, sch, lib, npz, f"Band:{sch.key()}"), sch, lib, ramp


@pytest.mark.parametrize("height_mode", ["luminance", "match"])
def test_processor_heights_and_stats(tmp_path, filaments_json, gradient_image, height_mode):
    bp, sch, lib, ramp = _processor(tmp_path, filaments_json)
    res = bp.process(gradient_image, 24.0, quantize_colors=64, smooth_sigma=10,
                     interpolate=True, engrave=True, gaussian_px=0.5, mesh_pitch_mm=0.2,
                     height_mode=height_mode)

    H, W = res.height_mm.shape
    assert res.k_map.shape == (H, W) and res.mask_solid.shape == (H, W) and res.matched_rgb.shape == (H, W, 3)
    assert res.mask_solid.all()
    assert res.k_map.min() >= 9 and res.k_map.max() <= 27
    z_min = sch.top_z(sch.base_min_layers)
    z_max = sch.top_z(27) + 1e-6                                    # mid-cell of the last layer, never above
    assert res.height_mm.min() >= z_min - 1e-6 and res.height_mm.max() <= z_max
    assert abs((W - 1) * res.pitch_mm - 24.0) < 1e-6

    st = res.stats
    per_layer = st["top_surface_fraction_per_layer"]
    assert set(per_layer) == set(range(1, 28))
    assert abs(sum(per_layer.values()) - 1.0) < 1e-6
    assert all(per_layer[n] == 0.0 for n in range(1, 6))          # nothing tops out below layer 6 (floor 0.521)
    per_fil = st["top_surface_fraction_per_filament"]
    assert set(per_fil) == {"Black", "Red", "Sunny Orange", "White"}
    assert abs(sum(per_fil.values()) - 1.0) < 1e-6
    # light and dark regions both present (gradient brightest corner L* 74, dark patch L* < 10)
    assert per_fil["Black"] > 0.02 and per_fil["Sunny Orange"] > 0.05
    if height_mode == "match":
        assert per_fil["White"] > 0.005 and res.k_map.max() >= 20
    else:
        # the reference tone curve puts L* 74 into the last orange layers (as the
        # reference does with its sky); the white band starts at L* ~76
        assert res.k_map.max() >= 18
    assert st["tool_changes"] == 3
    assert st["mean_dE"] >= 0 and st["mean_dL"] >= 0
    assert st["height_mode"] == height_mode
    assert st["engrave_floor_mm"] == pytest.approx(sch.top_z(5) + 0.04 + 1e-3)
    if height_mode == "luminance":
        assert st["tone_curve"] == {"q_lo": -0.13, "q_hi": 1.06, "gamma": 1.55}
        # luminance mapping spreads a smooth gradient over every layer its L* range
        # covers (L* ~2..74 -> layers 6..19), no plateau
        assert sum(1 for n in range(6, 20) if per_layer[n] > 0.002) >= 12
        assert max(per_layer.values()) < 0.25
    else:
        assert st["luma_weight"] == 5.0


def _slice_planes(sch):
    return np.array([sch.first_layer_mm / 2] + [sch.top_z(n) - sch.layer_h / 2 for n in range(2, 29)])


@pytest.mark.parametrize("height_mode", ["luminance", "match"])
def test_no_column_lands_on_a_slice_plane(tmp_path, filaments_json, gradient_image, height_mode):
    """Slice plane of layer n is top_z(n) - lh/2; a column exactly there has
    float-ambiguous layer membership.  No solid column may sit within 1e-4 mm of
    a plane in any mode (match mode: |f| <= 0.45 keeps >= 4 um away by
    construction; continuous maps are nudged 1 um off the plane; the clamps
    0.521 and top_z(K) are off-plane) - with and without smoothing/resampling."""
    bp, sch, lib, ramp = _processor(tmp_path, filaments_json)
    planes = _slice_planes(sch)
    for engrave, gaussian_px, pitch in ((False, 0, 0.1), (True, 0, 0.1), (True, 0.5, 0.15)):
        res = bp.process(gradient_image, 24.0, quantize_colors=64, smooth_sigma=10, interpolate=True,
                         engrave=engrave, gaussian_px=gaussian_px, mesh_pitch_mm=pitch, height_mode=height_mode)
        h = res.height_mm[res.mask_solid].astype(np.float64)
        d = np.abs(h[:, None] - planes[None, :]).min(axis=1)
        assert np.all(d > 1e-4), d.min()
        # what the writer emits (%.3f) must not land on a plane either
        h3 = np.round(h, 3)
        assert np.all(np.abs(h3[:, None] - planes[None, :]).min(axis=1) > 1e-6)
        assert res.stats["slice_plane_hits"] == 0
        assert res.stats.get("slice_plane_hits_mesh", 0) == 0
        assert res.height_mm.max() <= sch.top_z(27) + 1e-6
    # saturated projections are clipped strictly inside the cell
    from core.band.processor import SUBLAYER_MAX
    assert SUBLAYER_MAX < 0.5


def test_match_mode_saturated_projection_stays_inside_cell(tmp_path, filaments_json):
    """A pixel far brighter than its matched row projects past the cell edge;
    the old clip to 0.5 put it exactly on the slice plane (flat plateaus in the
    mesh).  Now |f| <= SUBLAYER_MAX, i.e. at least 0.05*lh from the plane."""
    from core.band.processor import SUBLAYER_MAX
    bp, sch, lib, ramp = _processor(tmp_path, filaments_json)
    img = np.zeros((32, 64, 3), np.uint8)
    img[:, :32] = (255, 255, 255)        # white: matched to the top row, f -> toward row K-1 or 0
    img[:, 32:] = (250, 140, 20)         # bright orange, brighter than the ramp's orange rows
    p = tmp_path / "sat.png"
    Image.fromarray(img).save(p)
    res = bp.process(str(p), 6.3, quantize_colors=4, smooth_sigma=0, interpolate=True, engrave=False,
                     gaussian_px=0, mesh_pitch_mm=0.1, height_mode="match")
    k = res.k_map[res.mask_solid].astype(np.float64)
    f = (res.height_mm[res.mask_solid] - (0.16 + 0.08 * (k - 1))) / 0.08
    assert np.abs(f).max() <= SUBLAYER_MAX + 1e-5
    assert res.stats["slice_plane_hits"] == 0


def test_luminance_mode_follows_tone_curve(tmp_path, filaments_json):
    """Flat patches of known L* land on the layers the tone curve predicts,
    independent of the ramp's colours, and the ramp row colours them."""
    from core.band.optics import srgb_to_lab_d65
    from core.band.tone import ToneCurve, engrave_floor
    bp, sch, lib, ramp = _processor(tmp_path, filaments_json)
    greys = [0, 40, 90, 140, 190, 235, 255]
    img = np.zeros((16, 16 * len(greys), 3), np.uint8)
    for i, g in enumerate(greys):
        img[:, 16 * i:16 * (i + 1)] = g
    p = tmp_path / "steps.png"
    Image.fromarray(img).save(p)
    tone = ToneCurve.reference()
    res = bp.process(str(p), 0.1 * (img.shape[1] - 1), quantize_colors=16, smooth_sigma=0, interpolate=True,
                     engrave=True, gaussian_px=0, mesh_pitch_mm=0.1, height_mode="luminance", tone=tone)
    z_lo, z_hi = engrave_floor(sch), sch.top_z(27)
    for i, g in enumerate(greys):
        L = srgb_to_lab_d65(np.array([[g, g, g]]) / 255.0)[0, 0]
        expect_h = float(tone.height(L, z_lo, z_hi))
        expect_k = int(tone.layer(L, sch, z_lo))
        col = res.height_mm[:, 16 * i + 8]
        # (1.1e-3: a patch that happens to fall within 1 um of a slice plane is nudged 1 um off it)
        assert np.allclose(col, expect_h, atol=1.1e-3), (g, col[0], expect_h)
        assert int(np.median(res.k_map[:, 16 * i + 8])) == expect_k
        assert np.array_equal(res.matched_rgb[8, 16 * i + 8], ramp.rgb[expect_k - 9])
    # black -> inside layer 6 (reference black floor ~0.58 mm, above the 0.521 floor);
    # white (L* 100) -> layer 25/26 like the reference (its L26/L27 carry 0.6 %/0.1 %)
    assert int(res.k_map[8, 8]) == 9
    assert sch.top_z(5) + 0.04 < res.height_mm[8, 8] < sch.top_z(6) + 0.04
    assert sch.top_z(24) + 0.04 < res.height_mm[8, -8] < sch.top_z(27) and int(res.k_map[8, -8]) in (25, 26)
    # the ramp's top layer is reached with a curve that spans the whole range
    res_full = bp.process(str(p), 0.1 * (img.shape[1] - 1), quantize_colors=16, smooth_sigma=0, interpolate=True,
                          engrave=True, gaussian_px=0, mesh_pitch_mm=0.1, height_mode="luminance",
                          tone="-0.13,1.0,1.55")
    assert res_full.height_mm[8, -8] == pytest.approx(sch.top_z(27), abs=1e-6) and int(res_full.k_map[8, -8]) == 27
    # linear curve: mid grey (L* 50) sits at mid relief
    res_lin = bp.process(str(p), 0.1 * (img.shape[1] - 1), quantize_colors=16, smooth_sigma=0, interpolate=True,
                         engrave=True, gaussian_px=0, mesh_pitch_mm=0.1, height_mode="luminance", tone="linear")
    L119 = srgb_to_lab_d65(np.array([[140, 140, 140]]) / 255.0)[0, 0]
    assert res_lin.height_mm[8, 16 * 3 + 8] == pytest.approx(z_lo + (z_hi - z_lo) * L119 / 100, abs=1e-5)


def test_tone_curve_api():
    from core.band.tone import ToneCurve
    t = ToneCurve.from_any("0,1,1")
    assert t == ToneCurve.linear() == ToneCurve(0.0, 1.0, 1.0)
    assert ToneCurve.from_any("reference") == ToneCurve() == ToneCurve.from_any(None)
    assert ToneCurve.from_any({"q_lo": -0.13, "q_hi": 1.06, "gamma": 1.55}) == ToneCurve.reference()
    tt = ToneCurve.linear().t(np.array([-10.0, 0.0, 25.0, 100.0, 140.0]))
    np.testing.assert_allclose(tt, [0, 0, 0.25, 1, 1])
    ref = ToneCurve.reference()
    L = np.linspace(0, 100, 201)
    assert np.all(np.diff(ref.t(L)) >= 0) and ref.t(0.0) > 0 and ref.t(100.0) < 1
    with pytest.raises(ValueError):
        ToneCurve(1.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        ToneCurve(0.0, 1.0, 0.0)


def test_luma_weight_prevents_chroma_attraction(tmp_path, filaments_json):
    """(height_mode='match') A blue patch lies off the warm ramp; with the
    luminance-weighted matcher it must land on the row of matching lightness,
    while plain Lab distance pulls it towards the least saturated (whitest) rows."""
    from core.band.optics import srgb_to_lab_d65
    bp, sch, lib, ramp = _processor(tmp_path, filaments_json)

    blue = (60, 110, 200)
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = blue
    p = tmp_path / "blue.png"
    Image.fromarray(img).save(p)
    kw = dict(quantize_colors=8, smooth_sigma=0, interpolate=False, engrave=False,
              gaussian_px=0, mesh_pitch_mm=0.1, height_mode="match")
    res_w = bp.process(str(p), 12.0, luma_weight=5.0, **kw)
    res_1 = bp.process(str(p), 12.0, luma_weight=1.0, **kw)

    L_blue = srgb_to_lab_d65(np.array([blue]) / 255.0)[0, 0]
    ramp_L = srgb_to_lab_d65(ramp.rgb / 255.0)[:, 0]
    k_expect = 9 + int(np.argmin(np.abs(ramp_L - L_blue)))
    k_w = int(np.median(res_w.k_map[:, :24]))
    k_1 = int(np.median(res_1.k_map[:, :24]))
    assert abs(k_w - k_expect) <= 1, (k_w, k_expect)
    # plain Lab distance is dominated by the hue mismatch (b* -50 vs a warm
    # ramp) and lands on the least-yellow row - a much worse lightness match
    assert k_1 != k_w, (k_1, k_w)
    assert abs(ramp_L[k_1 - 9] - L_blue) > abs(ramp_L[k_w - 9] - L_blue) + 5.0, (k_1, k_w)
    # the black half stays on the base row in both cases
    assert int(np.median(res_w.k_map[:, 40:])) == 9
    assert int(np.median(res_1.k_map[:, 40:])) == 9


def test_convert_album_end_to_end(tmp_path, filaments_json, gradient_image):
    from core.band.pipeline import convert_album_to_band_3mf

    out = tmp_path / "out"
    res = convert_album_to_band_3mf(
        gradient_image, width_mm=24.0, filaments_json=filaments_json, schedule=_schedule(),
        mesh_pitch_mm=0.2, engrave=True, out_dir=str(out), advisory="br", quantize_colors=64,
    )
    # ramp LUT defaults to the output directory (not lut-npy预设/Custom) and names the model
    assert res["ramp_npz"].startswith(str(out)) and os.path.exists(res["ramp_npz"])
    assert os.path.basename(res["ramp_npz"]) == f"Band_{_schedule().key()}_bl7.npz"
    assert os.path.exists(res["threemf"]) and res["threemf"].endswith(".3mf")
    assert res["stats"]["height_mode"] == "luminance"
    assert res["stats"]["slice_plane_hits"] == 0
    assert os.path.exists(res["preview_png"])
    assert os.path.exists(res["schedule_json"]) and os.path.exists(res["swaps_txt"])
    assert res["schedule"]["layer_counts"] == [9, 4, 6, 8]
    assert [round(z, 6) for z, _, _ in res["stats"]["swap_entries"]] == [0.88, 1.20, 1.68]
    assert [e for _, e, _ in res["stats"]["swap_entries"]] == [2, 3, 4]
    assert res["stats"]["mesh_watertight"]

    with zipfile.ZipFile(res["threemf"]) as z:
        names = set(z.namelist())
        assert "Metadata/custom_gcode_per_layer.xml" in names
        assert "3D/Objects/object_1.model" in names
        xml = z.read("Metadata/custom_gcode_per_layer.xml").decode()
        assert xml.count("<layer ") == 3
        assert 'type="2"' in xml and "MultiAsSingle" in xml
        ms = z.read("Metadata/model_settings.config").decode()
        assert 'key="extruder" value="1"' in ms
        ps = json.loads(z.read("Metadata/project_settings.config"))
        assert len(ps["filament_colour"]) == 4
        assert ps["filament_colour"] == ["#000000", "#C12E1F", "#FF9016", "#FFFFFF"]
        assert ps["enable_prime_tower"] == "0"
        assert ps["initial_layer_print_height"] == "0.16"
        assert ps["layer_height"] == "0.08"

    # schedule json round-trip
    with open(res["schedule_json"]) as fh:
        js = json.load(fh)
    assert js["schedule"]["filament_names"] == ["Black", "Red", "Sunny Orange", "White"]
    assert abs(sum(js["stats"]["top_surface_fraction_per_layer"].values()) - 1.0) < 1e-6


def test_render_preview_and_stamp(filaments_json):
    from core.band.optics import BeerLambertModel, load_filament_library
    from core.band.pipeline import apply_advisory_stamp, render_preview

    lib = load_filament_library(filaments_json)
    sch = _schedule()
    h = np.linspace(sch.top_z(9), sch.top_z(27), 60 * 60, dtype=np.float32).reshape(60, 60)
    img = render_preview(h, sch, lib, BeerLambertModel())
    arr = np.asarray(img)
    assert arr.shape == (60, 60, 3)
    # dark base at the low end, white at the high end
    assert arr[0, 0].mean() < arr[-1, -1].mean()
    h2, m2 = apply_advisory_stamp(h, np.ones_like(h, bool), sch, "br")
    assert h2.shape == h.shape and m2.all()
    # stamp box: LW = 0.13*W, LH = 0.6*LW, margin 0.035*W from the bottom/right
    W = h.shape[1]
    LW, LH, m2 = max(int(0.13 * W), 8), max(int(0.60 * max(int(0.13 * W), 8)), 4), int(0.035 * W)
    box = h2[h.shape[0] - m2 - LH:h.shape[0] - m2, W - m2 - LW:W - m2]
    assert box.shape == (LH, LW)
    near_dark = np.abs(box - sch.top_z(9)) < 1e-5
    near_light = np.abs(box - sch.top_z(27)) < 1e-5
    assert np.all(near_dark | near_light) and near_dark.any() and near_light.any()


def test_cli_parses_and_runs(tmp_path, filaments_json, gradient_image, monkeypatch):
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import album_plaque
    out = tmp_path / "cli_out"
    rc = album_plaque.main([
        "--image", gradient_image, "--width", "24", "--palette", "Black,Red,Sunny Orange,White",
        "--order", "fixed", "--pitch", "0.2", "--out", str(out), "--filaments", filaments_json,
        "--quantize", "64", "--advisory", "none", "--ramp-dir", str(out),
    ])
    assert rc == 0
    assert any(p.endswith(".3mf") for p in os.listdir(out))


def test_cli_layers_follow_their_filaments_under_auto_order(tmp_path, filaments_json, gradient_image):
    """--layers counts are attached to the filaments named in --palette even when
    --order auto re-sorts the palette by luminance."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import album_plaque
    out = tmp_path / "cli_out2"
    rc = album_plaque.main([
        "--image", gradient_image, "--width", "24", "--palette", "White,Red,Black,Sunny Orange",
        "--layers", "8,4,9,6", "--order", "auto", "--pitch", "0.2", "--out", str(out),
        "--filaments", filaments_json, "--quantize", "16", "--advisory", "none",
    ])
    assert rc == 0
    with open(out / "gradient_band_schedule.json") as fh:
        js = json.load(fh)
    assert js["schedule"]["filament_names"] == ["Black", "Red", "Sunny Orange", "White"]
    assert js["schedule"]["layer_counts"] == [9, 4, 6, 8]
    assert [round(z, 6) for z, _, _ in js["stats"]["swap_entries"]] == [0.88, 1.20, 1.68]
