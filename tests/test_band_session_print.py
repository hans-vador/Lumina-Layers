"""Regression coverage for Studio's physical preview and export contract."""
import json
import struct
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest
import trimesh

from core.band.optics import Filament, linear_to_srgb
from core.band.pipeline import composite_heights
from core.band.session import BandSession
from core.band.tone import ToneCurve


@pytest.fixture
def session():
    lightness = np.tile(np.linspace(20, 80, 24, dtype=np.float32), (16, 1))
    lib = {f.name: f for f in (Filament.from_hex('Black', '#111111', .2),
                               Filament.from_hex('White', '#FFFFFF', 4))}
    s = BandSession('gradient.png', lightness, np.zeros((16, 24, 3), np.uint8),
                    np.ones((16, 24), bool), .1, lib, ['Black', 'White'], [5, 12],
                    width_mm=2.4, first_layer_mm=.16, layer_h=.08,
                    tone=ToneCurve(0, 1, 1))
    s._refresh_small()
    return s


def test_mesh_is_closed_exact_size_and_uses_edited_heights(session):
    original, _ = session.print_mesh(.1)
    session.negative = True
    session.full_range = True
    session.border_width_mm = .2
    session.border_depth_mm = .1
    mesh, _ = session.print_mesh(.1)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    np.testing.assert_allclose(mesh.extents[:2], [2.4, 1.6])
    np.testing.assert_allclose(mesh.vertices[:384, 2], session.height_map().ravel())
    assert not np.allclose(mesh.vertices[:, 2], original.vertices[:, 2])
    for detail in (8, 16, 100):
        preview = session.mesh3d(detail)
        np.testing.assert_allclose(preview['size_mm'][:2], [2.4, 1.6])


def test_preview_ramp_is_optical_color_at_physical_height(session):
    data = session.mesh3d(16)
    zs = np.linspace(0, session.schedule.total_height_mm, 2048)
    expected = np.round(linear_to_srgb(composite_heights(
        zs, session.schedule, session.library, session.model)) * 255).astype(np.uint8)
    np.testing.assert_array_equal(data['ramp'], expected)
    session.lighting = 'back'
    back = session.mesh3d(16)
    assert back['backlit']
    assert not np.array_equal(back['ramp'], data['ramp'])
    session.led_hex = '#FF2020'
    assert not np.array_equal(back['ramp'], session.mesh3d(16)['ramp'])


def test_backlight_includes_entire_first_layer(session):
    # A thick first layer still belongs to the base, with no gap in absorption.
    session.first_layer_mm = .4
    session.layer_counts = [1, 12]
    h = np.array([.2, .35])
    rgb = session.library['Black'].rgb_lin
    from core.band.optics import srgb_to_linear, hex_to_rgb01
    led = srgb_to_linear(hex_to_rgb01(session.led_hex))
    expected = np.exp(-session.k_opaque * h[:, None] * (1-rgb) / .2) * led
    np.testing.assert_allclose(session.transmitted_heights(h), expected)


def test_stl_and_3mf_preserve_edits_and_layer_settings(session, tmp_path):
    session.negative = True
    session.full_range = True
    session.border_width_mm = .2
    session.border_depth_mm = .12
    stl = session.export_stl(str(tmp_path), mesh_pitch_mm=.1)
    package = session.export(str(tmp_path), mesh_pitch_mm=.1)
    expected, _ = session.print_mesh(.1)
    loaded = trimesh.load(stl['stl'], force='mesh')
    assert loaded.is_watertight
    np.testing.assert_allclose(loaded.extents, expected.extents, atol=1e-6)
    with zipfile.ZipFile(package['threemf']) as z:
        cfg = json.loads(z.read('Metadata/project_settings.config'))
        assert float(cfg['initial_layer_print_height']) == session.first_layer_mm
        assert float(cfg['layer_height']) == session.layer_h
        root = ET.fromstring(z.read('3D/Objects/object_1.model'))
        ns = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
        vertices = [[float(v.get(k)) for k in ('x', 'y', 'z')]
                    for v in root.findall('.//m:vertex', ns)]
        # The 3MF writer serializes coordinates to 0.001 mm.
        np.testing.assert_allclose(vertices, expected.vertices, atol=0.000501)
        faces = [[int(t.get(k)) for k in ('v1', 'v2', 'v3')]
                 for t in root.findall('.//m:triangle', ns)]
        np.testing.assert_array_equal(faces, expected.faces)
        swaps = ET.fromstring(z.read('Metadata/custom_gcode_per_layer.xml'))
        entries = swaps.findall('.//layer')
        assert len(entries) == session.schedule.n_bands - 1
        assert float(entries[0].get('top_z')) == session.schedule.swap_entries(session.library)[0][0]


def test_mesh_api_binary_and_boot_settings(session, monkeypatch):
    from fastapi.testclient import TestClient
    from core.band import studio_api
    monkeypatch.setattr(studio_api.STUDIO, 'session', session)
    monkeypatch.setattr(studio_api.STUDIO, 'library', session.library)
    with TestClient(studio_api.create_app()) as client:
        boot = client.get('/api/boot').json()
        assert boot['settings']['width_mm'] == 2.4
        assert boot['settings']['first_layer_mm'] == .16
        req = dict(names=session.filament_names, counts=session.layer_counts,
                   **boot['settings'], detail=60)
        response = client.post('/api/mesh', json=req)
        assert response.status_code == 200
        data = response.content
        n = struct.unpack('<I', data[:4])[0]
        assert (4+n) % 4 == 0
        head = json.loads(data[4:4+n])
        assert len(head['ramp']) == 2048
        assert len(data) == 4+n+head['nv']*15+head['nf']*12
