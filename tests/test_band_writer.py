"""Band mode: standalone 3MF writer (core/band/writer3mf.py)."""
import json
import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

from core.band.heightfield_mesh import heightfield_to_trimesh
from core.band import writer3mf
from core.band.writer3mf import (
    DEFAULT_TEMPLATE_SOURCES, TEMPLATE_CACHE, build_project_settings,
    classify_filament_keys, load_template, write_band_3mf, validate_swap_entries,
)

try:  # builder A's dataclass, if present
    from core.band.optics import Filament as _Filament

    def make_fil(name, hx):
        return _Filament(name=name, hex=hx, rgb_lin=np.zeros(3), td_mm=1.0)
except Exception:  # pragma: no cover - fallback keeps this test independent
    @dataclass
    class _Fake:
        name: str
        hex: str
        settings_id: str = 'Bambu PLA Basic @BBL X2D 0.4 nozzle'
        filament_id: str = 'GFA00'

    def make_fil(name, hx):
        return _Fake(name, hx)


FILS = [make_fil('Black', '#000000'), make_fil('Red', '#C12E1F'),
        make_fil('Sunny Orange', '#FF9016'), make_fil('White', '#FFFFFF')]
SWAPS = [(0.88, 2, '#C12E1F'), (1.20, 3, '#FF9016'), (1.68, 4, '#FFFFFF')]
F = len(FILS)

_sources_present = all(os.path.isfile(p) for p in DEFAULT_TEMPLATE_SOURCES)


@pytest.fixture(scope='module')
def written(tmp_path_factory):
    rng = np.random.default_rng(1)
    h = 0.6 + rng.random((20, 20)) * 1.6
    mesh = heightfield_to_trimesh(h, 0.15)
    out = str(tmp_path_factory.mktemp('band') / 'tiny_band.3mf')
    path = write_band_3mf(mesh, out, FILS, SWAPS, title='Tiny', description='desc & test',
                          size_mm=19 * 0.15)
    assert path == out and os.path.isfile(out)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        files = {n: zf.read(n) for n in names}
    return files


def test_package_contents(written):
    expected = {'[Content_Types].xml', '_rels/.rels', '3D/3dmodel.model',
                '3D/_rels/3dmodel.model.rels', '3D/Objects/object_1.model',
                'Metadata/model_settings.config', 'Metadata/custom_gcode_per_layer.xml',
                'Metadata/project_settings.config', 'Metadata/filament_sequence.json',
                'Metadata/slice_info.config', 'Metadata/cut_information.xml'}
    assert expected == set(written)  # no thumbnails when none given


def test_single_object_mesh_with_3dp_vertices(written):
    obj = written['3D/Objects/object_1.model'].decode()
    assert obj.count('<object ') == 1
    assert '<object id="1"' in obj
    verts = re.findall(r'<vertex x="([-\d.]+)" y="([-\d.]+)" z="([-\d.]+)"/>', obj)
    assert len(verts) == 2 * 20 * 20
    assert all(len(v.split('.')[1]) == 3 for row in verts for v in row)
    assert min(float(v[2]) for v in verts) == 0.0
    assert obj.count('<triangle ') == 4 * 19 * 19 + 2 * (2 * 19 + 2 * 19)

    main = written['3D/3dmodel.model'].decode()
    assert 'BambuStudio-02.08.02.61' in main
    assert '<metadata name="Title">Tiny</metadata>' in main
    assert 'desc &amp; test' in main
    assert main.count('<component ') == 1 and 'objectid="1"' in main
    m = re.search(r'<item objectid="2" p:UUID="[^"]+" transform="1 0 0 0 1 0 0 0 1 ([\d.]+) ([\d.]+) 0"', main)
    assert m, main
    size = 19 * 0.15
    assert float(m.group(1)) == pytest.approx((256 - size) / 2, abs=1e-3)
    assert float(m.group(2)) == pytest.approx((256 - size) / 2, abs=1e-3)


def test_model_settings_single_part_extruder1_filament_maps(written):
    ms = written['Metadata/model_settings.config'].decode()
    assert ms.count('<part ') == 1
    assert ms.count('<object ') == 1
    assert re.search(r'<object id="2">\s*<metadata key="name" value="Tiny"/>\s*<metadata key="extruder" value="1"/>', ms)
    part = ms[ms.index('<part '):ms.index('</part>')]
    assert '<metadata key="extruder" value="1"/>' in part
    maps = re.search(r'<metadata key="filament_maps" value="([^"]+)"/>', ms).group(1).split()
    assert maps == ['1'] * F
    assert 'filament_map_mode" value="Auto For Flush"' in ms
    assert '<model_instance>' in ms
    assert 'thumbnail_file' not in ms


def test_custom_gcode_per_layer(written):
    xml = written['Metadata/custom_gcode_per_layer.xml'].decode()
    layers = re.findall(r'<layer top_z="([\d.]+)" type="(\d)" extruder="(\d+)" color="(#[0-9A-F]{6})" extra="" gcode="tool_change"/>', xml)
    assert len(layers) == 3
    zs = [float(l[0]) for l in layers]
    assert zs == sorted(zs)
    assert zs == pytest.approx([0.88, 1.20, 1.68], abs=1e-6)
    assert [l[1] for l in layers] == ['2', '2', '2']
    assert [int(l[2]) for l in layers] == [2, 3, 4]
    assert [l[3] for l in layers] == ['#C12E1F', '#FF9016', '#FFFFFF']
    assert '<mode value="MultiAsSingle"/>' in xml
    assert '<plate_info id="1"/>' in xml


def test_project_settings_profile_and_filament_arrays(written):
    cfg = json.loads(written['Metadata/project_settings.config'])
    assert cfg['filament_colour'] == ['#000000', '#C12E1F', '#FF9016', '#FFFFFF']
    assert len(cfg['filament_colour']) == F
    assert len(cfg['flush_volumes_matrix']) == 2 * F * F
    assert len(cfg['flush_volumes_vector']) == 2 * F
    mtx = np.array(cfg['flush_volumes_matrix']).reshape(2, F, F)
    assert (mtx[0].diagonal() == '0').all() and (mtx[1].diagonal() == '0').all()
    assert cfg['enable_prime_tower'] == '0'
    assert cfg['initial_layer_print_height'] == '0.16'
    assert cfg['layer_height'] == '0.08'
    assert cfg['wall_loops'] == '2'
    assert cfg['top_shell_layers'] == '9999'
    assert cfg['bottom_shell_layers'] == '9999'
    assert cfg['sparse_infill_density'] == '100%'
    assert cfg['sparse_infill_pattern'] == 'zig-zag'
    assert cfg['top_surface_pattern'] == 'monotonicline'
    assert cfg['ironing_type'] == 'no ironing'
    assert cfg['printer_settings_id'] == 'Bambu Lab X2D 0.4 nozzle'
    assert cfg['print_settings_id'] == '0.08mm High Quality @BBL X2D'
    assert cfg['printer_model'] == 'Bambu Lab X2D'
    assert cfg['filament_settings_id'] == ['Bambu PLA Basic @BBL X2D 0.4 nozzle'] * F
    assert cfg['filament_ids'] == ['GFA00'] * F
    assert cfg['filament_self_index'] == ['1', '2', '3', '4']
    assert cfg['filament_map'] == ['1'] * F
    assert 'initial_layer_height' not in cfg
    assert len(cfg['different_settings_to_system']) == F + 2
    assert len(cfg['inherits_group']) == F + 2
    assert all(isinstance(v, (str, list)) for v in cfg.values())

    # every per-filament array length == multiplier * F, using the cached template
    tpl = load_template()
    mult = tpl['per_filament_multiplier']
    assert 'filament_colour' in mult and mult['filament_colour'] == 1
    assert mult.get('nozzle_temperature') == 1
    assert mult.get('filament_flush_temp') == 6
    bad = {k: (len(cfg[k]), m) for k, m in mult.items() if len(cfg[k]) != m * F}
    assert not bad, bad
    # non-per-filament lists keep their template length
    for k, v in tpl['base'].items():
        if isinstance(v, list) and k not in mult and k not in (
                'flush_volumes_matrix', 'flush_volumes_vector',
                'different_settings_to_system', 'inherits_group', 'print_compatible_printers'):
            assert len(cfg[k]) == len(v), k
    assert cfg['printable_area'] == ['0x0', '256x0', '256x256', '0x256']


@pytest.mark.skipif(not _sources_present, reason='known-good X2D 3mfs not available')
def test_multiplier_classification_against_known_good_configs(written):
    configs = [writer3mf._read_config_from_3mf(p) for p in DEFAULT_TEMPLATE_SOURCES]
    mult = classify_filament_keys(configs)
    cfg = json.loads(written['Metadata/project_settings.config'])
    for k, m in mult.items():
        assert isinstance(cfg[k], list) and len(cfg[k]) == m * F, (k, m, len(cfg[k]))
    assert mult['filament_settings_id'] == 1
    assert mult['filament_flush_volumetric_speed'] == 6
    assert mult['filament_dev_ams_drying_temperature'] == 4
    assert 'flush_volumes_matrix' not in mult and 'nozzle_flush_dataset' not in mult
    # writer also works with F=5 (X2D two extruders: 2*25 / 10)
    cfg5 = build_project_settings(load_template(), FILS + [make_fil('Pink', '#F5A3B7')])
    assert len(cfg5['flush_volumes_matrix']) == 50 and len(cfg5['flush_volumes_vector']) == 10
    assert all(len(cfg5[k]) == m * 5 for k, m in mult.items())


def test_template_cache_exists_and_is_keyed_on_sources(tmp_path):
    assert os.path.isfile(TEMPLATE_CACHE)
    # default sources (or none) -> served from the cache, even if the 3mfs are absent
    tpl = load_template()
    assert tpl['n_extruders'] == 2 and tpl['per_filament_multiplier']
    assert load_template(template_sources=DEFAULT_TEMPLATE_SOURCES)['sources'] == tpl['sources']
    # a different explicit source list must NOT be served from the cache
    with pytest.raises(FileNotFoundError):
        load_template(template_sources=[str(tmp_path / 'missing.3mf')])
    with pytest.raises(FileNotFoundError):
        load_template(template_sources=[str(tmp_path / 'missing.3mf')],
                      cache_path=str(tmp_path / 'nocache.json'))


@pytest.mark.skipif(not _sources_present, reason='known-good X2D 3mfs not available')
def test_template_sources_override_rebuilds_without_touching_cache(tmp_path):
    before = os.path.getmtime(TEMPLATE_CACHE)
    tpl = load_template(template_sources=DEFAULT_TEMPLATE_SOURCES[:2] + DEFAULT_TEMPLATE_SOURCES[2:])
    assert tpl['sources'] == [os.path.basename(p) for p in DEFAULT_TEMPLATE_SOURCES]
    # a genuinely different (sub)list is rebuilt from the files and not cached over the default
    tpl2 = load_template(template_sources=[DEFAULT_TEMPLATE_SOURCES[0], DEFAULT_TEMPLATE_SOURCES[2]])
    assert tpl2['sources'] == ['Blonde.3mf', 'Love_Sick.3mf'] and tpl2['source_filament_counts'] == [2, 4]
    assert os.path.getmtime(TEMPLATE_CACHE) == before


def test_print_preset_does_not_inherit_itself(written):
    cfg = json.loads(written['Metadata/project_settings.config'])
    parent = cfg['inherits_group'][0]
    assert parent != cfg['print_settings_id']
    # the parent recorded by Bambu Studio in every known-good source file
    assert parent == load_template()['base']['inherits_group'][0] == '0.08mm Extra Fine @BBL A1'
    assert cfg['different_settings_to_system'][0].startswith('bottom_shell_layers;')
    assert cfg['inherits_group'][1:] == [''] * (F + 1)


def test_swap_validation():
    assert validate_swap_entries(SWAPS, 4, 0.16, 0.08) == [10, 14, 20]
    with pytest.raises(ValueError):
        validate_swap_entries([(0.90, 2, '#FF0000')], 4, 0.16, 0.08)  # mid-layer
    with pytest.raises(ValueError):
        validate_swap_entries([(0.88, 5, '#FF0000')], 4, 0.16, 0.08)  # extruder > F
    with pytest.raises(ValueError):
        validate_swap_entries([(1.20, 2, '#FF0000'), (0.88, 3, '#00FF00')], 4, 0.16, 0.08)
    with pytest.raises(ValueError):
        validate_swap_entries([(0.16, 2, '#FF0000')], 4, 0.16, 0.08)  # n must be >= 2


def test_thumbnail_optional(tmp_path):
    from PIL import Image
    png = tmp_path / 't.png'
    Image.new('RGB', (64, 40), (200, 30, 30)).save(png)
    mesh = heightfield_to_trimesh(np.full((4, 4), 1.0), 0.5)
    out = write_band_3mf(mesh, str(tmp_path / 'thumb.3mf'), FILS, SWAPS, 'T', 'd', 1.5,
                         thumbnail_png=str(png))
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        ms = zf.read('Metadata/model_settings.config').decode()
        rels = zf.read('_rels/.rels').decode()
    assert {'Metadata/plate_1.png', 'Metadata/plate_1_small.png'} <= names
    assert 'thumbnail_file' in ms and 'plate_1_small.png' in rels
