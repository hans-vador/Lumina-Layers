import json
import zipfile
import numpy as np
from PIL import Image
from core.stack5.edges import contrast_edges
from core.stack5.cleanup import min_region_cleanup
from core.stack5.pipeline import convert_album_stack5, default_filaments_json
from core.band.optics import load_filament_library, BeerLambertModel


def test_contrast_boundary_protects_both_sides_but_not_flat_interiors():
    rgb = np.full((12,12,3), 255, np.uint8)
    rgb[:,6:] = 0
    edges = contrast_edges(rgb)
    assert edges[:,5:7].all()
    assert not edges[:,:5].any() and not edges[:,7:].any()
    gradient = np.broadcast_to(np.arange(12, dtype=np.uint8)[None,:,None]+100, (12,12,3))
    assert not contrast_edges(gradient).any()


def test_cleanup_keeps_protected_outline_and_removes_unprotected_noise():
    mm = np.zeros((12,12,1), np.int32)
    mm[3,3] = mm[8,8] = 1
    protected = np.zeros((12,12), bool); protected[3,3] = True
    clean, _ = min_region_cleanup(mm, np.ones((12,12),bool),16, protected_mask=protected)
    assert clean[3,3,0] == 1 and clean[8,8,0] == 0


def test_default_export_keeps_thin_outline_with_opaque_white(tmp_path):
    rgb = np.full((16,16,3),255,np.uint8)
    rgb[4:12,8] = 0  # eight-pixel line: below the 16-pixel cleanup threshold
    path = tmp_path/'outline.png'; Image.fromarray(rgb).save(path)
    result = convert_album_stack5(str(path), width_mm=1.6, palette=['Black','White'],
        backing='White', orientation='face-up', out_dir=str(tmp_path/'out'),
        metric='hue', min_region_px=16, tower_fit='auto')
    recipe = json.load(open(result['recipe_json']))
    assert recipe['settings']['first_layer_mm'] == .2
    assert recipe['stats']['actual_backing_mm'] == 1.0
    assert recipe['stats']['early_stopping']['protected_edge_pixels'] > 0
    preview = np.asarray(Image.open(result['preview_png']).convert('RGB'))
    assert preview[6:10,8].max() < 50
    assert preview[6:10,7].min() > 245
    with zipfile.ZipFile(result['threemf']) as z:
        cfg = json.loads(z.read('Metadata/project_settings.config'))
        assert float(cfg['initial_layer_print_height']) == .2
        assert 'max_z="1"' in z.read('Metadata/layer_config_ranges.xml').decode()
    white = load_filament_library(default_filaments_json())['White']
    assert white.td_mm == .1
    assert BeerLambertModel().opacity(.08,white.td_mm) > .995
