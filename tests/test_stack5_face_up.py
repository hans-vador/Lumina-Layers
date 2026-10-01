"""Physical layer boundaries and optical thickness must agree for face-up Stack5."""
import json
import re
import zipfile
import numpy as np
import pytest
from PIL import Image
from core.stack5.pipeline import convert_album_stack5, face_up_vertex_map

@pytest.mark.parametrize('height', [0.04, 0.08])
def test_rotation_preserves_backing_and_reverses_colour_order(height):
    transform = face_up_vertex_map(20, 1.6, height, 0.16)
    xml = b'\n'.join(f'<vertex x="3" y="7" z="{z}"/>'.encode() for z in [0,.08,.16,.24,.32,.4,2.0])
    result = transform(xml)
    assert re.findall(rb'x="([^"]+)"', result) == [b'17.000000'] * 7
    assert np.allclose([float(v) for v in re.findall(rb'z="([^"]+)"', result)],
                       [1.6+5*height,1.6+4*height,1.6+3*height,1.6+2*height,1.6+height,1.6,0])

@pytest.mark.parametrize('height', [0.04, 0.08])
@pytest.mark.parametrize('backing_height', [None, 0.2])
def test_face_up_export_matches_optics_and_printer_layers(tmp_path, height, backing_height):
    base = 1.56 if backing_height else 1.6
    # This geometry fixture needs translucent white to exercise every layer.
    from core.stack5.pipeline import default_filaments_json
    library = json.load(open(default_filaments_json()))
    next(f for f in library if f['name'] == 'White')['td_mm'] = 5.0
    library_path = tmp_path/'filaments.json'
    library_path.write_text(json.dumps(library))
    path=tmp_path/'asymmetric.png'
    img=np.zeros((16,16,3),np.uint8); img[:,:6]=255
    Image.fromarray(img).save(path)
    result=convert_album_stack5(str(path), width_mm=1.6, palette=['Black','White'], backing='Black',
        orientation='face-up',backing_layer_h=backing_height,layer_h=height,first_layer_mm=.16,spacer_mm=1.6,filaments_json=str(library_path),
        out_dir=str(tmp_path/'out'), smooth_sigma=0, min_region_px=0, metric='lab', tower_fit='auto')
    recipe=json.load(open(result['recipe_json']))
    assert recipe['settings']['optical_layer_thicknesses_mm']==[height]*5
    assert recipe['stats']['total_height_mm']==pytest.approx(base+height*5)
    sets=recipe['stats']['printed_layer_material_sets']
    nbase=8 if backing_height else 1+round((1.6-.16)/height)
    assert sets[:nbase]==[[0]]*nbase
    assert sets[nbase:]==recipe['stats']['materials_per_optical_layer_sets'][::-1]
    with zipfile.ZipFile(result['threemf']) as z:
        cfg=json.loads(z.read('Metadata/project_settings.config'))
        assert float(cfg['initial_layer_print_height'])==.16
        assert float(cfg['layer_height'])==height
        assert cfg['min_layer_height']==['0.08']
        if backing_height:
            ranges=z.read('Metadata/layer_config_ranges.xml').decode()
            assert 'max_z="1.56"' in ranges and '>0.2</option>' in ranges
            assert recipe['stats']['sliced_layers_est']==13
        else:
            assert 'Metadata/layer_config_ranges.xml' not in z.namelist()
        xml=b''.join(z.read(n) for n in z.namelist() if n.startswith('3D/Objects/') and n.endswith('.model'))
        zs=np.array([float(v) for v in re.findall(rb'<vertex [^>]*z="([^"]+)"',xml)])
        assert zs.min()==pytest.approx(0)
        assert zs.max()==pytest.approx(base+height*5)
        assert np.any(np.isclose(zs,base))
        assert np.allclose((zs[zs>=base]-base)/height,np.round((zs[zs>=base]-base)/height))
    data=np.load(result['lut_npz'])
    from core.band.optics import load_filament_library
    from core.stack5.lut import synth_lut
    lib=load_filament_library(recipe['settings']['filaments_json'])
    expected,_=synth_lut([lib['Black'],lib['White']],0,layer_h=height)
    assert np.array_equal(data['rgb'],expected)
