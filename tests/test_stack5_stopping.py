import json
import re
import zipfile
import numpy as np
import pytest
from PIL import Image
from core.band.optics import BeerLambertModel, Filament, linear_to_srgb
from core.stack5.stopping import stop_at_best_match
from core.stack5.pipeline import convert_album_stack5

@pytest.mark.parametrize('metric',['lumina','lab','hue'])
@pytest.mark.parametrize('layer_h',[.04,.08])
def test_exact_shades_stop_at_zero_through_five_supported_layers(metric,layer_h):
    fils=[Filament.from_hex('Black','#000000',.15),Filament.from_hex('White','#FFFFFF',5)]
    model=BeerLambertModel()
    # White over black at six physical thicknesses. Same white filament in
    # adjacent layers must NOT be collapsed unless its colour already matches.
    shades=np.array([np.round(linear_to_srgb(np.full(3,model.opacity(i*layer_h,5)))*255) for i in range(6)],np.uint8)[None]
    result={'material_matrix':np.ones((1,6,5),np.int32),'mask_solid':np.ones((1,6),bool),
            'quantized_image':shades,'matched_rgb':np.broadcast_to(shades[:,-1:],shades.shape).copy()}
    mm,preview,heights,report=stop_at_best_match(result,fils,0,model,layer_h,metric)
    assert heights.tolist()==[[0,1,2,3,4,5]]
    assert np.array_equal(preview,shades)
    assert np.all(np.diff((mm>=0).astype(int),axis=-1)>=0) # air only above material
    assert report['optical_voxels_after']==15
    assert report['max_match_cost_increase']<=1e-8
    assert np.all(result['material_matrix']==1) # no mutation of baseline


def test_transparency_is_not_backing_and_black_ties_choose_zero():
    fils=[Filament.from_hex('Black','#000000',.15)]
    mm=np.zeros((1,2,5),np.int32);mm[0,1]=-1
    result={'material_matrix':mm,'mask_solid':np.array([[True,False]]),
            'quantized_image':np.zeros((1,2,3),np.uint8),'matched_rgb':np.full((1,2,3),77,np.uint8)}
    mm,preview,heights,report=stop_at_best_match(result,fils,0,BeerLambertModel(),.08)
    assert np.all(mm==-1)
    assert np.all(preview[0,0]==0) and np.all(preview[0,1]==77)
    assert report['pixels_by_colour_layer_count']==[1,0,0,0,0,0]

@pytest.mark.parametrize('early_stop',[True,False])
def test_all_black_export_has_only_backing_when_stopped(tmp_path,early_stop):
    image=tmp_path/'black.png';Image.new('RGB',(16,16),'black').save(image)
    out=convert_album_stack5(str(image),width_mm=1.6,palette=['Black','White'],backing='Black',
        orientation='face-up',backing_layer_h=None,layer_h=.04,first_layer_mm=.16,spacer_mm=1.6,early_stop=early_stop,
        out_dir=str(tmp_path/'out'),smooth_sigma=0,min_region_px=0,metric='lab',tower_fit='auto')
    recipe=json.load(open(out['recipe_json']))
    height=1.6 if early_stop else 1.8
    assert recipe['stats']['total_height_mm']==pytest.approx(height)
    assert recipe['stats']['total_print_layers']==(37 if early_stop else 42)
    assert recipe['stats']['per_slot_viewing_surface_share']['Black']==1
    assert recipe['settings']['early_stop']==early_stop
    if early_stop:
        assert not np.load(recipe['outputs']['stop_layers']).any()
        assert recipe['stats']['early_stopping']['optical_material_saved_fraction']==1
    with zipfile.ZipFile(out['threemf']) as z:
        data=b''.join(z.read(n) for n in z.namelist() if n.startswith('3D/Objects/') and n.endswith('.model'))
        zs=[float(x) for x in re.findall(rb'<vertex [^>]*z="([^"]+)"',data)]
        assert min(zs)==0 and max(zs)==pytest.approx(height)


def test_stopped_column_mesh_has_no_cap_over_dark_region(tmp_path):
    import trimesh
    import xml.etree.ElementTree as ET
    image=tmp_path/'split.png'
    img=np.zeros((40,80,3),np.uint8);img[:,40:]=255;Image.fromarray(img).save(image)
    out=convert_album_stack5(str(image),width_mm=8,palette=['Black','White'],backing='Black',
        orientation='face-up',backing_layer_h=None,layer_h=.08,first_layer_mm=.16,spacer_mm=1.6,
        out_dir=str(tmp_path/'out'),smooth_sigma=0,min_region_px=0,metric='lab',tower_fit='auto')
    with zipfile.ZipFile(out['threemf']) as z:
        for name in z.namelist():
            if not (name.startswith('3D/Objects/') and name.endswith('.model')):continue
            root=ET.fromstring(z.read(name));ns={'m':'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}
            for obj in root.findall('.//m:object',ns):
                verts=np.array([[float(v.get(k)) for k in ('x','y','z')] for v in obj.findall('.//m:vertex',ns)])
                faces=np.array([[int(t.get(k)) for k in ('v1','v2','v3')] for t in obj.findall('.//m:triangle',ns)])
                mesh=trimesh.Trimesh(verts,faces,process=False)
                assert mesh.is_winding_consistent
                assert mesh.volume>0
                # Source black is on the left: no material above the base in
                # that half. White has actual material to the full shell height.
                high=verts[verts[:,2]>1.60001]
                if len(high): assert high[:,0].min()>=3.9

@pytest.mark.parametrize('metric',['lab','hue','lumina'])
def test_independent_pink_blue_same_height_and_dark_blue_blend(metric):
    from core.stack5.lut import synth_lut
    fils=[Filament.from_hex('Black','#000000',.15),
          Filament.from_hex('Pink','#F5A3B7',.3),
          Filament.from_hex('Blue','#1E44BE',.3)]
    model=BeerLambertModel()
    one,_=synth_lut(fils,0,model,layers=1,layer_h=.04)
    two,recipes=synth_lut(fils,0,model,layers=2,layer_h=.04)
    dark=two[np.flatnonzero(np.all(recipes==[0,2],axis=1))[0]]
    targets=np.array([[one[1],one[2],dark]],np.uint8)
    # Deliberately useless original stack: global matching must replace it,
    # not trim its prefixes. Pink and blue both sit on black backing here.
    result={'material_matrix':np.zeros((1,3,5),np.int32),'mask_solid':np.ones((1,3),bool),
            'quantized_image':targets,'matched_rgb':np.zeros_like(targets)}
    mm,preview,height,report=stop_at_best_match(result,fils,0,model,.04,metric)
    assert height[0,:2].tolist()==[1,1]
    assert mm[0,0,-1]==1 and mm[0,1,-1]==2
    used=mm[0,2][mm[0,2]>=0]
    assert 0 in used and 2 in used
    assert np.all(np.diff((mm>=0).astype(int),axis=-1)>=0)
    assert report['candidate_count']==sum(3**i for i in range(6))


def test_cleanup_copies_supported_whole_columns_and_keeps_preview_consistent():
    from core.stack5.stopping import variable_stack_candidates
    fils=[Filament.from_hex('Black','#000000',.15),Filament.from_hex('Blue','#1E44BE',.3)]
    model=BeerLambertModel()
    candidates,colours,_,_=variable_stack_candidates(fils,0,model,.04,5)
    ix=np.flatnonzero(np.all(candidates==[-1,-1,-1,-1,1],axis=1))[0]
    target=np.zeros((10,10,3),np.uint8);target[5,5]=colours[ix]
    result={'material_matrix':np.zeros((10,10,5),np.int32),'mask_solid':np.ones((10,10),bool),
            'quantized_image':target,'matched_rgb':np.zeros_like(target)}
    mm,preview,height,report=stop_at_best_match(result,fils,0,model,.04,min_region_px=4)
    assert np.all(np.diff((mm>=0).astype(int),axis=-1)>=0)
    assert not height.any() and not preview.any()
    assert report['whole_recipe_cleanup']['reassigned_px_total']>0
