"""Color-order, project roundtrip, mapping and snapshot regression tests."""
import copy
import io
import json
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from core.band.mapping import reference_ramp, match_heights, mapped_fraction
from core.band.optics import Filament, srgb_to_lab_d65
from core.band.session import BandSession
from core.band.project import dump_project, load_project, settings
from core.band.schedule import BandSchedule
from core.band.tone import ToneCurve


@pytest.fixture
def painting(tmp_path):
    image=np.full((32,32,3),[198,175,236],np.uint8)
    image[8:24,8:24]=[37,78,233]
    path=tmp_path/'purple-blue.png';Image.fromarray(image).save(path)
    library={f.name:f for f in [Filament.from_hex('Purple','#a78bd4',2.5),Filament.from_hex('Blue','#1e44be',.3)]}
    s=BandSession.load(str(path),library,['Purple','Blue'],[9,7],width_mm=3.2,smooth_sigma=0)
    s.mesh_mode='color_match';s.engrave=False
    s.mesh_core=[{'hex':'#c6afec','td_mm':2.5,'layer':9},{'hex':'#254ee9','td_mm':.3,'layer':16}]
    return s


def test_blue_details_above_purple_base(painting):
    h=painting.height_map()
    assert h[2,2] == pytest.approx(.72)
    assert h[16,16] > h[2,2]+.08
    mesh,_=painting.print_mesh(.1)
    assert mesh.is_watertight and mesh.volume>0
    assert painting.schedule.swap_entries(painting.library)==[(.8,2,'#1E44BE')]


def test_luminance_explains_original_recess(painting):
    painting.mesh_mode='luminance'
    h=painting.height_map()
    assert h[16,16]<h[2,2]


def test_filament_changes_do_not_move_reference_geometry(painting):
    before=painting.height_map();before_preview=np.asarray(painting.preview())
    painting.library['Blue']=Filament.from_hex('Blue','#00ff00',1.5)
    np.testing.assert_array_equal(painting.height_map(),before)
    assert not np.array_equal(before_preview,np.asarray(painting.preview()))
    painting.copy_filaments_to_mesh_core()
    assert not np.array_equal(before,painting.height_map())


@pytest.mark.parametrize('metric',['rgb','cielab','hsl','dot'])
def test_metrics_recognize_reference_endpoints(metric):
    colors=np.array([[.7,.5,.8],[.1,.2,.9],[.4,.4,.4]])
    expected=np.array([.72,1.2,.96])
    rgb=colors[None,...]
    np.testing.assert_allclose(match_heights(rgb,expected,colors,metric),expected[None,...])


def test_duplicate_reference_color_chooses_lowest():
    np.testing.assert_array_equal(match_heights(np.ones((1,1,3)),np.array([.5,1,1.5]),np.ones((3,3))),[[.5]])


def test_modes_use_documented_color_regions():
    rgb=np.array([[[1.,0,0],[0,1.,0],[0,0,1.],[.5,.5,.5]]])
    light=np.full((1,4),50.)
    tone=ToneCurve.linear()
    out=mapped_fraction(rgb,light,'color_aware',tone,channel_order='BGR')
    assert out[0,0]>out[0,1]>out[0,2]
    pop=mapped_fraction(rgb,light,'color_pop',tone)
    assert pop[0,0]>pop[0,3]
    reverse=mapped_fraction(rgb,light,'color_pop',tone,reverse=True)
    assert reverse[0,0]<reverse[0,3]
    np.testing.assert_array_equal(mapped_fraction(rgb,light,'combo',tone,combo=1),tone.t(light))
    np.testing.assert_array_equal(mapped_fraction(rgb,light,'combo',tone,combo=0),rgb.max(-1))


def test_project_roundtrip_keeps_source_cores_geometry_and_filaments(painting):
    painting.negative=True;painting.border_width_mm=.2;painting.border_depth_mm=.08
    painting.min_depth_mm=.72;painting.max_depth_mm=1.28
    doc=dump_project(painting)
    loaded=load_project(json.loads(json.dumps(doc)))
    assert settings(loaded)==settings(painting)
    assert loaded.describe()==painting.describe()
    np.testing.assert_array_equal(loaded.rgb,painting.rgb)
    np.testing.assert_array_equal(loaded.height_map(),painting.height_map())
    a,_=loaded.print_mesh(.15);b,_=painting.print_mesh(.15)
    np.testing.assert_array_equal(a.vertices,b.vertices)


def test_many_bands_and_reused_spools():
    sch=BandSchedule(('Purple','Blue','Purple','Blue','Purple','Blue'),(9,2,2,2,2,2))
    library={n:Filament.from_hex(n,h,1) for n,h in [('Purple','#a78bd4'),('Blue','#1e44be')]}
    assert sch.stacks().shape[1]==6
    np.testing.assert_array_equal(sch.stacks().sum(1),np.arange(9,20))
    assert [v[1] for v in sch.swap_entries(library)]==[2,1,2,1,2]


def test_api_invalid_edits_and_project_do_not_replace_document(painting,monkeypatch):
    from core.band import studio_api as api
    monkeypatch.setattr(api.STUDIO,'session',painting)
    monkeypatch.setattr(api.STUDIO,'library',painting.library)
    req=settings(painting)|{'names':painting.filament_names,'counts':painting.layer_counts}
    with TestClient(api.create_app()) as client:
        bad=req|{'counts':[0,7]}
        assert client.post('/api/state',json=bad).status_code==422
        assert api.STUDIO.session is painting
        bad=req|{'mesh_core':[{'hex':'#ff0000','layer':999,'td_mm':1}]}
        assert client.post('/api/state',json=bad).status_code==400
        assert api.STUDIO.session is painting
        assert client.post('/api/project/open',files={'file':('broken.json',b'{}')}).status_code==400
        assert api.STUDIO.session is painting
        assert client.post('/api/mesh',json=req|{'detail':60}).status_code==200
        assert api.STUDIO.session is painting  # viewport requests are read-only snapshots
        saved=client.post('/api/project/save',json=req)
        assert saved.status_code==200
        reopened=client.post('/api/project/open',files={'file':('painting.json',saved.content)})
        assert reopened.status_code==200
        assert reopened.json()['settings']['mesh_mode']=='color_match'
