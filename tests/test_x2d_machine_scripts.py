import json
from pathlib import Path
from scripts.refresh_x2d_machine_gcodes import resolve_profile


def test_include_overrides_inherited_generic_script(tmp_path):
    profiles={'base':{'machine_start_gcode':'generic'},'startup':{'machine_start_gcode':'X2D-specific'},
              'printer':{'inherits':'base','include':['startup'],'nozzle_diameter':['0.4','0.4']}}
    for name,data in profiles.items():
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
    cfg=resolve_profile(tmp_path,'printer')
    assert cfg['machine_start_gcode']=='X2D-specific'
    assert cfg['nozzle_diameter']==['0.4','0.4']


def test_snapshot_contains_real_x2d_startup_and_heating():
    cfg=json.loads((Path(__file__).resolve().parents[1]/'assets/x2d_machine_gcodes.json').read_text())['config']
    start=cfg['machine_start_gcode']
    assert 'X2D start gcode' in start
    assert 'M104' in start and 'M109' in start
    assert 'nozzle_temperature_initial_layer' in start
    assert 'G1 X10.1 Y200.0 Z0.28 F1500.0 E15' not in start


def test_snapshot_matches_included_installed_scripts_when_available():
    root=Path('/Applications/BambuStudio.app/Contents/Resources/profiles/BBL/machine')
    if not root.exists():
        import pytest
        pytest.skip('Bambu Studio not installed')
    cfg=json.loads((Path(__file__).resolve().parents[1]/'assets/x2d_machine_gcodes.json').read_text())['config']
    main=json.loads((root/'Bambu Lab X2D 0.4 nozzle.json').read_text())
    for name in main['include']:
        include=json.loads((root/(name+'.json')).read_text())
        for key,value in include.items():
            if key.endswith('_gcode'): assert cfg[key]==value
