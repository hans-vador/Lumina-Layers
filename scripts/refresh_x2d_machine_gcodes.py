"""Snapshot installed X2D scripts, resolving both inheritance and includes."""
import argparse
import json
from pathlib import Path


def resolve_profile(root, name, chain=()):
    if name in chain:
        raise ValueError(f'profile cycle: {chain + (name,)}')
    data=json.loads((Path(root)/(name+'.json')).read_text())
    chain=chain+(name,)
    merged=resolve_profile(root,data['inherits'],chain) if data.get('inherits') else {}
    for include in data.get('include',[]):
        merged.update(resolve_profile(root,include,chain))
    merged.update({k:v for k,v in data.items() if k not in ('include','inherits')})
    return merged


def snapshot(root):
    name='Bambu Lab X2D 0.4 nozzle'
    resolved=resolve_profile(root,name)
    codes={k:v for k,v in resolved.items() if k.endswith('_gcode')}
    if 'X2D start gcode' not in codes.get('machine_start_gcode',''):
        raise ValueError('missing X2D-specific included startup script')
    return {'source':f'Bambu Studio installed BBL/machine/{name}.json; inheritance AND includes resolved',
            'config':codes}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profiles',default='/Applications/BambuStudio.app/Contents/Resources/profiles/BBL/machine')
    p.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'assets/x2d_machine_gcodes.json'))
    a=p.parse_args();Path(a.out).write_text(json.dumps(snapshot(a.profiles),indent=2)+'\n')
