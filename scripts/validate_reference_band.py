#!/usr/bin/env python3
"""Compare the frozen Graduation tone rule with the original reference mesh.

Registration constants are specific to the 200 mm Graduation reference and its
embedded 639x639 source image. This measures geometry, not printed colour.
"""
import argparse
import json
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.band.reference import tone_height

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',required=True)
    ap.add_argument('--image',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    vertices=[]
    with zipfile.ZipFile(a.reference) as z, z.open('3D/3dmodel.model') as stream:
        for _,e in ET.iterparse(stream,events=['end']):
            if e.tag.endswith('vertex'):
                vertices.append([float(e.attrib[k]) for k in ('x','y','z')])
            e.clear()
    v=np.asarray(vertices);v[:,2]+=2
    v=v[(v[:,2]>.01)&(v[:,2]<2.4)&(abs(v[:,0])<96)&(abs(v[:,1])<96)]
    v=v[np.random.default_rng(987).choice(len(v),min(100000,len(v)),False)]
    rgb=np.asarray(Image.open(a.image).convert('RGB'))
    if rgb.shape!=(639,639,3):raise ValueError('Use the original embedded 639x639 Graduation source')
    h=tone_height(rgb)
    pred=map_coordinates(h,[(97-v[:,1]-.0506037879)/193.896092*638,
                            (v[:,0]+97-.052808021)/193.896092*638],order=1,mode='nearest')
    error=abs(pred-v[:,2])
    report={'samples':len(error),'mean_absolute_error_mm':float(error.mean()),
            'median_absolute_error_mm':float(np.median(error)),
            'within_half_layer_fraction':float(np.mean(error<.04)),
            'p95_error_mm':float(np.percentile(error,95)),
            'scope':'Registered tone rule vs reference interior; excludes frame; not printed-colour validation'}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
