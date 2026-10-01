"""Generate a face-up SUNLU test strip: 0..5 white layers over solid black.

Geometry is prescribed directly, so an inaccurate optical model cannot change
which thickness gets printed. The leftmost patch has a notch for orientation.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.band.optics import Filament
from core.stack5.lut import register_stack5_mode, synth_lut
from core.relief.mesh import mesh_relief_voxels
from core.relief.export3mf import write_relief_3mf


def generate(out_dir, layer_h=0.08):
    if layer_h not in (0.04, 0.08):
        raise ValueError('choose 0.08 or experimental 0.04 mm')
    os.makedirs(out_dir, exist_ok=True)
    # Postprocessor stretches the first voxel from layer_h to 0.16 mm.
    base_mm, first = 0.8, 0.16
    base_layers = 1 + round((base_mm-first)/layer_h)
    pixel_mm, patch_px, height_px = 0.5, 16, 20
    vox = np.full((base_layers+5,height_px,6*patch_px),-1,np.int8)
    vox[:base_layers] = 0
    for whites in range(6):
        # Complete five-layer shell; black under the white cap.
        x=slice(whites*patch_px,(whites+1)*patch_px)
        vox[base_layers:base_layers+5-whites,:,x]=0
        vox[base_layers+5-whites:,:,x]=1
    vox[:,-4:,:4]=-1  # top-left notch when viewed from above
    fils=[Filament.from_hex('Black','#000000',.15),Filament.from_hex('White','#FFFFFF',5)]
    lut,_=synth_lut(fils,0,layer_h=layer_h)
    recipe=SimpleNamespace(mode_key=register_stack5_mode(fils),filaments=fils,backing_slot=0,lut_rgb=lut)
    meshes=mesh_relief_voxels(vox,['Black','White'],pixel_mm,layer_h)
    sets=[sorted(int(i) for i in np.unique(layer) if i>=0) for layer in vox]
    name=f'sunlu_white_on_black_{layer_h:g}mm'
    dest=os.path.join(out_dir,name+'.3mf')
    post=write_relief_3mf(meshes,recipe,dest,name,sets,(48,10),layer_h,first_layer_mm=first,tower_fit='auto')
    manifest={'file':dest,'orientation':'face-up','notch':'top-left; patches read left to right',
              'first_layer_mm':first,'layer_h_mm':layer_h,'base_mm':base_mm,
              'patch_white_thickness_mm':[round(i*layer_h,4) for i in range(6)],
              'experimental':layer_h<.08,'filament':'Use your SUNLU PLA+ black and white',
              'calibration':'Prescribed geometry, not calibrated colour predictions. Do not infer TD from the supplied photograph.',
              'printer_minimum_mm':.08,'triangles':post['triangles_total']}
    with open(os.path.join(out_dir,name+'.json'),'w') as f: json.dump(manifest,f,indent=2)
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',default='output/stack5_sunlu_calibration')
    ap.add_argument('--layer-height',type=float,choices=[.08,.04],default=.08)
    args=ap.parse_args()
    print(json.dumps(generate(os.path.abspath(args.out),args.layer_height),indent=2))
