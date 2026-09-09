"""Relief Stack5 end-to-end: original image -> Stack5 recipe + provider depth
-> face-up relief voxels -> per-material meshes -> X2D 3MF (+ preview, recipe).

    original album image
        |-- core.relief.color_stage  (Stack5, unchanged)  -> 5-layer recipe / pixel
        |-- geometry provider         (depth / normals / mask)
                 -> depth_processing  (align, normalise, smooth, quantise, slope limit)
                 -> relief_stack      (backing to surface, stack[4]..stack[0] on top)
                 -> mesh              (one closed box-union per filament)
                 -> export3mf         (Lumina writer + Stack5 post-processor)
                 -> validate          (rasterise the 3MF back and compare)

Flat Stack5 (core/stack5/pipeline.py) is not modified; the flat mode still
prints face down with its own entry point.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Optional, Sequence

import numpy as np

from config import _BASE_DIR
from core.relief.color_stage import ColourRecipe, stack5_colour_recipe
from core.relief.depth_processing import DepthParams, ReliefField, process_depth
from core.relief.export3mf import write_relief_3mf
from core.relief.mesh import mesh_relief_voxels
from core.relief.preview import depth_preview, shaded_preview
from core.relief.provider import GeometryResult, get_provider, load_mask_file
from core.relief.relief_stack import ReliefDims, build_relief_voxels, check_relief_voxels
from core.relief.validate import verify_relief_3mf
from core.stack5.pipeline import TOWER_FIT_MODES, _jsonable, _slug

REPO = _BASE_DIR
DEFAULT_OUT_DIR = os.path.join(REPO, 'output', 'relief')
VERIFY_MODES = ('auto', 'yes', 'no')
VERIFY_AUTO_MAX_PX = 2_500_000        # 'auto' re-rasterises the 3MF up to this many grid pixels (150 mm = 2.25 M)


def _sanitize_stem(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', '_', s).strip('_') or 'relief'


def convert_album_relief(image_path: str, width_mm: float = 150.0, *,
                         geometry: str = 'imported-depth', depth_map: Optional[str] = None,
                         mesh: Optional[str] = None, normals: Optional[str] = None, normals_y_up: bool = False,
                         foreground_mask: Optional[str] = None, provider_options: Optional[dict] = None,
                         dims: Optional[ReliefDims] = None, depth_params: Optional[DepthParams] = None,
                         depth_weight: float = 1.0, normal_weight: float = 0.0, smoothness_weight: float = 0.0,
                         out_dir: str = DEFAULT_OUT_DIR, output: Optional[str] = None, title: Optional[str] = None,
                         verify: str = 'auto', keep_intermediate: bool = False,
                         flush_scale: float = 1.0, min_flush: Optional[float] = None, tower_fit: str = 'auto',
                         seed: int = 0, geometry_result: Optional[GeometryResult] = None,
                         height_edits=None, edit_clean_px: Optional[int] = None,
                         **colour_kwargs) -> dict:
    """Convert one album image into a face-up relief 3MF.

    geometry / depth_map / mesh / normals / foreground_mask: provider inputs
      (see core.relief.provider).  ``geometry_result`` bypasses the provider.
    dims: ReliefDims (base 0.8 mm, relief 4.0 mm, 0.08 mm layers, 5 colour layers).
    depth_params: DepthParams (normalisation, smoothing, slope limits ...).
    normal_weight > 0 with normals present: fuse depth + normals (Milestone 4).
    verify: 'auto' | 'yes' | 'no' - rasterise the written 3MF and compare with
      the voxel matrix ('auto' = only up to VERIFY_AUTO_MAX_PX grid pixels).
    height_edits: optional art-direction script (core.relief.art_direct.EditScript,
      its dict or a JSON path): mask operations replayed on the processed relief
      (geometry only; the colour recipe is untouched).  None = unchanged pipeline.
    edit_clean_px: min-feature kernel for the edited region (default: the script's, 4).
    colour_kwargs: forwarded to stack5_colour_recipe (palette, backing, metric,
      quantize_colors, smooth_sigma, advisory, min_region_px, filaments_json ...).
    Returns a dict with the output paths, geometry report, voxel check, 3MF
    verification and the flat-Stack5-compatible recipe fields.
    """
    t_all = time.perf_counter()
    timing: dict = {}
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if verify not in VERIFY_MODES:
        raise ValueError(f"verify must be one of {VERIFY_MODES}")
    if tower_fit not in TOWER_FIT_MODES:
        raise ValueError(f"tower_fit must be one of {TOWER_FIT_MODES}")
    dims = dims or ReliefDims()
    params = depth_params or DepthParams()
    slug = _slug(image_path)
    if output:
        output = os.path.abspath(output)
        job_dir = os.path.dirname(output)
        stem = _sanitize_stem(os.path.splitext(os.path.basename(output))[0])
    else:
        job_dir = os.path.join(out_dir, slug)
        stem = f"{slug}_relief"
        output = os.path.join(job_dir, f"{stem}.3mf")
    os.makedirs(job_dir, exist_ok=True)
    title = title or slug.replace('_', ' ')
    warnings: list[str] = []
    if abs(float(dims.layer_h) - 0.08) > 1e-9:
        warnings.append(f"layer height {dims.layer_h:g} differs from the 0.08 mm the Stack5 LUT / preset are "
                        "calibrated for: the colour recipe is re-synthesised for this thickness and is NOT the "
                        "flat Stack5 recipe; the process preset keeps its 0.08 mm name")

    # 0. preflight: geometry inputs and optional dependencies are checked BEFORE the
    #    slow colour stage so a typo or a missing package fails in a second
    provider = None
    if geometry_result is None:
        opts = dict(provider_options or {})
        opts.setdefault('depth_map', depth_map)
        opts.setdefault('mesh', mesh)
        opts.setdefault('normals', normals)
        opts.setdefault('normals_y_up', normals_y_up)
        if foreground_mask and 'mask' not in opts:
            opts['mask'] = foreground_mask
        provider = get_provider(geometry, **opts)
        pf = getattr(provider, 'preflight', None)
        if callable(pf):
            pf()
    for label, path in (('foreground mask', foreground_mask), ('normal map', normals)):
        if path and not os.path.isfile(path):
            raise FileNotFoundError(f"{label} file not found: {path}")

    # 1. colour: the unchanged Stack5 recipe on the original image
    t0 = time.perf_counter()
    recipe: ColourRecipe = stack5_colour_recipe(image_path, width_mm, job_dir, slug, seed=seed,
                                                layer_h=dims.layer_h, layers=dims.color_layers,
                                                lut_basename=f"{stem}_lut.npz", **colour_kwargs)
    timing['colour_s'] = time.perf_counter() - t0
    H, W = recipe.grid_hw

    # 2. geometry
    t0 = time.perf_counter()
    geom = provider.generate(image_path, target_hw=(H, W)) if geometry_result is None else geometry_result
    if normals and geom.normals is None:
        # a normal map given on the command line is attached whatever the provider
        from core.relief.normal_integration import load_normal_map
        nm = load_normal_map(normals, y_up=normals_y_up)
        if nm.shape[:2] != geom.depth.shape:
            import cv2
            nm = cv2.resize(nm, (geom.depth.shape[1], geom.depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        geom = GeometryResult(depth=geom.depth, normals=nm, mask=geom.mask, depth_convention=geom.depth_convention,
                              source=geom.source, meta=dict(geom.meta, normals_file=os.path.abspath(normals)))
    if float(normal_weight) > 0 and geom.normals is None:
        warnings.append("normal_weight > 0 but no normal map is available: no depth/normal fusion")
    fusion_info = None
    if geom.normals is not None and float(normal_weight) > 0:
        from core.relief.normal_integration import fuse_depth_normals
        fused, fusion_info = fuse_depth_normals(geom.raised(), geom.normals, depth_weight=float(depth_weight),
                                                normal_weight=float(normal_weight),
                                                smoothness_weight=float(smoothness_weight), mask=geom.mask)
        geom = GeometryResult(depth=fused, normals=geom.normals, mask=geom.mask, source=f"{geom.source}+normals",
                              meta=dict(geom.meta, fusion=fusion_info))
    fg = load_mask_file(foreground_mask) if foreground_mask else None
    field: ReliefField = process_depth(geom, (H, W), dims, params, foreground_mask=fg, plaque_mask=recipe.mask_solid)
    field.report['warnings'] = warnings + list(field.report.get('warnings', []))
    edit_report = None
    edits_json = None
    if height_edits is not None:
        # art direction: replay the mask edits on the processed relief (geometry only)
        from core.relief.art_direct import EditScript, apply_edit_script
        script = EditScript.coerce(height_edits)
        if len(script):
            field, edit_report = apply_edit_script(field, script, recipe.pixel_mm, recipe.mask_solid,
                                                   clean_px=edit_clean_px)
            field.report['warnings'] = list(field.report.get('warnings', [])) + list(edit_report.get('warnings', []))
            edits_json = script.save(os.path.join(job_dir, f"{stem}_edits.json"))
            edit_report['edits_file'] = edits_json
    timing['geometry_s'] = time.perf_counter() - t0

    # 3. face-up voxels
    t0 = time.perf_counter()
    vox, vmeta = build_relief_voxels(recipe.material_matrix, recipe.mask_solid, field.steps, dims, recipe.backing_slot,
                                     n_slots=len(recipe.names))
    vcheck = check_relief_voxels(vox, recipe.material_matrix, recipe.mask_solid, field.steps, dims, recipe.backing_slot)
    if not vcheck['ok']:
        raise RuntimeError(f"relief voxel structure check failed: {vcheck}")
    timing['voxels_s'] = time.perf_counter() - t0

    # 4. meshes
    t0 = time.perf_counter()
    meshes = mesh_relief_voxels(vox, recipe.names, recipe.pixel_mm, dims.layer_h)
    bad = [e['name'] for e in meshes if not e.get('volume_ok', True)]
    if bad:
        raise RuntimeError(f"mesh volume does not match the voxel volume for {bad}")
    timing['mesh_s'] = time.perf_counter() - t0

    # 5. 3MF
    t0 = time.perf_counter()
    post = write_relief_3mf(meshes, recipe, output, title, vmeta['printed_layer_material_sets'],
                            (W * recipe.pixel_mm, H * recipe.pixel_mm), dims.layer_h, flush_scale=flush_scale,
                            min_flush=min_flush, tower_fit=tower_fit, keep_intermediate=keep_intermediate)
    timing['export_s'] = time.perf_counter() - t0

    # 6. verification of the written file
    t0 = time.perf_counter()
    do_verify = verify == 'yes' or (verify == 'auto' and H * W <= VERIFY_AUTO_MAX_PX)
    verification = None
    if do_verify:
        verification = verify_relief_3mf(output, vox, [p['slot'] for p in post['parts']], recipe.pixel_mm,
                                         dims.layer_h, backing_slot=recipe.backing_slot)
        if not verification['ok']:
            raise RuntimeError(f"3MF verification failed: {verification}")
    else:
        field.report['warnings'].append(f"3MF re-rasterisation skipped (verify={verify}, {H * W} grid px); "
                                        "use --verify yes to run it")
    timing['verify_s'] = time.perf_counter() - t0

    # 7. previews + side files
    preview_png = os.path.join(job_dir, f"{stem}_preview.png")
    shaded_preview(recipe.matched_rgb, recipe.mask_solid, field.steps, dims.layer_h, recipe.pixel_mm).save(preview_png)
    depth_png = os.path.join(job_dir, f"{stem}_depth.png")
    depth_preview(field.steps, dims.relief_layers, recipe.mask_solid).save(depth_png)
    ams_lines = []
    for p in post['parts']:
        f = recipe.library[p['name']]
        tag = '  (backing / relief body)' if p['slot'] == recipe.backing_slot else ''
        ams_lines.append(f"AMS slot {p['extruder']} = {f.name} ({f.hex}, TD {f.td_mm:g} mm){tag}")
    if post['unused_slots']:
        ams_lines.append("unused (no pixels): " + ', '.join(recipe.names[s] for s in post['unused_slots']))
    ams_txt = os.path.join(job_dir, f"{stem}_ams.txt")
    with open(ams_txt, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(ams_lines) + '\n')

    tp = post['tower_plan']
    mesh_stats = [{k: v for k, v in e.items() if k != 'mesh'} for e in meshes]
    timing['total_s'] = time.perf_counter() - t_all
    recipe_out = {
        'mode': 'relief-stack5',
        'image': os.path.abspath(image_path),
        'title': title,
        'mode_key': recipe.mode_key,
        'orientation': 'face-up: base on the bed, stack[0] visible on top; no flipping after printing',
        'thickness': dims.summary(),
        'settings': dict(recipe.settings, geometry=geometry, depth_map=os.path.abspath(depth_map) if depth_map else None,
                         mesh=os.path.abspath(mesh) if mesh else None,
                         normals=os.path.abspath(normals) if normals else None,
                         foreground_mask=os.path.abspath(foreground_mask) if foreground_mask else None,
                         depth_params=params.to_dict(), depth_weight=float(depth_weight),
                         normal_weight=float(normal_weight), smoothness_weight=float(smoothness_weight),
                         flush_scale=float(tp['flush_scale']), flush_scale_requested=float(flush_scale),
                         min_flush=tp['flush_min_mm3'], tower_fit=tower_fit, verify=verify,
                         height_edits=edits_json),
        'palette': [{'slot': i, 'name': f.name, 'hex': f.hex, 'td_mm': f.td_mm, 'td_source': f.td_source}
                    for i, f in enumerate(recipe.filaments)],
        'backing': {'slot': recipe.backing_slot, 'name': recipe.filaments[recipe.backing_slot].name,
                    'extruder': post['backing_extruder'], 'role': 'flat base + relief body + support under the shell'},
        'slot_to_extruder': post['slot_to_extruder'],
        'parts': post['parts'],
        'ams': ams_lines,
        'colour': {'material_matrix_sha256': recipe.sha256, 'pixel_matcher': recipe.matcher,
                   'cleanup': recipe.cleanup, 'palette_report': recipe.palette_report,
                   'depth_independent': True,
                   'note': ('identical to flat Stack5 stats.material_matrix_sha256 for the same image/palette'
                            if abs(float(dims.layer_h) - 0.08) < 1e-9 else
                            f'LUT re-synthesised for {dims.layer_h:g} mm layers (not the flat Stack5 recipe); '
                            'still independent of the depth map')},
        'geometry': {'provider': geom.describe(), 'fusion': fusion_info, 'report': field.report,
                     'voxels': vmeta, 'voxel_check': vcheck},
        'art_direction': edit_report,
        'meshes': mesh_stats,
        'stats': {
            'resolution_px': [W, H], 'pixel_mm': recipe.pixel_mm, 'plaque_size_mm': [W * recipe.pixel_mm, H * recipe.pixel_mm],
            'total_print_layers': vmeta['total_layers'], 'total_height_mm': vmeta['total_mm'],
            'relief_layers_max': vmeta['relief_layers_present_max'],
            'n_objects': post['n_objects'], 'triangles_per_object': post['triangles_per_object'],
            'triangles_total': post['triangles_total'],
            'materials_per_printed_layer': vmeta['materials_per_printed_layer'],
            'tool_changes_planned': tp['tool_changes'], 'purge_total_mm3': tp['purge_total_mm3'],
            'purge_per_layer_mm3': tp['purge_per_layer_mm3'], 'tower': tp['tower'],
            'material_matrix_sha256': recipe.sha256,
        },
        'threemf_check': {k: post[k] for k in ('filament_colour', 'project_settings', 'flush_matrix_len',
                                               'flush_vector_len', 'build_transform', 'plaque_xy_mm',
                                               'process_preset', 'process_keys_changed_vs_system')},
        'verification': verification,
        'flush': {k: tp[k] for k in ('flush_matrix', 'flush_min_mm3', 'flush_scale', 'flush_max_mm3',
                                     'flush_formula', 'ordering', 'purge_total_mm3', 'worst_layer', 'tool_changes')},
        'tower': tp['tower'],
        'layout': tp['layout'],
        'outputs': {'threemf': output, 'preview_png': preview_png, 'depth_png': depth_png, 'lut_npz': recipe.lut_npz,
                    'ams_txt': ams_txt, 'stamped_input': recipe.stamped_input, 'edits_json': edits_json},
        'timing': dict(recipe.timing, **timing),
    }
    recipe_json = os.path.join(job_dir, f"{stem}_recipe.json")
    with open(recipe_json, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(recipe_out), fh, indent=2, ensure_ascii=False)

    return {
        'threemf': output, 'preview_png': preview_png, 'depth_png': depth_png, 'recipe_json': recipe_json,
        'ams_txt': ams_txt, 'lut_npz': recipe.lut_npz, 'palette': recipe.names,
        'backing': recipe.filaments[recipe.backing_slot].name, 'backing_slot': recipe.backing_slot,
        'mode_key': recipe.mode_key, 'material_matrix_sha256': recipe.sha256,
        'dims': dims.summary(), 'steps': field.steps, 'voxels': vox, 'voxel_meta': _jsonable(vmeta),
        'voxel_check': vcheck, 'verification': verification, 'geometry_report': _jsonable(field.report),
        'parts': post['parts'], 'slot_to_extruder': post['slot_to_extruder'],
        'stats': _jsonable(recipe_out['stats']), 'tower': _jsonable(tp['tower']), 'flush': _jsonable(recipe_out['flush']),
        'meshes': _jsonable(mesh_stats), 'timing': timing, 'ams': ams_lines,
        'art_direction': _jsonable(edit_report), 'edits_json': edits_json,
    }
