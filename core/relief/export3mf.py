"""3MF export for Relief Stack5 through the EXISTING writers.

1. utils.bambu_3mf_writer.export_scene_with_bambu_metadata - Lumina's Bambu
   writer: one part per material object, extruders 1..F in slot order, all
   objects in 3D/Objects/object_1.model (the layout flat Stack5 produces).
2. core.stack5.pipeline.postprocess_3mf - the flat Stack5 post-processor,
   unchanged: X2D project settings (self-contained 'Stack5 0.08mm @BBL X2D'
   process preset), palette flush matrix, prime tower sized from the per-layer
   purge plan (now over every relief layer), plaque placement, fixed zip
   timestamps (byte-identical output for identical input).

The intermediate writer output is deleted unless ``keep_intermediate``.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from config import ColorSystem
from core.relief.color_stage import ColourRecipe
from core.relief.mesh import scene_from_meshes
from core.stack5.pipeline import postprocess_3mf

# Same dict core.converter passes to the writer (only used to seed the writer's
# project_settings, which postprocess_3mf rebuilds from the X2D template).
WRITER_PRINT_SETTINGS = {
    'layer_height': '0.08', 'initial_layer_height': '0.08', 'wall_loops': '1',
    'top_shell_layers': '0', 'bottom_shell_layers': '0', 'sparse_infill_density': '100%',
    'sparse_infill_pattern': 'zig-zag', 'nozzle_temperature': ['220'] * 8, 'bed_temperature': ['60'] * 8,
    'filament_type': ['PLA'] * 8, 'print_speed': '100', 'travel_speed': '150', 'enable_support': '0',
    'brim_width': '5', 'brim_type': 'auto_brim',
}


def write_relief_3mf(meshes: Sequence[dict], recipe: ColourRecipe, out_path: str, title: str,
                     printed_layer_sets: Sequence[Sequence[int]], plaque_size_mm: Sequence[float],
                     layer_h: float, first_layer_mm: Optional[float] = None, flush_scale: float = 1.0,
                     min_flush: Optional[float] = None, tower_fit: str = 'auto',
                     keep_intermediate: bool = False) -> dict:
    """meshes: output of core.relief.mesh.mesh_relief_voxels (slot order).
    Returns postprocess_3mf's report (+ 'intermediate' path when kept)."""
    if not meshes:
        raise ValueError("no material meshes to export")
    from utils.bambu_3mf_writer import export_scene_with_bambu_metadata
    scene = scene_from_meshes(meshes)
    slot_names = [e['name'] for e in meshes]
    conf = ColorSystem.get(recipe.mode_key)
    preview_colors = conf['preview']
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    tmp = os.path.join(os.path.dirname(out_path), f".{stem}_lumina_writer.3mf")
    settings = dict(WRITER_PRINT_SETTINGS)
    settings['layer_height'] = settings['initial_layer_height'] = f"{float(layer_h):g}"
    try:
        export_scene_with_bambu_metadata(scene=scene, output_path=tmp, slot_names=slot_names,
                                         preview_colors=preview_colors, settings=settings,
                                         color_mode=recipe.mode_key)
        post = postprocess_3mf(tmp, out_path, recipe.filaments, recipe.backing_slot, recipe.lut_rgb,
                               first_layer_mm=float(first_layer_mm if first_layer_mm is not None else layer_h),
                               layer_h=float(layer_h), title=title,
                               printed_layer_sets=[list(s) for s in printed_layer_sets],
                               plaque_size_mm=(float(plaque_size_mm[0]), float(plaque_size_mm[1])),
                               flush_scale=float(flush_scale), min_flush=min_flush, tower_fit=tower_fit)
    finally:
        if not keep_intermediate and os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    if keep_intermediate:
        post['intermediate'] = tmp
    post['threemf'] = out_path
    return post
