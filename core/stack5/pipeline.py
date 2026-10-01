"""Stack5 end-to-end pipeline: album image -> 5-filament palette -> synthetic LUT
-> Lumina's NATIVE HiFi conversion (core.converter.convert_image_to_3d) -> 3MF
post-processed to an X2D-consistent project (core.band.writer3mf template).

Lumina quirks handled here (verified in core/converter.py, core/image_processing.py):
* structure_mode is a substring-tested string: "Double" / "双面" -> double-sided,
  anything else -> single-sided voxel matrix, but the X-mirror of single-sided
  output is only applied when the string contains "Single" / "单面".  We pass
  'Single-sided' / 'Double-sided'.
* backing_color_id is the palette SLOT index (0..n-1) of the filament that fills
  the spacer layers (validated against len(slots)); the backing voxels become
  part of that slot's mesh (no separate backing object).
* Only slots with a non-empty mesh are exported, in slot order, and the writer
  renumbers extruders 1..F by that order; project_settings.filament_colour is the
  LUT's pure-stack colour per slot (converter._get_actual_lut_slot_colors).  We
  read the exported part list back and rebuild project_settings for those F.
* convert_image_to_3d writes to config.OUTPUT_DIR with a generated name
  ('<base>_Lumina_HiFi_Unknown_<ts>.3mf' - the naming service has no tag for a
  dynamic mode) and does not return the material matrix; we capture the
  processor result by subclassing LuminaImageProcessor in core.converter's
  namespace for the duration of the call.
* The browser GLB preview (_create_preview_mesh) is a pure-Python per-pixel
  loop; it is skipped (patched to return None) - the 3MF does not depend on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import zipfile
from typing import Optional, Sequence
from xml.sax.saxutils import quoteattr

import numpy as np
from PIL import Image

from config import OUTPUT_DIR, ModelingMode, PrinterConfig, _BASE_DIR
from core.band import writer3mf
from core.band.optics import BeerLambertModel, Filament, load_filament_library, srgb_to_lab_d65
from core.band.pipeline import advisory_stamp_mask
from core.stack5 import flush as flushmod
from core.stack5.cleanup import min_region_cleanup, region_stats, resync_matched_rgb
from core.stack5.lut import (LAYER_H, N_LAYERS, choose_backing, is_pure, layer_thicknesses, lut_meta,
                             pure_stack_indices, register_stack5_mode, save_lut_npz, synth_lut)
from core.stack5.palette import (DEFAULT_CHROMA_WEIGHT, DEFAULT_DOMINANT_W, DEFAULT_METRIC,
                                 DEFAULT_NEED_SHARE, DEFAULT_SPOOL_PENALTY,
                                 DEFAULT_PREFER_TOL, DEFAULT_SPOOL_BONUS, DEFAULT_SPOOL_DE,
                                 evaluate_palette, image_hist, select_palette)
from core.stack5.metric import make_matcher, resolve_hue_params

REPO = _BASE_DIR
USER_FILAMENTS_JSON = os.path.join(REPO, 'assets', 'filaments_user.json')
MEASURED_FILAMENTS_JSON = os.path.join(REPO, 'assets', 'filaments_user_measured.json')


def default_filaments_json() -> str:
    """Library used when the caller passes none: the calibrated
    ``filaments_user_measured.json`` (TDs fitted to Lumina's measured Bambu LUTs, see
    docs/band/TD_CALIBRATION.md) when it exists, else the HueForge-style guesses."""
    return MEASURED_FILAMENTS_JSON if os.path.isfile(MEASURED_FILAMENTS_JSON) else USER_FILAMENTS_JSON


DEFAULT_FILAMENTS_JSON = default_filaments_json()
DEFAULT_OUT_DIR = os.path.join(REPO, 'output', 'stack5')
HIFI_PX_PER_MM = 10                         # core/image_processing.py HIGH_FIDELITY
STRUCTURE_MODES = {'single': 'Single-sided', 'double': 'Double-sided'}
ADVISORY_CORNERS = ('br', 'bl', 'bc')
ADVISORY_LABEL_FRAC = 0.13
ADVISORY_MARGIN_FRAC = 0.035

PRINTER_SETTINGS_ID = writer3mf.PRINTER_SETTINGS_ID       # 'Bambu Lab X2D 0.4 nozzle'
SYSTEM_PRINT_SETTINGS_ID = writer3mf.PRINT_SETTINGS_ID    # '0.08mm High Quality @BBL X2D' (the base we resolve)
PRINT_SETTINGS_ID = 'Stack5 0.08mm @BBL X2D'              # our self-contained "project-inside" process preset
SYSTEM_PROCESS_JSON = os.path.join(REPO, 'assets', 'x2d_process_0.08mm_high_quality_system.json')
# Every process key we deliberately change from the X2D 0.08 mm system preset
# (on top of stack5_print_overrides + the prime tower plan).  The plaque is a
# solid plaque with one wall, a skirt to prime, and no brim. Solid depth is
# set from the layer plan so Bambu uses monotonic rather than sparse infill.
STACK5_PROCESS_OVERRIDES = {
    'wall_loops': '1', 'top_shell_layers': '1', 'bottom_shell_layers': '0',
    'sparse_infill_density': '100%', 'sparse_infill_pattern': 'zig-zag',
    'internal_solid_infill_pattern': 'monotonic',
    'top_surface_pattern': 'monotonic', 'bottom_surface_pattern': 'monotonic', 'ironing_type': 'no ironing',
    'enable_prime_tower': '1', 'flush_into_infill': '0', 'flush_into_objects': '0', 'flush_into_support': '1',
    'seam_position': 'aligned', 'wall_generator': 'classic',
    'print_sequence': 'by layer', 'timelapse_type': '0', 'skirt_loops': '1', 'skirt_distance': '2',
    'skirt_height': '1', 'brim_type': 'no_brim', 'detect_thin_wall': '0',
}
# First-layer adhesion (Hans, Sep 5 2026: the 0.08 mm first layer was lifting).  The first
# layer stays 0.08 mm - it is the face-down viewing layer and the colours depend on it -
# so adhesion comes from a slow, unfanned first layer instead.  (Sep 21 2026: nozzle
# dropped from 220/225 to 205 on every layer at Hans's request; speed and fans unchanged.)  Process keys go
# through the self-contained process preset (list-valued X2D keys are expanded to the
# system list length); filament keys are written per filament AND listed in each
# filament's different_settings_to_system, otherwise Bambu Studio silently reverts them
# to the 'Bambu PLA Basic @BBL X2D' system values.
FIRST_LAYER_PROCESS_OVERRIDES = {
    'initial_layer_speed': '18',            # mm/s (user: 15-20; X2D system 40-50)
    'initial_layer_infill_speed': '20',     # mm/s (system 70-100)
    'elefant_foot_compensation': '0',       # user: 0 (system 0.15; Stack5 used 0.1)
    'initial_layer_acceleration': '500',    # keep the X2D system value explicit
}
FIRST_LAYER_FILAMENT_OVERRIDES = {
    'close_fan_the_first_x_layers': '1',    # part fan off on the first layer ...
    'first_x_layer_fan_speed': '0',         # ... and 0 % if the profile uses the explicit key
    'first_x_layer_part_fan_speed': '0',
    'close_additional_fan_first_x_layers': '1',   # aux fan off on the first layer too
    'hot_plate_temp_initial_layer': '65', 'hot_plate_temp': '65',            # smooth / high-temp PEI plate
    'textured_plate_temp_initial_layer': '65', 'textured_plate_temp': '65',  # textured PEI plate
    # Filament temperature is a per-filament key, so it lives here with the other
    # filament overrides and gets listed in different_settings_to_system - otherwise
    # Bambu Studio reverts it to the 220 system value.
    'nozzle_temperature': '205',
    'nozzle_temperature_initial_layer': '205',
}
STACK5_PROCESS_OVERRIDES.update(FIRST_LAYER_PROCESS_OVERRIDES)
FILAMENT_SETTINGS_ID = 'Bambu PLA Basic @BBL X2D 0.4 nozzle'
STACK5_DIFFERENT_SETTINGS = ('bottom_shell_layers;elefant_foot_compensation;enable_prime_tower;'
                             'initial_layer_print_height;ironing_type;prime_tower_brim_width;'
                             'prime_tower_infill_gap;prime_tower_rib_wall;prime_tower_width;'
                             'sparse_infill_density;sparse_infill_pattern;top_shell_layers;'
                             'top_surface_pattern;wall_loops')
DEFAULT_MIN_REGION_PX = 16          # 0.4 x 0.4 mm at 0.1 mm/px: the smallest island a 0.4 nozzle prints
TOWER_FIT_MODES = ('error', 'warn', 'auto')
# Backing (spacer) layers are printed at this layer height through a Bambu Studio
# "height range modifier" (Metadata/layer_config_ranges.xml, applied to the plaque
# object; see bbs_3mf.cpp _extract_layer_config_ranges_from_archive and
# Slicing.cpp layer_height_profile_from_ranges).  The colour layers keep layer_h.
# 1.6 mm of backing = 20 layers at 0.08 -> 8 layers at 0.2.  Boundaries fall on
# whole layers (5 x 0.08 = 0.4; 1.6 / 0.2 = 8), so the slicer keeps the colour
# layers exactly where the LUT expects them.  None disables the modifier.
DEFAULT_BACKING_LAYER_H = 0.2
TOWER_FIT_MIN_SCALE = 0.3          # 'auto' never purges less than 30 % of Bambu's formula
TOWER_FIT_STEP = 0.8


def stack5_print_overrides(first_layer_mm: float = 0.08, layer_h: float = LAYER_H,
                           tower_plan: Optional[dict] = None) -> dict:
    """Print profile for Lumina's face-down stacks on the X2D (applied on top of
    writer3mf.BAND_PRINT_OVERRIDES, which it replaces where they differ).

    The converter meshes the viewing-surface stack at Z 0..layer_h
    (transform[2,2] = LAYER_HEIGHT).  A thicker first layer is therefore NOT
    just a profile value: postprocess_3mf shifts every mesh vertex at
    z >= layer_h up by (first_layer_mm - layer_h) so the viewing voxel layer
    spans Z 0..first_layer_mm and the slicer's first layer samples it.  The
    synthetic LUT and the palette search model that layer at its real
    thickness (lut.layer_thicknesses).  A first layer thinner than layer_h
    would sample nothing and is refused.
    """
    if float(first_layer_mm) < float(layer_h) - 1e-9:
        raise ValueError(f"initial_layer_print_height ({first_layer_mm:g}) cannot be thinner than the colour "
                         f"layer height ({layer_h:g}): the face-down viewing layer must fill the first layer")
    ov = {
        'layer_height': f"{float(layer_h):g}",
        'initial_layer_print_height': f"{float(first_layer_mm):g}",
        'wall_loops': '1',
        'top_shell_layers': '1',
        'bottom_shell_layers': '0',
        'sparse_infill_density': '100%',
        'sparse_infill_pattern': 'zig-zag',
        'enable_prime_tower': '1',
        'printer_settings_id': PRINTER_SETTINGS_ID,
        'print_settings_id': PRINT_SETTINGS_ID,
        'printer_model': 'Bambu Lab X2D',
        'print_compatible_printers': [PRINTER_SETTINGS_ID],
    }
    if tower_plan is not None:
        # Cover the plaque depth with solid layers. Bambu otherwise classifies
        # even 100% fill as sparse, where monotonic is not an accepted pattern.
        ov['top_shell_layers'] = str(max(1, len(tower_plan['layers'])))
        ov.update(flushmod.tower_config_keys(tower_plan))
    return ov


def load_system_process(path: str = SYSTEM_PROCESS_JSON) -> dict:
    """The X2D 0.08 mm system process preset, fully resolved through its
    inherits chain (cached from the installed Bambu Studio vendor profiles by
    scripts/refresh_x2d_process_preset.py)."""
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def apply_stack5_process_preset(cfg: dict, overrides: dict, system: Optional[dict] = None) -> list[str]:
    """Make the process preset SELF-CONTAINED.

    Bambu Studio merges a project's process values onto the preset named by
    print_settings_id / inherits_group[0] and silently reverts every key that is
    not listed in different_settings_to_system[0]; a system-named preset can
    also be reset to the system values by a printer re-selection.  A custom
    name with no parent makes Studio create a "project-inside" preset that is
    used verbatim (PresetCollection::load_external_preset), but then EVERY key
    must be present or it falls back to PrintConfig defaults (0.2 mm first
    layer, 3 mm tower brim, 60 mm tower...).  So: all process keys = the
    resolved X2D 0.08 mm system values, then STACK5_PROCESS_OVERRIDES, then the
    Stack5 layer/tower overrides.  Returns the list of keys that differ from
    the system preset (also written to different_settings_to_system[0]).
    """
    sysp = (system or load_system_process())['config']
    for k, v in sysp.items():
        cfg[k] = [str(x) for x in v] if isinstance(v, list) else str(v)
    for k, v in STACK5_PROCESS_OVERRIDES.items():
        cfg[k] = _shaped_like(sysp.get(k), v)
    for k, v in overrides.items():
        cfg[k] = [str(x) for x in v] if isinstance(v, list) else str(v)
    cfg['print_settings_id'] = (PRINT_SETTINGS_ID if float(cfg['layer_height']) == LAYER_H
                                else f"Stack5 {float(cfg['layer_height']):g}mm experimental @BBL X2D")
    cfg['print_compatible_printers'] = [PRINTER_SETTINGS_ID]
    cfg['inherits_group'] = [''] * len(cfg.get('inherits_group') or [''])
    cfg['different_settings_to_system'][0] = ';'.join(process_diff_vs_system(cfg, sysp))
    return process_diff_vs_system(cfg, sysp)


def _shaped_like(system_value, v):
    """A scalar override for a list-valued X2D key (one entry per extruder /
    variant, e.g. initial_layer_speed has 6) is repeated to that length."""
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(system_value, list):
        return [str(v)] * len(system_value)
    return str(v)


def process_diff_vs_system(cfg: dict, sysp: dict) -> list[str]:
    return sorted(k for k in sysp
                  if cfg.get(k) != ([str(x) for x in sysp[k]] if isinstance(sysp[k], list) else str(sysp[k])))


def _filament_count(cfg: dict) -> int:
    return len(cfg.get('filament_colour') or [])


def apply_first_layer_filament_overrides(cfg: dict, overrides: Optional[dict] = None) -> list[str]:
    """Write FIRST_LAYER_FILAMENT_OVERRIDES for every filament of ``cfg`` and list
    the keys in each filament's different_settings_to_system[1..F] (Bambu Studio
    reverts unlisted filament values to the system filament preset).  Returns the
    keys written."""
    ov = dict(FIRST_LAYER_FILAMENT_OVERRIDES if overrides is None else overrides)
    F = _filament_count(cfg)
    if F < 1:
        raise ValueError("project settings have no filaments")
    for k, v in ov.items():
        cfg[k] = [str(v)] * F
    dsts = list(cfg.get('different_settings_to_system') or [])
    while len(dsts) < F + 2:
        dsts.append('')
    for i in range(1, F + 1):
        have = [x for x in str(dsts[i]).split(';') if x]
        dsts[i] = ';'.join(sorted(set(have) | set(ov)))
    cfg['different_settings_to_system'] = dsts
    return sorted(ov)


def apply_first_layer_settings(cfg: dict, system: Optional[dict] = None) -> dict:
    """Apply the first-layer adhesion settings to an EXISTING project config
    (used to patch already generated 3MFs): process keys shaped like the X2D
    system preset, different_settings_to_system[0] recomputed, filament keys +
    their difference lists.  The 0.08 mm first layer is never touched."""
    sysp = (system or load_system_process())['config']
    for k, v in FIRST_LAYER_PROCESS_OVERRIDES.items():
        cfg[k] = _shaped_like(sysp.get(k), v)
    diff = process_diff_vs_system(cfg, sysp)
    dsts = list(cfg.get('different_settings_to_system') or [''])
    dsts[0] = ';'.join(diff)
    cfg['different_settings_to_system'] = dsts
    fil = apply_first_layer_filament_overrides(cfg)
    return {'process_keys': sorted(FIRST_LAYER_PROCESS_OVERRIDES), 'filament_keys': fil,
            'process_keys_changed_vs_system': diff}


LAYER_CONFIG_RANGES_FILE = 'Metadata/layer_config_ranges.xml'


def layer_config_ranges_xml(ranges: Sequence[tuple[float, float, float]], object_index: int = 1) -> str:
    """Bambu Studio height-range modifiers for ONE model object.

    ranges: (min_z, max_z, layer_height) in the object's own coordinates (bottom
    at z = 0).  ``object_index`` is the 1-based index of the ModelObject in the
    model (the importer keys the file by index, not by the XML object id).
    Layout copied from bbs_3mf.cpp _add_layer_config_ranges_file_to_archive.
    """
    lines = ['<?xml version="1.0" encoding="utf-8"?>', '<objects>', f' <object id="{int(object_index)}">']
    for lo, hi, h in ranges:
        lines.append(f'  <range min_z="{float(lo):g}" max_z="{float(hi):g}">')
        lines.append(f'   <option opt_key="layer_height">{float(h):g}</option>')
        lines.append('  </range>')
    lines += [' </object>', '</objects>', '']
    return '\n'.join(lines)


def backing_layer_ranges(total_layers: int, structure: str, layer_h: float = LAYER_H,
                         colour_layers: int = N_LAYERS,
                         backing_layer_h: Optional[float] = DEFAULT_BACKING_LAYER_H,
                         first_layer_mm: Optional[float] = None) -> list[tuple[float, float, float]]:
    """Height range(s) covering the spacer between the colour stacks.

    single-sided (face down): colour at z 0 .. first_layer + (colour_layers-1)*layer_h,
    backing above it up to the top.  double-sided: colour on both faces, backing in
    between (the top face has no thick first layer).  first_layer_mm None -> layer_h.
    Returns [] when disabled or when the backing is thinner than one thick layer.
    """
    if not backing_layer_h or float(backing_layer_h) <= float(layer_h) + 1e-9:
        return []
    fl = float(layer_h) if first_layer_mm is None else float(first_layer_mm)
    lo = fl + (colour_layers - 1) * float(layer_h)
    hi = fl + (int(total_layers) - 1) * float(layer_h)
    if structure == 'double':
        hi -= colour_layers * float(layer_h)
    if hi - lo < float(backing_layer_h) - 1e-9:
        return []
    return [(round(lo, 6), round(hi, 6), float(backing_layer_h))]


def face_up_backing_schedule(spacer_mm, first_layer_mm, layer_h, backing_layer_h=None):
    """Snap backing to first layer + whole backing layers (nearest thickness)."""
    h = float(backing_layer_h or layer_h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError('backing layer height must be positive and finite')
    count = 1 + max(0, int(round((spacer_mm - first_layer_mm) / h)))
    height = round(first_layer_mm + (count - 1) * h, 6)
    return count, height, h


def _name_object(main_xml: str, ms_xml: str, name: str) -> tuple[str, str]:
    """Give the plaque ModelObject a name (Bambu's conflict warnings print it)."""
    safe = quoteattr(name)
    main_xml, _ = re.subn(r'<object id="(\d+)"(?![^>]*\bname=)([^>]*type="model")',
                          lambda m: f'<object id="{m.group(1)}" name={safe}{m.group(2)}', main_xml, count=1)
    ms_xml, _ = re.subn(r'(<object id="\d+">\s*\n)',
                        lambda m: m.group(1) + f'    <metadata key="name" value={safe} />\n', ms_xml, count=1)
    return main_xml, ms_xml


# --------------------------------------------------------------------------- helpers
def _slug(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    base = re.sub(r'[<>:"/\\|?*\s]+', '_', base).strip('_') or 'untitled'
    return base


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def apply_td_overrides(library: dict, td_scale: float = 1.0,
                       td_overrides: Optional[dict] = None) -> dict:
    """Return a copy of the library with every TD multiplied by ``td_scale`` and
    then per-filament values replaced by ``td_overrides`` ({name: td_mm}).

    The library TDs are HueForge-style guesses; Stack5 composites 0.08 mm layers,
    where the per-layer opacity o = 1 - exp(-k*0.08/TD) is very sensitive to TD
    (Lumina's photo-measured RYBW LUT implies TD ~0.05-0.2 mm for pigmented Bambu
    Basic spools and ~10 mm for White).  Calibrate here without editing the JSON.
    """
    scale = float(td_scale)
    if scale <= 0:
        raise ValueError("td_scale must be > 0")
    over = dict(td_overrides or {})
    missing = [k for k in over if k not in library]
    if missing:
        raise KeyError(f"td_overrides for unknown filaments: {missing}")
    out = {}
    for name, f in library.items():
        td = float(over[name]) if name in over else f.td_mm * scale
        src = f.td_source if (name not in over and scale == 1.0) else 'override'
        out[name] = Filament(name=f.name, hex=f.hex, rgb_lin=f.rgb_lin, td_mm=td, td_source=src,
                             settings_id=f.settings_id, filament_id=f.filament_id)
    return out


def set_determinism(seed: int = 0) -> None:
    """cv2 + numpy seeds, colour-recipe logger off, numpy.asscalar shim (main.py:24-28)."""
    import cv2
    cv2.setRNGSeed(int(seed))
    np.random.seed(int(seed))
    os.environ['LUMINA_COLOR_RECIPE_POLICY'] = 'off'
    if not hasattr(np, 'asscalar'):
        setattr(np, 'asscalar', lambda a: a.item())


def working_resolution(image_path: str, width_mm: float) -> tuple[int, int]:
    """Lumina HiFi target size: w = int(width_mm*10), h = int(w * H/W)."""
    with Image.open(image_path) as im:
        W, H = im.size
    tw = int(float(width_mm) * HIFI_PX_PER_MM)
    th = int(tw * H / W)
    return tw, th


def working_copy(image_path: str, width_mm: float, out_png: str) -> str:
    """Copy of the input at Lumina's working resolution (RGBA PNG, NEAREST - the
    same resampling Lumina applies, so its own resize becomes the identity)."""
    tw, th = working_resolution(image_path, width_mm)
    img = Image.open(image_path).convert('RGBA').resize((tw, th), Image.Resampling.NEAREST)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    img.save(out_png)
    return out_png


def _run_nonce() -> str:
    """Unique per process+instant.  convert_image_to_3d names its output
    '<image stem>_Lumina_HiFi_Unknown_<YYYYmmdd_HHMMSS>.3mf' in config.OUTPUT_DIR,
    so two runs of the same image within one second would clobber each other
    (observed: interleaved zip, Bad CRC); a nonce in the image stem prevents it."""
    return f"{os.getpid():x}{time.time_ns() & 0xFFFFFFFF:08x}"


def stamp_advisory(image_path: str, width_mm: float, corner: str, out_png: str) -> str:
    """Resize a copy of the image to the working resolution (NEAREST, as Lumina
    does) and draw the binary PARENTAL ADVISORY label (core.band.pipeline mask)
    into a bottom corner: label width 13 % of the image, margin 3.5 %."""
    corner = (corner or '').lower()
    if corner not in ADVISORY_CORNERS:
        raise ValueError(f"advisory corner must be one of {ADVISORY_CORNERS}, got {corner!r}")
    tw, th = working_resolution(image_path, width_mm)
    img = Image.open(image_path).convert('RGBA').resize((tw, th), Image.Resampling.NEAREST)
    arr = np.array(img)
    W, H = tw, th
    LW = max(int(ADVISORY_LABEL_FRAC * W), 8)
    LH = max(int(0.60 * LW), 4)
    lab = advisory_stamp_mask(LW, LH)                      # True = light
    m2 = int(ADVISORY_MARGIN_FRAC * W)
    r0 = max(H - m2 - LH, 0)
    c0 = {'br': W - m2 - LW, 'bl': m2, 'bc': (W - LW) // 2}[corner]
    c0 = int(np.clip(c0, 0, max(W - LW, 0)))
    patch = lab[:min(LH, H - r0), :min(LW, W - c0)]
    region = arr[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]]
    region[..., :3] = np.where(patch[..., None], 255, 0)
    region[..., 3] = 255
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    Image.fromarray(arr, 'RGBA').save(out_png)
    return out_png


# --------------------------------------------------------------------------- Lumina call
def _merge_layers_no_dilation(self, voxel_matrix, mat_id):
    """HighFidelityMesher._merge_layers_with_dilation without the 3x3 cv2.dilate.

    Lumina dilates every material mask by one pixel before meshing, so the parts
    of neighbouring materials overlap by 0.1 mm on every boundary; Bambu Studio
    carves overlapping parts in favour of the later part (PrintObjectSlice.cpp
    clip_multipart_objects, always on), which re-assigns 15-40 % of the pixel
    stacks.  The greedy rectangles are exact pixel boxes, so the undilated masks
    already tile the layer watertight.
    """
    layer_groups = []
    prev_mask = None
    start_z = 0
    for z in range(voxel_matrix.shape[0]):
        curr_mask = (voxel_matrix[z] == mat_id)
        if not np.any(curr_mask):
            if prev_mask is not None and np.any(prev_mask):
                layer_groups.append((start_z, z - 1, prev_mask))
                prev_mask = None
            continue
        if prev_mask is None:
            start_z = z
            prev_mask = curr_mask.copy()
        elif np.array_equal(curr_mask, prev_mask):
            pass
        else:
            layer_groups.append((start_z, z - 1, prev_mask))
            start_z = z
            prev_mask = curr_mask.copy()
    if prev_mask is not None and np.any(prev_mask):
        layer_groups.append((start_z, voxel_matrix.shape[0] - 1, prev_mask))
    return layer_groups


def run_lumina(image_path: str, lut_path: str, mode_key: str, width_mm: float, spacer_mm: float,
               structure: str, quantize_colors: int, smooth_sigma: float, backing_slot: int,
               hue_weight: float = 0.0, skip_glb: bool = True, enable_cleanup: bool = True,
               dilate: bool = False, min_region_px: int = DEFAULT_MIN_REGION_PX,
               metric: str = 'lumina', wL: float = 1.0, hue_params=None,
               stopping=None) -> dict:
    """Call core.converter.convert_image_to_3d (HiFi, native algorithm) and capture
    the processor result (material_matrix etc.) that Lumina does not return.

    dilate=False patches out the mesher's 1-px mask dilation (parts would overlap);
    min_region_px > 0 removes per-layer material islands smaller than that (see
    core.stack5.cleanup) between Lumina's matching and its meshing, keeping
    matched_rgb (preview) in sync.  Stats of both are returned under 'cleanup'.
    metric 'hue' / 'lab' installs a core.stack5.metric.Stack5Matcher as the
    processor's hue_matcher so every quantised colour is matched with the SAME
    metric the palette search used; 'lumina' keeps Lumina's 8-bit Lab KDTree.
    """
    import core.converter as conv
    import core.mesh_generators as meshgen
    from core.image_processing import LuminaImageProcessor

    if structure not in STRUCTURE_MODES:
        raise ValueError(f"structure must be one of {sorted(STRUCTURE_MODES)}, got {structure!r}")
    capture: dict = {}
    Orig = conv.LuminaImageProcessor

    class _Capturing(Orig):  # type: ignore[misc, valid-type]
        def _process_high_fidelity_mode(self, rgb_arr, *args, **kwargs):
            from core.stack5.edges import contrast_edges
            matched, matrix, target, debug = super()._process_high_fidelity_mode(rgb_arr, *args, **kwargs)
            edges = contrast_edges(rgb_arr)
            # Bound target diversity for the exhaustive variable-depth search.
            edge_rgb = (np.rint(rgb_arr[edges].astype(float) / 17) * 17).astype(np.uint8)
            target[edges] = edge_rgb
            if edges.any():
                edge_matcher = make_matcher('lab', self.lut_rgb, wL=2.0)
                unique_edges, inverse = np.unique(edge_rgb, axis=0, return_inverse=True)
                ids = edge_matcher.match_colors_batch(unique_edges, k=32)[inverse]
                matrix[edges] = self.ref_stacks[ids]
                matched[edges] = self.lut_rgb[ids]
            debug['quantized_image'] = target.copy()
            self._edge_restore = (edges, matrix[edges].copy(), matched[edges].copy())
            return matched, matrix, target, debug

        def process_image(self, *a, **k):
            matcher = make_matcher(metric, self.lut_rgb, wL=wL, hue_params=hue_params)
            if matcher is not None:
                self.hue_matcher = matcher          # LuminaImageProcessor uses it instead of its KDTree
                print(f"[STACK5] pixel matcher: {matcher.describe()}")
            capture['matcher'] = matcher.describe() if matcher is not None else 'lumina 8-bit Lab KDTree'
            res = Orig.process_image(self, *a, **k)
            edges, edge_stacks, edge_rgb = self._edge_restore
            res['material_matrix'][edges] = edge_stacks
            res['matched_rgb'][edges] = edge_rgb
            res['material_matrix'][~np.asarray(res['mask_solid'], bool)] = -1
            res['protected_edges'] = edges & np.asarray(res['mask_solid'], bool)
            mm = np.asarray(res['material_matrix'])
            mask = np.asarray(res['mask_solid'], bool)
            info = {'min_region_px': int(min_region_px), 'before': region_stats(mm, mask, max(int(min_region_px), 16))}
            if int(min_region_px) > 1 and stopping is None:
                new_mm, st = min_region_cleanup(mm, mask, int(min_region_px), protected_mask=res['protected_edges'])
                res['material_matrix'] = new_mm
                res['matched_rgb'] = resync_matched_rgb(new_mm, mask, res['matched_rgb'], self.lut_rgb, self.ref_stacks)
                info['cleanup'] = st
                info['after'] = region_stats(new_mm, mask, max(int(min_region_px), 16))
            else:
                info['cleanup'] = None
                info['after'] = info['before']
            if stopping is not None:
                from core.stack5.stopping import stop_at_best_match
                new_mm, preview, heights, report = stop_at_best_match(
                    res, **stopping, metric=metric, wL=wL, hue_params=hue_params,
                    min_region_px=min_region_px)
                res['material_matrix'], res['matched_rgb'] = new_mm, preview
                res['stop_layers'], res['early_stopping'] = heights, report
                info['cleanup'] = report['whole_recipe_cleanup']
                info['after'] = region_stats(new_mm, mask, max(int(min_region_px), 16))
            capture['cleanup'] = info
            capture['result'] = res
            capture['processor'] = self
            return res

    orig_preview = conv._create_preview_mesh
    orig_merge = meshgen.HighFidelityMesher._merge_layers_with_dilation
    conv.LuminaImageProcessor = _Capturing
    if skip_glb:
        conv._create_preview_mesh = lambda *a, **k: None
    if not dilate:
        meshgen.HighFidelityMesher._merge_layers_with_dilation = _merge_layers_no_dilation
    try:
        ret = conv.convert_image_to_3d(
            image_path, lut_path, float(width_mm), float(spacer_mm), STRUCTURE_MODES[structure],
            False, 0, mode_key,                 # auto_bg, bg_tol, color_mode
            False, 0.0, 0.0, 0.0, None,         # add_loop, loop_width, loop_length, loop_hole, loop_pos
            modeling_mode=ModelingMode.HIGH_FIDELITY, quantize_colors=int(quantize_colors),
            blur_kernel=0, smooth_sigma=float(smooth_sigma), backing_color_id=int(backing_slot),
            separate_backing=False, enable_cleanup=bool(enable_cleanup), hue_weight=float(hue_weight))
    finally:
        conv.LuminaImageProcessor = Orig
        conv._create_preview_mesh = orig_preview
        meshgen.HighFidelityMesher._merge_layers_with_dilation = orig_merge
    threemf, glb, preview_img, status, recipe_txt = ret
    if threemf is None:
        raise RuntimeError(f"Lumina conversion failed: {status}")
    if 'result' not in capture:
        raise RuntimeError("did not capture the processor result from convert_image_to_3d")
    return {'threemf': threemf, 'glb': glb, 'preview_img': preview_img, 'status': status,
            'recipe_txt': recipe_txt, 'result': capture['result'], 'processor': capture['processor'],
            'cleanup': capture.get('cleanup'), 'dilated': bool(dilate), 'matcher': capture.get('matcher')}


# --------------------------------------------------------------------------- stats
def material_stats(result: dict, n_slots: int, slot_names: Sequence[str], structure: str,
                   spacer_mm: float, backing_slot: int,
                   first_layer_mm: float = PrinterConfig.LAYER_HEIGHT,
                   layer_h: float = LAYER_H, orientation: str = 'face-down',
                   backing_layer_h: Optional[float] = None) -> dict:
    mm = np.asarray(result['material_matrix'])
    mask = np.asarray(result['mask_solid'], bool)
    solid = mm[mask]                                        # (P,L)
    P, L = solid.shape
    uniq = np.unique(solid, axis=0) if P else np.zeros((0, L), dtype=mm.dtype)
    valid = solid >= 0
    counts = np.bincount(solid[valid].ravel(), minlength=n_slots)[:n_slots].astype(np.float64)
    layer_share = counts / max(counts.sum(), 1.0)
    # First occupied viewing-order entry is the exposed material; all air means backing.
    first_occupied = np.argmax(valid, axis=1)
    surf = solid[np.arange(P), first_occupied].copy()
    surf[~valid.any(axis=1)] = backing_slot
    surf_counts = np.bincount(surf[surf >= 0], minlength=n_slots)[:n_slots].astype(np.float64)
    surf_share = surf_counts / max(P, 1)
    pure_px = np.all((solid == surf[:, None]) | ~valid, axis=1)
    pure_share_by = {}
    for i, nme in enumerate(slot_names):
        pure_share_by[nme] = float(np.count_nonzero(pure_px & (surf == i)) / max(P, 1))
    # tool changes: per printed layer (materials present - 1); spacer layers hold one material
    per_layer = []
    for j in range(L):
        ids = np.unique(solid[:, j][solid[:, j] >= 0])
        per_layer.append(int(ids.size))
    changes_optical = int(sum(max(0, n - 1) for n in per_layer))
    spacer_layers = max(1, int(round(float(spacer_mm) / PrinterConfig.LAYER_HEIGHT)))
    if orientation == 'face-up':
        spacer_layers, base_height, _ = face_up_backing_schedule(
            spacer_mm, first_layer_mm, layer_h, backing_layer_h)
    if structure == 'double':
        total_layers = 2 * L + spacer_layers
        tool_changes = 2 * changes_optical
    else:
        total_layers = L + spacer_layers
        tool_changes = changes_optical
    # transitions between consecutive layers whose material set changes (extra swaps)
    sets = [set(np.unique(solid[:, j][solid[:, j] >= 0]).tolist()) for j in range(L)]
    layer_seq = sets + [{int(backing_slot)}] * spacer_layers
    if structure == 'double':
        layer_seq = layer_seq + sets[::-1]
    if orientation == 'face-up':
        optical_sets = sets[::-1]
        while optical_sets and not optical_sets[-1]:
            optical_sets.pop()
        layer_seq = [{int(backing_slot)}] * spacer_layers + optical_sets
        total_layers = len(layer_seq)
    extra = 0
    for a, b in zip(layer_seq[:-1], layer_seq[1:]):
        if a and b and not (a & b):
            extra += 1
    q = result.get('debug_data', {}).get('quantized_image') if isinstance(result.get('debug_data'), dict) else None
    mean_de_matched = None
    if q is not None and P:
        qm = np.asarray(q)[mask].astype(np.float64) / 255.0
        mt = np.asarray(result['matched_rgb'])[mask].astype(np.float64) / 255.0
        d = np.linalg.norm(srgb_to_lab_d65(qm) - srgb_to_lab_d65(mt), axis=1)
        mean_de_matched = float(d.mean())
    h = hashlib.sha256(np.ascontiguousarray(mm.astype(np.int8)).tobytes()).hexdigest()
    tw, th = result['dimensions']
    return {
        'resolution_px': [int(tw), int(th)],
        'pixel_scale_mm': float(result['pixel_scale']),
        'n_solid_pixels': int(P),
        'n_unique_stacks_used': int(uniq.shape[0]),
        'n_pure_stacks_used': int(sum(len(set(row[row >= 0].tolist())) <= 1 for row in uniq)),
        'per_slot_layer_share': {n: float(layer_share[i]) for i, n in enumerate(slot_names)},
        'per_slot_viewing_surface_share': {n: float(surf_share[i]) for i, n in enumerate(slot_names)},
        'pure_stack_pixel_share': float(np.count_nonzero(pure_px) / max(P, 1)),
        'pure_stack_pixel_share_by_filament': pure_share_by,
        'materials_per_optical_layer': per_layer,
        'materials_per_optical_layer_sets': [sorted(int(x) for x in s) for s in sets],
        'printed_layer_material_sets': [sorted(int(x) for x in s) for s in layer_seq],
        'tool_changes_est': int(tool_changes),
        'tool_changes_est_with_layer_transitions': int(tool_changes + extra),
        'total_print_layers': int(total_layers),
        'total_height_mm': float(base_height + (total_layers - spacer_layers) * layer_h
                                 if orientation == 'face-up' else first_layer_mm + (total_layers - 1) * layer_h),
        'mean_dE_matched_quantized_vs_lut': mean_de_matched,
        'material_matrix_sha256': h,
    }


# --------------------------------------------------------------------------- 3MF post-processing
def _parse_parts(model_settings_xml: str) -> list[tuple[str, int]]:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(model_settings_xml)
    parts = []
    for part in root.iter('part'):
        name, ext = None, None
        for md in part.findall('metadata'):
            if md.get('key') == 'name':
                name = md.get('value')
            elif md.get('key') == 'extruder':
                ext = int(md.get('value'))
        parts.append((name, ext))
    return parts


_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _zinfo(name: str) -> zipfile.ZipInfo:
    """ZipInfo with a constant timestamp so identical inputs give a byte-identical 3MF."""
    zi = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = 0o644 << 16
    return zi


def _copy_entry_counting_triangles(zin: zipfile.ZipFile, info: zipfile.ZipInfo,
                                   zout: zipfile.ZipFile) -> list[int]:
    """Stream-copy one zip member; return '<triangle ' counts per '<object ' seen."""
    zi = _zinfo(info.filename)
    counts: list[int] = []
    cur = -1
    carry = b''
    with zin.open(info) as src, zout.open(zi, 'w', force_zip64=True) as dst:
        while True:
            chunk = src.read(8 << 20)
            if not chunk:
                break
            dst.write(chunk)
            buf = carry + chunk
            nl = buf.rfind(b'\n')
            if nl < 0:
                carry = buf
                continue
            block, carry = buf[:nl + 1], buf[nl + 1:]
            pos = 0
            while True:
                k = block.find(b'<object ', pos)
                if k < 0:
                    if cur >= 0:
                        counts[cur] += block.count(b'<triangle ', pos)
                    break
                if cur >= 0:
                    counts[cur] += block.count(b'<triangle ', pos, k)
                counts.append(0)
                cur = len(counts) - 1
                pos = k + 8
        if carry and cur >= 0:
            counts[cur] += carry.count(b'<triangle ')
    return counts


_VERTEX_Z = re.compile(rb'(<vertex [^>]*?z=")([^"]+)(")')


def _copy_entry_shifting_z(zin: zipfile.ZipFile, info: zipfile.ZipInfo, zout: zipfile.ZipFile,
                           dz: float, z_min: float, vertex_map=None) -> list[int]:
    """Stream-copy an object .model like _copy_entry_counting_triangles, but lift
    every vertex with z >= z_min by dz.

    This is how a first layer thicker than the colour layer is realised: the
    converter meshed the viewing layer at Z 0..layer_h, so shifting everything
    above it by (first_layer - layer_h) stretches that one voxel layer to the
    slicer's first-layer height and keeps every other boundary on a whole
    layer.  Vertices at z = 0 (the plate face) are untouched.  The file has a
    handful of distinct z values, so substitutions are memoised.
    """
    zi = _zinfo(info.filename)
    counts: list[int] = []
    cur = -1
    carry = b''
    memo: dict[bytes, bytes] = {}
    lim = float(z_min) - 1e-6

    def lift(m):
        raw = m.group(2)
        out = memo.get(raw)
        if out is None:
            v = float(raw)
            out = (('%.4f' % (v + dz)).rstrip('0').rstrip('.').encode() if v >= lim else raw)
            memo[raw] = out
        return m.group(1) + out + m.group(3)

    with zin.open(info) as src, zout.open(zi, 'w', force_zip64=True) as dst:
        while True:
            chunk = src.read(8 << 20)
            if not chunk:
                break
            buf = carry + chunk
            nl = buf.rfind(b'\n')
            if nl < 0:
                carry = buf
                continue
            block, carry = buf[:nl + 1], buf[nl + 1:]
            dst.write(vertex_map(block) if vertex_map else _VERTEX_Z.sub(lift, block))
            pos = 0
            while True:
                k = block.find(b'<object ', pos)
                if k < 0:
                    if cur >= 0:
                        counts[cur] += block.count(b'<triangle ', pos)
                    break
                if cur >= 0:
                    counts[cur] += block.count(b'<triangle ', pos, k)
                counts.append(0)
                cur = len(counts) - 1
                pos = k + 8
        if carry:
            dst.write(vertex_map(carry) if vertex_map else _VERTEX_Z.sub(lift, carry))
            if cur >= 0:
                counts[cur] += carry.count(b'<triangle ')
    return counts


def face_up_vertex_map(width_mm: float, spacer_mm: float, layer_h: float,
                       first_layer_mm: float, backing_layer_h: Optional[float] = None):
    """Rotate native face-down geometry 180 degrees about Y, then scale Z.

    The native mesh uses 0.08 mm voxels. Backing is mapped independently so
    the requested physical base thickness is preserved. The normal first-layer
    stretch is incorporated here. Two axis reversals preserve winding.
    """
    native_base = max(1, round(spacer_mm / LAYER_H)) * LAYER_H
    _, base, _ = face_up_backing_schedule(spacer_mm, first_layer_mm, layer_h, backing_layer_h)
    colour_end = N_LAYERS * LAYER_H
    vertex = re.compile(rb'<vertex\s+[^>]*?/?>')
    attr = re.compile(rb'([xz])="([^"]+)"')
    def rewrite(block):
        def change_vertex(match):
            def change_attr(m):
                value = float(m[2])
                if m[1] == b'x':
                    value = width_mm - value
                elif value <= colour_end + 1e-6:
                    value = base + (colour_end - value) * layer_h / LAYER_H
                else:
                    value = base * (colour_end + native_base - value) / native_base
                return m[1] + b'="' + f'{max(0.0, value):.6f}'.encode() + b'"'
            return attr.sub(change_attr, match[0])
        return vertex.sub(change_vertex, block)
    return rewrite


def _fmt_transform(tx: float, ty: float, tz: float = 0.0) -> str:
    return f"1 0 0 0 1 0 0 0 1 {tx:g} {ty:g} {tz:g}"


def _place_build_item(main_xml: str, transform: str) -> str:
    """Give the (single) build item a translation - Bambu Studio applies the item
    transform as the instance offset (bbs_3mf.cpp _handle_start_item) and does
    NOT centre 3MF imports (Plater.cpp: center_around_origin only for non-3MF)."""
    new, n = re.subn(r'<item\s+objectid="(\d+)"\s*/>',
                     lambda m: f'<item objectid="{m.group(1)}" transform="{transform}" printable="1"/>',
                     main_xml, count=1)
    if n != 1:
        raise ValueError("3D/3dmodel.model: expected exactly one bare <item objectid=.../>")
    return new


def _add_assemble(ms_xml: str, transform: str) -> str:
    m = re.search(r'<object\s+id="(\d+)"', ms_xml)
    if not m:
        raise ValueError("model_settings.config: no <object id=...>")
    oid = m.group(1)
    if '<assemble>' in ms_xml:
        return ms_xml
    block = ('  <assemble>\n'
             f'   <assemble_item object_id="{oid}" instance_id="0" transform="{transform}" offset="0 0 0"/>\n'
             '  </assemble>\n</config>')
    return ms_xml.replace('</config>', block, 1)


def postprocess_3mf(src_3mf: str, dst_3mf: str, palette: Sequence[Filament], backing_slot: int,
                    lut_rgb: np.ndarray, first_layer_mm: float = 0.08, layer_h: float = LAYER_H,
                    title: Optional[str] = None, template_sources=None,
                    printed_layer_sets: Optional[Sequence[Sequence[int]]] = None,
                    plaque_size_mm: Optional[Sequence[float]] = None,
                    flush_scale: float = 1.0, min_flush: Optional[float] = None,
                    tower_fit: str = 'error',
                    layer_ranges: Optional[Sequence[tuple[float, float, float]]] = None,
                    vertex_map=None) -> dict:
    """Rewrite Lumina's 3MF with an X2D-consistent Metadata/project_settings.config
    for the F parts actually exported (slot order preserved, extruders 1..F), a
    palette-specific flush matrix, a prime tower sized from the modelled per-layer
    purge, and a build-item transform placing the plaque next to the tower.

    printed_layer_sets: material slot ids present in every printed layer (optical
    layers, spacer layers, mirrored layers for double-sided) - from material_stats.
    plaque_size_mm: (w, h) of the mesh footprint.  min_flush None -> derived from
    the template (nozzle_volume minus long-retraction volume, Bambu's rule).
    layer_ranges: optional [(min_z, max_z, layer_height)] height-range modifiers
    written to Metadata/layer_config_ranges.xml (backing_layer_ranges()).
    """
    if tower_fit not in TOWER_FIT_MODES:
        raise ValueError(f"tower_fit must be one of {TOWER_FIT_MODES}")
    names = [f.name for f in palette]
    with zipfile.ZipFile(src_3mf) as zin:
        ms_xml = zin.read('Metadata/model_settings.config').decode('utf-8')
        main_xml = zin.read('3D/3dmodel.model').decode('utf-8')
        lumina_cfg = json.loads(zin.read('Metadata/project_settings.config').decode('utf-8'))
        parts = _parse_parts(ms_xml)
        if not parts:
            raise ValueError(f"{src_3mf}: no parts in model_settings.config")
        used_slots = []
        for pname, ext in parts:
            if pname not in names:
                raise ValueError(f"part {pname!r} is not a palette slot {names}")
            used_slots.append(names.index(pname))
        exts = [e for _, e in parts]
        if exts != list(range(1, len(parts) + 1)):
            raise ValueError(f"unexpected extruder numbering {exts}")
        F = len(parts)
        # verify Lumina's filament_colour order: pure-stack LUT colour of each used slot
        pure_rows = pure_stack_indices(len(palette), N_LAYERS)
        expected = ['#%02X%02X%02X' % tuple(int(c) for c in lut_rgb[pure_rows[s]]) for s in used_slots]
        got = [str(h).upper() for h in lumina_cfg.get('filament_colour', [])]
        order_ok = got == expected
        used_fils = [palette[s] for s in used_slots]
        template = writer3mf.load_template(template_sources)
        # flush matrix + prime tower from the palette actually exported (extruder order)
        if min_flush is None:
            min_flush = flushmod.min_flush_from_config(template['base'])
        if printed_layer_sets is None:
            printed_layer_sets = [list(range(len(used_slots)))] * N_LAYERS
        else:
            slot_to_ext = {s: i for i, s in enumerate(used_slots)}
            printed_layer_sets = [[slot_to_ext[int(s)] for s in layer if int(s) in slot_to_ext]
                                  for layer in printed_layer_sets]
        if plaque_size_mm is None:
            raise ValueError("plaque_size_mm (mesh footprint w, h in mm) is required")
        scale = float(flush_scale)
        while True:
            tower_plan = flushmod.plan_tower([f.hex for f in used_fils], printed_layer_sets,
                                             float(plaque_size_mm[0]), float(plaque_size_mm[1]), layer_h,
                                             min_flush=float(min_flush), scale=scale)
            tw_ = tower_plan['tower']
            if tw_['fits'] or tower_fit != 'auto' or scale <= TOWER_FIT_MIN_SCALE + 1e-9:
                break
            new_scale = max(TOWER_FIT_MIN_SCALE, round(scale * TOWER_FIT_STEP, 3))
            print(f"[STACK5] prime tower does not fit at purge scale {scale:g} "
                  f"(depth {tw_['depth_modelled']:.0f} > {tw_['depth_available']:.0f} mm): trying {new_scale:g}")
            scale = new_scale
        tower_plan['flush_scale_requested'] = float(flush_scale)
        if not tw_['fits']:
            msg = (f"prime tower does not fit: modelled depth {tw_['depth_modelled']:.1f} mm > "
                   f"{tw_['depth_available']:.1f} mm available (width {tw_['width']:g} mm beside a "
                   f"{plaque_size_mm[0]:g} mm plaque; worst layer {max(tower_plan['purge_per_layer_mm3'])} mm3). "
                   f"Reduce --width or --flush-scale.")
            if tower_fit in ('error', 'auto'):
                raise ValueError(msg)
            print(f"[STACK5] WARNING: {msg}")
        cfg = writer3mf.build_project_settings(template, used_fils,
                                               stack5_print_overrides(first_layer_mm, layer_h, tower_plan))
        process_diff = apply_stack5_process_preset(cfg, stack5_print_overrides(first_layer_mm, layer_h, tower_plan))
        first_layer_filament_keys = apply_first_layer_filament_overrides(cfg)
        # Old album templates contain A1 machine scripts despite their X2D label.
        # Use the actual resolved X2D scripts, and persist the printer overrides.
        with open(os.path.join(REPO, 'assets', 'x2d_machine_gcodes.json'), encoding='utf-8') as fh:
            machine_gcodes = json.load(fh)['config']
        cfg.update(machine_gcodes)
        printer_diff = set(filter(None, cfg['different_settings_to_system'][-1].split(';')))
        cfg['different_settings_to_system'][-1] = ';'.join(sorted(printer_diff | set(machine_gcodes)))
        main_xml, ms_xml = _name_object(main_xml, ms_xml, title or os.path.splitext(os.path.basename(dst_3mf))[0])
        lay = tower_plan['layout']
        transform = _fmt_transform(round(lay['plaque_xy'][0], 4), round(lay['plaque_xy'][1], 4), 0.0)
        main_xml = _place_build_item(main_xml, transform)
        ms_xml = _add_assemble(ms_xml, transform)
        if title:
            ms_xml = re.sub(r'(key="plater_name"\s+value=)""', lambda m: m.group(1) + quoteattr(title), ms_xml, count=1)
        os.makedirs(os.path.dirname(os.path.abspath(dst_3mf)) or '.', exist_ok=True)
        tri_counts: list[int] = []
        ranges = [tuple(r) for r in (layer_ranges or [])]
        with zipfile.ZipFile(dst_3mf, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename == LAYER_CONFIG_RANGES_FILE:
                    continue                                  # rewritten below
                if info.filename == 'Metadata/project_settings.config':
                    zout.writestr(_zinfo(info.filename), json.dumps(cfg, indent=4, ensure_ascii=False))
                elif info.filename == 'Metadata/model_settings.config':
                    zout.writestr(_zinfo(info.filename), ms_xml)
                elif info.filename == '3D/3dmodel.model':
                    zout.writestr(_zinfo(info.filename), main_xml)
                elif info.filename.startswith('3D/Objects/') and info.filename.endswith('.model'):
                    dz = float(first_layer_mm) - float(layer_h)
                    if vertex_map is not None:
                        tri_counts += _copy_entry_shifting_z(zin, info, zout, 0, 0, vertex_map)
                    elif dz > 1e-9:
                        tri_counts += _copy_entry_shifting_z(zin, info, zout, dz, float(layer_h))
                    else:
                        tri_counts += _copy_entry_counting_triangles(zin, info, zout)
                else:
                    zout.writestr(_zinfo(info.filename), zin.read(info.filename))
            if ranges:
                zout.writestr(_zinfo(LAYER_CONFIG_RANGES_FILE), layer_config_ranges_xml(ranges, object_index=1))
    return {
        'n_objects': F,
        'tower_plan': tower_plan,
        'layer_ranges': [list(r) for r in ranges],
        'process_preset': PRINT_SETTINGS_ID,
        'process_keys_changed_vs_system': process_diff,
        'first_layer': {k: cfg[k] for k in list(FIRST_LAYER_PROCESS_OVERRIDES) + list(FIRST_LAYER_FILAMENT_OVERRIDES)},
        'first_layer_filament_keys': first_layer_filament_keys,
        'build_transform': transform,
        'plaque_xy_mm': list(lay['plaque_xy']),
        'parts': [{'part_id': i + 1, 'name': n, 'extruder': e, 'slot': s}
                  for i, ((n, e), s) in enumerate(zip(parts, used_slots))],
        'used_slots': used_slots,
        'unused_slots': [s for s in range(len(palette)) if s not in used_slots],
        'slot_to_extruder': {names[s]: i + 1 for i, s in enumerate(used_slots)},
        'filament_colour': cfg['filament_colour'],
        'lumina_filament_colour': got,
        'lumina_filament_colour_expected_pure_lut': expected,
        'lumina_filament_colour_order_verified': bool(order_ok),
        'triangles_per_object': tri_counts,
        'triangles_total': int(sum(tri_counts)),
        'backing_extruder': (used_slots.index(int(backing_slot)) + 1) if int(backing_slot) in used_slots else None,
        'project_settings': {k: cfg[k] for k in ('printer_settings_id', 'print_settings_id', 'printer_model',
                                                 'layer_height', 'initial_layer_print_height', 'wall_loops',
                                                 'initial_layer_speed', 'initial_layer_infill_speed',
                                                 'elefant_foot_compensation', 'hot_plate_temp_initial_layer',
                                                 'textured_plate_temp_initial_layer', 'close_fan_the_first_x_layers',
                                                 'nozzle_temperature_initial_layer',
                                                 'top_shell_layers', 'bottom_shell_layers',
                                                 'sparse_infill_density', 'sparse_infill_pattern',
                                                 'enable_prime_tower', 'prime_tower_rib_wall',
                                                 'prime_tower_width', 'prime_tower_infill_gap',
                                                 'prime_tower_brim_width', 'wipe_tower_x', 'wipe_tower_y',
                                                 'flush_volumes_matrix', 'filament_settings_id',
                                                 'filament_colour')},
        'flush_matrix_len': len(cfg[writer3mf.FLUSH_MATRIX_KEY]),
        'flush_vector_len': len(cfg[writer3mf.FLUSH_VECTOR_KEY]),
    }


# --------------------------------------------------------------------------- main entry
def convert_album_stack5(image_path: str, width_mm: float = 150.0,
                         filaments_json: Optional[str] = None,
                         palette: Optional[Sequence[str]] = None, backing: Optional[str] = None,
                         quantize_colors: int = 96, smooth_sigma: float = 10,
                         structure: str = 'single', spacer_mm: float = 1.0,
                         advisory: Optional[str] = None, out_dir: str = DEFAULT_OUT_DIR,
                         title: Optional[str] = None, seed: int = 0,
                         must_include: Sequence[str] = (), first_layer_mm: float = 0.20,
                         wL: float = 1.0, hist_k: int = 64, k_opaque: Optional[float] = None,
                         hue_weight: float = 0.0, keep_lumina_output: bool = False,
                         skip_glb: bool = True, td_scale: float = 1.0,
                         td_overrides: Optional[dict] = None,
                         lut_npz: Optional[str] = None,
                         chroma_weight: float = DEFAULT_CHROMA_WEIGHT,
                         spool_bonus: float = DEFAULT_SPOOL_BONUS, spool_de: float = DEFAULT_SPOOL_DE,
                         dominant_w: float = DEFAULT_DOMINANT_W,
                         prefer: Sequence[str] = (), prefer_tol: float = DEFAULT_PREFER_TOL,
                         metric: str = DEFAULT_METRIC, hue_params: dict | None = None,
                         max_spools: int = 5, min_spools: int | None = None,
                         spool_penalty: float = DEFAULT_SPOOL_PENALTY, need_share: float = DEFAULT_NEED_SHARE,
                         min_region_px: int = DEFAULT_MIN_REGION_PX,
                         dilate: bool = False, flush_scale: float = 1.0,
                         min_flush: Optional[float] = None, tower_fit: str = 'error',
                         backing_layer_h: Optional[float] = DEFAULT_BACKING_LAYER_H,
                         orientation: str = 'face-down', layer_h: float = LAYER_H,
                         early_stop: bool = True) -> dict:
    """image -> {threemf, preview_png, palette, backing, lut_npz, recipe_json, ams_txt, stats}.

    palette: 5 filament names (slot order) or None for the exhaustive search;
    backing: filament name (None -> searched: the backing of the winning
    configuration, palette_report['best']['backing']);
    advisory: 'br' | 'bl' | 'bc' | None; structure: 'single' | 'double';
    first_layer_mm: slicer first layer (default 0.20). Face-down stretches the
        viewing layer; face-up keeps it in the backing.
    orientation: face-down (legacy) or face-up (single sided).
    layer_h: optical layer thickness, 0.08 or experimental 0.04 mm;
    early_stop: face-up columns search all 0..5-layer recipes (default True);
        near-equal matches prefer simpler recipes. Ignored for face-down jobs.
    td_scale / td_overrides: optical calibration of the library TDs (see
    apply_td_overrides) used for BOTH the palette search and the LUT.
    lut_npz: a MEASURED LUT {rgb (N,3) uint8, stacks (N,5) int32 in palette-slot
    ids} to use instead of the synthetic one (requires an explicit ``palette``
    in the slot order the board was printed with).
    chroma_weight / spool_bonus / spool_de / dominant_w: palette scoring (see
    core.stack5.palette); prefer / prefer_tol: soft must-include; metric:
    'lumina' (OpenCV 8-bit Lab, what Lumina's KDTree uses) or 'lab' (weighted
    true Lab, wL).
    min_region_px: remove per-layer material islands below this many pixels
    (0 = off); dilate: keep Lumina's 1-px mask dilation (parts overlap; off).
    flush_scale / min_flush: Bambu flush-volume formula scaling and per-nozzle
    minimum (None -> from the X2D template); tower_fit: 'error' (default) or
    'warn' when the modelled prime tower does not fit beside the plaque.
    backing_layer_h: layer height for the backing layers via a Bambu height-range
    modifier (default 0.2; None keeps everything at layer_h).
    """
    t_all = time.perf_counter()
    timing: dict = {}
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if structure not in STRUCTURE_MODES:
        raise ValueError(f"structure must be one of {sorted(STRUCTURE_MODES)}")
    if orientation not in ('face-down', 'face-up'):
        raise ValueError("orientation must be face-down or face-up")
    if not np.isfinite(layer_h) or layer_h not in (0.04, 0.08):
        raise ValueError("colour layer height must be 0.08 or experimental 0.04 mm")
    if not np.isfinite(first_layer_mm) or not np.isfinite(spacer_mm):
        raise ValueError("first layer and spacer must be finite")
    if orientation == 'face-down' and layer_h != LAYER_H:
        raise ValueError("0.04 mm colour layers require face-up orientation")
    if orientation == 'face-up':
        if structure != 'single':
            raise ValueError("face-up currently requires single-sided structure")
        if spacer_mm < first_layer_mm:
            raise ValueError("face-up backing must be at least the first-layer thickness")
        face_up_backing_schedule(spacer_mm, first_layer_mm, layer_h, backing_layer_h)
        if not backing_layer_h and abs((spacer_mm - first_layer_mm) / layer_h - round((spacer_mm - first_layer_mm) / layer_h)) > 1e-6:
            raise ValueError("backing must equal first_layer_mm plus whole colour-height increments")
    stack5_print_overrides(first_layer_mm, layer_h)
    stopping_enabled = bool(early_stop and orientation == 'face-up')
    if stopping_enabled and dilate:
        raise ValueError('early stopping requires dilation off to keep stopped regions clear')
    optical_first = first_layer_mm if orientation == 'face-down' else layer_h
    if layer_h < 0.08:
        print('[STACK5] EXPERIMENTAL: 0.04 mm is below the X2D 0.4 nozzle profile minimum of 0.08 mm; '
              'printer minimum is preserved. Validate slicer output and a coupon before printing.')
    if tower_fit not in TOWER_FIT_MODES:
        raise ValueError(f"tower_fit must be one of {TOWER_FIT_MODES}")
    if advisory is not None and str(advisory).lower() in ('', 'none'):
        advisory = None
    slug = _slug(image_path)
    title = title or slug.replace('_', ' ')
    job_dir = os.path.join(out_dir, slug)
    os.makedirs(job_dir, exist_ok=True)

    set_determinism(seed)
    filaments_json = filaments_json or default_filaments_json()
    library = apply_td_overrides(load_filament_library(filaments_json), td_scale, td_overrides)
    model = BeerLambertModel(k_opaque)

    # 1. palette
    t0 = time.perf_counter()
    hist = image_hist(image_path, k=hist_k, seed=seed)
    scoring_kw = dict(chroma_weight=float(chroma_weight), spool_bonus=float(spool_bonus),
                      spool_de=float(spool_de), dominant_w=float(dominant_w))
    palette_report = None
    if palette:
        names = [p.strip() for p in palette if p and p.strip()]
        missing = [n for n in names if n not in library]
        if missing:
            raise KeyError(f"filaments not in library: {missing}; available: {sorted(library)}")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate filaments in palette: {names}")
        if len(names) < 2 or len(names) > 5:
            raise ValueError("palette must have 2..5 filaments")
        if backing and backing not in names:
            raise KeyError(f"backing {backing!r} must be one of the palette {names}")
        thick = layer_thicknesses(optical_first, layer_h, N_LAYERS)
        palette_eval = evaluate_palette(hist, names, library, backing=backing, model=model, wL=wL,
                                        layer_h=thick,
                                        metric=metric, hue_params=hue_params, **scoring_kw)
        palette_report = {'search': 'explicit palette' + ('' if backing else ' x all backings'),
                          'metric': metric, 'best': palette_eval, 'top': [palette_eval],
                          'backing_rule': f"fixed: {backing}" if backing else 'searched: lowest cost member'}
    else:
        if not 1 <= int(max_spools) <= 5:
            raise ValueError("max_spools must be within 1..5 (one AMS)")
        thick = layer_thicknesses(optical_first, layer_h, N_LAYERS)
        names, palette_report = select_palette(hist, library, model, n=int(max_spools), n_min=min_spools,
                                               layer_h=thick,
                                               must_include=must_include, backing=backing, wL=wL, prefer=prefer,
                                               prefer_tol=prefer_tol, metric=metric, hue_params=hue_params,
                                               spool_penalty=float(spool_penalty), need_share=float(need_share),
                                               **scoring_kw)
    fils = [library[n] for n in names]
    # the backing is part of the scored configuration: take it from the report
    backing_slot = choose_backing(fils, backing or palette_report['best']['backing'])
    timing['palette_s'] = time.perf_counter() - t0

    # 2. LUT + mode
    t0 = time.perf_counter()
    if lut_npz:
        if not palette:
            raise ValueError("a measured lut_npz needs an explicit palette in the board's slot order")
        data = np.load(lut_npz)
        lut_rgb = np.asarray(data['rgb'], dtype=np.uint8).reshape(-1, 3)
        lut_stacks = np.asarray(data['stacks'], dtype=np.int32)
        if lut_stacks.ndim != 2 or lut_stacks.shape[0] != lut_rgb.shape[0]:
            raise ValueError(f"{lut_npz}: stacks must be (N,L) matching rgb (N,3)")
        if lut_stacks.min() < 0 or lut_stacks.max() >= len(fils):
            raise ValueError(f"{lut_npz}: stack slot ids must be within 0..{len(fils) - 1}")
        pure_rows = pure_stack_indices(len(fils), lut_stacks.shape[1])
        if not all(np.all(lut_stacks[r] == i) for i, r in enumerate(pure_rows) if r < lut_stacks.shape[0]):
            raise ValueError(f"{lut_npz}: rows are not in enumerate_stacks() order (pure stacks misplaced)")
        lut_source = os.path.abspath(lut_npz)
        if orientation == 'face-up' or layer_h != LAYER_H:
            raise ValueError('face-up requires a synthetic LUT until a matching face-up calibration is available')
        if abs(float(first_layer_mm) - LAYER_H) > 1e-9:
            print(f"[STACK5] warning: measured LUT {lut_npz} was photographed from a board printed with a "
                  f"{LAYER_H:g} mm first layer; this print uses {first_layer_mm:g}, so the viewing layer is "
                  f"thicker than the board measured (translucent top colours will drift)")
        lut_npz = save_lut_npz(os.path.join(job_dir, f"{slug}_stack5_lut.npz"), lut_rgb, lut_stacks,
                               dict(lut_meta(fils, backing_slot, model, layer_h=layer_h, first_layer_mm=optical_first), measured_from=lut_source,
                                    model='measured'))
    else:
        lut_rgb, lut_stacks = synth_lut(fils, backing_slot, model, layer_h=layer_h, first_layer_mm=optical_first)
        lut_source = None
        lut_npz = save_lut_npz(os.path.join(job_dir, f"{slug}_stack5_lut.npz"), lut_rgb, lut_stacks,
                               dict(lut_meta(fils, backing_slot, model, layer_h=layer_h, first_layer_mm=optical_first),
                                    orientation=orientation, print_first_layer_mm=first_layer_mm))
    mode_key = register_stack5_mode(fils)
    timing['lut_s'] = time.perf_counter() - t0

    # 3. working copy of the input (unique name per run, see _run_nonce), with the
    #    optional advisory stamp drawn at the working resolution
    stamped_png = None
    if advisory:
        stamped_png = stamp_advisory(image_path, width_mm, advisory,
                                     os.path.join(job_dir, f"{slug}_input_stamped_{advisory.lower()}.png"))
    work_dir = os.path.join(job_dir, '.work')
    os.makedirs(work_dir, exist_ok=True)
    image_for_lumina = os.path.join(work_dir, f"{slug}_s5{_run_nonce()}.png")
    if stamped_png:
        Image.open(stamped_png).save(image_for_lumina)
    else:
        working_copy(image_path, width_mm, image_for_lumina)

    # 4. Lumina native conversion
    t0 = time.perf_counter()
    set_determinism(seed)
    try:
        lum = run_lumina(image_for_lumina, lut_npz, mode_key, width_mm, spacer_mm, structure,
                         quantize_colors, smooth_sigma, backing_slot, hue_weight=hue_weight,
                         skip_glb=skip_glb, dilate=dilate, min_region_px=min_region_px,
                         metric=metric, wL=wL, hue_params=hue_params,
                         stopping=(dict(filaments=fils, backing_slot=backing_slot, model=model, layer_h=layer_h)
                                   if stopping_enabled else None))
    finally:
        try:
            os.remove(image_for_lumina)
            if not os.listdir(work_dir):
                os.rmdir(work_dir)
        except OSError:
            pass
    timing['lumina_s'] = time.perf_counter() - t0

    # 5. post-process the 3MF (X2D project settings, flush matrix, prime tower,
    #    plaque placement) into the job directory
    t0 = time.perf_counter()
    stats = material_stats(lum['result'], len(fils), names, structure, spacer_mm, backing_slot,
                           first_layer_mm=first_layer_mm, layer_h=layer_h, orientation=orientation,
                           backing_layer_h=backing_layer_h)
    tw_px, th_px = lum['result']['dimensions']
    px_mm = float(lum['result']['pixel_scale'])
    threemf = os.path.join(job_dir, f"{slug}_stack5.3mf")
    ranges = backing_layer_ranges(stats['total_print_layers'], structure, LAYER_H, N_LAYERS, backing_layer_h,
                                  first_layer_mm=first_layer_mm)
    if orientation == 'face-up':
        _, actual_backing_mm, base_h = face_up_backing_schedule(
            spacer_mm, first_layer_mm, layer_h, backing_layer_h)
        ranges = [(0.0, actual_backing_mm, base_h)] if backing_layer_h else []
        stats['actual_backing_mm'] = actual_backing_mm
    try:
        post = postprocess_3mf(lum['threemf'], threemf, fils, backing_slot, lut_rgb,
                               first_layer_mm=first_layer_mm, layer_h=layer_h, title=title,
                               printed_layer_sets=stats['printed_layer_material_sets'],
                               plaque_size_mm=(tw_px * px_mm, th_px * px_mm),
                               flush_scale=flush_scale, min_flush=min_flush, tower_fit=tower_fit,
                               layer_ranges=ranges,
                               vertex_map=(face_up_vertex_map(tw_px * px_mm, spacer_mm, layer_h, first_layer_mm, backing_layer_h)
                                           if orientation == 'face-up' else None))
    finally:
        # always drop Lumina's intermediate export (tens of MB per album) unless asked
        # to keep it - a failed post-process (e.g. prime tower does not fit) must not
        # leave it behind in output/
        if not keep_lumina_output:
            for p in (lum['threemf'], lum['glb']):
                if p and os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    timing['postprocess_s'] = time.perf_counter() - t0

    stop_layers_path = None
    if stopping_enabled:
        stop_layers_path = os.path.join(job_dir, f'{slug}_stop_layers.npy')
        np.save(stop_layers_path, lum['result']['stop_layers'])
    preview_png = os.path.join(job_dir, f"{slug}_preview.png")
    lum['preview_img'].save(preview_png)

    tp = post['tower_plan']
    cl = lum.get('cleanup') or {}
    stats.update({
        'n_objects': post['n_objects'],
        'triangles_per_object': post['triangles_per_object'],
        'triangles_total': post['triangles_total'],
        'mean_dE_predicted_palette': float(palette_report['best']['mean_dE']),
        'palette_cost': float(palette_report['best']['cost']),
        'palette_cost_dE': float(palette_report['best']['cost_dE']),
        'dominant_exact_spool_share': float(palette_report['best'].get('dominant_exact_spool_share', 0.0)),
        'prefer_applied': bool((palette_report.get('prefer') or {}).get('applied', False)),
        'purge_per_layer_mm3': tp['purge_per_layer_mm3'],
        'purge_total_mm3': tp['purge_total_mm3'],
        'tool_changes_planned': tp['tool_changes'],
        'tower': tp['tower'],
        'islands_per_optical_layer': (cl.get('after') or {}).get('islands_per_layer'),
        'islands_per_optical_layer_before_cleanup': (cl.get('before') or {}).get('islands_per_layer'),
        'sub_nozzle_share_per_optical_layer': (cl.get('after') or {}).get('thin_share_per_layer'),
        'sub_nozzle_share_per_optical_layer_before_cleanup': (cl.get('before') or {}).get('thin_share_per_layer'),
        'small_island_share_per_optical_layer_before_cleanup': (cl.get('before') or {}).get('small_share_per_layer'),
        'cleanup_reassigned_px': ((cl.get('cleanup') or {}).get('reassigned_px_total')
                                  if cl.get('cleanup') else 0),
        'mesh_dilated': bool(lum.get('dilated', False)),
        'lumina_status': lum['status'],
        'early_stopping': lum['result'].get('early_stopping', {'enabled': False}),
        'backing_layer_h': float(backing_layer_h) if ranges else None,
        'backing_layer_ranges': [list(r) for r in ranges],
        'sliced_layers_est': stats['total_print_layers'] if orientation == 'face-up' else int(N_LAYERS * (2 if structure == 'double' else 1)
                                 + (round((ranges[0][1] - ranges[0][0]) / ranges[0][2]) if ranges
                                    else stats['total_print_layers'] - N_LAYERS * (2 if structure == 'double' else 1))),
    })
    timing['total_s'] = time.perf_counter() - t_all

    ams_lines = []
    for p in post['parts']:
        f = library[p['name']]
        tag = '  (backing)' if p['slot'] == backing_slot else ''
        ams_lines.append(f"AMS slot {p['extruder']} = {f.name} ({f.hex}, TD {f.td_mm:g} mm){tag}")
    if post['unused_slots']:
        ams_lines.append("unused (no pixels): " + ', '.join(names[s] for s in post['unused_slots']))
    ams_txt = os.path.join(job_dir, f"{slug}_ams.txt")
    with open(ams_txt, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(ams_lines) + '\n')

    recipe = {
        'image': os.path.abspath(image_path),
        'title': title,
        'mode_key': mode_key,
        'settings': {'width_mm': float(width_mm), 'quantize_colors': int(quantize_colors),
                     'smooth_sigma': float(smooth_sigma), 'structure': structure,
                     'spacer_mm': float(spacer_mm), 'advisory': advisory, 'seed': int(seed),
                     'first_layer_mm': float(first_layer_mm), 'layer_h': layer_h,
                     'orientation': orientation, 'experimental_layer_height': layer_h < 0.08,
                     'early_stop': stopping_enabled,
                     'optical_layer_thicknesses_mm': thick.tolist(), 'wL': float(wL),
                     'hist_k': int(hist_k), 'hue_weight': float(hue_weight),
                     'td_scale': float(td_scale), 'td_overrides': dict(td_overrides or {}),
                     'must_include': list(must_include or ()), 'prefer': list(prefer or ()),
                     'prefer_tol': float(prefer_tol), 'scoring': scoring_kw, 'metric': metric,
                     'hue_params': resolve_hue_params(hue_params) if metric == 'hue' else None,
                     'pixel_matcher': lum.get('matcher'),
                     'max_spools': int(max_spools), 'min_spools': min_spools,
                     'spool_penalty': float(spool_penalty), 'need_share': float(need_share),
                     'min_region_px': int(min_region_px), 'dilate': bool(dilate),
                     'flush_scale': float(tp['flush_scale']), 'flush_scale_requested': float(flush_scale),
                     'min_flush': tp['flush_min_mm3'],
                     'tower_fit': tower_fit, 'backing_layer_h': backing_layer_h,
                     'model': repr(model) if lut_source is None else f"measured:{lut_source}",
                     'filaments_json': os.path.abspath(filaments_json)},
        'palette': [{'slot': i, 'name': f.name, 'hex': f.hex, 'td_mm': f.td_mm, 'td_source': f.td_source}
                    for i, f in enumerate(fils)],
        'backing': {'slot': int(backing_slot), 'name': fils[backing_slot].name,
                    'extruder': post['backing_extruder']},
        'slot_to_extruder': post['slot_to_extruder'],
        'parts': post['parts'],
        'ams': ams_lines,
        'stats': stats,
        'threemf_check': {k: post[k] for k in ('filament_colour', 'lumina_filament_colour',
                                               'lumina_filament_colour_expected_pure_lut',
                                               'lumina_filament_colour_order_verified',
                                               'project_settings', 'flush_matrix_len', 'flush_vector_len',
                                               'build_transform', 'plaque_xy_mm')},
        'flush': {k: tp[k] for k in ('flush_matrix', 'flush_min_mm3', 'flush_scale', 'flush_max_mm3',
                                     'flush_formula', 'ordering', 'purge_per_layer_mm3', 'purge_total_mm3',
                                     'worst_layer', 'tool_changes')},
        'flush_extruder_order': [f.name for f in [fils[s] for s in post['used_slots']]],
        'tower': tp['tower'],
        'layout': tp['layout'],
        'purge_plan_per_layer': tp['layers'],
        'cleanup': cl,
        'palette_report': palette_report,
        'outputs': {'threemf': threemf, 'preview_png': preview_png, 'lut_npz': lut_npz,
                    'ams_txt': ams_txt, 'stamped_input': stamped_png, 'stop_layers': stop_layers_path},
        'timing': timing,
    }
    recipe_json = os.path.join(job_dir, f"{slug}_recipe.json")
    with open(recipe_json, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable(recipe), fh, indent=2, ensure_ascii=False)

    return {
        'threemf': threemf,
        'preview_png': preview_png,
        'palette': names,
        'backing': fils[backing_slot].name,
        'backing_slot': int(backing_slot),
        'lut_npz': lut_npz,
        'mode_key': mode_key,
        'recipe_json': recipe_json,
        'ams_txt': ams_txt,
        'ams': ams_lines,
        'slot_to_extruder': post['slot_to_extruder'],
        'stats': _jsonable(stats),
        'palette_report': _jsonable(palette_report),
        'flush': _jsonable(recipe['flush']),
        'tower': _jsonable(tp['tower']),
        'layout': _jsonable(tp['layout']),
        'timing': timing,
    }
