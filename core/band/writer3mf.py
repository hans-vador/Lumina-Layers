"""Standalone Bambu Studio 3MF writer for band mode (HueForge-style output).

Emits exactly the reference package (docs/band/PLAN.md, output_spec):
ONE watertight heightfield solid as a single part (object extruder 1) and the
colour schedule as plate-wide tool changes in Metadata/custom_gcode_per_layer.xml
(type=2, MultiAsSingle).  Metadata/project_settings.config is derived from the
user's known-good X2D project files: per-filament arrays are classified by their
length multiplier across three files with 2/3/4 filaments and resized to F.

This module does not touch utils/bambu_3mf_writer.py (it only borrows the
triangle byte streamer).
"""
from __future__ import annotations

import json
import os
import uuid
import zipfile
from datetime import datetime
from typing import Sequence
from xml.sax.saxutils import escape, quoteattr

import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEMPLATE_CACHE = os.path.join(REPO, 'assets', 'x2d_project_settings_template.json')

_ALBUMS = '/Users/hans_vador/Downloads/BambuAlbums'
DEFAULT_TEMPLATE_SOURCES = [
    os.path.join(_ALBUMS, 'Blonde.3mf'),      # 2 filaments
    os.path.join(_ALBUMS, 'Octane.3mf'),      # 3 filaments
    os.path.join(_ALBUMS, 'Love_Sick.3mf'),   # 4 filaments
]

APP_VERSION = 'BambuStudio-02.08.02.61'
N_EXTRUDERS = 2  # X2D

# Keys whose list holds one entry per *preset* (print, filament_1..F, printer).
PRESET_LIST_KEYS = ('different_settings_to_system', 'inherits_group')
# Key whose length is E*F*F / 2*F.
FLUSH_MATRIX_KEY = 'flush_volumes_matrix'
FLUSH_VECTOR_KEY = 'flush_volumes_vector'
# Per-filament keys with an irregular length in one of the three source files
# (values verified by eye: '1','0' pairs per filament => multiplier 2).
KNOWN_MULTIPLIERS = {'filament_dev_ams_drying_ams_limitations': 2}

PRINT_SETTINGS_ID = '0.08mm High Quality @BBL X2D'
PRINTER_SETTINGS_ID = 'Bambu Lab X2D 0.4 nozzle'

# Reference profile facts (Astroworld HueForge reference ported onto X2D keys).
BAND_PRINT_OVERRIDES: dict = {
    'layer_height': '0.08',
    'initial_layer_print_height': '0.16',
    'wall_loops': '2',
    'top_shell_layers': '9999',
    'bottom_shell_layers': '9999',
    'sparse_infill_density': '100%',
    'sparse_infill_pattern': 'zig-zag',
    'top_surface_pattern': 'monotonicline',
    'ironing_type': 'no ironing',
    'enable_prime_tower': '0',
    'single_extruder_multi_material': '1',
    'flush_into_infill': '0',
    'flush_into_objects': '0',
    'flush_into_support': '1',
    'elefant_foot_compensation': '0.1',
    'seam_position': 'aligned',
    'wall_generator': 'classic',
    'print_sequence': 'by layer',
    'timelapse_type': '0',
    'skirt_loops': '1',
    'brim_type': 'auto_brim',
    'detect_thin_wall': '0',
    'print_settings_id': PRINT_SETTINGS_ID,
    'print_compatible_printers': [PRINTER_SETTINGS_ID],
    'printer_settings_id': PRINTER_SETTINGS_ID,
    'printer_model': 'Bambu Lab X2D',
}
DIFFERENT_SETTINGS = ('bottom_shell_layers;elefant_foot_compensation;enable_prime_tower;'
                      'initial_layer_print_height;ironing_type;sparse_infill_density;'
                      'sparse_infill_pattern;top_shell_layers;top_surface_pattern;wall_loops')


# --------------------------------------------------------------------------
# project_settings template: classification + cache
# --------------------------------------------------------------------------
def _read_config_from_3mf(path: str) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read('Metadata/project_settings.config').decode('utf-8'))


def _count_filaments(cfg: dict) -> int:
    return len(cfg['filament_colour'])


def classify_filament_keys(configs: Sequence[dict]) -> dict[str, int]:
    """Return {key: multiplier m} for every key whose list length is m*F in
    every config (F = that config's filament count) — plus KNOWN_MULTIPLIERS.
    Non-list keys and fixed-length lists are not returned."""
    ns = [_count_filaments(c) for c in configs]
    if len(set(ns)) < 2:
        raise ValueError("need template configs with different filament counts")
    keys = set().union(*[set(c) for c in configs])
    out: dict[str, int] = {}
    for k in sorted(keys):
        if k in (FLUSH_MATRIX_KEY, FLUSH_VECTOR_KEY) or k in PRESET_LIST_KEYS:
            continue
        if k in KNOWN_MULTIPLIERS:
            out[k] = KNOWN_MULTIPLIERS[k]
            continue
        vals = [c.get(k) for c in configs]
        if not all(isinstance(v, list) for v in vals):
            continue
        ms = {len(v) / n for v, n in zip(vals, ns)}
        if len(ms) == 1:
            m = ms.pop()
            if m >= 1 and float(m).is_integer():
                out[k] = int(m)
    return out


def _pick_fixed_value(vals: list):
    """For a non-per-filament key with disagreeing values across sources:
    majority length, then the value with the most distinct entries."""
    lists = [v for v in vals if isinstance(v, list)]
    if len(lists) != len(vals):
        return vals[-1]
    lens = [len(v) for v in lists]
    maj = max(set(lens), key=lambda L: (lens.count(L), L))
    cands = [v for v in lists if len(v) == maj]
    return max(cands, key=lambda v: len(set(map(str, v))))


def build_template(sources: Sequence[str] = DEFAULT_TEMPLATE_SOURCES) -> dict:
    """Merge the known-good configs into a filament-count-agnostic template."""
    configs = [_read_config_from_3mf(p) for p in sources]
    ns = [_count_filaments(c) for c in configs]
    mult = classify_filament_keys(configs)
    base: dict = {}
    keys = set().union(*[set(c) for c in configs])
    for k in sorted(keys):
        vals = [c[k] for c in configs if k in c]
        if k in mult:
            m = mult[k]
            block = None
            for v, n in zip([c.get(k) for c in configs], ns):
                if isinstance(v, list) and len(v) == m * n:
                    block = list(v[:m])           # filament-major: first filament's block
                    break
            if block is None:                     # irregular in every file: take first m
                block = list(vals[-1][:m])
            base[k] = block
        elif k in (FLUSH_MATRIX_KEY, FLUSH_VECTOR_KEY, *PRESET_LIST_KEYS):
            base[k] = vals[-1]                    # regenerated at expansion time
        else:
            base[k] = vals[-1] if all(v == vals[-1] for v in vals) else _pick_fixed_value(vals)
    return {
        'schema': 1,
        'sources': [os.path.basename(p) for p in sources],
        'source_filament_counts': ns,
        'n_extruders': N_EXTRUDERS,
        'per_filament_multiplier': mult,
        'preset_list_keys': list(PRESET_LIST_KEYS),
        'base': base,
    }


def load_template(template_sources: Sequence[str] | None = DEFAULT_TEMPLATE_SOURCES,
                  cache_path: str = TEMPLATE_CACHE) -> dict:
    """Return the project_settings template for ``template_sources``.

    The cache (assets/x2d_project_settings_template.json) is keyed on the
    source list it was built from: it is served only when the requested
    sources match it (by basename) - or when no sources are given.  A
    different explicit source list is rebuilt from those files (which must
    exist) and does not overwrite the default cache.
    """
    sources = list(template_sources or [])
    cached = None
    if os.path.isfile(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            tpl = json.load(f)
        if tpl.get('schema') == 1 and 'base' in tpl:
            cached = tpl
    if cached is not None:
        if not sources or [os.path.basename(p) for p in sources] == list(cached.get('sources', [])):
            return cached
    missing = [p for p in sources if not os.path.isfile(p)]
    if not sources or missing:
        raise FileNotFoundError(
            f"template sources missing: {missing or '(none given)'}"
            + (f" and no matching cache at {cache_path}" if cached is None
               else f"; cache {cache_path} was built from {cached.get('sources')}"))
    tpl = build_template(sources)
    is_default = [os.path.basename(p) for p in sources] == [os.path.basename(p) for p in DEFAULT_TEMPLATE_SOURCES]
    if cached is None or is_default:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(tpl, f, indent=1, ensure_ascii=False)
    return tpl


def _fil_attr(fil, name: str, default):
    v = fil.get(name, default) if isinstance(fil, dict) else getattr(fil, name, default)
    return default if v is None else v


def _norm_hex(s: str) -> str:
    return '#' + str(s).strip().lstrip('#').upper()


def build_project_settings(template: dict, filaments: Sequence,
                           print_overrides: dict | None = None) -> dict:
    """Expand the template to F filaments and apply the band profile."""
    F = len(filaments)
    if F < 1:
        raise ValueError("at least one filament is required")
    E = int(template.get('n_extruders', N_EXTRUDERS))
    mult = template['per_filament_multiplier']
    cfg = json.loads(json.dumps(template['base']))  # deep copy

    for k, m in mult.items():
        block = list(cfg.get(k, [''] * m))[:m]
        if len(block) < m:
            block = block + [block[-1] if block else ''] * (m - len(block))
        cfg[k] = block * F

    hexes = [_norm_hex(_fil_attr(f, 'hex', '#808080')) for f in filaments]
    cfg['filament_colour'] = hexes
    cfg['filament_multi_colour'] = list(hexes)
    cfg['default_filament_colour'] = list(hexes)
    cfg['filament_settings_id'] = [str(_fil_attr(f, 'settings_id', 'Bambu PLA Basic @BBL X2D 0.4 nozzle'))
                                   for f in filaments]
    cfg['filament_ids'] = [str(_fil_attr(f, 'filament_id', 'GFA00')) for f in filaments]
    cfg['filament_vendor'] = [str(_fil_attr(f, 'vendor', 'Bambu Lab')) for f in filaments]
    cfg['filament_type'] = [str(_fil_attr(f, 'type', 'PLA')) for f in filaments]
    cfg['filament_self_index'] = [str(i + 1) for i in range(F)]
    cfg['filament_map'] = ['1'] * F

    matrix = []
    for _e in range(E):
        for i in range(F):
            for j in range(F):
                matrix.append('0' if i == j else '350')
    cfg[FLUSH_MATRIX_KEY] = matrix
    cfg[FLUSH_VECTOR_KEY] = ['140'] * (E * F)

    # Preset lists: [print, filament_1..F, printer].  The print preset keeps the
    # parent recorded by Bambu Studio in the source files (inherits_group[0] =
    # '0.08mm Extra Fine @BBL A1' for print_settings_id '0.08mm High Quality
    # @BBL X2D' in every known-good file) - never itself - and lists our
    # overrides in different_settings_to_system.
    for k in template.get('preset_list_keys', PRESET_LIST_KEYS):
        old = cfg.get(k) or ['']
        first = str(old[0]) if old else ''
        cfg[k] = [first] + [''] * F + ['']
    cfg['different_settings_to_system'][0] = DIFFERENT_SETTINGS
    if cfg['inherits_group'][0] == PRINT_SETTINGS_ID:
        cfg['inherits_group'][0] = ''

    overrides = dict(BAND_PRINT_OVERRIDES)
    if print_overrides:
        overrides.update(print_overrides)
    for k, v in overrides.items():
        if isinstance(v, list):
            cfg[k] = [str(x) for x in v]
        elif isinstance(cfg.get(k), list):
            # Per-extruder / per-filament keys are lists in Bambu's format.  A
            # scalar override broadcasts across the template's existing length
            # so callers can say initial_layer_speed=30 without knowing E.
            cfg[k] = [str(v)] * len(cfg[k])
        else:
            cfg[k] = str(v)
    cfg.pop('initial_layer_height', None)  # not a Bambu key
    return cfg


# --------------------------------------------------------------------------
# package pieces
# --------------------------------------------------------------------------
def _write_vertices_bytes(raw, vertices: np.ndarray, fmt: str = '%.3f'):
    verts = np.asarray(vertices, dtype=np.float64)
    if len(verts) == 0:
        return
    x = np.char.mod(fmt, verts[:, 0])
    y = np.char.mod(fmt, verts[:, 1])
    z = np.char.mod(fmt, verts[:, 2])
    chunk = 100_000
    for i in range(0, len(verts), chunk):
        j = min(i + chunk, len(verts))
        lines = '     <vertex x="' + x[i:j] + '" y="' + y[i:j] + '" z="' + z[i:j] + '"/>\n'
        raw.write(''.join(lines.tolist()).encode('ascii'))


def _write_triangles_bytes(raw, faces: np.ndarray):
    try:
        from utils.bambu_3mf_writer import BambuStudio3MFWriter
        BambuStudio3MFWriter._write_triangles_bytes(raw, faces)
        return
    except Exception:
        pass
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return
    v1, v2, v3 = (np.char.mod('%d', f[:, i]) for i in range(3))
    chunk = 100_000
    for i in range(0, len(f), chunk):
        j = min(i + chunk, len(f))
        lines = '     <triangle v1="' + v1[i:j] + '" v2="' + v2[i:j] + '" v3="' + v3[i:j] + '"/>\n'
        raw.write(''.join(lines.tolist()).encode('ascii'))


_MODEL_NS = ('xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
             'xmlns:BambuStudio="http://schemas.bambulab.com/package/2021" '
             'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06" '
             'requiredextensions="p"')


def _content_types(with_png: bool) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
             ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
             ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>',
             ' <Default Extension="png" ContentType="image/png"/>',
             ' <Default Extension="gcode" ContentType="text/x.gcode"/>',
             '</Types>']
    return '\n'.join(lines) + '\n'


def _root_rels(with_thumb: bool) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
             ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>']
    if with_thumb:
        lines += [' <Relationship Target="/Metadata/plate_1.png" Id="rel-2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>',
                  ' <Relationship Target="/Metadata/plate_1.png" Id="rel-4" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-middle"/>',
                  ' <Relationship Target="/Metadata/plate_1_small.png" Id="rel-5" Type="http://schemas.bambulab.com/package/2021/cover-thumbnail-small"/>']
    lines.append('</Relationships>')
    return '\n'.join(lines) + '\n'


def _model_rels() -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
            ' <Relationship Target="/3D/Objects/object_1.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
            '</Relationships>\n')


def _fmt_transform(tx: float, ty: float, tz: float = 0.0) -> str:
    return f"1 0 0 0 1 0 0 0 1 {tx:g} {ty:g} {tz:g}"


def _main_model(title: str, description: str, transform: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" {_MODEL_NS}>
 <metadata name="Application">{APP_VERSION}</metadata>
 <metadata name="BambuStudio:3mfVersion">1</metadata>
 <metadata name="Title">{escape(title)}</metadata>
 <metadata name="Description">{escape(description)}</metadata>
 <metadata name="CreationDate">{datetime.now().strftime('%Y-%m-%d')}</metadata>
 <resources>
  <object id="2" p:UUID="{uuid.uuid4()}" type="model">
   <components>
    <component p:path="/3D/Objects/object_1.model" objectid="1" p:UUID="{uuid.uuid4()}" transform="1 0 0 0 1 0 0 0 1 0 0 0"/>
   </components>
  </object>
 </resources>
 <build p:UUID="{uuid.uuid4()}">
  <item objectid="2" p:UUID="{uuid.uuid4()}" transform="{transform}" printable="1"/>
 </build>
</model>
'''


def _model_settings(title: str, n_filaments: int, transform: str, thumbnail: bool) -> str:
    t = quoteattr(title)
    thumb = ''
    if thumbnail:
        thumb = ('    <metadata key="thumbnail_file" value="Metadata/plate_1.png"/>\n'
                 '    <metadata key="thumbnail_no_light_file" value="Metadata/plate_1.png"/>\n')
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<config>
  <object id="2">
    <metadata key="name" value={t}/>
    <metadata key="extruder" value="1"/>
    <part id="1" subtype="normal_part" uuid="{uuid.uuid4()}">
      <metadata key="name" value="plaque"/>
      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>
      <metadata key="extruder" value="1"/>
    </part>
  </object>
  <plate>
    <metadata key="plater_id" value="1"/>
    <metadata key="plater_name" value={t}/>
    <metadata key="locked" value="false"/>
    <metadata key="filament_map_mode" value="Auto For Flush"/>
    <metadata key="filament_maps" value="{' '.join(['1'] * n_filaments)}"/>
{thumb}    <model_instance>
      <metadata key="object_id" value="2"/>
      <metadata key="instance_id" value="0"/>
      <metadata key="identify_id" value="1"/>
    </model_instance>
  </plate>
  <assemble>
   <assemble_item object_id="2" instance_id="0" transform="{transform}" offset="0 0 0"/>
  </assemble>
</config>
'''


def _custom_gcode(swap_entries: Sequence[tuple[float, int, str]]) -> str:
    lines = ['<?xml version="1.0" encoding="utf-8"?>',
             '<custom_gcodes_per_layer>',
             '<plate>',
             '<plate_info id="1"/>']
    for top_z, ext, hx in swap_entries:
        lines.append(f'<layer top_z="{float(top_z):.8f}" type="2" extruder="{int(ext)}" '
                     f'color="{_norm_hex(hx)}" extra="" gcode="tool_change"/>')
    lines += ['<mode value="MultiAsSingle"/>', '</plate>', '</custom_gcodes_per_layer>']
    return '\n'.join(lines) + '\n'


def _slice_info() -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <header>\n'
            '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
            f'    <header_item key="X-BBL-Client-Version" value="{APP_VERSION.split("-")[-1]}"/>\n'
            '  </header>\n</config>\n')


def _cut_information() -> str:
    return ('<?xml version="1.0" encoding="utf-8"?>\n<objects>\n <object id="1">\n'
            '  <cut_id id="0" check_sum="1" connectors_cnt="0"/>\n </object>\n</objects>')


def validate_swap_entries(swap_entries: Sequence[tuple[float, int, str]], n_filaments: int,
                          first_layer_mm: float, layer_h: float, tol: float = 1e-6) -> list[int]:
    """Assert every swap top_z == FL + lh*(n-1) for an integer n >= 2, entries
    ascending, extruders within 1..F.  Returns the layer numbers n."""
    layers = []
    prev = -1.0
    for top_z, ext, _hx in swap_entries:
        n_f = (float(top_z) - first_layer_mm) / layer_h + 1.0
        n = int(round(n_f))
        if abs(n_f - n) > tol / layer_h or n < 2:
            raise ValueError(f"swap top_z={top_z} is not a layer top for FL={first_layer_mm}, lh={layer_h}")
        if not (1 <= int(ext) <= n_filaments):
            raise ValueError(f"swap extruder {ext} outside 1..{n_filaments}")
        if float(top_z) <= prev:
            raise ValueError("swap entries must be strictly ascending in top_z")
        prev = float(top_z)
        layers.append(n)
    return layers


def _thumbnails(thumbnail_png: str) -> tuple[bytes, bytes]:
    import io
    from PIL import Image
    im = Image.open(thumbnail_png).convert('RGBA')
    out = []
    for size in (512, 128):
        canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        im2 = im.copy()
        im2.thumbnail((size, size), Image.Resampling.LANCZOS)
        canvas.paste(im2, ((size - im2.width) // 2, (size - im2.height) // 2))
        buf = io.BytesIO()
        canvas.save(buf, format='PNG')
        out.append(buf.getvalue())
    return out[0], out[1]


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def write_band_3mf(mesh: trimesh.Trimesh, out_path: str, filaments: Sequence,
                   swap_entries: Sequence[tuple[float, int, str]], title: str,
                   description: str, size_mm: float, bed_mm: float = 256.0,
                   template_sources: Sequence[str] | None = None,
                   print_overrides: dict | None = None,
                   thumbnail_png: str | None = None) -> str:
    """Write the band-mode 3MF package.  ``filaments[i]`` is extruder i+1.

    Returns out_path.
    """
    if template_sources is None:
        template_sources = DEFAULT_TEMPLATE_SOURCES
    F = len(filaments)
    if F < 1:
        raise ValueError("at least one filament required")
    if len(mesh.faces) == 0:
        raise ValueError("refusing to write an empty mesh")

    template = load_template(template_sources)
    cfg = build_project_settings(template, filaments, print_overrides)
    fl = float(cfg['initial_layer_print_height'])
    lh = float(cfg['layer_height'])
    validate_swap_entries(swap_entries, F, fl, lh)

    # centre the object on the bed: mesh min corner is expected at (0, 0, 0)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    ext_x = float(bounds[1, 0] - bounds[0, 0]) if len(mesh.vertices) else float(size_mm)
    ext_y = float(bounds[1, 1] - bounds[0, 1]) if len(mesh.vertices) else float(size_mm)
    if ext_x <= 0 or ext_y <= 0:
        ext_x = ext_y = float(size_mm)
    tx = (float(bed_mm) - ext_x) / 2.0 - float(bounds[0, 0])
    ty = (float(bed_mm) - ext_y) / 2.0 - float(bounds[0, 1])
    transform = _fmt_transform(round(tx, 4), round(ty, 4), 0.0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    thumbs = _thumbnails(thumbnail_png) if thumbnail_png else None

    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', _content_types(thumbs is not None))
        zf.writestr('_rels/.rels', _root_rels(thumbs is not None))
        zf.writestr('3D/3dmodel.model', _main_model(title, description, transform))
        zf.writestr('3D/_rels/3dmodel.model.rels', _model_rels())

        zi = zipfile.ZipInfo('3D/Objects/object_1.model')
        zi.compress_type = zipfile.ZIP_DEFLATED
        with zf.open(zi, 'w', force_zip64=True) as raw:
            raw.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            raw.write(f'<model unit="millimeter" xml:lang="en-US" {_MODEL_NS}>\n'.encode())
            raw.write(b' <metadata name="BambuStudio:3mfVersion">1</metadata>\n <resources>\n')
            raw.write(f'  <object id="1" p:UUID="{uuid.uuid4()}" type="model">\n'.encode())
            raw.write(b'   <mesh>\n    <vertices>\n')
            _write_vertices_bytes(raw, mesh.vertices, '%.3f')
            raw.write(b'    </vertices>\n    <triangles>\n')
            _write_triangles_bytes(raw, mesh.faces)
            raw.write(b'    </triangles>\n   </mesh>\n  </object>\n')
            raw.write(b' </resources>\n <build/>\n</model>\n')

        zf.writestr('Metadata/model_settings.config',
                    _model_settings(title, F, transform, thumbs is not None))
        zf.writestr('Metadata/custom_gcode_per_layer.xml', _custom_gcode(swap_entries))
        zf.writestr('Metadata/project_settings.config',
                    json.dumps(cfg, indent=4, ensure_ascii=False))
        zf.writestr('Metadata/filament_sequence.json',
                    '{"plate_1":{"nozzle_sequence":[],"optimal_assignment":[],"sequence":[]}}')
        zf.writestr('Metadata/slice_info.config', _slice_info())
        zf.writestr('Metadata/cut_information.xml', _cut_information())
        if thumbs is not None:
            zf.writestr('Metadata/plate_1.png', thumbs[0])
            zf.writestr('Metadata/plate_1_small.png', thumbs[1])
    return out_path
