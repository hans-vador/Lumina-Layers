"""Portable Lumina projects: source image, both cores and all print settings."""
from __future__ import annotations

import base64
import io
import os
import tempfile
from PIL import Image, ImageOps

from core.band.session import BandSession, LED_PRESETS
from core.band.optics import Filament
from core.band.tone import ToneCurve

SETTINGS = ('width_mm', 'first_layer_mm', 'layer_h', 'negative', 'full_range',
            'engrave', 'k_opaque', 'lighting', 'border_width_mm', 'border_depth_mm',
            'smooth_sigma', 'mesh_mode', 'mesh_core', 'match_metric', 'combo',
            'tolerance', 'region_split', 'reverse_regions', 'channel_order',
            'min_depth_mm', 'max_depth_mm', 'spike_removal')


def settings(s: BandSession) -> dict:
    return {k: getattr(s, k) for k in SETTINGS} | {
        'q_lo': s.tone.q_lo, 'q_hi': s.tone.q_hi, 'gamma': s.tone.gamma,
        'intensity': s.light_intensity,
        'led': next((k for k, v in LED_PRESETS.items() if v == s.led_hex),
                    'Natural White 4000K')}


def dump_project(s: BandSession) -> dict:
    with Image.open(s.image_path) as original:
        image = ImageOps.exif_transpose(original).convert('RGBA')
        buf = io.BytesIO(); image.save(buf, format='PNG')
    return {'format': 'lumina-filament-project', 'version': 1,
            'image': {'name': os.path.basename(s.image_path),
                      'png_base64': base64.b64encode(buf.getvalue()).decode('ascii')},
            'settings': settings(s), 'names': list(s.filament_names),
            'counts': list(s.layer_counts),
            'filaments': [f.to_dict() for f in s.library.values()]}


def load_project(doc: dict) -> BandSession:
    if doc.get('format') != 'lumina-filament-project' or doc.get('version') != 1:
        raise ValueError('Not a supported Lumina project (version 1 required)')
    if len(doc['image']['png_base64']) > 48_000_000:
        raise ValueError('Project image is too large')
    raw = base64.b64decode(doc['image']['png_base64'], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        dest = os.path.join(tempfile.mkdtemp(prefix='lumina_project_'),
                            os.path.basename(doc['image']['name']) or 'image.png')
        image.save(dest, format='PNG')
    library = {}
    for entry in doc['filaments']:
        f = Filament.from_hex(entry['name'], entry['hex'], entry['td_mm'],
                              entry.get('td_source', 'project'),
                              entry.get('settings_id', ''), entry.get('filament_id', ''))
        library[f.name] = f
    # Reuse the HTTP schema so files and interactive controls accept the same
    # values. Validate before publishing the new session to the application.
    from core.band.studio_api import StateReq
    req = StateReq(**(doc['settings'] | {'names': doc['names'], 'counts': doc['counts']}))
    values = req.model_dump()
    if len(req.names) != len(req.counts) or any(n not in library for n in req.names):
        raise ValueError('Project filament stack is invalid')
    s = BandSession.load(dest, library, req.names, req.counts,
                         width_mm=req.width_mm, smooth_sigma=req.smooth_sigma)
    for key in SETTINGS:
        setattr(s, key, values[key])
    s.base_min_layers = min(5, req.counts[0])
    s.tone = ToneCurve(req.q_lo, req.q_hi, req.gamma)
    s.light_intensity = req.intensity; s.led_hex = LED_PRESETS[req.led]
    s.height_map()
    return s
