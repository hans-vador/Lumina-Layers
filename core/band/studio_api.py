"""HTTP backend for Band Studio's drag-and-drop editor.

The editor is a vertical layer bar: you drag filament swatches onto it, the app
divides the layers between them, and the preview re-renders.  That interaction
needs real drag-and-drop and a layout that mirrors the print stack, which is why
this is a small hand-written front end over :class:`core.band.session.BandSession`
rather than a widget toolkit.

Everything expensive (image load, bilateral filter, L*) is cached in the session;
every request here only touches the cheap half, so a drag re-renders in ~90 ms.
"""
from __future__ import annotations

import base64
import copy
import json
import io
import os
import tempfile
from typing import List, Optional, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict, model_validator

from core.band.optics import (Filament, hex_to_rgb01, load_filament_library,
                              save_filament_library, srgb_to_linear)
from core.band.session import LED_PRESETS, BandSession
from core.band.tone import ToneCurve

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC = os.path.join(REPO, 'ui', 'studio')
DEFAULT_LIBRARY = os.path.join(REPO, 'assets', 'filaments_user_measured.json')
DEFAULT_OUT = os.path.join(REPO, 'output', 'band_studio')


class Studio:
    """Whole-app state: one library, one open image, one colour plan."""

    def __init__(self, library_path: str = DEFAULT_LIBRARY):
        self.library_path = library_path
        self.library = load_filament_library(library_path)
        self.session: Optional[BandSession] = None

    # ----- library ------------------------------------------------------- #
    def library_json(self) -> list:
        out = []
        for f in self.library.values():
            lum = float(srgb_to_linear(hex_to_rgb01(f.hex)).mean())
            out.append({'name': f.name, 'hex': f.hex, 'td_mm': round(float(f.td_mm), 3),
                        'source': f.td_source, 'lightness': round(lum, 4)})
        return out

    def set_td(self, name: str, td_mm: float) -> None:
        if name not in self.library:
            raise KeyError(name)
        if not td_mm > 0:
            raise ValueError("TD must be > 0")
        f = self.library[name]
        self.library[name] = Filament.from_hex(f.name, f.hex, float(td_mm), 'edited',
                                               f.settings_id, f.filament_id)
        if self.session is not None:
            self.session.library = self.library

    def add_filament(self, name: str, hex_str: str, td_mm: float) -> None:
        name = (name or '').strip()
        if not name:
            raise ValueError("name is required")
        if name in self.library:
            raise ValueError(f"{name} already exists")
        self.library[name] = Filament.from_hex(name, hex_str, float(td_mm), 'user')
        if self.session is not None:
            self.session.library = self.library

    # ----- image --------------------------------------------------------- #
    def open_image(self, path: str, width_mm: float = 150.0) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        names = self._default_stack()
        self.session = BandSession.load(path, self.library, names,
                                        [9] + [4] * (len(names) - 1), width_mm=width_mm)

    def _default_stack(self) -> List[str]:
        """Darkest first, then spread across lightness - HueForge's ordering rule."""
        fils = sorted(self.library.values(),
                      key=lambda f: float(srgb_to_linear(hex_to_rgb01(f.hex)).mean()))
        if not fils:
            raise RuntimeError("empty filament library")
        picks = [fils[0]]
        for frac in (0.4, 0.7, 1.0):
            i = min(len(fils) - 1, int(round(frac * (len(fils) - 1))))
            if fils[i].name not in [p.name for p in picks]:
                picks.append(fils[i])
        return [p.name for p in picks]

    # ----- render -------------------------------------------------------- #
    def require(self) -> BandSession:
        if self.session is None:
            raise HTTPException(400, "no image open")
        return self.session


STUDIO = Studio()


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #
class OpenReq(BaseModel):
    path: str
    width_mm: float = 150.0


class TdReq(BaseModel):
    name: str
    td_mm: float


class NewFilamentReq(BaseModel):
    name: str
    hex: str
    td_mm: float = 1.0


class MeshStop(BaseModel):
    hex: str = Field(pattern=r'^#[0-9a-fA-F]{6}$')
    layer: int = Field(ge=1, le=1000)
    td_mm: float = Field(default=1, gt=0, le=100)


class StateReq(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    names: List[str] = Field(min_length=1, max_length=32)
    counts: List[int] = Field(min_length=1, max_length=32)
    width_mm: float = Field(default=150, gt=0, le=500)
    q_lo: float = -.13
    q_hi: float = 1.06
    gamma: float = Field(default=1.55, gt=0, le=10)
    first_layer_mm: float = Field(default=.16, gt=0, le=1)
    layer_h: float = Field(default=.08, gt=0, le=1)
    negative: bool = False
    full_range: bool = False
    engrave: bool = True
    k_opaque: float = Field(default=7, gt=0, le=20)
    lighting: Literal['front', 'back'] = 'front'
    led: str = 'Natural White 4000K'
    intensity: float = Field(default=2.5, gt=0, le=100)
    border_width_mm: float = Field(default=0, ge=0)
    border_depth_mm: float = Field(default=0, ge=0)
    smooth_sigma: float = Field(default=10, ge=0, le=50)
    nozzle_temp_c: float = Field(default=205, ge=150, le=300)
    first_layer_speed: float = Field(default=50, ge=5, le=200)
    mesh_mode: Literal['luminance', 'combo', 'max_channel', 'color_match', 'color_aware', 'color_pop'] = 'luminance'
    mesh_core: List[MeshStop] = Field(default_factory=list, max_length=32)
    match_metric: Literal['rgb', 'cielab', 'hsl', 'dot'] = 'cielab'
    combo: float = Field(default=1, ge=0, le=1)
    tolerance: float = Field(default=8, ge=0, le=255)
    region_split: float = Field(default=.5, gt=0, lt=1)
    reverse_regions: bool = False
    channel_order: Literal['RGB', 'RBG', 'GRB', 'GBR', 'BRG', 'BGR'] = 'BGR'
    min_depth_mm: Optional[float] = Field(default=None, gt=0)
    max_depth_mm: Optional[float] = Field(default=None, gt=0)
    spike_removal: Literal['off', 'fast', 'moderate', 'aggressive'] = 'off'

    @model_validator(mode='after')
    def validate_stack(self):
        if len(self.names) != len(self.counts) or any(c < 1 for c in self.counts):
            raise ValueError('Each filament needs a positive layer count')
        if sum(self.counts) > 1000:
            raise ValueError('The print stack must fit within 1000 layers')
        if self.q_hi <= self.q_lo:
            raise ValueError('White point must be above black point')
        if self.led not in LED_PRESETS:
            raise ValueError('Unknown LED preset')
        if self.border_width_mm * 2 >= self.width_mm:
            raise ValueError('Border must be smaller than half the model width')
        return self


class MeshReq(StateReq):
    detail: int = 260          # grid samples along the longest edge


class ExportReq(StateReq):
    kind: str = '3mf'
    out_dir: str = DEFAULT_OUT


def _png_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def _apply(req: StateReq, commit: bool = True) -> BandSession:
    s = copy.copy(STUDIO.require())
    names, counts = list(req.names), [max(1, int(c)) for c in req.counts]
    if not names:
        raise HTTPException(400, "drag at least one filament onto the bar")
    if len(counts) != len(names):
        counts = (counts + [4] * len(names))[:len(names)]
    unknown = [n for n in names if n not in STUDIO.library]
    if unknown:
        raise HTTPException(400, f"unknown filament(s): {unknown}")

    if abs(s.width_mm - req.width_mm) > 1e-9 or abs(s.smooth_sigma - req.smooth_sigma) > 1e-9:
        s.width_mm, s.smooth_sigma = float(req.width_mm), float(req.smooth_sigma)
        s.reload_image()

    s.library = dict(STUDIO.library)
    s.filament_names, s.layer_counts = names, counts
    s.tone = ToneCurve(float(req.q_lo), float(req.q_hi), float(req.gamma))
    s.first_layer_mm, s.layer_h = float(req.first_layer_mm), float(req.layer_h)
    s.base_min_layers = min(5, counts[0])
    s.negative, s.full_range, s.engrave = req.negative, req.full_range, req.engrave
    s.k_opaque = float(req.k_opaque)
    s.lighting = 'back' if str(req.lighting).startswith('back') else 'front'
    s.led_hex = LED_PRESETS.get(req.led, s.led_hex)
    s.light_intensity = float(req.intensity)
    s.border_width_mm, s.border_depth_mm = float(req.border_width_mm), float(req.border_depth_mm)
    s.nozzle_temp_c = float(req.nozzle_temp_c)
    s.first_layer_speed = float(req.first_layer_speed)
    for key in ('mesh_mode', 'match_metric', 'combo', 'tolerance', 'region_split',
                'reverse_regions', 'channel_order', 'min_depth_mm', 'max_depth_mm', 'spike_removal'):
        setattr(s, key, getattr(req, key))
    s.mesh_core = [stop.model_dump() for stop in req.mesh_core]
    s.height_map()  # reject invalid geometry before replacing the current project
    if commit:
        STUDIO.session = s
    return s


def _state_payload(s: BandSession) -> dict:
    sched = s.schedule
    shares = s.surface_shares()
    bands = []
    for b, (name, n) in enumerate(zip(s.filament_names, s.layer_counts)):
        f = s.library[name]
        bands.append({
            'name': name, 'hex': f.hex, 'td_mm': round(float(f.td_mm), 3), 'layers': int(n),
            'swap_z': None if b == 0 else round(float(sched.top_z(sched.band_first_layer(b))), 3),
            'surface': round(float(shares.get(name, 0.0)), 4),
        })
    h = s.height_map(full=False)
    m = s._mask_small
    return {
        'preview': _png_data_url(s.preview()),
        'mesh_core': s.mesh_core,
        'settings': __import__('core.band.project', fromlist=['settings']).settings(s),
        'layers': [{'layer': n, 'z': round(sched.top_z(n), 4),
                    'hex': s.filaments[int(sched.band_of_layer(n))].hex,
                    'name': s.filament_names[int(sched.band_of_layer(n))]}
                   for n in range(1, sched.n_layers+1)],
        'describe': s.describe(),
        'bands': bands,
        'stats': {
            'layers': int(sched.n_layers),
            'height_mm': round(float(sched.total_height_mm), 3),
            'swaps': int(sched.n_bands - 1),
            'relief_mm': [round(float(h[m].min()), 3), round(float(h[m].max()), 3)]
            if m.any() else [0, 0],
        },
    }


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    app = FastAPI(title="Band Studio")
    app.mount('/vendor', StaticFiles(directory=os.path.join(STATIC, 'vendor')), name='vendor')

    @app.get('/')
    def index():
        return FileResponse(os.path.join(STATIC, 'index.html'))

    @app.get('/api/boot')
    def api_boot():
        """Everything the page needs on first paint."""
        if STUDIO.session is None:
            return {'opened': False, 'out_dir': DEFAULT_OUT}
        s = STUDIO.session
        return {'opened': True, 'out_dir': DEFAULT_OUT,
                'image': os.path.basename(s.image_path),
                'names': s.filament_names, 'counts': s.layer_counts,
                **_state_payload(s)}

    @app.get('/api/source')
    def api_source():
        from PIL import Image, ImageOps
        with Image.open(STUDIO.require().image_path) as image:
            image = ImageOps.exif_transpose(image).convert('RGBA')
            image.thumbnail((1200, 1200))
            buf = io.BytesIO(); image.save(buf, format='PNG')
        return Response(buf.getvalue(), media_type='image/png',
                        headers={'Cache-Control': 'no-store'})

    @app.post('/api/project/save')
    def api_save_project(req: StateReq):
        from core.band.project import dump_project
        s = _apply(req, commit=False)
        return Response(json.dumps(dump_project(s)), media_type='application/json',
                        headers={'Content-Disposition': 'attachment; filename="painting.lumina.json"'})

    @app.post('/api/project/open')
    async def api_open_project(file: UploadFile = File(...)):
        from core.band.project import load_project
        try:
            doc = json.loads(await file.read())
            s = load_project(doc)
        except Exception as exc:
            raise HTTPException(400, f'Cannot open project: {exc}')
        STUDIO.session, STUDIO.library = s, s.library
        return {'image': os.path.basename(s.image_path), 'names': s.filament_names,
                'counts': s.layer_counts, 'filaments': STUDIO.library_json(), **_state_payload(s)}

    @app.get('/api/library')
    def api_library():
        return {'filaments': STUDIO.library_json(), 'path': STUDIO.library_path,
                'leds': list(LED_PRESETS)}

    @app.post('/api/library/td')
    def api_td(req: TdReq):
        try:
            STUDIO.set_td(req.name, req.td_mm)
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        return {'filaments': STUDIO.library_json()}

    @app.post('/api/library/new')
    def api_new(req: NewFilamentReq):
        try:
            STUDIO.add_filament(req.name, req.hex, req.td_mm)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {'filaments': STUDIO.library_json()}

    @app.post('/api/library/save')
    def api_save_library():
        p = save_filament_library(STUDIO.library_path, STUDIO.library)
        return {'saved': p, 'n': len(STUDIO.library)}

    @app.post('/api/open')
    def api_open(req: OpenReq):
        try:
            STUDIO.open_image(req.path, req.width_mm)
        except FileNotFoundError:
            raise HTTPException(404, f"no such file: {req.path}")
        s = STUDIO.require()
        return {'names': s.filament_names, 'counts': s.layer_counts,
                'image': os.path.basename(req.path), **_state_payload(s)}

    @app.post('/api/upload')
    async def api_upload(file: UploadFile = File(...), width_mm: float = 150.0):
        dst = os.path.join(tempfile.mkdtemp(prefix='band_up_'), os.path.basename(file.filename or 'image.png'))
        with open(dst, 'wb') as fh:
            fh.write(await file.read())
        STUDIO.open_image(dst, width_mm)
        s = STUDIO.require()
        return {'names': s.filament_names, 'counts': s.layer_counts,
                'image': os.path.basename(dst), **_state_payload(s)}

    @app.post('/api/state')
    def api_state(req: StateReq):
        return _state_payload(_apply(req))

    @app.post('/api/auto')
    def api_auto(req: StateReq):
        """Divide the layers between the dragged colours to maximise detail."""
        s = _apply(req)
        try:
            s.layer_counts = s.auto_thickness()
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        return {'counts': s.layer_counts, **_state_payload(s)}

    @app.post('/api/mesh')
    def api_mesh(req: MeshReq):
        """Geometry for the 3D viewport as one packed binary blob.

        Framing: uint32 header length, then that many bytes of JSON, then
        Float32 positions, Uint32 indices, Uint8 colours - laid out so each
        typed array starts 4-byte aligned and the client can wrap the buffer
        directly instead of parsing millions of JSON numbers.
        """
        import json as _json
        import struct
        s = _apply(req, commit=False)
        m = s.mesh3d(max_dim=max(60, min(600, int(req.detail))))
        pos = np.ascontiguousarray(m['positions'], dtype=np.float32)
        idx = np.ascontiguousarray(m['indices'], dtype=np.uint32)
        col = np.ascontiguousarray(m['colors'], dtype=np.uint8)
        head = _json.dumps({'nv': int(pos.shape[0]), 'nf': int(idx.shape[0]),
                            'size_mm': m['size_mm'], 'pitch_mm': m['pitch_mm'],
                            'ramp': m['ramp'], 'ramp_height_mm': m['ramp_height_mm'],
                            'backlit': m['backlit']}).encode()
        pad = (-len(head) - 4) % 4
        head = head + b' ' * pad
        blob = (struct.pack('<I', len(head)) + head
                + pos.tobytes() + idx.tobytes() + col.tobytes())
        return Response(content=blob, media_type='application/octet-stream')

    @app.post('/api/export')
    def api_export(req: ExportReq):
        s = _apply(req, commit=False)
        try:
            if req.kind == 'stl':
                return s.export_stl(req.out_dir)
            return {k: v for k, v in s.export(req.out_dir).items()
                    if isinstance(v, (str, int, float))}
        except Exception as exc:
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    @app.exception_handler(ValueError)
    def _value_err(request, exc):
        return JSONResponse({'error': str(exc)}, status_code=400)

    @app.exception_handler(HTTPException)
    def _http_err(request, exc):
        return JSONResponse({'error': exc.detail}, status_code=exc.status_code)

    return app
