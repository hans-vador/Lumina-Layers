"""Art-directed relief: edit scripts, the interactive edit session and the hook
that applies a script inside the Relief Stack5 pipeline.

    Depth Anything V2 (full image) -> normal relief processing -> steps (layers)
        -> ORIGINAL height map in mm  (steps * 0.08)
        -> user edits on masks (core.relief.heightfield)  -> edited mm map
        -> quantize_mm -> steps -> voxels / mesh / 3MF as before

An :class:`EditScript` is a JSON document (``relief-edits/1``) holding the
operations with their masks (PNG, base64) plus the grid they were authored on.
Masks live on the relief grid, which is the original image resampled NEAREST to
Lumina's working resolution (10 px/mm) - the same grid the Stack5 colours are
matched on, so a mask pixel is a colour pixel is a relief pixel.  A script
replays on a different grid (other plaque width) by resampling its masks.

Nothing here touches colour: :func:`apply_edit_script` only rewrites the relief
steps of a :class:`~core.relief.depth_processing.ReliefField`.  With no script
the pipeline never calls it, so unedited Relief output is byte-identical.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from core.relief.depth_processing import DepthParams, ReliefField, morphological_clean, process_depth, relief_stats
from core.relief.heightfield import OPS, apply_op, quantize_mm
from core.relief.relief_stack import ReliefDims

FORMAT = 'relief-edits/1'
DEFAULT_CLEAN_PX = 4          # same printability rule as DepthParams.min_feature_px (0.4 mm at 0.1 mm/px)
ALPHA_BACKGROUND = 10         # Lumina: alpha < 10 = transparent (core/image_processing.py)


# --------------------------------------------------------------------------- mask / array codecs
def encode_mask_png(mask: np.ndarray) -> str:
    m = np.asarray(mask).astype(bool)
    buf = io.BytesIO()
    Image.fromarray(m).convert('1').save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def decode_mask_png(data: str) -> np.ndarray:
    im = Image.open(io.BytesIO(base64.b64decode(data)))
    return np.asarray(im.convert('L')) > 127


def encode_profile_png(profile: np.ndarray) -> str:
    p = np.clip(np.asarray(profile, dtype=np.float64), 0.0, 1.0)
    buf = io.BytesIO()
    Image.fromarray(np.round(p * 65535).astype(np.uint16)).save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def decode_profile_png(data: str) -> np.ndarray:
    im = Image.open(io.BytesIO(base64.b64decode(data)))
    return (np.asarray(im).astype(np.float32) / 65535.0).astype(np.float32)


def resample_mask(mask: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    """NEAREST resample with PIL - the same resampler Lumina's working copy uses,
    so a mask made on the original image lands on the same pixels as the colours."""
    m = np.asarray(mask).astype(bool)
    H, W = int(target_hw[0]), int(target_hw[1])
    if m.shape == (H, W):
        return m.copy()
    im = Image.fromarray(m.astype(np.uint8) * 255, 'L').resize((W, H), Image.Resampling.NEAREST)
    return np.asarray(im) > 127


def resample_profile(profile: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float32)
    H, W = int(target_hw[0]), int(target_hw[1])
    if p.shape == (H, W):
        return p.copy()
    im = Image.fromarray(p, 'F').resize((W, H), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def grid_to_image_xy(x: float, y: float, grid_hw: Sequence[int], image_hw: Sequence[int]) -> tuple[float, float]:
    """Relief-grid pixel (x, y) -> original image pixel (centre-aligned)."""
    return ((x + 0.5) * image_hw[1] / grid_hw[1], (y + 0.5) * image_hw[0] / grid_hw[0])


def image_to_grid_xy(x: float, y: float, grid_hw: Sequence[int], image_hw: Sequence[int]) -> tuple[float, float]:
    return (x * grid_hw[1] / image_hw[1] - 0.5, y * grid_hw[0] / image_hw[0] - 0.5)


# --------------------------------------------------------------------------- edit script
def _op_summary(op: dict) -> str:
    kind = op.get('op')
    n = int(np.count_nonzero(op['mask'])) if op.get('mask') is not None else 0
    if kind == 'dome':
        s = f"dome to {float(op.get('height_mm', 0)):g} mm, roundness {float(op.get('roundness', 1)):g}"
    elif kind == 'extrude':
        s = f"extrude to {float(op.get('height_mm', 0)):g} mm"
    elif kind == 'offset':
        s = f"raise/lower {float(op.get('delta_mm', 0)):+g} mm"
    elif kind == 'feather':
        s = f"feather {float(op.get('width_mm', op.get('feather_mm', 0)) or 0):g} mm"
    elif kind == 'reset':
        s = "reset to the Depth Anything relief"
    else:
        s = f"object profile to {float(op.get('height_mm', 0)):g} mm"
    if kind in ('dome', 'extrude', 'offset', 'object') and float(op.get('feather_mm', 0) or 0) > 0:
        s += f", edge feather {float(op['feather_mm']):g} mm"
    return f"{s} on {n} px" + (f" ({op['label']})" if op.get('label') else '')


class EditScript:
    """Ordered edit operations + the grid / depth settings they were authored on."""

    META_KEYS = ('image', 'width_mm', 'grid_hw', 'pixel_mm', 'relief_mm', 'base_mm', 'depth', 'clean_px', 'created',
                 'note')

    def __init__(self, ops: Optional[Sequence[dict]] = None, **meta):
        self.ops: list[dict] = []
        for op in ops or ():
            self.add(op)
        self.meta: dict = {k: v for k, v in meta.items() if v is not None}
        self.meta.setdefault('clean_px', DEFAULT_CLEAN_PX)

    # -- construction -------------------------------------------------------
    def add(self, op: dict) -> None:
        kind = str(op.get('op', '')).lower()
        if kind not in OPS:
            raise ValueError(f"unknown edit op {kind!r}; choose from {OPS}")
        d = dict(op)
        d['op'] = kind
        if d.get('mask') is not None:
            d['mask'] = np.asarray(d['mask']).astype(bool)
        if d.get('profile') is not None:
            d['profile'] = np.asarray(d['profile'], dtype=np.float32)
        self.ops.append(d)

    @property
    def grid_hw(self) -> Optional[tuple[int, int]]:
        g = self.meta.get('grid_hw')
        if g is None:
            for op in self.ops:
                if op.get('mask') is not None:
                    return tuple(int(v) for v in op['mask'].shape)
            return None
        return int(g[0]), int(g[1])

    def __len__(self) -> int:
        return len(self.ops)

    def summary(self) -> list[str]:
        return [_op_summary(op) for op in self.ops]

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        ops = []
        for op in self.ops:
            d = {k: v for k, v in op.items() if k not in ('mask', 'profile')}
            for k, v in list(d.items()):
                if isinstance(v, np.generic):
                    d[k] = v.item()
            if op.get('mask') is not None:
                d['mask_png'] = encode_mask_png(op['mask'])
                d['mask_hw'] = [int(op['mask'].shape[0]), int(op['mask'].shape[1])]
            if op.get('profile') is not None:
                d['profile_png'] = encode_profile_png(op['profile'])
            ops.append(d)
        meta = dict(self.meta)
        if self.grid_hw is not None:
            meta['grid_hw'] = [int(self.grid_hw[0]), int(self.grid_hw[1])]
        meta.setdefault('created', time.strftime('%Y-%m-%dT%H:%M:%S'))
        return {'format': FORMAT, **meta, 'summary': self.summary(), 'ops': ops}

    @classmethod
    def from_dict(cls, d: dict) -> 'EditScript':
        fmt = d.get('format')
        if fmt != FORMAT:
            raise ValueError(f"not a relief edit script (format {fmt!r}, expected {FORMAT!r})")
        ops = []
        for od in d.get('ops', []):
            op = {k: v for k, v in od.items() if k not in ('mask_png', 'mask_hw', 'profile_png')}
            if od.get('mask_png'):
                op['mask'] = decode_mask_png(od['mask_png'])
            if od.get('profile_png'):
                op['profile'] = decode_profile_png(od['profile_png'])
            ops.append(op)
        meta = {k: d[k] for k in cls.META_KEYS if k in d}
        return cls(ops, **meta)

    def save(self, path: str) -> str:
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return path

    @classmethod
    def load(cls, path: str) -> 'EditScript':
        with open(path, encoding='utf-8') as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def coerce(cls, obj) -> 'EditScript':
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)
        if isinstance(obj, str):
            if not os.path.isfile(obj):
                raise FileNotFoundError(f"edit script not found: {obj}")
            return cls.load(obj)
        raise TypeError(f"height_edits must be an EditScript, a dict or a JSON path, got {type(obj).__name__}")

    def ops_resampled(self, target_hw: Sequence[int]) -> list[dict]:
        """Ops with masks / profiles on ``target_hw`` (NEAREST / bilinear)."""
        out = []
        for op in self.ops:
            d = dict(op)
            if d.get('mask') is not None:
                d['mask'] = resample_mask(d['mask'], target_hw)
            if d.get('profile') is not None:
                d['profile'] = resample_profile(d['profile'], target_hw)
            out.append(d)
        return out


# --------------------------------------------------------------------------- session (GUI / programmatic)
def replay_ops(original: np.ndarray, ops: Sequence[dict], pixel_mm: float, relief_max: float) -> tuple[np.ndarray, list[dict]]:
    h = np.asarray(original, dtype=np.float32).copy()
    infos = []
    for op in ops:
        h, info = apply_op(h, original, op, pixel_mm, relief_max)
        infos.append(info)
    return h, infos


class EditSession:
    """Original mm map + an ordered list of applied ops, with undo / reset.
    Undo replays the remaining ops from the original (an op costs ~0.1 s on a
    1500 x 1500 grid), so no per-step copies are kept."""

    def __init__(self, original_mm: np.ndarray, pixel_mm: float, relief_max_mm: float):
        self.original = np.ascontiguousarray(np.asarray(original_mm, dtype=np.float32))
        if self.original.ndim != 2:
            raise ValueError("original_mm must be (H, W)")
        self.pixel_mm = float(pixel_mm)
        self.relief_max = float(relief_max_mm)
        self.current = self.original.copy()
        self.ops: list[dict] = []
        self.infos: list[dict] = []

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.original.shape[0]), int(self.original.shape[1])

    @property
    def can_undo(self) -> bool:
        return bool(self.ops)

    def apply(self, op: dict) -> dict:
        d = dict(op)
        d['op'] = str(d.get('op', '')).lower()
        if d['op'] not in OPS:
            raise ValueError(f"unknown edit op {d['op']!r}; choose from {OPS}")
        new, info = apply_op(self.current, self.original, d, self.pixel_mm, self.relief_max)
        if info.get('noop'):
            return info
        self.current = new
        self.ops.append(d)
        self.infos.append(info)
        return info

    def undo(self) -> bool:
        if not self.ops:
            return False
        self.ops.pop()
        self.infos.pop()
        self.current, self.infos = replay_ops(self.original, self.ops, self.pixel_mm, self.relief_max)
        return True

    def reset(self) -> None:
        self.ops, self.infos = [], []
        self.current = self.original.copy()

    def changed(self) -> np.ndarray:
        return self.current != self.original

    def load_script(self, script: EditScript) -> list[dict]:
        self.reset()
        for op in script.ops_resampled(self.shape):
            self.apply(op)
        return list(self.infos)

    def to_script(self, **meta) -> EditScript:
        meta.setdefault('grid_hw', list(self.shape))
        meta.setdefault('pixel_mm', self.pixel_mm)
        meta.setdefault('relief_mm', self.relief_max)
        return EditScript([dict(op) for op in self.ops], **meta)


# --------------------------------------------------------------------------- pipeline hook
def original_mm_from_field(field: ReliefField) -> np.ndarray:
    return (field.steps.astype(np.float64) * float(field.dims.layer_h)).astype(np.float32)


def apply_edit_script(field: ReliefField, script, pixel_mm: float, plaque_mask: Optional[np.ndarray] = None,
                      clean_px: Optional[int] = None) -> tuple[ReliefField, dict]:
    """Replay an edit script on the pipeline's relief field.

    The field's steps (whole layers) become the ORIGINAL mm map; the edited map is
    quantised back (round half up) and clamped to the relief budget.  Pixels the
    edits did not change keep their exact steps.  Where steps changed, the
    pipeline's min-feature rule (grey closing + opening, ``clean_px``, default
    the script's ``clean_px`` = 4) removes slivers thinner than the nozzle,
    restricted to the changed region (dilated by the kernel).  The plaque
    cut-out stays 0.  Returns (new field, report)."""
    dims: ReliefDims = field.dims
    layer_h = float(dims.layer_h)
    relief_layers = int(dims.relief_layers)
    relief_max = relief_layers * layer_h
    H, W = field.steps.shape
    sc = EditScript.coerce(script)
    warnings: list[str] = []
    if sc.grid_hw is not None and tuple(sc.grid_hw) != (H, W):
        warnings.append(f"edits were authored on a {sc.grid_hw[1]}x{sc.grid_hw[0]} grid; masks are resampled to "
                        f"the {W}x{H} relief grid")
    rm = sc.meta.get('relief_mm')
    if rm is not None and abs(float(rm) - relief_max) > 1e-6:
        warnings.append(f"edits were authored for a {float(rm):g} mm relief budget; heights are clamped to "
                        f"{relief_max:g} mm")
    ops = sc.ops_resampled((H, W))
    plaque = None if plaque_mask is None else np.asarray(plaque_mask).astype(bool)
    if plaque is not None and plaque.shape != (H, W):
        raise ValueError("plaque_mask must match the relief grid")
    original = original_mm_from_field(field)
    edited, infos = replay_ops(original, ops, float(pixel_mm), relief_max)
    steps = quantize_mm(edited, layer_h, relief_layers)
    if plaque is not None:
        steps[~plaque] = 0
    changed = steps != field.steps
    k = int(sc.meta.get('clean_px', DEFAULT_CLEAN_PX) if clean_px is None else clean_px)
    cleaned_px = 0
    if k > 1 and changed.any():
        import cv2
        region = cv2.dilate(changed.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        if plaque is not None:
            region &= plaque
        cleaned = morphological_clean(steps, k, mask=None if region.all() else region)
        if plaque is not None:
            cleaned[~plaque] = 0
        cleaned_px = int(np.count_nonzero(cleaned != steps))
        steps = cleaned
    steps = np.clip(steps, 0, relief_layers).astype(np.int16)
    final_changed = steps != field.steps
    report = {
        'format': FORMAT, 'n_ops': len(ops), 'ops': infos, 'summary': sc.summary(),
        'pixels_changed': int(final_changed.sum()),
        'share_changed': float(final_changed.mean()) if final_changed.size else 0.0,
        'cleanup_px': k, 'cleanup_pixels_changed': cleaned_px,
        'relief_mm_before': [round(float(original.min()), 4), round(float(original.max()), 4)],
        'relief_mm_after': [round(float(steps.min()) * layer_h, 4), round(float(steps.max()) * layer_h, 4)],
        'clamped_to_mm': relief_max, 'warnings': warnings,
        'colour_untouched': True,
    }
    new_report = dict(field.report)
    new_report['art_direction'] = report
    new_report['stats'] = relief_stats(steps, dims, mask=plaque)
    norm = np.clip(steps.astype(np.float32) / max(relief_layers, 1), 0.0, 1.0).astype(np.float32)
    return ReliefField(steps=steps, normalized=norm, mask=field.mask, dims=dims, report=new_report), report


# --------------------------------------------------------------------------- editor input
def relief_for_editor(image_path: str, width_mm: float, *, dims: Optional[ReliefDims] = None,
                      params: Optional[DepthParams] = None, geometry: str = 'depth-anything',
                      depth_map: Optional[str] = None, provider_options: Optional[dict] = None,
                      foreground_mask: Optional[str] = None, geometry_result=None) -> dict:
    """Everything the editor needs WITHOUT running the Stack5 colour stage:
    the relief grid (Lumina's working resolution, NEAREST copy of the image),
    the plaque mask (alpha >= 10, Lumina's rule), the provider geometry and the
    normal-pipeline relief field.  The pipeline itself recomputes the field
    with ``plaque_mask = recipe.mask_solid`` at export; for an opaque cover
    both masks are all-True, so the editor's ORIGINAL equals the export's."""
    from core.relief.provider import get_provider, load_mask_file
    from core.stack5.pipeline import HIFI_PX_PER_MM, working_resolution
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    dims = dims or ReliefDims()
    params = params or DepthParams()
    tw, th = working_resolution(image_path, width_mm)
    with Image.open(image_path) as im0:
        image_hw = (int(im0.size[1]), int(im0.size[0]))
        rgba = np.asarray(im0.convert('RGBA').resize((tw, th), Image.Resampling.NEAREST))
    plaque = rgba[..., 3] >= ALPHA_BACKGROUND
    rgb_grid = np.ascontiguousarray(rgba[..., :3])
    geom = geometry_result
    if geom is None:
        opts = dict(provider_options or {})
        if depth_map:
            opts.setdefault('depth_map', depth_map)
        provider = get_provider(geometry, **opts)
        pf = getattr(provider, 'preflight', None)
        if callable(pf):
            pf()
        geom = provider.generate(image_path, target_hw=(th, tw))
    fg = load_mask_file(foreground_mask) if foreground_mask else None
    field = process_depth(geom, (th, tw), dims, params, foreground_mask=fg, plaque_mask=plaque)
    return {'field': field, 'geom': geom, 'plaque_mask': plaque, 'grid_hw': (th, tw),
            'pixel_mm': 1.0 / float(HIFI_PX_PER_MM), 'rgb_grid': rgb_grid, 'image_hw': image_hw,
            'original_mm': original_mm_from_field(field),
            'relief_max_mm': dims.relief_layers * float(dims.layer_h), 'dims': dims, 'params': params,
            'geometry': geometry, 'depth_map': depth_map}
