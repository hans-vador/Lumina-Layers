"""Band mode end-to-end pipeline: album image -> HueForge-style plaque 3MF.

image -> (palette/schedule) -> ramp LUT (.npz, Lumina interchange format)
      -> BandProcessor (Lumina matcher + heightfield)
      -> heightfield_to_trimesh -> write_band_3mf (single part + plate-wide swaps)
      -> shaded predicted preview + swap instructions + stats.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from config import OUTPUT_DIR, _BASE_DIR
from core.band.optics import (BeerLambertModel, Filament, MeasuredRampModel, linear_to_srgb,
                              load_filament_library)
from core.band.schedule import BandSchedule
from core.band.tone import ToneCurve

REPO = _BASE_DIR
DEFAULT_FILAMENTS_JSON = os.path.join(REPO, 'assets', 'filaments_user.json')
# Measured (calibration-board) ramps are looked up here; synthetic ramp LUTs are
# written next to the output instead (the legacy LUT manager lists every .npz in
# this directory as a 'Merged' LUT, which a Band ramp is not).
CUSTOM_LUT_DIR = os.path.join(REPO, 'lut-npy预设', 'Custom')
TEMPLATE_SOURCES = [
    '/Users/hans_vador/Downloads/BambuAlbums/Blonde.3mf',
    '/Users/hans_vador/Downloads/BambuAlbums/Octane.3mf',
    '/Users/hans_vador/Downloads/BambuAlbums/Love_Sick.3mf',
]

FilamentsArg = Union[Dict[str, Filament], Sequence[Filament]]


# --------------------------------------------------------------------------- utils
def _band_filaments(schedule: BandSchedule, filaments: FilamentsArg) -> List[Filament]:
    """Return the schedule's filaments as a list in band order (bottom -> top)."""
    if isinstance(filaments, dict):
        return [filaments[n] for n in schedule.filament_names]
    lst = list(filaments)
    by_name = {f.name: f for f in lst}
    if all(n in by_name for n in schedule.filament_names):
        return [by_name[n] for n in schedule.filament_names]
    if len(lst) != schedule.n_bands:
        raise ValueError("filaments list must be in band order or contain all schedule names")
    return lst


def _luminance_y(f: Filament) -> float:
    c = np.asarray(f.rgb_lin, dtype=np.float64)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def _slug(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    base = re.sub(r'[<>:"/\\|?*]+', '_', base).strip() or 'untitled'
    return base


def _schedule_dict(schedule: BandSchedule) -> dict:
    d = asdict(schedule) if is_dataclass(schedule) else dict(schedule)
    d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
    d['n_layers'] = int(schedule.n_layers)
    d['key'] = schedule.key()
    return d


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


# --------------------------------------------------------------------------- preview
def composite_heights(height_mm: np.ndarray, schedule: BandSchedule, filaments: FilamentsArg,
                      model) -> np.ndarray:
    """Predicted linear-RGB colour of every column of a heightfield.

    Continuous Beer-Lambert compositing (port of rebuild3.comp) when the model
    exposes ``k_opaque`` (BeerLambertModel); otherwise (measured ramps or any
    model that only knows discrete rows) the model's ramp rows are interpolated
    linearly in height.  Columns below the base band's top take the base colour.
    """
    fils = _band_filaments(schedule, filaments)
    h = np.asarray(height_mm, dtype=np.float64)
    k_opaque = getattr(model, 'k_opaque', None)
    if k_opaque is not None:
        # bottom-up: colour = base; each band b>=1 of thickness t_b(h) = clip(h - z_start_b, 0, n_b*lh)
        color = np.zeros(h.shape + (3,), dtype=np.float64)
        thru = np.ones(h.shape, dtype=np.float64)
        for b in range(schedule.n_bands - 1, 0, -1):
            z_lo = schedule.top_z(schedule.band_first_layer(b)) - schedule.layer_h   # bottom of band b
            t_full = schedule.layer_counts[b] * schedule.layer_h
            t = np.clip(h - z_lo, 0.0, t_full)
            td = max(float(fils[b].td_mm), 1e-6)
            a = 1.0 - np.exp(-float(k_opaque) * t / td)
            color += (thru * a)[..., None] * np.asarray(fils[b].rgb_lin, dtype=np.float64)
            thru *= (1.0 - a)
        return color + thru[..., None] * np.asarray(fils[0].rgb_lin, dtype=np.float64)

    ramp = np.asarray(model.ramp(schedule, fils), dtype=np.float64)      # (K',3) linear
    zs = np.asarray(schedule.heights(), dtype=np.float64)                 # (K',)
    out = np.empty(h.shape + (3,), dtype=np.float64)
    for c in range(3):
        out[..., c] = np.interp(h, zs, ramp[:, c])
    return out


def render_preview(height_mm: np.ndarray, schedule: BandSchedule, filaments: FilamentsArg, model,
                   mask: Optional[np.ndarray] = None, relief_gain: float = 3.0) -> Image.Image:
    """Shaded predicted preview: colour through the optical model + hillshade (rebuild3 L185-190)."""
    h = np.asarray(height_mm, dtype=np.float64)
    pred = np.clip(linear_to_srgb(np.clip(composite_heights(h, schedule, filaments, model), 0, 1)), 0, 1)
    gy, gx = np.gradient(h * relief_gain)
    nrm = np.sqrt(1.0 + gx ** 2 + gy ** 2)
    shade = 1.0 / nrm
    shade = 0.72 + 0.28 * (shade + 0.35 * gx / nrm)
    img = np.clip(pred * shade[..., None], 0, 1)
    if mask is not None:
        img = np.where(np.asarray(mask, bool)[..., None], img, 0.85)
    return Image.fromarray((img * 255).astype(np.uint8))


# --------------------------------------------------------------------------- advisory stamp
def _font(px: int):
    for fp in ("/System/Library/Fonts/Supplemental/Arial Black.ttf",
               "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(fp, px)
        except Exception:
            continue
    return ImageFont.load_default()


def advisory_stamp_mask(width_px: int, height_px: int) -> np.ndarray:
    """Binary 'PARENTAL ADVISORY' label (True = light), rendered at 6x and downsampled."""
    LW, LH, Sc = max(width_px, 8), max(height_px, 4), 6
    im2 = Image.new("L", (LW * Sc, LH * Sc), 255)
    dd = ImageDraw.Draw(im2)
    w2, h2 = LW * Sc, LH * Sc
    bw = max(Sc, w2 // 34)
    dd.rectangle([0, 0, w2 - 1, h2 - 1], outline=0, width=bw)
    dd.rectangle([bw, bw, w2 - bw - 1, int(h2 * 0.34)], fill=0)
    dd.rectangle([bw, h2 - int(h2 * 0.30), w2 - bw - 1, h2 - bw - 1], fill=0)

    def ctr(y, txt, fpx, fill):
        ft = _font(max(int(fpx), 4))
        bb = dd.textbbox((0, 0), txt, font=ft)
        dd.text(((w2 - (bb[2] - bb[0])) // 2, y - bb[1]), txt, font=ft, fill=fill)

    ctr(int(h2 * 0.055), "PARENTAL", int(h2 * 0.235), 255)
    ctr(int(h2 * 0.40), "ADVISORY", int(h2 * 0.26), 0)
    ctr(int(h2 * 0.745), "EXPLICIT CONTENT", int(h2 * 0.155), 255)
    return np.asarray(im2.resize((LW, LH), Image.LANCZOS)) > 128


def apply_advisory_stamp(height_mm: np.ndarray, mask_solid: np.ndarray, schedule: BandSchedule,
                         corner: str = 'br') -> np.ndarray:
    """Stamp a binary advisory label into the heightfield: dark -> top_z(n0), light -> top_z(K).

    corner: 'br' | 'bl' | 'bc' (bottom-right / -left / -centre, image space)."""
    corner = (corner or '').lower()
    if corner not in ('br', 'bl', 'bc'):
        raise ValueError("advisory corner must be 'br', 'bl' or 'bc'")
    H, W = height_mm.shape
    LW = max(int(0.13 * W), 8)
    LH = max(int(0.60 * LW), 4)
    lab = advisory_stamp_mask(LW, LH)
    z_d = schedule.top_z(int(schedule.layer_counts[0]))
    z_l = schedule.top_z(int(schedule.n_layers))
    m2 = int(0.035 * W)
    r0 = max(H - m2 - LH, 0)
    c0 = {"br": W - m2 - LW, "bl": m2, "bc": (W - LW) // 2}[corner]
    c0 = int(np.clip(c0, 0, max(W - LW, 0)))
    h = np.asarray(height_mm, dtype=np.float32).copy()
    patch = np.where(lab, z_l, z_d).astype(np.float32)[:min(LH, H - r0), :min(LW, W - c0)]
    h[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]] = patch
    mask_solid = np.asarray(mask_solid, bool).copy()
    mask_solid[r0:r0 + patch.shape[0], c0:c0 + patch.shape[1]] = True
    return h, mask_solid


# --------------------------------------------------------------------------- schedule selection
def _pick_model(schedule: Optional[BandSchedule], analytic=None, ramp_dir: Optional[str] = None):
    """MeasuredRampModel if a measured ramp exists for this schedule key (in
    ramp_dir or lut-npy预设/Custom), else the analytic model (default
    BeerLambertModel with optics.DEFAULT_K_OPAQUE)."""
    if schedule is not None:
        for d in ([ramp_dir] if ramp_dir else []) + [CUSTOM_LUT_DIR]:
            p = os.path.join(d, f"Band_{schedule.key()}_measured.npz")
            if os.path.exists(p):
                return MeasuredRampModel.from_npz(p), f"measured:{p}"
    model = analytic if analytic is not None else BeerLambertModel()
    return model, repr(model)


def _assignment_kwargs(height_mode: str, luma_weight: Optional[float], tone) -> dict:
    """How the optimiser must assign histogram colours to ramp rows so that it
    predicts what BandProcessor(height_mode=...) will do."""
    from core.band import palette as pal
    if height_mode == 'luminance':
        return {'tone': ToneCurve.from_any(tone), 'wL_match': None}
    wL_match = pal.matcher_wL(luma_weight) if luma_weight and float(luma_weight) != 1.0 else None
    return {'tone': None, 'wL_match': wL_match}


def resolve_schedule_detailed(image_path: str, library: Dict[str, Filament],
                              palette: Optional[List[str]], order: str,
                              schedule: Optional[BandSchedule], n0: int = 9,
                              max_layers: int = 27, model=None,
                              luma_weight: Optional[float] = 5.0,
                              n_filaments=(4, 5), height_mode: str = 'luminance',
                              tone=None) -> tuple:
    """(schedule, optimiser report | None).

    The optimiser assigns histogram colours to ramp rows the way the processor
    will (height_mode 'luminance': tone curve on L*; 'match': the luminance-
    weighted matcher metric) and scores the perceptual error (wL=1.5, true
    CIELAB) of that assignment, so the report's ``mean_dE`` is the predicted
    mean dE of the print.  None report when the schedule was given explicitly."""
    if schedule is not None:
        return schedule, None
    from core.band import palette as pal
    if model is None:
        model = BeerLambertModel()
    if n0 < 1:
        raise ValueError("n0 must be >= 1")
    assign = _assignment_kwargs(height_mode, luma_weight, tone)
    hist = pal.image_hist_lab(image_path)
    if palette:
        names = [p.strip() for p in palette if p.strip()]
        missing = [n for n in names if n not in library]
        if missing:
            raise KeyError(f"filaments not in library: {missing}; available: {sorted(library)}")
        if order != 'fixed':
            names = sorted(names, key=lambda n: _luminance_y(library[n]))   # darkest base -> lightest top
        sched, rep = pal.optimise_thickness_detailed(hist, tuple(names), library, model,
                                                     n0=n0, max_layers=max_layers, **assign)
        rep['search'] = 'thickness-only (fixed filament order)'
        return sched, rep
    sched, rep = pal.select_palette_detailed(hist, library, model, n_filaments=n_filaments,
                                             n0=n0, max_layers=max_layers, **assign)
    sizes = (n_filaments,) if isinstance(n_filaments, int) else tuple(n_filaments)
    rep['search'] = (f"palette ({'/'.join(str(b) for b in sizes)}-filament subsets, luminance order) "
                     f"+ thickness")
    return sched, rep


def resolve_schedule(image_path: str, library: Dict[str, Filament], palette: Optional[List[str]],
                     order: str, schedule: Optional[BandSchedule], n0: int = 9,
                     max_layers: int = 27, model=None, luma_weight: Optional[float] = 5.0,
                     height_mode: str = 'luminance', tone=None) -> BandSchedule:
    return resolve_schedule_detailed(image_path, library, palette, order, schedule,
                                     n0=n0, max_layers=max_layers, model=model,
                                     luma_weight=luma_weight, height_mode=height_mode, tone=tone)[0]


# --------------------------------------------------------------------------- main entry
def convert_album_to_band_3mf(image_path: str, width_mm: float = 150.0,
                              filaments_json: str = DEFAULT_FILAMENTS_JSON,
                              palette: Optional[List[str]] = None, order: str = 'auto',
                              schedule: Optional[BandSchedule] = None, mesh_pitch_mm: float = 0.15,
                              engrave: bool = True, out_dir: str = OUTPUT_DIR,
                              title: Optional[str] = None, advisory: Optional[str] = None,
                              quantize_colors: int = 256, smooth_sigma: float = 10,
                              interpolate: bool = True, gaussian_px: float = 0.5,
                              bed_mm: float = 256.0, print_overrides: Optional[dict] = None,
                              template_sources: Optional[List[str]] = None,
                              embed_thumbnail: bool = True,
                              ramp_dir: Optional[str] = None,
                              luma_weight: float = 5.0,
                              k_opaque: Optional[float] = None,
                              n0: int = 9, max_layers: int = 27,
                              n_filaments=(4, 5),
                              height_mode: str = 'luminance',
                              tone=None,
                              engrave_floor_mm: Optional[float] = None) -> dict:
    """Convert one album cover to a HueForge-style single-part plaque 3MF.

    ramp_dir: where the ramp LUT .npz is saved (default: out_dir; the legacy
    LUT manager would list a ramp in lut-npy预设/Custom as a 'Merged' LUT).
    height_mode: 'luminance' (HueForge-style tone mapping, default) or 'match'
    (Lumina nearest-row matcher); tone: ToneCurve / 'reference' / 'linear' /
    'q_lo,q_hi,gamma' for 'luminance' mode.
    luma_weight: matcher lightness weight ('match' mode, see BandProcessor.process).
    k_opaque: Beer-Lambert steepness for the synthetic ramp/palette search
    (None = optics.DEFAULT_K_OPAQUE); ignored when a measured ramp exists.
    engrave_floor_mm: relief floor (None = reference 0.521 mm, see BandProcessor).

    Returns {threemf, preview_png, schedule (dict), swap_instructions, stats, ...}.
    """
    from core.band.heightfield_mesh import heightfield_to_trimesh
    from core.band.processor import BandProcessor
    from core.band.ramp_lut import build_ramp_lut, ramp_filename
    from core.band.writer3mf import write_band_3mf

    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    os.makedirs(out_dir, exist_ok=True)
    slug = _slug(image_path)
    title = title or slug.replace('_', ' ')
    tone_curve = ToneCurve.from_any(tone)

    library = load_filament_library(filaments_json)
    analytic = BeerLambertModel(k_opaque)
    t_sched = datetime.now()
    schedule, palette_report = resolve_schedule_detailed(image_path, library, palette, order, schedule,
                                                         n0=n0, max_layers=max_layers, model=analytic,
                                                         luma_weight=luma_weight, n_filaments=n_filaments,
                                                         height_mode=height_mode, tone=tone_curve)
    schedule_seconds = (datetime.now() - t_sched).total_seconds()
    if schedule.n_layers > 27:
        print(f"[BAND] warning: schedule has {schedule.n_layers} layers (> 27 reference max)")
    fils = [library[n] for n in schedule.filament_names]
    lut_dir = ramp_dir or out_dir
    model, model_name = _pick_model(schedule, analytic, lut_dir)

    # ramp LUT in Lumina's .npz interchange format (rgb uint8 (K',3), stacks int32 (K',5))
    ramp = build_ramp_lut(schedule, library, model)
    os.makedirs(lut_dir, exist_ok=True)
    ramp_npz = os.path.join(lut_dir, ramp_filename(schedule, model))
    ramp.save(ramp_npz)
    mode_key = f"Band:{schedule.key()}"

    bp = BandProcessor(ramp, schedule, library, ramp_npz, mode_key, engrave_floor_mm=engrave_floor_mm)
    res = bp.process(image_path, width_mm, quantize_colors=quantize_colors, smooth_sigma=smooth_sigma,
                     interpolate=interpolate, engrave=engrave, gaussian_px=gaussian_px,
                     mesh_pitch_mm=mesh_pitch_mm, luma_weight=luma_weight,
                     height_mode=height_mode, tone=tone_curve)
    height, mask = res.height_mm, res.mask_solid
    stats = dict(res.stats)
    if advisory and advisory.lower() != 'none':
        height, mask = apply_advisory_stamp(height, mask, schedule, advisory)
        stats['advisory_stamp'] = advisory.lower()
        # report the surface distribution of the height field that is actually meshed
        stats['pre_stamp_top_surface_fraction_per_filament'] = stats['top_surface_fraction_per_filament']
        stats.update(bp.surface_fractions(height, mask))

    mesh = heightfield_to_trimesh(height, res.pitch_mm, mask=mask if not mask.all() else None)

    # preview
    preview = render_preview(height, schedule, fils, model, mask=mask)
    preview_png = os.path.join(out_dir, f"{slug}_band_preview.png")
    preview.save(preview_png)

    swaps = schedule.swap_entries(library)
    instructions = schedule.swap_instructions(library)
    td_lines = ", ".join(f"{f.name} TD {f.td_mm:g} mm ({f.td_source})" for f in fils)
    if height_mode == 'luminance':
        hm = (f"Height mapping: luminance (L* -> height, tone curve q_lo {tone_curve.q_lo:g} "
              f"q_hi {tone_curve.q_hi:g} gamma {tone_curve.gamma:g})")
    else:
        hm = f"Height mapping: nearest ramp colour (luma weight {luma_weight:g})"
    description = (f"{instructions}\n\nFilaments: {td_lines}\nOptical model: {model_name}\n{hm}\n"
                   f"Layer height {schedule.layer_h} mm, first layer {schedule.first_layer_mm} mm, "
                   f"{schedule.n_layers} layers, {schedule.n_bands - 1} filament changes.\n"
                   f"Generated by Lumina Studio Band mode.")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    threemf = os.path.join(out_dir, f"{slug}_Lumina_HiFi_HF_{ts}.3mf")
    size_mm = float(width_mm)
    write_kwargs = dict(
        filaments=fils, swap_entries=swaps, title=f"{title} (Album Art - HueForge style)",
        description=description, size_mm=size_mm, bed_mm=bed_mm,
        template_sources=template_sources or TEMPLATE_SOURCES, print_overrides=print_overrides,
        thumbnail_png=preview_png if embed_thumbnail else None,
    )
    threemf = write_band_3mf(mesh, threemf, **write_kwargs) or threemf

    stats.update({
        'optical_model': model_name,
        'schedule_seconds': schedule_seconds,
        'palette_report': _jsonable(palette_report) if palette_report else None,
        'mesh_vertices': int(len(mesh.vertices)),
        'mesh_faces': int(len(mesh.faces)),
        'mesh_watertight': bool(mesh.is_watertight),
        'mesh_extent_mm': [float(x) for x in mesh.extents],
        'swap_entries': [(float(z), int(e), str(hx)) for z, e, hx in swaps],
    })
    sched_dict = _schedule_dict(schedule)
    sched_dict['filament_hex'] = [f.hex for f in fils]

    schedule_json = os.path.join(out_dir, f"{slug}_band_schedule.json")
    with open(schedule_json, 'w', encoding='utf-8') as fh:
        json.dump(_jsonable({'schedule': sched_dict, 'stats': stats, 'ramp_npz': ramp_npz,
                             'image': os.path.abspath(image_path), 'width_mm': width_mm}), fh, indent=2)
    swaps_txt = os.path.join(out_dir, f"{slug}_swaps.txt")
    with open(swaps_txt, 'w', encoding='utf-8') as fh:
        fh.write(description + "\n")

    return {
        'threemf': threemf,
        'preview_png': preview_png,
        'schedule': sched_dict,
        'schedule_obj': schedule,
        'swap_instructions': instructions,
        'stats': _jsonable(stats),
        'ramp_npz': ramp_npz,
        'schedule_json': schedule_json,
        'swaps_txt': swaps_txt,
        'mode_key': mode_key,
    }
