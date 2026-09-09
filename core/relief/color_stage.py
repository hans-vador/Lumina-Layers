"""The Stack5 colour stage, unchanged, for Relief mode.

Runs exactly what flat Stack5 (core/stack5/pipeline.py convert_album_stack5
steps 1-4) runs on the ORIGINAL RGB image - palette search, synthetic (or
measured) LUT, dynamic colour-mode registration, Lumina's HiFi processor with
the hue-first matcher, min-region cleanup - and returns the per-pixel 5-layer
recipe instead of meshing it face down.  Nothing about the depth map reaches
this module, so for one image + palette the recipe is identical whatever
geometry is used (the SHA-256 of the material matrix equals the flat Stack5
``stats.material_matrix_sha256``; see tests/test_relief_stack5.py).

The only deliberate difference from run_lumina: the processor is called
directly (LuminaImageProcessor.process_image) instead of through
core.converter.convert_image_to_3d, which would also mesh and write a flat
face-down 3MF that Relief mode has no use for.  The converter passes the same
arguments (verified against core/converter.py lines 859-870).
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from config import ModelingMode
from core.band.optics import BeerLambertModel, Filament, load_filament_library
from core.stack5.cleanup import min_region_cleanup, region_stats, resync_matched_rgb
from core.stack5.lut import (LAYER_H, N_LAYERS, choose_backing, lut_meta, pure_stack_indices,
                             register_stack5_mode, save_lut_npz, synth_lut)
from core.stack5.metric import make_matcher, resolve_hue_params
from core.stack5.palette import (DEFAULT_CHROMA_WEIGHT, DEFAULT_DOMINANT_W, DEFAULT_METRIC, DEFAULT_NEED_SHARE,
                                 DEFAULT_PREFER_TOL, DEFAULT_SPOOL_BONUS, DEFAULT_SPOOL_DE, DEFAULT_SPOOL_PENALTY,
                                 evaluate_palette, image_hist, select_palette)
from core.stack5.pipeline import (DEFAULT_MIN_REGION_PX, _run_nonce, _slug, apply_td_overrides,
                                  default_filaments_json, set_determinism, stamp_advisory, working_copy,
                                  working_resolution)


def material_matrix_sha256(material_matrix: np.ndarray) -> str:
    """Same digest flat Stack5 records in stats.material_matrix_sha256."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(material_matrix).astype(np.int8)).tobytes()).hexdigest()


@dataclass
class ColourRecipe:
    """Everything the geometry stage needs from Stack5."""
    names: list[str]
    filaments: list[Filament]
    backing_slot: int
    lut_rgb: np.ndarray
    lut_stacks: np.ndarray
    lut_npz: str
    lut_source: Optional[str]
    mode_key: str
    material_matrix: np.ndarray       # (H, W, L) slot ids, [..., 0] = viewing surface, -1 outside mask
    matched_rgb: np.ndarray           # (H, W, 3) uint8 predicted colour per pixel
    mask_solid: np.ndarray            # (H, W) bool
    quantized_image: Optional[np.ndarray]
    dimensions: tuple[int, int]       # (W, H) px
    pixel_mm: float
    palette_report: dict
    cleanup: dict
    matcher: str
    sha256: str
    library: dict
    model: object
    settings: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    stamped_input: Optional[str] = None

    @property
    def grid_hw(self) -> tuple[int, int]:
        return int(self.dimensions[1]), int(self.dimensions[0])


def stack5_colour_recipe(image_path: str, width_mm: float, job_dir: str, slug: Optional[str] = None, *,
                         filaments_json: Optional[str] = None, palette: Optional[Sequence[str]] = None,
                         backing: Optional[str] = None, quantize_colors: int = 96, smooth_sigma: float = 10,
                         advisory: Optional[str] = None, seed: int = 0, must_include: Sequence[str] = (),
                         prefer: Sequence[str] = (), prefer_tol: float = DEFAULT_PREFER_TOL,
                         metric: str = DEFAULT_METRIC, hue_params: Optional[dict] = None, wL: float = 1.0,
                         hist_k: int = 64, k_opaque: Optional[float] = None, td_scale: float = 1.0,
                         td_overrides: Optional[dict] = None, lut_npz: Optional[str] = None,
                         chroma_weight: float = DEFAULT_CHROMA_WEIGHT, spool_bonus: float = DEFAULT_SPOOL_BONUS,
                         spool_de: float = DEFAULT_SPOOL_DE, dominant_w: float = DEFAULT_DOMINANT_W,
                         max_spools: int = 5, min_spools: Optional[int] = None,
                         spool_penalty: float = DEFAULT_SPOOL_PENALTY, need_share: float = DEFAULT_NEED_SHARE,
                         min_region_px: int = DEFAULT_MIN_REGION_PX, hue_weight: float = 0.0,
                         enable_cleanup: bool = True, layer_h: float = LAYER_H, layers: int = N_LAYERS,
                         lut_basename: Optional[str] = None) -> ColourRecipe:
    """Stack5 steps 1-4 on the original image.  Parameters mirror
    core.stack5.pipeline.convert_album_stack5 (same defaults)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    if advisory is not None and str(advisory).lower() in ('', 'none'):
        advisory = None
    slug = slug or _slug(image_path)
    os.makedirs(job_dir, exist_ok=True)
    timing: dict = {}

    set_determinism(seed)
    filaments_json = filaments_json or default_filaments_json()
    library = apply_td_overrides(load_filament_library(filaments_json), td_scale, td_overrides)
    model = BeerLambertModel(k_opaque)

    # 1. palette (identical to convert_album_stack5)
    t0 = time.perf_counter()
    hist = image_hist(image_path, k=hist_k, seed=seed)
    scoring_kw = dict(chroma_weight=float(chroma_weight), spool_bonus=float(spool_bonus),
                      spool_de=float(spool_de), dominant_w=float(dominant_w))
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
        palette_eval = evaluate_palette(hist, names, library, backing=backing, model=model, wL=wL,
                                        metric=metric, hue_params=hue_params, **scoring_kw)
        palette_report = {'search': 'explicit palette' + ('' if backing else ' x all backings'),
                          'metric': metric, 'best': palette_eval, 'top': [palette_eval],
                          'backing_rule': f"fixed: {backing}" if backing else 'searched: lowest cost member'}
    else:
        if not 1 <= int(max_spools) <= 5:
            raise ValueError("max_spools must be within 1..5 (one AMS)")
        names, palette_report = select_palette(hist, library, model, n=int(max_spools), n_min=min_spools,
                                               must_include=must_include, backing=backing, wL=wL, prefer=prefer,
                                               prefer_tol=prefer_tol, metric=metric, hue_params=hue_params,
                                               spool_penalty=float(spool_penalty), need_share=float(need_share),
                                               **scoring_kw)
    fils = [library[n] for n in names]
    backing_slot = choose_backing(fils, backing or palette_report['best']['backing'])
    timing['palette_s'] = time.perf_counter() - t0

    # 2. LUT + colour mode
    t0 = time.perf_counter()
    lut_base = lut_basename or f"{slug}_relief_lut.npz"
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
        lut_path = save_lut_npz(os.path.join(job_dir, lut_base), lut_rgb, lut_stacks,
                                dict(lut_meta(fils, backing_slot, model, layers=lut_stacks.shape[1], layer_h=layer_h),
                                     measured_from=lut_source, model='measured'))
    else:
        lut_rgb, lut_stacks = synth_lut(fils, backing_slot, model, layers=int(layers), layer_h=float(layer_h))
        lut_source = None
        lut_path = save_lut_npz(os.path.join(job_dir, lut_base), lut_rgb, lut_stacks,
                                lut_meta(fils, backing_slot, model, layers=int(layers), layer_h=float(layer_h)))
    mode_key = register_stack5_mode(fils)
    timing['lut_s'] = time.perf_counter() - t0

    # 3. working copy at Lumina's resolution (+ optional advisory stamp)
    stamped_png = None
    if advisory:
        stamped_png = stamp_advisory(image_path, width_mm, advisory,
                                     os.path.join(job_dir, f"{slug}_input_stamped_{advisory.lower()}.png"))
    work_dir = os.path.join(job_dir, '.work')
    os.makedirs(work_dir, exist_ok=True)
    image_for_lumina = os.path.join(work_dir, f"{slug}_relief{_run_nonce()}.png")   # unique per run (see _run_nonce)
    if stamped_png:
        Image.open(stamped_png).save(image_for_lumina)
    else:
        working_copy(image_path, width_mm, image_for_lumina)

    # 4. Lumina's processor, exactly as run_lumina / convert_image_to_3d call it
    t0 = time.perf_counter()
    from core.image_processing import LuminaImageProcessor
    set_determinism(seed)
    try:
        proc = LuminaImageProcessor(lut_path, mode_key, hue_weight=float(hue_weight))
        proc.enable_cleanup = bool(enable_cleanup)
        matcher = make_matcher(metric, proc.lut_rgb, wL=wL, hue_params=hue_params)
        if matcher is not None:
            proc.hue_matcher = matcher
            print(f"[RELIEF] pixel matcher: {matcher.describe()}")
        matcher_desc = matcher.describe() if matcher is not None else 'lumina 8-bit Lab KDTree'
        res = proc.process_image(image_path=image_for_lumina, target_width_mm=float(width_mm),
                                 modeling_mode=ModelingMode.HIGH_FIDELITY, quantize_colors=int(quantize_colors),
                                 auto_bg=False, bg_tol=0, blur_kernel=0, smooth_sigma=float(smooth_sigma))
    finally:
        try:
            os.remove(image_for_lumina)
            if not os.listdir(work_dir):
                os.rmdir(work_dir)
        except OSError:
            pass
    mm = np.asarray(res['material_matrix'])
    mask = np.asarray(res['mask_solid'], bool)
    info = {'min_region_px': int(min_region_px), 'before': region_stats(mm, mask, max(int(min_region_px), 16))}
    if int(min_region_px) > 1:
        new_mm, st = min_region_cleanup(mm, mask, int(min_region_px))
        matched = resync_matched_rgb(new_mm, mask, res['matched_rgb'], proc.lut_rgb, proc.ref_stacks)
        mm = new_mm
        info['cleanup'] = st
        info['after'] = region_stats(mm, mask, max(int(min_region_px), 16))
    else:
        matched = np.asarray(res['matched_rgb'])
        info['cleanup'] = None
        info['after'] = info['before']
    timing['lumina_s'] = time.perf_counter() - t0

    tw, th = res['dimensions']
    exp_w, exp_h = working_resolution(image_path, width_mm)
    if (tw, th) != (exp_w, exp_h):
        raise RuntimeError(f"Lumina grid {tw}x{th} differs from the expected working resolution {exp_w}x{exp_h}")

    settings = {
        'width_mm': float(width_mm), 'quantize_colors': int(quantize_colors), 'smooth_sigma': float(smooth_sigma),
        'advisory': advisory, 'seed': int(seed), 'layer_h': float(layer_h), 'wL': float(wL), 'hist_k': int(hist_k),
        'hue_weight': float(hue_weight), 'td_scale': float(td_scale), 'td_overrides': dict(td_overrides or {}),
        'must_include': list(must_include or ()), 'prefer': list(prefer or ()), 'prefer_tol': float(prefer_tol),
        'scoring': scoring_kw, 'metric': metric,
        'hue_params': resolve_hue_params(hue_params) if metric == 'hue' else None,
        'pixel_matcher': matcher_desc, 'max_spools': int(max_spools), 'min_spools': min_spools,
        'spool_penalty': float(spool_penalty), 'need_share': float(need_share),
        'min_region_px': int(min_region_px), 'enable_cleanup': bool(enable_cleanup),
        'model': repr(model) if lut_source is None else f"measured:{lut_source}",
        'filaments_json': os.path.abspath(filaments_json),
        'colour_source': 'original image (Stack5 unchanged); depth never influences the recipe',
    }
    return ColourRecipe(
        names=list(names), filaments=fils, backing_slot=int(backing_slot), lut_rgb=lut_rgb, lut_stacks=lut_stacks,
        lut_npz=lut_path, lut_source=lut_source, mode_key=mode_key, material_matrix=mm,
        matched_rgb=np.asarray(matched), mask_solid=mask, quantized_image=res.get('quantized_image'),
        dimensions=(int(tw), int(th)), pixel_mm=float(res['pixel_scale']), palette_report=palette_report,
        cleanup=info, matcher=matcher_desc, sha256=material_matrix_sha256(mm), library=library, model=model,
        settings=settings, timing=timing, stamped_input=stamped_png)
