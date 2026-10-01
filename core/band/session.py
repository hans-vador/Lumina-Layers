"""BandSession: the editable state behind an interactive Band-mode editor.

HueForge splits its work the way its UI does.  Loading and filtering an image is
expensive and completely independent of the colour plan, while everything the
sliders touch - filament order, per-band layer counts, TDs, the tone curve - is
a handful of numpy ops over a cached (H, W) lightness map.  That split is why
HueForge can show a live preview and still keep "Remesh" as a separate button.

``BandSession`` caches the expensive half at :meth:`load` and recomputes the
cheap half on demand, so a slider drag costs a composite over a downscaled
lightness map (~10 ms) instead of a full re-run of the image pipeline (~2.5 s).

The session deliberately owns no UI types: ``ui/band_tab.py`` drives it, and so
can a script. Both export formats and the viewport mesh use the edited session
height field, including inversion, range stretching and borders.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from core.band.optics import (BeerLambertModel, Filament, DEFAULT_K_OPAQUE, hex_to_rgb01,
                              linear_to_srgb, load_filament_library, save_filament_library,
                              srgb_to_lab_d65, srgb_to_linear)
from core.band.pipeline import render_preview
from core.band.schedule import BandSchedule
from core.band.tone import ToneCurve, engrave_floor

PX_PER_MM = 10.0          # Lumina's raster density (pixel_scale 0.1 mm)
PREVIEW_MAX_PX = 720      # longest edge of the interactive preview

# HueForge's LED presets for back-lit (lithophane) preview, as sRGB hex.
LED_PRESETS = {
    'Warm White 2700K': '#FFB16E',
    'Natural White 4000K': '#FFD5AE',
    'Cold White 6500K': '#FFFAFD',
    'Red': '#FF2020', 'Green': '#20FF20', 'Blue': '#2020FF',
    'Cyan': '#20FFFF', 'Magenta': '#FF20FF',
}


# --------------------------------------------------------------------------- #
# Image loading (the expensive, schedule-independent half)
# --------------------------------------------------------------------------- #
def load_lightness(image_path: str, width_mm: float, smooth_sigma: float = 10.0,
                   px_per_mm: float = PX_PER_MM) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """image -> (L* map, bilateral RGB, solid mask, pixel_scale_mm).

    The RGB is kept as well as the lightness because Color Match mode matches
    pixel *colour* against the ramp, while Luminance mode only needs L*.

    Mirrors what LuminaImageProcessor does ahead of matching: EXIF-correct,
    resample to the print raster with LANCZOS, bilateral-filter, then take
    CIELAB L*.  Alpha (when present) becomes the solid mask.
    """
    img = ImageOps.exif_transpose(Image.open(image_path))
    has_alpha = img.mode in ('RGBA', 'LA') or 'transparency' in img.info
    rgba = img.convert('RGBA')
    target_w = max(1, int(round(float(width_mm) * float(px_per_mm))))
    scale = target_w / float(rgba.width)
    target_h = max(1, int(round(rgba.height * scale)))
    rgba = rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)

    arr = np.asarray(rgba)
    rgb = np.ascontiguousarray(arr[..., :3])
    mask = (arr[..., 3] > 127) if has_alpha else np.ones(arr.shape[:2], dtype=bool)
    if smooth_sigma and smooth_sigma > 0:
        rgb = cv2.bilateralFilter(rgb, d=9, sigmaColor=float(smooth_sigma),
                                  sigmaSpace=float(smooth_sigma))
    L = srgb_to_lab_d65(rgb.astype(np.float64) / 255.0)[..., 0]
    return (L.astype(np.float32), rgb.astype(np.uint8), mask,
            float(width_mm) / float(target_w))


def _downscale(L: np.ndarray, rgb: np.ndarray, mask: np.ndarray,
               max_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = L.shape
    if max(h, w) <= max_px:
        return L, rgb, mask
    s = max_px / float(max(h, w))
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    L_s = cv2.resize(L, size, interpolation=cv2.INTER_AREA)
    rgb_s = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    m_s = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return L_s, rgb_s, m_s


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
@dataclass
class BandSession:
    """Editable Band-mode document: an image plus the colour plan over it."""

    image_path: str
    L: np.ndarray                      # (H, W) float32 CIELAB L*
    rgb: np.ndarray                    # (H, W, 3) uint8, bilateral-filtered
    mask: np.ndarray                   # (H, W) bool
    pixel_scale: float                 # mm per pixel
    library: Dict[str, Filament]

    filament_names: List[str] = field(default_factory=list)   # bottom -> top
    layer_counts: List[int] = field(default_factory=list)

    width_mm: float = 150.0
    first_layer_mm: float = 0.16
    layer_h: float = 0.08
    base_min_layers: int = 5
    smooth_sigma: float = 10.0

    tone: ToneCurve = field(default_factory=ToneCurve.reference)
    k_opaque: float = DEFAULT_K_OPAQUE
    engrave: bool = True

    lighting: str = 'front'            # 'front' (filament painting) | 'back' (lithophane)
    led_hex: str = LED_PRESETS['Natural White 4000K']
    light_intensity: float = 1.0

    negative: bool = False             # HueForge 'Negative': flip dark<->light
    full_range: bool = False           # HueForge 'Full Range': stretch luminance
    border_width_mm: float = 0.0       # HueForge border
    border_depth_mm: float = 0.0
    nozzle_temp_c: float = 205.0       # print temperature written into the 3MF filament profile
    first_layer_speed: float = 50.0    # mm/s; Bambu stock for 0.08mm High Quality @BBL X2D

    # --- Model Geometry (HueForge's panel of the same name) ----------------
    min_depth_mm: Optional[float] = None   # where colour blending begins; None = engrave floor
    max_depth_mm: Optional[float] = None   # cap on mesh height; None = top of the stack
    depth_mode: str = 'static'             # 'static' | 'dynamic' | 'clipped'
    spike_removal: str = 'off'             # 'off' | 'fast' | 'moderate' | 'aggressive'
    mesh_mode: str = 'luminance'
    mesh_core: List[dict] = field(default_factory=list)
    combo: float = 1.0
    tolerance: float = 8.0
    region_split: float = 0.5
    reverse_regions: bool = False
    channel_order: str = 'BGR'
    match_metric: str = 'cielab'        # 'luminance' | 'rgb' | 'cielab' | 'hsl' | 'dot'

    _L_small: np.ndarray = field(default=None, repr=False)
    _rgb_small: np.ndarray = field(default=None, repr=False)
    _mask_small: np.ndarray = field(default=None, repr=False)

    # ----- construction --------------------------------------------------- #
    @classmethod
    def load(cls, image_path: str, library: Dict[str, Filament] | str,
             filament_names: Optional[List[str]] = None,
             layer_counts: Optional[List[int]] = None,
             width_mm: float = 150.0, smooth_sigma: float = 10.0,
             **kw) -> 'BandSession':
        if isinstance(library, str):
            library = load_filament_library(library)
        L, rgb, mask, px = load_lightness(image_path, width_mm, smooth_sigma)
        names = list(filament_names or list(library)[:4])
        counts = list(layer_counts or ([9] + [4] * (len(names) - 1)))
        s = cls(image_path=image_path, L=L, rgb=rgb, mask=mask, pixel_scale=px, library=library,
                filament_names=names, layer_counts=counts, width_mm=width_mm,
                smooth_sigma=smooth_sigma, **kw)
        s._refresh_small()
        return s

    def _refresh_small(self) -> None:
        self._L_small, self._rgb_small, self._mask_small = _downscale(
            self.L, self.rgb, self.mask, PREVIEW_MAX_PX)

    def reload_image(self, image_path: Optional[str] = None,
                     width_mm: Optional[float] = None) -> 'BandSession':
        """Re-run the expensive half (new image, width or smoothing)."""
        if image_path:
            self.image_path = image_path
        if width_mm:
            self.width_mm = float(width_mm)
        self.L, self.rgb, self.mask, self.pixel_scale = load_lightness(
            self.image_path, self.width_mm, self.smooth_sigma)
        self._refresh_small()
        return self

    # ----- derived state -------------------------------------------------- #
    @property
    def schedule(self) -> BandSchedule:
        return BandSchedule(
            filament_names=tuple(self.filament_names),
            layer_counts=tuple(int(c) for c in self.layer_counts),
            first_layer_mm=self.first_layer_mm,
            layer_h=self.layer_h,
            base_min_layers=min(self.base_min_layers, int(self.layer_counts[0])),
        )

    @property
    def model(self) -> BeerLambertModel:
        return BeerLambertModel(k_opaque=self.k_opaque)

    @property
    def filaments(self) -> List[Filament]:
        return [self.library[n] for n in self.filament_names]

    def z_bounds(self) -> tuple[float, float]:
        sched = self.schedule
        z_lo = engrave_floor(sched) if self.engrave else float(sched.top_z(sched.n0))
        z_hi = float(sched.top_z(sched.n_layers))
        if self.min_depth_mm is not None:
            z_lo = float(self.min_depth_mm)
        if self.max_depth_mm is not None:
            z_hi = float(self.max_depth_mm)
        if not 0 < z_lo <= z_hi <= sched.total_height_mm:
            raise ValueError('Depth bounds must be positive, ordered, and within the print stack')
        return z_lo, z_hi

    # ----- the cheap half ------------------------------------------------- #
    def _tone_input(self, L: np.ndarray, mask: np.ndarray) -> np.ndarray:
        q = np.asarray(L, dtype=np.float64)
        if self.full_range and np.any(mask):
            lo, hi = np.percentile(q[mask], (0.5, 99.5))
            if hi - lo > 1e-6:
                q = np.clip((q - lo) / (hi - lo), 0.0, 1.0) * 100.0
        if self.negative:
            q = 100.0 - q
        return q

    def _apply_border(self, h: np.ndarray, z_lo: float, z_hi: float) -> np.ndarray:
        if self.border_width_mm <= 0:
            return h
        px = max(1, int(round(self.border_width_mm / max(self.pixel_scale, 1e-9)
                              * (h.shape[1] / max(self.L.shape[1], 1)))))
        z = float(np.clip(z_lo + self.border_depth_mm, z_lo, z_hi))
        h = h.copy()
        h[:px, :] = z
        h[-px:, :] = z
        h[:, :px] = z
        h[:, -px:] = z
        return h

    def copy_filaments_to_mesh_core(self) -> None:
        self.mesh_core = [dict(hex=f.hex, td_mm=f.td_mm, layer=self.schedule.cum(i))
                          for i, f in enumerate(self.filaments)]

    def mesh_reference(self):
        from core.band.mapping import reference_ramp
        if not self.mesh_core:
            self.copy_filaments_to_mesh_core()
        if self.mesh_core[-1]['layer'] > self.schedule.n_layers:
            raise ValueError('Mesh Core extends above the print stack; add print layers or lower its last stop')
        return reference_ramp(self.mesh_core, self.first_layer_mm, self.layer_h, self.k_opaque)

    def height_map(self, full: bool = True) -> np.ndarray:
        """Authoritative height field; preview resamples the completed mapping."""
        from core.band.mapping import MODES, mapped_fraction, match_heights
        if self.mesh_mode not in MODES:
            raise ValueError(f'Unknown mesh mode: {self.mesh_mode}')
        if self.mesh_mode == 'color_match' and not self.mesh_core:
            self.copy_filaments_to_mesh_core()
        key = repr((id(self.L), id(self.rgb), self.tone, self.mesh_mode, self.mesh_core,
                    self.layer_counts, self.first_layer_mm, self.layer_h, self.base_min_layers,
                    self.engrave, self.negative, self.full_range, self.min_depth_mm,
                    self.max_depth_mm, self.border_width_mm, self.border_depth_mm,
                    self.combo, self.tolerance, self.region_split, self.reverse_regions,
                    self.channel_order, self.match_metric, self.k_opaque, self.spike_removal))
        if getattr(self, '_height_key', None) != key:
            z_lo, z_hi = self.z_bounds()
            if self.mesh_mode == 'color_match':
                zs, colors = self.mesh_reference()
                h = match_heights(self.rgb / 255., zs, colors, self.match_metric)
                if self.negative:
                    h = zs[0] + zs[-1] - h
                h = np.clip(h, z_lo, z_hi)
            else:
                t = mapped_fraction(self.rgb / 255., self._tone_input(self.L, self.mask),
                                    self.mesh_mode, self.tone, self.combo, self.tolerance,
                                    self.region_split, self.reverse_regions, self.channel_order)
                h = z_lo + (z_hi-z_lo)*t
            if self.spike_removal != 'off':
                from scipy.ndimage import median_filter
                size = {'fast': 3, 'moderate': 5, 'aggressive': 7}.get(self.spike_removal)
                if size is None:
                    raise ValueError('Unknown spike removal setting')
                h = median_filter(h, size=size, mode='nearest')
            h = self._apply_border(h, z_lo, z_hi)
            self._height_full = np.where(self.mask, h, 0.)
            self._height_key = key
        h = self._height_full
        if not full and h.shape != self._L_small.shape:
            h = cv2.resize(h, (self._L_small.shape[1], self._L_small.shape[0]),
                           interpolation=cv2.INTER_AREA)
            h = np.where(self._mask_small, h, 0.)
        return h.copy()

    def transmitted(self, full: bool = False) -> np.ndarray:
        """Linear RGB seen through the plaque with the light behind it.

        Front-lit compositing asks what comes *back* off the stack; a lithophane
        asks what gets *through* it, which is a different sum.  Each band is a
        filter of thickness t: per channel it absorbs in proportion to how much
        that channel is missing from its own colour, so black blocks everything,
        red passes red, and a thicker band passes less.  That matches the
        per-channel TDs measured in docs/band/TD_CALIBRATION.md (Red
        td_rgb_mm [3.3, 0.29, 0.20] - red light goes through red filament).
        """
        return self.transmitted_heights(self.height_map(full=full))

    def transmitted_heights(self, h: np.ndarray) -> np.ndarray:
        """Backlit optical prediction for arbitrary column heights in mm."""
        sched = self.schedule
        fils = self.filaments
        T = np.ones(h.shape + (3,), dtype=np.float64)
        for b in range(sched.n_bands):
            if b == 0:                       # base band: everything below its top
                t = np.clip(h, 0.0, sched.top_z(sched.layer_counts[0]))
            else:
                z_lo = sched.top_z(sched.band_first_layer(b)) - self.layer_h
                t = np.clip(h - z_lo, 0.0, sched.layer_counts[b] * self.layer_h)
            c = np.clip(np.asarray(fils[b].rgb_lin, dtype=np.float64), 0.0, 1.0)
            td = max(float(fils[b].td_mm), 1e-6)
            T *= np.exp(-self.k_opaque * t[..., None] * (1.0 - c) / td)
        led = srgb_to_linear(np.asarray(hex_to_rgb01(self.led_hex), dtype=np.float64))
        return T * led * float(self.light_intensity)

    def preview(self, full: bool = False, lighting: Optional[str] = None) -> Image.Image:
        """Live view. ``lighting`` 'front' (filament painting) or 'back' (lithophane)."""
        mask = self.mask if full else self._mask_small
        mode = (lighting or self.lighting).lower()
        if mode.startswith('back'):
            lin = np.clip(self.transmitted(full=full), 0.0, 1.0)
            img = np.clip(linear_to_srgb(lin), 0.0, 1.0)
            img = np.where(np.asarray(mask, bool)[..., None], img, 0.05)
            return Image.fromarray((img * 255).astype(np.uint8))
        return render_preview(self.height_map(full=full), self.schedule,
                              self.library, self.model, mask=mask)

    # ----- reporting ------------------------------------------------------ #
    def describe(self) -> str:
        """HueForge's 'Describe' popup: the swap instructions for the slicer."""
        return self.schedule.swap_instructions(self.library)

    def surface_shares(self) -> Dict[str, float]:
        """Fraction of the visible surface whose top layer is each filament."""
        sched = self.schedule
        h = self.height_map(full=False)
        mask = self._mask_small
        if not np.any(mask):
            return {n: 0.0 for n in self.filament_names}
        n = np.clip(sched.layer_of_height(h[mask]), 1, sched.n_layers)
        band = sched.band_of_layer(n)
        total = float(band.size)
        return {name: float(np.isin(band, [i for i,n in enumerate(self.filament_names) if n == name]).sum()) / total
                for name in dict.fromkeys(self.filament_names)}

    def stats(self) -> dict:
        sched = self.schedule
        h = self.height_map(full=False)
        vis = h[self._mask_small] if np.any(self._mask_small) else np.zeros(1)
        return {
            'layers': sched.n_layers,
            'total_height_mm': round(float(sched.total_height_mm), 3),
            'swaps': sched.n_bands - 1,
            'height_range_mm': (round(float(vis.min()), 3), round(float(vis.max()), 3)),
            'surface_shares': {k: round(v, 4) for k, v in self.surface_shares().items()},
        }

    # ----- 3D viewport ---------------------------------------------------- #
    def print_mesh(self, mesh_pitch_mm: float = 0.15, max_dim: int | None = None):
        """One geometry path for STL, 3MF and the interactive detail levels."""
        from core.band.heightfield_mesh import heightfield_to_trimesh

        if not np.isfinite(mesh_pitch_mm) or mesh_pitch_mm <= 0:
            raise ValueError("mesh pitch must be finite and positive")
        h, mask = self.height_map(full=True), self.mask
        height_mm = self.width_mm * h.shape[0] / h.shape[1]
        w = max(2, min(h.shape[1], int(round(self.width_mm / mesh_pitch_mm)) + 1))
        rows = max(2, min(h.shape[0], int(round(height_mm / mesh_pitch_mm)) + 1))
        if max_dim is not None:
            scale = min(1.0, max(2, max_dim) / max(w, rows))
            w, rows = max(2, round(w * scale)), max(2, round(rows * scale))
        if (rows, w) != h.shape:
            h = cv2.resize(h.astype(np.float32), (w, rows), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (w, rows),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        floor = min(self.z_bounds()[0], self.layer_h)
        h = np.where(mask, np.maximum(h, floor), self.layer_h)
        pitch = self.width_mm / (w - 1)
        mesh = heightfield_to_trimesh(h, pitch, mask)
        mesh.vertices[:, 1] *= height_mm / ((rows - 1) * pitch)
        return mesh, pitch

    def mesh3d(self, max_dim: int = 400) -> dict:
        """Closed print geometry and a height-based optical colour ramp.

        The viewport samples the ramp per fragment, so colour changes follow
        physical Z rather than interpolation of colours across triangles.
        """
        from core.band.pipeline import composite_heights
        mesh, pitch = self.print_mesh(max_dim=max_dim)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        z_max = float(self.schedule.total_height_mm)
        zs = np.linspace(0.0, z_max, 2048)
        backlit = self.lighting.startswith('back')
        def predict(z):
            return (self.transmitted_heights(z) if backlit else
                    composite_heights(z, self.schedule, self.library, self.model))
        rgb = np.round(linear_to_srgb(predict(verts[:, 2])) * 255).astype(np.uint8)
        ramp = np.round(linear_to_srgb(predict(zs)) * 255).astype(np.uint8)
        return {'positions': verts, 'indices': np.asarray(mesh.faces, dtype=np.uint32),
                'colors': rgb, 'pitch_mm': pitch, 'size_mm': mesh.extents.tolist(),
                'ramp': ramp.tolist(), 'ramp_height_mm': z_max, 'backlit': backlit}

    # ----- optimiser ------------------------------------------------------ #
    def auto_thickness(self, max_layers: int = 27) -> List[int]:
        """Best per-band layer counts for the current filament order.

        HueForge's slider auto-arrange: the filament choice stays yours, only
        the thicknesses are searched.
        """
        from core.band.palette import image_hist_lab, optimise_thickness
        if self.mesh_mode == 'color_match' or len(self.layer_counts) > 5:
            # Fit print thickness to the fixed reference geometry rather than
            # running the luminance optimizer, which would undo Color Match.
            from core.band.pipeline import composite_heights
            target = self._rgb_small.astype(float)/255
            h = self.height_map(full=False)
            counts = list(self.layer_counts)
            def score(candidate):
                schedule = BandSchedule(tuple(self.filament_names), tuple(candidate),
                                        self.first_layer_mm, self.layer_h,
                                        min(self.base_min_layers, candidate[0]))
                predicted = linear_to_srgb(composite_heights(h, schedule, self.library, self.model))
                delta = srgb_to_lab_d65(predicted)-srgb_to_lab_d65(target)
                return float(np.mean(np.sum(delta[self._mask_small]**2, axis=-1)))
            best = score(counts)
            for _ in range(12):
                improved = False
                for i in range(len(counts)-1):
                    for step in (-1, 1):
                        candidate = counts.copy()
                        candidate[i] += step; candidate[i+1] -= step
                        if min(candidate) < 1:
                            continue
                        value = score(candidate)
                        if value < best - 1e-8:
                            counts, best, improved = candidate, value, True
                if not improved:
                    break
            return counts
        hist = image_hist_lab(self.image_path)
        sched = optimise_thickness(
            hist, list(self.filament_names), self.library, self.model,
            n0=int(self.layer_counts[0]), max_layers=max_layers,
            first_layer_mm=self.first_layer_mm, layer_h=self.layer_h,
            base_min_layers=min(self.base_min_layers, int(self.layer_counts[0])),
            tone=self.tone)
        return [int(c) for c in sched.layer_counts]

    # ----- export --------------------------------------------------------- #
    def library_json(self, path: Optional[str] = None) -> str:
        """Write the (possibly edited) library so the exporter sees current TDs."""
        path = path or os.path.join(tempfile.mkdtemp(prefix='band_lib_'), 'filaments.json')
        return save_filament_library(path, self.library)

    def export_stl(self, out_dir: str, mesh_pitch_mm: float = 0.15,
                   basename: Optional[str] = None) -> dict:
        """HueForge's native pair: a single-solid STL plus Describe.txt.

        The STL carries no colour - the swap instructions in the .txt are what
        you type into the slicer, exactly as HueForge does it.
        """
        from core.band.heightfield_mesh import heightfield_to_trimesh

        os.makedirs(out_dir, exist_ok=True)
        base = basename or os.path.splitext(os.path.basename(self.image_path))[0]

        mesh, _ = self.print_mesh(mesh_pitch_mm)
        stl_path = os.path.join(out_dir, f"{base}.stl")
        mesh.export(stl_path)

        txt_path = os.path.join(out_dir, f"{base}_describe.txt")
        with open(txt_path, 'w', encoding='utf-8') as fh:
            fh.write(self.describe())
            fh.write(f"\n\nModel: {base}.stl\n"
                     f"Size: {self.width_mm:.1f} mm wide, "
                     f"{self.schedule.total_height_mm:.2f} mm tall.\n"
                     f"Slice at layer height {self.layer_h} mm with a "
                     f"{self.first_layer_mm} mm first layer.\n")

        return {'stl': stl_path, 'describe_txt': txt_path,
                'triangles': int(len(mesh.faces)), 'watertight': bool(mesh.is_watertight)}

    def export(self, out_dir: str, mesh_pitch_mm: float = 0.15, **kw) -> dict:
        """Export the current edited geometry and its exact layer schedule."""
        from core.band.writer3mf import write_band_3mf
        mesh, _ = self.print_mesh(mesh_pitch_mm)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        preview_png = os.path.join(out_dir, f"{base}_band_preview.png")
        self.preview(full=True).save(preview_png)
        swaps_txt = os.path.join(out_dir, f"{base}_swaps.txt")
        with open(swaps_txt, 'w', encoding='utf-8') as fh:
            fh.write(self.describe())
        overrides = dict(kw.pop('print_overrides', None) or {})
        overrides.update(layer_height=str(self.layer_h),
                         initial_layer_print_height=str(self.first_layer_mm))
        # Temperature is a per-filament key, so it must be a list one entry
        # per spool in the 3MF. First-layer speed is deliberately left on the
        # template value - only the temperature is ours.
        n_spools = len(dict.fromkeys(self.filament_names))
        temp = [str(int(round(self.nozzle_temp_c)))] * n_spools
        overrides.setdefault('nozzle_temperature', temp)
        overrides.setdefault('nozzle_temperature_initial_layer', list(temp))
        # Bambu's speed keys are per-extruder lists; the template carries one
        # entry per extruder, so match its length rather than the spool count.
        overrides.setdefault('initial_layer_speed', str(int(round(self.first_layer_speed))))
        path = write_band_3mf(
            mesh, os.path.join(out_dir, f"{base}_band.3mf"), [self.library[n] for n in dict.fromkeys(self.filament_names)],
            self.schedule.swap_entries(self.library), title=base,
            description=self.describe(), size_mm=self.width_mm,
            thumbnail_png=preview_png, print_overrides=overrides, **kw)
        return {'threemf': path, 'preview_png': preview_png, 'swaps_txt': swaps_txt,
                'swap_instructions': self.describe(), 'stats': self.stats(),
                'watertight': bool(mesh.is_watertight), 'triangles': len(mesh.faces)}
