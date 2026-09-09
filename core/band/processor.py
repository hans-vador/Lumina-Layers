"""Band mode processor: image -> continuous heightfield through Lumina's pipeline.

Wraps :class:`core.image_processing.LuminaImageProcessor` with a ramp LUT
(``.npz`` with ``rgb (K',3) uint8`` and ``stacks (K',5) int32``) whose rows are
the K' = K-n0+1 tones of a :class:`core.band.schedule.BandSchedule`.

Two height sources are available (``height_mode``):

* ``'luminance'`` (default) - HueForge-style tone mapping: the column height is a
  smooth monotone function of the pixel's CIELAB lightness
  (:class:`core.band.tone.ToneCurve`, fitted to the professional reference),
  h = z_lo + (z_hi - z_lo) * curve(L*).  Sub-layer detail and base-band relief
  (engraving) are inherent; the printed layer follows from the slice-plane
  rule and the colour of that layer is the ramp row.  This reproduces the
  reference's surface distribution (every tonal layer carries 1-10 %).
* ``'match'`` - Lumina's Lab-KDTree nearest-row match (luminance-weighted), so
  ``material_matrix.sum(-1)`` is the printed-layer count k; height
  h = FL + lh*(k-1+f) with a sub-layer refinement f in (-0.5, 0.5) from the
  bilateral-filtered colour, plus optional base-band engraving by L*.

Both paths end with a light gaussian and an INTER_AREA resample to the mesh pitch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from config import ColorSystem, ModelingMode
from core.image_processing import LuminaImageProcessor
from core.band.optics import Filament, srgb_to_lab_d65
from core.band.ramp_lut import RampLUT
from core.band.schedule import BandSchedule
from core.band.tone import ToneCurve, engrave_floor

# Sub-layer refinement range.  |f| < 0.5 keeps a column strictly inside layer
# k's slice cell (slice plane at top_z(k) +- lh/2); the margin keeps vertices
# off the plane even after %.3f rounding (0.05*lh = 4 um >> 0.5 um).
SUBLAYER_MAX = 0.45
HEIGHT_MODES = ('luminance', 'match')


@dataclass
class BandResult:
    """Output of :meth:`BandProcessor.process`.

    All raster arrays share the same (H, W) shape at ``pitch_mm`` (the mesh
    pitch): ``k_map``/``matched_rgb``/``mask_solid`` are NEAREST-resampled from
    the 10 px/mm processing grid, ``height_mm`` is INTER_AREA-resampled.
    """
    height_mm: np.ndarray          # (H, W) float32, 0 outside mask_solid
    k_map: np.ndarray              # (H, W) int16, printed-layer count (>= n0 where solid, 0 elsewhere)
    matched_rgb: np.ndarray        # (H, W, 3) uint8 ramp colour of the matched row
    mask_solid: np.ndarray         # (H, W) bool
    pitch_mm: float
    stats: dict = field(default_factory=dict)


def band_mode_conf(schedule: BandSchedule, filaments: Dict[str, Filament]) -> dict:
    """Build the ColorSystem config dict for a Band schedule (5 slots, zero-padded)."""
    names = list(schedule.filament_names)
    slots = names + [f"unused{i}" for i in range(len(names), 5)]
    preview = {}
    cmap = {}
    for i, nm in enumerate(slots):
        if nm in filaments:
            h = filaments[nm].hex.lstrip('#')
            rgb = [int(h[j:j + 2], 16) for j in (0, 2, 4)]
        else:
            rgb = [128, 128, 128]
        preview[i] = rgb + [255]
        cmap[nm] = i
    return {
        'name': 'Band',
        'slots': slots,
        'preview': preview,
        'map': cmap,
        'layer_count': 5,
        'corner_labels': ['TL', 'TR', 'BR', 'BL'],
    }


def _hex_to_rgb01(hx: str) -> np.ndarray:
    h = hx.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0


class BandProcessor:
    """Drive LuminaImageProcessor with a ramp LUT and turn its output into a heightfield."""

    def __init__(self, ramp: RampLUT, schedule: BandSchedule, filaments: Dict[str, Filament],
                 ramp_npz_path: str, mode_key: str, engrave_floor_mm: Optional[float] = None):
        if not str(mode_key).startswith('Band'):
            raise ValueError(f"mode_key must start with 'Band' (got {mode_key!r})")
        if not os.path.exists(ramp_npz_path):
            raise FileNotFoundError(ramp_npz_path)
        self.ramp = ramp
        self.schedule = schedule
        self.filaments = filaments
        self.ramp_npz_path = ramp_npz_path
        self.mode_key = mode_key
        # Seam (2): make ColorSystem.get(mode_key) resolve to a Band config so
        # LuminaImageProcessor picks layer_count=5 and no substring heuristic fires.
        ColorSystem.register_dynamic(mode_key, band_mode_conf(schedule, filaments))

        self.n0 = int(schedule.layer_counts[0])
        self.K = int(schedule.n_layers)
        self.fl = float(schedule.first_layer_mm)
        self.lh = float(schedule.layer_h)
        k_vals = np.asarray(ramp.k_values).astype(int)
        if k_vals.shape[0] != len(schedule.k_rows()) or int(k_vals[0]) != self.n0:
            raise ValueError("ramp.k_values does not match schedule.k_rows()")
        self.k_values = k_vals
        # Engraving / relief floor.  Default = reference relief minimum 0.521 mm
        # (slice plane between layers base_min_layers and +1, plus 1e-3 so the
        # darkest pixels are unambiguously inside layer 6 - the reference has no
        # surface in layer 5).  The contract's top_z(base_min_layers) = 0.48 is
        # available via engrave_floor_mm=schedule.top_z(schedule.base_min_layers).
        z_floor = engrave_floor(schedule) if engrave_floor_mm is None else float(engrave_floor_mm)
        z_n0 = float(schedule.top_z(self.n0))
        if not (0.0 < z_floor <= z_n0):
            raise ValueError(f"engrave floor {z_floor} must be in (0, top_z(n0)={z_n0}]")
        self.z_floor = z_floor

    # ------------------------------------------------------------------ helpers
    def _top_z_array(self, k: np.ndarray) -> np.ndarray:
        return self.fl + self.lh * (k.astype(np.float64) - 1.0)

    @staticmethod
    def _install_luma_weight(proc: LuminaImageProcessor, weight: float) -> None:
        """Make the matcher luminance-dominant (used by height_mode='match').

        Lumina matches in OpenCV 8-bit Lab (L scaled x2.55, a/b offset 128) with
        a plain Euclidean KDTree.  A 1-D tonal ramp cannot reproduce hues that lie
        off it (a blue sky against a black/red/orange/white ramp), and with an
        unweighted metric such pixels are "chroma-attracted" to the least
        saturated row - the whitest one - regardless of their lightness, which
        piles a third of the image onto the top layer.  Multiplying the L axis by
        ``weight`` on BOTH sides of the query (LUT rows and image colours go
        through the same instance method) makes lightness decide and chroma
        break ties.  The instance attribute shadows the class staticmethod;
        nothing in core/image_processing.py changes.
        """
        from scipy.spatial import KDTree
        base = LuminaImageProcessor._rgb_to_lab
        w = float(weight)

        def weighted_lab(rgb_array):
            lab = base(rgb_array)
            lab[..., 0] *= w
            return lab

        proc._rgb_to_lab = weighted_lab
        proc.lut_lab = weighted_lab(proc.lut_rgb)
        proc.kdtree = KDTree(proc.lut_lab)
        if getattr(proc, 'hue_matcher', None) is not None:
            # hue-aware matching is the opposite intent; not used in band mode
            proc.hue_matcher = None

    def _sublayer_fraction(self, proc: LuminaImageProcessor, bilateral_rgb: np.ndarray,
                           row_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """f in [-SUBLAYER_MAX, SUBLAYER_MAX] per pixel: projection of the pixel
        colour onto the segment from its matched ramp row towards the adjacent
        ramp row that is its other nearest neighbour (2-NN in the matcher's Lab
        space).  The clip stays strictly inside the slice cell so no column can
        land on a slice plane (top_z(k) +- lh/2)."""
        H, W = row_map.shape
        f = np.zeros((H, W), dtype=np.float32)
        n_rows = int(proc.lut_lab.shape[0])
        if n_rows < 2 or not np.any(valid):
            return f
        flat_rgb = bilateral_rgb.reshape(-1, 3)
        vi = np.flatnonzero(valid.ravel())
        lab = proc._rgb_to_lab(flat_rgb[vi].astype(np.uint8))            # (N,3) OpenCV-scaled Lab
        _, idx2 = proc.kdtree.query(lab, k=2)                              # (N,2)
        r0 = row_map.ravel()[vi].astype(np.int64)
        # neighbour candidate: the nearest of the two that is not the anchor row
        cand = np.where(idx2[:, 0] == r0, idx2[:, 1], idx2[:, 0]).astype(np.int64)
        adjacent = np.abs(cand - r0) == 1
        # if the 2-NN pair does not contain an adjacent row, fall back to the
        # adjacent row that the pixel colour projects positively onto
        lut_lab = np.asarray(proc.lut_lab, dtype=np.float64)
        c0 = lut_lab[r0]
        fv = np.zeros(len(vi), dtype=np.float64)
        if np.any(adjacent):
            a = np.flatnonzero(adjacent)
            d = lut_lab[cand[a]] - c0[a]
            den = np.maximum((d * d).sum(1), 1e-9)
            t = ((lab[a] - c0[a]) * d).sum(1) / den
            fv[a] = np.clip(t, 0.0, SUBLAYER_MAX) * np.sign(cand[a] - r0[a])
        na = np.flatnonzero(~adjacent)
        if len(na):
            best = np.zeros(len(na))
            for step in (-1, 1):
                rj = r0[na] + step
                ok = (rj >= 0) & (rj < n_rows)
                if not np.any(ok):
                    continue
                d = lut_lab[np.clip(rj, 0, n_rows - 1)] - c0[na]
                den = np.maximum((d * d).sum(1), 1e-9)
                t = np.clip(((lab[na] - c0[na]) * d).sum(1) / den, 0.0, SUBLAYER_MAX) * ok
                take = t > np.abs(best)
                best = np.where(take, t * step, best)
            fv[na] = best
        f.ravel()[vi] = fv.astype(np.float32)
        return f

    def _engrave(self, h: np.ndarray, bilateral_rgb: np.ndarray, k_map: np.ndarray,
                 valid: np.ndarray) -> tuple[np.ndarray, dict]:
        """(height_mode='match') Base-tone pixels (k == n0) get relief between
        the engraving floor and top_z(n0) proportional to perceptual lightness L*
        relative to the brightest (95th pct) base-tone pixel."""
        base_mask = valid & (k_map == self.n0)
        info = {'engraved_pixels': int(base_mask.sum()), 'engrave_L_ref': None}
        if base_mask.sum() < 100:
            return h, info
        rgb01 = bilateral_rgb[base_mask].reshape(-1, 3).astype(np.float64) / 255.0
        L = srgb_to_lab_d65(rgb01)[:, 0]
        L_ref = float(np.percentile(L, 95))
        info['engrave_L_ref'] = L_ref
        if L_ref <= 1e-6:
            return h, info
        z_min = self.z_floor
        z_n0 = self.schedule.top_z(self.n0)
        rel = np.clip(L / L_ref, 0.0, 1.0)
        h = h.copy()
        h[base_mask] = (z_min + (z_n0 - z_min) * rel).astype(np.float32)
        return h, info

    def surface_fractions(self, height_mm: np.ndarray, mask: np.ndarray) -> dict:
        """Top-surface area fraction per printed layer and per filament.

        Slice-plane rule: a column of height h contains layer n (>= 2) iff
        h > top_z(n) - lh/2, and layer 1 iff h > FL/2; the layer a column *tops
        out in* is the highest layer it contains.  Works on any (H, W) grid, so
        it can be re-run on the final height field (e.g. after the advisory
        stamp) or at the mesh pitch.
        """
        valid = np.asarray(mask, bool)
        hv = np.asarray(height_mm, dtype=np.float64)[valid]
        n_solid = max(int(hv.size), 1)
        K = self.K
        contains = np.zeros(K + 2, dtype=np.float64)      # index n, 1..K ; K+1 = 0
        for n in range(1, K + 1):
            thr = (self.fl / 2.0) if n == 1 else (self.schedule.top_z(n) - self.lh / 2.0)
            contains[n] = float(np.count_nonzero(hv > thr)) / n_solid
        per_layer = {n: float(max(contains[n] - contains[n + 1], 0.0)) for n in range(1, K + 1)}
        per_filament = {}
        for b, name in enumerate(self.schedule.filament_names):
            lo = self.schedule.band_first_layer(b)
            hi = self.schedule.cum(b)
            per_filament[name] = float(sum(per_layer[n] for n in range(lo, hi + 1)))
        return {
            'top_surface_fraction_per_layer': per_layer,
            'top_surface_fraction_per_filament': per_filament,
            'height_min_mm': float(hv.min()) if hv.size else 0.0,
            'height_max_mm': float(hv.max()) if hv.size else 0.0,
            'solid_fraction': float(valid.mean()) if valid.size else 0.0,
        }

    def _plane_distance(self, hv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance of heights to their nearest slice plane
        (FL/2 and FL + lh*(n-1.5), n >= 2) -> (signed d, plane z)."""
        n = np.clip(np.round((hv - self.fl) / self.lh + 1.5), 2, self.K + 1)
        p = self.fl + self.lh * (n - 1.5)
        p0 = np.full_like(hv, self.fl / 2.0)
        use0 = np.abs(hv - p0) < np.abs(hv - p)
        p = np.where(use0, p0, p)
        return hv - p, p

    def nudge_off_planes(self, h: np.ndarray, valid: np.ndarray, z_lo: float, z_hi: float,
                         eps: float = 1e-3) -> np.ndarray:
        """Move any solid column closer than ``eps`` to a slice plane to exactly
        eps away from it (towards the side it was on), keeping [z_lo, z_hi].

        A continuous height map hits a plane by chance in ~0.25 % of columns
        and the writer's %.3f rounding would then put those vertices *on* the
        plane (planes are multiples of 0.04 mm).  eps = 1 um survives the
        rounding and is far below the printer's resolution.
        """
        hv = np.asarray(h, dtype=np.float64)
        d, p = self._plane_distance(hv)
        close = valid & (np.abs(d) < eps)
        if np.any(close):
            side = np.where(d >= 0, 1.0, -1.0)
            fixed = p + side * eps
            # never leave the allowed range (the floor is 1e-3 above a plane by construction)
            fixed = np.where(fixed < z_lo, p + eps, fixed)
            fixed = np.where(fixed > z_hi, p - eps, fixed)
            hv = np.where(close, fixed, hv)
        return hv.astype(np.float32)

    def slice_plane_hits(self, height_mm: np.ndarray, mask: np.ndarray, tol: float = 1e-4) -> int:
        """Number of solid columns whose height lies within ``tol`` of a slice
        plane top_z(n) - lh/2 (n >= 2) or FL/2 - float-ambiguous layer membership."""
        hv = np.asarray(height_mm, dtype=np.float64)[np.asarray(mask, bool)]
        if hv.size == 0:
            return 0
        d, _ = self._plane_distance(hv)
        return int(np.count_nonzero(np.abs(d) <= tol))

    def _stats(self, h_full: np.ndarray, valid: np.ndarray, k_map: np.ndarray,
               matched_rgb: np.ndarray, bilateral_rgb: np.ndarray) -> dict:
        """Surface fractions + colour-match error (true CIELAB: full dE and the
        lightness-only |dL*|, the quantity a tonal ramp can actually control)."""
        K = self.K
        n_solid = max(int(np.count_nonzero(valid)), 1)
        out = self.surface_fractions(h_full, valid)
        vi = np.flatnonzero(valid.ravel())
        if vi.size > 200_000:
            rng = np.random.default_rng(0)
            vi = rng.choice(vi, 200_000, replace=False)
        src = bilateral_rgb.reshape(-1, 3)[vi].astype(np.float64) / 255.0
        dst = matched_rgb.reshape(-1, 3)[vi].astype(np.float64) / 255.0
        if vi.size:
            diff = srgb_to_lab_d65(src) - srgb_to_lab_d65(dst)
            dE = np.linalg.norm(diff, axis=1)
            dL = np.abs(diff[:, 0])
        else:
            dE = dL = np.zeros(0)
        k_hist = {int(k): float(np.count_nonzero(k_map[valid] == k)) / n_solid
                  for k in np.unique(k_map[valid])} if np.any(valid) else {}
        out.update({
            'n_layers': K,
            'n0': self.n0,
            'first_layer_mm': self.fl,
            'layer_h': self.lh,
            'k_map_fraction': k_hist,
            'mean_dE': float(dE.mean()) if dE.size else 0.0,
            'p95_dE': float(np.percentile(dE, 95)) if dE.size else 0.0,
            'mean_dL': float(dL.mean()) if dL.size else 0.0,
            'p95_dL': float(np.percentile(dL, 95)) if dL.size else 0.0,
            'tool_changes': int(self.schedule.n_bands - 1),
            'slice_plane_hits': self.slice_plane_hits(h_full, valid),
        })
        return out

    # ------------------------------------------------------------------ main
    def process(self, image_path: str, width_mm: float, quantize_colors: int = 256,
                smooth_sigma: float = 10, interpolate: bool = True, engrave: bool = True,
                gaussian_px: float = 0.5, mesh_pitch_mm: float = 0.15,
                luma_weight: float = 5.0, height_mode: str = 'luminance',
                tone: Optional[ToneCurve] = None) -> BandResult:
        """Run Lumina's image pipeline with the ramp LUT and build the height field.

        height_mode: 'luminance' (default) maps CIELAB L* of the bilateral-
            filtered pixel to height through ``tone`` (default: the curve fitted
            to the HueForge reference); 'match' uses Lumina's nearest-row match.
        tone: ToneCurve for 'luminance' mode (None = ToneCurve.reference()).
        interpolate: 'match' - sub-layer refinement f in (-0.5, 0.5);
            'luminance' - False snaps tonal columns to mid-cell heights top_z(k).
        engrave: relief inside the base band (floor = engraving floor) for the
            darkest pixels; False clamps the base band flat at top_z(n0).
        luma_weight: ('match' only) multiplier on the L axis of the matcher's
            (OpenCV) Lab space, applied to LUT rows and image colours alike.
            1.0 = Lumina's plain Lab distance; 5.0 = luminance-dominant.
        quantize_colors: K-Means colours of Lumina's posterisation ('match'
            only - 'luminance' works on the un-quantised bilateral image and
            runs the mandatory K-Means with a small k to save ~20 s).
        """
        if height_mode not in HEIGHT_MODES:
            raise ValueError(f"height_mode must be one of {HEIGHT_MODES}, got {height_mode!r}")
        tone = ToneCurve.from_any(tone)
        t0 = time.time()
        proc = LuminaImageProcessor(self.ramp_npz_path, self.mode_key)
        if proc.lut_rgb.shape[0] != len(self.k_values):
            raise ValueError("ramp npz row count does not match schedule")
        if luma_weight is not None and abs(float(luma_weight) - 1.0) > 1e-9:
            if float(luma_weight) <= 0:
                raise ValueError("luma_weight must be > 0")
            self._install_luma_weight(proc, float(luma_weight))

        q_eff = int(quantize_colors) if height_mode == 'match' else int(min(int(quantize_colors), 16))
        r = proc.process_image(
            image_path, width_mm, ModelingMode.HIGH_FIDELITY, q_eff,
            auto_bg=False, bg_tol=0, blur_kernel=0, smooth_sigma=smooth_sigma,
            resample=Image.Resampling.LANCZOS,
        )
        material = np.asarray(r['material_matrix'])
        matched_rgb = np.asarray(r['matched_rgb']).astype(np.uint8)
        mask_solid = np.asarray(r['mask_solid']).astype(bool)
        pixel_scale = float(r.get('pixel_scale', 0.1))
        H, W = mask_solid.shape

        # bilateral-filtered image: prefer debug_data, else re-filter the quantised image ourselves
        dbg = r.get('debug_data') or {}
        bilateral = dbg.get('bilateral_filtered')
        if bilateral is None or np.asarray(bilateral).shape != (H, W, 3):
            src = np.asarray(r.get('quantized_image', matched_rgb)).astype(np.uint8)
            if smooth_sigma > 0:
                bilateral = cv2.bilateralFilter(src, d=9, sigmaColor=float(smooth_sigma),
                                                sigmaSpace=float(smooth_sigma))
            else:
                bilateral = src
        bilateral = np.asarray(bilateral).astype(np.uint8)

        z_n0 = float(self.schedule.top_z(self.n0))
        z_lo = self.z_floor if engrave else z_n0
        z_hi = float(self.schedule.top_z(self.K))          # mid-cell of the last layer
        interp_info = {'sublayer_mean_abs_f': 0.0, 'sublayer_nonzero_fraction': 0.0}
        engrave_info: dict = {}

        if height_mode == 'luminance':
            valid = mask_solid.copy()
            L = np.zeros((H, W), dtype=np.float64)
            if np.any(valid):
                L[valid] = srgb_to_lab_d65(bilateral[valid].reshape(-1, 3).astype(np.float64) / 255.0)[:, 0]
            h = tone.height(L, z_lo, z_hi).astype(np.float32)
            k_map = np.clip(self.schedule.layer_of_height(h), self.n0, self.K).astype(np.int32)
            k_map[~valid] = 0
            row_map = np.clip(k_map - self.n0, 0, len(self.k_values) - 1)
            matched_rgb = np.asarray(self.ramp.rgb, dtype=np.uint8)[row_map]
            matched_rgb[~valid] = 0
            if not interpolate:
                # snap tonal columns to mid-cell heights; keep the base relief if engraving
                snapped = self._top_z_array(k_map.astype(np.float64))
                keep_relief = engrave & (k_map == self.n0)
                h = np.where(keep_relief, h, snapped).astype(np.float32)
            base_mask = valid & (k_map == self.n0)
            engrave_info = {'engraved_pixels': int(base_mask.sum()) if engrave else 0,
                            'engrave_L_ref': float(100.0 * (tone.q_lo + (tone.q_hi - tone.q_lo) *
                                                            ((z_n0 - z_lo) / max(z_hi - z_lo, 1e-9)) ** (1.0 / tone.gamma)))
                            if engrave else None}
            tonal = valid & (k_map > self.n0)          # base-band relief is engraving, not sub-layer detail
            if np.any(tonal):
                fa = np.abs((h[tonal] - self._top_z_array(k_map[tonal].astype(np.float64))) / self.lh)
                interp_info = {'sublayer_mean_abs_f': float(fa.mean()),
                               'sublayer_nonzero_fraction': float((fa > 1e-6).mean())}
        else:
            # k_map: sum of per-band layer counts; transparent pixels carry -1 in every slot
            k_map = material.sum(axis=-1).astype(np.int32)
            k_map[~mask_solid] = 0
            valid = mask_solid & (k_map >= self.n0)
            k_map[~valid] = 0
            row_map = np.clip(k_map - self.n0, 0, len(self.k_values) - 1)

            kf = np.where(valid, k_map, self.n0).astype(np.float64)
            h = (self.fl + self.lh * (kf - 1.0)).astype(np.float32)
            if interpolate:
                f = self._sublayer_fraction(proc, bilateral, row_map, valid)
                h = (h + self.lh * f).astype(np.float32)
                if np.any(valid):
                    fa = np.abs(f[valid])
                    interp_info = {'sublayer_mean_abs_f': float(fa.mean()),
                                   'sublayer_nonzero_fraction': float((fa > 1e-6).mean())}
            if engrave:
                h, engrave_info = self._engrave(h, bilateral, k_map, valid)
        h = np.where(valid, h, 0.0).astype(np.float32)

        if gaussian_px and gaussian_px > 0:
            # normalised (masked) gaussian so the mask edge does not bleed zeros inward
            num = gaussian_filter(h * valid, float(gaussian_px))
            den = gaussian_filter(valid.astype(np.float32), float(gaussian_px))
            h = np.where(valid, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)

        # column heights must stay inside the schedule's range: [floor, mid-cell of layer K]
        # and off every slice plane (unambiguous layer membership after %.3f rounding)
        h = np.where(valid, np.clip(h, z_lo, z_hi), 0.0).astype(np.float32)
        h = np.where(valid, self.nudge_off_planes(h, valid, z_lo, z_hi), 0.0).astype(np.float32)

        stats = self._stats(h, valid, k_map, matched_rgb, bilateral)
        stats.update(engrave_info)
        stats.update(interp_info)
        stats['height_mode'] = height_mode
        stats['tone_curve'] = tone.to_dict() if height_mode == 'luminance' else None
        stats['luma_weight'] = float(luma_weight) if luma_weight is not None else 1.0
        stats['quantize_colors'] = int(q_eff)
        stats['engrave_floor_mm'] = float(self.z_floor)
        stats['process_px'] = (int(W), int(H))
        stats['process_pitch_mm'] = pixel_scale

        # resample to mesh pitch
        pitch = pixel_scale
        if mesh_pitch_mm and mesh_pitch_mm > pixel_scale * 1.001:
            new_w = max(2, int(round(W * pixel_scale / mesh_pitch_mm)))
            new_h = max(2, int(round(H * pixel_scale / mesh_pitch_mm)))
            h_rs = cv2.resize(h, (new_w, new_h), interpolation=cv2.INTER_AREA)
            m_rs = cv2.resize(valid.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA) > 0.5
            k_rs = cv2.resize(k_map.astype(np.int16), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            rgb_rs = cv2.resize(matched_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # INTER_AREA over the mask edge averages in zeros: renormalise by mask coverage
            cov = cv2.resize(valid.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
            h_rs = np.where(m_rs, h_rs / np.maximum(cov, 1e-6), 0.0).astype(np.float32)
            h_rs = np.where(m_rs, np.clip(h_rs, z_lo, z_hi), 0.0).astype(np.float32)
            h_rs = np.where(m_rs, self.nudge_off_planes(h_rs, m_rs, z_lo, z_hi), 0.0).astype(np.float32)
            h, valid, k_map, matched_rgb = h_rs, m_rs, k_rs, rgb_rs
            stats['slice_plane_hits_mesh'] = self.slice_plane_hits(h, valid)
            # exact physical width: (W'-1) * pitch == width_mm
            pitch = float(width_mm) / max(new_w - 1, 1)
        else:
            pitch = float(width_mm) / max(W - 1, 1)

        stats['mesh_px'] = (int(h.shape[1]), int(h.shape[0]))
        stats['mesh_pitch_mm'] = pitch
        stats['process_seconds'] = time.time() - t0
        k_map[~valid] = 0
        return BandResult(
            height_mm=h.astype(np.float32),
            k_map=k_map.astype(np.int16),
            matched_rgb=matched_rgb.astype(np.uint8),
            mask_solid=valid.astype(bool),
            pitch_mm=pitch,
            stats=stats,
        )
