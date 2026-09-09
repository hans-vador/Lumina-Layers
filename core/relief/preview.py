"""Preview images for Relief Stack5: the Stack5 predicted colour shaded by the
relief (Lambertian, light from the top-left) and a grey map of the quantised
relief steps.  Diagnostics only - never used as a colour source."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image


def shaded_preview(matched_rgb: np.ndarray, mask_solid: np.ndarray, steps: np.ndarray, layer_h: float,
                   pixel_mm: float, light=(-0.6, -0.6, 1.0), ambient: float = 0.55) -> Image.Image:
    rgb = np.asarray(matched_rgb, dtype=np.float64) / 255.0
    mask = np.asarray(mask_solid).astype(bool)
    h = np.asarray(steps, dtype=np.float64) * float(layer_h)
    gy, gx = np.gradient(h, float(pixel_mm))            # dz/dy (rows), dz/dx (cols) in mm/mm
    n = np.stack([-gx, -gy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    l = np.asarray(light, dtype=np.float64)
    l = l / np.linalg.norm(l)
    lam = np.clip(n @ l, 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * lam
    out = np.clip(rgb * shade[..., None], 0.0, 1.0)
    rgba = np.zeros(rgb.shape[:2] + (4,), dtype=np.uint8)
    rgba[..., :3] = np.round(out * 255).astype(np.uint8)
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')


def depth_preview(steps: np.ndarray, relief_layers: int, mask_solid: Optional[np.ndarray] = None) -> Image.Image:
    s = np.asarray(steps, dtype=np.float64)
    g = np.clip(s / max(int(relief_layers), 1), 0.0, 1.0)
    img = np.round(g * 255).astype(np.uint8)
    if mask_solid is None:
        return Image.fromarray(img, 'L')
    rgba = np.zeros(img.shape + (4,), np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = img
    rgba[..., 3] = np.where(np.asarray(mask_solid).astype(bool), 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')
