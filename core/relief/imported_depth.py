"""Milestone 1 provider: a user-supplied grey depth map.

Convention: black = lowest, white = highest ('higher-is-closer').  Pass
``invert=True`` to the depth processing (CLI ``--invert-depth``) for maps
authored the other way round (e.g. Lumina's own heightmap convention, where
black = tallest).

Accepted files: 8/16-bit PNG, JPEG, TIFF (incl. float), WebP, .npy, .npz.
An alpha channel becomes the foreground mask; an explicit ``mask`` file or a
normal map (RGB, x right / y down / z out, see normal_integration) may be added.
The map may have any resolution or aspect ratio - the pipeline resamples it onto
the Stack5 grid (and records an aspect-ratio warning when they differ).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from core.relief.provider import (HIGHER_IS_CLOSER, GeometryError, GeometryResult, load_depth_file,
                                  load_mask_file)


class ImportedDepthProvider:
    name = 'imported-depth'

    def __init__(self, depth_map: Optional[str] = None, mask: Optional[str] = None,
                 normals: Optional[str] = None, normals_y_up: bool = False, **_ignored):
        if not depth_map:
            raise GeometryError("imported-depth needs a depth map file (--depth-map)")
        self.depth_map = depth_map
        self.mask_path = mask
        self.normals_path = normals
        self.normals_y_up = bool(normals_y_up)

    def preflight(self) -> None:
        """Cheap input checks before the (slow) colour stage."""
        for label, path in (('depth map', self.depth_map), ('mask', self.mask_path), ('normal map', self.normals_path)):
            if path and not os.path.isfile(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

    def generate(self, image_path: str, target_hw: Optional[tuple[int, int]] = None) -> GeometryResult:
        depth, alpha_mask = load_depth_file(self.depth_map)
        mask = alpha_mask
        if self.mask_path:
            m = load_mask_file(self.mask_path)
            if m.shape != depth.shape:
                import cv2
                m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            mask = m
        normals = None
        if self.normals_path:
            from core.relief.normal_integration import load_normal_map
            normals = load_normal_map(self.normals_path, y_up=self.normals_y_up)
            if normals.shape[:2] != depth.shape:
                import cv2
                normals = cv2.resize(normals, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        meta = {'depth_map': os.path.abspath(self.depth_map),
                'native_convention': 'black = lowest, white = highest',
                'mask_from_alpha': bool(alpha_mask is not None and not self.mask_path)}
        if self.mask_path:
            meta['mask_file'] = os.path.abspath(self.mask_path)
        if self.normals_path:
            meta['normals_file'] = os.path.abspath(self.normals_path)
        return GeometryResult(depth=depth, normals=normals, mask=mask, depth_convention=HIGHER_IS_CLOSER,
                              source=self.name, meta=meta)
