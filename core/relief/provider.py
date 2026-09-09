"""Geometry providers: original image -> depth (+ optional normals / mask).

Relief Stack5 keeps two independent branches:

* colour  - the ORIGINAL RGB album image through the unchanged Stack5 matcher;
* geometry - a :class:`GeometryProvider` that turns the same image (or a file
  derived from it) into a :class:`GeometryResult`.

Only geometry ever comes from a provider.  No provider output is used as a
colour source (AI-generated views change faces, text and colours).

Depth convention
----------------
``GeometryResult.depth`` is an (H, W) float array in the provider's own units.
``depth_convention`` says which way is "up":

* ``'higher-is-closer'`` (default) - larger values are closer to the viewer and
  are raised higher.  Imported grey maps (white = high), Depth Anything V2
  (relative inverse depth) and integrated normals use this.
* ``'higher-is-farther'`` - larger values are farther from the viewer (a camera
  z-buffer distance).  :meth:`GeometryResult.raised` negates these so the
  pipeline always works with "larger = raised".

The pipeline's ``invert`` option flips the result afterwards for maps that were
authored the other way round.  Normalisation to 0..1 happens in
:mod:`core.relief.depth_processing`, never here, so a provider may return any
finite range.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

HIGHER_IS_CLOSER = 'higher-is-closer'
HIGHER_IS_FARTHER = 'higher-is-farther'
DEPTH_CONVENTIONS = (HIGHER_IS_CLOSER, HIGHER_IS_FARTHER)

PROVIDER_NAMES = ('imported-depth', 'depth-anything', 'wonder3d')
DEFAULT_PROVIDER = 'imported-depth'

# Alpha below this (0..255) counts as background in a depth / normal image.
ALPHA_BACKGROUND = 10


class MissingDependencyError(RuntimeError):
    """An optional AI provider is unavailable (package or weights missing).

    The message always says what is missing and how to install it; the core
    imported-depth workflow never raises it.
    """


class GeometryError(ValueError):
    """A provider could not produce geometry from the given inputs."""


@dataclass
class GeometryResult:
    """Depth (+ optional normals / mask) for one image.

    depth:   (H, W) float32.  See the module docstring for the convention.
    normals: optional (H, W, 3) float32 unit normals in image space
             (x right, y DOWN the image, z toward the viewer).  Only used by
             :mod:`core.relief.normal_integration`.
    mask:    optional (H, W) bool, True = foreground / object.  Used to flatten
             the background and to protect silhouettes from slope limiting.
    """
    depth: np.ndarray
    normals: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    depth_convention: str = HIGHER_IS_CLOSER
    source: str = ''
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        d = np.asarray(self.depth, dtype=np.float32)
        if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 2:
            raise GeometryError(f"depth must be an (H>=2, W>=2) array, got shape {d.shape}")
        if not np.all(np.isfinite(d)):
            raise GeometryError("depth contains NaN / inf")
        self.depth = d
        if self.depth_convention not in DEPTH_CONVENTIONS:
            raise GeometryError(f"depth_convention must be one of {DEPTH_CONVENTIONS}")
        if self.normals is not None:
            n = np.asarray(self.normals, dtype=np.float32)
            if n.ndim != 3 or n.shape[2] != 3:
                raise GeometryError(f"normals must be (H, W, 3), got {n.shape}")
            self.normals = n
        if self.mask is not None:
            m = np.asarray(self.mask).astype(bool)
            if m.ndim != 2:
                raise GeometryError(f"mask must be (H, W), got {m.shape}")
            self.mask = m

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def raised(self) -> np.ndarray:
        """Depth as 'larger = raised' regardless of the provider's convention."""
        if self.depth_convention == HIGHER_IS_FARTHER:
            return -self.depth
        return self.depth

    def describe(self) -> dict:
        d = self.depth
        out = {'source': self.source, 'shape': [int(d.shape[0]), int(d.shape[1])],
               'depth_convention': self.depth_convention,
               'depth_min': float(d.min()), 'depth_max': float(d.max()),
               'has_normals': self.normals is not None, 'has_mask': self.mask is not None}
        if self.mask is not None:
            out['mask_share'] = float(self.mask.mean())
        out.update({k: v for k, v in self.meta.items() if isinstance(v, (str, int, float, bool, list))})
        return out


@runtime_checkable
class GeometryProvider(Protocol):
    """Common provider interface.  ``target_hw`` is the Stack5 output grid
    (H, W); providers MAY use it to pick a render resolution but the pipeline
    always resamples the result onto that grid itself."""
    name: str

    def generate(self, image_path: str, target_hw: Optional[tuple[int, int]] = None) -> GeometryResult: ...


# --------------------------------------------------------------------------- file loading
def _reduce_to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 2 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim == 3:
        a = a[..., 0]
    return a


def load_depth_file(path: str) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a depth image / array -> (depth float32 (H, W), mask or None).

    * 8-bit grey / RGB(A): 0..1 (RGB is converted to luminance)
    * 16-bit PNG / TIFF, grey or RGB(A): 0..1 (read with OpenCV at full depth)
    * 32-bit int: divided by 65535 when the maximum exceeds 255, else 255
    * float TIFF / .npy / .npz['depth' or the first non-mask array]: as stored
    * an alpha channel below ALPHA_BACKGROUND (or .npz['mask']) becomes the mask
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = _reduce_to_2d(np.load(path))
        return np.asarray(arr, dtype=np.float32), None
    if ext == '.npz':
        data = np.load(path)
        keys = [k for k in data.files if k not in ('mask', 'normals')]
        if not keys:
            raise GeometryError(f"{path}: no depth array (keys: {data.files}); expected 'depth'")
        key = 'depth' if 'depth' in keys else keys[0]
        mask = np.asarray(data['mask']).astype(bool) if 'mask' in data.files else None
        return np.asarray(_reduce_to_2d(data[key]), dtype=np.float32), mask
    mask = None
    arr = None
    if ext in ('.png', '.tif', '.tiff', '.webp', '.bmp', '.jpg', '.jpeg'):
        try:
            import cv2
            raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            raw = None
        if raw is not None and raw.dtype in (np.uint8, np.uint16, np.float32):
            if raw.ndim == 3 and raw.shape[2] == 4:
                alpha = raw[..., 3]
                mask = alpha >= (ALPHA_BACKGROUND * (257 if raw.dtype == np.uint16 else 1)
                                 if raw.dtype != np.float32 else ALPHA_BACKGROUND / 255.0)
                raw = raw[..., :3]
            if raw.ndim == 3:
                raw = cv2.cvtColor(np.ascontiguousarray(raw), cv2.COLOR_BGR2GRAY)
            if raw.dtype == np.uint16:
                arr = raw.astype(np.float32) / 65535.0
            elif raw.dtype == np.uint8:
                arr = raw.astype(np.float32) / 255.0
            else:
                arr = raw.astype(np.float32)
    if arr is None:
        from PIL import Image
        im = Image.open(path)
        mode = im.mode
        if mode in ('RGBA', 'LA', 'PA') or (mode == 'P' and 'transparency' in im.info):
            alpha = np.asarray(im.convert('RGBA'))[..., 3]
            mask = alpha >= ALPHA_BACKGROUND
        if mode.startswith('I;16'):
            arr = np.asarray(im).astype(np.float32) / 65535.0
        elif mode == 'I':
            arr = np.asarray(im).astype(np.float32)
            arr = arr / (65535.0 if float(arr.max()) > 255.0 else 255.0)
        elif mode == 'F':
            arr = np.asarray(im, dtype=np.float32)
        else:
            arr = np.asarray(im.convert('L'), dtype=np.float32) / 255.0
    arr = _reduce_to_2d(arr)
    return np.asarray(arr, dtype=np.float32), mask


def load_mask_file(path: str) -> np.ndarray:
    """Foreground mask image -> (H, W) bool.  Alpha (if any) wins over grey;
    otherwise grey > 50 % is foreground."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.asarray(np.load(path)).astype(bool)
    from PIL import Image
    im = Image.open(path)
    if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = np.asarray(im.convert('RGBA'))[..., 3]
        return alpha >= ALPHA_BACKGROUND
    return np.asarray(im.convert('L')) > 127


# --------------------------------------------------------------------------- registry
def get_provider(name: str, **options) -> GeometryProvider:
    """Instantiate a provider by name.  Heavy modules import lazily; a provider
    whose dependencies are missing raises MissingDependencyError from
    ``generate`` (not from here) with an install hint."""
    key = (name or DEFAULT_PROVIDER).strip().lower()
    if key == 'imported-depth':
        from core.relief.imported_depth import ImportedDepthProvider
        return ImportedDepthProvider(**options)
    if key == 'depth-anything':
        from core.relief.depth_anything import DepthAnythingProvider
        return DepthAnythingProvider(**options)
    if key == 'wonder3d':
        from core.relief.wonder3d import Wonder3DProvider
        return Wonder3DProvider(**options)
    raise ValueError(f"unknown geometry provider {name!r}; choose from {PROVIDER_NAMES}")
