"""Object masks for art-directed relief: click-to-select with a local SAM
(Segment Anything) and manual brush strokes.

* :class:`SamMaskProvider` wraps Hugging Face ``transformers`` ``SamModel`` /
  ``SamProcessor``.  It is OPTIONAL and LAZY: nothing is imported or loaded until
  :meth:`SamMaskProvider.load` / :meth:`set_image`, and nothing is downloaded
  unless ``allow_download=True``.  Without the packages or the cached weights
  :meth:`available` returns ``(False, message)`` with the exact command to run;
  the brush keeps working in the meantime.
* Brush helpers paint discs / strokes into a bool mask on the relief grid.

The image embedding is computed once per image (the slow part, a few seconds
on CPU for ViT-B); every click then only runs the light mask decoder with all
accumulated positive / negative points.  SAM returns three candidate masks per
prompt (roughly part / object / group); :class:`MaskProposal` keeps them all so
the GUI can offer 'smaller / larger'.

Default checkpoint: ``facebook/sam-vit-base`` (~375 MB).  ``Zigeng/SlimSAM-uniform-77``
(~39 MB, same architecture, lower quality) loads with the same code through
``weights=``.  Masks come back at the ORIGINAL image resolution; the caller
resamples them onto the relief grid (core.relief.art_direct.resample_mask).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from core.relief.provider import MissingDependencyError

DEFAULT_SAM_MODEL_ID = 'facebook/sam-vit-base'
LIGHT_SAM_MODEL_ID = 'Zigeng/SlimSAM-uniform-77'
SAM_INSTALL_HINT = ("pip install torch transformers   (already present when Depth Anything works)\n"
                    f"then run  scripts/convert_relief.py --fetch-sam-model  once (about 375 MB, '{DEFAULT_SAM_MODEL_ID}'),\n"
                    f"or  --fetch-sam-model --sam-weights {LIGHT_SAM_MODEL_ID}  for the 39 MB light model,\n"
                    "or pass a local checkpoint directory with --sam-weights.  The brush works without SAM.")


def sam_dependency_status() -> dict:
    st = {'torch': False, 'transformers': False}
    try:
        import torch  # noqa: F401
        st['torch'] = True
    except Exception:
        pass
    try:
        import transformers  # noqa: F401
        st['transformers'] = True
    except Exception:
        pass
    return st


def sam_weights_cached(model_id: str = DEFAULT_SAM_MODEL_ID) -> bool:
    from core.relief.depth_anything import weights_cached
    return weights_cached(model_id)


def sam_status(model_id: str = DEFAULT_SAM_MODEL_ID) -> tuple[bool, str]:
    """(ready, human message) without loading anything."""
    st = sam_dependency_status()
    missing = [k for k, ok in st.items() if not ok]
    if missing:
        return False, f"SAM click-to-select needs the optional packages {missing}.\n{SAM_INSTALL_HINT}"
    w = str(model_id)
    if os.path.isdir(os.path.expanduser(w)):
        return True, f"SAM weights: local directory {w}"
    if not sam_weights_cached(w):
        return False, (f"SAM weights '{w}' are not downloaded yet (nothing is fetched automatically).\n"
                       f"In Terminal:  cd ~/Lumina-Layers && .venv/bin/python scripts/convert_relief.py --fetch-sam-model"
                       + ("" if w == DEFAULT_SAM_MODEL_ID else f" --sam-weights {w}")
                       + "\nUntil then use the brush to paint the selection.")
    return True, f"SAM ready: {w} (cached)"


def fetch_sam_weights(model_id: str = DEFAULT_SAM_MODEL_ID) -> str:
    """Download (once) the SAM processor + model into the Hugging Face cache.
    Explicit user action only - never called implicitly."""
    st = sam_dependency_status()
    missing = [k for k, ok in st.items() if not ok]
    if missing:
        raise MissingDependencyError(f"SAM needs the optional packages {missing}.\n{SAM_INSTALL_HINT}")
    from transformers import SamModel, SamProcessor
    SamProcessor.from_pretrained(model_id)
    SamModel.from_pretrained(model_id)
    return str(model_id)


# --------------------------------------------------------------------------- proposals
@dataclass
class MaskProposal:
    """Candidate masks for one prompt, all at the image resolution the provider saw."""
    masks: np.ndarray                     # (K, H, W) bool
    scores: np.ndarray                    # (K,) predicted IoU / quality
    points: list = field(default_factory=list)   # [(x, y), ...] in image pixels
    labels: list = field(default_factory=list)   # 1 = object, 0 = background
    source: str = ''

    @property
    def best(self) -> int:
        return int(np.argmax(self.scores)) if len(self.scores) else 0

    def by_size(self) -> list[int]:
        """Candidate indices from the smallest to the largest area."""
        return [int(i) for i in np.argsort(self.masks.reshape(len(self.masks), -1).sum(axis=1), kind='stable')]

    def choose(self, which='best') -> np.ndarray:
        """'best' (highest score), 'small' / 'medium' / 'large' (by area) or an index."""
        if isinstance(which, str):
            if which == 'best':
                return self.masks[self.best]
            order = self.by_size()
            pick = {'small': order[0], 'medium': order[len(order) // 2], 'large': order[-1]}
            if which not in pick:
                raise ValueError("which must be 'best', 'small', 'medium', 'large' or an index")
            return self.masks[pick[which]]
        return self.masks[int(which)]


@runtime_checkable
class MaskProvider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...
    def set_image(self, rgb: np.ndarray) -> None: ...
    def predict(self, points_xy: Sequence[Sequence[float]], labels: Sequence[int]) -> MaskProposal: ...


# --------------------------------------------------------------------------- SAM
class SamMaskProvider:
    name = 'sam'

    def __init__(self, weights: Optional[str] = None, device: str = 'auto', allow_download: bool = False,
                 model=None, processor=None):
        self.weights = weights or DEFAULT_SAM_MODEL_ID
        self.device = device
        self.allow_download = bool(allow_download)
        self._model = model
        self._processor = processor
        self._device = 'cpu'
        self._emb = None
        self._sizes = None
        self.image_hw: Optional[tuple[int, int]] = None

    # -- status -------------------------------------------------------------
    def available(self) -> tuple[bool, str]:
        if self._model is not None and self._processor is not None:
            return True, 'SAM ready (preloaded model)'
        if self.allow_download and all(sam_dependency_status().values()):
            return True, f"SAM: {self.weights} (download allowed)"
        return sam_status(self.weights)

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def image_set(self) -> bool:
        return self._emb is not None

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        ok, msg = self.available()
        if not ok:
            raise MissingDependencyError(msg)
        import torch
        from transformers import SamModel, SamProcessor
        kw = {'local_files_only': not self.allow_download}
        try:
            self._processor = SamProcessor.from_pretrained(self.weights, **kw)
            self._model = SamModel.from_pretrained(self.weights, **kw)
        except Exception as e:  # OSError / hub errors
            raise MissingDependencyError(f"could not load SAM weights '{self.weights}' ({e.__class__.__name__}: {e}).\n"
                                         f"{SAM_INSTALL_HINT}") from e
        self._model.eval()
        self._pick_device(torch)

    def _pick_device(self, torch) -> None:
        if self.device == 'auto':
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            dev = self.device
        try:
            self._model.to(dev)
        except Exception:
            dev = 'cpu'
            self._model.to(dev)
        self._device = dev

    # -- inference ----------------------------------------------------------
    def set_image(self, rgb: np.ndarray) -> None:
        """Compute the image embedding once (RGB uint8 (H, W, 3))."""
        self.load()
        import torch
        arr = np.asarray(rgb)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("rgb must be (H, W, 3) uint8")
        arr = np.ascontiguousarray(arr[..., :3].astype(np.uint8))
        if self._device == 'cpu':
            torch.manual_seed(0)
        inputs = self._processor(images=arr, return_tensors='pt')
        with torch.no_grad():
            self._emb = self._model.get_image_embeddings(inputs['pixel_values'].to(self._device))
        self._sizes = (inputs['original_sizes'], inputs['reshaped_input_sizes'])
        self.image_hw = (int(arr.shape[0]), int(arr.shape[1]))

    def predict(self, points_xy: Sequence[Sequence[float]], labels: Sequence[int]) -> MaskProposal:
        """All points prompt ONE object (labels 1 = inside, 0 = not this)."""
        if self._emb is None:
            raise RuntimeError("call set_image() before predict()")
        pts = [[float(p[0]), float(p[1])] for p in points_xy]
        labs = [int(l) for l in labels]
        if not pts or len(pts) != len(labs):
            raise ValueError("need one label per point and at least one point")
        import torch
        H, W = self.image_hw
        dummy = np.zeros((H, W, 3), np.uint8)   # only used by the processor to scale the points
        inputs = self._processor(images=dummy, input_points=[[pts]], input_labels=[[labs]], return_tensors='pt')
        with torch.no_grad():
            out = self._model(input_points=inputs['input_points'].to(self._device),
                              input_labels=inputs['input_labels'].to(self._device),
                              image_embeddings=self._emb, multimask_output=True)
        masks = self._processor.image_processor.post_process_masks(out.pred_masks.cpu(), inputs['original_sizes'],
                                                                   inputs['reshaped_input_sizes'])
        m = np.asarray(masks[0][0].numpy()).astype(bool)          # (K, H, W)
        scores = np.asarray(out.iou_scores.detach().cpu().numpy()).reshape(-1)[:len(m)].astype(np.float64)
        return MaskProposal(masks=m, scores=scores, points=[tuple(p) for p in pts], labels=labs, source=self.name)


# --------------------------------------------------------------------------- brush
def paint_disc(mask: np.ndarray, x: float, y: float, radius_px: float, value: bool = True) -> np.ndarray:
    """In-place disc into a bool (H, W) mask (centre in grid pixels)."""
    import cv2
    m = np.asarray(mask)
    if m.dtype != bool or m.ndim != 2:
        raise ValueError("mask must be a bool (H, W) array")
    r = max(int(round(float(radius_px))), 0)
    buf = m.view(np.uint8)
    cv2.circle(buf, (int(round(x)), int(round(y))), r, 1 if value else 0, thickness=-1, lineType=cv2.LINE_8)
    return m


def paint_stroke(mask: np.ndarray, p0: Sequence[float], p1: Sequence[float], radius_px: float,
                 value: bool = True) -> np.ndarray:
    """In-place capsule (line with round caps) between two grid points."""
    import cv2
    m = np.asarray(mask)
    if m.dtype != bool or m.ndim != 2:
        raise ValueError("mask must be a bool (H, W) array")
    r = max(int(round(float(radius_px))), 0)
    buf = m.view(np.uint8)
    a = (int(round(p0[0])), int(round(p0[1])))
    b = (int(round(p1[0])), int(round(p1[1])))
    cv2.line(buf, a, b, 1 if value else 0, thickness=max(2 * r, 1), lineType=cv2.LINE_8)
    cv2.circle(buf, a, r, 1 if value else 0, thickness=-1, lineType=cv2.LINE_8)
    cv2.circle(buf, b, r, 1 if value else 0, thickness=-1, lineType=cv2.LINE_8)
    return m


def component_containing(mask: np.ndarray, x: float, y: float) -> np.ndarray:
    """The connected component of ``mask`` that contains grid pixel (x, y); the
    whole mask when the point is outside it."""
    import cv2
    m = np.asarray(mask).astype(bool)
    xi, yi = int(round(x)), int(round(y))
    if not (0 <= yi < m.shape[0] and 0 <= xi < m.shape[1]) or not m[yi, xi]:
        return m.copy()
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    return lab == lab[yi, xi]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return ndimage.binary_fill_holes(np.asarray(mask).astype(bool))
