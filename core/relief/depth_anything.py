"""Optional geometry provider: Depth Anything V2 (relative monocular depth).

    original image -> Depth Anything V2 -> relative depth -> common relief processing

The model is loaded through Hugging Face ``transformers`` (AutoModelForDepthEstimation).
Nothing is downloaded unless ``allow_download=True`` (CLI ``--allow-download``):
by default the weights must already be in the local HF cache or given as a
directory with ``weights=``.  Missing packages or weights raise
:class:`MissingDependencyError` with the exact install / download command.

Output convention: Depth Anything V2 predicts relative INVERSE depth - larger
values are CLOSER to the camera - so the result is 'higher-is-closer' and needs
no flip.  The prediction is resampled to the original image size (bicubic) so
it stays aligned with the artwork; the pipeline then resamples it again onto
the Stack5 grid.  The model's colourised visualisation is never produced or
used - only the float depth.

Model choice: the -Small-hf checkpoint (~100 MB) is plenty for a 4 mm relief;
-Base-hf / -Large-hf work with the same code (``weights=`` the model id).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from core.relief.provider import HIGHER_IS_CLOSER, HIGHER_IS_FARTHER, GeometryResult, MissingDependencyError

DEFAULT_MODEL_ID = 'depth-anything/Depth-Anything-V2-Small-hf'
INSTALL_HINT = ("pip install torch torchvision transformers   (CPU is fine; ~1-2 GB of packages)\n"
                "then either pass --allow-download once to fetch the ~100 MB checkpoint\n"
                f"'{DEFAULT_MODEL_ID}' into the Hugging Face cache, or --weights /path/to/checkpoint-dir")


def weights_cached(model_id: str = DEFAULT_MODEL_ID) -> bool:
    """True when the checkpoint is a local directory or already in the Hugging Face
    cache (no network access)."""
    if os.path.isdir(str(model_id)):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        got = try_to_load_from_cache(str(model_id), 'config.json')
    except Exception:
        return False
    return isinstance(got, str) and os.path.isfile(got)


def fetch_weights(model_id: str = DEFAULT_MODEL_ID) -> str:
    """Download (once) the processor + model files into the Hugging Face cache and
    return the model id.  Explicit user action only - never called implicitly."""
    st = dependency_status()
    missing = [k for k, ok in st.items() if not ok]
    if missing:
        raise MissingDependencyError(
            f"Depth Anything V2 needs the optional packages {missing}, which are not installed.\n" + INSTALL_HINT)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    AutoImageProcessor.from_pretrained(model_id)
    AutoModelForDepthEstimation.from_pretrained(model_id)
    return str(model_id)


def dependency_status() -> dict:
    """Which optional pieces are importable (no model loading, no downloads)."""
    st = {'torch': False, 'torchvision': False, 'transformers': False}
    try:
        import torch  # noqa: F401
        st['torch'] = True
    except Exception:
        pass
    try:
        import torchvision  # noqa: F401
        st['torchvision'] = True
    except Exception:
        pass
    try:
        import transformers  # noqa: F401
        st['transformers'] = True
    except Exception:
        pass
    return st


class DepthAnythingProvider:
    name = 'depth-anything'

    def __init__(self, weights: Optional[str] = None, device: str = 'auto', allow_download: bool = False,
                 max_side: int = 1036, depth_convention: Optional[str] = None, **_ignored):
        self.weights = weights or DEFAULT_MODEL_ID
        self.device = device
        self.allow_download = bool(allow_download)
        self.max_side = int(max_side)
        # relative checkpoints predict inverse depth (larger = closer); the *-Metric-* checkpoints
        # predict metres (larger = farther) and are flipped unless told otherwise
        if depth_convention is None:
            depth_convention = HIGHER_IS_FARTHER if 'metric' in str(self.weights).lower() else HIGHER_IS_CLOSER
        if depth_convention not in (HIGHER_IS_CLOSER, HIGHER_IS_FARTHER):
            raise ValueError("depth_convention must be 'higher-is-closer' or 'higher-is-farther'")
        self.depth_convention = depth_convention
        self._model = None
        self._processor = None

    def preflight(self) -> None:
        """Fail before the colour stage when the optional packages are missing or a
        weights DIRECTORY does not exist (hub cache lookups happen in generate)."""
        st = dependency_status()
        missing = [k for k, ok in st.items() if not ok]
        if missing:
            raise MissingDependencyError(
                f"Depth Anything V2 needs the optional packages {missing}, which are not installed.\n" + INSTALL_HINT)
        w = str(self.weights)
        looks_like_path = w.startswith(('/', '.', '~')) or w.count('/') > 1 or os.path.isdir(w)
        if looks_like_path:
            if not os.path.isdir(os.path.expanduser(w)):
                raise MissingDependencyError(f"Depth Anything weights directory not found: {w}\n{INSTALL_HINT}")
        elif not self.allow_download and not weights_cached(w):
            raise MissingDependencyError(
                f"Depth Anything weights '{w}' are not in the local Hugging Face cache and downloads are disabled.\n"
                f"Run  scripts/convert_relief.py --fetch-depth-model  once (about 100 MB), or pass --allow-download.")

    # -- loading ------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        st = dependency_status()
        missing = [k for k, ok in st.items() if not ok]
        if missing:
            raise MissingDependencyError(
                f"Depth Anything V2 needs the optional packages {missing}, which are not installed.\n" + INSTALL_HINT)
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        kw = {'local_files_only': not self.allow_download}
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.weights, **kw)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.weights, **kw)
        except Exception as e:  # OSError / HF hub errors
            where = self.weights
            if os.path.isdir(str(where)):
                raise MissingDependencyError(f"could not load Depth Anything weights from {where}: {e}") from e
            if self.allow_download:
                raise MissingDependencyError(
                    f"could not download / load Depth Anything weights '{where}' ({e.__class__.__name__}: {e}). "
                    f"Check the network connection and the model id.") from e
            raise MissingDependencyError(
                f"Depth Anything weights '{where}' are not in the local Hugging Face cache and downloads are "
                f"disabled ({e.__class__.__name__}).\n{INSTALL_HINT}") from e
        if self.device == 'auto':
            dev = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None)
                                                            and torch.backends.mps.is_available() else 'cpu')
        else:
            dev = self.device
        self._device = dev
        self._model.to(dev).eval()

    # -- inference ----------------------------------------------------------
    def generate(self, image_path: str, target_hw: Optional[tuple[int, int]] = None) -> GeometryResult:
        self._load()
        import torch
        from PIL import Image
        from core.relief.provider import ALPHA_BACKGROUND
        src = Image.open(image_path)
        mask = None
        if src.mode in ('RGBA', 'LA', 'PA') or (src.mode == 'P' and 'transparency' in src.info):
            alpha = np.asarray(src.convert('RGBA'))[..., 3]
            mask = alpha >= ALPHA_BACKGROUND
        im = src.convert('RGB')
        W0, H0 = im.size
        work = im
        if max(W0, H0) > self.max_side:
            s = self.max_side / max(W0, H0)
            work = im.resize((max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))), Image.Resampling.LANCZOS)
        torch.manual_seed(0)
        inputs = self._processor(images=work, return_tensors='pt').to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        pred = out.predicted_depth            # (1, h, w)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(H0, W0), mode='bicubic',
                                               align_corners=False).squeeze().float().cpu().numpy()
        depth = np.asarray(pred, dtype=np.float32)
        return GeometryResult(depth=depth, mask=mask, depth_convention=self.depth_convention, source=self.name,
                              meta={'model': str(self.weights), 'device': str(self._device),
                                    'inference_size': [int(work.size[1]), int(work.size[0])],
                                    'mask_from_alpha': mask is not None,
                                    'output': ('relative inverse depth (larger = closer)' if self.depth_convention == HIGHER_IS_CLOSER
                                               else 'metric depth (larger = farther), flipped by the pipeline')
                                              + ', bicubic to image size'})
