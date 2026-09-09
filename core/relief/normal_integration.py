"""Normal maps -> height, and depth + normal fusion (extension point, Milestone 4).

Conventions
-----------
Normals are in IMAGE space: x right, y DOWN the image, z toward the viewer,
unit length.  A tangent-space normal map stored as RGB decodes as
``n = rgb * 2 - 1``; OpenGL-style maps have y UP and need ``y_up=True``
(which negates the y component).  For a height field z(x, y) (in "height units
per pixel") the surface normal is proportional to (-dz/dx, -dz/dy, 1), so

    dz/dx = -nx / nz          dz/dy = -ny / nz

Integration solves the Poisson equation  lap z = div(p, q)  with Neumann
boundaries in the DCT domain (Frankot-Chellappa / Simchony): exact for
integrable fields, least-squares for the rest, O(N log N), deterministic.

Fusion minimises

    E = depth_weight * |z - d|^2 + normal_weight * |grad z - g|^2 + smoothness_weight * |lap z|^2

whose normal equation (depth_weight - normal_weight * lap + smoothness_weight * lap^2) z
= depth_weight * d - normal_weight * div(g) is diagonal in the same DCT basis, so it is
solved in closed form.  Because normals give slopes in height-per-pixel while a
provider's depth has arbitrary units, the integrated normal height is first
scaled to the depth by least squares unless ``normal_scale`` is given.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

NZ_MIN = 0.05


def load_normal_map(path: str, y_up: bool = False) -> np.ndarray:
    """RGB normal map -> (H, W, 3) float32 unit normals (image space)."""
    from PIL import Image
    im = Image.open(path)
    if im.mode.startswith('I;16'):
        arr = np.asarray(im).astype(np.float32) / 65535.0
        arr = np.stack([arr] * 3, axis=-1)
    else:
        arr = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0
    n = arr * 2.0 - 1.0
    if y_up:
        n[..., 1] *= -1.0
    return normalize_normals(n)


def normalize_normals(n: np.ndarray, nz_min: float = NZ_MIN) -> np.ndarray:
    n = np.asarray(n, dtype=np.float32).copy()
    n[..., 2] = np.maximum(n[..., 2], nz_min)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return (n / norm).astype(np.float32)


def normals_from_height(z: np.ndarray) -> np.ndarray:
    """Unit normals of a height field (height units per pixel) - the inverse of
    :func:`gradients_from_normals`; used by the tests."""
    gy, gx = np.gradient(np.asarray(z, dtype=np.float64))
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)


def gradients_from_normals(normals: np.ndarray, nz_min: float = NZ_MIN) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(normals, dtype=np.float64)
    nz = np.maximum(n[..., 2], nz_min)
    p = -n[..., 0] / nz
    q = -n[..., 1] / nz
    return p, q


def _divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Neumann-consistent divergence of pixel-centred gradients."""
    H, W = p.shape
    pf = np.zeros((H, W + 1))
    pf[:, 1:W] = 0.5 * (p[:, :-1] + p[:, 1:])        # flux across the vertical cell walls
    qf = np.zeros((H + 1, W))
    qf[1:H, :] = 0.5 * (q[:-1, :] + q[1:, :])
    return (pf[:, 1:] - pf[:, :-1]) + (qf[1:, :] - qf[:-1, :])


def _laplacian_eigenvalues(H: int, W: int) -> np.ndarray:
    ky = 2.0 * np.cos(np.pi * np.arange(H) / H) - 2.0
    kx = 2.0 * np.cos(np.pi * np.arange(W) / W) - 2.0
    return ky[:, None] + kx[None, :]


def solve_poisson_dct(f: np.ndarray) -> np.ndarray:
    """lap z = f with Neumann boundaries; z has zero mean."""
    from scipy.fft import dctn, idctn
    H, W = f.shape
    F = dctn(np.asarray(f, dtype=np.float64), type=2, norm='ortho')
    lam = _laplacian_eigenvalues(H, W)
    lam[0, 0] = 1.0
    Zh = F / lam
    Zh[0, 0] = 0.0
    return idctn(Zh, type=2, norm='ortho')


def integrate_normals(normals: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Height (height units per pixel, zero mean) from a normal map.  Outside
    ``mask`` the gradients are set to zero (flat background)."""
    p, q = gradients_from_normals(normals)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        p = np.where(m, p, 0.0)
        q = np.where(m, q, 0.0)
    return solve_poisson_dct(_divergence(p, q))


def fuse_depth_normals(depth: np.ndarray, normals: np.ndarray, depth_weight: float = 1.0,
                       normal_weight: float = 1.0, smoothness_weight: float = 0.0,
                       normal_scale: Optional[float] = None, mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """Closed-form minimiser of the depth + normal + smoothness energy (module
    docstring).  ``depth`` is 'larger = raised' in any units; the result is in
    the same units.  Returns (fused (H, W) float32, info)."""
    from scipy.fft import dctn, idctn
    d = np.asarray(depth, dtype=np.float64)
    if normals.shape[:2] != d.shape:
        raise ValueError("normals must match the depth shape")
    wd, wn, ws = float(depth_weight), float(normal_weight), float(smoothness_weight)
    if wd < 0 or wn < 0 or ws < 0 or (wd == 0 and wn == 0):
        raise ValueError("weights must be >= 0 and depth_weight + normal_weight > 0")
    p, q = gradients_from_normals(normals)
    m = None if mask is None else np.asarray(mask).astype(bool)
    if m is not None:
        p = np.where(m, p, 0.0)
        q = np.where(m, q, 0.0)
    # scale the normal slopes (height/px) to the depth's units
    if normal_scale is None:
        zn = solve_poisson_dct(_divergence(p, q))
        sel = m if m is not None else np.ones(d.shape, bool)
        zc = zn[sel] - zn[sel].mean()
        dc = d[sel] - d[sel].mean()
        denom = float(zc @ zc)
        scale = float(zc @ dc) / denom if denom > 1e-12 else 1.0
        if abs(scale) < 1e-12:
            scale = 1.0
    else:
        scale = float(normal_scale)
    p, q = p * scale, q * scale
    f = _divergence(p, q)
    H, W = d.shape
    lam = _laplacian_eigenvalues(H, W)
    denom = wd - wn * lam + ws * lam * lam
    num = wd * dctn(d, type=2, norm='ortho') - wn * dctn(f, type=2, norm='ortho')
    if wd == 0:
        denom[0, 0] = 1.0
        num[0, 0] = 0.0
    Zh = num / denom
    z = idctn(Zh, type=2, norm='ortho')
    if wd == 0:
        z = z - z.mean() + d.mean()
    info = {'depth_weight': wd, 'normal_weight': wn, 'smoothness_weight': ws, 'normal_scale': scale,
            'method': 'screened Poisson, DCT-II (Neumann), closed form'}
    return z.astype(np.float32), info
