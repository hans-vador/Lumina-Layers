"""Protect both sides of sharp source boundaries, before colour reduction."""
import numpy as np
from core.band.optics import srgb_to_lab_d65


def contrast_edges(rgb, threshold=18.0):
    """Mark neighbouring pixels separated by at least threshold CIELAB units.

    Both lightness and chromatic boundaries count; smooth gradients do not.
    No dilation: preserve the source outline without inventing a black stroke.
    """
    lab = srgb_to_lab_d65(np.asarray(rgb, dtype=float) / 255.)
    edges = np.zeros(lab.shape[:2], bool)
    dx = np.linalg.norm(lab[:, 1:] - lab[:, :-1], axis=-1) >= threshold
    dy = np.linalg.norm(lab[1:] - lab[:-1], axis=-1) >= threshold
    edges[:, 1:] |= dx
    edges[:, :-1] |= dx
    edges[1:] |= dy
    edges[:-1] |= dy
    return edges
