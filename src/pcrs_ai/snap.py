"""Snap drawn lines to the nearest strong image edge (sub-pixel), then smooth the correction along each line.

Motivation: drawn lines show no systematic offset but about ±7 cm position scatter, while PCRS curb lines sit on
image edges. For each vertex sampled every `step`, search ±`search_m` along the local normal for the
maximum gradient magnitude (bilinear), median-smooth the offsets over `smooth` neighbours, and move the vertex.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import median_filter
from shapely.geometry import LineString


def snap_lines(lines, grad: np.ndarray, transform, step=0.05, search_m=0.10, res_m=0.01, smooth=9):
    inv = ~transform
    offs = np.arange(-search_m, search_m + 1e-9, res_m)
    out = []
    for ln in lines:
        n = max(2, int(ln.length / step) + 1)
        d = np.linspace(0, ln.length, n)
        p = np.array([ln.interpolate(x).coords[0][:2] for x in d])
        t = np.gradient(p, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        q = p[:, None, :] + offs[None, :, None] * nrm[:, None, :]           # (n, k, 2) candidate positions
        col = inv.a * q[..., 0] + inv.c - 0.5; row = inv.e * q[..., 1] + inv.f - 0.5
        g = ndimage.map_coordinates(grad, [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(q.shape[:2])
        best = offs[np.argmax(g, axis=1)]
        best = median_filter(best, size=min(smooth, n), mode="nearest")
        out.append(LineString(p + best[:, None] * nrm))
    return out
