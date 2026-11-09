"""Measure the offset between an orthophoto and vector lines (PCRS) by maximising edge energy along the lines.

Idea: a correctly placed curb line lies on an image edge. Shift all line samples by (dx, dy) on a grid and pick the shift
that maximises the mean gradient magnitude sampled under them. Sub-pixel via bilinear sampling and a parabolic peak fit.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def densify(geoms, step: float, return_ids: bool = False):
    """Sample points every `step` metres along (Multi)LineStrings (vectorised, shapely 2).
    Returns (xy[N,2], tangent unit vectors[N,2]) [, ids[N] = index of the source geometry]; a line of length L gives
    ceil(L / step) points starting at 0."""
    import shapely
    g = np.asarray(geoms, dtype=object)
    empty = (np.zeros((0, 2)), np.zeros((0, 2))) + ((np.zeros(0, int),) if return_ids else ())
    if g.size == 0:
        return empty
    src = np.array([i for i, x in enumerate(g) if x is not None and not x.is_empty], dtype=int)
    if len(src) == 0:
        return empty
    parts, pidx = shapely.get_parts(np.asarray([g[i] for i in src], dtype=object), return_index=True)
    pid = src[pidx]
    keep = shapely.get_type_id(parts) == 1                              # LineStrings only
    parts, pid = parts[keep], pid[keep]
    lens = shapely.length(parts)
    keep = lens >= step
    parts, lens, pid = parts[keep], lens[keep], pid[keep]
    if len(parts) == 0:
        return empty
    counts = np.ceil(lens / step - 1e-9).astype(int)
    idx = np.repeat(np.arange(len(parts)), counts)
    start = np.repeat(np.cumsum(counts) - counts, counts)
    d = (np.arange(counts.sum()) - start) * step
    p = shapely.get_coordinates(shapely.line_interpolate_point(parts[idx], d))[:, :2]
    q = shapely.get_coordinates(shapely.line_interpolate_point(parts[idx], np.minimum(d + step / 2, lens[idx])))[:, :2]
    v = q - p
    n = np.linalg.norm(v, axis=1, keepdims=True)
    ok = n[:, 0] > 1e-9
    if return_ids:
        return p[ok], v[ok] / n[ok], pid[idx][ok]
    return p[ok], v[ok] / n[ok]


def gradient_magnitude(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = ndimage.gaussian_filter(g, sigma, order=(0, 1))
    gy = ndimage.gaussian_filter(g, sigma, order=(1, 0))
    return np.hypot(gx, gy)


def edge_energy_grid(grad: np.ndarray, cols: np.ndarray, rows: np.ndarray, shifts_px: np.ndarray) -> np.ndarray:
    """Mean gradient magnitude under points shifted by every (dx, dy) in shifts_px x shifts_px (pixel units,
    dx along columns / east, dy along rows / south). cols/rows are pixel-centre coordinates (integer = centre)."""
    out = np.zeros((len(shifts_px), len(shifts_px)), dtype=np.float64)
    for i, dy in enumerate(shifts_px):
        for j, dx in enumerate(shifts_px):
            v = ndimage.map_coordinates(grad, [rows + dy, cols + dx], order=1, mode="nearest")
            out[i, j] = v.mean()
    return out


def peak_subpixel(E: np.ndarray, shifts: np.ndarray) -> tuple[float, float, float]:
    """Argmax of a 2D grid with a separable parabolic refinement. Returns (dx, dy, sharpness)."""
    i, j = np.unravel_index(np.argmax(E), E.shape)
    step = shifts[1] - shifts[0]

    def para(a, b, c):
        den = a - 2 * b + c
        return 0.0 if den == 0 else 0.5 * (a - c) / den

    oi = para(E[i - 1, j], E[i, j], E[i + 1, j]) if 0 < i < E.shape[0] - 1 else 0.0
    oj = para(E[i, j - 1], E[i, j], E[i, j + 1]) if 0 < j < E.shape[1] - 1 else 0.0
    sharp = (E.max() - np.median(E)) / (E.std() + 1e-12)
    return shifts[j] + oj * step, shifts[i] + oi * step, float(sharp)


def phase_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (dx, dy) in pixels such that b(x) ≈ a(x - d). Hann window, sub-pixel by parabolic fit."""
    a = a.astype(np.float64) - a.mean(); b = b.astype(np.float64) - b.mean()
    w = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    Fa, Fb = np.fft.fft2(a * w), np.fft.fft2(b * w)
    R = Fb * np.conj(Fa); R /= np.abs(R) + 1e-12
    r = np.fft.ifft2(R).real
    i, j = np.unravel_index(np.argmax(r), r.shape)

    def para(m, c, p):
        den = m - 2 * c + p
        return 0.0 if den == 0 else 0.5 * (m - p) / den

    H, W = r.shape
    di = para(r[(i - 1) % H, j], r[i, j], r[(i + 1) % H, j])
    dj = para(r[i, (j - 1) % W], r[i, j], r[i, (j + 1) % W])
    dy = i + di if i <= H // 2 else i + di - H
    dx = j + dj if j <= W // 2 else j + dj - W
    return float(dx), float(dy)


def estimate_offset(grad: np.ndarray, ortho, xy: np.ndarray, search_m: float = 0.40, grid_m: float = 0.02,
                    margin_px: int = 20) -> dict | None:
    """Offset (metres, x east / y north) to ADD to the vector points so that they sit on image edges."""
    if len(xy) == 0:
        return None
    shifts_px = np.arange(-search_m, search_m + 1e-9, grid_m) / ortho.gsd
    col, row = ortho.world_to_pixel(xy[:, 0], xy[:, 1])
    col, row = col - 0.5, row - 0.5                                   # corner coords -> centre-index coords
    ok = (col > margin_px) & (row > margin_px) & (col < ortho.width - margin_px) & (row < ortho.height - margin_px)
    if ok.sum() < 50:
        return None
    E = edge_energy_grid(grad, col[ok], row[ok], shifts_px)
    dx, dy, sharp = peak_subpixel(E, shifts_px)
    c = len(shifts_px) // 2
    return {"n_pts": int(ok.sum()), "dx_m": dx * ortho.gsd, "dy_m": -dy * ortho.gsd, "sharpness": sharp,
            "E_gain": float(E.max() / E[c, c]), "E": E}
