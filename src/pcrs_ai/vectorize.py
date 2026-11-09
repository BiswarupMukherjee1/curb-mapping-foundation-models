"""Probability raster -> polylines (world coordinates) with a per-line mean probability.

threshold -> skeleton (1 px) -> trace chains between endpoints / junctions (8-connectivity) -> pixel centres to world
coordinates -> Douglas-Peucker simplification -> drop short pieces.
"""
from __future__ import annotations

import numpy as np
import shapely
from affine import Affine
from scipy import ndimage
from shapely.geometry import LineString
from skimage.morphology import skeletonize

NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(sk: np.ndarray) -> list[np.ndarray]:
    """Return pixel chains [(N,2) rows, cols] covering the skeleton; each chain ends at an endpoint or junction."""
    rr, cc = np.nonzero(sk)
    if len(rr) == 0:
        return []
    H, W = sk.shape
    idx = -np.ones(sk.shape, np.int64)
    idx[rr, cc] = np.arange(len(rr))
    nbrs = [[] for _ in range(len(rr))]
    for dr, dc in NB:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.full(len(rr), -1)
        j[ok] = idx[r2[ok], c2[ok]]
        for i in np.flatnonzero(j >= 0):
            nbrs[i].append(int(j[i]))
    deg = np.array([len(n) for n in nbrs])
    visited_edge = set()
    chains = []
    nodes = np.flatnonzero(deg != 2)
    for s in nodes:
        for n in nbrs[s]:
            if (s, n) in visited_edge:
                continue
            chain = [s, n]
            visited_edge.add((s, n)); visited_edge.add((n, s))
            prev, cur = s, n
            while deg[cur] == 2:
                nxt = nbrs[cur][0] if nbrs[cur][0] != prev else nbrs[cur][1]
                if (cur, nxt) in visited_edge:
                    break
                visited_edge.add((cur, nxt)); visited_edge.add((nxt, cur))
                chain.append(nxt)
                prev, cur = cur, nxt
            chains.append(np.array(chain))
    # pure cycles (all degree 2) were not reached from any node
    seen = np.zeros(len(rr), bool)
    for ch in chains:
        seen[ch] = True
    for s in np.flatnonzero((deg == 2) & ~seen):
        if seen[s]:
            continue
        chain, prev, cur = [s], -1, s
        while True:
            seen[cur] = True
            cand = [x for x in nbrs[cur] if x != prev and not seen[x]]
            if not cand:
                break
            prev, cur = cur, cand[0]
            chain.append(cur)
        if len(chain) > 2:
            chains.append(np.array(chain))
    return [np.stack([rr[ch], cc[ch]], 1) for ch in chains]


def prob_to_lines(prob: np.ndarray, transform: Affine, threshold: float = 0.5, min_len_m: float = 0.5,
                  simplify_m: float = 0.02, valid: np.ndarray | None = None):
    """prob: (H, W) in [0, 1] for a raster whose pixel-corner affine is `transform`.
    Returns (list[LineString], np.ndarray mean_prob per line)."""
    m = prob >= threshold
    if valid is not None:
        m &= valid
    m = ndimage.binary_closing(m, iterations=1)
    sk = skeletonize(m)
    lines, conf = [], []
    for ch in trace_skeleton(sk):
        if len(ch) < 2:
            continue
        r, c = ch[:, 0].astype(float), ch[:, 1].astype(float)
        x = transform.a * (c + 0.5) + transform.b * (r + 0.5) + transform.c
        y = transform.d * (c + 0.5) + transform.e * (r + 0.5) + transform.f
        ln = LineString(np.stack([x, y], 1))
        if ln.length < min_len_m:
            continue
        lines.append(shapely.simplify(ln, simplify_m))
        conf.append(float(prob[ch[:, 0], ch[:, 1]].mean()))
    return lines, np.asarray(conf)
