"""M0: classical curb-line extraction from the 5 cm ortho (no learning).

1. LSD line segments on luminance, tile processed in overlapping blocks
2. per-segment cues: length, luminance contrast across the line, NDVI on the line, nDSM at the midpoint
3. optional "curb band" cue: a parallel segment 10-40 cm away (a curb stone has two edges)
4. thresholds tuned on validation blocks only
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from ..io import Ortho


@dataclass
class Segments:
    xy: np.ndarray            # [N, 4] x1, y1, x2, y2 in tile pixel-centre coordinates
    feats: dict               # name -> [N] array

    def __len__(self):
        return len(self.xy)


def detect_lsd(gray: np.ndarray, block: int = 4000, overlap: int = 64, min_len_px: float = 10.0) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    H, W = gray.shape
    out = []
    for r0 in range(0, H, block):
        for c0 in range(0, W, block):
            rr0, cc0 = max(r0 - overlap, 0), max(c0 - overlap, 0)
            sub = gray[rr0:min(r0 + block + overlap, H), cc0:min(c0 + block + overlap, W)]
            lines = lsd.detect(sub)[0]
            if lines is None:
                continue
            L = lines.reshape(-1, 4).astype(np.float64)
            L[:, [0, 2]] += cc0; L[:, [1, 3]] += rr0
            mx, my = (L[:, 0] + L[:, 2]) / 2, (L[:, 1] + L[:, 3]) / 2
            core = (mx >= c0) & (mx < c0 + block) & (my >= r0) & (my < r0 + block)      # avoid duplicates
            length = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
            out.append(L[core & (length >= min_len_px)])
    return np.concatenate(out) if out else np.zeros((0, 4))


def segment_features(seg: np.ndarray, lum: np.ndarray, ndvi: np.ndarray, ndsm_at, side_px: float = 3.0,
                     n_samples: int = 8) -> dict:
    p1, p2 = seg[:, :2], seg[:, 2:]
    d = p2 - p1
    length = np.hypot(d[:, 0], d[:, 1])
    t = d / length[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    s = np.linspace(0.1, 0.9, n_samples)
    P = p1[:, None, :] + s[None, :, None] * d[:, None, :]                      # [N, S, 2] (x=col, y=row)

    def samp(img, Q):
        v = ndimage.map_coordinates(img, [Q[..., 1].ravel(), Q[..., 0].ravel()], order=1, mode="nearest")
        return v.reshape(Q.shape[:2]).mean(axis=1)

    left = samp(lum, P + side_px * n[:, None, :])
    right = samp(lum, P - side_px * n[:, None, :])
    mid = ((p1 + p2) / 2)
    return {"length_px": length, "contrast": np.abs(left - right), "ndvi_on": samp(ndvi, P),
            "ndsm_mid": ndsm_at(mid), "angle": np.arctan2(t[:, 1], t[:, 0]) % np.pi}


def has_parallel_partner(seg: np.ndarray, angle: np.ndarray, gap_px=(2.0, 8.0), max_dangle_deg=8.0) -> np.ndarray:
    mid = (seg[:, :2] + seg[:, 2:]) / 2
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    tree = cKDTree(mid)
    t = np.stack([np.cos(angle), np.sin(angle)], 1)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    out = np.zeros(len(seg), bool)
    for i, nb in enumerate(tree.query_ball_point(mid, r=length / 2 + gap_px[1])):
        for j in nb:
            if j == i:
                continue
            da = abs(angle[i] - angle[j]); da = min(da, np.pi - da)
            if np.rad2deg(da) > max_dangle_deg:
                continue
            v = mid[j] - mid[i]
            perp = abs(v @ n[i])
            along = abs(v @ t[i])
            if gap_px[0] <= perp <= gap_px[1] and along < (length[i] + length[j]) / 2:
                out[i] = True
                break
    return out


def select(seg: Segments, min_len_px=20, min_contrast=12.0, max_ndvi=0.2, max_ndsm=0.4, require_pair=False) -> np.ndarray:
    f = seg.feats
    m = (f["length_px"] >= min_len_px) & (f["contrast"] >= min_contrast) & (f["ndvi_on"] <= max_ndvi) & (f["ndsm_mid"] <= max_ndsm)
    if require_pair:
        m &= f["pair"]
    return m


def to_world_lines(seg_xy: np.ndarray, ortho: Ortho) -> list[LineString]:
    tr = ortho.transform
    out = []
    for x1, y1, x2, y2 in seg_xy:
        a = (tr.a * (x1 + 0.5) + tr.b * (y1 + 0.5) + tr.c, tr.d * (x1 + 0.5) + tr.e * (y1 + 0.5) + tr.f)
        b = (tr.a * (x2 + 0.5) + tr.b * (y2 + 0.5) + tr.c, tr.d * (x2 + 0.5) + tr.e * (y2 + 0.5) + tr.f)
        out.append(LineString([a, b]))
    return out
