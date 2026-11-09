"""Is a ground object (curb) visible from the air? Per-point reason codes from infrared vegetation and height.

- vegetation : NDVI from the CIR ortho (bands NIR, R, G) above a threshold (canopy over the curb)
- elevated   : nDSM = DSM - DTM above a threshold in a small neighbourhood (cars, awnings, canopy without NDVI)
- shadow     : low RGB brightness (the curb may still be visible; reported, not excluded)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .io import Ortho, load_dem, load_ortho

NDVI_VEG = 0.20          # canopy threshold (checked against the NDVI histogram at curb points)
NDSM_ELEVATED_M = 0.8    # above a 15-20 cm curb step and most street furniture bases
SHADOW_LUM = 55.0        # mean RGB value (0-255)


@dataclass
class PointVisibility:
    ndvi: np.ndarray
    ndsm: np.ndarray          # nDSM directly at the point (50 cm cell)
    lum: np.ndarray
    ndsm_nbhd: np.ndarray | None = None   # max nDSM in a 1.5 m neighbourhood (conservative variant)

    @property
    def vegetation(self):
        return self.ndvi > NDVI_VEG

    @property
    def elevated(self):
        return self.ndsm > NDSM_ELEVATED_M

    @property
    def visible(self):
        return ~(self.vegetation | self.elevated)

    @property
    def shadow(self):
        return self.lum < SHADOW_LUM

    def reason(self) -> np.ndarray:
        r = np.full(len(self.ndvi), "visible", dtype=object)
        r[self.elevated & ~self.vegetation] = "elevated_object"
        r[self.vegetation] = "vegetation"
        r[self.visible & self.shadow] = "visible_shadow"
        return r


def _sample_window_mean(o: Ortho, xy: np.ndarray, bands, radius_px: int = 1, block: int = 4000) -> np.ndarray:
    """Mean over a (2r+1)^2 neighbourhood of each point, read block by block to bound memory. Returns [N, len(bands)]."""
    col, row = o.world_to_pixel(xy[:, 0], xy[:, 1])
    col, row = np.floor(col).astype(int), np.floor(row).astype(int)
    out = np.full((len(xy), len(bands)), np.nan, np.float32)
    for r0 in range(0, o.height, block):
        for c0 in range(0, o.width, block):
            m = (row >= r0) & (row < r0 + block) & (col >= c0) & (col < c0 + block)
            if not m.any():
                continue
            pad = radius_px
            rr0, cc0 = max(r0 - pad, 0), max(c0 - pad, 0)
            h = min(r0 + block + pad, o.height) - rr0
            w = min(c0 + block + pad, o.width) - cc0
            a = o.window(cc0, rr0, w, h, bands=list(bands)).astype(np.float32)
            a = ndimage.uniform_filter(a, size=(1, 2 * radius_px + 1, 2 * radius_px + 1))
            out[m] = a[:, row[m] - rr0, col[m] - cc0].T
    return out


def point_visibility(tile: str, xy: np.ndarray) -> PointVisibility:
    ir = load_ortho("orthoir2023", tile)
    nir_r = _sample_window_mean(ir, xy, bands=(1, 2))                    # rasterio bands are 1-based: NIR, R
    ndvi = (nir_r[:, 0] - nir_r[:, 1]) / (nir_r[:, 0] + nir_r[:, 1] + 1e-6)
    rgb = load_ortho("ortho2023", tile)
    lum = _sample_window_mean(rgb, xy, bands=(1, 2, 3)).mean(axis=1)
    dsm, tr = load_dem("mns2023", tile)
    dtm, _ = load_dem("mnt2023", tile)
    ndsm0 = np.nan_to_num(dsm - dtm, nan=0.0)
    ndsm3 = ndimage.maximum_filter(ndsm0, size=3)                                  # 1.5 m neighbourhood
    inv = ~tr
    c = inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c
    r = inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f
    ci = np.clip(np.floor(c).astype(int), 0, ndsm0.shape[1] - 1)
    ri = np.clip(np.floor(r).astype(int), 0, ndsm0.shape[0] - 1)
    return PointVisibility(ndvi=ndvi, ndsm=ndsm0[ri, ci], lum=lum, ndsm_nbhd=ndsm3[ri, ci])
