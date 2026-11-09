"""In-memory raster stack of one tile for learning: RGB 5 cm, curb target, ignore band, split mask.

target = 2 channels, 1 within POS_M of a PCRS curb line of that channel:
         channel 0 "bottom / outer" = BORD TROT, CHAU, CIRCU, FRAN (gutter line or outer limit)
         channel 1 "top"            = BORD HAUT
         (one merged band for both lines makes the model draw a single line between them, 15 cm from each)
ignore = per channel, POS_M < distance < IGN_M from that channel's lines (label uncertainty ring; excluded from the loss)
split  = 0 outside the evaluation region, 1 train, 2 val, 3 test (see protocol.region)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box

from . import protocol
from .io import load_ortho

POS_M = 0.05
IGN_M = 0.10
CHANNELS = {"bottom": ["TROT", "CHAU", "CIRCU", "FRAN"], "top": ["HAUT"]}
SPLITS = {"train": 1, "val": 2, "test": 3}
WIN = 512


@dataclass
class TileData:
    tile: str
    rgb: np.ndarray          # (3, H, W) uint8
    target: np.ndarray       # (2, H, W) uint8
    ignore: np.ndarray       # (2, H, W) uint8
    split: np.ndarray        # (H, W) uint8
    transform: Affine        # pixel-corner affine (EPSG:3945)
    nir: np.ndarray | None = None   # (H, W) uint8 near-infrared, loaded only for encoders that use it

    @property
    def shape(self):
        return self.split.shape

    def window(self, r0: int, c0: int, size: int = WIN):
        s = np.s_[r0:r0 + size, c0:c0 + size]
        a = (slice(None),) + s
        return self.rgb[a], self.target[a], self.ignore[a], self.split[s]

    def image(self, r0: int, c0: int, size: int = WIN) -> np.ndarray:
        """(3 or 4, size, size) uint8: RGB, plus NIR when loaded."""
        s = np.s_[r0:r0 + size, c0:c0 + size]
        rgb = self.rgb[(slice(None),) + s]
        return rgb if self.nir is None else np.concatenate([rgb, self.nir[s][None]], 0)

    def band_stats(self):
        """Per-band mean / std (RGB[+NIR]) on a 1/16 subsample, in raw 0-255 units."""
        a = self.rgb[:, ::16, ::16].astype(np.float32)
        if self.nir is not None:
            a = np.concatenate([a, self.nir[None, ::16, ::16].astype(np.float32)], 0)
        return a.reshape(a.shape[0], -1).mean(1), a.reshape(a.shape[0], -1).std(1)

    def window_bounds(self, r0: int, c0: int, size: int = WIN):
        x0, y0 = self.transform * (c0, r0)
        x1, y1 = self.transform * (c0 + size, r0 + size)
        return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def load_tile(tile: str, verbose: bool = True, with_nir: bool = False) -> TileData:
    o = load_ortho("ortho2023", tile)
    rgb = o.read_all()
    H, W = rgb.shape[1:]
    tr = o.transform
    cb = protocol.curbs(tile)
    tgt = np.zeros((len(CHANNELS), H, W), np.uint8)
    ign = np.zeros((len(CHANNELS), H, W), np.uint8)
    for k, subs in enumerate(CHANNELS.values()):
        lines = cb[cb.ssclasse.isin(subs)].geometry
        tgt[k] = rasterize(((g.buffer(POS_M), 1) for g in lines), out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
        near = rasterize(((g.buffer(IGN_M), 1) for g in lines), out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
        ign[k] = (near.astype(bool) & ~tgt[k].astype(bool)).astype(np.uint8)
        del near
    split = np.zeros((H, W), np.uint8)
    for name, code in SPLITS.items():
        reg = protocol.region(tile, name)
        if not reg.is_empty:
            m = rasterize([(reg, 1)], out_shape=(H, W), transform=tr, fill=0, dtype="uint8")
            split[m.astype(bool)] = code
    if verbose:
        print(f"[tiledata] {tile}: {H}x{W}, curb px {tgt.sum() / 1e6:.2f} M, split px "
              + ", ".join(f"{k} {(split == v).mean():.3f}" for k, v in SPLITS.items()), flush=True)
    nir = load_ortho("orthoir2023", tile).read_all(bands=[1])[0] if with_nir else None   # CIR band order: NIR, R, G
    return TileData(tile, rgb, tgt, ign, split, tr, nir)


def grid_windows(td: TileData, split: str, size: int = WIN, stride: int | None = None, min_frac: float = 0.15):
    """Top-left corners of a regular grid of windows whose area is >= min_frac inside `split`."""
    stride = stride or size
    code = SPLITS[split]
    H, W = td.shape
    out = []
    for r0 in range(0, H - size + 1, stride):
        for c0 in range(0, W - size + 1, stride):
            if (td.split[r0:r0 + size:8, c0:c0 + size:8] == code).mean() >= min_frac:
                out.append((r0, c0))
    return out


def curb_length_m(td: TileData, r0: int, c0: int, size: int = WIN) -> float:
    """Approximate curb length inside a window from the target band (band ≈ 2*POS_M wide)."""
    px = int(td.target[:, r0:r0 + size, c0:c0 + size].sum())
    gsd = td.transform.a
    return px * gsd * gsd / (2 * POS_M)
