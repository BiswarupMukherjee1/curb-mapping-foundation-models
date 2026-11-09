"""Spatial train/val/test split: 100 m blocks of the native grid, assigned by a stable hash."""
from __future__ import annotations

import hashlib

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from .datahub import NATIVE_CRS

BLOCK_M = 100.0
FRACTIONS = {"train": 0.60, "val": 0.15, "test": 0.25}
SALT = "pcrs-ai-split-v1"
EVAL_SHRINK_M = 2.0
# lots whose PCRS disagrees with the 2023 orthophoto by 20 cm or more (data audit): excluded from ground truth, kept for QC
EXCLUDED_LOTS = {"RL": ["C12", "BC02", "BC03", "30", "BC06"], "OH": []}


def block_split(ix: int, iy: int) -> str:
    h = int(hashlib.sha1(f"{SALT}:{ix}:{iy}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, f in FRACTIONS.items():
        acc += f
        if h < acc:
            return name
    return "test"


def blocks_for_tile(tile_polygon) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = tile_polygon.bounds
    rows = []
    for ix in range(int(np.floor(minx / BLOCK_M)), int(np.ceil(maxx / BLOCK_M))):
        for iy in range(int(np.floor(miny / BLOCK_M)), int(np.ceil(maxy / BLOCK_M))):
            b = box(ix * BLOCK_M, iy * BLOCK_M, (ix + 1) * BLOCK_M, (iy + 1) * BLOCK_M)
            inter = b.intersection(tile_polygon)
            if inter.area < 0.5 * BLOCK_M ** 2:
                continue
            rows.append({"ix": ix, "iy": iy, "split": block_split(ix, iy), "geometry": inter,
                         "eval_geometry": inter.buffer(-EVAL_SHRINK_M)})
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=NATIVE_CRS)
    return g


def split_region(blocks: gpd.GeoDataFrame, split: str, shrink: bool = True):
    sel = blocks[blocks.split == split]
    geoms = sel["eval_geometry"] if shrink else sel.geometry
    return gpd.GeoSeries(geoms, crs=blocks.crs).union_all()
