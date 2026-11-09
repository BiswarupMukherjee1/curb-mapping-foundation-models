"""Evaluation protocol shared by all learned-model experiments.

- region(tile, split): 100 m blocks of that split, shrunk by 2 m, intersected with PCRS coverage (lots CONSTITUE),
  minus lots whose PCRS disagrees with the 2023 orthophoto (found in the data audit).
- ground truth: PCRS curb lines (classe BORD, all sub-classes) clipped to the region.
- reason_of(xy): visibility reason code of the nearest curb point (visible / visible_shadow / vegetation / elevated_object).
"""
from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .io import load_pcrs, load_tile_polygon
from .metrics import line_metrics
from .paths import DATA_PROCESSED
from .splits import EXCLUDED_LOTS, blocks_for_tile, split_region

REASONS = ["visible", "visible_shadow", "vegetation", "elevated_object"]


@lru_cache(maxsize=None)
def curbs(tile: str) -> gpd.GeoDataFrame:
    infra = load_pcrs(tile, "cr_objet_infra_l")
    return infra[infra.classe == "BORD"].reset_index(drop=True)


@lru_cache(maxsize=None)
def region(tile: str, split: str, shrink: bool = True):
    blocks = blocks_for_tile(load_tile_polygon(tile))
    reg = split_region(blocks, split, shrink=shrink)
    lots = load_pcrs(tile, "cr_zone_couverture_s")
    covered = lots[(lots.etat == "CONSTITUE") & ~lots.lot.isin(EXCLUDED_LOTS.get(tile, []))].geometry.union_all()
    reg = reg.intersection(covered)
    ex = lots[lots.lot.isin(EXCLUDED_LOTS.get(tile, []))]
    if len(ex):
        reg = reg.difference(ex.geometry.union_all())
    return reg


def ground_truth(tile: str, split: str) -> gpd.GeoSeries:
    return curbs(tile).geometry.clip(region(tile, split))


@lru_cache(maxsize=None)
def _reason_index(tile: str):
    P = pd.read_csv(DATA_PROCESSED / f"curb_points_{tile}.csv.gz", usecols=["x", "y", "reason"])
    return cKDTree(P[["x", "y"]].values), P.reason.values


def reason_of(tile: str, xy: np.ndarray) -> np.ndarray:
    tree, reasons = _reason_index(tile)
    _, j = tree.query(xy, k=1)
    return reasons[j]


def evaluate(tile: str, gt: gpd.GeoSeries, pred) -> dict:
    """All-GT and visible-GT metrics at 5/10/20 cm, plus 10 cm completeness per visibility reason."""
    pred = list(pred)
    out = {"all": line_metrics(gt.values, pred)}
    out["visible"] = line_metrics(gt.values, pred, gt_weight=lambda xy: np.isin(reason_of(tile, xy), REASONS[:2]).astype(float))
    for r in REASONS:
        m = line_metrics(gt.values, pred, tolerances=(0.10,), gt_weight=lambda xy, r=r: (reason_of(tile, xy) == r).astype(float))
        out[f"completeness_10cm_{r}"] = m["completeness_10cm"]
    return out
