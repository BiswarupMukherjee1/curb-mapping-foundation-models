"""Assign bottom / top to drawn curb lines geometrically, using the road side.

The model finds both edges of a curb (86-88 % of drawn lines have a parallel partner 8-35 cm away) but sometimes
labels them in the wrong order. The bottom (gutter) line is always on the carriageway side. So within each pair,
the line closer to the nearest road centreline (IGN BD TOPO carriageways, independent of the PCRS) is labelled
bottom and the other top. Unpaired lines keep the model's label.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .align import densify
from .datahub import NATIVE_CRS
from .paths import DATA_RAW

CARRIAGEWAY = {"Route à 1 chaussée", "Route à 2 chaussées", "Rond-point", "Bretelle", "Route empierrée", "Type autoroutier"}


def road_centrelines(tile: str):
    g = gpd.read_file(DATA_RAW / "bdtopo" / f"roads_{tile}.geojson")
    g = g[g["nature"].isin(CARRIAGEWAY)]
    return g.set_crs("EPSG:4326", allow_override=True).to_crs(NATIVE_CRS).geometry.values


def reassign(lines, chan: np.ndarray, roads, pair=(0.08, 0.35), min_pair_share=0.3, step=0.10):
    """Return (new channel array, info DataFrame). 0 = bottom, 1 = top."""
    chan = np.asarray(chan).copy()
    xy, _, ids = densify(lines, step, return_ids=True)
    rxy, _ = densify(roads, 0.5)
    d_road = pd.Series(cKDTree(rxy).query(xy)[0]).groupby(ids).mean()
    tree = cKDTree(xy)
    nb = tree.query_ball_point(xy, r=pair[1])
    votes = {}
    for i, js in enumerate(nb):
        a = ids[i]
        for j in js:
            b = ids[j]
            if b == a:
                continue
            if np.hypot(*(xy[j] - xy[i])) >= pair[0]:
                votes.setdefault(a, {}).setdefault(b, set()).add(i)
    n_pts = pd.Series(ids).value_counts()
    partner = {}
    for a, cand in votes.items():
        b, pts = max(cand.items(), key=lambda kv: len(kv[1]))
        if len(pts) >= min_pair_share * n_pts[a]:
            partner[a] = b
    new = chan.copy()
    for a, b in partner.items():
        if a in d_road.index and b in d_road.index:
            new[a] = 0 if d_road[a] < d_road[b] else 1
    info = pd.DataFrame({"paired": [i in partner for i in range(len(lines))],
                         "changed": new != chan})
    return new, info
