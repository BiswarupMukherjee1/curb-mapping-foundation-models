"""Where do the wrong drawn lines come from, and do two simple post-processing fixes help?

Input: the test lines written by the review experiment (data/processed/review_lines_<run>.gpkg).
1. Decomposition of drawn length: correct (same class within 10 cm), on the other curb line, same class 10-20 cm
   away, near other PCRS objects (steps, gutters, walls), or far from any PCRS line.
2. Pairing: share of lines with a parallel partner 8-35 cm away (both curb edges found) or a duplicate within 5 cm.
3. Signed offset of drawn points from their own PCRS line, positive towards the road.
4. Fix 1: relabel bottom/top inside each pair by distance to the road centreline (IGN BD TOPO).
5. Fix 2: snap each line to the strongest image edge within ±10 cm.
Usage: PYTHONPATH=src python experiments/04_review_ranking/error_analysis.py --lines data/processed/review_lines_<run>.gpkg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pcrs_ai import paths  # noqa: E402,F401

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from pcrs_ai import protocol  # noqa: E402
from pcrs_ai.align import densify, gradient_magnitude  # noqa: E402
from pcrs_ai.io import load_ortho, load_pcrs  # noqa: E402
from pcrs_ai.metrics import line_metrics  # noqa: E402
from pcrs_ai.orient import reassign, road_centrelines  # noqa: E402
from pcrs_ai.snap import snap_lines  # noqa: E402
from pcrs_ai.tiledata import CHANNELS  # noqa: E402


def same_class_metrics(tile, lines, chan):
    cb = protocol.curbs(tile); reg = protocol.region(tile, "test")
    out = {}
    for k, (name, subs) in enumerate(CHANNELS.items()):
        m = line_metrics(cb[cb.ssclasse.isin(subs)].geometry.clip(reg).values, [l for l, c in zip(lines, chan) if c == k],
                         tolerances=(0.05, 0.10))
        out[name] = {"completeness_10cm": m["completeness_10cm"], "correctness_10cm": m["correctness_10cm"],
                     "completeness_5cm": m["completeness_5cm"]}
    return out


def analyse_tile(tile, D):
    lines = list(D.geometry.values); chan = D.channel.values.astype(int)
    cb = protocol.curbs(tile); reg = protocol.region(tile, "test").buffer(1.0)
    infra = load_pcrs(tile, "cr_objet_infra_l"); delim = load_pcrs(tile, "cr_objet_delim_l")
    dense = lambda g: densify(g.clip(reg).values, 0.02)[0]
    own = {k: cKDTree(dense(cb[cb.ssclasse.isin(s)].geometry)) for k, s in enumerate(CHANNELS.values())}
    oth_infra = cKDTree(dense(infra[infra.classe != "BORD"].geometry)); oth_delim = cKDTree(dense(delim.geometry))
    xy, _, ids = densify(lines, 0.05, return_ids=True); ch = chan[ids]
    d_own = np.where(ch == 0, own[0].query(xy)[0], own[1].query(xy)[0])
    d_oth = np.where(ch == 0, own[1].query(xy)[0], own[0].query(xy)[0])
    cat = np.full(len(xy), "far from any PCRS line", object)
    cat[oth_delim.query(xy)[0] <= 0.10] = "other PCRS objects"
    cat[oth_infra.query(xy)[0] <= 0.10] = "other PCRS objects"
    cat[(d_own > 0.10) & (d_own <= 0.20)] = "right line type, 10-20 cm away"
    cat[d_oth <= 0.10] = "on the other curb line"
    cat[d_own <= 0.10] = "correct (<= 10 cm)"
    res = {"decomposition": pd.Series(cat).value_counts(normalize=True).round(3).to_dict()}

    b = xy[ch == 0]; t = xy[ch == 1]
    db = cKDTree(t).query(b)[0]
    res["pairing_bottom_lines"] = {"duplicate_within_5cm": float((db <= 0.05).mean()),
                                   "partner_8_35cm": float(((db > 0.08) & (db <= 0.35)).mean())}

    rxy, _ = densify(road_centrelines(tile), 0.5); rtree = cKDTree(rxy)
    offs = {}
    for k, name in enumerate(CHANNELS):
        gxy, gtan = densify(cb[cb.ssclasse.isin(list(CHANNELS.values())[k])].geometry.clip(reg).values, 0.02)
        pxy = xy[ch == k]
        d, j = cKDTree(gxy).query(pxy, distance_upper_bound=0.30); ok = np.isfinite(d)
        n = np.stack([-gtan[j[ok], 1], gtan[j[ok], 0]], 1)
        _, ri = rtree.query(gxy[j[ok]])
        n *= np.sign((n * (rxy[ri] - gxy[j[ok]])).sum(1))[:, None]
        o = ((pxy[ok] - gxy[j[ok]]) * n).sum(1) * 100
        offs[name] = {"median_cm": float(np.median(o)), "iqr_cm": [float(np.percentile(o, 25)), float(np.percentile(o, 75))]}
    res["signed_offset_towards_road"] = offs

    res["model_labels"] = same_class_metrics(tile, lines, chan)
    new, info = reassign(lines, chan, road_centrelines(tile))
    res["fix_road_side_relabel"] = {"changed_share": float(info.changed.mean()), **same_class_metrics(tile, lines, new)}
    o = load_ortho("ortho2023", tile)
    grad = gradient_magnitude(o.read_all().astype(np.float32).mean(0), 1.0)
    res["fix_snap_to_edge"] = same_class_metrics(tile, snap_lines(lines, grad, o.transform), chan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = gpd.read_file(a.lines)
    out = {t: analyse_tile(t, L[L.tile == t].reset_index(drop=True)) for t in sorted(L.tile.unique())}
    txt = json.dumps(out, indent=1, default=float)
    print(txt)
    Path(a.out or Path(a.lines).with_suffix(".error_analysis.json")).write_text(txt)


if __name__ == "__main__":
    main()
