"""Evaluation protocol and the classical (non-learned) baseline.

Protocol: 100 m blocks, train/val/test by hash, evaluation inside blocks shrunk by 2 m; ground truth
= PCRS curbs (classe BORD, all sub-classes) minus the lots found inconsistent in the data audit. Metrics: completeness / correctness / quality
at 5 / 10 / 20 cm (src/pcrs_ai/metrics.py), on all GT and on GT visible from the air (visibility reason codes), plus
completeness per reason code.
M0: LSD segments + contrast / NDVI / nDSM / parallel-partner cues; thresholds tuned on VAL blocks only, reported on TEST.
Usage: PYTHONPATH=src python experiments/02_classical_baseline/run.py --tiles OH RL
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pcrs_ai import paths  # noqa: E402,F401

import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from pcrs_ai.align import densify  # noqa: E402
from pcrs_ai.baselines import classical as m0  # noqa: E402
from pcrs_ai.io import load_dem, load_ortho, load_pcrs, load_tile_polygon  # noqa: E402
from pcrs_ai.metrics import line_metrics  # noqa: E402
from pcrs_ai.paths import DATA_PROCESSED  # noqa: E402
from pcrs_ai.splits import EXCLUDED_LOTS, blocks_for_tile, split_region  # noqa: E402

GRID = {"min_len_px": [10, 20, 40], "min_contrast": [6.0, 12.0, 20.0, 30.0], "max_ndsm": [0.3, 1.0],
        "require_pair": [False, True]}
REASONS = ["visible", "visible_shadow", "vegetation", "elevated_object"]


def tile_segments(tile: str) -> m0.Segments:
    cache = DATA_PROCESSED / f"m0_segments_{tile}.npz"
    if cache.exists():
        z = np.load(cache)
        return m0.Segments(z["xy"], {k[2:]: z[k] for k in z.files if k.startswith("f_")})
    t0 = time.time()
    rgb = load_ortho("ortho2023", tile)
    lum = rgb.read_all().astype(np.float32).mean(axis=0)
    seg = m0.detect_lsd(lum.astype(np.uint8))
    print(f"  {tile}: {len(seg)} LSD segments >= 10 px ({time.time() - t0:.0f}s)", flush=True)
    ir = load_ortho("orthoir2023", tile).read_all().astype(np.float32)
    ndvi = (ir[0] - ir[1]) / (ir[0] + ir[1] + 1e-6)
    del ir
    dsm, dtr = load_dem("mns2023", tile)
    dtm, _ = load_dem("mnt2023", tile)
    ndsm = np.nan_to_num(dsm - dtm, nan=0.0)
    inv = ~dtr
    tr = rgb.transform

    def ndsm_at(pix):
        x = tr.a * (pix[:, 0] + 0.5) + tr.c; y = tr.e * (pix[:, 1] + 0.5) + tr.f
        c = np.clip(np.floor(inv.a * x + inv.c).astype(int), 0, ndsm.shape[1] - 1)
        r = np.clip(np.floor(inv.e * y + inv.f).astype(int), 0, ndsm.shape[0] - 1)
        return ndsm[r, c]

    f = m0.segment_features(seg, lum, ndvi, ndsm_at)
    cand = (f["contrast"] >= min(GRID["min_contrast"])) & (f["ndvi_on"] <= 0.2) & (f["ndsm_mid"] <= max(GRID["max_ndsm"]))
    pair = np.zeros(len(seg), bool)
    pair[cand] = m0.has_parallel_partner(seg[cand], f["angle"][cand])
    f["pair"] = pair
    print(f"  {tile}: features + pairs done ({time.time() - t0:.0f}s); candidates {cand.sum()}, with partner {pair.sum()}", flush=True)
    np.savez_compressed(cache, xy=seg, **{f"f_{k}": v for k, v in f.items()})
    return m0.Segments(seg, f)


def ground_truth(tile: str, region) -> gpd.GeoSeries:
    infra = load_pcrs(tile, "cr_objet_infra_l")
    curbs = infra[infra.classe == "BORD"]
    lots = load_pcrs(tile, "cr_zone_couverture_s")
    # the PCRS only exists inside surveyed street corridors: evaluate inside complete lots only
    covered = lots[(lots.etat == "CONSTITUE") & ~lots.lot.isin(EXCLUDED_LOTS.get(tile, []))].geometry.union_all()
    region = region.intersection(covered)
    ex = lots[lots.lot.isin(EXCLUDED_LOTS.get(tile, []))]
    if len(ex):
        region = region.difference(ex.geometry.union_all())
    return curbs.geometry.clip(region), region


def reason_lookup(tile: str):
    P = pd.read_csv(DATA_PROCESSED / f"curb_points_{tile}.csv.gz", usecols=["x", "y", "reason"])
    tree = cKDTree(P[["x", "y"]].values)
    reasons = P.reason.values

    def f(xy):
        _, j = tree.query(xy, k=1)
        return reasons[j]
    return f


def evaluate(gt, pred, reason_of) -> dict:
    out = {"all": line_metrics(gt.values, pred.values)}
    out["visible"] = line_metrics(gt.values, pred.values,
                                  gt_weight=lambda xy: np.isin(reason_of(xy), ["visible", "visible_shadow"]).astype(float))
    for r in REASONS:
        m = line_metrics(gt.values, pred.values, tolerances=(0.10,), gt_weight=lambda xy, r=r: (reason_of(xy) == r).astype(float))
        out[f"completeness_10cm_{r}"] = m["completeness_10cm"]
        out[f"gt_km_{r}"] = m["gt_length_weighted_m"] / 1000
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--tag", default="m0")
    a = ap.parse_args()
    run = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{a.tag}"
    run.mkdir(parents=True, exist_ok=False)
    data = {}
    for t in a.tiles:
        print("==", t, flush=True)
        tp = load_tile_polygon(t)
        blocks = blocks_for_tile(tp)
        seg = tile_segments(t)
        loose = m0.select(seg, min_len_px=min(GRID["min_len_px"]), min_contrast=min(GRID["min_contrast"]),
                          max_ndsm=max(GRID["max_ndsm"]))
        cand_idx = np.flatnonzero(loose)
        lines = gpd.GeoSeries(m0.to_world_lines(seg.xy[cand_idx], load_ortho("ortho2023", t)), index=cand_idx, crs=blocks.crs)
        regions, gts, clipped, dense = {}, {}, {}, {}
        for s in ("val", "test"):
            gts[s], regions[s] = ground_truth(t, split_region(blocks, s))
            clipped[s] = lines.clip(regions[s])                       # keeps the segment index
            dense[s] = densify(gts[s].values, 0.02)
        print(f"  {t}: {len(cand_idx)} candidate segments; val/test clipped {len(clipped['val'])}/{len(clipped['test'])}", flush=True)
        data[t] = {"seg": seg, "clipped": clipped, "regions": regions, "gts": gts, "dense": dense, "reason_of": reason_lookup(t)}

    # ---- tune on VAL (pooled over tiles), objective quality@10cm on all GT
    rows = []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(GRID.keys(), vals))
        q_num = q_den = 0.0
        per = {}
        for t, d in data.items():
            m = m0.select(d["seg"], **p)
            cv = d["clipped"]["val"]
            pred = cv[m[cv.index]]
            r = line_metrics(None, pred.values, tolerances=(0.10,), gt_dense=d["dense"]["val"])
            per[t] = r
            gl, pl = r["gt_length_m"], r["pred_length_m"]
            tp_ = r["completeness_10cm"] * gl
            fp_ = (1 - r["correctness_10cm"]) * pl if pl > 0 else 0.0
            q_num += tp_; q_den += gl + fp_
        rows.append({**p, "val_quality_10cm": q_num / q_den if q_den else 0.0,
                     **{f"{t}_compl": per[t]["completeness_10cm"] for t in per},
                     **{f"{t}_corr": per[t]["correctness_10cm"] for t in per}})
        print("  val", p, round(rows[-1]["val_quality_10cm"], 4), flush=True)
    tune = pd.DataFrame(rows).sort_values("val_quality_10cm", ascending=False)
    tune.to_csv(run / "val_grid.csv", index=False)
    best = {k: (bool(v) if k == "require_pair" else v) for k, v in tune.iloc[0][list(GRID)].to_dict().items()}
    best["min_len_px"] = int(best["min_len_px"])
    print("best (val):", best, flush=True)

    # ---- report on TEST
    summary = {"best_params_val": best, "grid_size": len(rows)}
    for t, d in data.items():
        m = m0.select(d["seg"], **best)
        ct = d["clipped"]["test"]
        pred = ct[m[ct.index]]
        res = evaluate(d["gts"]["test"], pred, d["reason_of"])
        summary[t] = res
        print(t, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["all"].items()}), flush=True)
        print(t, "visible:", json.dumps({k: round(v, 3) for k, v in res["visible"].items() if "completeness" in k or "correctness" in k}), flush=True)
        print(t, "by reason:", {k: round(v, 3) for k, v in res.items() if k.startswith(("completeness_10cm_", "gt_km_"))}, flush=True)
        gpd.GeoDataFrame(geometry=pred, crs=pred.crs).to_file(DATA_PROCESSED / f"m0_pred_test_{t}.gpkg", driver="GPKG")
        # figure: one dense test block
        b = blocks_for_tile(load_tile_polygon(t))
        tb = b[b.split == "test"]
        dens = [d["gts"]["test"].clip(g).length.sum() for g in tb.geometry]
        g = tb.geometry.iloc[int(np.argmax(dens))]
        cx, cy = g.centroid.x, g.centroid.y; h = 15.0
        o = load_ortho("ortho2023", t)
        c, r_ = o.world_to_pixel(cx - h, cy + h)
        w = int(2 * h / o.gsd)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(o.window(int(c), int(r_), w, w).transpose(1, 2, 0), extent=[cx - h, cx + h, cy - h, cy + h])
        for ln in d["gts"]["test"].cx[cx - h:cx + h, cy - h:cy + h]:
            for part in getattr(ln, "geoms", [ln]):
                ax.plot(*part.xy, color="yellow", lw=1.2)
        for ln in pred.cx[cx - h:cx + h, cy - h:cy + h]:
            for part in getattr(ln, "geoms", [ln]):
                ax.plot(*part.xy, color="red", lw=0.9)
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{t} test block: PCRS curbs (yellow) vs M0 (red), 30 m", fontsize=9)
        fig.tight_layout(); fig.savefig(run / f"m0_test_{t}.png", dpi=110); plt.close(fig)
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("->", run)


if __name__ == "__main__":
    main()
