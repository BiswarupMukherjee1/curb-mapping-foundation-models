"""Data audit, part 2: visibility of curbs from the air, and orthophoto <-> PCRS alignment on visible curbs only.

1. per curb point (every 10 cm): NDVI (CIR ortho), nDSM (DSM - DTM, 50 cm), brightness -> reason code
2. fraction of curb length visible per sub-class (upper bound for aerial extraction completeness)
3. global offset on visible points (TROT / HAUT / ALL) + recovery test with an injected shift
4. per-lot offsets on visible points with split-half reliability
5. zoomed previews (4 m) and the NDVI / nDSM histograms used to set the thresholds
Usage: PYTHONPATH=src python experiments/01_data_audit/run_visible.py --tiles OH RL
"""
from __future__ import annotations

import argparse
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

from pcrs_ai import visibility as vis  # noqa: E402
from pcrs_ai.align import densify, estimate_offset, gradient_magnitude  # noqa: E402
from pcrs_ai.io import load_ortho, load_pcrs, load_tile_polygon  # noqa: E402
from pcrs_ai.paths import DATA_PROCESSED  # noqa: E402

INJECT = np.array([0.15, -0.10])      # metres (east, north) for the recovery test


def curb_points(tile: str, step=0.10):
    infra = load_pcrs(tile, "cr_objet_infra_l")
    curbs = infra[infra.classe == "BORD"].reset_index(drop=True)
    lots = load_pcrs(tile, "cr_zone_couverture_s")
    xs, subs = [], []
    for sub, g in curbs.groupby("ssclasse"):
        xy, _ = densify(g.geometry.values, step)
        xs.append(xy); subs += [sub] * len(xy)
    xy = np.concatenate(xs)
    subs = np.asarray(subs)
    inside = gpd.GeoSeries(gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=curbs.crs).within(load_tile_polygon(tile).buffer(-2.0)).values
    xy, subs = xy[inside], subs[inside]           # points outside the raster would get no image values
    pts = gpd.GeoDataFrame({"ssclasse": subs}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=curbs.crs)
    j = gpd.sjoin(pts, lots[["lot", "etat", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    return xy, j["ssclasse"].values, j["lot"].values, j["etat"].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--tag", default="visible")
    ap.add_argument("--max-pts", type=int, default=120_000)
    a = ap.parse_args()
    run = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{a.tag}"
    run.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(0)
    summary = {"thresholds": {"NDVI_VEG": vis.NDVI_VEG, "NDSM_ELEVATED_M": vis.NDSM_ELEVATED_M, "SHADOW_LUM": vis.SHADOW_LUM}}
    for t in a.tiles:
        print("==", t, flush=True)
        xy, sub, lot, etat = curb_points(t)
        pv = vis.point_visibility(t, xy)
        reason = pv.reason()
        df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "ssclasse": sub, "lot": lot, "etat": etat,
                           "ndvi": pv.ndvi, "ndsm": pv.ndsm, "ndsm_nbhd": pv.ndsm_nbhd, "lum": pv.lum, "reason": reason})
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PROCESSED / f"curb_points_{t}.csv.gz", index=False, float_format="%.4f")
        sens = {}
        for thr in (0.5, 0.8, 1.5):
            for nm, nd in (("at_point", pv.ndsm), ("nbhd_1.5m", pv.ndsm_nbhd)):
                sens[f"{nm}_ndsm>{thr}"] = float((~(pv.vegetation | (nd > thr))).mean())
        print("visible-fraction sensitivity:", {k: round(v, 3) for k, v in sens.items()}, flush=True)

        vis_tab = (df.assign(n=1).pivot_table(index="ssclasse", columns="reason", values="n", aggfunc="sum", fill_value=0))
        vis_tab["length_km"] = vis_tab.sum(axis=1) * 0.10 / 1000
        vis_tab["visible_frac"] = (vis_tab.get("visible", 0) + vis_tab.get("visible_shadow", 0)) / vis_tab.drop(columns="length_km").sum(axis=1)
        vis_tab.to_csv(run / f"visibility_by_class_{t}.csv")
        print(vis_tab.round(3).to_string(), flush=True)

        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        ax[0].hist(np.clip(pv.ndvi, -0.5, 1), 100); ax[0].axvline(vis.NDVI_VEG, color="r"); ax[0].set_title(f"{t} NDVI at curb points")
        ax[1].hist(np.clip(pv.ndsm, -1, 15), 100); ax[1].axvline(vis.NDSM_ELEVATED_M, color="r"); ax[1].set_yscale("log"); ax[1].set_title("nDSM (m, 1.5 m max)")
        fig.tight_layout(); fig.savefig(run / f"thresholds_{t}.png", dpi=100); plt.close(fig)

        ortho = load_ortho("ortho2023", t)
        t0 = time.time()
        grad = gradient_magnitude(ortho.read_all().astype(np.float32).mean(axis=0), 1.0)
        res = {"load_s": round(time.time() - t0, 1)}
        visible = pv.visible
        for name, m in [("ALL_all", np.ones(len(xy), bool)), ("ALL_visible", visible),
                        ("TROT_visible", visible & (sub == "TROT")), ("HAUT_visible", visible & (sub == "HAUT"))]:
            idx = np.flatnonzero(m)
            if len(idx) > a.max_pts:
                idx = rng.choice(idx, a.max_pts, replace=False)
            r = estimate_offset(grad, ortho, xy[idx])
            r.pop("E")
            res[name] = {k: (round(v * 100, 2) if k in ("dx_m", "dy_m") else v) for k, v in r.items()}
            if name == "ALL_visible":
                rr = estimate_offset(grad, ortho, xy[idx] + INJECT); rr.pop("E")
                rec = np.array([rr["dx_m"] - r["dx_m"], rr["dy_m"] - r["dy_m"]])
                res["recovery_test"] = {"injected_cm": (INJECT * 100).tolist(), "recovered_correction_cm": (rec * 100).round(2).tolist(),
                                        "error_cm": ((rec + INJECT) * 100).round(2).tolist()}
        print(json.dumps(res, indent=1), flush=True)

        lots_out = []
        for lname, idx in pd.Series(np.arange(len(xy)))[visible].groupby(lot[visible]).groups.items():
            idx = np.asarray(list(idx))
            if len(idx) < 3000:
                continue
            idx = rng.permutation(idx)[:30_000]
            r = estimate_offset(grad, ortho, xy[idx]); r.pop("E")
            h1 = estimate_offset(grad, ortho, xy[idx[::2]]); h2 = estimate_offset(grad, ortho, xy[idx[1::2]])
            lots_out.append({"lot": lname, "etat": etat[idx[0]], "n_pts": r["n_pts"],
                             "dx_cm": r["dx_m"] * 100, "dy_cm": r["dy_m"] * 100, "sharpness": r["sharpness"],
                             "split_half_diff_cm": float(np.hypot(h1["dx_m"] - h2["dx_m"], h1["dy_m"] - h2["dy_m"]) * 100),
                             "visible_frac": float(visible[lot == lname].mean())})
        L = pd.DataFrame(lots_out)
        L["abs_cm"] = np.hypot(L.dx_cm, L.dy_cm)
        L["reliable"] = (L.split_half_diff_cm < 4) & (L.sharpness > 3)
        L.to_csv(run / f"per_lot_visible_{t}.csv", index=False)
        print(L.round(2).to_string(), flush=True)

        # zoomed previews on visible curb points
        sel = rng.choice(np.flatnonzero(visible), 6, replace=False)
        fig, axes = plt.subplots(1, 6, figsize=(18, 3.4))
        allxy = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "s": sub})
        for ax, k in zip(axes, sel):
            cx, cy = xy[k]; h = 2.0
            c, r_ = ortho.world_to_pixel(cx - h, cy + h)
            w = int(2 * h / ortho.gsd)
            ax.imshow(ortho.window(int(c), int(r_), w, w).transpose(1, 2, 0), extent=[cx - h, cx + h, cy - h, cy + h])
            near = allxy[(abs(allxy.x - cx) < h) & (abs(allxy.y - cy) < h)]
            for s_, col in [("TROT", "yellow"), ("HAUT", "cyan"), ("CHAU", "magenta"), ("FRAN", "orange"), ("CIRCU", "lime")]:
                q = near[near.s == s_]
                ax.scatter(q.x, q.y, s=1.5, c=col, label=s_)
            ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h); ax.set_xticks([]); ax.set_yticks([])
        axes[0].legend(fontsize=6, markerscale=4, loc="lower left")
        fig.suptitle(f"{t}: visible PCRS curb points on 5 cm ortho (4 m crops; yellow = bottom/gutter, cyan = top)", fontsize=9)
        fig.tight_layout(); fig.savefig(run / f"zoom_{t}.png", dpi=120); plt.close(fig)

        summary[t] = {"n_points": len(xy), "curb_km": len(xy) * 0.10 / 1000, "visible_frac_sensitivity": sens,
                      "visible_frac_all": float(visible.mean()),
                      "reason_frac": pd.Series(reason).value_counts(normalize=True).round(4).to_dict(),
                      "alignment": res,
                      "lots": {"n": len(L), "n_reliable": int(L.reliable.sum()),
                               "median_abs_cm_reliable": float(L[L.reliable].abs_cm.median()) if L.reliable.any() else None,
                               "p90_abs_cm_reliable": float(L[L.reliable].abs_cm.quantile(.9)) if L.reliable.any() else None,
                               "max_abs_cm_reliable": float(L[L.reliable].abs_cm.max()) if L.reliable.any() else None}}
        del grad
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print("->", run)


if __name__ == "__main__":
    main()
