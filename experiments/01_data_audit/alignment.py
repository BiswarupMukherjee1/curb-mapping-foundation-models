"""Data audit, part 1: is the open data usable for 10 cm curb work?

A. inventory: raster grids, PCRS classes and lengths, delivery-lot states, object dates vs the Sep-2023 flight
B. RGB <-> infrared co-registration (phase correlation on random blocks, after applying both world files)
C. ortho <-> PCRS alignment: global and per-lot (dx, dy) that maximises edge energy under curb lines (BORD TROT / HAUT)
D. preview crops with the PCRS overlaid
Usage: PYTHONPATH=src python experiments/01_data_audit/run.py --tiles OH RL
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pcrs_ai import paths  # noqa: E402,F401  (cache redirection)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pcrs_ai.align import densify, edge_energy_grid, gradient_magnitude, peak_subpixel, phase_correlation  # noqa: E402
from pcrs_ai.io import load_ortho, load_pcrs  # noqa: E402

FLIGHT = pd.Timestamp("2023-09-01", tz="UTC")
CURB_SUB = {"TROT": "curb bottom (gutter line)", "HAUT": "curb top", "CHAU": "roadway curb", "FRAN": "mountable curb",
            "CIRCU": "lane/cycle curb"}


def inventory(tile: str) -> dict:
    infra = load_pcrs(tile, "cr_objet_infra_l")
    delim = load_pcrs(tile, "cr_objet_delim_l")
    lots = load_pcrs(tile, "cr_zone_couverture_s")
    rows = []
    for name, g in (("infra", infra), ("delim", delim)):
        L = g.geometry.length
        for (c, s), idx in g.groupby(["classe", "ssclasse"]).groups.items():
            rows.append({"layer": name, "classe": c, "ssclasse": s, "n": len(idx), "length_m": float(L.loc[idx].sum())})
    inv = pd.DataFrame(rows).sort_values("length_m", ascending=False)
    curbs = infra[infra.classe == "BORD"].copy()
    md = pd.to_datetime(curbs["mdate"], utc=True, errors="coerce")
    cd = pd.to_datetime(curbs["cdate"], utc=True, errors="coerce")
    return {
        "inventory": inv,
        "lots_state": lots["etat"].value_counts().to_dict(),
        "lots_reception_years": pd.to_datetime(lots["date_reception"], errors="coerce").dt.year.value_counts().sort_index().to_dict(),
        "curb_total_km": float(curbs.geometry.length.sum() / 1000),
        "curb_created_years": cd.dt.year.value_counts().sort_index().to_dict(),
        "curb_modified_after_flight_frac": float((md > FLIGHT).mean()),
        "curbs": curbs, "lots": lots,
    }


def rgb_ir_registration(tile: str, n_blocks: int = 24, size: int = 512, seed: int = 0) -> dict:
    rgb, ir = load_ortho("ortho2023", tile), load_ortho("orthoir2023", tile)
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_blocks):
        c0, r0 = int(rng.integers(size, rgb.width - 2 * size)), int(rng.integers(size, rgb.height - 2 * size))
        x, y = rgb.transform * (c0, r0)                            # world coords of the block corner
        ci, ri = ir.world_to_pixel(x, y)
        ci0, ri0 = int(np.floor(ci)), int(np.floor(ri))
        frac = (ci - ci0, ri - ri0)                                  # world-file-implied residual (IR px)
        a = rgb.window(c0, r0, size, size).astype(np.float32)       # (3,H,W)
        b = ir.window(ci0, ri0, size, size).astype(np.float32)
        best = None
        for ka in range(3):                                          # pick the band pair that correlates best
            for kb in range(3):
                cc = np.corrcoef(a[ka].ravel(), b[kb].ravel())[0, 1]
                if best is None or abs(cc) > abs(best[0]):
                    best = (cc, ka, kb)
        if abs(best[0]) < 0.5:
            continue
        dx, dy = phase_correlation(a[best[1]], b[best[2]])         # b(x) ≈ a(x - d) in IR window pixels
        # residual = measured shift minus the offset the world files already predict
        res.append({"dx_px": dx + frac[0], "dy_px": dy + frac[1], "corr": best[0], "bands": (best[1], best[2])})
    df = pd.DataFrame(res)
    gsd = rgb.gsd
    return {"n_blocks": len(df), "median_dx_cm": float(df.dx_px.median() * gsd * 100) if len(df) else None,
            "median_dy_cm": float(df.dy_px.median() * gsd * 100) if len(df) else None,
            "iqr_dx_cm": float(df.dx_px.quantile(.75) - df.dx_px.quantile(.25)) * gsd * 100 if len(df) else None,
            "band_pairs": {str(k): int(v) for k, v in df.bands.value_counts().items()} if len(df) else {},
            "rgb_origin": [rgb.transform.c, rgb.transform.f], "ir_origin": [ir.transform.c, ir.transform.f]}


def pcrs_alignment(tile: str, curbs, lots, step_m=0.10, max_pts=120_000, search_m=0.40, grid_m=0.02, seed=0) -> dict:
    ortho = load_ortho("ortho2023", tile)
    t0 = time.time()
    gray = ortho.read_all().astype(np.float32).mean(axis=0)
    grad = gradient_magnitude(gray, sigma=1.0)
    del gray
    shifts_px = np.arange(-search_m, search_m + 1e-9, grid_m) / ortho.gsd
    out = {"t_load_s": round(time.time() - t0, 1)}
    rng = np.random.default_rng(seed)

    def search(geoms, cap):
        xy, _ = densify(geoms, step_m)
        if len(xy) == 0:
            return None
        if len(xy) > cap:
            xy = xy[rng.choice(len(xy), cap, replace=False)]
        col, row = ortho.world_to_pixel(xy[:, 0], xy[:, 1])
        col, row = col - 0.5, row - 0.5                               # corner coords -> centre-index coords
        ok = (col > 20) & (row > 20) & (col < ortho.width - 20) & (row < ortho.height - 20)
        E = edge_energy_grid(grad, col[ok], row[ok], shifts_px)
        dx, dy, sharp = peak_subpixel(E, shifts_px)
        # image shift (dx east, dy south in px) that best fits the lines -> PCRS-to-image offset in metres (x east, y north)
        return {"n_pts": int(ok.sum()), "dx_cm": dx * ortho.gsd * 100, "dy_cm": -dy * ortho.gsd * 100,
                "sharpness": sharp, "E_gain": float(E.max() / E[len(E) // 2, len(E) // 2]), "E": E}

    for sub in ["TROT", "HAUT", "ALL"]:
        g = curbs if sub == "ALL" else curbs[curbs.ssclasse == sub]
        r = search(g.geometry.values, max_pts)
        if r:
            out[f"global_{sub}"] = {k: v for k, v in r.items() if k != "E"}
            if sub == "ALL":
                out["_E_all"] = r["E"]
    per_lot = []
    import geopandas as gpd
    j = gpd.sjoin(curbs[["geometry"]], lots[["lot", "etat", "date_reception", "geometry"]], predicate="intersects")
    for lot, g in j.groupby("lot"):
        r = search(g.geometry.values, 20_000)
        if r and r["n_pts"] >= 1500:
            per_lot.append({"lot": lot, "etat": g.etat.iloc[0], "date_reception": str(g.date_reception.iloc[0])[:10],
                            **{k: v for k, v in r.items() if k != "E"}})
    out["per_lot"] = pd.DataFrame(per_lot)
    out["t_total_s"] = round(time.time() - t0, 1)
    return out


def previews(tile: str, curbs, shift_cm, out_dir: Path, n=4, half_m=10.0, seed=1):
    ortho = load_ortho("ortho2023", tile)
    rng = np.random.default_rng(seed)
    trot = curbs[curbs.ssclasse.isin(["TROT", "HAUT"])]
    picks = trot.sample(n=min(n, len(trot)), random_state=seed)
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4.4))
    for ax, (_, row) in zip(np.atleast_1d(axes), picks.iterrows()):
        cx, cy = row.geometry.interpolate(0.5, normalized=True).coords[0][:2]
        c, r = ortho.world_to_pixel(cx - half_m, cy + half_m)
        w = int(2 * half_m / ortho.gsd)
        img = ortho.window(int(c), int(r), w, w).transpose(1, 2, 0)
        ext = [cx - half_m, cx + half_m, cy - half_m, cy + half_m]
        ax.imshow(img, extent=ext)
        near = trot.cx[ext[0]:ext[1], ext[2]:ext[3]]
        for _, s in near.iterrows():
            col = "yellow" if s.ssclasse == "TROT" else "cyan"
            for part in getattr(s.geometry, "geoms", [s.geometry]):
                x, y = part.xy
                ax.plot(x, y, color=col, lw=0.8)
                if shift_cm is not None:
                    ax.plot(np.array(x) + shift_cm[0] / 100, np.array(y) + shift_cm[1] / 100, color="red", lw=0.6, ls="--")
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{tile}: PCRS curbs on ortho 2023 (yellow = bottom/gutter, cyan = top; red dashed = after global shift)", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / f"preview_{tile}.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--tag", default="audit")
    a = ap.parse_args()
    run = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{a.tag}"
    run.mkdir(parents=True, exist_ok=False)
    summary = {}
    for t in a.tiles:
        print(f"== {t}", flush=True)
        inv = inventory(t)
        inv["inventory"].to_csv(run / f"inventory_{t}.csv", index=False)
        reg = rgb_ir_registration(t)
        print("  rgb-ir:", reg, flush=True)
        al = pcrs_alignment(t, inv["curbs"], inv["lots"])
        al["per_lot"].to_csv(run / f"per_lot_shift_{t}.csv", index=False)
        E = al.pop("_E_all")
        np.savetxt(run / f"energy_grid_ALL_{t}.csv", E, delimiter=",", fmt="%.4f")
        pl = al.pop("per_lot")
        g = al["global_ALL"]
        previews(t, inv["curbs"], (g["dx_cm"], g["dy_cm"]), run)
        summary[t] = {
            "curb_total_km": inv["curb_total_km"], "lots_state": inv["lots_state"],
            "lots_reception_years": {str(k): v for k, v in inv["lots_reception_years"].items()},
            "curb_created_years": {str(k): v for k, v in inv["curb_created_years"].items()},
            "curb_modified_after_flight_frac": inv["curb_modified_after_flight_frac"],
            "rgb_ir": reg, "alignment": al,
            "per_lot": {"n_lots": len(pl),
                        "median_abs_offset_cm": float(np.hypot(pl.dx_cm, pl.dy_cm).median()) if len(pl) else None,
                        "p90_abs_offset_cm": float(np.hypot(pl.dx_cm, pl.dy_cm).quantile(.9)) if len(pl) else None},
        }
        print(json.dumps(summary[t], indent=1, default=str), flush=True)
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("->", run)


if __name__ == "__main__":
    main()
