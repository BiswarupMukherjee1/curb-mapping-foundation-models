"""Block-bootstrap confidence intervals for trained models (test squares, 10 cm, both curb lines pooled).

Per 100 m test block: GT length, GT length matched within 10 cm, drawn length, drawn length matched. The bootstrap
resamples blocks (B = 1000) and recomputes completeness / correctness / quality for each run, and paired differences
between runs on the same resample.
Usage: PYTHONPATH=src python scripts/compare_runs.py --runs <tag> <tag> ... [--tiles OH RL] [--ref <tag>]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcrs_ai import paths  # noqa: E402,F401

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pcrs_ai import protocol  # noqa: E402
from pcrs_ai.io import load_tile_polygon  # noqa: E402
from pcrs_ai.metrics import line_metrics  # noqa: E402
from pcrs_ai.splits import blocks_for_tile  # noqa: E402


def block_table(tile: str, pred: gpd.GeoSeries) -> pd.DataFrame:
    blocks = blocks_for_tile(load_tile_polygon(tile))
    blocks = blocks[blocks.split == "test"]
    reg = protocol.region(tile, "test")
    gt = protocol.ground_truth(tile, "test")
    rows = []
    for _, b in blocks.iterrows():
        cell = b.geometry.intersection(reg)
        if cell.is_empty:
            continue
        g = gt.clip(cell); p = pred.clip(cell)
        m = line_metrics(g.values, p.values, tolerances=(0.10,))
        if m["gt_length_m"] == 0 and m["pred_length_m"] == 0:
            continue
        rows.append({"block": f"{tile}_{b.ix}_{b.iy}", "gt": m["gt_length_m"], "gt_ok": m["completeness_10cm"] * m["gt_length_m"] if m["gt_length_m"] else 0.0,
                     "pr": m["pred_length_m"], "pr_ok": m["correctness_10cm"] * m["pred_length_m"] if m["pred_length_m"] else 0.0})
    return pd.DataFrame(rows).set_index("block")


def scores(t: pd.DataFrame) -> dict:
    g, go, p, po = t["gt"].sum(), t["gt_ok"].sum(), t["pr"].sum(), t["pr_ok"].sum()
    return {"completeness": go / g, "correctness": po / max(p, 1e-9), "quality": go / (g + p - po)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--ref", default=None)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="test_lines", help="test_lines (re-scored) or test_lines_as_trained (as trained)")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for tile in a.tiles:
        tabs = {}
        for tag in a.runs:
            f = sorted(glob.glob(str(ROOT / f"data/processed/{a.prefix}_{tile}_*{tag}.gpkg")))
            if not f:
                continue
            tabs[tag] = block_table(tile, gpd.read_file(f[-1]).geometry)
        if not tabs:
            continue
        idx = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))
        tabs = {k: v.loc[idx] for k, v in tabs.items()}
        boots = {k: [] for k in tabs}
        for _ in range(a.B):
            s = rng.choice(len(idx), len(idx), replace=True)
            for k, t in tabs.items():
                boots[k].append(scores(t.iloc[s]))
        ref = a.ref if a.ref in tabs else None
        for k, t in tabs.items():
            pt = scores(t); B = pd.DataFrame(boots[k])
            row = {"tile": tile, "run": k, "n_blocks": len(idx)}
            for m in ("completeness", "correctness", "quality"):
                row[m] = pt[m]; row[f"{m}_lo"], row[f"{m}_hi"] = np.percentile(B[m], [2.5, 97.5])
                if ref and k != ref:
                    d = B[m] - pd.DataFrame(boots[ref])[m]
                    row[f"d_{m}_vs_ref"] = pt[m] - scores(tabs[ref])[m]
                    row[f"d_{m}_lo"], row[f"d_{m}_hi"] = np.percentile(d, [2.5, 97.5])
            rows.append(row)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = ["tile", "run", "n_blocks"] + [c for c in out.columns if c.startswith(("quality", "d_quality", "completeness", "correctness")) and not c.endswith(("_lo", "_hi"))]
    print(out.round(3).to_string(index=False))
    if a.out:
        out.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
