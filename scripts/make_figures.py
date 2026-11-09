"""Figures for the README, generated from results/ (and, for the example map, the open orthophoto and PCRS).

Usage: PYTHONPATH=src python scripts/make_figures.py [--no-map]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8d8c86"
FAMILY = {"supervised": "#2a78d6", "natural-image FM": "#eb6834", "EO FM": "#1baf7a"}
FAMILY_LABEL = {"supervised": "no foundation model", "natural-image FM": "foundation model, natural images",
                "EO FM": "foundation model, Earth observation"}
CITIES = ["Pessac", "Bordeaux"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})


def legend_families(ax, fams, loc="lower right"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAMILY[f], label=FAMILY_LABEL[f]) for f in fams], loc=loc, frameon=False, fontsize=9)


def main_comparison():
    t = pd.read_csv(RES / "main_comparison.csv")
    cl = json.loads((RES / "classical_baseline.json").read_text())
    order = t[t.test_area == "Pessac"].sort_values("quality").model.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, city, code in zip(axes, CITIES, ["OH", "RL"]):
        s = t[t.test_area == city].set_index("model").loc[order]
        y = np.arange(len(s))
        ax.barh(y, s.quality, height=0.62, color=[FAMILY[f] for f in s.family], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(s.quality, y, xerr=[s.quality - s.quality_lo, s.quality_hi - s.quality], fmt="none",
                    ecolor=INK, elinewidth=1, capsize=3)
        for yi, (q, hi) in enumerate(zip(s.quality, s.quality_hi)):
            ax.text(hi + 0.01, yi, f"{q:.2f}", va="center", fontsize=9, color=INK)
        qc = cl[code]["all"]["quality_10cm"]
        ax.axvline(qc, color=NEUTRAL, linestyle="--", linewidth=1.2)
        ax.text(qc + 0.005, len(s) - 0.45, f"classical, no learning: {qc:.2f}", color=INK2, fontsize=8.5)
        ax.set_yticks(y, s.index)
        ax.set_xlim(0, 0.68)
        ax.set_title(city, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("quality at 10 cm (higher is better)")
        ax.grid(axis="y", visible=False)
    legend_families(axes[1], ["supervised", "natural-image FM", "EO FM"])
    fig.tight_layout()
    fig.savefig(FIG / "main_comparison.png", dpi=150)
    plt.close(fig)


def review_and_qc():
    r = json.loads((RES / "review_ranking.json").read_text())["per_tile"]
    q = json.loads((RES / "map_quality_control.json").read_text())["controlled_shifts_auroc"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    xs = [0, 10, 20, 30, 50]
    for city, code, ls in [("Pessac", "OH", "-"), ("Bordeaux", "RL", "--")]:
        c = r[code]["calibrated confidence"]
        ys = [0] + [100 * c[f"caught_at_{p}pct"] for p in xs[1:]]
        ax.plot(xs, ys, ls, color=FAMILY["supervised"], linewidth=2, marker="o", markersize=5, label=f"{city}: model confidence")
    ax.plot([0, 50], [0, 50], color=NEUTRAL, linewidth=1.2, label="random order")
    ax.set_xlabel("share of drawn length reviewed, least confident first (%)")
    ax.set_ylabel("share of wrong length caught (%)")
    ax.set_xlim(0, 52); ax.set_ylim(0, 75)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Review ranking", loc="left", fontsize=11)
    ax = axes[1]
    shifts = [5, 10, 20, 50]
    for key, name, ls in [("visible", "curbs visible from the air", "-"), ("all", "all curbs", "--")]:
        ax.plot(shifts, [q[f"{key}_{s}cm"] for s in shifts], ls, color=FAMILY["supervised"], linewidth=2, marker="o",
                markersize=5, label=name)
    ax.axhline(0.5, color=NEUTRAL, linewidth=1.2)
    ax.text(50, 0.51, "chance", ha="right", va="bottom", color=INK2, fontsize=8.5)
    ax.set_xscale("log"); ax.set_xticks(shifts, [f"{s} cm" for s in shifts]); ax.minorticks_off()
    ax.set_xlabel("displacement injected into the map")
    ax.set_ylabel("AUROC, displaced vs untouched pieces")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Map quality control", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "review_and_qc.png", dpi=150)
    plt.close(fig)


def example_map(run="unet", tile="OH"):
    """Two 100 m test blocks with surveyed PCRS curbs and the lines drawn by the model, chosen by rule among blocks
    with >= 200 m of surveyed curb: 'typical' = completeness at 10 cm closest to the tile-wide value, 'hardest' =
    lowest completeness. Areas outside the evaluation region are greyed out."""
    import geopandas as gpd
    from shapely.geometry import box

    from pcrs_ai import protocol
    from pcrs_ai.io import load_ortho, load_tile_polygon
    from pcrs_ai.metrics import line_metrics
    from pcrs_ai.splits import blocks_for_tile

    pred = gpd.read_file(sorted(ROOT.glob(f"data/processed/test_lines_{tile}_{run}.gpkg"))[-1]).geometry
    gt = protocol.ground_truth(tile, "test")
    reg = protocol.region(tile, "test")
    tile_c = line_metrics(gt.values, pred.values, tolerances=(0.10,))["completeness_10cm"]
    blocks = blocks_for_tile(load_tile_polygon(tile))
    cells = []
    for _, bl in blocks[blocks.split == "test"].iterrows():
        cell = bl.geometry.intersection(reg)
        g = gt.clip(cell)
        if cell.is_empty or g.length.sum() < 200:
            continue
        m = line_metrics(g.values, pred.clip(cell).values, tolerances=(0.10,))
        cells.append((bl.geometry, cell, m["completeness_10cm"], m["correctness_10cm"]))
    typical = min(cells, key=lambda c: abs(c[2] - tile_c))
    hard = min(cells, key=lambda c: c[2])
    o = load_ortho("ortho2023", tile)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4))
    for row, (cell, name) in zip(axes, [(typical, "typical test block"), (hard, "hardest test block")]):
        b, ev = cell[0], cell[1]
        x0, y0, x1, y1 = b.bounds
        c0, r0 = [int(v) for v in o.world_to_pixel(x0, y1)]
        n = int(round((x1 - x0) / o.gsd))
        img = np.moveaxis(o.window(c0, r0, n, n), 0, -1)
        ext = (x0, x0 + n * o.gsd, y1 - n * o.gsd, y1)
        outside = gpd.GeoSeries([box(x0, y0, x1, y1).difference(ev)])
        for ax, title in zip(row, [f"{name}: surveyed curb lines (PCRS)",
                                   f"drawn by the model: completeness {cell[2]:.2f}, correctness {cell[3]:.2f}"]):
            ax.imshow(img, extent=ext)
            outside.plot(ax=ax, color="#ffffff", alpha=0.55, linewidth=0)
            ax.set_title(title, loc="left", fontsize=9.5)
        gt.clip(ev).plot(ax=row[0], color="#ffffff", linewidth=1.0)
        pred.clip(ev).plot(ax=row[1], color="#eda100", linewidth=1.0)
        for ax in row:
            ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
            ax.plot([ext[1] - 24, ext[1] - 4], [ext[2] + 4, ext[2] + 4], color=INK, linewidth=3)
            ax.text(ext[1] - 14, ext[2] + 5.5, "20 m", color=INK, ha="center", fontsize=9)
    fig.text(0.01, 0.005, f"U-Net, Pessac test blocks (tile-wide completeness {tile_c:.2f}); greyed: outside the evaluated "
             "area. Orthophoto 2023 and PCRS: Bordeaux Métropole, Licence Ouverte 2.0.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(FIG / "example_pessac.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-map", action="store_true")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    main_comparison()
    review_and_qc()
    if not a.no_map:
        example_map()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
