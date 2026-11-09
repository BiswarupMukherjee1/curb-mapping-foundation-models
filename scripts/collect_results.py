"""Collect the final result tables into results/ (CSV and JSON), from the run folders and the saved test lines.

Writes
  results/models.csv            every compared model (all labels, 6000 iterations): settings and test scores at 5 / 10 / 20 cm (common threshold grid)
  results/main_comparison.csv   block-bootstrap 95 % CIs, and paired differences against the colour-only network and
                                against frozen DINOv3-sat
  results/review_ranking.json, results/map_quality_control.json, results/explainability.json, results/error_analysis.json
Usage: PYTHONPATH=src python scripts/collect_results.py
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/03_learned_models/runs"
OUT = ROOT / "results"
TILES = {"OH": "Pessac", "RL": "Bordeaux"}

# run tag -> (model name, family, training iterations)
MODELS = {
    "colour_only": ("Colour-only network", "supervised", 6000),
    "unet": ("U-Net ResNet-34, fine-tuned", "supervised", 6000),
    "unet_seed1": ("U-Net ResNet-34, fine-tuned (seed 1)", "supervised", 6000),
    "dinov3_sat": ("DINOv3 ViT-L, satellite", "natural-image FM", 6000),
    "dinov3_sat_seed1": ("DINOv3 ViT-L, satellite (seed 1)", "natural-image FM", 6000),
    "dinov3_web": ("DINOv3 ViT-B, web", "natural-image FM", 6000),
    "anysat": ("AnySat", "EO FM", 6000),
    "dofa": ("DOFA ViT-B", "EO FM", 6000),
}
MAIN = ["colour_only", "anysat", "dofa", "dinov3_web", "dinov3_sat",
        "unet"]


def run_dir(tag):
    d = sorted(glob.glob(str(RUNS / f"*_{tag}")))
    d = [x for x in d if Path(x, "summary_rescored.json").exists()]
    return Path(d[-1]) if d else None


def model_table():
    rows = []
    for tag, (name, fam, iters) in MODELS.items():
        d = run_dir(tag)
        if d is None:
            print("missing", tag)
            continue
        s = json.loads((d / "summary_rescored.json").read_text())
        cfg = json.loads((d / "summary.json").read_text())
        for t, city in TILES.items():
            if t not in s:
                continue
            a = s[t]["all"]
            rows.append({"run": tag, "model": name, "family": fam, "iterations": iters,
                         "labels": cfg["budget"], "labelled_curb_m": round(cfg["labelled_curb_m"]),
                         "trained_on": "+".join(TILES[x] for x in cfg["train_tiles"]),
                         "trainable_params": cfg["head_params"], "threshold": s["threshold"], "test_area": city,
                         **{f"{m}_{tol}": round(a[f"{m}_{tol}"], 4) for m in ("completeness", "correctness", "quality")
                            for tol in ("5cm", "10cm", "20cm")},
                         "completeness_10cm_visible_curbs": round(s[t]["visible"]["completeness_10cm"], 4)})
    return pd.DataFrame(rows)


def bootstrap(ref, out):
    tags = [t for t in MAIN if run_dir(t) is not None]
    subprocess.run([sys.executable, str(ROOT / "scripts/compare_runs.py"), "--ref", ref, "--out", str(out), "--runs", *tags],
                   check=True, cwd=ROOT)
    return pd.read_csv(out)


def latest(pattern):
    f = sorted(glob.glob(str(ROOT / pattern)))
    return json.loads(Path(f[-1]).read_text()) if f else None


def main():
    OUT.mkdir(exist_ok=True)
    m = model_table()
    m.to_csv(OUT / "models.csv", index=False)

    b1 = bootstrap("colour_only", OUT / ".ci_vs_colour.csv")
    b2 = bootstrap("dinov3_sat", OUT / ".ci_vs_dino.csv")
    keep = ["tile", "run", "n_blocks"] + [f"{a}{s}" for a in ("completeness", "correctness", "quality") for s in ("", "_lo", "_hi")]
    main_t = b1[keep + [c for c in b1.columns if c.startswith("d_quality")]].rename(
        columns={"d_quality_vs_ref": "d_quality_vs_colour", "d_quality_lo": "d_quality_vs_colour_lo", "d_quality_hi": "d_quality_vs_colour_hi"})
    d2 = b2[["tile", "run", "d_quality_vs_ref", "d_quality_lo", "d_quality_hi"]].rename(
        columns={"d_quality_vs_ref": "d_quality_vs_dinov3", "d_quality_lo": "d_quality_vs_dinov3_lo", "d_quality_hi": "d_quality_vs_dinov3_hi"})
    main_t = main_t.merge(d2, on=["tile", "run"])
    main_t.insert(2, "model", main_t.run.map(lambda r: MODELS[r][0]))
    main_t.insert(3, "family", main_t.run.map(lambda r: MODELS[r][1]))
    main_t["test_area"] = main_t.tile.map(TILES)
    main_t.drop(columns="tile").round(4).to_csv(OUT / "main_comparison.csv", index=False)
    for f in (OUT / ".ci_vs_colour.csv", OUT / ".ci_vs_dino.csv"):
        f.unlink()

    au = latest("experiments/01_data_audit/runs/*_visible_v2/summary.json")
    if au is not None:
        audit = {}
        for t, city in TILES.items():
            s = au[t]; al = s["alignment"]
            audit[city] = {"surveyed_curb_km_in_tile": round(s["curb_km"], 1),
                           "visible_from_air_share": round(s["visible_frac_all"], 3),
                           "share_by_reason": s["reason_frac"],
                           # the audit stores these offsets in centimetres
                           "ortho_to_pcrs_offset_visible_curbs_cm": {"east": al["ALL_visible"]["dx_m"], "north": al["ALL_visible"]["dy_m"]},
                           "injected_shift_recovery_error_cm": al["recovery_test"]["error_cm"],
                           "per_lot_offset_cm_reliable_lots": {k: round(v, 2) for k, v in s["lots"].items() if k.endswith("_cm_reliable")}}
        (OUT / "data_audit.json").write_text(json.dumps(audit, indent=1))
    for name, pat in [("classical_baseline", "experiments/02_classical_baseline/runs/*_m0_coverage/summary.json"),
                      ("review_ranking", "experiments/04_review_ranking/runs/*_unet/summary.json"),
                      ("map_quality_control", "experiments/05_map_quality_control/runs/*_unet/summary.json"),
                      ("explainability", "experiments/06_explainability/runs/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]/summary.json"),
                      ("error_analysis", "data/processed/review_lines_*_unet.error_analysis.json")]:
        s = latest(pat)
        if s is not None:
            (OUT / f"{name}.json").write_text(json.dumps(s, indent=1, default=str))
        else:
            print("missing", name)
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
