"""Confidence, reason codes and the review curve for the lines drawn by a trained model.

1. Re-run the trained model on VAL and TEST blocks; vectorise at its selected threshold.
2. Per line: share within 10 cm of a same-class PCRS line, confidence statistics, visibility features, reason code.
3. Confidence: (a) mean probability along the line, (b) a logistic model of P(line correct) on those features,
   fitted on VAL lines only (length-weighted), applied to TEST.
4. Review curve on TEST: least-confident first; share of wrong length caught vs share of length reviewed.
5. Wrong-length share per reason code (which situations produce errors).
Usage: PYTHONPATH=src python experiments/04_review_ranking/run.py --model-run experiments/03_learned_models/runs/<run>
"""
from __future__ import annotations

import argparse
import importlib.util
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
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pcrs_ai import protocol, review  # noqa: E402
from pcrs_ai.align import densify  # noqa: E402
from pcrs_ai.gpu import wait_for_memory  # noqa: E402
from pcrs_ai.models import load_trained, needs_nir  # noqa: E402
from pcrs_ai.paths import DATA_PROCESSED  # noqa: E402
from pcrs_ai.tiledata import CHANNELS, SPLITS, load_tile  # noqa: E402

spec = importlib.util.spec_from_file_location("train", ROOT / "experiments/03_learned_models/train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)


def gt_dense(tile, split):
    cb = protocol.curbs(tile); reg = protocol.region(tile, split)
    out = {}
    for k, subs in enumerate(CHANNELS.values()):
        out[k] = densify(cb[cb.ssclasse.isin(subs)].geometry.clip(reg).values, 0.02)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    src = Path(a.model_run)
    cfg = json.loads((src / "config.json").read_text())
    s02 = json.loads((src / ("summary_rescored.json" if (src / "summary_rescored.json").exists() else "summary.json")).read_text())
    run = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{a.tag or src.name.split('_', 1)[1]}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps({"model_run": str(src), **vars(a)}, indent=1))
    dev = "cuda"
    wait_for_memory(6.5)
    _, enc, head = load_trained(src, dev)
    thr = s02["threshold"]
    tables = []
    for t in cfg["test_tiles"]:
        td = load_tile(t, with_nir=needs_nir(cfg["encoder"]))
        prob, _ = train.predict_tile(td, enc, head, dev, [SPLITS["val"], SPLITS["test"]])
        for split in ("val", "test"):
            ls, cf, ch = train.lines_for_split(td, prob, SPLITS[split], thr)
            tab = review.segment_table(t, ls, ch, prob, td.transform, gt_dense(t, split))
            tab["tile"], tab["split"] = t, split
            tab["geometry"] = [ls[i] for i in tab.index]
            tables.append(tab)
            print(f"{t} {split}: {len(tab)} lines, {tab.length_m.sum():.0f} m, correct share "
                  f"{np.average(tab.frac_correct, weights=tab.length_m):.3f}", flush=True)
        del td, prob
    T = pd.concat(tables, ignore_index=True)
    V, X = T[T.split == "val"], T[T.split == "test"].copy()

    sc = StandardScaler().fit(V[review.FEATURES])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(V[review.FEATURES]), (V.frac_correct >= 0.5).astype(int),
                                                       sample_weight=V.length_m)
    X["p_correct"] = clf.predict_proba(sc.transform(X[review.FEATURES]))[:, 1]
    coefs = dict(zip(review.FEATURES, np.round(clf.coef_[0], 3)))

    res = {"model_run": src.name, "threshold": thr, "coefficients": coefs, "per_tile": {}}
    fig, axes = plt.subplots(1, len(X.tile.unique()), figsize=(5 * len(X.tile.unique()), 4), squeeze=False)
    for ax, (t, D) in zip(axes[0], X.groupby("tile")):
        L, F = D.length_m.values, D.frac_correct.values
        curves = {"random order": review.review_curve(np.random.default_rng(0).random(len(D)), L, F),
                  "mean probability": review.review_curve(D.conf_mean.values, L, F),
                  "calibrated confidence": review.review_curve(D.p_correct.values, L, F)}
        r = {"lines": len(D), "drawn_m": float(L.sum()), "wrong_share": float(1 - np.average(F, weights=L))}
        for name, (x, y) in curves.items():
            r[name] = {"auc": review.auc(x, y), **{f"caught_at_{int(s * 100)}pct": review.caught_at(x, y, s) for s in (0.1, 0.2, 0.3, 0.5)}}
            ax.plot(x, y, label=f"{name} (AUC {r[name]['auc']:.2f})")
        rs = D.groupby("reason").apply(lambda g: pd.Series({"drawn_m": g.length_m.sum(),
                                                             "wrong_share": 1 - np.average(g.frac_correct, weights=g.length_m)}),
                                       include_groups=False)
        r["by_reason"] = rs.round(3).to_dict("index")
        res["per_tile"][t] = r
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("share of drawn length reviewed (least confident first)"); ax.set_ylabel("share of wrong length caught")
        ax.set_title(f"{t}: review curve (test)"); ax.legend(fontsize=8, frameon=False)
        print(t, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()})
                             for k, v in r.items() if k != "by_reason"}), flush=True)
        print(t, "by reason:", r["by_reason"], flush=True)
    fig.tight_layout(); fig.savefig(run / "review_curves.png", dpi=120); plt.close(fig)
    print("coefficients:", coefs, flush=True)
    (run / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    out = gpd.GeoDataFrame(X.drop(columns=["geometry"]), geometry=X.geometry.values, crs="EPSG:3945")
    out.to_file(DATA_PROCESSED / f"review_lines_{run.name}.gpkg", driver="GPKG")
    print("->", run, flush=True)


if __name__ == "__main__":
    main()
