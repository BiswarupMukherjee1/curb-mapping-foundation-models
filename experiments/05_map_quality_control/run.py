"""Quality control of an existing PCRS against the 2023 imagery, using a trained curb model.

Support of a map line = mean over its points (every 5 cm) of the model probability for its class (bottom / top),
max-pooled over ±1 px. Correct, visible lines should have high support; displaced or outdated lines low support.

A. Controlled edits on TEST-block PCRS curbs: lines cut into 5 m pieces; each piece kept (control) or shifted
   perpendicular by 5 / 10 / 20 / 50 cm (random side). AUROC(shifted vs untouched) per magnitude, on visible and on
   all pieces; false-alarm rate on untouched pieces at the threshold that detects 80 % of 20 cm shifts.
B. Missing lines: 10 % of the pieces are deleted from the map; model lines (selected threshold) with no map line within
   20 cm are "missing-line candidates". Recall of deleted length vs candidate length elsewhere (false alarms per km).
C. Real case: support per delivery lot outside the training blocks, including the lots excluded by the data audit because their
   PCRS disagreed with the ortho (C12, BC02, BC03, 30, BC06). Do they rank lowest?
Usage: PYTHONPATH=src python experiments/05_map_quality_control/run.py --model-run experiments/03_learned_models/runs/<run>
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
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
import torch  # noqa: E402
from rasterio.features import rasterize  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from pcrs_ai import protocol  # noqa: E402
from pcrs_ai.align import densify  # noqa: E402
from pcrs_ai.io import load_pcrs  # noqa: E402
from pcrs_ai.gpu import wait_for_memory  # noqa: E402
from pcrs_ai.models import load_trained, needs_nir  # noqa: E402
from pcrs_ai.paths import DATA_PROCESSED  # noqa: E402
from pcrs_ai.splits import EXCLUDED_LOTS  # noqa: E402
from pcrs_ai.tiledata import CHANNELS, SPLITS, load_tile  # noqa: E402

spec = importlib.util.spec_from_file_location("train", ROOT / "experiments/03_learned_models/train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
SHIFTS = [0.05, 0.10, 0.20, 0.50]
PIECE_M = 5.0


def cut(line, L=PIECE_M):
    n = max(1, int(round(line.length / L)))
    return [shapely.ops.substring(line, i * line.length / n, (i + 1) * line.length / n) for i in range(n)]


def support(lines, chans, probmax, transform):
    """Mean max-pooled probability of each line's own class, sampled every 5 cm."""
    xy, _, ids = densify(lines, 0.05, return_ids=True)
    inv = ~transform
    c = np.clip(np.floor(inv.a * xy[:, 0] + inv.c).astype(int), 0, probmax.shape[2] - 1)
    r = np.clip(np.floor(inv.e * xy[:, 1] + inv.f).astype(int), 0, probmax.shape[1] - 1)
    v = probmax[np.asarray(chans)[ids], r, c].astype(np.float32) / 255.0
    s = pd.Series(v).groupby(ids).mean()
    out = np.full(len(lines), np.nan); out[s.index.values] = s.values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-run", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = Path(a.model_run)
    cfg = json.loads((src / "config.json").read_text()); s02 = json.loads((src / ("summary_rescored.json" if (src / "summary_rescored.json").exists() else "summary.json")).read_text())
    run = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{src.name.split('_', 1)[1]}"
    run.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    wait_for_memory(6.5)
    _, enc, head = load_trained(src, dev)
    res = {"model_run": src.name}
    piecesA, lotrows = [], []
    for t in cfg["test_tiles"]:
        td = load_tile(t, with_nir=needs_nir(cfg["encoder"]))
        lots = load_pcrs(t, "cr_zone_couverture_s")
        ex = lots[lots.lot.isin(EXCLUDED_LOTS.get(t, []))]
        if len(ex):     # also predict over excluded lots (never trained on: outside every split region)
            m = rasterize([(g, 1) for g in ex.geometry], out_shape=td.shape, transform=td.transform, fill=0, dtype="uint8")
            td.split[(m > 0) & (td.split == 0)] = 9
        prob, _ = train.predict_tile(td, enc, head, dev, [SPLITS["val"], SPLITS["test"], 9])
        pmax = np.stack([ndimage.maximum_filter(prob[k], size=3) for k in range(prob.shape[0])])
        cb = protocol.curbs(t)
        reg = protocol.region(t, "test")
        # ---------------- A: controlled shifts on test pieces
        pieces, chans = [], []
        for k, subs in enumerate(CHANNELS.values()):
            for g in cb[cb.ssclasse.isin(subs)].geometry.clip(reg):
                for part in getattr(g, "geoms", [g]):
                    if part.length >= 2.0:
                        for p in cut(part):
                            pieces.append(p); chans.append(k)
        n = len(pieces)
        kind = rng.choice(["control"] * 4 + [f"shift_{int(s * 100)}cm" for s in SHIFTS] + ["delete"], size=n)
        edited = []
        for p, kd in zip(pieces, kind):
            if kd.startswith("shift"):
                d = float(kd.split("_")[1][:-2]) / 100 * rng.choice([-1, 1])
                q = shapely.offset_curve(p, d)
                edited.append(q if not q.is_empty else p)
            else:
                edited.append(p)
        mid = np.array([p.interpolate(0.5, normalized=True).coords[0][:2] for p in pieces])
        rs = protocol.reason_of(t, mid)
        sup = support(edited, chans, pmax, td.transform)
        piecesA.append(pd.DataFrame({"tile": t, "kind": kind, "support": sup, "visible": np.isin(rs, protocol.REASONS[:2]),
                                     "reason": rs, "channel": chans, "length": [p.length for p in pieces]}))
        # ---------------- B: missing lines (deleted pieces)
        ls, _, ch = train.lines_for_split(td, prob, SPLITS["test"], s02["threshold"])
        kept = [e for e, kd in zip(edited, kind) if kd != "delete"]
        deleted = [p for p, kd in zip(pieces, kind) if kd == "delete"]
        mxy, _ = densify(kept, 0.05)
        pxy, _, pid = densify(ls, 0.05, return_ids=True)
        far = cKDTree(mxy).query(pxy, distance_upper_bound=0.20)[0] == np.inf if len(mxy) else np.ones(len(pxy), bool)
        cand_xy = pxy[far]
        dxy, _ = densify(deleted, 0.05)
        rec = (cKDTree(cand_xy).query(dxy, distance_upper_bound=0.20)[0] < np.inf).mean() if len(cand_xy) and len(dxy) else 0.0
        on_del = (cKDTree(dxy).query(cand_xy, distance_upper_bound=0.30)[0] < np.inf) if len(dxy) and len(cand_xy) else np.zeros(len(cand_xy), bool)
        kept_km = sum(p.length for p in kept) / 1000
        res[f"{t}_missing"] = {"deleted_m": float(len(dxy) * 0.05), "recall_deleted_length": float(rec),
                               "false_candidates_m_per_km_map": float((~on_del).sum() * 0.05 / max(kept_km, 1e-9))}
        # ---------------- C: support per lot outside training blocks
        nontrain = (td.split == SPLITS["val"]) | (td.split == SPLITS["test"]) | (td.split == 9)
        for _, lot in lots.iterrows():
            lc = cb[cb.geometry.intersects(lot.geometry)]
            if not len(lc):
                continue
            ls_l, ch_l = [], []
            for k, subs in enumerate(CHANNELS.values()):
                for g in lc[lc.ssclasse.isin(subs)].geometry.intersection(lot.geometry):
                    for part in getattr(g, "geoms", [g]):
                        if part.geom_type == "LineString" and part.length >= 1.0:
                            ls_l.append(part); ch_l.append(k)
            if not ls_l:
                continue
            xy, _, ids = densify(ls_l, 0.10, return_ids=True)
            inv = ~td.transform
            c = np.floor(inv.a * xy[:, 0] + inv.c).astype(int); r = np.floor(inv.e * xy[:, 1] + inv.f).astype(int)
            ok = (c >= 0) & (r >= 0) & (c < td.shape[1]) & (r < td.shape[0])
            ok[ok] &= nontrain[r[ok], c[ok]]
            vis = np.isin(protocol.reason_of(t, xy), protocol.REASONS[:2])
            sel = ok & vis
            if sel.sum() < 300:
                continue
            v = pmax[np.asarray(ch_l)[ids[sel]], r[sel], c[sel]].astype(np.float32) / 255.0
            lotrows.append({"tile": t, "lot": lot.lot, "etat": lot.etat, "flagged_by_audit": lot.lot in EXCLUDED_LOTS.get(t, []),
                            "n_points": int(sel.sum()), "support": float(v.mean()), "share_low_support": float((v < 0.2).mean())})
        del td, prob, pmax
        print(f"{t}: pieces {n}, missing-line test {res[f'{t}_missing']}", flush=True)

    A = pd.concat(piecesA, ignore_index=True)
    A.to_csv(run / "pieces.csv", index=False)
    auc = {}
    for subset, D in [("visible", A[A.visible]), ("all", A)]:
        ctl = D[D.kind == "control"].support.dropna()
        for s in SHIFTS:
            sh = D[D.kind == f"shift_{int(s * 100)}cm"].support.dropna()
            auc[f"{subset}_{int(s * 100)}cm"] = float(roc_auc_score(np.r_[np.zeros(len(ctl)), np.ones(len(sh))], -np.r_[ctl, sh]))
        sh20 = D[D.kind == "shift_20cm"].support.dropna()
        thr = np.quantile(sh20, 0.8)                      # flag if support <= thr -> detects 80 % of 20 cm shifts
        auc[f"{subset}_false_alarm_at_80pct_detection_20cm"] = float((ctl <= thr).mean())
    res["controlled_shifts_auroc"] = auc
    res["unverifiable_share_of_control_pieces"] = float(1 - A[A.kind == "control"].visible.mean())
    Lt = pd.DataFrame(lotrows)
    Lt["lot"] = Lt["lot"].fillna("(unnamed " + Lt["etat"].str.lower() + ")")
    Lt["w"] = Lt.support * Lt.n_points; Lt["wl"] = Lt.share_low_support * Lt.n_points
    Lt = Lt.groupby(["tile", "lot", "etat", "flagged_by_audit"], as_index=False).agg(n_points=("n_points", "sum"), w=("w", "sum"), wl=("wl", "sum"))
    Lt["support"] = Lt.w / Lt.n_points; Lt["share_low_support"] = Lt.wl / Lt.n_points
    Lt = Lt.drop(columns=["w", "wl"]).sort_values("support").reset_index(drop=True)
    from sklearn.metrics import roc_auc_score as _auc
    res["lots_support_by_city_state"] = Lt.groupby("etat").apply(lambda g: float(np.average(g.support, weights=g.n_points)), include_groups=False).to_dict()
    for name, flag in [("flagged_by_audit", Lt.flagged_by_audit), ("city_state_not_complete", Lt.etat != "CONSTITUE")]:
        if flag.any() and (~flag).any():
            res[f"lots_auroc_low_support_vs_{name}"] = float(_auc(flag, -Lt.support))
    Lt.to_csv(run / "lot_support.csv", index=False)
    res["lots"] = {"n": len(Lt), "excluded_lots_rank": {r.lot: int(i + 1) for i, r in enumerate(Lt.itertuples()) if r.flagged_by_audit}}
    print(json.dumps(res, indent=1, default=float), flush=True)
    print(Lt.head(12).round(3).to_string(index=False), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for subset, st in [("visible", "-o"), ("all", "--o")]:
        ax[0].plot([s * 100 for s in SHIFTS], [auc[f"{subset}_{int(s * 100)}cm"] for s in SHIFTS], st, label=f"{subset} curbs")
    ax[0].axhline(0.5, color="k", lw=0.6, ls=":"); ax[0].set_xlabel("shift applied to map line (cm)"); ax[0].set_ylabel("AUROC shifted vs untouched")
    ax[0].set_xscale("log"); ax[0].set_xticks([5, 10, 20, 50]); ax[0].set_xticklabels(["5", "10", "20", "50"]); ax[0].legend(frameon=False)
    ax[0].set_title("Detecting displaced PCRS lines")
    cols = ["tab:red" if e else "tab:gray" for e in Lt.flagged_by_audit]
    ax[1].bar(range(len(Lt)), Lt.support, color=cols); ax[1].set_xlabel("delivery lots, sorted by support (red = flagged by the data audit)")
    ax[1].set_ylabel("mean model support on visible map lines"); ax[1].set_title("Lot-level QC ranking")
    fig.tight_layout(); fig.savefig(run / "qc.png", dpi=120); plt.close(fig)
    (run / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print("->", run, flush=True)


if __name__ == "__main__":
    main()
