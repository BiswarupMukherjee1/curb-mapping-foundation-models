"""Per-segment correctness, features, reason codes and the review curve.

Review curve: segments are sorted by confidence (least confident first). For each reviewed share x of the drawn length,
y = share of the *wrong* drawn length (points farther than the tolerance from a same-class PCRS line) that the reviewer
has seen. Random ordering gives y = x; a useful confidence gives y >> x for small x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .align import densify
from .metrics import _match
from .visibility import NDSM_ELEVATED_M, NDVI_VEG, SHADOW_LUM, point_visibility

FEATURES = ["conf_mean", "conf_min", "conf_std", "log_len", "frac_veg", "frac_elev", "frac_shadow", "lum_mean", "is_top"]
REASON_ORDER = ["vegetation", "elevated_object", "shadow", "bright_marking", "short", "clear"]


def segment_table(tile: str, lines, chan: np.ndarray, prob_u8: np.ndarray, transform, gt_dense_by_chan: dict,
                  step: float = 0.05, tol: float = 0.10) -> pd.DataFrame:
    """One row per predicted line: length, share correct at `tol`, confidence statistics, visibility features."""
    xy, tg, ids = densify(lines, step, return_ids=True)
    ok = np.zeros(len(xy), bool)
    for k, (gxy, gt_) in gt_dense_by_chan.items():
        m = chan[ids] == k
        if m.any() and len(gxy):
            ok[m], _ = _match(xy[m], tg[m], gxy, gt_, tol, 30.0)
    inv = ~transform
    col = np.floor(inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c).astype(int)
    row = np.floor(inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f).astype(int)
    col = np.clip(col, 0, prob_u8.shape[2] - 1); row = np.clip(row, 0, prob_u8.shape[1] - 1)
    p = prob_u8[chan[ids], row, col].astype(np.float32) / 255.0
    pv = point_visibility(tile, xy)
    d = pd.DataFrame({"seg": ids, "ok": ok, "p": p, "veg": pv.ndvi > NDVI_VEG, "elev": pv.ndsm > NDSM_ELEVATED_M,
                      "shadow": pv.lum < SHADOW_LUM, "lum": pv.lum})
    g = d.groupby("seg")
    t = pd.DataFrame({
        "n": g.size(), "frac_correct": g.ok.mean(), "conf_mean": g.p.mean(), "conf_min": g.p.min(), "conf_std": g.p.std().fillna(0),
        "frac_veg": g.veg.mean(), "frac_elev": g.elev.mean(), "frac_shadow": g.shadow.mean(), "lum_mean": g.lum.mean()})
    t["length_m"] = t.n * step
    t["log_len"] = np.log(t.length_m)
    t["channel"] = chan[t.index.values]
    t["is_top"] = (t.channel == 1).astype(float)
    t["reason"] = reason_codes(t)
    return t


def reason_codes(t: pd.DataFrame) -> np.ndarray:
    r = np.full(len(t), "clear", dtype=object)
    r[t.length_m.values < 1.0] = "short"
    r[t.lum_mean.values > 170] = "bright_marking"
    r[t.frac_shadow.values > 0.5] = "shadow"
    r[t.frac_elev.values > 0.3] = "elevated_object"
    r[t.frac_veg.values > 0.3] = "vegetation"
    return r


def review_curve(score: np.ndarray, length: np.ndarray, frac_correct: np.ndarray):
    """score: higher = more confident. Returns (x reviewed length share, y share of wrong length caught)."""
    o = np.argsort(score, kind="stable")
    L = length[o]; E = length[o] * (1 - frac_correct[o])
    x = np.concatenate([[0], np.cumsum(L) / L.sum()])
    y = np.concatenate([[0], np.cumsum(E) / max(E.sum(), 1e-9)])
    return x, y


def caught_at(x: np.ndarray, y: np.ndarray, share: float) -> float:
    return float(np.interp(share, x, y))


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))
