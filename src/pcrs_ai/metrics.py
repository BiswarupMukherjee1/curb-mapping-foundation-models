"""Vector-level evaluation of extracted lines against surveyed PCRS lines, at cm tolerances (buffer method).

completeness(τ) = share of ground-truth length lying within τ of a prediction       ("how much did we draw")
correctness(τ)  = share of predicted length lying within τ of the ground truth        ("how much of it is right")
quality(τ)      = TP / (TP + FP + FN) on length (Wiedemann-style)
Lengths are approximated by densifying every line at `step` metres; the point-to-line distance is approximated by the
distance to the other set's densified points (error <= step / 2).
An optional direction check rejects matches whose local directions differ by more than `max_angle_deg`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .align import densify

TOLERANCES_M = (0.05, 0.10, 0.20)


def _match(src_xy, src_t, dst_xy, dst_t, tol, max_angle_deg):
    if len(src_xy) == 0:
        return np.zeros(0, bool), np.zeros(0)
    if len(dst_xy) == 0:
        return np.zeros(len(src_xy), bool), np.full(len(src_xy), np.inf)
    tree = cKDTree(dst_xy)
    d, j = tree.query(src_xy, k=8, distance_upper_bound=tol)
    ok = np.isfinite(d)
    if max_angle_deg is not None:
        jj = np.where(ok, j, 0)
        cos = np.abs(np.einsum("nd,nkd->nk", src_t, dst_t[jj]))     # undirected lines
        ok &= cos >= np.cos(np.deg2rad(max_angle_deg))
    d = np.where(ok, d, np.inf)
    best = d.min(axis=1)
    return np.isfinite(best), best


def line_metrics(gt_geoms, pred_geoms, tolerances=TOLERANCES_M, step=0.02, max_angle_deg=30.0,
                 gt_weight: np.ndarray | None = None, gt_dense=None) -> dict:
    """gt_weight: optional per-GT-point weight (e.g. 1 = visible, 0 = occluded) evaluated on the step-densified GT.
    Pass a callable(xy)->weights instead to compute it on the densified points.
    gt_dense: optional precomputed densify(gt_geoms, step) to avoid recomputation (e.g. in a tuning loop)."""
    gxy, gt_ = gt_dense if gt_dense is not None else densify(gt_geoms, step)
    pxy, pt_ = densify(pred_geoms, step)
    if callable(gt_weight):
        w = np.asarray(gt_weight(gxy), float)
    elif gt_weight is None:
        w = np.ones(len(gxy))
    else:
        w = np.asarray(gt_weight, float)
    out = {"gt_length_m": float(len(gxy) * step), "gt_length_weighted_m": float(w.sum() * step),
           "pred_length_m": float(len(pxy) * step)}
    for tol in tolerances:
        g_ok, g_d = _match(gxy, gt_, pxy, pt_, tol, max_angle_deg)
        p_ok, p_d = _match(pxy, pt_, gxy, gt_, tol, max_angle_deg)
        tp_g = float((w * g_ok).sum()); fn = float((w * ~g_ok).sum())
        tp_p = float(p_ok.sum()); fp = float((~p_ok).sum())
        k = f"{int(round(tol * 100))}cm"
        out[f"completeness_{k}"] = tp_g / (tp_g + fn) if tp_g + fn > 0 else float("nan")
        out[f"correctness_{k}"] = tp_p / (tp_p + fp) if tp_p + fp > 0 else float("nan")
        out[f"quality_{k}"] = tp_g / (tp_g + fn + fp) if tp_g + fn + fp > 0 else float("nan")
        out[f"rmse_matched_{k}_cm"] = float(np.sqrt(np.mean(g_d[g_ok] ** 2)) * 100) if g_ok.any() else float("nan")
    return out
