"""Explainability of the trained curb models: what do they rely on?

A. Branch test (frozen-encoder + head models only): on TEST squares, predict with
     - both branches (normal),
     - the foundation-model features replaced by zeros (pixel stem only),
     - the head's pixel input replaced by a flat grey image (foundation-model features only),
   using the trained head and threshold, and score at 10 cm. The head is not retrained, so this measures how much
   the trained model depends on each branch, not what each branch could learn alone.
B. RISE attribution (Petsiuk et al., 2018) for drawn test lines that are correct or wrong (same class, 10 cm):
   score = mean predicted probability of the line's class along the line; 500 random 8x8 masks upsampled to the
   512 px window, masked pixels replaced by the window's mean colour. For each line: share of positive attribution
   within 0.5 m of the line ("on the curb") vs 0.5-3 m ("surroundings"), normalised by area (focus ratio > 1 means
   attribution concentrates on the curb).
Usage: PYTHONPATH=src python experiments/06_explainability/run.py --runs <run_dir> [<run_dir> ...] --tiles OH RL
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
import torch.nn.functional as F  # noqa: E402
from affine import Affine  # noqa: E402
from rasterio.features import rasterize  # noqa: E402


from pcrs_ai import protocol  # noqa: E402
from pcrs_ai.align import densify  # noqa: E402
from pcrs_ai.gpu import wait_for_memory  # noqa: E402
from pcrs_ai.metrics import _match  # noqa: E402
from pcrs_ai.models import load_trained, needs_nir  # noqa: E402
from pcrs_ai.tiledata import CHANNELS, SPLITS, WIN, load_tile  # noqa: E402

spec = importlib.util.spec_from_file_location("train", ROOT / "experiments/03_learned_models/train.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
N_MASKS, GRID, P_KEEP, BATCH = 500, 8, 0.5, 8


class Ablated(torch.nn.Module):
    """Wrap a head so predict_tile runs it with one branch switched off."""

    def __init__(self, head, mode):
        super().__init__()
        self.head, self.mode = head, mode

    def forward(self, x, feats):
        if self.mode == "pixel_only":
            feats = torch.zeros_like(feats)
        elif self.mode == "features_only":
            x = torch.full_like(x, 0.5)
        return self.head(x, feats)


def branch_test(run, cfg, enc, head, td, thr, dev):
    out = {}
    for mode in ("both", "pixel_only", "features_only"):
        h = head if mode == "both" else Ablated(head, mode)
        prob, _ = train.predict_tile(td, enc, h, dev, [SPLITS["test"]])
        ls, _, _ = train.lines_for_split(td, prob, SPLITS["test"], thr)
        m = protocol.evaluate(td.tile, protocol.ground_truth(td.tile, "test"), ls)["all"]
        out[mode] = {k: m[k] for k in ("completeness_10cm", "correctness_10cm", "quality_10cm")}
        print(f"  branch {mode:14s}", {k: round(v, 3) for k, v in out[mode].items()}, flush=True)
    return out


def pick_lines(td, enc, head, thr, dev, n_each=15, seed=0):
    """Drawn test lines (>= 2 m), labelled correct (>= 80 % within 10 cm of a same-class line) or wrong (< 20 %)."""
    prob, _ = train.predict_tile(td, enc, head, dev, [SPLITS["test"]])
    ls, _, ch = train.lines_for_split(td, prob, SPLITS["test"], thr)
    cb = protocol.curbs(td.tile); reg = protocol.region(td.tile, "test")
    gt = {k: densify(cb[cb.ssclasse.isin(s)].geometry.clip(reg).values, 0.02) for k, s in enumerate(CHANNELS.values())}
    rows = []
    for i, (l, k) in enumerate(zip(ls, ch)):
        if l.length < 2.0:
            continue
        xy, tg = densify([l], 0.05)
        ok, _ = _match(xy, tg, gt[k][0], gt[k][1], 0.10, 30.0)
        rows.append({"i": i, "channel": int(k), "frac": float(ok.mean()), "line": l})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    good = df[df.frac >= 0.8]; bad = df[df.frac < 0.2]
    pick = pd.concat([good.sample(min(n_each, len(good)), random_state=seed).assign(label="correct"),
                      bad.sample(min(n_each, len(bad)), random_state=seed).assign(label="wrong")])
    del prob
    return pick, rng


@torch.no_grad()
def rise(td, enc, head, line, channel, dev, rng):
    inv = ~td.transform
    cx, cy = line.interpolate(0.5, normalized=True).coords[0][:2]
    c0 = int(np.clip(inv.a * cx + inv.c - WIN // 2, 0, td.shape[1] - WIN))
    r0 = int(np.clip(inv.e * cy + inv.f - WIN // 2, 0, td.shape[0] - WIN))
    img = torch.from_numpy(td.image(r0, c0)).float().to(dev)[None] / 255.0

    tr = td.transform @ Affine.translation(c0, r0)
    on = rasterize([(line.buffer(0.05), 1)], out_shape=(WIN, WIN), transform=tr, fill=0, dtype="uint8").astype(bool)
    near = rasterize([(line.buffer(0.5), 1)], out_shape=(WIN, WIN), transform=tr, fill=0, dtype="uint8").astype(bool)
    ring = rasterize([(line.buffer(3.0), 1)], out_shape=(WIN, WIN), transform=tr, fill=0, dtype="uint8").astype(bool) & ~near
    if on.sum() < 10:
        return None
    on_t = torch.from_numpy(on).to(dev)
    fill = img.mean(dim=(2, 3), keepdim=True)
    cell = int(np.ceil(WIN / GRID))
    sal = torch.zeros(WIN, WIN, device=dev); scores = []
    for b in range(0, N_MASKS, BATCH):
        n = min(BATCH, N_MASKS - b)
        grid = (torch.from_numpy(rng.random((n, 1, GRID, GRID))).to(dev) < P_KEEP).float()
        up = F.interpolate(grid, size=((GRID + 1) * cell, (GRID + 1) * cell), mode="bilinear", align_corners=False)
        dx, dy = rng.integers(0, cell, 2)
        masks = up[:, :, dy:dy + WIN, dx:dx + WIN]
        x = img * masks + fill * (1 - masks)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = enc(x) if enc is not None else None
            p = torch.sigmoid(head(x, feats)).float()[:, channel]
        s = (p * on_t).sum(dim=(1, 2)) / on_t.sum()
        sal += (s[:, None, None] * masks[:, 0]).sum(0); scores.append(s)
    sal = (sal / (N_MASKS * P_KEEP)).cpu().numpy()
    base = float(torch.cat(scores).mean())
    pos = np.clip(sal - np.median(sal), 0, None)
    tot = pos.sum() + 1e-9
    share_near, share_ring = pos[near].sum() / tot, pos[ring].sum() / tot
    focus = (share_near / near.mean()) / max(share_ring / max(ring.mean(), 1e-9), 1e-9)
    return {"share_within_0.5m": float(share_near), "share_0.5_3m": float(share_ring), "focus_ratio": float(focus),
            "mean_score": base, "_sal": sal, "_img": img[0, :3].permute(1, 2, 0).cpu().numpy(), "_on": on}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--n-each", type=int, default=15)
    ap.add_argument("--min-free", type=float, default=8.0, help="GB of free GPU memory to wait for")
    a = ap.parse_args()
    out_dir = Path(__file__).parent / "runs" / time.strftime("%Y%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=False)
    dev = "cuda"
    wait_for_memory(a.min_free)
    summary = {}
    for rd in map(Path, a.runs):
        cfg, enc, head = load_trained(rd, dev)
        s02 = json.loads((rd / ("summary_rescored.json" if (rd / "summary_rescored.json").exists() else "summary.json")).read_text())
        thr = s02["threshold"]; tag = rd.name.split("_", 1)[1]
        print(f"== {tag} (threshold {thr})", flush=True)
        res = {"threshold": thr}
        rows, examples = [], []
        for t in a.tiles:
            td = load_tile(t, verbose=False, with_nir=needs_nir(cfg["encoder"]))
            if enc is not None:
                res[f"branch_{t}"] = branch_test(rd, cfg, enc, head, td, thr, dev)
            pick, rng = pick_lines(td, enc, head, thr, dev, a.n_each)
            for _, r in pick.iterrows():
                o = rise(td, enc, head, r.line, r.channel, dev, rng)
                if o is None:
                    continue
                rows.append({"tile": t, "label": r.label, **{k: v for k, v in o.items() if not k.startswith("_")}})
                if len([e for e in examples if e[0] == r.label]) < 3:
                    examples.append((r.label, t, o))
            del td
        D = pd.DataFrame(rows)
        D.to_csv(out_dir / f"rise_{tag}.csv", index=False)
        res["rise"] = D.groupby("label")[["share_within_0.5m", "share_0.5_3m", "focus_ratio"]].median().round(3).to_dict("index")
        res["rise_n"] = D.label.value_counts().to_dict()
        print("  RISE medians:", res["rise"], flush=True)
        fig, axes = plt.subplots(2, len(examples), figsize=(2.6 * len(examples), 5.4), squeeze=False)
        for j, (lab, t, o) in enumerate(examples):
            axes[0, j].imshow(np.clip(o["_img"], 0, 1)); axes[0, j].contour(o["_on"], levels=[0.5], colors="yellow", linewidths=0.6)
            axes[0, j].set_title(f"{lab} ({t})", fontsize=8)
            axes[1, j].imshow(np.clip(o["_img"].mean(2), 0, 1), cmap="gray"); axes[1, j].imshow(o["_sal"], cmap="jet", alpha=0.5)
            axes[1, j].contour(o["_on"], levels=[0.5], colors="white", linewidths=0.6)
            for ax in axes[:, j]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"RISE attribution for drawn curb lines (25.6 m windows), {tag}", fontsize=9)
        fig.tight_layout(); fig.savefig(out_dir / f"rise_examples_{tag}.png", dpi=110); plt.close(fig)
        summary[tag] = res
        del enc, head; torch.cuda.empty_cache()
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("->", out_dir, flush=True)


if __name__ == "__main__":
    main()
