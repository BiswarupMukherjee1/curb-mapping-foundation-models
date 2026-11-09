"""Train and evaluate a curb-line model: frozen encoder + light head, or a fine-tuned U-Net, with a label budget.

Training data: 512 x 512 px (25.6 m) label windows on the TRAIN blocks; a budget of B metres of labelled curb is met
by drawing windows at random (80 % containing curbs). Every curb inside a chosen window is labelled, as an annotator
would label small areas.
Model: frozen DINOv3 (or none) + EdgeHead (src/pcrs_ai/models.py); masked BCE + Dice; flips / 90° rotations.
Selection: probability threshold chosen on VAL blocks by line quality at 10 cm (after vectorisation).
Report: TEST blocks, protocol.evaluate (all / visible GT at 5 / 10 / 20 cm, completeness per visibility reason).
Usage: PYTHONPATH=src python experiments/03_learned_models/train.py --encoder dinov3_vitl16_sat --budget all
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from affine import Affine  # noqa: E402

from pcrs_ai import protocol  # noqa: E402
from pcrs_ai.gpu import wait_for_memory  # noqa: E402
from pcrs_ai.metrics import line_metrics  # noqa: E402
from pcrs_ai.models import anysat_modules, build, n_params, needs_nir  # noqa: E402
from pcrs_ai.paths import DATA_PROCESSED, MODELS  # noqa: E402
from pcrs_ai.tiledata import CHANNELS, SPLITS, WIN, curb_length_m, grid_windows, load_tile  # noqa: E402
from pcrs_ai.vectorize import prob_to_lines  # noqa: E402

THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
BLOCK_PX = 2000          # 100 m blocks, vectorised one at a time


def pick_windows(tds, budget, rng):
    cand = []
    for t, td in tds.items():
        for r0, c0 in grid_windows(td, "train"):
            cand.append((t, r0, c0, curb_length_m(td, r0, c0)))
    pos = [w for w in cand if w[3] > 5]
    neg = [w for w in cand if w[3] <= 5]
    rng.shuffle(pos); rng.shuffle(neg)
    if budget == "all":
        return pos + neg
    out, L, ip, ineg = [], 0.0, 0, 0
    while L < float(budget) and (ip < len(pos) or ineg < len(neg)):
        if rng.random() < 0.8 and ip < len(pos):
            out.append(pos[ip]); L += pos[ip][3]; ip += 1
        elif ineg < len(neg):
            out.append(neg[ineg]); ineg += 1
    return out


def batch_from(tds, wins, rng, bs, dev):
    xs, ys, ms = [], [], []
    for i in rng.integers(0, len(wins), bs):
        t, r0, c0, _ = wins[i]
        _, tgt, ign, spl = tds[t].window(r0, c0)
        rgb = tds[t].image(r0, c0)                                # RGB (+ NIR for encoders that use it)
        m = (spl == SPLITS["train"])[None] & (ign == 0)          # (2, H, W) per-channel loss mask
        k = int(rng.integers(4)); f = bool(rng.integers(2))
        arr = [np.rot90(a, k, axes=(-2, -1)) for a in (rgb, tgt, m)]
        if f:
            arr = [a[..., ::-1] for a in arr]
        xs.append(np.ascontiguousarray(arr[0])); ys.append(np.ascontiguousarray(arr[1])); ms.append(np.ascontiguousarray(arr[2]))
    x = torch.from_numpy(np.stack(xs)).to(dev).float() / 255.0
    y = torch.from_numpy(np.stack(ys)).to(dev).float()          # (B, 2, H, W)
    m = torch.from_numpy(np.stack(ms)).to(dev).float()
    return x, y, m


def loss_fn(logit, y, m, pos_weight=4.0):
    bce = F.binary_cross_entropy_with_logits(logit, y, pos_weight=torch.tensor(pos_weight, device=y.device), reduction="none")
    bce = (bce * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logit) * m
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * m).sum() + 1)
    return bce + dice


@torch.no_grad()
def predict_tile(td, enc, head, dev, codes, bs=8, stride=448):
    """uint8 probability maps (n_channels, H, W) (0-255) over windows touching the given split codes."""
    H, W = td.shape
    acc = np.zeros((len(CHANNELS), H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    corners = []
    for r0 in list(range(0, H - WIN, stride)) + [H - WIN]:
        for c0 in list(range(0, W - WIN, stride)) + [W - WIN]:
            if np.isin(td.split[r0:r0 + WIN:16, c0:c0 + WIN:16], codes).any():
                corners.append((r0, c0))
    wy = np.minimum(np.arange(WIN) + 1, np.arange(WIN)[::-1] + 1).astype(np.float32)
    wy = np.minimum(wy / 32.0, 1.0); w2 = np.outer(wy, wy)                     # feather window borders
    for i in range(0, len(corners), bs):
        cs = corners[i:i + bs]
        x = torch.from_numpy(np.stack([td.image(r, c) for r, c in cs])).to(dev).float() / 255.0
        with torch.autocast("cuda", dtype=torch.float16):
            feats = enc(x) if enc is not None else None
            p = torch.sigmoid(head(x, feats)).float().cpu().numpy()
        for (r, c), pi in zip(cs, p):
            acc[:, r:r + WIN, c:c + WIN] += pi * w2; cnt[r:r + WIN, c:c + WIN] += w2
    out = np.zeros(acc.shape, np.uint8)
    for k in range(acc.shape[0]):
        out[k] = (np.where(cnt > 0, acc[k] / np.maximum(cnt, 1e-6), 0) * 255).astype(np.uint8)
    return out, len(corners)


def lines_for_split(td, prob_u8, code, thr):
    """Vectorise each channel inside the split region. Returns (lines, conf, channel index)."""
    H, W = td.shape
    lines, conf, chan = [], [], []
    for r0 in range(0, H, BLOCK_PX):
        for c0 in range(0, W, BLOCK_PX):
            sp = td.split[r0:r0 + BLOCK_PX, c0:c0 + BLOCK_PX] == code
            if not sp.any():
                continue
            tr = td.transform @ Affine.translation(c0, r0)
            for k in range(prob_u8.shape[0]):
                pr = prob_u8[k, r0:r0 + BLOCK_PX, c0:c0 + BLOCK_PX].astype(np.float32) / 255.0
                ls, cf = prob_to_lines(pr, tr, threshold=thr, valid=sp)
                lines += ls; conf += list(cf); chan += [k] * len(ls)
    return lines, np.asarray(conf), np.asarray(chan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="dinov3_vitl16_sat", help="dinov3_vitl16_sat | dinov3_vitb16_web | none")
    ap.add_argument("--budget", default="all", help="labelled curb metres: all | 100 | 1000 | ...")
    ap.add_argument("--train-tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--test-tiles", nargs="*", default=["OH", "RL"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upsample", type=int, default=1, help="resize factor applied to the encoder input only (2 -> 40 cm tokens)")
    ap.add_argument("--width", type=int, default=64, help="head width")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    tag = a.tag or f"{a.encoder}_b{a.budget}_tr{''.join(a.train_tiles)}_te{''.join(a.test_tiles)}"
    run = Path(__file__).parent / "runs" / f"{time.strftime('%Y%m%d-%H%M')}_{tag}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda"
    need = 12.0 if a.upsample > 1 else (9.0 if "+" in a.encoder else 6.5)
    waited = wait_for_memory(need)
    if waited > 0:
        print(f"[gpu] waited {waited / 60:.0f} min for {need} GB", flush=True)
    t0 = time.time()
    tiles = sorted(set(a.train_tiles) | set(a.test_tiles))
    tds = {t: load_tile(t, with_nir=needs_nir(a.encoder)) for t in tiles}
    print(f"tiles loaded ({time.time() - t0:.0f}s)", flush=True)

    enc, head = build(a.encoder, width=a.width, upsample=a.upsample)
    enc = enc.to(dev) if enc is not None else None
    head = head.to(dev)
    for m in anysat_modules(enc):                          # per-band standardisation from the training tiles
        st = [tds[t].band_stats() for t in a.train_tiles]
        mean, std = np.mean([s_[0] for s_ in st], 0), np.mean([s_[1] for s_ in st], 0)
        m.set_stats(mean, std)
        (run / "anysat_stats.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    print(f"head params {n_params(head) / 1e6:.2f} M", flush=True)
    wins = pick_windows({t: tds[t] for t in a.train_tiles}, a.budget, rng)
    budget_m = sum(w[3] for w in wins)
    print(f"label windows {len(wins)}, labelled curb ≈ {budget_m:.0f} m", flush=True)

    if hasattr(head, "encoder"):                         # pretrained backbone: 10x lower learning rate
        groups = [{"params": head.encoder.parameters(), "lr": a.lr / 10},
                  {"params": [p for n, p in head.named_parameters() if not n.startswith("encoder.")], "lr": a.lr}]
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        max_lr = [a.lr / 10, a.lr]
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
        max_lr = a.lr
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=a.iters, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    head.train()
    for it in range(a.iters):
        x, y, m = batch_from(tds, wins, rng, a.batch, dev)
        with torch.autocast("cuda", dtype=torch.float16):
            feats = enc(x) if enc is not None else None
            logit = head(x, feats)
        loss = loss_fn(logit.float(), y, m)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % 200 == 0 or it == a.iters - 1:
            print(f"it {it} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    head.eval()
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), MODELS / f"head_{run.name}.pt")

    # ---- predict val + test regions
    probs = {}
    for t, td in tds.items():
        codes = [SPLITS["val"]] + ([SPLITS["test"]] if t in a.test_tiles else [])
        probs[t], n = predict_tile(td, enc, head, dev, codes)
        print(f"predicted {t}: {n} windows ({time.time() - t0:.0f}s)", flush=True)

    # ---- threshold on VAL (pooled over all loaded tiles): line quality at 10 cm
    sel = []
    for thr in THRESHOLDS:
        tp = gl = fp = 0.0
        for t, td in tds.items():
            gt = protocol.ground_truth(t, "val")
            ls, _, _ = lines_for_split(td, probs[t], SPLITS["val"], thr)
            r = line_metrics(gt.values, ls, tolerances=(0.10,))
            tp += r["completeness_10cm"] * r["gt_length_m"]; gl += r["gt_length_m"]
            fp += (1 - r["correctness_10cm"]) * r["pred_length_m"] if r["pred_length_m"] else 0.0
        q = tp / (gl + fp) if gl + fp else 0.0
        sel.append({"threshold": thr, "val_quality_10cm": q})
        print(f"val thr {thr}: quality@10cm {q:.4f}", flush=True)
    best = max(sel, key=lambda d: d["val_quality_10cm"])["threshold"]

    # ---- TEST
    summary = {"encoder": a.encoder, "budget": a.budget, "labelled_curb_m": budget_m, "n_label_windows": len(wins),
               "train_tiles": a.train_tiles, "head_params": n_params(head), "val_selection": sel, "threshold": best,
               "time_s": round(time.time() - t0)}
    for t in a.test_tiles:
        ls, cf, ch = lines_for_split(tds[t], probs[t], SPLITS["test"], best)
        res = protocol.evaluate(t, protocol.ground_truth(t, "test"), ls)
        cb = protocol.curbs(t); reg = protocol.region(t, "test")
        for k, (cname, subs) in enumerate(CHANNELS.items()):          # per-class: bottom pred vs bottom GT, top vs top
            gtk = cb[cb.ssclasse.isin(subs)].geometry.clip(reg)
            res[f"class_{cname}"] = line_metrics(gtk.values, [l for l, c in zip(ls, ch) if c == k])
        summary[t] = res
        print(t, json.dumps({k: round(v, 3) for k, v in res["all"].items() if k.startswith(("completeness", "correctness", "quality"))}), flush=True)
        print(t, "visible", json.dumps({k: round(v, 3) for k, v in res["visible"].items() if k.startswith(("completeness", "correctness"))}), flush=True)
        print(t, "per class @10cm", {c: (round(res[f"class_{c}"]["completeness_10cm"], 3), round(res[f"class_{c}"]["correctness_10cm"], 3)) for c in CHANNELS}, flush=True)
        gpd.GeoDataFrame({"conf": cf, "channel": [list(CHANNELS)[k] for k in ch]}, geometry=ls, crs="EPSG:3945").to_file(DATA_PROCESSED / f"test_lines_as_trained_{t}_{run.name}.gpkg", driver="GPKG")
    (run / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("->", run, flush=True)


if __name__ == "__main__":
    main()
