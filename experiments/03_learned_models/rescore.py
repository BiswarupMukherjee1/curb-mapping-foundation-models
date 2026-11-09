"""Re-score trained models with one common threshold grid (selection on VAL, report on TEST).

Training runs used two threshold grids over time; this puts every model on the same footing. For each run: reload the
head, predict VAL and TEST blocks, pick the threshold maximising line quality at 10 cm on VAL from COMMON_GRID,
evaluate TEST with the shared protocol, and write `summary_rescored.json` into the run folder plus the test lines.
Usage: PYTHONPATH=src python experiments/03_learned_models/rescore.py <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pcrs_ai import paths  # noqa: E402,F401

import geopandas as gpd  # noqa: E402
import torch  # noqa: E402

from pcrs_ai import protocol  # noqa: E402
from pcrs_ai.metrics import line_metrics  # noqa: E402
from pcrs_ai.gpu import wait_for_memory  # noqa: E402
from pcrs_ai.models import load_trained, needs_nir  # noqa: E402
from pcrs_ai.paths import DATA_PROCESSED  # noqa: E402
from pcrs_ai.tiledata import CHANNELS, SPLITS, load_tile  # noqa: E402

spec = importlib.util.spec_from_file_location("train", Path(__file__).parent / "run.py")
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)
COMMON_GRID = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
_TILES = {}


def tile(t, nir=False):
    key = (t, nir)
    if key not in _TILES:
        _TILES[key] = load_tile(t, verbose=False, with_nir=nir)
    return _TILES[key]


def rescore(run: Path, dev="cuda"):
    wait_for_memory(6.5)
    cfg, enc, head = load_trained(run, dev)
    nir = needs_nir(cfg["encoder"])
    tiles = sorted(set(cfg["train_tiles"]) | set(cfg["test_tiles"]))
    probs = {}
    for t in tiles:
        codes = [SPLITS["val"]] + ([SPLITS["test"]] if t in cfg["test_tiles"] else [])
        probs[t], _ = train.predict_tile(tile(t, nir), enc, head, dev, codes)
    sel = []
    for thr in COMMON_GRID:
        tp = gl = fp = 0.0
        for t in tiles:
            ls, _, _ = train.lines_for_split(tile(t, nir), probs[t], SPLITS["val"], thr)
            r = line_metrics(protocol.ground_truth(t, "val").values, ls, tolerances=(0.10,))
            tp += r["completeness_10cm"] * r["gt_length_m"]; gl += r["gt_length_m"]
            fp += (1 - r["correctness_10cm"]) * r["pred_length_m"] if r["pred_length_m"] else 0.0
        sel.append({"threshold": thr, "val_quality_10cm": tp / (gl + fp) if gl + fp else 0.0})
    best = max(sel, key=lambda d: d["val_quality_10cm"])["threshold"]
    out = {"run": run.name, "encoder": cfg["encoder"], "budget": cfg["budget"], "train_tiles": cfg["train_tiles"],
           "seed": cfg.get("seed", 0), "threshold": best, "val_selection": sel}
    tag = run.name.split("_", 1)[1]
    for t in cfg["test_tiles"]:
        ls, cf, ch = train.lines_for_split(tile(t, nir), probs[t], SPLITS["test"], best)
        res = protocol.evaluate(t, protocol.ground_truth(t, "test"), ls)
        cb = protocol.curbs(t); reg = protocol.region(t, "test")
        for k, (cname, subs) in enumerate(CHANNELS.items()):
            res[f"class_{cname}"] = line_metrics(cb[cb.ssclasse.isin(subs)].geometry.clip(reg).values,
                                                 [l for l, c in zip(ls, ch) if c == k])
        out[t] = res
        gpd.GeoDataFrame({"conf": cf, "channel": [list(CHANNELS)[k] for k in ch]}, geometry=ls, crs="EPSG:3945").to_file(
            DATA_PROCESSED / f"test_lines_{t}_{tag}.gpkg", driver="GPKG")
    (run / "summary_rescored.json").write_text(json.dumps(out, indent=1, default=float))
    print(run.name, "thr", best, {t: round(out[t]["all"]["quality_10cm"], 3) for t in cfg["test_tiles"]}, flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        rescore(Path(r))
