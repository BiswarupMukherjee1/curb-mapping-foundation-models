"""Download the phase-1 data for the tiles in configs/aoi.yaml into data/raw/ (idempotent, writes a manifest).

data/raw/
  tiles/<T>.geojson                    tile footprint (WGS84)
  ortho2023/<T>/...  orthoir2023/<T>/...  lidar2020/<T>/...   extracted downloader archives
  pcrs/<T>/<layer>.geojson             PCRS layers clipped to the tile footprint (WGS84, from the datahub)
  pcrs/doc/*.xlsx                      Bordeaux PCRS structure + correspondence to the CNIG standard
  manifest.json
Usage: python scripts/download_data.py [--tiles OH RL] [--skip-rasters]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcrs_ai import datahub as dh  # noqa: E402
from pcrs_ai.paths import CONFIGS, DATA_RAW  # noqa: E402

DOC_FILES = ["PCRS_BM_StructurationFichiersEtObjets.xlsx",
             "PCRS_BM_TableCorrespondanceBordeauxMetropoleVersStandard.xlsx"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch_raster(layer: str, tile: str, manifest: dict):
    out = DATA_RAW / layer / tile
    if out.exists() and any(out.iterdir()):
        print(f"[skip] {layer}/{tile} exists")
        return
    mb = dh.download_size_mb(layer, [tile])
    url = dh.download_link(layer, [tile])
    print(f"[get ] {layer}/{tile} ~{mb} MB")
    arc = DATA_RAW / layer / f"{tile}.download"
    dh.download_file(url, arc)
    out.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(arc):
        with zipfile.ZipFile(arc) as z:
            z.extractall(out)
        arc.unlink()
    else:                                   # single file: keep it, name unknown -> keep extension-less
        arc.rename(out / f"{tile}.bin")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest["rasters"][f"{layer}/{tile}"] = {
        "layer": dh.LAYERS[layer], "size_mb_reported": mb,
        "files": [{"path": str(p.relative_to(DATA_RAW)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files],
    }


def fetch_vectors(tile: str, ring: list, layers: list[str], manifest: dict):
    wkt = dh.wkt_polygon(ring)
    for layer in layers:
        out = DATA_RAW / "pcrs" / tile / f"{layer}.geojson"
        if out.exists():
            print(f"[skip] pcrs/{tile}/{layer}")
            continue
        gj = dh.export_geojson(layer, where=dh.intersects(wkt))
        dh.save_json(gj, out)
        n = len(gj.get("features", []))
        print(f"[get ] pcrs/{tile}/{layer}: {n} features")
        manifest["vectors"][f"{tile}/{layer}"] = {"n": n, "bytes": out.stat().st_size, "sha256": sha256(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--skip-rasters", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((CONFIGS / "aoi.yaml").read_text())
    tiles = args.tiles or cfg["download"]["tiles"]
    mpath = DATA_RAW / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"rasters": {}, "vectors": {}, "tiles": {}}
    manifest["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["licence"] = "Licence Ouverte / Open Licence (Etalab) — Bordeaux Métropole"

    for f in DOC_FILES:
        p = DATA_RAW / "pcrs" / "doc" / f
        if not p.exists():
            dh.download_file(f"{dh.PCRS_CDN}/{f}", p)
            print(f"[get ] doc {f}")

    for t in tiles:
        rec = dh.tile_record(name=t)
        ring = rec["geo_shape"]["geometry"]["coordinates"][0]
        dh.save_json({"type": "Feature", "properties": {"nom": t}, "geometry": rec["geo_shape"]["geometry"]},
                     DATA_RAW / "tiles" / f"{t}.geojson")
        manifest["tiles"][t] = {"centroid": rec["geo_point_2d"]}
        fetch_vectors(t, ring, cfg["pcrs_layers"], manifest)
        if not args.skip_rasters:
            for layer in cfg["download"]["rasters"]:
                fetch_raster(layer, t, manifest)
            if t in cfg["download"].get("lidar_tiles", []):
                fetch_raster("lidar2020", t, manifest)
        dh.save_json(manifest, mpath)
    print("done ->", mpath)


if __name__ == "__main__":
    main()
