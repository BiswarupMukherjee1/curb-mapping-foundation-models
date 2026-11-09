"""Bordeaux Métropole open data access (Licence Ouverte).

Two services:
- Opendatasoft "datahub" API (vector layers, incl. the PCRS) — records / geojson exports filtered by a WKT polygon.
- The public tile downloader's WPS service (orthophotos, LiDAR) — the same calls the official "Graphic Downloader"
  web page makes, with the public key embedded in that page.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
from pathlib import Path

import requests

DATAHUB = "https://datahub.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets"
WPS = "https://data.bordeaux-metropole.fr/wps"
WPS_PUBLIC_KEY = "258BILMNYZ"          # public key of https://data.bordeaux-metropole.fr/opendata/graphic_downloader/
PCRS_CDN = "https://cdn.scnbdx.fr/web-geo/QILTMET01/PCRS"
NATIVE_CRS = "EPSG:3945"               # RGF93 / CC45, used by the orthophotos, LiDAR and the PCRS files
UA = {"User-Agent": "pcrs-ai research (Univ. Bordeaux student project)"}

# Raster/LiDAR layers in the downloader: (WPS layer name, format)
LAYERS = {
    "ortho2023": ("ORTHO2023", "JPEG"),
    "orthoir2023": ("ORTHO2023_IR", "JPEG"),
    "lidar2020": ("NUAGE_POINTS_LASER_3D", "LAZ"),
    "mns2023": ("MNS_LIDAR", "TIFF"),
    "mnt2023": ("MNT2023", "TIFF"),
}


def _get(url: str, params: dict | None = None, timeout: int = 120, retries: int = 3) -> requests.Response:
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------- datahub (vectors)
def records(dataset: str, **params) -> dict:
    return _get(f"{DATAHUB}/{dataset}/records", params=params).json()


def count(dataset: str, where: str | None = None) -> int:
    p = {"limit": 0}
    if where:
        p["where"] = where
    return records(dataset, **p)["total_count"]


def export_geojson(dataset: str, where: str | None = None) -> dict:
    """Full export (no 100-record paging limit). Geometry in WGS84 lon/lat."""
    p = {"where": where} if where else {}
    return _get(f"{DATAHUB}/{dataset}/exports/geojson", params=p, timeout=600).json()


def wkt_polygon(ring_lonlat: list) -> str:
    return "POLYGON((" + ",".join(f"{x} {y}" for x, y in ring_lonlat) + "))"


def intersects(wkt: str) -> str:
    return f"intersects(geo_shape, geom'{wkt}')"


def tile_record(name: str | None = None, lon: float | None = None, lat: float | None = None) -> dict:
    """Tile of the 1.4 km x 1 km orthophoto/LiDAR grid (dataset to_dalle_s), by name or by a point."""
    where = f"nom='{name}'" if name else intersects(f"POINT({lon} {lat})")
    res = records("to_dalle_s", where=where, limit=1)["results"]
    if not res:
        raise KeyError(f"no tile for {name or (lon, lat)}")
    return res[0]


# ---------------------------------------------------------------- WPS (raster / LiDAR tiles)
def _wps(identifier: str, inputs: dict) -> str:
    parts = []
    for k, v in inputs.items():
        vals = v if isinstance(v, (list, tuple)) else [v]
        parts += [f"{k}={x}" for x in vals]
    params = {"key": WPS_PUBLIC_KEY, "service": "wps", "version": "1.0.0", "request": "execute",
              "identifier": identifier, "datainputs": ";".join(parts)}
    url = WPS + "?" + urllib.parse.urlencode(params, safe="=;")
    return _get(url).content.decode("latin-1")


def _literals(xml: str) -> list[str]:
    return re.findall(r"<wps:LiteralData[^>]*>([^<]*)<", xml)


def available_tiles(layer: str) -> set[str]:
    couche, fmt = LAYERS[layer]
    return set(_literals(_wps("gd_dalles_dispo", {"couche": couche, "format": fmt})))


def download_size_mb(layer: str, tiles: list[str]) -> float:
    couche, fmt = LAYERS[layer]
    lit = _literals(_wps("gd_taille_fichiers", {"couche": couche, "format": fmt, "dalle": tiles}))
    return float(lit[0]) if lit else float("nan")


def download_link(layer: str, tiles: list[str]) -> str:
    couche, fmt = LAYERS[layer]
    xml = _wps("gd_download", {"couche": couche, "format": fmt, "dalle": tiles})
    m = re.search(r'<wps:Reference[^>]*href="([^"]+)"', xml)
    if not m:
        raise RuntimeError(f"no download link for {layer} {tiles}: {xml[-400:]}")
    return m.group(1).replace("&amp;", "&")


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=UA, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for b in r.iter_content(chunk):
                f.write(b)
    tmp.rename(dest)
    return dest


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
