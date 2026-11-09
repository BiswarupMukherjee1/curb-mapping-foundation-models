"""Load downloaded tiles (orthophoto JPEG + world file) and PCRS vectors in the native CRS (EPSG:3945)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from .datahub import NATIVE_CRS
from .paths import DATA_RAW


def read_world_file(path: Path) -> Affine:
    """World file (.wld) -> affine of the pixel *corner* (world files give the centre of the upper-left pixel)."""
    a, d, b, e, c, f = [float(x) for x in path.read_text().split()]
    return Affine(a, b, c, d, e, f) @ Affine.translation(-0.5, -0.5)


@dataclass
class Ortho:
    path: Path
    transform: Affine          # pixel-corner affine, EPSG:3945
    width: int
    height: int

    @property
    def gsd(self) -> float:
        return self.transform.a

    def window(self, col0: int, row0: int, w: int, h: int, bands=None) -> np.ndarray:
        with rasterio.open(self.path) as ds:
            arr = ds.read(bands, window=Window(col0, row0, w, h))
        return arr

    def read_all(self, bands=None) -> np.ndarray:
        with rasterio.open(self.path) as ds:
            return ds.read(bands)

    def world_to_pixel(self, x, y):
        """World coords -> fractional (col, row) in pixel units where integer = pixel corner."""
        i = ~self.transform
        x, y = np.asarray(x, float), np.asarray(y, float)
        return i.a * x + i.b * y + i.c, i.d * x + i.e * y + i.f


def load_ortho(layer: str, tile: str, snap_to_tile: bool = True) -> Ortho:
    """snap_to_tile: the 2023 RGB world files put the grid corner half a pixel (2.5 cm) off the tile corner, while the
    infrared ortho, DSM, DTM and the tile footprint all start exactly on it, and RGB and IR are co-registered pixel for
    pixel (measured residual 2.5 ± 0.06 cm, exactly the world-file difference). So the corner is snapped to the tile footprint."""
    d = DATA_RAW / layer / tile
    jpg = next(d.glob("*.jpg"))
    tr = read_world_file(jpg.with_suffix(".wld"))
    with rasterio.open(jpg) as ds:
        w, h = ds.width, ds.height
    if snap_to_tile:
        minx, _, _, maxy = load_tile_polygon(tile).bounds
        if abs(tr.c - minx) < tr.a and abs(tr.f - maxy) < tr.a:
            tr = Affine(tr.a, tr.b, round(minx, 3), tr.d, tr.e, round(maxy, 3))
    return Ortho(jpg, tr, w, h)


def load_dem(layer: str, tile: str):
    """50 cm DSM ('mns2023') or DTM ('mnt2023') as (array float32 with NaN nodata, affine)."""
    tif = next((DATA_RAW / layer / tile).glob("*.tif"))
    with rasterio.open(tif) as ds:
        a = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        a[a < -1000] = np.nan
        return a, ds.transform


def load_pcrs(tile: str, layer: str = "cr_objet_infra_l") -> gpd.GeoDataFrame:
    gdf = gpd.read_file(DATA_RAW / "pcrs" / tile / f"{layer}.geojson")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(NATIVE_CRS)


def load_tile_polygon(tile: str):
    return gpd.read_file(DATA_RAW / "tiles" / f"{tile}.geojson").set_crs("EPSG:4326", allow_override=True).to_crs(NATIVE_CRS).geometry.iloc[0]
