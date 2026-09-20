"""Shared test helpers (tiny synthetic rasters, never real scenes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

ORIGIN = (500000.0, 3420000.0)  # EPSG:32643, ~30.9N 75E
CRS_UTM = "EPSG:32643"


def write_multiband(
    path: Path,
    data: np.ndarray,
    crs: str = CRS_UTM,
    pixel: float = 30.0,
    nodata: float | None = None,
    band_tags: list[dict[str, str]] | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    b, h, w = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype=data.dtype,
        count=b,
        height=h,
        width=w,
        crs=crs,
        transform=from_origin(*ORIGIN, pixel, pixel),
        nodata=nodata,
    ) as dst:
        dst.write(data)
        for i, t in enumerate(band_tags or [], start=1):
            dst.update_tags(i, **t)
        if tags:
            dst.update_tags(**tags)
    return path
