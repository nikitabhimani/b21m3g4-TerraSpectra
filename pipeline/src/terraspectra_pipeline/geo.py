"""Georeferencing: UTM target grids, windowed reprojection, AOI clipping and pixel masks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio import features, windows
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_bounds,
    transform_geom,
)
from rasterio.windows import Window

from terraspectra_contracts.constants import NODATA

Geometry = dict[str, Any]
WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class Grid:
    """A raster grid: CRS, affine transform and size."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return windows.bounds(Window(0, 0, self.width, self.height), self.transform)

    def window_transform(self, window: Window) -> Affine:
        return windows.transform(window, self.transform)


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    """WGS84 / UTM zone CRS containing the given point."""
    zone = int((lon + 180.0) // 6.0) % 60 + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def is_utm(crs: CRS | None) -> bool:
    if crs is None or not crs.is_projected:
        return False
    epsg = crs.to_epsg() or 0
    return 32601 <= epsg <= 32660 or 32701 <= epsg <= 32760 or "UTM" in crs.to_wkt().upper()


def grid_center_lonlat(grid: Grid) -> tuple[float, float]:
    left, bottom, right, top = transform_bounds(grid.crs, WGS84, *grid.bounds)
    return (left + right) / 2.0, (bottom + top) / 2.0


def plan_target_grid(
    src: Grid, dst_crs: CRS | str | None = None, resolution: float | None = None
) -> Grid:
    """Target grid in ``dst_crs`` (default: UTM zone of the scene centre).

    Returns ``src`` unchanged when it is already in the target CRS at the requested resolution.
    """
    target = (
        CRS.from_user_input(dst_crs) if dst_crs else utm_crs_for_lonlat(*grid_center_lonlat(src))
    )
    same_res = resolution is None or np.isclose(abs(src.transform.a), resolution)
    if target == src.crs and same_res:
        return src
    kwargs: dict[str, Any] = {"resolution": resolution} if resolution else {}
    transform, width, height = calculate_default_transform(
        src.crs, target, src.width, src.height, *src.bounds, **kwargs
    )
    return Grid(target, transform, int(width), int(height))


def load_aoi(aoi: str | Path | Geometry) -> list[Geometry]:
    """Load geometries from a GeoJSON path/dict (FeatureCollection, Feature or geometry)."""
    data: Any = aoi
    if isinstance(aoi, (str, Path)):
        data = json.loads(Path(aoi).read_text())
    kind = data.get("type")
    if kind == "FeatureCollection":
        geoms = [f["geometry"] for f in data["features"] if f.get("geometry")]
    elif kind == "Feature":
        geoms = [data["geometry"]]
    elif kind in {"Polygon", "MultiPolygon"}:
        geoms = [data]
    else:
        raise ValueError(f"unsupported AOI GeoJSON type: {kind!r}")
    if not geoms:
        raise ValueError("AOI contains no geometries")
    return geoms


def transform_geoms(geoms: Sequence[Geometry], src_crs: CRS | str, dst_crs: CRS) -> list[Geometry]:
    return [transform_geom(src_crs, dst_crs, g) for g in geoms]


def geoms_bounds(geoms: Sequence[Geometry]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    def walk(coords: Any) -> None:
        if isinstance(coords[0], (int, float)):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
        else:
            for c in coords:
                walk(c)

    for g in geoms:
        walk(g["coordinates"])
    return min(xs), min(ys), max(xs), max(ys)


def crop_grid_to_geoms(grid: Grid, geoms: Sequence[Geometry], pad: int = 1) -> Grid:
    """Sub-grid (pixel-aligned with ``grid``) covering the geometries' bounds."""
    win = windows.from_bounds(*geoms_bounds(geoms), transform=grid.transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    win = Window(win.col_off - pad, win.row_off - pad, win.width + 2 * pad, win.height + 2 * pad)
    full = Window(0, 0, grid.width, grid.height)
    try:
        win = win.intersection(full)
    except rasterio.errors.WindowError as exc:
        raise ValueError("AOI does not intersect the scene") from exc
    return Grid(grid.crs, grid.window_transform(win), int(win.width), int(win.height))


def aligned_offset(outer: Grid, inner: Grid) -> tuple[int, int] | None:
    """(row, col) offset of ``inner`` in ``outer`` if both grids share pixels exactly."""
    if outer.crs != inner.crs or not np.allclose(
        [outer.transform.a, outer.transform.b, outer.transform.d, outer.transform.e],
        [inner.transform.a, inner.transform.b, inner.transform.d, inner.transform.e],
    ):
        return None
    col, row = ~outer.transform @ (inner.transform.c, inner.transform.f)
    if not (np.isclose(col, round(col), atol=1e-6) and np.isclose(row, round(row), atol=1e-6)):
        return None
    return round(row), round(col)


def aoi_mask(geoms: Sequence[Geometry], transform: Affine, shape: tuple[int, int]) -> np.ndarray:
    """True for pixels inside the AOI."""
    return features.geometry_mask(geoms, out_shape=shape, transform=transform, invert=True)


def source_window_for(dst_window: Window, dst: Grid, src: Grid, pad: int = 2) -> Window | None:
    """Source-pixel window covering a destination window (padded), or None if disjoint."""
    bounds = windows.bounds(dst_window, dst.transform)
    if dst.crs != src.crs:
        bounds = transform_bounds(dst.crs, src.crs, *bounds, densify_pts=21)
    win = windows.from_bounds(*bounds, transform=src.transform)
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    win = Window(win.col_off - pad, win.row_off - pad, win.width + 2 * pad, win.height + 2 * pad)
    try:
        return win.intersection(Window(0, 0, src.width, src.height))
    except rasterio.errors.WindowError:
        return None


def reproject_block(
    data: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_transform: Affine,
    dst_crs: CRS,
    dst_shape: tuple[int, int],
    nodata: float = NODATA,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Reproject a ``(B, h, w)`` block onto a destination window grid."""
    out = np.full((data.shape[0], *dst_shape), nodata, dtype=np.float32)
    reproject(
        source=data,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=nodata,
        resampling=resampling,
    )
    return out


def clip_to_aoi(
    src_path: str | Path, aoi: str | Path | Geometry, aoi_crs: CRS | str = WGS84
) -> tuple[np.ndarray, Affine, CRS]:
    """Crop a (small) raster to an AOI in memory; pixels outside become nodata."""
    with rasterio.open(src_path) as ds:
        geoms = transform_geoms(load_aoi(aoi), aoi_crs, ds.crs)
        nodata = ds.nodata if ds.nodata is not None else NODATA
        data, transform = rio_mask(ds, geoms, crop=True, nodata=nodata, filled=True)
        return data.astype(np.float32), transform, ds.crs


def nodata_mask(cube: np.ndarray, nodata: float | None = NODATA) -> np.ndarray:
    """True where any band is nodata or non-finite."""
    bad = ~np.isfinite(cube).all(axis=0)
    if nodata is not None and not np.isnan(nodata):
        bad |= (cube == nodata).any(axis=0)
    return bad


def _band(cube: np.ndarray, wavelengths: np.ndarray, nm: float) -> np.ndarray:
    return cube[int(np.argmin(np.abs(np.asarray(wavelengths) - nm)))]


def cloud_shadow_mask(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    cloud_blue: float = 0.25,
    cloud_ndvi: float = 0.2,
    shadow_nir: float = 0.04,
    shadow_swir: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """HEURISTIC cloud and shadow masks from reflectance thresholds -> ``(cloud, shadow)``.

    Cloud: bright in blue and spectrally flat (low NDVI). Shadow: dark in NIR and SWIR.
    Not a substitute for a proper cloud product (e.g. EnMAP L2A quality layers).
    """
    # TODO(Day 6): replace with sensor QA layers or a trained cloud classifier where available.
    blue = _band(cube, wavelengths, 470.0)
    red = _band(cube, wavelengths, 660.0)
    nir = _band(cube, wavelengths, 860.0)
    swir = _band(cube, wavelengths, 1650.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    cloud = (blue > cloud_blue) & (np.nan_to_num(ndvi, nan=0.0) < cloud_ndvi)
    shadow = (nir < shadow_nir) & (swir < shadow_swir) & (nir >= 0) & ~cloud
    return cloud, shadow
