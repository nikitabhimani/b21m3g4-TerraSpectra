"""Pre-visual stress indices from canonical-grid reflectance cubes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

from terraspectra_contracts.constants import NODATA, WAVELENGTHS_NM
from terraspectra_pipeline.chunking import iter_blocks

INDEX_NAMES = ("ndvi", "ndre", "rep", "pri", "cci", "mcari", "ndwi")
INDEX_DESCRIPTIONS = {
    "ndvi": "NDVI (R800-R670)/(R800+R670)",
    "ndre": "NDRE (R790-R720)/(R790+R720)",
    "rep": "Red-edge position, nm (linear four-point)",
    "pri": "PRI (R531-R570)/(R531+R570)",
    "cci": "CCI (R531-R645)/(R531+R645)",
    "mcari": "MCARI [(R700-R670)-0.2(R700-R550)](R700/R670)",
    "ndwi": "NDWI Gao (R860-R1240)/(R860+R1240)",
}
INDEX_NODATA = np.float32(np.nan)


class BandLookup:
    """Nearest-band accessor ``R(nm)`` for a ``(B, H, W)`` cube."""

    def __init__(self, cube: np.ndarray, wavelengths: np.ndarray | None = None) -> None:
        self.cube = cube
        self.wl = np.asarray(WAVELENGTHS_NM if wavelengths is None else wavelengths)
        if self.wl.size != cube.shape[0]:
            raise ValueError(f"{self.wl.size} wavelengths for {cube.shape[0]} bands")

    def index_of(self, nm: float) -> int:
        return int(np.argmin(np.abs(self.wl - nm)))

    def __call__(self, nm: float) -> np.ndarray:
        return self.cube[self.index_of(nm)].astype(np.float64)


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a + b != 0, (a - b) / (a + b), np.nan)


def ndvi(r: BandLookup) -> np.ndarray:
    return _nd(r(800), r(670))


def ndre(r: BandLookup) -> np.ndarray:
    return _nd(r(790), r(720))


def red_edge_position(r: BandLookup) -> np.ndarray:
    """Guyot & Baret linear four-point REP (nm)."""
    r670, r700, r740, r780 = r(670), r(700), r(740), r(780)
    r_re = (r670 + r780) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rep = 700.0 + 40.0 * (r_re - r700) / (r740 - r700)
    return np.where(r740 != r700, rep, np.nan)


def pri(r: BandLookup) -> np.ndarray:
    return _nd(r(531), r(570))


def cci(r: BandLookup) -> np.ndarray:
    return _nd(r(531), r(645))


def mcari(r: BandLookup) -> np.ndarray:
    r550, r670, r700 = r(550), r(670), r(700)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r670 != 0, ((r700 - r670) - 0.2 * (r700 - r550)) * (r700 / r670), np.nan)


def ndwi(r: BandLookup) -> np.ndarray:
    return _nd(r(860), r(1240))


INDEX_FUNCS: dict[str, Callable[[BandLookup], np.ndarray]] = {
    "ndvi": ndvi,
    "ndre": ndre,
    "rep": red_edge_position,
    "pri": pri,
    "cci": cci,
    "mcari": mcari,
    "ndwi": ndwi,
}


def compute_indices(
    cube: np.ndarray,
    wavelengths: np.ndarray | None = None,
    names: tuple[str, ...] = INDEX_NAMES,
    nodata: float = NODATA,
) -> dict[str, np.ndarray]:
    """Return ``{name: float32 (H, W)}``; nodata pixels are NaN."""
    lookup = BandLookup(cube, wavelengths)
    invalid = (cube == nodata).any(axis=0)
    out: dict[str, np.ndarray] = {}
    for name in names:
        arr = INDEX_FUNCS[name](lookup).astype(np.float32)
        arr[invalid] = INDEX_NODATA
        out[name] = arr
    return out


def _profile(
    crs: CRS | None, transform: Affine, width: int, height: int, count: int
) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "count": count,
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "nodata": INDEX_NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "ZSTD",
    }


def _describe(dst: rasterio.io.DatasetWriter, names: tuple[str, ...]) -> None:
    for i, name in enumerate(names, start=1):
        dst.set_band_description(i, name)
        dst.update_tags(i, index=name, formula=INDEX_DESCRIPTIONS[name])


def write_indices(
    path: str | Path,
    indices: dict[str, np.ndarray],
    crs: CRS | None,
    transform: Affine,
) -> Path:
    """Write an in-memory index dict as a multi-band GeoTIFF with band descriptions."""
    names = tuple(indices)
    h, w = next(iter(indices.values())).shape
    with rasterio.open(path, "w", **_profile(crs, transform, w, h, len(names))) as dst:
        for i, name in enumerate(names, start=1):
            dst.write(indices[name].astype(np.float32), i)
        _describe(dst, names)
    return Path(path)


def indices_from_cube(
    cube_path: str | Path,
    out_path: str | Path,
    names: tuple[str, ...] = INDEX_NAMES,
    block_size: int = 512,
) -> Path:
    """Compute indices block by block from a C1 cube and write ``indices.tif``."""
    with rasterio.open(cube_path) as src:
        wl = np.array([float(src.tags(i)["wavelength_nm"]) for i in range(1, src.count + 1)])
        nodata = src.nodata if src.nodata is not None else NODATA
        profile = _profile(src.crs, src.transform, src.width, src.height, len(names))
        with rasterio.open(out_path, "w", **profile) as dst:
            for win in iter_blocks(src.height, src.width, block_size):
                block = src.read(window=win, out_dtype="float32")
                result = compute_indices(block, wl, names, nodata)
                dst.write(np.stack([result[n] for n in names]), window=win)
            _describe(dst, names)
    return Path(out_path)
