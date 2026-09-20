"""C1 Cloud-Optimized GeoTIFF writer (windowed) and optional Zarr export."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import rasterio
import rasterio.shutil
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from terraspectra_contracts.constants import (
    CONTRACT_VERSION,
    N_BANDS,
    NODATA,
    WAVELENGTH_TAG,
    WAVELENGTHS_NM,
)

VALID_SOURCES = {"hyperion", "enmap", "prisma", "synthetic"}

COG_OPTIONS: dict[str, Any] = {
    "compress": "ZSTD",
    "blocksize": 256,
    "overviews": "AUTO",
    "overview_resampling": "AVERAGE",
    "bigtiff": "IF_SAFER",
    "num_threads": "ALL_CPUS",
}


class CubeWriter:
    """Write a C1 cube block by block, then convert to COG on close.

    Blocks go to a tiled scratch GeoTIFF next to the output, so memory stays bounded;
    the COG (with internal overviews) is produced by a GDAL CreateCopy at the end.
    """

    def __init__(
        self,
        path: str | Path,
        crs: CRS | str,
        transform: Affine,
        width: int,
        height: int,
        source: str,
        acquired_at: datetime | None = None,
        wavelengths: tuple[float, ...] = WAVELENGTHS_NM,
        extra_tags: dict[str, str] | None = None,
    ) -> None:
        if source not in VALID_SOURCES:
            raise ValueError(f"source must be one of {sorted(VALID_SOURCES)}, got {source!r}")
        self.path = Path(path)
        self.wavelengths = wavelengths
        self.tags: dict[str, str] = {"contract_version": CONTRACT_VERSION, "source": source}
        if acquired_at is not None:
            self.tags["acquired_at"] = acquired_at.isoformat()
        self.tags.update(extra_tags or {})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".tmp.tif", dir=self.path.parent)
        os.close(fd)
        self._tmp = Path(tmp)
        self._dst = rasterio.open(
            self._tmp,
            "w",
            driver="GTiff",
            dtype="float32",
            count=len(wavelengths),
            width=width,
            height=height,
            crs=crs,
            transform=transform,
            nodata=NODATA,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            compress="ZSTD",
            bigtiff="IF_SAFER",
        )

    @property
    def width(self) -> int:
        return int(self._dst.width)

    @property
    def height(self) -> int:
        return int(self._dst.height)

    def write(self, block: np.ndarray, window: Window) -> None:
        """Write a ``(B, h, w)`` float32 block at ``window``."""
        if block.shape[0] != len(self.wavelengths):
            raise ValueError(f"block has {block.shape[0]} bands, expected {len(self.wavelengths)}")
        self._dst.write(block.astype(np.float32, copy=False), window=window)

    def close(self) -> Path:
        """Finalise: tag bands, build the COG and remove the scratch file."""
        try:
            if not self._dst.closed:
                for i, wl in enumerate(self.wavelengths, start=1):
                    self._dst.update_tags(i, **{WAVELENGTH_TAG: f"{wl:.2f}"})
                    self._dst.set_band_description(i, f"{wl:.2f} nm")
                self._dst.update_tags(**self.tags)
                self._dst.close()
                rasterio.shutil.copy(self._tmp, self.path, driver="COG", **COG_OPTIONS)
        finally:
            self._tmp.unlink(missing_ok=True)
        return self.path

    def abort(self) -> None:
        if not self._dst.closed:
            self._dst.close()
        self._tmp.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_cog(
    cube: np.ndarray,
    path: str | Path,
    crs: CRS | str,
    transform: Affine,
    source: str,
    acquired_at: datetime | None = None,
) -> Path:
    """Write an in-memory ``(200, H, W)`` cube as a C1 COG."""
    if cube.shape[0] != N_BANDS:
        raise ValueError(f"C1 cubes need {N_BANDS} bands, got {cube.shape[0]}")
    _, h, w = cube.shape
    with CubeWriter(path, crs, transform, w, h, source, acquired_at) as writer:
        writer.write(cube, Window(0, 0, w, h))
    return Path(path)


def export_zarr(cog_path: str | Path, zarr_path: str | Path, chunk: int = 256) -> Path:
    """Copy a C1 COG into a chunked Zarr array (needs the ``zarr`` extra)."""
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError("Zarr export needs: uv sync --extra zarr") from exc

    from terraspectra_pipeline.chunking import iter_blocks

    with rasterio.open(cog_path) as src:
        arr = zarr.open_array(
            str(zarr_path),
            mode="w",
            shape=(src.count, src.height, src.width),
            chunks=(src.count, chunk, chunk),
            dtype="float32",
            fill_value=NODATA,
        )
        for win in iter_blocks(src.height, src.width, chunk):
            r, c = int(win.row_off), int(win.col_off)
            arr[:, r : r + int(win.height), c : c + int(win.width)] = src.read(window=win)
        arr.attrs.update(
            {
                **src.tags(),
                "crs": src.crs.to_wkt() if src.crs else None,
                "transform": list(src.transform)[:6],
                "nodata": NODATA,
                WAVELENGTH_TAG: [
                    float(src.tags(i)[WAVELENGTH_TAG]) for i in range(1, src.count + 1)
                ],
            }
        )
    return Path(zarr_path)
