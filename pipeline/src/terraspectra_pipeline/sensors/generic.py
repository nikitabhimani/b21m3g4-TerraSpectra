"""Reader for any multi-band GeoTIFF whose bands carry ``wavelength_nm`` tags (incl. C1 cubes)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from terraspectra_contracts.constants import WAVELENGTH_TAG
from terraspectra_pipeline.sensors.base import (
    Quantity,
    SceneMetadata,
    SensorReader,
    register_reader,
)

TIFF_SUFFIXES = {".tif", ".tiff"}


def estimate_fwhm(wavelengths: np.ndarray) -> np.ndarray:
    """FWHM ~ local band spacing when the file does not provide it."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    if wl.size < 2:
        return np.full(wl.shape, 10.0)
    return np.abs(np.gradient(wl))


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


class RasterioBandFileReader(SensorReader):
    """Shared windowed reading for single multi-band raster files."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._ds: DatasetReader | None = None

    @property
    def raster_path(self) -> Path:
        return self.path

    @property
    def dataset(self) -> DatasetReader:
        if self._ds is None or self._ds.closed:
            self._ds = rasterio.open(self.raster_path)
        return self._ds

    def read_window(self, window: Window) -> np.ndarray:
        return self.dataset.read(window=window, out_dtype="float32", boundless=False)

    def close(self) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_ds"] = None  # dataset handles are not picklable
        return state


@register_reader
class GenericGeoTIFFReader(RasterioBandFileReader):
    """Bands must carry ``wavelength_nm`` (and optionally ``fwhm_nm``) tags."""

    name = "generic"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        if path.suffix.lower() not in TIFF_SUFFIXES or not path.is_file():
            return False
        try:
            with rasterio.open(path) as ds:
                return WAVELENGTH_TAG in ds.tags(1)
        except rasterio.errors.RasterioIOError:
            return False

    def _load_metadata(self) -> SceneMetadata:
        ds = self.dataset
        band_tags = [ds.tags(i) for i in range(1, ds.count + 1)]
        missing = [i + 1 for i, t in enumerate(band_tags) if WAVELENGTH_TAG not in t]
        if missing:
            raise ValueError(f"{self.path}: bands {missing[:5]}... lack '{WAVELENGTH_TAG}' tags")
        wl = np.array([float(t[WAVELENGTH_TAG]) for t in band_tags])
        if all("fwhm_nm" in t for t in band_tags):
            fwhm = np.array([float(t["fwhm_nm"]) for t in band_tags])
        else:
            fwhm = estimate_fwhm(wl)
        tags = ds.tags()
        return SceneMetadata(
            sensor=self.name,
            wavelengths_nm=wl,
            fwhm_nm=fwhm,
            width=ds.width,
            height=ds.height,
            crs=ds.crs,
            transform=ds.transform,
            scale=np.asarray(ds.scales, dtype=np.float64),
            offset=np.asarray(ds.offsets, dtype=np.float64),
            quantity=Quantity(tags.get("quantity", Quantity.REFLECTANCE.value)),
            bad_bands=np.zeros(ds.count, dtype=bool),
            nodata=ds.nodata,
            acquired_at=parse_datetime(tags.get("acquired_at")),
            sun_elevation_deg=float(tags["sun_elevation"]) if "sun_elevation" in tags else None,
            scene_id=self.path.stem,
            extra={"source": tags.get("source", "unknown")},
        )
