"""PRISMA L2D HE5 reader (needs the ``prisma`` extra: h5py). Georeferencing is still TODO."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from rasterio.windows import Window

from terraspectra_pipeline.sensors.base import (
    Quantity,
    SceneMetadata,
    SensorReader,
    register_reader,
)
from terraspectra_pipeline.sensors.generic import parse_datetime

SWATH = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields"
CUBES = ("VNIR_Cube", "SWIR_Cube")
SUFFIXES = {".he5", ".h5"}


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError("PRISMA support needs: uv sync --extra prisma") from exc
    return h5py


def _attr(attrs: Any, key: str, default: Any = None) -> Any:
    val = attrs.get(key, default)
    return val.decode() if isinstance(val, bytes) else val


@register_reader
class PrismaReader(SensorReader):
    """Reads VNIR+SWIR cubes from a PRISMA L2D HE5 by slicing (lazy, windowed).

    Cubes are stored as (rows, cols, bands) with wavelengths in descending order; the reader
    concatenates VNIR then SWIR and sorts bands ascending.
    """

    name = "prisma"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._h5: Any = None
        self._order: np.ndarray | None = None

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in SUFFIXES and path.name.startswith("PRS_")

    @property
    def h5(self) -> Any:
        if self._h5 is None:
            self._h5 = _h5py().File(self.path, "r")
        return self._h5

    def _load_metadata(self) -> SceneMetadata:
        f = self.h5
        attrs = f.attrs
        wl_parts, fwhm_parts, scale_parts, offset_parts = [], [], [], []
        shape: tuple[int, int] | None = None
        for cube, tag in zip(CUBES, ("Vnir", "Swir"), strict=True):
            ds = f[f"{SWATH}/{cube}"]
            n = ds.shape[-1]
            shape = shape or (ds.shape[0], ds.shape[1])
            wl = np.asarray(_attr(attrs, f"List_Cw_{tag}"), dtype=np.float64)[:n]
            fwhm = np.asarray(_attr(attrs, f"List_Fwhm_{tag}"), dtype=np.float64)[:n]
            lo = float(_attr(attrs, f"L2Scale{tag}Min", 0.0))
            hi = float(_attr(attrs, f"L2Scale{tag}Max", 1.0))
            wl_parts.append(wl)
            fwhm_parts.append(fwhm)
            # L2D stores reflectance as uint16: refl = lo + DN * (hi - lo) / 65535.
            scale_parts.append(np.full(n, (hi - lo) / 65535.0))
            offset_parts.append(np.full(n, lo))
        assert shape is not None
        wl = np.concatenate(wl_parts)
        # PRISMA flags unused channels with wavelength 0.
        order = np.argsort(np.where(wl > 0, wl, np.inf), kind="stable")
        self._order = order
        acquired: datetime | None = parse_datetime(_attr(attrs, "Product_StartTime"))
        sun_zenith = _attr(attrs, "Sun_zenith_angle")
        # TODO(Day 3): derive CRS/transform from Latitude/Longitude geolocation fields
        # (L2D is UTM-projected; Product_ULcorner_easting/northing + Epsg_Code attrs).
        return SceneMetadata(
            sensor=self.name,
            wavelengths_nm=wl[order],
            fwhm_nm=np.concatenate(fwhm_parts)[order],
            width=shape[1],
            height=shape[0],
            crs=None,
            transform=Affine.identity(),
            scale=np.concatenate(scale_parts)[order],
            offset=np.concatenate(offset_parts)[order],
            quantity=Quantity.REFLECTANCE,
            bad_bands=(wl[order] <= 0),
            nodata=None,
            acquired_at=acquired,
            sun_elevation_deg=90.0 - float(sun_zenith) if sun_zenith is not None else None,
            scene_id=self.path.stem,
        )

    def read_window(self, window: Window) -> np.ndarray:
        m = self.metadata
        assert self._order is not None
        r0, c0 = int(window.row_off), int(window.col_off)
        r1, c1 = r0 + int(window.height), c0 + int(window.width)
        parts = [
            np.asarray(self.h5[f"{SWATH}/{cube}"][r0:r1, c0:c1, :], dtype=np.float32)
            for cube in CUBES
        ]
        stacked = np.concatenate(parts, axis=-1).transpose(2, 0, 1)
        out = stacked[self._order]
        assert out.shape[0] == m.band_count
        return np.ascontiguousarray(out)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state
