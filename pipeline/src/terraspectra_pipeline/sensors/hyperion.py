"""EO-1 Hyperion L1 GeoTIFF bundle: one GeoTIFF per band + an MTL text file."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from terraspectra_pipeline.sensors.base import (
    Quantity,
    SceneMetadata,
    SensorReader,
    register_reader,
)

N_HYPERION_BANDS = 242
VNIR_BANDS = range(1, 71)  # 1-based; bands 71-242 are SWIR
# Calibrated bands per the USGS Hyperion handbook: VNIR 8-57, SWIR 77-224 (1-based).
CALIBRATED_BANDS = (range(8, 58), range(77, 225))
VNIR_SCALE, SWIR_SCALE = 40.0, 80.0  # DN / scale = radiance (W m-2 sr-1 um-1)

_BAND_RE = re.compile(r"_B(\d{3})(?:_L1[A-Z0-9]*)?\.TIF{1,2}$", re.IGNORECASE)
_KV_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.+?)\s*$")


def hyperion_wavelengths() -> tuple[np.ndarray, np.ndarray]:
    """Nominal centre wavelengths and FWHM (nm) for the 242 bands.

    VNIR and SWIR centres are linear to within ~0.1 nm of the published table.
    """
    # TODO(Day 2): prefer per-scene values from the L1R SPECTRUM/.hdr table when bundled.
    vnir = np.linspace(355.59, 1057.68, 70)
    swir = np.linspace(851.92, 2577.08, 172)
    wl = np.concatenate([vnir, swir])
    fwhm = np.full(N_HYPERION_BANDS, 10.9)
    return wl, fwhm


def hyperion_bad_bands() -> np.ndarray:
    """True for uncalibrated / overlapping bands (1-7, 58-76, 225-242)."""
    bad = np.ones(N_HYPERION_BANDS, dtype=bool)
    for rng in CALIBRATED_BANDS:
        bad[[b - 1 for b in rng]] = False
    return bad


def parse_mtl(path: Path) -> dict[str, str]:
    """Flatten an ODL-style ``KEY = VALUE`` MTL file (group structure ignored)."""
    out: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = _KV_RE.match(line)
        if m and m.group(1) not in {"GROUP", "END_GROUP"}:
            out[m.group(1)] = m.group(2).strip('"')
    return out


def _parse_acquisition(mtl: dict[str, str]) -> datetime | None:
    date = mtl.get("ACQUISITION_DATE") or mtl.get("DATE_ACQUIRED")
    if not date:
        return None
    time = (mtl.get("SCENE_CENTER_SCAN_TIME") or mtl.get("SCENE_CENTER_TIME") or "").strip('"')
    try:
        stamp = datetime.fromisoformat(f"{date}T{time[:8]}" if time else date)
    except ValueError:
        return None
    return stamp.replace(tzinfo=UTC)


def _float(mtl: dict[str, str], *keys: str) -> float | None:
    for k in keys:
        if k in mtl:
            try:
                return float(mtl[k])
            except ValueError:
                continue
    return None


@register_reader
class HyperionReader(SensorReader):
    """Reads a Hyperion L1T/L1Gst bundle directory lazily, one band file at a time."""

    name = "hyperion"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.root = self.path if self.path.is_dir() else self.path.parent
        self.band_files = self._find_band_files(self.root)
        self._datasets: dict[int, DatasetReader] = {}

    @staticmethod
    def _find_band_files(root: Path) -> dict[int, Path]:
        files: dict[int, Path] = {}
        for p in root.iterdir() if root.is_dir() else []:
            m = _BAND_RE.search(p.name)
            if m and 1 <= int(m.group(1)) <= N_HYPERION_BANDS:
                files[int(m.group(1))] = p
        return files

    @staticmethod
    def _find_mtl(root: Path) -> Path | None:
        hits = sorted(p for p in root.glob("*") if re.search(r"MTL.*\.TXT$", p.name, re.I))
        return hits[0] if hits else None

    @classmethod
    def can_read(cls, path: Path) -> bool:
        root = path if path.is_dir() else path.parent
        if not root.is_dir() or cls._find_mtl(root) is None:
            return False
        return any(
            _BAND_RE.search(p.name) and p.name.upper().startswith("EO1H") for p in root.iterdir()
        )

    def _dataset(self, band: int) -> DatasetReader:
        ds = self._datasets.get(band)
        if ds is None or ds.closed:
            ds = rasterio.open(self.band_files[band])
            self._datasets[band] = ds
        return ds

    def _load_metadata(self) -> SceneMetadata:
        if not self.band_files:
            raise FileNotFoundError(f"no Hyperion band files found in {self.root}")
        mtl_path = self._find_mtl(self.root)
        mtl = parse_mtl(mtl_path) if mtl_path else {}
        wl, fwhm = hyperion_wavelengths()
        bad = hyperion_bad_bands()
        missing = [b for b in range(1, N_HYPERION_BANDS + 1) if b not in self.band_files]
        bad[[b - 1 for b in missing]] = True

        vnir_scale = _float(mtl, "SCALING_FACTOR_VNIR", "RADIANCE_SCALING_FACTOR_VNIR")
        swir_scale = _float(mtl, "SCALING_FACTOR_SWIR", "RADIANCE_SCALING_FACTOR_SWIR")
        scale = np.array(
            [
                1.0 / (vnir_scale or VNIR_SCALE)
                if b in VNIR_BANDS
                else 1.0 / (swir_scale or SWIR_SCALE)
                for b in range(1, N_HYPERION_BANDS + 1)
            ]
        )
        first = self._dataset(min(self.band_files))
        extra: dict[str, Any] = {"missing_band_files": len(missing)}
        if mtl_path:
            extra["mtl"] = mtl_path.name
        return SceneMetadata(
            sensor=self.name,
            wavelengths_nm=wl,
            fwhm_nm=fwhm,
            width=first.width,
            height=first.height,
            crs=first.crs,
            transform=first.transform,
            scale=scale,
            offset=np.zeros(N_HYPERION_BANDS),
            quantity=Quantity.RADIANCE,
            bad_bands=bad,
            nodata=0.0,
            acquired_at=_parse_acquisition(mtl),
            sun_elevation_deg=_float(mtl, "SUN_ELEVATION"),
            scene_id=mtl.get("ENTITY_ID") or self.root.name,
            extra=extra,
        )

    def read_window(self, window: Window) -> np.ndarray:
        """Read only calibrated bands; flagged bands are returned as zeros."""
        m = self.metadata
        h, w = int(window.height), int(window.width)
        out = np.zeros((N_HYPERION_BANDS, h, w), dtype=np.float32)
        for idx in np.flatnonzero(~m.bad_bands):
            out[idx] = self._dataset(int(idx) + 1).read(1, window=window, out_dtype="float32")
        return out

    def close(self) -> None:
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_datasets"] = {}
        return state
