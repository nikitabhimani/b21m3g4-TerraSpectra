"""Contract C1 validator: returns human-readable violations (empty list == valid)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from terraspectra_contracts.constants import (
    CONTRACT_VERSION,
    N_BANDS,
    NODATA,
    REFLECTANCE_RANGE,
    WAVELENGTH_TAG,
    WAVELENGTHS_NM,
)
from terraspectra_pipeline.geo import is_utm
from terraspectra_pipeline.writer import VALID_SOURCES

WAVELENGTH_TOLERANCE_NM = 0.05
SAMPLE_SIZE = 256
EXPECTED_BLOCK = 256


def validate_cube(path: str | Path, check_values: bool = True) -> list[str]:
    """Check ``path`` against C1 and return a list of violations."""
    try:
        ds = rasterio.open(path)
    except RasterioIOError as exc:
        return [f"cannot open: {exc}"]
    errors: list[str] = []
    with ds:
        if ds.driver != "GTiff":
            errors.append(f"driver is {ds.driver}, expected a (COG) GeoTIFF")
        layout = ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT", "")
        if layout.upper() != "COG":
            errors.append("file is not laid out as a Cloud-Optimized GeoTIFF (LAYOUT != COG)")
        if not ds.profile.get("tiled", False) or ds.block_shapes[0] != (
            EXPECTED_BLOCK,
            EXPECTED_BLOCK,
        ):
            errors.append(f"block shape {ds.block_shapes[0]}, expected 256x256 tiles")
        if (ds.compression is None) or ds.compression.name.upper() != "ZSTD":
            errors.append(f"compression is {ds.compression}, expected ZSTD")
        if max(ds.width, ds.height) > 2 * EXPECTED_BLOCK and not ds.overviews(1):
            errors.append("missing internal overviews")

        if ds.count != N_BANDS:
            errors.append(f"band count is {ds.count}, expected {N_BANDS}")
        if set(ds.dtypes) != {"float32"}:
            errors.append(f"dtype is {sorted(set(ds.dtypes))}, expected float32")
        if ds.nodata is None or not np.isclose(ds.nodata, NODATA):
            errors.append(f"nodata is {ds.nodata}, expected {NODATA}")

        if ds.crs is None:
            errors.append("no CRS")
        elif not is_utm(ds.crs):
            errors.append(f"CRS {ds.crs.to_string()} is not a projected UTM CRS")

        tags = ds.tags()
        if tags.get("contract_version") != CONTRACT_VERSION:
            errors.append(
                f"contract_version tag is {tags.get('contract_version')!r}, "
                f"expected {CONTRACT_VERSION!r}"
            )
        if tags.get("source") not in VALID_SOURCES:
            errors.append(f"source tag {tags.get('source')!r} not in {sorted(VALID_SOURCES)}")

        errors.extend(_check_wavelengths(ds))
        if check_values and ds.count == N_BANDS:
            errors.extend(_check_values(ds))
    return errors


def _check_wavelengths(ds: rasterio.io.DatasetReader) -> list[str]:
    missing: list[int] = []
    mismatched: list[str] = []
    for i in range(1, ds.count + 1):
        raw = ds.tags(i).get(WAVELENGTH_TAG)
        if raw is None:
            missing.append(i)
            continue
        if i > len(WAVELENGTHS_NM):
            continue
        try:
            value = float(raw)
        except ValueError:
            mismatched.append(f"band {i}: {raw!r}")
            continue
        if abs(value - WAVELENGTHS_NM[i - 1]) > WAVELENGTH_TOLERANCE_NM:
            mismatched.append(f"band {i}: {value} != {WAVELENGTHS_NM[i - 1]}")
    out: list[str] = []
    if missing:
        out.append(f"{len(missing)} band(s) lack '{WAVELENGTH_TAG}' tags (first: {missing[:5]})")
    if mismatched:
        out.append(f"{len(mismatched)} wavelength tag(s) off the canonical grid: {mismatched[:3]}")
    return out


def _check_values(ds: rasterio.io.DatasetReader) -> list[str]:
    h, w = min(SAMPLE_SIZE, ds.height), min(SAMPLE_SIZE, ds.width)
    # Decimated read over the whole extent (uses overviews when present).
    sample = ds.read(out_shape=(ds.count, h, w), out_dtype="float32")
    valid = sample[sample != NODATA]
    out: list[str] = []
    if valid.size == 0:
        return ["sampled pixels are all nodata"]
    if not np.isfinite(valid).all():
        out.append("sample contains NaN/inf values (use nodata=-1 instead)")
    lo, hi = REFLECTANCE_RANGE
    finite = valid[np.isfinite(valid)]
    if finite.size and (finite.min() < lo or finite.max() > hi):
        out.append(f"values outside [{lo}, {hi}]: min={finite.min():.4f}, max={finite.max():.4f}")
    return out
