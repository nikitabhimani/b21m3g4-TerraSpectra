"""Tests for terraspectra_pipeline.writer — C1 COG writer."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from terraspectra_contracts.constants import N_BANDS, NODATA, WAVELENGTH_TAG
from terraspectra_pipeline.writer import CubeWriter, export_zarr, write_cog


def _sample_cube(bands=N_BANDS, h=32, w=32):
    rng = np.random.default_rng(0)
    return rng.random((bands, h, w), dtype=np.float32)


def _sample_crs_transform():
    crs = CRS.from_epsg(32643)  # UTM zone 43N
    transform = Affine.translation(500000, 3000000) * Affine.scale(30, -30)
    return crs, transform


def test_write_cog_creates_contract_compliant_file(tmp_path):
    """write_cog should produce a file with correct bands, dtype, nodata and tags."""
    cube = _sample_cube()
    crs, transform = _sample_crs_transform()
    out_path = tmp_path / "cube.tif"

    result_path = write_cog(cube, out_path, crs, transform, source="synthetic")

    with rasterio.open(result_path) as ds:
        assert ds.count == N_BANDS
        assert set(ds.dtypes) == {"float32"}
        assert ds.nodata == NODATA
        assert ds.tags().get("source") == "synthetic"
        assert ds.tags().get("contract_version") is not None
        assert ds.tags(1).get(WAVELENGTH_TAG) is not None


def test_write_cog_rejects_wrong_band_count(tmp_path):
    """write_cog should reject a cube that doesn't have exactly N_BANDS bands."""
    cube = _sample_cube(bands=5)
    crs, transform = _sample_crs_transform()
    out_path = tmp_path / "wrong_bands.tif"

    with pytest.raises(ValueError):
        write_cog(cube, out_path, crs, transform, source="synthetic")


def test_cube_writer_rejects_invalid_source(tmp_path):
    """CubeWriter should reject a source not in VALID_SOURCES."""
    crs, transform = _sample_crs_transform()
    out_path = tmp_path / "invalid_source.tif"

    with pytest.raises(ValueError):
        CubeWriter(out_path, crs, transform, width=16, height=16, source="not_a_real_sensor")


def test_cube_writer_reports_width_and_height(tmp_path):
    """CubeWriter.width/.height should match the requested dimensions."""
    crs, transform = _sample_crs_transform()
    out_path = tmp_path / "dims.tif"

    writer = CubeWriter(out_path, crs, transform, width=20, height=10, source="synthetic")
    try:
        assert writer.width == 20
        assert writer.height == 10
    finally:
        writer.abort()


def test_cube_writer_write_rejects_wrong_block_bands(tmp_path):
    """Writing a block with the wrong number of bands should raise ValueError."""
    crs, transform = _sample_crs_transform()
    out_path = tmp_path / "wrong_block.tif"

    writer = CubeWriter(out_path, crs, transform, width=16, height=16, source="synthetic")
    try:
        bad_block = np.zeros((3, 16, 16), dtype=np.float32)  # wrong band count
        with pytest.raises(ValueError):
            writer.write(bad_block, Window(0, 0, 16, 16))
    finally:
        writer.abort()


def test_export_zarr_without_zarr_installed_raises_clear_error(synthetic_cog, tmp_path, monkeypatch):
    """Without the zarr extra installed, export_zarr should raise a clear ImportError."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("No module named 'zarr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    zarr_out = tmp_path / "out.zarr"

    with pytest.raises(ImportError, match="Zarr export needs"):
        export_zarr(synthetic_cog, zarr_out)