"""Tests for terraspectra_pipeline.validate — Contract C1 validation checks."""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from terraspectra_pipeline.validate import validate_cube


def _write_basic_tif(path, bands=5, dtype="float32", crs="EPSG:4326", nodata=None):
    """Write a minimal, non-COG GeoTIFF, for testing individual validation checks."""
    data = np.zeros((bands, 16, 16), dtype=dtype)
    transform = from_origin(0, 0, 1, 1)

    kwargs = dict(
        driver="GTiff",
        height=16,
        width=16,
        count=bands,
        dtype=dtype,
        transform=transform,
    )
    if crs is not None:
        kwargs["crs"] = crs
    if nodata is not None:
        kwargs["nodata"] = nodata

    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data)


def test_validate_cube_missing_file_reports_cannot_open(tmp_path):
    """A path that doesn't exist should report a 'cannot open' error, not crash."""
    missing = tmp_path / "does_not_exist.tif"

    errors = validate_cube(missing)

    assert len(errors) == 1
    assert "cannot open" in errors[0]


def test_validate_cube_reports_wrong_band_count(tmp_path):
    """A file with the wrong number of bands should list a band-count violation."""
    path = tmp_path / "wrong_bands.tif"
    _write_basic_tif(path, bands=5)

    errors = validate_cube(path, check_values=False)

    assert any("band count" in e for e in errors)


def test_validate_cube_reports_wrong_dtype(tmp_path):
    """A file with an integer dtype (not float32) should list a dtype violation."""
    path = tmp_path / "wrong_dtype.tif"
    _write_basic_tif(path, bands=200, dtype="uint16")

    errors = validate_cube(path, check_values=False)

    assert any("dtype" in e for e in errors)


def test_validate_cube_reports_missing_crs(tmp_path):
    """A file with no CRS at all should list a 'no CRS' violation."""
    path = tmp_path / "no_crs.tif"
    _write_basic_tif(path, bands=200, crs=None)

    errors = validate_cube(path, check_values=False)

    assert any("no CRS" in e for e in errors)


def test_validate_cube_reports_wrong_nodata(tmp_path):
    """A file whose nodata value isn't -1.0 should list a nodata violation."""
    path = tmp_path / "wrong_nodata.tif"
    _write_basic_tif(path, bands=200, nodata=0.0)

    errors = validate_cube(path, check_values=False)

    assert any("nodata" in e for e in errors)


def test_validate_cube_synthetic_fixture_has_no_structural_errors(synthetic_cog):
    """The team's own synthetic fixture should be contract-compliant."""
    errors = validate_cube(synthetic_cog)

    assert not any("band count" in e for e in errors)
    assert not any("no CRS" in e for e in errors)
    assert not any("dtype" in e for e in errors)