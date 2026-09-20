from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terraspectra_contracts.constants import N_BANDS, WAVELENGTHS_NM
from terraspectra_contracts.fixtures import make_synthetic_cube
from terraspectra_pipeline.validate import validate_cube
from terraspectra_pipeline.writer import CubeWriter, export_zarr, write_cog

from .helpers import CRS_UTM, ORIGIN, write_multiband


def test_fixture_cube_is_valid(synthetic_cog: Path) -> None:
    assert validate_cube(synthetic_cog) == []


def test_writer_roundtrip_is_valid(tmp_path: Path) -> None:
    cube, _ = make_synthetic_cube(40, 50)
    when = datetime(2024, 7, 1, 5, 30, tzinfo=UTC)
    out = write_cog(
        cube, tmp_path / "out.tif", CRS_UTM, from_origin(*ORIGIN, 30, 30), "enmap", when
    )
    assert validate_cube(out) == []
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(), cube)
        assert ds.tags()["acquired_at"] == when.isoformat()
        assert float(ds.tags(200)["wavelength_nm"]) == pytest.approx(WAVELENGTHS_NM[-1])
    assert not list(tmp_path.glob("*.tmp.tif"))


def test_large_cube_gets_overviews(tmp_path: Path) -> None:
    """Block-wise writes of a raster wider than 512 px must produce internal overviews."""
    path = tmp_path / "big.tif"
    # A smooth cube (not noise) keeps this test fast: 200 bands of random floats do not compress.
    block = np.tile(np.linspace(0.05, 0.6, N_BANDS, dtype=np.float32)[:, None, None], (1, 300, 600))
    with CubeWriter(path, CRS_UTM, from_origin(*ORIGIN, 30, 30), 600, 530, "synthetic") as w:
        w.write(block, rasterio.windows.Window(0, 0, 600, 300))
        w.write(block[:, :230], rasterio.windows.Window(0, 300, 600, 230))
    with rasterio.open(path) as ds:
        assert ds.overviews(1)
    assert validate_cube(path) == []


def test_three_band_file_fails(tmp_path: Path) -> None:
    data = np.full((3, 16, 16), 0.5, dtype=np.float32)
    path = write_multiband(tmp_path / "rgb.tif", data)
    problems = validate_cube(path)
    joined = "\n".join(problems)
    assert "band count is 3" in joined
    assert "COG" in joined and "nodata" in joined and "contract_version" in joined


def test_detects_bad_values_crs_and_tags(tmp_path: Path) -> None:
    data = np.full((N_BANDS, 8, 8), 1.5, dtype=np.float32)
    tags = [{"wavelength_nm": str(w + 1.0)} for w in WAVELENGTHS_NM]
    path = write_multiband(
        tmp_path / "bad.tif", data, crs="EPSG:4326", pixel=0.001, nodata=-1.0, band_tags=tags
    )
    joined = "\n".join(validate_cube(path))
    assert "values outside" in joined
    assert "not a projected UTM" in joined
    assert "off the canonical grid" in joined


def test_unopenable_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk.tif"
    junk.write_text("not a tiff")
    assert validate_cube(junk)[0].startswith("cannot open")


def test_writer_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CubeWriter(tmp_path / "x.tif", CRS_UTM, from_origin(*ORIGIN, 30, 30), 4, 4, "landsat")
    with pytest.raises(ValueError):
        write_cog(
            np.zeros((3, 4, 4), np.float32),
            tmp_path / "x.tif",
            CRS_UTM,
            from_origin(*ORIGIN, 30, 30),
            "synthetic",
        )
    with (
        pytest.raises(RuntimeError),
        CubeWriter(tmp_path / "y.tif", CRS_UTM, from_origin(*ORIGIN, 30, 30), 4, 4, "synthetic"),
    ):
        raise RuntimeError("boom")
    assert not list(tmp_path.glob("*.tif"))


def test_zarr_export(tmp_path: Path, synthetic_cog: Path) -> None:
    zarr = pytest.importorskip("zarr")
    store = export_zarr(synthetic_cog, tmp_path / "cube.zarr", chunk=32)
    arr = zarr.open_array(str(store), mode="r")
    with rasterio.open(synthetic_cog) as ds:
        np.testing.assert_array_equal(arr[:], ds.read())
    assert arr.attrs["source"] == "synthetic"
    assert len(arr.attrs["wavelength_nm"]) == N_BANDS
