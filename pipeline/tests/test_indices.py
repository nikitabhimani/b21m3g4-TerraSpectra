from pathlib import Path

import numpy as np
import rasterio

from terraspectra_contracts.constants import NODATA, WAVELENGTHS_NM
from terraspectra_contracts.fixtures import _vegetation_spectrum, make_synthetic_cube
from terraspectra_pipeline.indices import (
    INDEX_NAMES,
    BandLookup,
    compute_indices,
    indices_from_cube,
    write_indices,
)

WL = np.asarray(WAVELENGTHS_NM)


def _flat_cube(values: dict[float, float], default: float = 0.2) -> np.ndarray:
    cube = np.full((WL.size, 2, 2), default, dtype=np.float32)
    lookup = BandLookup(cube)
    for nm, v in values.items():
        cube[lookup.index_of(nm)] = v
    return cube


def test_ndvi_known_values() -> None:
    cube = _flat_cube({800: 0.5, 670: 0.1})
    ndvi = compute_indices(cube, names=("ndvi",))["ndvi"]
    np.testing.assert_allclose(ndvi, (0.5 - 0.1) / (0.5 + 0.1), rtol=1e-6)


def test_rep_of_healthy_spectrum_in_red_edge_and_shifts_when_stressed() -> None:
    healthy = _vegetation_spectrum(WL, 1.0)[:, None, None].astype(np.float32)
    stressed = _vegetation_spectrum(WL, 0.6)[:, None, None].astype(np.float32)
    rep_h = float(compute_indices(healthy, names=("rep",))["rep"][0, 0])
    rep_s = float(compute_indices(stressed, names=("rep",))["rep"][0, 0])
    assert 700.0 <= rep_h <= 740.0
    assert rep_s < rep_h


def test_all_indices_finite_and_nodata_is_nan() -> None:
    cube, _ = make_synthetic_cube(16, 16)
    out = compute_indices(cube)
    assert set(out) == set(INDEX_NAMES)
    for arr in out.values():
        assert arr.shape == (16, 16) and arr.dtype == np.float32
        assert np.isnan(arr[0, 0])  # fixture nodata corner
        assert np.isfinite(arr[4:, 4:]).all()
    assert (out["ndvi"][4:, 4:] > 0.5).all()


def test_write_and_blockwise_indices(tmp_path: Path, synthetic_cog: Path) -> None:
    out = indices_from_cube(synthetic_cog, tmp_path / "indices.tif", block_size=40)
    with rasterio.open(out) as ds, rasterio.open(synthetic_cog) as src:
        assert ds.count == len(INDEX_NAMES)
        assert ds.descriptions == INDEX_NAMES
        cube = src.read()
        expected = compute_indices(cube, nodata=NODATA)
        np.testing.assert_allclose(ds.read(1), expected["ndvi"], equal_nan=True, rtol=1e-6)

    single = write_indices(tmp_path / "one.tif", {"ndvi": expected["ndvi"]}, src.crs, src.transform)
    with rasterio.open(single) as ds:
        assert ds.descriptions == ("ndvi",)
