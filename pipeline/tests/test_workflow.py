import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from terraspectra_contracts.constants import N_BANDS, NODATA, WAVELENGTH_TAG, WAVELENGTHS_NM
from terraspectra_contracts.fixtures import _vegetation_spectrum, make_synthetic_cube
from terraspectra_pipeline import geo
from terraspectra_pipeline.normalize import compute_file_stats
from terraspectra_pipeline.radiometry import earth_sun_distance_au, solar_irradiance
from terraspectra_pipeline.sensors import open_scene
from terraspectra_pipeline.sensors.hyperion import hyperion_wavelengths
from terraspectra_pipeline.validate import validate_cube
from terraspectra_pipeline.workflow import ProcessOptions, process_scene

from .helpers import CRS_UTM, ORIGIN, write_multiband
from .test_sensors import make_enmap_product, make_hyperion_bundle


def _aoi(path: Path, bounds_utm: tuple[float, float, float, float]) -> Path:
    w, s, e, n = transform_bounds(CRS_UTM, "EPSG:4326", *bounds_utm)
    ring = [[w, s], [e, s], [e, n], [w, n], [w, s]]
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    return path


def test_reprocess_c1_cube_is_identity(tmp_path: Path, synthetic_cog: Path) -> None:
    out = tmp_path / "out.tif"
    with open_scene(synthetic_cog) as reader:
        res = process_scene(reader, out, options=ProcessOptions(block_size=40, cloud_mask=False))
    assert validate_cube(out) == []
    assert (res.width, res.height) == (64, 64) and res.blocks == 4
    with rasterio.open(out) as a, rasterio.open(synthetic_cog) as b:
        src, dst = b.read(), a.read()
        valid = dst[0] != NODATA
        assert valid.sum() == 64 * 64 - 4  # the fixture's nodata corner
        # Canonical-grid input only gets a mild SRF smoothing.
        assert np.abs(dst[:, valid] - src[:, valid]).mean() < 0.01
        assert a.tags()["source"] == "synthetic"


def test_aoi_clip_and_normalisation(tmp_path: Path, synthetic_cog: Path) -> None:
    left, top = ORIGIN
    aoi = _aoi(tmp_path / "farm.geojson", (left + 300, top - 900, left + 900, top - 300))
    stats_path = compute_file_stats(synthetic_cog, n_windows=4, window_size=32).save(
        tmp_path / "stats.json"
    )
    from terraspectra_pipeline.normalize import BandStats

    opts = ProcessOptions(stats=BandStats.load(stats_path), cloud_mask=False)
    with open_scene(synthetic_cog) as reader:
        res = process_scene(reader, tmp_path / "clip.tif", aoi=aoi, options=opts)
    assert validate_cube(res.path) == []
    assert 20 <= res.width <= 24 and 20 <= res.height <= 24
    with rasterio.open(res.path) as ds:
        data = ds.read()
        assert ds.tags()["normalized"] == "true"
    valid = data[0] != NODATA
    assert valid.any() and not valid.all()  # padded border lies outside the AOI polygon
    assert data[:, valid].min() >= 0 and data[:, valid].max() <= 1


def test_reprojects_geographic_input_to_utm(tmp_path: Path) -> None:
    cube, _ = make_synthetic_cube(32, 32)
    tags = [{WAVELENGTH_TAG: f"{w:.2f}"} for w in WAVELENGTHS_NM]
    src = tmp_path / "geo.tif"
    with rasterio.open(
        src,
        "w",
        driver="GTiff",
        dtype="float32",
        count=N_BANDS,
        width=32,
        height=32,
        crs="EPSG:4326",
        transform=from_origin(75.0, 31.0, 0.0003, 0.0003),
        nodata=NODATA,
    ) as dst:
        dst.write(cube)
        for i, t in enumerate(tags, start=1):
            dst.update_tags(i, **t)
    out = tmp_path / "utm.tif"
    with open_scene(src) as reader:
        res = process_scene(
            reader, out, options=ProcessOptions(block_size=16, cloud_mask=False, source="synthetic")
        )
    assert res.crs == "EPSG:32643"
    assert validate_cube(out) == []
    assert res.valid_fraction > 0.5


def test_hyperion_end_to_end(tmp_path: Path) -> None:
    root = make_hyperion_bundle(tmp_path / "hyp", size=6)
    wl, fwhm = hyperion_wavelengths()
    rho = _vegetation_spectrum(wl, 1.0)
    esun = solar_irradiance(wl, fwhm)
    d = earth_sun_distance_au(123)
    radiance = rho * esun * np.sin(np.deg2rad(62.5)) / (np.pi * d**2)
    scale = np.where(np.arange(1, 243) <= 70, 40.0, 80.0)
    dn = np.round(radiance * scale).astype(np.int16)
    stem = "EO1H1470392004123110KZ"
    for b in range(1, 243):
        band = np.full((1, 6, 6), dn[b - 1], dtype=np.int16)
        band[:, 0, 0] = 0
        write_multiband(root / f"{stem}_B{b:03d}_L1T.TIF", band, nodata=0)

    out = tmp_path / "hyp.tif"
    with open_scene(root) as reader:
        res = process_scene(reader, out, options=ProcessOptions(smooth=True))
    assert validate_cube(out) == []
    with rasterio.open(out) as ds:
        data = ds.read()
        assert ds.tags()["source"] == "hyperion" and "acquired_at" in ds.tags()
    assert data[0, 0, 0] == NODATA
    expected = _vegetation_spectrum(np.asarray(WAVELENGTHS_NM), 1.0)
    got = data[:, 3, 3]
    covered = (np.asarray(WAVELENGTHS_NM) > 450) & (np.asarray(WAVELENGTHS_NM) < 2350)
    assert np.abs(got[covered] - expected[covered]).max() < 0.05
    assert res.valid_fraction == pytest.approx(35 / 36)


def test_enmap_parallel_workers(tmp_path: Path) -> None:
    root = make_enmap_product(tmp_path / "enmap", n=40, size=20)
    with open_scene(root) as reader:
        res = process_scene(
            reader, tmp_path / "enmap.tif", options=ProcessOptions(block_size=8, workers=2)
        )
    assert validate_cube(res.path) == []
    with rasterio.open(res.path) as ds:
        data = ds.read()
    assert data[0, 0, 0] == NODATA
    np.testing.assert_allclose(data[:, 5, 5], 0.25, atol=1e-4)


def test_process_rejects_scene_without_crs_or_source(tmp_path: Path) -> None:
    data = np.full((N_BANDS, 4, 4), 0.2, dtype=np.float32)
    tags = [{WAVELENGTH_TAG: str(w)} for w in WAVELENGTHS_NM]
    path = write_multiband(tmp_path / "nosrc.tif", data, band_tags=tags)
    with open_scene(path) as reader, pytest.raises(ValueError, match="source"):
        process_scene(reader, tmp_path / "x.tif")


def test_grid_helpers() -> None:
    assert geo.utm_crs_for_lonlat(75.1, 30.9).to_epsg() == 32643
    assert geo.utm_crs_for_lonlat(-70.0, -33.0).to_epsg() == 32719
    grid = geo.Grid(rasterio.crs.CRS.from_string(CRS_UTM), from_origin(*ORIGIN, 30, 30), 10, 10)
    assert geo.plan_target_grid(grid) is grid
    coarser = geo.plan_target_grid(grid, resolution=60.0)
    assert coarser.width == 5
    sub = geo.Grid(grid.crs, from_origin(ORIGIN[0] + 60, ORIGIN[1] - 30, 30, 30), 3, 3)
    assert geo.aligned_offset(grid, sub) == (1, 2)
    assert geo.aligned_offset(grid, coarser) is None
    with pytest.raises(ValueError):
        geo.load_aoi({"type": "Point", "coordinates": [0, 0]})


def test_cloud_shadow_heuristic() -> None:
    wl = np.asarray(WAVELENGTHS_NM)
    veg = _vegetation_spectrum(wl, 1.0)
    cube = np.stack([veg, np.full_like(wl, 0.5), np.full_like(wl, 0.01)], axis=1)[:, :, None]
    cloud, shadow = geo.cloud_shadow_mask(cube, wl)
    assert cloud[:, 0].tolist() == [False, True, False]
    assert shadow[:, 0].tolist() == [False, False, True]
    assert geo.nodata_mask(np.array([[[NODATA, 0.1]]])).tolist() == [[True, False]]


def test_clip_to_aoi(tmp_path: Path, synthetic_cog: Path) -> None:
    left, top = ORIGIN
    aoi = _aoi(tmp_path / "a.geojson", (left + 30, top - 300, left + 300, top - 30))
    data, transform, crs = geo.clip_to_aoi(synthetic_cog, aoi)
    assert data.shape[0] == N_BANDS and 9 <= data.shape[1] <= 11
    assert crs.to_epsg() == 32643 and transform.a == 30.0
