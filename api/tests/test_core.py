"""Unit tests for chunker, stitcher, zones, engine and tiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terraspectra_api.core.chunker import RasterChunker, iter_windows, window_offsets
from terraspectra_api.core.engine import InferenceEngine, ModelError
from terraspectra_api.core.stitcher import Stitcher, feather_weights, write_risk_cog
from terraspectra_api.core.tiles import colorize, transparent_tile
from terraspectra_api.core.zones import (
    ZoneConfig,
    extract_zones,
    pixel_area_acres,
    recommended_action,
)
from terraspectra_contracts import N_BANDS, N_CLASSES, WINDOW_SIZE

TRANSFORM = from_origin(500000.0, 3420000.0, 30.0, 30.0)
ACRES_PER_PIXEL = 900.0 / 4046.8564224


@pytest.mark.parametrize(("size", "overlap"), [(64, 16), (100, 16), (128, 0), (130, 32), (20, 8)])
def test_window_offsets_cover_axis(size: int, overlap: int) -> None:
    offs = window_offsets(size, WINDOW_SIZE, overlap)
    covered = np.zeros(size, dtype=bool)
    for o in offs:
        covered[o : o + WINDOW_SIZE] = True
    assert covered.all()
    assert offs[0] == 0
    assert all(o + WINDOW_SIZE <= max(size, WINDOW_SIZE) for o in offs)


@pytest.mark.parametrize(("h", "w", "overlap"), [(128, 128, 16), (100, 150, 16), (40, 70, 8)])
def test_chunker_and_stitcher_cover_every_pixel(
    tmp_path: Path, h: int, w: int, overlap: int
) -> None:
    path = tmp_path / "cube.tif"
    rng = np.random.default_rng(0)
    data = rng.random((N_BANDS, h, w), dtype=np.float32)
    data[:, 0, 0] = -1.0
    profile = {"driver": "GTiff", "dtype": "float32", "count": N_BANDS, "height": h, "width": w,
               "crs": "EPSG:32643", "transform": TRANSFORM, "nodata": -1.0}  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    chunker = RasterChunker(path, overlap=overlap, batch_size=5, prefetch=2)
    stitcher = Stitcher(h, w, WINDOW_SIZE, overlap)
    seen = 0
    for batch in chunker.batches():
        assert batch.data.shape[1:] == (N_BANDS, WINDOW_SIZE, WINDOW_SIZE)
        assert len(batch) <= 5
        assert (batch.data[~np.broadcast_to(batch.valid[:, None], batch.data.shape)] == 0).all()
        n = len(batch)
        stitcher.add(
            np.ones((n, N_CLASSES, WINDOW_SIZE, WINDOW_SIZE), np.float32),
            np.ones((n, 1, WINDOW_SIZE, WINDOW_SIZE), np.float32),
            batch.rows,
            batch.cols,
            batch.valid,
        )
        seen += n
    assert seen == len(chunker) == len(list(iter_windows(h, w, WINDOW_SIZE, overlap)))
    weight = stitcher.weight.copy()
    assert weight[0, 0] == 0  # nodata pixel
    weight[0, 0] = 1
    assert (weight > 0).all()


def test_chunker_aoi_mask_skips_windows(tmp_path: Path) -> None:
    path = tmp_path / "cube.tif"
    profile = {"driver": "GTiff", "dtype": "float32", "count": N_BANDS, "height": 128,
               "width": 128, "crs": "EPSG:32643", "transform": TRANSFORM}  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.full((N_BANDS, 128, 128), 0.3, np.float32))
    aoi = np.zeros((128, 128), dtype=bool)
    aoi[:10, :10] = True
    batches = list(RasterChunker(path, overlap=16, aoi_mask=aoi).batches())
    assert sum(len(b) for b in batches) == 1
    assert batches[0].valid.sum() == 100


def test_feather_weights_positive() -> None:
    w = feather_weights(64, 16)
    assert w.shape == (64, 64)
    assert (w > 0).all()
    assert w.max() == 1.0
    assert (feather_weights(64, 0) == 1).all()


def test_stitcher_reproduces_constants(tmp_path: Path) -> None:
    h, w, overlap = 100, 90, 16
    stitcher = Stitcher(h, w, WINDOW_SIZE, overlap)
    const_p = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    windows = list(iter_windows(h, w, WINDOW_SIZE, overlap))
    rows = np.array([r for r, _ in windows])
    cols = np.array([c for _, c in windows])
    n = len(windows)
    probs = np.broadcast_to(const_p[None, :, None, None], (n, 4, 64, 64)).copy()
    onset = np.full((n, 1, 64, 64), 12.5, np.float32)
    stitcher.add(probs, onset, rows, cols)
    mask = np.ones((h, w), bool)
    mask[5, 5] = False
    p, o, valid = stitcher.finalize(mask)
    np.testing.assert_allclose(p[:, valid], const_p[:, None] * np.ones((1, valid.sum())), atol=1e-5)
    np.testing.assert_allclose(o[valid], 12.5, atol=1e-4)
    assert (p[:, 5, 5] == -1).all() and o[5, 5] == -1

    out = write_risk_cog(tmp_path / "risk.tif", p, o, "EPSG:32643", TRANSFORM)
    with rasterio.open(out) as src:
        assert src.count == 5 and src.shape == (h, w)
        assert src.descriptions == (
            "p_healthy", "p_early_stress", "p_high_blight_risk", "p_visible_disease",
            "days_to_onset",
        )  # fmt: skip
        assert src.nodata == -1
        assert src.transform == TRANSFORM


def _probs_for(class_map: np.ndarray) -> np.ndarray:
    probs = np.full((4, *class_map.shape), 0.05, np.float32)
    for c in range(4):
        probs[c][class_map == c] = 0.85
    return probs


def test_zones_from_hand_made_class_map(tmp_path: Path) -> None:
    h = w = 60
    class_map = np.zeros((h, w), np.int64)
    class_map[5:15, 5:15] = 2  # 100 px → ~22.2 acres
    class_map[30:50, 30:40] = 3  # 200 px → ~44.5 acres
    class_map[40:45, 5:10] = 1  # 25 px → ~5.6 acres
    class_map[0, 59] = 2  # single pixel, removed by opening
    probs = _probs_for(class_map)
    onset = np.full((h, w), 10.0, np.float32)
    valid = np.ones((h, w), bool)
    zones, summary = extract_zones(
        probs, onset, valid, TRANSFORM, "EPSG:32643", "job_t", "scn_t",
        ZoneConfig(min_prob=0.5, min_acres=1.0),
    )  # fmt: skip
    assert len(zones.features) == 3
    assert summary.zones_by_class == {
        "healthy": 0, "early_stress": 1, "high_blight_risk": 1, "visible_disease": 1,
    }  # fmt: skip
    by_class = {f.properties.risk_class: f.properties for f in zones.features}
    assert by_class[2].area_acres == pytest.approx(100 * ACRES_PER_PIXEL, rel=0.02)
    assert by_class[3].area_acres == pytest.approx(200 * ACRES_PER_PIXEL, rel=0.02)
    assert by_class[3].risk_score == pytest.approx(0.95, abs=1e-3)
    assert by_class[2].days_to_onset == pytest.approx(10.0)
    assert zones.features[0].properties.zone_id == "z-001"
    assert zones.features[0].properties.risk_class == 3
    lon, lat = zones.features[0].geometry.coordinates[0][0]
    assert 74 < lon < 76 and 30 < lat < 32  # UTM 43N near the fixture origin
    assert summary.acres_analyzed == pytest.approx(h * w * ACRES_PER_PIXEL, rel=1e-3)
    assert summary.acres_at_risk == pytest.approx(325 * ACRES_PER_PIXEL, rel=0.03)

    # min_acres filter drops the small early-stress zone.
    zones, _ = extract_zones(
        probs, onset, valid, TRANSFORM, "EPSG:32643", "j", "s", ZoneConfig(min_acres=10.0)
    )
    assert len(zones.features) == 2


def test_zones_with_cube_indicator(cube_path: Path) -> None:
    with rasterio.open(cube_path) as src:
        transform, crs, (h, w) = src.transform, src.crs, src.shape
    class_map = np.zeros((h, w), np.int64)
    class_map[64:92, 64:92] = 2  # the planted stressed patch
    zones, _ = extract_zones(
        _probs_for(class_map), np.full((h, w), 5.0, np.float32), np.ones((h, w), bool),
        transform, crs, "j", "s", ZoneConfig(min_acres=0.0), cube_path=cube_path,
    )  # fmt: skip
    assert len(zones.features) == 1
    props = zones.features[0].properties
    assert props.dominant_indicator in {
        "red_edge_shift", "pri_drop", "chlorophyll_loss", "water_stress",
    }  # fmt: skip
    assert "48 hours" in props.recommended_action


def test_pixel_area_geographic() -> None:
    from pyproj import CRS

    t = from_origin(75.0, 31.0, 0.0003, 0.0003)  # ~30 m at 31°N
    acres = pixel_area_acres(t, CRS.from_epsg(4326), 10, 10)
    assert 0.15 < acres < 0.25


def test_recommended_action_rules() -> None:
    assert "curative" in recommended_action(3, 0, "pri_drop")
    assert "7 days" in recommended_action(2, 20, "red_edge_shift")
    assert "irrigation" in recommended_action(1, 20, "water_stress")


def test_engine_predict_shapes(stub_model: Path) -> None:
    engine = InferenceEngine(stub_model, device="cpu", batch_size=3)
    engine.load()
    assert engine.model_loaded and engine.device_name == "cpu"
    probs, onset = engine.predict(np.zeros((7, N_BANDS, 64, 64), np.float32))
    assert probs.shape == (7, 4, 64, 64) and onset.shape == (7, 1, 64, 64)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    with pytest.raises(ModelError):
        engine.predict(np.zeros((1, 3, 64, 64), np.float32))


def test_engine_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ModelError):
        InferenceEngine(tmp_path / "missing.pt", device="cpu").load()
    engine = InferenceEngine(tmp_path / "missing.pt", device="cpu", allow_stub=True,
                             stub_dir=tmp_path)  # fmt: skip
    engine.load()
    assert engine.ready and not engine.model_loaded and engine.using_stub


def test_colorize_and_transparent_tile() -> None:
    rgb, alpha = colorize(np.array([[0.0, 1.0]]), np.array([[True, False]]))
    assert tuple(rgb[:, 0, 0]) == (26, 152, 80)
    assert tuple(rgb[:, 0, 1]) == (215, 48, 39)
    assert alpha.tolist() == [[210, 0]]
    assert transparent_tile().startswith(b"\x89PNG")
