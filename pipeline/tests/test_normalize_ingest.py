import io
import json
import tarfile
import zipfile
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pytest

from terraspectra_contracts.constants import NODATA
from terraspectra_pipeline.config import PipelineSettings
from terraspectra_pipeline.ingest import M2MError, USGSClient, register_scene, safe_extract
from terraspectra_pipeline.normalize import (
    BandStats,
    apply_scaling,
    compute_band_stats,
    compute_file_stats,
    sample_windows,
)


def test_band_stats_roundtrip_and_scaling(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    block = rng.uniform(0.1, 0.5, size=(3, 50, 50)).astype(np.float32)
    block[:, 0, 0] = NODATA
    stats = compute_band_stats([block])
    assert stats.n_pixels == 50 * 50 - 1
    assert all(0.1 < lo < 0.13 for lo in stats.low)
    loaded = BandStats.load(stats.save(tmp_path / "stats.json"))
    assert loaded == stats
    scaled = apply_scaling(block, loaded)
    assert scaled[:, 0, 0].tolist() == [NODATA] * 3
    valid = scaled[:, 1:, :]
    assert valid.min() == 0.0 and valid.max() == 1.0
    with pytest.raises(ValueError):
        apply_scaling(block[:2], loaded)
    with pytest.raises(ValueError):
        compute_band_stats([np.full((3, 2, 2), NODATA, np.float32)])


def test_sample_windows_within_bounds() -> None:
    for win in sample_windows(100, 40, size=64, n=10):
        assert win.row_off + win.height <= 100 and win.col_off + win.width <= 40


def test_file_stats(synthetic_cog: Path) -> None:
    stats = compute_file_stats(synthetic_cog, n_windows=3, window_size=16)
    assert len(stats.low) == 200 and stats.wavelengths_nm is not None
    assert all(h >= lo for lo, h in zip(stats.low, stats.high, strict=True))


def test_register_zip_and_tar(tmp_path: Path, synthetic_cog: Path) -> None:
    archive = tmp_path / "scene.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(synthetic_cog, "wrapper/cube/synthetic.tif")
    rec = register_scene(archive, tmp_path / "raw")
    assert rec.path.name == "cube" and (rec.path / "scene.json").exists()
    assert json.loads((rec.path / "scene.json").read_text())["scene_dir"] == rec.scene_dir

    tgz = tmp_path / "scene2.tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(synthetic_cog, "synthetic.tif")
    rec2 = register_scene(tgz, tmp_path / "raw", sensor="generic")
    assert rec2.path == (tmp_path / "raw" / "scene2").resolve() and rec2.sensor == "generic"

    single = register_scene(synthetic_cog, tmp_path / "raw")
    assert single.sensor == "generic"
    with pytest.raises(FileNotFoundError):
        register_scene(tmp_path / "missing.zip", tmp_path / "raw")


def test_safe_extract_blocks_traversal(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ValueError):
        safe_extract(evil, tmp_path / "out")
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"nope")
    with pytest.raises(ValueError):
        safe_extract(plain, tmp_path / "out")


def _usgs_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path.rsplit("/", 1)[-1]
    if path == "login-token":
        assert json.loads(request.content)["username"] == "alice"
        return httpx.Response(200, json={"data": "KEY", "errorCode": None})
    if request.url.host != "dl.example":  # pre-signed download URLs need no token
        assert request.headers.get("X-Auth-Token") == "KEY"
    if path == "scene-search":
        body = json.loads(request.content)
        corner = body["sceneFilter"]["spatialFilter"]["lowerLeft"]
        assert corner == {"latitude": 30, "longitude": 75}
        results = [{"entityId": "E1", "displayId": "D1", "temporalCoverage": {"startDate": "x"}}]
        return httpx.Response(200, json={"data": {"results": results}})
    if path == "download-options":
        return httpx.Response(
            200,
            json={
                "data": [
                    {"entityId": "E1", "id": "P1", "available": True},
                    {"entityId": "E1", "id": "P2", "available": False},
                ]
            },
        )
    if path == "download-request":
        assert json.loads(request.content)["downloads"] == [{"entityId": "E1", "productId": "P1"}]
        return httpx.Response(
            200, json={"data": {"availableDownloads": [{"url": "https://dl.example/files/E1.tgz"}]}}
        )
    if path == "E1.tgz":
        return httpx.Response(200, content=io.BytesIO(b"payload").getvalue())
    if path == "logout":
        return httpx.Response(200, json={"data": True})
    if path == "bad":
        return httpx.Response(200, json={"errorCode": "AUTH_INVALID", "errorMessage": "no"})
    return httpx.Response(404)


def test_usgs_client_flow(tmp_path: Path) -> None:
    settings = PipelineSettings(USGS_M2M_USERNAME="alice", USGS_M2M_TOKEN="secret")
    with USGSClient(settings, transport=httpx.MockTransport(_usgs_handler)) as client:
        hits = client.scene_search(
            (75, 30, 76, 31), date(2004, 1, 1), date(2004, 12, 31), max_cloud_cover=20
        )
        assert [h.entity_id for h in hits] == ["E1"]
        files = list(client.iter_download(["E1"], tmp_path / "dl"))
        assert files[0].read_bytes() == b"payload"
        with pytest.raises(M2MError):
            client._post("bad", {})


def test_usgs_client_requires_credentials() -> None:
    client = USGSClient(PipelineSettings(), transport=httpx.MockTransport(_usgs_handler))
    with pytest.raises(M2MError):
        client.login()
    client.close()
