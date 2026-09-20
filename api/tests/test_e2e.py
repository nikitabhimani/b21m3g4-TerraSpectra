"""End-to-end: upload → job → zones / risk.tif / tiles, all in-process."""

from __future__ import annotations

import io
import json
from pathlib import Path

import jsonschema
import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from terraspectra_contracts import N_BANDS
from tests.conftest import CONTRACTS

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_rgba(data: bytes) -> np.ndarray:
    with rasterio.MemoryFile(data) as mem, mem.open() as src:
        assert (src.width, src.height, src.count) == (256, 256, 4)
        return src.read()


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    lat_r = np.radians(lat)
    y = int((1.0 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / np.pi) / 2.0 * n)
    return x, y


def _upload(client: TestClient, auth: dict[str, str], path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        resp = client.post(
            "/v1/scenes",
            files={"file": (path.name, fh, "image/tiff")},
            data={"name": "synthetic"},
            headers=auth,
        )
    assert resp.status_code == 201, resp.text
    return dict(resp.json())


def test_full_job_lifecycle(client: TestClient, auth: dict[str, str], cube_path: Path) -> None:
    scene = _upload(client, auth, cube_path)
    assert scene["bands"] == N_BANDS
    assert (scene["width"], scene["height"]) == (128, 128)
    assert scene["crs"] == "EPSG:32643"
    bounds = scene["bounds"]
    assert isinstance(bounds, list) and len(bounds) == 4
    assert client.get(f"/v1/scenes/{scene['scene_id']}", headers=auth).status_code == 200
    assert len(client.get("/v1/scenes", headers=auth).json()) == 1

    resp = client.post("/v1/jobs", json={"scene_id": scene["scene_id"]}, headers=auth)
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "queued"
    job_id = resp.json()["job_id"]

    job = client.get(f"/v1/jobs/{job_id}", headers=auth).json()
    assert job["status"] == "succeeded", job
    assert job["progress"] == 1.0
    summary = job["summary"]
    assert summary["acres_analyzed"] > 0
    assert set(summary["zones_by_class"]) == {
        "healthy",
        "early_stress",
        "high_blight_risk",
        "visible_disease",
    }
    assert client.get("/v1/jobs", headers=auth).json()[0]["job_id"] == job_id

    zones_resp = client.get(f"/v1/jobs/{job_id}/zones", headers=auth)
    assert zones_resp.status_code == 200
    assert zones_resp.headers["content-type"].startswith("application/geo+json")
    zones = zones_resp.json()
    schema = json.loads((CONTRACTS / "zones.schema.json").read_text())
    jsonschema.validate(zones, schema)
    assert zones["job_id"] == job_id
    assert len(zones["features"]) == sum(summary["zones_by_class"].values())
    assert zones["features"], "stub model should yield at least one zone"

    risk = client.get(f"/v1/jobs/{job_id}/risk.tif", headers=auth)
    assert risk.status_code == 200
    assert risk.headers["content-type"] == "image/tiff"
    with rasterio.MemoryFile(risk.content) as mem, mem.open() as src:
        assert src.count == 5
        assert (src.height, src.width) == (128, 128)
        assert src.dtypes[0] == "float32"
        assert src.nodata == -1
        assert src.crs.to_string() == "EPSG:32643"
        data = src.read()
    probs = data[:4, 10:, 10:]
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-4)
    assert (data[:, :2, :2] == -1).all(), "nodata corner must stay nodata"
    assert ((data[4, 10:, 10:] >= 0) & (data[4, 10:, 10:] <= 30)).all()

    lon = (bounds[0] + bounds[2]) / 2  # type: ignore[index]
    lat = (bounds[1] + bounds[3]) / 2  # type: ignore[index]
    x, y = _lonlat_to_tile(lon, lat, 14)
    tile = client.get(f"/v1/jobs/{job_id}/tiles/14/{x}/{y}.png")
    assert tile.status_code == 200
    assert tile.content.startswith(PNG_MAGIC)
    assert "max-age" in tile.headers["cache-control"]
    assert _png_rgba(tile.content)[3].max() > 0, "tile inside bounds must have data"

    far = client.get(f"/v1/jobs/{job_id}/tiles/14/0/0.png")
    assert far.status_code == 200
    assert _png_rgba(far.content)[3].max() == 0, "far-away tile must be transparent"

    assert client.get(f"/v1/jobs/{job_id}/tiles/1/5/0.png").status_code == 404
    assert client.get("/v1/jobs/job_nope/tiles/1/0/0.png").status_code == 404


def test_job_with_field_and_aoi(client: TestClient, auth: dict[str, str], cube_path: Path) -> None:
    scene = _upload(client, auth, cube_path)
    b = scene["bounds"]
    assert isinstance(b, list)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    ring = [[b[0], b[1]], [cx, b[1]], [cx, cy], [b[0], cy], [b[0], b[1]]]
    aoi = {"type": "Polygon", "coordinates": [ring]}
    job = client.post(
        "/v1/jobs", json={"scene_id": scene["scene_id"], "aoi": aoi}, headers=auth
    ).json()
    job = client.get(f"/v1/jobs/{job['job_id']}", headers=auth).json()
    assert job["status"] == "succeeded", job
    risk = client.get(f"/v1/jobs/{job['job_id']}/risk.tif", headers=auth)
    with rasterio.MemoryFile(risk.content) as mem, mem.open() as src:
        band = src.read(1)
    valid = band >= 0
    assert 0.1 < valid.mean() < 0.5  # roughly the lower-left quarter

    # The demo field lies elsewhere (Ludhiana) → the job fails with a clear error.
    job = client.post(
        "/v1/jobs",
        json={"scene_id": scene["scene_id"], "field_id": "fld_demo_farm"},
        headers=auth,
    ).json()
    job = client.get(f"/v1/jobs/{job['job_id']}", headers=auth).json()
    assert job["status"] == "failed"
    assert "AOI" in job["error"]
    assert client.get(f"/v1/jobs/{job['job_id']}/zones", headers=auth).status_code == 409
    assert client.get(f"/v1/jobs/{job['job_id']}/risk.tif", headers=auth).status_code == 409


def test_job_errors(client: TestClient, auth: dict[str, str], cube_path: Path) -> None:
    resp = client.post("/v1/jobs", json={"scene_id": "scn_missing"}, headers=auth)
    assert resp.status_code == 404
    scene = _upload(client, auth, cube_path)
    resp = client.post(
        "/v1/jobs", json={"scene_id": scene["scene_id"], "field_id": "nope"}, headers=auth
    )
    assert resp.status_code == 404
    resp = client.post(
        "/v1/jobs",
        json={"scene_id": scene["scene_id"], "aoi": {"type": "Point", "coordinates": [0, 0]}},
        headers=auth,
    )
    assert resp.status_code == 422


def test_register_scene_by_uri(
    client: TestClient, auth: dict[str, str], cube_path: Path, tmp_path: Path
) -> None:
    storage = tmp_path / "storage" / "incoming"
    storage.mkdir(parents=True)
    target = storage / "cube.tif"
    target.write_bytes(cube_path.read_bytes())
    resp = client.post("/v1/scenes", data={"uri": target.as_uri()}, headers=auth)
    assert resp.status_code == 201, resp.text
    assert resp.json()["uri"] == target.as_uri()
    # Outside the allowed roots → rejected.
    resp = client.post("/v1/scenes", data={"uri": cube_path.as_uri()}, headers=auth)
    assert resp.status_code == 422
    resp = client.post("/v1/scenes", data={"uri": "s3://bucket/cube.tif"}, headers=auth)
    assert resp.status_code == 422
    assert client.post("/v1/scenes", data={}, headers=auth).status_code == 422


def test_upload_rejects_three_band_file(
    client: TestClient, auth: dict[str, str], tmp_path: Path
) -> None:
    path = tmp_path / "rgb.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 3,
        "height": 16,
        "width": 16,
        "crs": "EPSG:32643",
        "transform": from_origin(500000, 3420000, 30, 30),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.zeros((3, 16, 16), dtype=np.float32))
    resp = client.post(
        "/v1/scenes", files={"file": ("rgb.tif", path.read_bytes(), "image/tiff")}, headers=auth
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_cube"
    assert "200 bands" in resp.json()["detail"]
    assert not list((tmp_path / "storage" / "scenes").glob("*.tif")), "rejected file removed"


def test_upload_rejects_non_raster(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.post(
        "/v1/scenes",
        files={"file": ("x.tif", io.BytesIO(b"not a tiff"), "image/tiff")},
        headers=auth,
    )
    assert resp.status_code == 422


def test_upload_too_large(client: TestClient, auth: dict[str, str]) -> None:
    client.app.state.settings.max_upload_mb = 1  # type: ignore[attr-defined]
    resp = client.post(
        "/v1/scenes",
        files={"file": ("big.tif", io.BytesIO(b"0" * (2 * 1024 * 1024)), "image/tiff")},
        headers=auth,
    )
    assert resp.status_code == 413
