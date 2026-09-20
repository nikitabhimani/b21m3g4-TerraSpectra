from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from terraspectra_api.main import create_app
from terraspectra_api.security import RateLimiter
from terraspectra_api.settings import Settings


def test_health_is_public(client: TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"
    assert resp.headers["X-Request-ID"]


def test_request_id_is_propagated(client: TestClient) -> None:
    resp = client.get("/v1/health", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/v1/fields"),
        ("get", "/v1/scenes"),
        ("get", "/v1/scenes/scn_x"),
        ("get", "/v1/jobs"),
        ("get", "/v1/jobs/job_x"),
        ("get", "/v1/jobs/job_x/zones"),
        ("get", "/v1/jobs/job_x/risk.tif"),
    ],
)
def test_protected_endpoints_require_key(client: TestClient, method: str, path: str) -> None:
    resp = client.request(method, path)
    assert resp.status_code == 401
    assert resp.json()["code"] == "unauthorized"
    assert client.request(method, path, headers={"X-API-Key": "wrong"}).status_code == 401


def test_post_endpoints_require_key(client: TestClient) -> None:
    assert client.post("/v1/jobs", json={"scene_id": "x"}).status_code == 401
    assert client.post("/v1/scenes", data={"uri": "file:///x"}).status_code == 401


def test_error_model(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.get("/v1/scenes/does-not-exist", headers=auth)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "scene 'does-not-exist' not found", "code": "scene_not_found"}
    resp = client.post("/v1/jobs", json={}, headers=auth)
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_degraded_without_model(settings: Settings, tmp_path: object) -> None:
    settings.model_path = settings.storage_dir / "missing.pt"
    with TestClient(create_app(settings)) as c:
        body = c.get("/v1/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False


def test_fields(client: TestClient, auth: dict[str, str]) -> None:
    resp = client.get("/v1/fields", headers=auth)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/geo+json")
    assert resp.json()["features"][0]["properties"]["field_id"] == "fld_demo_farm"


def test_metrics_exposed(client: TestClient) -> None:
    client.get("/v1/health")
    assert client.get("/metrics").status_code == 200


def test_rate_limiter() -> None:
    limiter = RateLimiter("2/minute")
    assert limiter.hit("k", now=0) is None
    assert limiter.hit("k", now=1) is None
    retry = limiter.hit("k", now=2)
    assert retry is not None and retry > 0
    assert limiter.hit("k", now=61) is None


def test_rate_limit_returns_429(settings: Settings, auth: dict[str, str]) -> None:
    settings.rate_limit = "1/minute"
    with TestClient(create_app(settings)) as c:
        assert c.get("/v1/jobs", headers=auth).status_code == 200
        resp = c.get("/v1/jobs", headers=auth)
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_cors(settings: Settings) -> None:
    settings.cors_origins = ["http://localhost:5173"]
    with TestClient(create_app(settings)) as c:
        resp = c.options(
            "/v1/jobs",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
    assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_settings_parse_csv() -> None:
    s = Settings(_env_file=None, api_keys="a, b,,c", cors_origins="http://x,http://y")  # type: ignore[call-arg]
    assert s.api_keys == ["a", "b", "c"]
    assert s.cors_origins == ["http://x", "http://y"]


def test_production_rejects_default_key() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, env="production", api_keys="dev-key-change-me")  # type: ignore[call-arg]
