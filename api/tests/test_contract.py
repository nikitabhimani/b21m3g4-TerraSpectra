"""Contract C3: the app exposes exactly the paths/methods/operationIds in openapi.yaml."""

from __future__ import annotations

from typing import Any

import yaml
from fastapi.testclient import TestClient

from tests.conftest import CONTRACTS

METHODS = {"get", "post", "put", "patch", "delete"}


def _operations(spec: dict[str, Any]) -> dict[tuple[str, str], str | None]:
    return {
        (path, method): op.get("operationId")
        for path, ops in spec["paths"].items()
        for method, op in ops.items()
        if method in METHODS
    }


def test_routes_match_contract(client: TestClient) -> None:
    contract = yaml.safe_load((CONTRACTS / "openapi.yaml").read_text())
    app_spec = client.get("/openapi.json").json()
    expected = _operations(contract)
    actual = _operations(app_spec)
    assert set(actual) == set(expected)
    assert actual == expected


def test_info_matches_contract(client: TestClient) -> None:
    contract = yaml.safe_load((CONTRACTS / "openapi.yaml").read_text())
    info = client.get("/openapi.json").json()["info"]
    assert info["title"] == contract["info"]["title"]
    assert info["version"] == contract["info"]["version"]


def test_public_operations_match_contract(client: TestClient) -> None:
    contract = yaml.safe_load((CONTRACTS / "openapi.yaml").read_text())
    public = {
        (p, m)
        for p, ops in contract["paths"].items()
        for m, op in ops.items()
        if op.get("security") == []
    }
    for path, method in _operations(contract):
        url = path.format(job_id="job_x", scene_id="scn_x", z=0, x=0, y=0)
        resp = client.request(method, url, json={} if method == "post" else None)
        if (path, method) in public:
            assert resp.status_code != 401, url
        else:
            assert resp.status_code == 401, url


def test_fixture_job_matches_schema() -> None:
    from terraspectra_contracts import JobStatus

    JobStatus.model_validate_json((CONTRACTS / "fixtures" / "sample_job.json").read_text())
