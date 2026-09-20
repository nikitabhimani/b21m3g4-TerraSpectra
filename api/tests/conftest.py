"""Shared fixtures: an isolated app (sqlite + storage in tmp, inline queue, test env)."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from terraspectra_api.main import create_app
from terraspectra_api.settings import Settings
from terraspectra_contracts.fixtures import export_stub_model, write_synthetic_cube

API_KEY = "test-key"
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = REPO_ROOT / "contracts"


@pytest.fixture(scope="session")
def stub_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return export_stub_model(tmp_path_factory.mktemp("model") / "model.pt")


@pytest.fixture(scope="session")
def cube_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_synthetic_cube(tmp_path_factory.mktemp("cube") / "cube.tif", 128, 128)


@pytest.fixture
def settings(tmp_path: Path, stub_model: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        env="test",
        api_keys=[API_KEY],
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        storage_dir=tmp_path / "storage",
        model_path=stub_model,
        device="cpu",
        batch_size=8,
        window_overlap=16,
        queue_mode="inline",
        rate_limit="1000/minute",
        zone_min_acres=0.0,
        zone_min_prob=0.0,  # the random stub is never confident; still exercise zones
        prefetch_batches=1,
        log_level="WARNING",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": API_KEY}
