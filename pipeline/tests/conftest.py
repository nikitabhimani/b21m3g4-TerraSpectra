from __future__ import annotations

from pathlib import Path

import pytest

from terraspectra_contracts.fixtures import write_synthetic_cube


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep tests independent of the developer's .env / credentials."""
    from terraspectra_pipeline.config import get_settings

    for var in ("USGS_M2M_USERNAME", "USGS_M2M_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TS_PIPELINE_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()


@pytest.fixture
def synthetic_cog(tmp_path: Path) -> Path:
    return write_synthetic_cube(tmp_path / "synthetic.tif", 64, 64, seed=1)
