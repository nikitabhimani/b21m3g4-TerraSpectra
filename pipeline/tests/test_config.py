"""Tests for terraspectra_pipeline.config — pipeline settings and env var overrides."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from terraspectra_pipeline.config import PipelineSettings


def test_default_values(monkeypatch):
    """With no env vars set, defaults should match the documented values."""
    monkeypatch.delenv("TS_PIPELINE_WORKERS", raising=False)
    monkeypatch.delenv("TS_PIPELINE_BLOCK_SIZE", raising=False)
    monkeypatch.delenv("TS_PIPELINE_LOG_LEVEL", raising=False)
    monkeypatch.delenv("TS_PIPELINE_DATA_DIR", raising=False)

    settings = PipelineSettings(_env_file=None)

    assert settings.data_dir == Path("data")

def test_env_var_overrides_workers(monkeypatch):
    """TS_PIPELINE_WORKERS should override the default worker count."""
    monkeypatch.setenv("TS_PIPELINE_WORKERS", "4")

    settings = PipelineSettings(_env_file=None)

    assert settings.workers == 4


def test_env_var_overrides_block_size(monkeypatch):
    """TS_PIPELINE_BLOCK_SIZE should override the default block size."""
    monkeypatch.setenv("TS_PIPELINE_BLOCK_SIZE", "1024")

    settings = PipelineSettings(_env_file=None)

    assert settings.block_size == 1024


def test_env_var_overrides_log_level(monkeypatch):
    """TS_PIPELINE_LOG_LEVEL should override the default log level."""
    monkeypatch.setenv("TS_PIPELINE_LOG_LEVEL", "DEBUG")

    settings = PipelineSettings(_env_file=None)

    assert settings.log_level == "DEBUG"


def test_raw_dir_property():
    """raw_dir should be data_dir joined with 'raw'."""
    settings = PipelineSettings(_env_file=None, data_dir=Path("mydata"))

    assert settings.raw_dir == Path("mydata/raw")


def test_processed_dir_property():
    """processed_dir should be data_dir joined with 'processed'."""
    settings = PipelineSettings(_env_file=None, data_dir=Path("mydata"))

    assert settings.processed_dir == Path("mydata/processed")


def test_usgs_username_alias(monkeypatch):
    """USGS_M2M_USERNAME (no TS_PIPELINE_ prefix) should also populate usgs_username."""
    monkeypatch.delenv("TS_PIPELINE_USGS_USERNAME", raising=False)
    monkeypatch.setenv("USGS_M2M_USERNAME", "someuser")

    settings = PipelineSettings(_env_file=None)

    assert settings.usgs_username == "someuser"


def test_block_size_minimum_is_enforced():
    """block_size below 16 should raise a validation error."""
    with pytest.raises(ValidationError):
        PipelineSettings(_env_file=None, block_size=8)


def test_workers_minimum_is_enforced():
    """workers below 1 should raise a validation error."""
    with pytest.raises(ValidationError):
        PipelineSettings(_env_file=None, workers=0)