"""Pipeline settings (env prefix ``TS_PIPELINE_``; USGS credentials use their own names)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """Runtime configuration for the pipeline CLI and library."""

    model_config = SettingsConfigDict(
        env_prefix="TS_PIPELINE_",
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    data_dir: Path = Path("data")
    workers: int = Field(default=1, ge=1)
    block_size: int = Field(default=512, ge=16, description="Processing window size (px)")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    usgs_api_url: str = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    usgs_username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("USGS_M2M_USERNAME", "TS_PIPELINE_USGS_USERNAME"),
    )
    usgs_token: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("USGS_M2M_TOKEN", "TS_PIPELINE_USGS_TOKEN"),
    )
    http_timeout_s: float = 60.0

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> PipelineSettings:
    """Return the cached process-wide settings."""
    return PipelineSettings()
