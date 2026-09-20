"""Runtime configuration (pydantic-settings, env prefix ``TS_``)."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from terraspectra_contracts import WINDOW_SIZE


def _default_fields_path() -> Path:
    return Path(str(resources.files("terraspectra_api").joinpath("data/fields.geojson")))


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    """All service settings. Every field can be set via ``TS_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(env_prefix="TS_", env_file=".env", extra="ignore")

    env: Literal["development", "test", "staging", "production"] = "development"
    api_keys: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["dev-key-change-me"])
    database_url: str = "sqlite:///./terraspectra.db"
    redis_url: str = "redis://localhost:6379/0"
    storage_dir: Path = Path("./data")
    model_path: Path = Path("./models/model.pt")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = Field(default=32, ge=1, le=4096)
    window_overlap: int = Field(default=16, ge=0, lt=WINDOW_SIZE)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    log_level: str = "INFO"

    # Service behaviour
    queue_mode: Literal["rq", "inline"] = "rq"
    queue_name: str = "terraspectra"
    job_timeout_s: int = Field(default=3600, ge=1)
    fields_path: Path = Field(default_factory=_default_fields_path)
    max_upload_mb: int = Field(default=4096, ge=1)
    rate_limit: str = "120/minute"
    uri_allowed_roots: Annotated[list[Path], NoDecode] = Field(default_factory=list)
    host: str = "127.0.0.1"
    port: int = 8000

    # Zone extraction / tiles
    zone_min_prob: float = Field(default=0.4, ge=0.0, le=1.0)
    zone_min_acres: float = Field(default=1.0, ge=0.0)
    zone_simplify_px: float = Field(default=0.5, ge=0.0)
    tile_cache_size: int = Field(default=1024, ge=0)
    prefetch_batches: int = Field(default=2, ge=0)

    @field_validator("api_keys", "cors_origins", "uri_allowed_roots", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> Any:
        return _split_csv(value)

    @field_validator("log_level")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("rate_limit")
    @classmethod
    def _check_rate(cls, value: str) -> str:
        parse_rate_limit(value)
        return value

    @model_validator(mode="after")
    def _check_production(self) -> Settings:
        if self.env == "production" and (not self.api_keys or "dev-key-change-me" in self.api_keys):
            raise ValueError("TS_API_KEYS must be set to non-default keys in production")
        return self

    @property
    def is_dev(self) -> bool:
        """True for development/test environments (auto-create tables, stub model)."""
        return self.env in ("development", "test")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def allowed_roots(self) -> list[Path]:
        """Directories that ``file://`` scene URIs may point into."""
        return [p.resolve() for p in (self.uri_allowed_roots or [self.storage_dir])]


_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_rate_limit(value: str) -> tuple[int, int]:
    """Parse ``"<n>/<second|minute|hour|day>"`` into ``(limit, period_seconds)``; ``0`` disables."""
    try:
        count, period = value.strip().split("/")
        return int(count), _PERIODS[period.strip().lower().rstrip("s")]
    except (ValueError, KeyError) as exc:
        raise ValueError(f"invalid rate limit {value!r}, expected e.g. '120/minute'") from exc


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings loaded from the environment."""
    return Settings()
