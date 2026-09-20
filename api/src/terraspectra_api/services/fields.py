"""Farm field boundaries (``GET /v1/fields``) loaded from ``TS_FIELDS_PATH``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from terraspectra_api.schemas import FieldCollection


@lru_cache(maxsize=8)
def _load(path: str, mtime_ns: int) -> FieldCollection:
    return FieldCollection.model_validate_json(Path(path).read_bytes())


def load_fields(path: Path) -> FieldCollection:
    """Parse and validate the field GeoJSON (re-read when the file changes).

    TODO(Day 11): store fields in the database with CRUD endpoints.
    """
    return _load(str(path), path.stat().st_mtime_ns)
