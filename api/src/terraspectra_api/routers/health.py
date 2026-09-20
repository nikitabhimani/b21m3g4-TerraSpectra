"""``GET /v1/health`` (public)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from terraspectra_api import __version__
from terraspectra_api.core.engine import InferenceEngine
from terraspectra_api.deps import get_engine
from terraspectra_api.schemas import Health

router = APIRouter(tags=["health"])


@router.get("/health", operation_id="getHealth", response_model=Health)
def get_health(engine: InferenceEngine | None = Depends(get_engine)) -> Health:
    """Liveness plus model status; ``degraded`` when serving the stub or no model."""
    loaded = bool(engine and engine.model_loaded)
    # TODO(Day 11): add DB/Redis readiness probes (separate /v1/ready endpoint).
    return Health(
        status="ok" if loaded else "degraded",
        version=__version__,
        device=engine.device_name if engine else "cpu",
        model_loaded=loaded,
    )
