"""``GET /v1/jobs/{job_id}/tiles/{z}/{x}/{y}.png`` (public)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi import Path as PathParam
from fastapi.responses import Response
from sqlalchemy.orm import Session

from terraspectra_api.core.tiles import TileRenderer
from terraspectra_api.db import get_session
from terraspectra_api.deps import get_tiles
from terraspectra_api.errors import ApiError
from terraspectra_api.routers.jobs import get_job_or_404, require_succeeded
from terraspectra_api.schemas import ERROR_RESPONSES

router = APIRouter(tags=["tiles"])
CACHE_CONTROL = "public, max-age=86400, immutable"


@router.get(
    "/jobs/{job_id}/tiles/{z}/{x}/{y}.png",
    operation_id="getJobTile",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "256x256 RGBA heatmap tile"},
        404: ERROR_RESPONSES[404],
    },
)
def get_job_tile(
    job_id: str,
    z: int = PathParam(..., ge=0, le=22),
    x: int = PathParam(..., ge=0),
    y: int = PathParam(..., ge=0),
    session: Session = Depends(get_session),
    tiles: TileRenderer = Depends(get_tiles),
) -> Response:
    """Heatmap of risk score (1 - p_healthy), transparent outside data."""
    if x >= 2**z or y >= 2**z:
        raise ApiError(404, "tile index out of range for zoom", "tile_not_found")
    job = get_job_or_404(session, job_id)
    try:
        path = require_succeeded(job, job.risk_path)
    except ApiError as exc:  # contract only allows 404 here
        raise ApiError(404, exc.detail, exc.code) from exc
    png = tiles.render(path, z, x, y)
    return Response(png, media_type="image/png", headers={"Cache-Control": CACHE_CONTROL})
