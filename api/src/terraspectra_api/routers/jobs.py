"""Inference jobs and their artifacts (zones, risk raster)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse, Response
from shapely.errors import GEOSException
from shapely.geometry import shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from terraspectra_api.db import get_session
from terraspectra_api.deps import get_queue, get_settings_dep
from terraspectra_api.errors import ApiError, not_found
from terraspectra_api.models import Job, Scene, as_utc, new_id
from terraspectra_api.schemas import ERROR_RESPONSES
from terraspectra_api.services.fields import load_fields
from terraspectra_api.services.queue import JobQueue
from terraspectra_api.settings import Settings
from terraspectra_contracts import JobCreate, JobStatus, JobSummary, ZoneFeatureCollection
from terraspectra_contracts.schemas import JobState

log = logging.getLogger(__name__)
router = APIRouter(tags=["jobs"])
GEOJSON = "application/geo+json"


def to_status(job: Job) -> JobStatus:
    return JobStatus(
        job_id=job.id,
        scene_id=job.scene_id,
        field_id=job.field_id,
        status=JobState(job.status),
        progress=job.progress,
        created_at=as_utc(job.created_at),
        updated_at=as_utc(job.updated_at),
        error=job.error,
        summary=JobSummary.model_validate(job.summary) if job.summary else None,
    )


def get_job_or_404(session: Session, job_id: str) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise not_found("job", job_id)
    return job


def require_succeeded(job: Job, artifact: str | None) -> Path:
    """409 until the job has succeeded and produced ``artifact``."""
    if job.status != JobState.SUCCEEDED:
        raise ApiError(409, f"job is {job.status}, results not available", "job_not_ready")
    if not artifact or not Path(artifact).is_file():
        raise ApiError(404, "job artifact missing", "artifact_not_found")
    return Path(artifact)


def _validate_aoi(aoi: dict[str, object]) -> None:
    if aoi.get("type") not in ("Polygon", "MultiPolygon"):
        raise ApiError(422, "aoi must be a GeoJSON Polygon or MultiPolygon", "invalid_aoi")
    try:
        geom = shape(aoi)
    except (GEOSException, ValueError, TypeError, KeyError, IndexError) as exc:
        raise ApiError(422, f"invalid aoi geometry: {exc}", "invalid_aoi") from exc
    if geom.is_empty or not geom.is_valid:
        raise ApiError(422, "aoi geometry is empty or invalid", "invalid_aoi")
    minx, miny, maxx, maxy = geom.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise ApiError(422, "aoi must be in EPSG:4326 lon/lat", "invalid_aoi")


@router.get("/jobs", operation_id="listJobs", response_model=list[JobStatus])
def list_jobs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> list[JobStatus]:
    """Jobs, newest first."""
    rows = session.scalars(
        select(Job).order_by(Job.created_at.desc(), Job.id.desc()).limit(limit).offset(offset)
    )
    return [to_status(j) for j in rows]


@router.post(
    "/jobs",
    operation_id="createJob",
    status_code=202,
    response_model=JobStatus,
    responses={404: ERROR_RESPONSES[404], 422: ERROR_RESPONSES[422]},
)
def create_job(
    body: JobCreate,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    queue: JobQueue = Depends(get_queue),
    settings: Settings = Depends(get_settings_dep),
) -> JobStatus:
    """Queue an inference job for a registered scene (optionally clipped to a field/AOI)."""
    if session.get(Scene, body.scene_id) is None:
        raise not_found("scene", body.scene_id)
    if body.field_id is not None and load_fields(settings.fields_path).get(body.field_id) is None:
        raise not_found("field", body.field_id)
    if body.aoi is not None:
        _validate_aoi(body.aoi)
    job = Job(
        id=new_id("job"),
        scene_id=body.scene_id,
        field_id=body.field_id,
        aoi=body.aoi,
        status=JobState.QUEUED,
        progress=0.0,
    )
    session.add(job)
    session.commit()  # the worker must see the row before it runs
    try:
        queue.enqueue(job.id, background)
    except Exception as exc:
        log.exception("enqueue failed", extra={"job_id": job.id})
        job.status, job.error = JobState.FAILED, f"enqueue failed: {exc}"
        session.commit()
        raise ApiError(503, "job queue unavailable", "queue_unavailable") from exc
    return to_status(job)


@router.get(
    "/jobs/{job_id}",
    operation_id="getJob",
    response_model=JobStatus,
    responses={404: ERROR_RESPONSES[404]},
)
def get_job(job_id: str, session: Session = Depends(get_session)) -> JobStatus:
    return to_status(get_job_or_404(session, job_id))


@router.get(
    "/jobs/{job_id}/zones",
    operation_id="getJobZones",
    response_model=ZoneFeatureCollection,
    responses={
        200: {"content": {GEOJSON: {}}},
        404: ERROR_RESPONSES[404],
        409: ERROR_RESPONSES[409],
    },
)
def get_job_zones(job_id: str, session: Session = Depends(get_session)) -> Response:
    """Risk zones (contract C4)."""
    job = get_job_or_404(session, job_id)
    path = require_succeeded(job, job.zones_path)
    return Response(path.read_bytes(), media_type=GEOJSON)


@router.get(
    "/jobs/{job_id}/risk.tif",
    operation_id="getJobRiskRaster",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/tiff": {}}, "description": "5-band risk COG"},
        404: ERROR_RESPONSES[404],
        409: ERROR_RESPONSES[409],
    },
)
def get_job_risk_raster(job_id: str, session: Session = Depends(get_session)) -> FileResponse:
    """Bands 1-4 class probabilities, band 5 days_to_onset (float32, nodata -1)."""
    job = get_job_or_404(session, job_id)
    path = require_succeeded(job, job.risk_path)
    return FileResponse(path, media_type="image/tiff", filename=f"{job_id}_risk.tif")
