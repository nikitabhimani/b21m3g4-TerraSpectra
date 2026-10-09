"""Inference jobs and their artifacts (zones, risk raster)."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from shapely.errors import GEOSException
from shapely.geometry import shape
from sqlalchemy import select
from sqlalchemy.orm import Session

from terraspectra_api.db import get_session
from terraspectra_api.deps import get_queue, get_redis, get_settings_dep
from terraspectra_api.errors import ApiError, not_found
from terraspectra_api.models import Job, Scene, as_utc, new_id, utcnow
from terraspectra_api.schemas import ERROR_RESPONSES
from terraspectra_api.services.events import publish_job_event, stream_job_events
from terraspectra_api.services.fields import load_fields
from terraspectra_api.services.jobs import mark_job_cancelled, unmark_job_cancelled
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


@router.get(
    "/jobs/{job_id}/stream",
    operation_id="streamJobEvents",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "Server-Sent Events stream for live job progress and status",
        },
        404: ERROR_RESPONSES[404],
    },
)
async def stream_job_events_endpoint(
    job_id: str,
    request: Request,
    session: Session = Depends(get_session),
    redis: object | None = Depends(get_redis),
) -> StreamingResponse:
    """Stream live Server-Sent Events (progress, step, and status transitions) for a job."""
    get_job_or_404(session, job_id)
    db = request.app.state.db

    return StreamingResponse(
        stream_job_events(job_id=job_id, db=db, redis_client=redis),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/jobs/{job_id}/cancel",
    operation_id="cancelJob",
    response_model=JobStatus,
    responses={
        404: ERROR_RESPONSES[404],
        409: ERROR_RESPONSES[409],
    },
)
def cancel_job(
    job_id: str,
    session: Session = Depends(get_session),
    redis: object | None = Depends(get_redis),
) -> JobStatus:
    """Cancel a queued or running inference job."""
    job = get_job_or_404(session, job_id)
    if job.status == JobState.SUCCEEDED:
        raise ApiError(409, "cannot cancel succeeded job", "job_already_finished")
    if job.status == JobState.FAILED:
        raise ApiError(409, "cannot cancel failed job", "job_already_finished")
    if job.status == JobState.CANCELLED:
        return to_status(job)

    job.status = JobState.CANCELLED
    job.error = "Cancelled by user"
    job.finished_at = utcnow()
    session.commit()

    mark_job_cancelled(job.id, redis)
    publish_job_event(
        job.id,
        "cancelled",
        {
            "job_id": job.id,
            "status": "cancelled",
            "progress": job.progress,
            "step": "Job cancelled by user",
            "timestamp": utcnow().isoformat(),
        },
        redis,
    )
    return to_status(job)


@router.post(
    "/jobs/{job_id}/retry",
    operation_id="retryJob",
    status_code=202,
    response_model=JobStatus,
    responses={
        404: ERROR_RESPONSES[404],
        409: ERROR_RESPONSES[409],
    },
)
def retry_job(
    job_id: str,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    queue: JobQueue = Depends(get_queue),
    redis: object | None = Depends(get_redis),
) -> JobStatus:
    """Retry a failed or cancelled inference job."""
    job = get_job_or_404(session, job_id)
    if job.status not in (JobState.FAILED, JobState.CANCELLED):
        raise ApiError(409, f"cannot retry job in {job.status} state", "job_cannot_retry")

    unmark_job_cancelled(job.id, redis)
    job.status = JobState.QUEUED
    job.progress = 0.0
    job.error = None
    job.summary = None
    job.started_at = None
    job.finished_at = None
    job.risk_path = None
    job.zones_path = None
    session.commit()

    try:
        queue.enqueue(job.id, background)
    except Exception as exc:
        log.exception("retry enqueue failed", extra={"job_id": job.id})
        job.status, job.error = JobState.FAILED, f"enqueue failed: {exc}"
        session.commit()
        raise ApiError(503, "job queue unavailable", "queue_unavailable") from exc

    publish_job_event(
        job.id,
        "progress",
        {
            "job_id": job.id,
            "status": "queued",
            "progress": 0.0,
            "step": "Job re-queued for execution",
            "timestamp": utcnow().isoformat(),
        },
        redis,
    )
    return to_status(job)

