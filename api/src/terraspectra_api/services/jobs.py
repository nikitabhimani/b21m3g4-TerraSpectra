"""Job execution: scene → chunk → infer → stitch → COG → zones → summary."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from shapely.geometry import shape

from terraspectra_api.core.chunker import RasterChunker
from terraspectra_api.core.engine import InferenceEngine
from terraspectra_api.core.stitcher import Stitcher, write_risk_cog
from terraspectra_api.core.zones import ZoneConfig, extract_zones
from terraspectra_api.db import Database
from terraspectra_api.models import Job, utcnow
from terraspectra_api.services.fields import load_fields
from terraspectra_api.settings import Settings, get_settings
from terraspectra_api.storage import Storage, build_storage
from terraspectra_contracts import WINDOW_SIZE
from terraspectra_contracts.schemas import JobState

log = logging.getLogger(__name__)


@dataclass
class JobContext:
    """Everything a job needs; built once per API/worker process."""

    settings: Settings
    db: Database
    storage: Storage
    engine: InferenceEngine
    extras: dict[str, Any] = field(default_factory=dict)


_context: JobContext | None = None


def build_engine(settings: Settings, storage: Storage) -> InferenceEngine:
    """Engine configured from settings (stub fallback only in development/test)."""
    return InferenceEngine(
        model_path=settings.model_path,
        device=settings.device,
        batch_size=settings.batch_size,
        allow_stub=settings.is_dev,
        stub_dir=storage.cache_dir(),
    )


def build_context(settings: Settings | None = None, load_engine: bool = True) -> JobContext:
    """Create DB/storage/engine from settings (used by the RQ worker)."""
    settings = settings or get_settings()
    storage = build_storage(settings.storage_dir, settings.allowed_roots)
    db = Database(settings.database_url)
    if settings.is_dev:
        db.create_all()
    engine = build_engine(settings, storage)
    if load_engine:
        engine.load()
    return JobContext(settings=settings, db=db, storage=storage, engine=engine)


def set_context(ctx: JobContext | None) -> None:
    global _context
    _context = ctx


def get_context() -> JobContext:
    """Process-wide context, created lazily (e.g. in an RQ work-horse)."""
    global _context
    if _context is None:
        _context = build_context()
    return _context


class ProgressReporter:
    """Throttled progress writes (at most every ``min_interval`` s or ``min_step``)."""

    def __init__(
        self, db: Database, job_id: str, min_interval: float = 1.0, min_step: float = 0.05
    ) -> None:
        self.db, self.job_id = db, job_id
        self.min_interval, self.min_step = min_interval, min_step
        self._last_t = 0.0
        self._last_p = 0.0

    def __call__(self, progress: float, force: bool = False) -> None:
        progress = float(min(max(progress, 0.0), 1.0))
        now = time.monotonic()
        if not force and (
            now - self._last_t < self.min_interval and progress - self._last_p < self.min_step
        ):
            return
        self._last_t, self._last_p = now, progress
        _update(self.db, self.job_id, progress=progress)


def _update(db: Database, job_id: str, **values: Any) -> None:
    with db.session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise LookupError(f"job {job_id} disappeared")
        for key, value in values.items():
            setattr(job, key, value)
        job.updated_at = utcnow()


def _aoi_mask(geometry: dict[str, Any], src: rasterio.DatasetReader) -> np.ndarray:
    """Boolean ``[H,W]`` mask, True inside the AOI (given in EPSG:4326)."""
    geom = transform_geom("EPSG:4326", src.crs, geometry)
    if not shape(geom).intersects(shape(_bounds_geom(src))):
        raise ValueError("AOI does not intersect the scene")
    inside = ~geometry_mask([geom], out_shape=(src.height, src.width), transform=src.transform)
    if not inside.any():
        raise ValueError("AOI does not cover any scene pixel")
    return inside


def _bounds_geom(src: rasterio.DatasetReader) -> dict[str, Any]:
    left, bottom, right, top = src.bounds
    ring = [(left, bottom), (right, bottom), (right, top), (left, top), (left, bottom)]
    return {"type": "Polygon", "coordinates": [ring]}


def run_job(job_id: str, ctx: JobContext | None = None) -> None:
    """Execute one job end-to-end. Never raises: failures are recorded on the job."""
    ctx = ctx or get_context()
    started = time.perf_counter()
    try:
        with ctx.db.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                log.error("job not found", extra={"job_id": job_id})
                return
            if job.status != JobState.QUEUED:
                log.warning("job not queued; skipping", extra={"job_id": job_id})
                return
            job.status, job.progress = JobState.RUNNING, 0.01
            job.started_at = job.updated_at = utcnow()
            scene_id, scene_uri = job.scene_id, job.scene.uri
            aoi, field_id = job.aoi, job.field_id
        log.info("job started", extra={"job_id": job_id, "scene_id": scene_id})
        _execute(ctx, job_id, scene_id, scene_uri, aoi, field_id)
        log.info(
            "job succeeded",
            extra={"job_id": job_id, "seconds": round(time.perf_counter() - started, 2)},
        )
    except Exception as exc:
        log.exception("job failed", extra={"job_id": job_id})
        try:
            _update(
                ctx.db,
                job_id,
                status=JobState.FAILED,
                error=f"{type(exc).__name__}: {exc}"[:2000],
                finished_at=utcnow(),
            )
        except Exception:
            log.exception("could not record job failure", extra={"job_id": job_id})


def _execute(
    ctx: JobContext,
    job_id: str,
    scene_id: str,
    scene_uri: str,
    aoi: dict[str, Any] | None,
    field_id: str | None,
) -> None:
    settings, storage, engine = ctx.settings, ctx.storage, ctx.engine
    if not engine.ready:
        engine.load()
    report = ProgressReporter(ctx.db, job_id)
    cube_path: Path = storage.resolve_uri(scene_uri)
    if aoi is None and field_id is not None:
        aoi = load_fields(settings.fields_path).geometry_of(field_id)

    with rasterio.open(cube_path) as src:
        crs, transform = src.crs, src.transform
        height, width = src.height, src.width
        aoi_mask = _aoi_mask(aoi, src) if aoi else None

    chunker = RasterChunker(
        cube_path,
        window_size=WINDOW_SIZE,
        overlap=settings.window_overlap,
        batch_size=settings.batch_size,
        aoi_mask=aoi_mask,
        prefetch=settings.prefetch_batches,
    )
    stitcher = Stitcher(
        height, width, WINDOW_SIZE, settings.window_overlap, scratch_dir=storage.job_dir(job_id)
    )
    total = max(1, len(chunker))
    done = 0
    report(0.05, force=True)
    for batch in chunker.batches():
        probs, onset = engine.predict(batch.data)
        stitcher.add(probs, onset, batch.rows, batch.cols, batch.valid)
        done += len(batch)
        report(0.05 + 0.80 * done / total)
    # TODO(Day 10): count skipped (all-nodata) windows too so progress is exact.

    probs_full, onset_full, valid = stitcher.finalize(aoi_mask)
    risk_path = storage.risk_path(job_id)
    write_risk_cog(
        risk_path, probs_full, onset_full, crs, transform, {"job_id": job_id, "scene_id": scene_id}
    )
    report(0.9, force=True)

    zones, summary = extract_zones(
        probs_full,
        onset_full,
        valid,
        transform,
        crs,
        job_id=job_id,
        scene_id=scene_id,
        config=ZoneConfig(
            min_prob=settings.zone_min_prob,
            min_acres=settings.zone_min_acres,
            simplify_px=settings.zone_simplify_px,
        ),
        cube_path=cube_path,
    )
    zones_path = storage.zones_path(job_id)
    zones_path.write_text(zones.model_dump_json())
    _update(
        ctx.db,
        job_id,
        status=JobState.SUCCEEDED,
        progress=1.0,
        summary=summary.model_dump(mode="json"),
        risk_path=str(risk_path),
        zones_path=str(zones_path),
        finished_at=utcnow(),
        error=None,
    )
