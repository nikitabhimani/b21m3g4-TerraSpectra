"""Server-Sent Events (SSE) streaming and pub/sub for live job progress."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

from terraspectra_api.db import Database
from terraspectra_api.models import Job, as_utc
from terraspectra_contracts.schemas import JobState

log = logging.getLogger(__name__)

# In-process listener queues for pub/sub within the same process (dev/inline mode/tests)
_subscribers: dict[str, list[asyncio.Queue[tuple[str, dict[str, Any]]]]] = defaultdict(list)
_subscribers_lock = threading.Lock()


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format an SSE message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def publish_job_event(
    job_id: str,
    event_type: str,
    data: dict[str, Any],
    redis_client: Any | None = None,
) -> None:
    """Publish a job event to in-process listeners and optionally to Redis."""
    # 1. Notify in-process async queues
    with _subscribers_lock:
        queues = list(_subscribers.get(job_id, []))
    for q in queues:
        with contextlib.suppress(Exception):
            q.put_nowait((event_type, data))

    # 2. Publish to Redis channel if Redis client is available
    if redis_client is not None:
        try:
            channel = f"ts:events:{job_id}"
            payload = json.dumps({"event": event_type, "data": data})
            redis_client.publish(channel, payload)
        except Exception as exc:
            log.debug("Redis publish event error: %s", exc)


async def stream_job_events(
    job_id: str,
    db: Database,
    redis_client: Any | None = None,
    poll_interval: float = 0.5,
    timeout_s: float = 3600.0,
) -> AsyncIterator[str]:
    """Async generator yielding Server-Sent Events for job execution progress."""
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    # Register in-process subscriber
    with _subscribers_lock:
        _subscribers[job_id].append(queue)

    start_time = time.monotonic()
    last_ping = time.monotonic()
    last_progress = -1.0
    last_status = ""

    try:
        # 1. Fetch current status from database
        with db.session() as session:
            job = session.get(Job, job_id)
            if job is None:
                yield format_sse("error", {"job_id": job_id, "detail": f"job {job_id} not found"})
                return

            last_progress = float(job.progress)
            last_status = str(job.status)
            created_at = as_utc(job.created_at).isoformat()
            updated_at = as_utc(job.updated_at).isoformat()

            initial_data = {
                "job_id": job.id,
                "scene_id": job.scene_id,
                "field_id": job.field_id,
                "status": job.status,
                "progress": job.progress,
                "step": f"Job {job.status}",
                "created_at": created_at,
                "updated_at": updated_at,
                "summary": job.summary,
                "error": job.error,
            }

            if job.status == JobState.SUCCEEDED:
                yield format_sse("complete", initial_data)
                return
            if job.status == JobState.FAILED:
                yield format_sse("error", initial_data)
                return
            if job.status == JobState.CANCELLED:
                yield format_sse("cancelled", initial_data)
                return

            # Yield initial status
            yield format_sse("progress", initial_data)

        # 2. Stream subsequent updates
        while time.monotonic() - start_time < timeout_s:
            # Check queue with short timeout
            try:
                event_type, event_data = await asyncio.wait_for(queue.get(), timeout=poll_interval)
                yield format_sse(event_type, event_data)
                if event_type in ("complete", "error", "cancelled"):
                    return
                last_ping = time.monotonic()
                continue
            except TimeoutError:
                pass

            # Fallback/Safety Check: Inspect database for status or progress updates
            with db.session() as session:
                job = session.get(Job, job_id)
                if job is None:
                    yield format_sse(
                        "error", {"job_id": job_id, "detail": f"job {job_id} disappeared"}
                    )
                    return

                current_progress = float(job.progress)
                current_status = str(job.status)

                if current_status != last_status or current_progress > last_progress:
                    last_progress = current_progress
                    last_status = current_status
                    data = {
                        "job_id": job.id,
                        "status": job.status,
                        "progress": job.progress,
                        "step": f"Status: {job.status} ({int(job.progress * 100)}%)",
                        "summary": job.summary,
                        "error": job.error,
                    }
                    if job.status == JobState.SUCCEEDED:
                        yield format_sse("complete", data)
                        return
                    if job.status == JobState.FAILED:
                        yield format_sse("error", data)
                        return
                    if job.status == JobState.CANCELLED:
                        yield format_sse("cancelled", data)
                        return
                    yield format_sse("progress", data)
                    last_ping = time.monotonic()

            # Periodic SSE keepalive comment
            now = time.monotonic()
            if now - last_ping >= 15.0:
                yield ": keep-alive\n\n"
                last_ping = now

    finally:
        with _subscribers_lock:
            if queue in _subscribers[job_id]:
                _subscribers[job_id].remove(queue)
            if not _subscribers[job_id]:
                _subscribers.pop(job_id, None)
