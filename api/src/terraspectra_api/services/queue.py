"""Job dispatch: RQ on Redis (production) or inline in-process (dev/tests)."""

from __future__ import annotations

from typing import Protocol

from fastapi import BackgroundTasks

from terraspectra_api.services.jobs import JobContext, run_job
from terraspectra_api.settings import Settings

RUN_JOB_PATH = "terraspectra_api.services.jobs.run_job"


class JobQueue(Protocol):
    def enqueue(self, job_id: str, background: BackgroundTasks | None = None) -> None: ...


class InlineJobQueue:
    """Runs the job in-process after the response is sent (or immediately)."""

    def __init__(self, ctx: JobContext) -> None:
        self.ctx = ctx

    def enqueue(self, job_id: str, background: BackgroundTasks | None = None) -> None:
        if background is not None:
            background.add_task(run_job, job_id, self.ctx)
        else:
            run_job(job_id, self.ctx)


class RQJobQueue:
    """Enqueues ``run_job(job_id)`` onto an RQ queue; the worker holds the model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._queue: object | None = None

    def _get_queue(self) -> object:
        if self._queue is None:
            from redis import Redis
            from rq import Queue

            conn = Redis.from_url(self.settings.redis_url)
            self._queue = Queue(self.settings.queue_name, connection=conn)
        return self._queue

    def enqueue(self, job_id: str, background: BackgroundTasks | None = None) -> None:
        queue = self._get_queue()
        queue.enqueue(  # type: ignore[attr-defined]
            RUN_JOB_PATH,
            job_id,
            job_id=job_id,
            job_timeout=self.settings.job_timeout_s,
            result_ttl=86400,
            failure_ttl=7 * 86400,
        )


def build_queue(settings: Settings, ctx: JobContext) -> JobQueue:
    if settings.queue_mode == "inline":
        return InlineJobQueue(ctx)
    return RQJobQueue(settings)
