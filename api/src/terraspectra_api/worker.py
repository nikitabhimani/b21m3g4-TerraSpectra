"""RQ worker entry point: ``python -m terraspectra_api.worker``.

Uses ``SimpleWorker`` (no fork per job) so the model is loaded once per process and
CUDA contexts stay valid.
"""

from __future__ import annotations

import argparse
import logging

from terraspectra_api.logging import configure_logging
from terraspectra_api.services.jobs import build_context, set_context
from terraspectra_api.settings import get_settings

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Preload the engine, then process jobs from Redis until stopped."""
    from redis import Redis
    from rq import Queue, SimpleWorker

    settings = get_settings()
    parser = argparse.ArgumentParser(description="TerraSpectra inference worker")
    parser.add_argument("--queue", default=settings.queue_name)
    parser.add_argument("--burst", action="store_true", help="exit when the queue is empty")
    args = parser.parse_args(argv)

    configure_logging(settings.log_level)
    ctx = build_context(settings, load_engine=True)
    set_context(ctx)
    log.info("worker ready", extra={"device": ctx.engine.device_name, "queue": args.queue})

    conn = Redis.from_url(settings.redis_url)
    worker = SimpleWorker([Queue(args.queue, connection=conn)], connection=conn)
    # TODO(Day 10): one worker per GPU (CUDA_VISIBLE_DEVICES) and a scale-out guide.
    worker.work(burst=args.burst, with_scheduler=False)


if __name__ == "__main__":
    main()
