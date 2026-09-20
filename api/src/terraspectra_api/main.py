"""Application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from terraspectra_api import __version__
from terraspectra_api.core.tiles import TileRenderer
from terraspectra_api.db import Database
from terraspectra_api.errors import error_response, install_exception_handlers
from terraspectra_api.logging import RequestIdMiddleware, configure_logging
from terraspectra_api.routers import fields, health, jobs, scenes, tiles
from terraspectra_api.security import RateLimiter, rate_limit
from terraspectra_api.services.jobs import JobContext, build_engine
from terraspectra_api.services.queue import build_queue
from terraspectra_api.settings import Settings, get_settings
from terraspectra_api.storage import build_storage

log = logging.getLogger(__name__)
API_PREFIX = "/v1"


class UploadSizeLimitMiddleware:
    """Reject bodies whose declared ``Content-Length`` exceeds the upload limit early."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        # multipart framing overhead allowance
        self.max_bytes = max_bytes + 64 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in ("POST", "PUT"):
            length = Headers(scope=scope).get("content-length")
            if length and length.isdigit() and int(length) > self.max_bytes:
                response = error_response(413, "request body too large", "payload_too_large")
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired FastAPI app. Heavy resources are created in the lifespan."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.is_dev:
            await run_in_threadpool(app.state.db.create_all)
        engine = build_engine(settings, app.state.storage)
        await run_in_threadpool(engine.load)  # fails fast in production if model is missing
        ctx = JobContext(
            settings=settings, db=app.state.db, storage=app.state.storage, engine=engine
        )
        app.state.engine = engine
        app.state.queue = build_queue(settings, ctx)
        log.info(
            "api started",
            extra={"env": settings.env, "queue": settings.queue_mode, "device": engine.device_name},
        )
        try:
            yield
        finally:
            app.state.db.dispose()

    app = FastAPI(
        title="TerraSpectra Inference API",
        version=__version__,
        description="Contract C3. Hyperspectral crop-disease forecasting service.",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.storage = build_storage(settings.storage_dir, settings.allowed_roots)
    app.state.db = Database(settings.database_url)
    app.state.tiles = TileRenderer(settings.tile_cache_size)
    app.state.rate_limiter = RateLimiter(settings.rate_limit)
    app.state.engine = None

    install_exception_handlers(app)

    public = [health.router, tiles.router]
    protected = [fields.router, scenes.router, jobs.router]
    for router in public:
        app.include_router(router, prefix=API_PREFIX)
    for router in protected:
        app.include_router(router, prefix=API_PREFIX, dependencies=[Depends(rate_limit)])

    Instrumentator(excluded_handlers=["/metrics"]).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
    # Middleware: last added = outermost.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(UploadSizeLimitMiddleware, max_bytes=settings.max_upload_bytes)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
            max_age=600,
        )
    app.add_middleware(RequestIdMiddleware)
    return app
