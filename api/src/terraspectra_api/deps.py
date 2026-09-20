"""Shared FastAPI dependencies pulling singletons off ``app.state``."""

from __future__ import annotations

from fastapi import Request

from terraspectra_api.core.engine import InferenceEngine
from terraspectra_api.core.tiles import TileRenderer
from terraspectra_api.services.queue import JobQueue
from terraspectra_api.settings import Settings
from terraspectra_api.storage import Storage


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_storage(request: Request) -> Storage:
    storage: Storage = request.app.state.storage
    return storage


def get_engine(request: Request) -> InferenceEngine | None:
    return getattr(request.app.state, "engine", None)


def get_queue(request: Request) -> JobQueue:
    queue: JobQueue = request.app.state.queue
    return queue


def get_tiles(request: Request) -> TileRenderer:
    tiles: TileRenderer = request.app.state.tiles
    return tiles
