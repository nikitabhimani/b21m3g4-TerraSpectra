"""Artifact storage. Local filesystem now; S3/MinIO later."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import unquote, urlparse

CHUNK = 8 * 1024 * 1024


class StorageError(Exception):
    """Base storage error."""


class UploadTooLargeError(StorageError):
    """Upload exceeded ``TS_MAX_UPLOAD_MB``."""


class UnsupportedUriError(StorageError):
    """URI scheme is not supported or points outside the allowed roots."""


class Storage(Protocol):
    """What the service needs from a storage backend."""

    root: Path

    def save_scene(self, scene_id: str, src: BinaryIO, max_bytes: int) -> Path: ...
    def scene_path(self, scene_id: str) -> Path: ...
    def delete(self, path: Path) -> None: ...
    def resolve_uri(self, uri: str) -> Path: ...
    def job_dir(self, job_id: str) -> Path: ...
    def risk_path(self, job_id: str) -> Path: ...
    def zones_path(self, job_id: str) -> Path: ...
    def cache_dir(self) -> Path: ...


class LocalStorage:
    """Local layout under ``TS_STORAGE_DIR``.

    ``scenes/<scene_id>.tif`` and ``jobs/<job_id>/{risk.tif,zones.geojson}``.
    """

    def __init__(self, root: Path, allowed_roots: list[Path] | None = None) -> None:
        self.root = root.resolve()
        self.allowed_roots = [p.resolve() for p in (allowed_roots or [self.root])]
        for sub in ("scenes", "jobs", "cache"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def scene_path(self, scene_id: str) -> Path:
        return self.root / "scenes" / f"{scene_id}.tif"

    def save_scene(self, scene_id: str, src: BinaryIO, max_bytes: int) -> Path:
        """Stream ``src`` to disk atomically, enforcing ``max_bytes``."""
        dest = self.scene_path(scene_id)
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        written = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while chunk := src.read(CHUNK):
                    written += len(chunk)
                    if written > max_bytes:
                        raise UploadTooLargeError(f"upload exceeds {max_bytes} bytes")
                    out.write(chunk)
            os.replace(tmp_name, dest)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return dest

    def delete(self, path: Path) -> None:
        path = path.resolve()
        if path.is_relative_to(self.root):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def resolve_uri(self, uri: str) -> Path:
        """Map a ``file://`` URI (or bare path) to a local path inside the allowed roots."""
        parsed = urlparse(uri)
        if parsed.scheme in ("", "file"):
            raw = unquote(parsed.path if parsed.scheme else uri)
            path = Path(raw).resolve()
            if not any(path.is_relative_to(root) for root in self.allowed_roots):
                raise UnsupportedUriError("URI points outside the allowed storage roots")
            return path
        # TODO(Day 3): s3:// and MinIO support via an S3Storage implementation (boto3/fsspec).
        raise UnsupportedUriError(f"unsupported URI scheme {parsed.scheme!r}")

    def job_dir(self, job_id: str) -> Path:
        path = self.root / "jobs" / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def risk_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "risk.tif"

    def zones_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "zones.geojson"

    def cache_dir(self) -> Path:
        return self.root / "cache"


def build_storage(root: Path, allowed_roots: list[Path] | None = None) -> Storage:
    """Factory; TODO(Day 3): select S3Storage when ``TS_STORAGE_DIR`` is an ``s3://`` URI."""
    return LocalStorage(root, allowed_roots)
