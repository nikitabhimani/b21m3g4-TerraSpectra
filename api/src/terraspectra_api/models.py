"""ORM models: registered scenes and inference jobs."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from terraspectra_api.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Short prefixed id, e.g. ``scn_1a2b3c4d5e6f``."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def as_utc(value: datetime) -> datetime:
    """SQLite drops tzinfo; treat naive timestamps as UTC."""
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class Scene(Base):
    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    uri: Mapped[str] = mapped_column(Text)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    bands: Mapped[int] = mapped_column(Integer)
    crs: Mapped[str] = mapped_column(String(64))
    bounds: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    uploaded: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list[Job]] = relationship(back_populates="scene")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scene_id: Mapped[str] = mapped_column(ForeignKey("scenes.id"), index=True)
    field_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    aoi: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    risk_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    zones_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    scene: Mapped[Scene] = relationship(back_populates="jobs")
