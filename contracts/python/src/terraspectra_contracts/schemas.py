"""Contract C3/C4 - API and zone payload schemas (mirrors openapi.yaml and zones.schema.json)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobCreate(BaseModel):
    scene_id: str
    field_id: str | None = None
    aoi: dict[str, Any] | None = Field(default=None, description="GeoJSON Polygon in EPSG:4326")


class JobSummary(BaseModel):
    acres_analyzed: float
    acres_at_risk: float
    zones_by_class: dict[str, int]


class JobStatus(BaseModel):
    job_id: str
    scene_id: str
    field_id: str | None = None
    status: JobState
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    summary: JobSummary | None = None


class ZoneProperties(BaseModel):
    zone_id: str
    risk_class: int = Field(ge=0, le=3)
    risk_class_name: Literal["healthy", "early_stress", "high_blight_risk", "visible_disease"]
    risk_score: float = Field(ge=0.0, le=1.0)
    area_acres: float = Field(ge=0.0)
    days_to_onset: float = Field(ge=0.0, le=30.0)
    dominant_indicator: str
    recommended_action: str


class PolygonGeometry(BaseModel):
    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[tuple[float, float]]]


class ZoneFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PolygonGeometry
    properties: ZoneProperties


class ZoneFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    job_id: str
    scene_id: str
    generated_at: datetime
    features: list[ZoneFeature]
