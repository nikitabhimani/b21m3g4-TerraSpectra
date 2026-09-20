"""API schemas not already provided by ``terraspectra_contracts``."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from terraspectra_contracts import N_BANDS
from terraspectra_contracts.schemas import PolygonGeometry


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    device: str
    model_loaded: bool


class ErrorOut(BaseModel):
    detail: str
    code: str | None = None


class SceneOut(BaseModel):
    scene_id: str
    name: str
    uri: str
    width: int
    height: int
    bands: int = Field(json_schema_extra={"const": N_BANDS})
    crs: str
    bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    created_at: datetime


class FieldProperties(BaseModel):
    model_config = ConfigDict(extra="allow")

    field_id: str
    name: str
    crop: str | None = None
    area_acres: float


class FieldFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: PolygonGeometry
    properties: FieldProperties


class FieldCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[FieldFeature]

    def get(self, field_id: str) -> FieldFeature | None:
        return next((f for f in self.features if f.properties.field_id == field_id), None)

    def geometry_of(self, field_id: str) -> dict[str, Any] | None:
        feature = self.get(field_id)
        return feature.geometry.model_dump(mode="json") if feature else None


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorOut, "description": "Error"} for code in (404, 409, 422)
}
