"""``GET /v1/fields``."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response

from terraspectra_api.deps import get_settings_dep
from terraspectra_api.schemas import FieldCollection
from terraspectra_api.services.fields import load_fields
from terraspectra_api.settings import Settings

GEOJSON = "application/geo+json"
router = APIRouter(tags=["fields"])


@router.get(
    "/fields",
    operation_id="listFields",
    response_model=FieldCollection,
    responses={200: {"content": {GEOJSON: {}}}},
)
def list_fields(settings: Settings = Depends(get_settings_dep)) -> Response:
    """Farm field boundaries as a GeoJSON FeatureCollection."""
    fields = load_fields(settings.fields_path)
    return Response(fields.model_dump_json(exclude_none=True), media_type=GEOJSON)
