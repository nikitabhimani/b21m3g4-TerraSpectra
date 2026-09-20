"""Scene registry: upload or reference C1 cubes."""

from __future__ import annotations

import logging
from pathlib import Path

import rasterio
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from rasterio.errors import RasterioError
from rasterio.warp import transform_bounds
from sqlalchemy import select
from sqlalchemy.orm import Session

from terraspectra_api.db import get_session
from terraspectra_api.deps import get_settings_dep, get_storage
from terraspectra_api.errors import ApiError, not_found
from terraspectra_api.models import Scene, as_utc, new_id
from terraspectra_api.schemas import ERROR_RESPONSES, SceneOut
from terraspectra_api.settings import Settings
from terraspectra_api.storage import Storage, UnsupportedUriError, UploadTooLargeError
from terraspectra_contracts import N_BANDS, WAVELENGTHS_NM
from terraspectra_contracts.constants import WAVELENGTH_TAG

log = logging.getLogger(__name__)
router = APIRouter(tags=["scenes"])
WAVELENGTH_TOLERANCE_NM = 0.5


class CubeValidationError(ValueError):
    """The raster violates contract C1."""


def validate_cube(path: Path) -> dict[str, object]:
    """Check C1 (200 float32 bands with wavelength tags, georeferenced); return metadata."""
    try:
        with rasterio.open(path) as src:
            if src.count != N_BANDS:
                raise CubeValidationError(f"expected {N_BANDS} bands, got {src.count}")
            if any(dt != "float32" for dt in src.dtypes):
                raise CubeValidationError(f"expected float32 bands, got {sorted(set(src.dtypes))}")
            if src.crs is None:
                raise CubeValidationError("raster has no CRS")
            for i, expected in enumerate(WAVELENGTHS_NM, start=1):
                tag = src.tags(i).get(WAVELENGTH_TAG)
                if tag is None:
                    raise CubeValidationError(f"band {i} is missing the {WAVELENGTH_TAG} tag")
                if abs(float(tag) - expected) > WAVELENGTH_TOLERANCE_NM:
                    raise CubeValidationError(
                        f"band {i} wavelength {tag} nm does not match the C1 grid ({expected:.2f})"
                    )
            if src.nodata is not None and src.nodata != -1.0:
                log.warning("unexpected nodata value", extra={"nodata": src.nodata})
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": src.crs.to_string(),
                "bounds": [round(float(b), 7) for b in bounds],
            }
    except RasterioError as exc:
        raise CubeValidationError(f"not a readable raster: {exc}") from exc
    except ValueError as exc:
        if isinstance(exc, CubeValidationError):
            raise
        raise CubeValidationError(f"invalid raster metadata: {exc}") from exc


def to_out(scene: Scene) -> SceneOut:
    return SceneOut(
        scene_id=scene.id,
        name=scene.name,
        uri=scene.uri,
        width=scene.width,
        height=scene.height,
        bands=scene.bands,
        crs=scene.crs,
        bounds=scene.bounds,
        created_at=as_utc(scene.created_at),
    )


@router.get("/scenes", operation_id="listScenes", response_model=list[SceneOut])
def list_scenes(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
) -> list[SceneOut]:
    """Registered scenes, newest first."""
    rows = session.scalars(
        select(Scene).order_by(Scene.created_at.desc(), Scene.id).limit(limit).offset(offset)
    )
    return [to_out(s) for s in rows]


@router.post(
    "/scenes",
    operation_id="createScene",
    status_code=201,
    response_model=SceneOut,
    responses={422: ERROR_RESPONSES[422]},
)
def create_scene(
    file: UploadFile | None = File(None, description="C1-compliant COG"),
    uri: str | None = Form(None, description="Alternative to upload: s3:// or file:// URI"),
    name: str | None = Form(None, max_length=255),
    session: Session = Depends(get_session),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings_dep),
) -> SceneOut:
    """Register a scene from a multipart upload or an existing ``file://`` URI."""
    scene_id = new_id("scn")
    uploaded = file is not None and bool(file.filename)
    if uploaded:
        assert file is not None
        if file.size is not None and file.size > settings.max_upload_bytes:
            raise ApiError(413, "upload too large", "payload_too_large")
        try:
            path = storage.save_scene(scene_id, file.file, settings.max_upload_bytes)
        except UploadTooLargeError as exc:
            raise ApiError(413, str(exc), "payload_too_large") from exc
        scene_uri = path.as_uri()
        default_name = Path(file.filename or scene_id).name
    elif uri:
        try:
            path = storage.resolve_uri(uri)
        except UnsupportedUriError as exc:
            raise ApiError(422, str(exc), "unsupported_uri") from exc
        if not path.is_file():
            raise ApiError(422, "URI does not point to an existing file", "invalid_uri")
        scene_uri = path.as_uri()
        default_name = path.name
    else:
        raise ApiError(422, "provide either 'file' or 'uri'", "validation_error")

    try:
        meta = validate_cube(path)
    except CubeValidationError as exc:
        if uploaded:
            storage.delete(path)
        raise ApiError(422, str(exc), "invalid_cube") from exc

    scene = Scene(id=scene_id, name=name or default_name, uri=scene_uri, uploaded=uploaded, **meta)
    session.add(scene)
    session.flush()
    log.info("scene registered", extra={"scene_id": scene_id, "uploaded": uploaded})
    return to_out(scene)


@router.get(
    "/scenes/{scene_id}",
    operation_id="getScene",
    response_model=SceneOut,
    responses={404: ERROR_RESPONSES[404]},
)
def get_scene(scene_id: str, session: Session = Depends(get_session)) -> SceneOut:
    scene = session.get(Scene, scene_id)
    if scene is None:
        raise not_found("scene", scene_id)
    return to_out(scene)
