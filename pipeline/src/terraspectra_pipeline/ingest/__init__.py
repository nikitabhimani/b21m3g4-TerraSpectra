"""Scene acquisition: USGS M2M downloads and local archive registration."""

from terraspectra_pipeline.ingest.local import SceneRecord, register_scene, safe_extract
from terraspectra_pipeline.ingest.usgs import M2MError, SceneHit, USGSClient

__all__ = ["M2MError", "SceneHit", "SceneRecord", "USGSClient", "register_scene", "safe_extract"]
