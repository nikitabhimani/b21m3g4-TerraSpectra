"""USGS Machine-to-Machine (M2M) API client for EO-1 Hyperion downloads."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx

from terraspectra_pipeline.config import PipelineSettings, get_settings
from terraspectra_pipeline.logging import get_logger

log = get_logger(__name__)

HYPERION_DATASET = "eo1_hyp_pub"


class M2MError(RuntimeError):
    """Raised when the M2M API returns an ``errorCode``."""


@dataclass(frozen=True)
class SceneHit:
    entity_id: str
    display_id: str
    acquired: str | None
    raw: dict[str, Any]


class USGSClient:
    """Thin JSON client: ``login-token`` -> ``scene-search`` -> ``download-*`` -> ``logout``.

    Pass ``transport=httpx.MockTransport(...)`` in tests; nothing here touches the network
    until a method is called.
    """

    def __init__(
        self,
        settings: PipelineSettings | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._http = httpx.Client(
            base_url=self.settings.usgs_api_url,
            timeout=self.settings.http_timeout_s,
            transport=transport,
        )
        self._api_key: str | None = None

    def _post(self, endpoint: str, payload: dict[str, Any]) -> Any:
        headers = {"X-Auth-Token": self._api_key} if self._api_key else {}
        resp = self._http.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        if body.get("errorCode"):
            raise M2MError(f"{endpoint}: {body['errorCode']}: {body.get('errorMessage')}")
        return body.get("data")

    def login(self) -> None:
        """Authenticate with ``USGS_M2M_USERNAME`` / ``USGS_M2M_TOKEN``."""
        s = self.settings
        if not s.usgs_username or s.usgs_token is None:
            raise M2MError("set USGS_M2M_USERNAME and USGS_M2M_TOKEN to use the USGS client")
        self._api_key = self._post(
            "login-token", {"username": s.usgs_username, "token": s.usgs_token.get_secret_value()}
        )

    def logout(self) -> None:
        if self._api_key:
            try:
                self._post("logout", {})
            finally:
                self._api_key = None

    def scene_search(
        self,
        bbox: tuple[float, float, float, float],
        start: date,
        end: date,
        dataset: str = HYPERION_DATASET,
        max_results: int = 50,
        max_cloud_cover: int | None = None,
    ) -> list[SceneHit]:
        """Search scenes intersecting ``bbox`` (lon_min, lat_min, lon_max, lat_max)."""
        west, south, east, north = bbox
        scene_filter: dict[str, Any] = {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": south, "longitude": west},
                "upperRight": {"latitude": north, "longitude": east},
            },
            "acquisitionFilter": {"start": start.isoformat(), "end": end.isoformat()},
        }
        if max_cloud_cover is not None:
            scene_filter["cloudCoverFilter"] = {"min": 0, "max": max_cloud_cover}
        data = self._post(
            "scene-search",
            {"datasetName": dataset, "maxResults": max_results, "sceneFilter": scene_filter},
        )
        return [
            SceneHit(
                entity_id=r["entityId"],
                display_id=r.get("displayId", r["entityId"]),
                acquired=(r.get("temporalCoverage") or {}).get("startDate"),
                raw=r,
            )
            for r in (data or {}).get("results", [])
        ]

    def download_options(
        self, entity_ids: list[str], dataset: str = HYPERION_DATASET
    ) -> list[dict[str, Any]]:
        data = self._post("download-options", {"datasetName": dataset, "entityIds": entity_ids})
        return [o for o in (data or []) if o.get("available")]

    def download_request(self, options: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        """Request downloads; returns entries with ``url`` once ready."""
        downloads = [{"entityId": o["entityId"], "productId": o["id"]} for o in options]
        data = self._post("download-request", {"downloads": downloads, "label": label}) or {}
        # TODO(Day 2): poll `download-retrieve` for `preparingDownloads` until URLs are available.
        return list(data.get("availableDownloads", []))

    def download_file(self, url: str, dest_dir: Path, chunk_size: int = 1 << 20) -> Path:
        """Stream a file to ``dest_dir`` (resumable downloads are a TODO)."""
        # TODO(Day 2): add retries/Range-resume and checksum verification for multi-GB bundles.
        dest_dir.mkdir(parents=True, exist_ok=True)
        with self._http.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            name = _filename_from(resp) or url.rstrip("/").rsplit("/", 1)[-1]
            target = dest_dir / name
            part = target.with_suffix(target.suffix + ".part")
            with part.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size):
                    fh.write(chunk)
            part.replace(target)
        log.info("downloaded %s", target)
        return target

    def iter_download(
        self, entity_ids: list[str], dest_dir: Path, label: str = "terraspectra"
    ) -> Iterator[Path]:
        """Convenience: options -> request -> stream each available file."""
        for item in self.download_request(self.download_options(entity_ids), label):
            yield self.download_file(item["url"], dest_dir)

    def close(self) -> None:
        self.logout()
        self._http.close()

    def __enter__(self) -> Self:
        self.login()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _filename_from(resp: httpx.Response) -> str | None:
    disp = resp.headers.get("content-disposition", "")
    for part in disp.split(";"):
        key, _, value = part.strip().partition("=")
        if key.lower() == "filename" and value:
            return Path(value.strip('"')).name
    return None
