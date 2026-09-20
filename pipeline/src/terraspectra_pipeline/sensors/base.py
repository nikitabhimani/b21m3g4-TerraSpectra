"""Sensor reader interface, scene metadata and reader registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Self, TypeVar

import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from terraspectra_pipeline.chunking import iter_blocks


class Quantity(StrEnum):
    """What ``scale * DN + offset`` yields for a product."""

    RADIANCE = "radiance"  # W m-2 sr-1 um-1 -> needs TOA conversion
    REFLECTANCE = "reflectance"  # surface/TOA reflectance in [0, 1]


@dataclass
class SceneMetadata:
    """Sensor-agnostic scene description. Arrays are per band, in file band order."""

    sensor: str
    wavelengths_nm: np.ndarray
    fwhm_nm: np.ndarray
    width: int
    height: int
    crs: CRS | None
    transform: Affine
    scale: np.ndarray
    offset: np.ndarray
    quantity: Quantity
    bad_bands: np.ndarray
    nodata: float | None = None
    acquired_at: datetime | None = None
    sun_elevation_deg: float | None = None
    scene_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.wavelengths_nm)
        for name in ("fwhm_nm", "scale", "offset", "bad_bands"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, expected {n}")

    @property
    def band_count(self) -> int:
        return len(self.wavelengths_nm)

    def summary(self) -> dict[str, Any]:
        """JSON-friendly summary for ``info``."""
        wl = self.wavelengths_nm
        return {
            "sensor": self.sensor,
            "scene_id": self.scene_id,
            "bands": self.band_count,
            "bad_bands": int(np.count_nonzero(self.bad_bands)),
            "wavelength_range_nm": [round(float(wl.min()), 2), round(float(wl.max()), 2)],
            "width": self.width,
            "height": self.height,
            "crs": self.crs.to_string() if self.crs else None,
            "transform": list(self.transform)[:6],
            "quantity": self.quantity.value,
            "nodata": self.nodata,
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
            "sun_elevation_deg": self.sun_elevation_deg,
            **self.extra,
        }


class SensorReader(ABC):
    """Lazy, windowed access to a raw scene. Never loads the full cube."""

    name: ClassVar[str]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._metadata: SceneMetadata | None = None

    @classmethod
    @abstractmethod
    def can_read(cls, path: Path) -> bool:
        """Cheap check whether ``path`` looks like this sensor's product."""

    @abstractmethod
    def _load_metadata(self) -> SceneMetadata: ...

    @abstractmethod
    def read_window(self, window: Window) -> np.ndarray:
        """Raw values ``(B, h, w)`` as float32 for ``window`` (file band order)."""

    @property
    def metadata(self) -> SceneMetadata:
        if self._metadata is None:
            self._metadata = self._load_metadata()
        return self._metadata

    def iter_blocks(self, block_size: int = 512) -> Iterator[tuple[Window, np.ndarray]]:
        """Iterate non-overlapping blocks covering the scene."""
        m = self.metadata
        for win in iter_blocks(m.height, m.width, block_size):
            yield win, self.read_window(win)

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release file handles."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


R = TypeVar("R", bound=type[SensorReader])
_REGISTRY: dict[str, type[SensorReader]] = {}


def register_reader(cls: R) -> R:
    """Class decorator adding a reader to the registry under ``cls.name``."""
    _REGISTRY[cls.name] = cls
    return cls


def available_sensors() -> list[str]:
    return sorted(_REGISTRY)


def get_reader_class(name: str) -> type[SensorReader]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown sensor {name!r}; available: {available_sensors()}") from exc


def detect_sensor(path: str | Path) -> str | None:
    """Return the first registered sensor whose ``can_read`` accepts ``path``."""
    p = Path(path)
    for name in _detection_order():
        if _REGISTRY[name].can_read(p):
            return name
    return None


def _detection_order() -> list[str]:
    # Specific product formats first, the generic GeoTIFF reader last.
    return sorted(_REGISTRY, key=lambda n: (n == "generic", n))


def open_scene(path: str | Path, sensor: str | None = None) -> SensorReader:
    """Open ``path`` with the named reader, or auto-detect it."""
    name = sensor or detect_sensor(path)
    if name is None:
        raise ValueError(f"could not detect sensor for {path}; pass sensor= explicitly")
    return get_reader_class(name)(path)
