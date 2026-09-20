"""EnMAP L2A (surface reflectance): ``*SPECTRAL_IMAGE.TIF`` + ``*METADATA.XML``."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from terraspectra_pipeline.sensors.base import Quantity, SceneMetadata, register_reader
from terraspectra_pipeline.sensors.generic import RasterioBandFileReader, parse_datetime

L2A_DEFAULT_GAIN = 1e-4  # reflectance stored as int16 * 10000
L2A_DEFAULT_NODATA = -32768.0


@dataclass(frozen=True)
class EnmapXmlMetadata:
    wavelengths_nm: np.ndarray
    fwhm_nm: np.ndarray
    gain: np.ndarray
    offset: np.ndarray
    acquired_at: datetime | None
    sun_elevation_deg: float | None
    product_level: str | None
    scene_id: str | None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(root: ET.Element, *path: str) -> ET.Element | None:
    """Namespace-agnostic nested lookup by local tag names."""
    node: ET.Element | None = root
    for name in path:
        if node is None:
            return None
        node = next((c for c in node.iter() if c is not node and _local(c.tag) == name), None)
    return node


def _first(*elements: ET.Element | None) -> ET.Element | None:
    # Element truthiness depends on child count, so never use ``or`` on elements.
    return next((e for e in elements if e is not None), None)


def _text_float(el: ET.Element | None) -> float | None:
    if el is None or el.text is None:
        return None
    try:
        return float(el.text)
    except ValueError:
        return None


def parse_enmap_metadata(xml_path: Path) -> EnmapXmlMetadata:
    """Parse band characterisation, timing and sun geometry from an EnMAP METADATA.XML."""
    root = ET.parse(xml_path).getroot()
    bands = [el for el in root.iter() if _local(el.tag) == "bandID"]
    rows: list[tuple[int, float, float, float | None, float | None]] = []
    for i, band in enumerate(bands, start=1):
        num = int(band.get("number", i))
        wl = _text_float(_find(band, "wavelengthCenterOfBand"))
        fwhm = _text_float(_find(band, "FWHMOfBand"))
        if wl is None or fwhm is None:
            continue
        rows.append(
            (
                num,
                wl,
                fwhm,
                _text_float(_find(band, "GainOfBand")),
                _text_float(_find(band, "OffsetOfBand")),
            )
        )
    if not rows:
        raise ValueError(f"{xml_path}: no <bandID> wavelength entries found")
    rows.sort(key=lambda r: r[0])
    gains = np.array([L2A_DEFAULT_GAIN if r[3] in (None, 0.0) else r[3] for r in rows])
    offsets = np.array([0.0 if r[4] is None else r[4] for r in rows])

    start = _find(root, "temporalCoverage", "startTime")
    sun = _find(root, "sunElevationAngle", "center")
    level = _first(_find(root, "processingLevel"), _find(root, "level"))
    name = _first(_find(root, "metadata", "name"), _find(root, "name"))
    return EnmapXmlMetadata(
        wavelengths_nm=np.array([r[1] for r in rows]),
        fwhm_nm=np.array([r[2] for r in rows]),
        gain=gains,
        offset=offsets,
        acquired_at=parse_datetime(start.text if start is not None else None),
        sun_elevation_deg=_text_float(sun),
        product_level=level.text.strip() if level is not None and level.text else None,
        scene_id=name.text.strip() if name is not None and name.text else None,
    )


def _find_one(root: Path, pattern: str) -> Path | None:
    hits = sorted(root.glob(pattern)) + sorted(root.glob(pattern.lower()))
    return hits[0] if hits else None


@register_reader
class EnmapL2AReader(RasterioBandFileReader):
    """Accepts the product directory or the SPECTRAL_IMAGE file itself."""

    name = "enmap"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.root = self.path if self.path.is_dir() else self.path.parent

    @classmethod
    def can_read(cls, path: Path) -> bool:
        root = path if path.is_dir() else path.parent
        if not root.is_dir():
            return False
        return (
            _find_one(root, "*SPECTRAL_IMAGE*.TIF") is not None
            and _find_one(root, "*METADATA*.XML") is not None
        )

    @property
    def raster_path(self) -> Path:
        if self.path.is_file() and self.path.suffix.lower() in {".tif", ".tiff"}:
            return self.path
        found = _find_one(self.root, "*SPECTRAL_IMAGE*.TIF")
        if found is None:
            raise FileNotFoundError(f"no SPECTRAL_IMAGE GeoTIFF in {self.root}")
        return found

    def _load_metadata(self) -> SceneMetadata:
        xml_path = _find_one(self.root, "*METADATA*.XML")
        if xml_path is None:
            raise FileNotFoundError(f"no METADATA.XML in {self.root}")
        meta = parse_enmap_metadata(xml_path)
        ds = self.dataset
        if ds.count != meta.wavelengths_nm.size:
            raise ValueError(
                f"raster has {ds.count} bands but XML lists {meta.wavelengths_nm.size}"
            )
        # TODO(Day 3): honour the L2A quality layers (QL_QUALITY_CLOUD etc.) as extra masks.
        return SceneMetadata(
            sensor=self.name,
            wavelengths_nm=meta.wavelengths_nm,
            fwhm_nm=meta.fwhm_nm,
            width=ds.width,
            height=ds.height,
            crs=ds.crs,
            transform=ds.transform,
            scale=meta.gain,
            offset=meta.offset,
            quantity=Quantity.REFLECTANCE,
            bad_bands=np.zeros(ds.count, dtype=bool),
            nodata=ds.nodata if ds.nodata is not None else L2A_DEFAULT_NODATA,
            acquired_at=meta.acquired_at,
            sun_elevation_deg=meta.sun_elevation_deg,
            scene_id=meta.scene_id or self.root.name,
            extra={"product_level": meta.product_level, "metadata_xml": xml_path.name},
        )
