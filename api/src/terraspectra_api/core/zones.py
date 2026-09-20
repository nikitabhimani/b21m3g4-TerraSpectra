"""Risk-zone extraction (contract C4) from the stitched risk raster."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Geod
from rasterio.features import shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry

from terraspectra_contracts import CLASS_NAMES, MAX_ONSET_DAYS, NODATA, WAVELENGTHS_NM, RiskClass
from terraspectra_contracts.schemas import JobSummary, ZoneFeatureCollection

SQM_PER_ACRE = 4046.8564224
_GEOD = Geod(ellps="WGS84")
_CROSS = ndimage.generate_binary_structure(2, 1)  # 4-connectivity for labelling
_SQUARE = np.ones((3, 3), dtype=bool)  # opening element (keeps rectangle corners)


@dataclass(frozen=True)
class ZoneConfig:
    min_prob: float = 0.4
    min_acres: float = 1.0
    simplify_px: float = 0.5
    opening_iterations: int = 1


# --------------------------------------------------------------------------- areas


def pixel_area_acres(transform: Affine, crs: CRS | None, height: int, width: int) -> float:
    """Approximate area of one pixel in acres (geodesic at scene centre if geographic)."""
    if crs is not None and crs.is_geographic:
        x0, y0 = transform @ (width / 2, height / 2)
        x1, y1 = transform @ (width / 2 + 1, height / 2 + 1)
        area, _ = _GEOD.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        return abs(area) / SQM_PER_ACRE
    unit = crs.axis_info[0].unit_conversion_factor if crs is not None and crs.axis_info else 1.0
    return abs(transform.a * transform.e - transform.b * transform.d) * unit**2 / SQM_PER_ACRE


def geometry_area_acres(geom: BaseGeometry, crs: CRS | None) -> float:
    """Area of a geometry expressed in ``crs`` coordinates, in acres."""
    if crs is not None and crs.is_geographic:
        return abs(_GEOD.geometry_area_perimeter(geom)[0]) / SQM_PER_ACRE
    unit = crs.axis_info[0].unit_conversion_factor if crs is not None and crs.axis_info else 1.0
    return float(geom.area) * unit**2 / SQM_PER_ACRE


# --------------------------------------------------------------------- indicators

INDICATOR_NM = {"b531": 531, "b570": 570, "b670": 670, "b700": 700, "b710": 710,
                "b740": 740, "b780": 780, "b860": 860, "b1240": 1240}  # fmt: skip


def band_for(nm: float) -> int:
    """1-based band index nearest to ``nm`` on the C1 wavelength grid."""
    return int(np.argmin(np.abs(np.asarray(WAVELENGTHS_NM) - nm))) + 1


def spectral_indices(bands: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean stress indices over the given pixels (1-D reflectance arrays per band)."""
    eps = 1e-6
    b = {k: float(np.mean(v)) for k, v in bands.items()}
    reip = 700 + 40 * ((b["b670"] + b["b780"]) / 2 - b["b700"]) / (b["b740"] - b["b700"] + eps)
    return {
        "reip": reip,
        "pri": (b["b531"] - b["b570"]) / (b["b531"] + b["b570"] + eps),
        "ci_re": b["b780"] / (b["b710"] + eps) - 1.0,
        "ndwi": (b["b860"] - b["b1240"]) / (b["b860"] + b["b1240"] + eps),
    }


class IndicatorEstimator:
    """Heuristic ``dominant_indicator`` from a few diagnostic bands of the input cube.

    Compares zone-mean indices with a scene reference (median over valid pixels of a
    decimated read) and returns the index with the largest normalised stress deviation:
    ``red_edge_shift`` (REIP blue-shift), ``pri_drop`` (photochemical reflectance index),
    ``chlorophyll_loss`` (red-edge chlorophyll index) or ``water_stress`` (NDWI 860/1240).

    TODO(Day 13): replace with P2's model explanation output (per-zone attributions).
    """

    SCALES: ClassVar[dict[str, float]] = {"reip": 5.0, "pri": 0.02, "ci_re": 0.25, "ndwi": 0.05}
    NAMES: ClassVar[dict[str, str]] = {
        "reip": "red_edge_shift",
        "pri": "pri_drop",
        "ci_re": "chlorophyll_loss",
        "ndwi": "water_stress",
    }

    def __init__(self, cube_path: Path | None) -> None:
        self.cube_path = cube_path
        self.indexes = [band_for(nm) for nm in INDICATOR_NM.values()]
        self.reference: dict[str, float] | None = None
        if cube_path is not None:
            with rasterio.open(cube_path) as src:
                scale = max(1, int(np.ceil(max(src.height, src.width) / 512)))
                shape_out = (
                    len(self.indexes),
                    src.height // scale or 1,
                    src.width // scale or 1,
                )
                arr = src.read(self.indexes, out_shape=shape_out)
            valid = (arr != NODATA).all(axis=0) & np.isfinite(arr).all(axis=0)
            if valid.any():
                self.reference = spectral_indices(
                    {k: np.median(arr[i][valid]) for i, k in enumerate(INDICATOR_NM)}
                )

    def dominant(self, window: Window, mask: np.ndarray) -> str:
        """Dominant indicator for the pixels ``mask`` (shaped like ``window``)."""
        if self.cube_path is None or self.reference is None:
            return self.NAMES["reip"]
        with rasterio.open(self.cube_path) as src:
            arr = src.read(self.indexes, window=window)
        sel = mask & (arr != NODATA).all(axis=0)
        if not sel.any():
            return self.NAMES["reip"]
        zone = spectral_indices({k: arr[i][sel] for i, k in enumerate(INDICATOR_NM)})
        # Stress lowers every index; positive deviation = more stressed than the scene.
        dev = {k: (self.reference[k] - zone[k]) / self.SCALES[k] for k in self.SCALES}
        return self.NAMES[max(dev, key=lambda k: dev[k])]


# ------------------------------------------------------------------------ actions


def recommended_action(risk_class: int, days_to_onset: float, indicator: str) -> str:
    """Rule-based agronomic recommendation for a zone.

    TODO(Day 8): make rules crop-specific and configurable (YAML) with agronomist input.
    """
    if risk_class == RiskClass.VISIBLE_DISEASE:
        action = "Apply curative fungicide immediately and remove infected plants"
    elif risk_class == RiskClass.HIGH_BLIGHT_RISK:
        within = "48 hours" if days_to_onset < 7 else "7 days"
        action = f"Apply targeted preventative fungicide within {within}"
    elif risk_class == RiskClass.EARLY_STRESS:
        action = "Scout zone within 3 days and increase monitoring frequency"
    else:
        action = "No action required"
    if indicator == "water_stress" and risk_class != RiskClass.HEALTHY:
        action += "; check irrigation"
    return action


# -------------------------------------------------------------------------- zones


def _as_polygon(geom: BaseGeometry) -> Polygon | None:
    if geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom if isinstance(geom, Polygon) and geom.area > 0 else None


def _to_wgs84(geom: Polygon, crs: CRS | None) -> dict[str, Any]:
    src = crs.to_wkt() if crs is not None else "EPSG:4326"
    out = transform_geom(src, "EPSG:4326", mapping(geom), precision=7)
    coords = [[[float(x), float(y)] for x, y, *_ in ring] for ring in out["coordinates"]]
    return {"type": "Polygon", "coordinates": coords}


def extract_zones(
    probs: np.ndarray,
    onset: np.ndarray,
    valid: np.ndarray,
    transform: Affine,
    crs: Any,
    job_id: str,
    scene_id: str,
    config: ZoneConfig | None = None,
    cube_path: Path | None = None,
) -> tuple[ZoneFeatureCollection, JobSummary]:
    """Threshold → opening → connected components → polygons (EPSG:4326) + summary."""
    config = config or ZoneConfig()
    crs_obj = CRS.from_user_input(crs) if crs is not None else None
    height, width = valid.shape
    px_acres = pixel_area_acres(transform, crs_obj, height, width)
    min_pixels = max(1, int(np.ceil(config.min_acres / px_acres))) if px_acres > 0 else 1
    class_map = np.where(valid, np.argmax(np.where(valid, probs, 0.0), axis=0), -1)
    risk = np.clip(1.0 - probs[RiskClass.HEALTHY], 0.0, 1.0)
    indicators = IndicatorEstimator(cube_path)

    features: list[dict[str, Any]] = []
    counts = {name: 0 for name in CLASS_NAMES.values()}
    acres_at_risk = 0.0
    for cls in (RiskClass.EARLY_STRESS, RiskClass.HIGH_BLIGHT_RISK, RiskClass.VISIBLE_DISEASE):
        mask = (class_map == cls) & (probs[cls] >= config.min_prob)
        if config.opening_iterations > 0:
            mask = ndimage.binary_opening(
                mask, structure=_SQUARE, iterations=config.opening_iterations
            )
        labels, n = ndimage.label(mask, structure=_CROSS)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel())
        keep = sizes >= min_pixels
        keep[0] = False
        labels = np.where(keep[labels], labels, 0).astype(np.int32)
        slices = ndimage.find_objects(labels)
        polygons = shapes(labels, mask=labels > 0, connectivity=4, transform=transform)
        for geom_json, label in polygons:
            label = int(label)
            sl = slices[label - 1]
            if sl is None:
                continue
            comp = labels[sl] == label
            simplified = shape(geom_json).simplify(
                config.simplify_px * abs(transform.a), preserve_topology=True
            )
            poly = _as_polygon(simplified)
            if poly is None:
                continue
            window = Window(
                sl[1].start,
                sl[0].start,
                sl[1].stop - sl[1].start,
                sl[0].stop - sl[0].start,
            )
            area = geometry_area_acres(poly, crs_obj)
            days = float(np.clip(onset[sl][comp].mean(), 0.0, MAX_ONSET_DAYS))
            indicator = indicators.dominant(window, comp)
            counts[CLASS_NAMES[cls]] += 1
            acres_at_risk += area
            features.append(
                {
                    "type": "Feature",
                    "geometry": _to_wgs84(poly, crs_obj),
                    "properties": {
                        "zone_id": "",
                        "risk_class": int(cls),
                        "risk_class_name": CLASS_NAMES[cls],
                        "risk_score": round(float(np.clip(risk[sl][comp].mean(), 0, 1)), 4),
                        "area_acres": round(area, 3),
                        "days_to_onset": round(days, 2),
                        "dominant_indicator": indicator,
                        "recommended_action": recommended_action(cls, days, indicator),
                    },
                }
            )
    features.sort(key=lambda f: (-f["properties"]["risk_class"], -f["properties"]["risk_score"]))
    for i, feat in enumerate(features, start=1):
        feat["properties"]["zone_id"] = f"z-{i:03d}"

    collection = ZoneFeatureCollection.model_validate(
        {
            "type": "FeatureCollection",
            "job_id": job_id,
            "scene_id": scene_id,
            "generated_at": datetime.now(UTC),
            "features": features,
        }
    )
    summary = JobSummary(
        acres_analyzed=round(float(valid.sum()) * px_acres, 3),
        acres_at_risk=round(acres_at_risk, 3),
        zones_by_class=counts,
    )
    return collection, summary
