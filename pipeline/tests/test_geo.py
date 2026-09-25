"""Tests for terraspectra_pipeline.geo — pure functions, no real raster files needed."""

import numpy as np
import pytest
from rasterio.crs import CRS

from terraspectra_pipeline.geo import (
    geoms_bounds,
    is_utm,
    load_aoi,
    nodata_mask,
    utm_crs_for_lonlat,
)


def test_utm_crs_jaipur():
    """Jaipur (~75.8E, 26.9N) should fall in UTM zone 43N (EPSG:32643)."""
    crs = utm_crs_for_lonlat(75.8, 26.9)

    assert crs.to_epsg() == 32643


def test_utm_crs_southern_hemisphere():
    """Negative latitude should give a 327xx southern-hemisphere EPSG code."""
    crs = utm_crs_for_lonlat(18.4, -33.9)  # Cape Town

    epsg = crs.to_epsg()

    assert epsg is not None
    assert 32701 <= epsg <= 32760


def test_utm_crs_zone_boundaries():
    """Longitude 0.5E should fall in UTM zone 31N."""
    crs = utm_crs_for_lonlat(0.5, 10.0)

    assert crs.to_epsg() == 32631


def test_is_utm_true_for_utm_crs():
    """A UTM CRS should be detected as UTM."""
    utm = CRS.from_epsg(32643)

    assert is_utm(utm) is True


def test_is_utm_false_for_wgs84():
    """Plain WGS84 geographic CRS should not be detected as UTM."""
    wgs84 = CRS.from_epsg(4326)

    assert is_utm(wgs84) is False


def test_is_utm_false_for_none():
    """None should not be detected as UTM."""
    assert is_utm(None) is False


def test_nodata_mask_flags_nodata_value():
    """A pixel equal to the nodata value in any band should be masked True."""
    cube = np.ones((3, 2, 2), dtype=np.float32)

    # Set the same pixel to nodata in all three bands.
    cube[:, 0, 0] = -1.0

    mask = nodata_mask(cube, nodata=-1.0)

    assert mask[0, 0] == True
    assert mask[0, 1] == False
    assert mask[1, 0] == False
    assert mask[1, 1] == False


def test_nodata_mask_flags_nan():
    """A NaN in any band should also be masked True."""
    cube = np.ones((2, 2, 2), dtype=np.float32)

    # Introduce NaN into one band at pixel (1, 1).
    cube[0, 1, 1] = np.nan

    mask = nodata_mask(cube, nodata=-1.0)

    assert mask[1, 1] == True
    assert mask[0, 0] == False


def test_geoms_bounds_single_polygon():
    """Bounds of a simple square polygon should match its corners."""
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [10.0, 20.0],
                [30.0, 20.0],
                [30.0, 40.0],
                [10.0, 40.0],
                [10.0, 20.0],
            ]
        ],
    }

    min_x, min_y, max_x, max_y = geoms_bounds([polygon])

    assert min_x == 10.0
    assert min_y == 20.0
    assert max_x == 30.0
    assert max_y == 40.0


def test_geoms_bounds_multiple_geometries():
    """Bounds across two polygons should cover both."""
    poly1 = {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [5.0, 0.0],
                [5.0, 5.0],
                [0.0, 0.0],
            ]
        ],
    }

    poly2 = {
        "type": "Polygon",
        "coordinates": [
            [
                [8.0, 8.0],
                [12.0, 8.0],
                [12.0, 12.0],
                [8.0, 8.0],
            ]
        ],
    }

    min_x, min_y, max_x, max_y = geoms_bounds([poly1, poly2])

    assert min_x == 0.0
    assert min_y == 0.0
    assert max_x == 12.0
    assert max_y == 12.0


def test_load_aoi_from_feature_collection():
    """A FeatureCollection with one feature should return one geometry."""
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            }
        ],
    }

    geoms = load_aoi(feature_collection)

    assert len(geoms) == 1
    assert geoms[0]["type"] == "Polygon"


def test_load_aoi_from_bare_polygon():
    """A bare Polygon geometry should work without a Feature wrapper."""
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ]
        ],
    }

    geoms = load_aoi(polygon)

    assert len(geoms) == 1
    assert geoms[0] == polygon


def test_load_aoi_rejects_empty_feature_collection():
    """An empty FeatureCollection should raise ValueError."""
    empty_feature_collection = {
        "type": "FeatureCollection",
        "features": [],
    }

    with pytest.raises(ValueError):
        load_aoi(empty_feature_collection)