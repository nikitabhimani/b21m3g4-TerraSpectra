"""Tests for terraspectra_pipeline.geo — pure functions, no real raster files needed."""

import numpy as np
from rasterio.crs import CRS

from terraspectra_pipeline.geo import is_utm, nodata_mask, utm_crs_for_lonlat


def test_utm_crs_jaipur():
    """Jaipur (~75.8E, 26.9N) should fall in UTM zone 43N (EPSG:32643)."""
    crs = utm_crs_for_lonlat(75.8, 26.9)
    assert crs.to_epsg() == 32643


def test_utm_crs_southern_hemisphere():
    """Negative latitude should give a 327xx (southern) EPSG code, not 326xx (northern)."""
    crs = utm_crs_for_lonlat(18.4, -33.9)  # Cape Town
    epsg = crs.to_epsg()
    assert epsg is not None
    assert 32701 <= epsg <= 32760


def test_utm_crs_zone_boundaries():
    """Longitude 0 should sit at the start of zone 31, not zone 30 or 32."""
    crs = utm_crs_for_lonlat(0.5, 10.0)
    assert crs.to_epsg() == 32631


def test_is_utm_true_for_utm_crs():
    utm = CRS.from_epsg(32643)  # UTM zone 43N
    assert is_utm(utm) is True


def test_is_utm_false_for_wgs84():
    wgs84 = CRS.from_epsg(4326)  # plain lat/lon, not UTM
    assert is_utm(wgs84) is False


def test_is_utm_false_for_none():
    assert is_utm(None) is False


def test_nodata_mask_flags_nodata_value():
    """A pixel equal to the nodata value in any band should be masked True."""
    cube = np.ones((3, 2, 2), dtype=np.float32)
    cube[:, 0, 0] = -1.0  # nodata pixel
    mask = nodata_mask(cube, nodata=-1.0)
    assert mask[0, 0] == True   # noqa: E712
    assert mask[0, 1] == False  # noqa: E712
    assert mask[1, 0] == False  # noqa: E712
    assert mask[1, 1] == False  # noqa: E712


def test_nodata_mask_flags_nan():
    """A NaN in any band should also be masked True, even without a nodata match."""
    cube = np.ones((2, 2, 2), dtype=np.float32)
    cube[0, 1, 1] = np.nan
    mask = nodata_mask(cube, nodata=-1.0)
    assert mask[1, 1] == True   # noqa: E712
    assert mask[0, 0] == False  # noqa: E712