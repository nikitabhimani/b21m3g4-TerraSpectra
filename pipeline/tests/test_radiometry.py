from datetime import datetime

import numpy as np
import pytest

from terraspectra_pipeline.radiometry import (
    clip_reflectance,
    dark_object_subtraction,
    day_of_year,
    dn_to_radiance,
    earth_sun_distance_au,
    radiance_to_toa_reflectance,
    solar_irradiance,
)


def test_dn_to_radiance_per_band() -> None:
    dn = np.full((2, 2, 2), 400.0)
    out = dn_to_radiance(dn, np.array([1 / 40, 1 / 80]), np.array([0.0, 1.0]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0, 0], [10.0, 6.0])
    with pytest.raises(ValueError):
        dn_to_radiance(dn, np.ones(3))


def test_earth_sun_distance_extremes() -> None:
    assert earth_sun_distance_au(4) == pytest.approx(0.9833, abs=1e-3)  # perihelion
    assert earth_sun_distance_au(186) == pytest.approx(1.0167, abs=1e-3)  # aphelion
    assert day_of_year(datetime(2024, 2, 1)) == 32
    assert day_of_year(None) is None


def test_solar_irradiance_is_plausible() -> None:
    esun = solar_irradiance(np.array([500.0, 1000.0, 2200.0]), np.array([10.0, 10.0, 10.0]))
    assert 1600 < esun[0] < 2200
    assert esun[0] > esun[1] > esun[2] > 0


def test_toa_reflectance_roundtrip() -> None:
    esun = np.array([1800.0, 700.0])
    rho = np.array([0.1, 0.4])[:, None, None]
    elev, doy = 50.0, 120
    d = earth_sun_distance_au(doy)
    radiance = rho * esun[:, None, None] * np.sin(np.deg2rad(elev)) / (np.pi * d**2)
    out = radiance_to_toa_reflectance(radiance, esun, elev, doy)
    np.testing.assert_allclose(out, rho, rtol=1e-5)
    with pytest.raises(ValueError):
        radiance_to_toa_reflectance(radiance, esun, -5.0)


def test_clip_and_invalid_mask() -> None:
    refl = np.array([[[1.5, -0.2], [np.nan, 0.5]]], dtype=np.float32)
    invalid = np.array([[False, True], [False, False]])
    out = clip_reflectance(refl, invalid)
    assert out[0].tolist() == [[1.0, -1.0], [-1.0, 0.5]]


def test_dark_object_subtraction_non_negative() -> None:
    refl = np.full((2, 1, 1), 0.05, dtype=np.float32)
    out = dark_object_subtraction(refl, np.array([0.02, 0.1], dtype=np.float32))
    np.testing.assert_allclose(out[:, 0, 0], [0.03, 0.0], atol=1e-6)
