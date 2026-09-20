"""Radiometric conversion: DN -> radiance -> TOA reflectance, plus atmospheric-correction hooks.

Radiance units: W m-2 sr-1 um-1. Solar irradiance units: W m-2 um-1.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

_SUN_RADIUS_M = 6.957e8
_AU_M = 1.495978707e11
_SUN_TEMP_K = 5778.0
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


def _per_band(values: np.ndarray | float, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"expected {n} per-band values, got shape {arr.shape}")
    return arr[:, None, None]


def dn_to_radiance(
    dn: np.ndarray, gain: np.ndarray | float, offset: np.ndarray | float = 0.0
) -> np.ndarray:
    """``L = gain * DN + offset`` applied per band on a ``(B, H, W)`` array."""
    b = dn.shape[0]
    return (dn * _per_band(gain, b) + _per_band(offset, b)).astype(np.float32)


def earth_sun_distance_au(day_of_year: int) -> float:
    """Earth-Sun distance in AU (low-order approximation, <0.1% error)."""
    return float(1.0 - 0.01672 * np.cos(np.deg2rad(0.9856 * (day_of_year - 4))))


def day_of_year(when: datetime | None) -> int | None:
    return when.timetuple().tm_yday if when is not None else None


def solar_irradiance(wavelengths_nm: np.ndarray, fwhm_nm: np.ndarray | None = None) -> np.ndarray:
    """Band-averaged exo-atmospheric irradiance (W m-2 um-1) at 1 AU.

    Approximated by a 5778 K blackbody diluted to 1 AU, averaged over each
    band's Gaussian SRF when ``fwhm_nm`` is given.
    """
    # TODO(Day 4): replace the blackbody with the Thuillier (2003) / sensor-provided ESUN table.
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    fwhm = np.full_like(wl, 0.0) if fwhm_nm is None else np.asarray(fwhm_nm, dtype=np.float64)
    fine = np.arange(250.0, 4000.0, 1.0)
    spec = _blackbody_irradiance(fine)
    out = np.empty_like(wl)
    for i, (c, f) in enumerate(zip(wl, fwhm, strict=True)):
        if f <= 0:
            out[i] = np.interp(c, fine, spec)
            continue
        sigma = f / 2.3548
        w = np.exp(-0.5 * ((fine - c) / sigma) ** 2)
        out[i] = float((w * spec).sum() / w.sum())
    return out


def _blackbody_irradiance(wl_nm: np.ndarray) -> np.ndarray:
    lam = wl_nm * 1e-9
    radiance = 2 * _H * _C**2 / lam**5 / np.expm1(_H * _C / (lam * _KB * _SUN_TEMP_K))
    dilution = (_SUN_RADIUS_M / _AU_M) ** 2
    return np.asarray(np.pi * radiance * dilution * 1e-6)  # W m-2 um-1 at 1 AU


def radiance_to_toa_reflectance(
    radiance: np.ndarray,
    esun: np.ndarray,
    sun_elevation_deg: float,
    doy: int | None = None,
) -> np.ndarray:
    """``rho = pi * L * d^2 / (ESUN * sin(elevation))`` per band."""
    if not 0.0 < sun_elevation_deg <= 90.0:
        raise ValueError(f"sun elevation must be in (0, 90], got {sun_elevation_deg}")
    d = earth_sun_distance_au(doy) if doy is not None else 1.0
    cos_sza = np.sin(np.deg2rad(sun_elevation_deg))
    b = radiance.shape[0]
    rho = np.pi * radiance * d**2 / (_per_band(esun, b) * cos_sza)
    return rho.astype(np.float32)


def clip_reflectance(
    refl: np.ndarray, invalid: np.ndarray | None = None, nodata: float = -1.0
) -> np.ndarray:
    """Clip to [0, 1]; NaN/inf and ``invalid`` (H, W) pixels become ``nodata``."""
    out = np.clip(refl, 0.0, 1.0).astype(np.float32, copy=False)
    bad = ~np.isfinite(refl).all(axis=0)
    if invalid is not None:
        bad |= invalid
    out[:, bad] = nodata
    return out


def dark_object_values(samples: np.ndarray, percentile: float = 0.5) -> np.ndarray:
    """Per-band dark-object estimate from a ``(B, N)`` sample of valid pixels."""
    return np.percentile(samples, percentile, axis=1).astype(np.float32)


def dark_object_subtraction(refl: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Simple DOS atmospheric correction hook (fallback when no L2A product exists)."""
    # TODO(Day 4): swap in a proper correction (e.g. Py6S / sensor L2A) when available.
    return np.clip(refl - dark[:, None, None], 0.0, None).astype(np.float32)
