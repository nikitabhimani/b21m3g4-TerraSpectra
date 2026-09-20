"""Band cleaning and FWHM-aware spectral resampling onto the canonical C1 grid."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import resources

import numpy as np
from scipy.signal import savgol_filter

from terraspectra_contracts.constants import NODATA, WAVELENGTHS_NM

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

# Atmospheric water-vapour absorption windows (nm); C1 says these are interpolated, not dropped.
WATER_ABSORPTION_NM: tuple[tuple[float, float], ...] = ((1340.0, 1460.0), (1790.0, 1960.0))

CANONICAL_WAVELENGTHS = np.asarray(WAVELENGTHS_NM, dtype=np.float64)


def _canonical_fwhm() -> float:
    try:
        text = resources.files("terraspectra_contracts").joinpath("wavelengths.json").read_text()
        return float(json.loads(text)["fwhm_nm"])
    except (KeyError, FileNotFoundError, ValueError):
        return float(np.median(np.diff(CANONICAL_WAVELENGTHS)))


CANONICAL_FWHM_NM = _canonical_fwhm()


def in_ranges(wavelengths: np.ndarray, ranges: Sequence[tuple[float, float]]) -> np.ndarray:
    """Boolean mask of wavelengths falling inside any ``(lo, hi)`` range."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    mask = np.zeros(wl.shape, dtype=bool)
    for lo, hi in ranges:
        mask |= (wl >= lo) & (wl <= hi)
    return mask


def bad_band_mask(
    wavelengths: np.ndarray,
    sensor_bad: np.ndarray | None = None,
    ranges: Sequence[tuple[float, float]] = WATER_ABSORPTION_NM,
    valid_range: tuple[float, float] = (380.0, 2510.0),
) -> np.ndarray:
    """True for bands to drop: sensor-flagged, water absorption, or outside ``valid_range``."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    mask = in_ranges(wl, ranges) | (wl < valid_range[0]) | (wl > valid_range[1])
    mask |= ~np.isfinite(wl)
    if sensor_bad is not None:
        mask |= np.asarray(sensor_bad, dtype=bool)
    return mask


def gaussian_srf_matrix(
    src_wl: np.ndarray,
    src_fwhm: np.ndarray,
    dst_wl: np.ndarray,
    dst_fwhm: np.ndarray | float,
    src_bad: np.ndarray | None = None,
    min_coverage: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(weights[Bdst, Bsrc], covered[Bdst])``.

    Each weight is the overlap integral of two Gaussian SRFs, so sensors coarser or finer than
    the target are both handled. Rows are normalised over valid source bands; target bands with
    too little overlap (``covered=False``) are left for spectral interpolation.
    """
    src_wl = np.asarray(src_wl, dtype=np.float64)
    dst_wl = np.asarray(dst_wl, dtype=np.float64)
    s_src = np.asarray(src_fwhm, dtype=np.float64) * FWHM_TO_SIGMA
    s_dst = np.broadcast_to(np.asarray(dst_fwhm, dtype=np.float64), dst_wl.shape) * FWHM_TO_SIGMA
    var = s_dst[:, None] ** 2 + s_src[None, :] ** 2
    diff = dst_wl[:, None] - src_wl[None, :]
    overlap = np.exp(-0.5 * diff**2 / var) / np.sqrt(2.0 * np.pi * var)
    if src_bad is not None:
        overlap[:, np.asarray(src_bad, dtype=bool)] = 0.0
    total = overlap.sum(axis=1)
    # Coverage relative to an ideal, densely sampled source (~1/spacing per nm).
    spacing = np.median(np.diff(np.sort(src_wl))) if src_wl.size > 1 else 1.0
    covered = total * spacing >= min_coverage
    weights = np.divide(overlap, total[:, None], out=np.zeros_like(overlap), where=covered[:, None])
    return weights.astype(np.float32), covered


def resample_cube(cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply a ``(Bdst, Bsrc)`` weight matrix to a ``(Bsrc, H, W)`` cube."""
    b, h, w = cube.shape
    flat = cube.reshape(b, h * w)
    return (weights @ flat).reshape(weights.shape[0], h, w).astype(np.float32)


def interpolate_bands(cube: np.ndarray, wavelengths: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``bad`` bands from their nearest good neighbours (edges: nearest)."""
    bad = np.asarray(bad, dtype=bool)
    if not bad.any():
        return cube
    good_idx = np.flatnonzero(~bad)
    if good_idx.size == 0:
        raise ValueError("cannot interpolate: every band is flagged bad")
    wl = np.asarray(wavelengths, dtype=np.float64)
    out = cube.copy()
    for b in np.flatnonzero(bad):
        pos = np.searchsorted(good_idx, b)
        if pos == 0:
            out[b] = cube[good_idx[0]]
        elif pos == good_idx.size:
            out[b] = cube[good_idx[-1]]
        else:
            lo, hi = good_idx[pos - 1], good_idx[pos]
            t = (wl[b] - wl[lo]) / (wl[hi] - wl[lo])
            out[b] = (1.0 - t) * cube[lo] + t * cube[hi]
    return out


def savgol_smooth(cube: np.ndarray, window: int = 7, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along the spectral axis."""
    if cube.shape[0] < window:
        return cube
    return savgol_filter(cube, window_length=window, polyorder=polyorder, axis=0).astype(np.float32)


class SpectralResampler:
    """Sensor bands -> canonical 200-band grid (bad-band removal, SRF resampling, gap fill)."""

    def __init__(
        self,
        src_wavelengths: np.ndarray,
        src_fwhm: np.ndarray,
        src_bad: np.ndarray | None = None,
        dst_wavelengths: np.ndarray = CANONICAL_WAVELENGTHS,
        dst_fwhm: float | np.ndarray = CANONICAL_FWHM_NM,
    ) -> None:
        self.src_wavelengths = np.asarray(src_wavelengths, dtype=np.float64)
        self.dst_wavelengths = np.asarray(dst_wavelengths, dtype=np.float64)
        self.src_bad = bad_band_mask(self.src_wavelengths, src_bad)
        self.weights, covered = gaussian_srf_matrix(
            self.src_wavelengths, src_fwhm, self.dst_wavelengths, dst_fwhm, self.src_bad
        )
        # C1: target bands inside water windows are always interpolated.
        self.fill_bands = ~covered | in_ranges(self.dst_wavelengths, WATER_ABSORPTION_NM)
        self.weights[self.fill_bands] = 0.0
        self._src_used = self.weights.any(axis=0)

    def __call__(self, cube: np.ndarray, nodata: float = NODATA) -> np.ndarray:
        """Resample a ``(Bsrc, H, W)`` reflectance cube; pixels with nodata stay nodata."""
        if cube.shape[0] != self.src_wavelengths.size:
            raise ValueError(f"expected {self.src_wavelengths.size} bands, got {cube.shape[0]}")
        invalid = (cube[self._src_used] == nodata).any(axis=0)
        invalid |= ~np.isfinite(cube[self._src_used]).all(axis=0)
        src = np.where(invalid[None], 0.0, cube).astype(np.float32)
        src[~self._src_used] = 0.0  # unused bands may hold NaN; 0 * NaN would leak
        out = resample_cube(src, self.weights)
        out = interpolate_bands(out, self.dst_wavelengths, self.fill_bands)
        out[:, invalid] = nodata
        return out
