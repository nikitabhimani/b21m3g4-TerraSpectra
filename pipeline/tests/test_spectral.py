import numpy as np
import pytest

from terraspectra_contracts.constants import N_BANDS, NODATA
from terraspectra_contracts.fixtures import _vegetation_spectrum
from terraspectra_pipeline.spectral import (
    CANONICAL_WAVELENGTHS,
    SpectralResampler,
    bad_band_mask,
    gaussian_srf_matrix,
    interpolate_bands,
    savgol_smooth,
)


def _sensor_grid(start: float = 420.0, stop: float = 2450.0, step: float = 6.5):
    wl = np.arange(start, stop, step)
    return wl, np.full(wl.shape, step * 1.1)


def test_bad_band_mask_flags_water_and_sensor_bands() -> None:
    wl = np.array([500.0, 1400.0, 1850.0, 2200.0, 2600.0])
    mask = bad_band_mask(wl, sensor_bad=np.array([True, False, False, False, False]))
    assert mask.tolist() == [True, True, True, False, True]


def test_srf_rows_normalised_where_covered() -> None:
    wl, fwhm = _sensor_grid()
    weights, covered = gaussian_srf_matrix(wl, fwhm, CANONICAL_WAVELENGTHS, 10.55)
    assert weights.shape == (N_BANDS, wl.size)
    np.testing.assert_allclose(weights[covered].sum(axis=1), 1.0, atol=1e-5)
    assert not covered[0]  # 400 nm lies outside a sensor starting at 420 nm
    assert covered[CANONICAL_WAVELENGTHS.searchsorted(700.0)]


def test_resampling_smooth_spectrum_preserves_shape() -> None:
    wl, fwhm = _sensor_grid()
    spectrum = _vegetation_spectrum(wl, 1.0)
    cube = np.broadcast_to(spectrum[:, None, None], (wl.size, 3, 4)).astype(np.float32)
    out = SpectralResampler(wl, fwhm)(cube)
    assert out.shape == (N_BANDS, 3, 4) and out.dtype == np.float32
    expected = _vegetation_spectrum(CANONICAL_WAVELENGTHS, 1.0)
    # Water-absorption bands are interpolated; everything stays close to the true curve.
    assert np.max(np.abs(out[:, 0, 0] - expected)) < 0.03
    assert np.corrcoef(out[:, 0, 0], expected)[0, 1] > 0.995


def test_resampler_propagates_nodata_and_ignores_nan_bad_bands() -> None:
    wl, fwhm = _sensor_grid()
    cube = np.full((wl.size, 2, 2), 0.3, dtype=np.float32)
    water = bad_band_mask(wl)
    cube[water] = np.nan
    cube[:, 0, 0] = NODATA
    out = SpectralResampler(wl, fwhm)(cube)
    assert np.all(out[:, 0, 0] == NODATA)
    np.testing.assert_allclose(out[:, 1, 1], 0.3, atol=1e-5)


def test_interpolate_bands_linear_and_edges() -> None:
    wl = np.array([1.0, 2.0, 3.0, 4.0])
    cube = np.array([10.0, 0.0, 30.0, 0.0])[:, None, None]
    out = interpolate_bands(cube, wl, np.array([False, True, False, True]))
    assert out[:, 0, 0].tolist() == [10.0, 20.0, 30.0, 30.0]
    with pytest.raises(ValueError):
        interpolate_bands(cube, wl, np.ones(4, dtype=bool))


def test_savgol_reduces_noise() -> None:
    rng = np.random.default_rng(0)
    clean = _vegetation_spectrum(CANONICAL_WAVELENGTHS, 1.0)[:, None, None]
    noisy = (clean + rng.normal(0, 0.01, clean.shape)).astype(np.float32)
    smoothed = savgol_smooth(noisy)
    assert np.abs(smoothed - clean).mean() < np.abs(noisy - clean).mean()
