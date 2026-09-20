"""Spectral vegetation-stress indices on the canonical grid.

All functions accept reflectance with the spectral axis first (``[B, ...]``, e.g. ``[B]`` or
``[B, H, W]``) and return arrays shaped like the trailing axes.
"""

from __future__ import annotations

import numpy as np
from terraspectra_contracts import WAVELENGTHS_NM

_WL = np.asarray(WAVELENGTHS_NM, dtype=np.float64)
_EPS = 1e-6


def band_at(cube: np.ndarray, nm: float) -> np.ndarray:
    """Reflectance at ``nm`` by linear interpolation between neighbouring canonical bands."""
    i = int(np.clip(np.searchsorted(_WL, nm) - 1, 0, len(_WL) - 2))
    t = (nm - _WL[i]) / (_WL[i + 1] - _WL[i])
    return np.asarray((1 - t) * cube[i] + t * cube[i + 1])


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray((a - b) / (a + b + _EPS))


def ndvi(cube: np.ndarray) -> np.ndarray:
    """Normalised difference vegetation index (800, 670 nm)."""
    return _nd(band_at(cube, 800), band_at(cube, 670))


def ndre(cube: np.ndarray) -> np.ndarray:
    """Normalised difference red-edge index (790, 720 nm)."""
    return _nd(band_at(cube, 790), band_at(cube, 720))


def pri(cube: np.ndarray) -> np.ndarray:
    """Photochemical reflectance index (531, 570 nm)."""
    return _nd(band_at(cube, 531), band_at(cube, 570))


def cci(cube: np.ndarray) -> np.ndarray:
    """Chlorophyll/carotenoid index (531, 645 nm; Gamon et al. 2016)."""
    return _nd(band_at(cube, 531), band_at(cube, 645))


def ndwi(cube: np.ndarray) -> np.ndarray:
    """Gao normalised difference water index (860, 1240 nm)."""
    return _nd(band_at(cube, 860), band_at(cube, 1240))


def red_edge_position(cube: np.ndarray) -> np.ndarray:
    """Red-edge position in nm (Guyot & Baret linear four-point method)."""
    r670, r700, r740, r780 = (band_at(cube, w) for w in (670, 700, 740, 780))
    r_re = (r670 + r780) / 2
    rep = 700.0 + 40.0 * (r_re - r700) / (r740 - r700 + _EPS)
    return np.asarray(np.clip(rep, 670.0, 780.0))
