"""Spectral resampling of arbitrary sensors onto the canonical C1 wavelength grid."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from terraspectra_contracts import WAVELENGTHS_NM

log = logging.getLogger(__name__)

CANONICAL_WL = np.asarray(WAVELENGTHS_NM, dtype=np.float64)
CANONICAL_FWHM = float(np.mean(np.diff(CANONICAL_WL)))


def _as_matrix(cube: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape ``[B, ...]`` to ``[B, P]`` and return the trailing shape."""
    return cube.reshape(cube.shape[0], -1), cube.shape[1:]


def gaussian_srf_matrix(
    src_wl: np.ndarray,
    dst_wl: np.ndarray = CANONICAL_WL,
    dst_fwhm: float | np.ndarray = CANONICAL_FWHM,
) -> np.ndarray:
    """Row-normalised ``[len(dst), len(src)]`` weights of Gaussian target SRFs sampled at src."""
    fwhm = np.broadcast_to(np.asarray(dst_fwhm, dtype=np.float64), dst_wl.shape)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    w = np.exp(-0.5 * ((src_wl[None, :] - dst_wl[:, None]) / sigma[:, None]) ** 2)
    norm = w.sum(axis=1, keepdims=True)
    return np.asarray(np.where(norm > 1e-12, w / np.maximum(norm, 1e-12), 0.0))


def resample_to_canonical(
    cube: np.ndarray,
    src_wavelengths: np.ndarray | list[float],
    method: Literal["linear", "gaussian"] = "linear",
    fill: Literal["edge", "zero"] = "edge",
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``cube[B_src, ...]`` to the 200-band canonical grid.

    Returns ``(resampled[200, ...] float32, covered[200] bool)``; ``covered`` flags canonical
    bands inside the source's spectral range. Outside the range values are edge-extended
    (``fill="edge"``) or zeroed. Interior gaps (e.g. removed water bands) are bridged by
    interpolation, as C1 requires.
    """
    src = np.asarray(src_wavelengths, dtype=np.float64)
    if cube.shape[0] != src.size:
        raise ValueError(f"cube has {cube.shape[0]} bands but {src.size} wavelengths given")
    order = np.argsort(src)
    src = src[order]
    mat, trailing = _as_matrix(np.asarray(cube, dtype=np.float64)[order])
    covered = (src[0] - CANONICAL_FWHM / 2 <= CANONICAL_WL) & (
        src[-1] + CANONICAL_FWHM / 2 >= CANONICAL_WL
    )

    if method == "linear":
        # Vectorised np.interp over pixels: per canonical band find bracketing source bands.
        idx = np.clip(np.searchsorted(src, CANONICAL_WL) - 1, 0, src.size - 2)
        lo, hi = src[idx], src[idx + 1]
        t = np.clip((CANONICAL_WL - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
        out = (1 - t)[:, None] * mat[idx] + t[:, None] * mat[idx + 1]
    elif method == "gaussian":
        weights = gaussian_srf_matrix(src)
        out = weights @ mat
        # Canonical bands in interior gaps get no SRF support: fall back to linear there.
        nearest = np.min(np.abs(src[None, :] - CANONICAL_WL[:, None]), axis=1)
        empty = covered & (nearest > CANONICAL_FWHM)
        if empty.any():
            lin, _ = resample_to_canonical(cube, src_wavelengths, "linear", fill)
            out[empty] = lin.reshape(len(CANONICAL_WL), -1)[empty]
    else:  # pragma: no cover - guarded by Literal
        raise ValueError(method)

    if fill == "zero":
        out[~covered] = 0.0
    elif not covered.all():
        # Edge-extend: linear method with clipped t already repeats the edge bands.
        log.debug("source covers %d/%d canonical bands", covered.sum(), covered.size)
        first, last = np.argmax(covered), len(covered) - 1 - np.argmax(covered[::-1])
        out[: int(first)] = out[int(first)]
        out[int(last) + 1 :] = out[int(last)]
    return out.reshape((len(CANONICAL_WL), *trailing)).astype(np.float32), covered
