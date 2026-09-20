"""Feathered overlap blending of window predictions and risk COG writing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from terraspectra_contracts import CLASS_NAMES, CONTRACT_VERSION, N_CLASSES, NODATA, WINDOW_SIZE

MEMMAP_THRESHOLD_PIXELS = 64_000_000  # ~1.3 GB of accumulators


def feather_weights(size: int = WINDOW_SIZE, overlap: int = 0) -> np.ndarray:
    """2D weights ramping linearly over ``overlap`` pixels at each edge (always > 0)."""
    idx = np.arange(size, dtype=np.float32) + 0.5
    if overlap > 0:
        ramp = np.minimum(1.0, np.minimum(idx, size - idx) / overlap)
    else:
        ramp = np.ones(size, dtype=np.float32)
    return np.outer(ramp, ramp).astype(np.float32)


def _alloc(shape: tuple[int, ...], scratch: Path | None) -> np.ndarray:
    if scratch is not None and int(np.prod(shape[-2:])) >= MEMMAP_THRESHOLD_PIXELS:
        f = tempfile.NamedTemporaryFile(dir=scratch, suffix=".npy", delete=False)  # noqa: SIM115
        f.close()
        return np.lib.format.open_memmap(f.name, mode="w+", dtype=np.float32, shape=shape)
    return np.zeros(shape, dtype=np.float32)


class Stitcher:
    """Accumulates weighted probabilities/onset and normalises by total weight."""

    def __init__(
        self,
        height: int,
        width: int,
        window_size: int = WINDOW_SIZE,
        overlap: int = 0,
        n_classes: int = N_CLASSES,
        scratch_dir: Path | None = None,
    ) -> None:
        self.height, self.width = height, width
        self.weights_2d = feather_weights(window_size, overlap)
        self.probs = _alloc((n_classes, height, width), scratch_dir)
        self.onset = _alloc((height, width), scratch_dir)
        self.weight = _alloc((height, width), scratch_dir)

    def add(
        self,
        probs: np.ndarray,
        onset: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> None:
        """Blend ``probs[N,C,S,S]`` / ``onset[N,1,S,S]`` placed at ``(rows[i], cols[i])``."""
        for i in range(probs.shape[0]):
            r, c = int(rows[i]), int(cols[i])
            h = min(probs.shape[2], self.height - r)
            w = min(probs.shape[3], self.width - c)
            wgt = self.weights_2d[:h, :w]
            if valid is not None:
                wgt = wgt * valid[i, :h, :w]
            self.probs[:, r : r + h, c : c + w] += probs[i, :, :h, :w] * wgt
            self.onset[r : r + h, c : c + w] += onset[i, 0, :h, :w] * wgt
            self.weight[r : r + h, c : c + w] += wgt

    def finalize(
        self, valid_mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(probs[C,H,W], onset[H,W], valid[H,W])`` with ``NODATA`` where invalid."""
        valid = self.weight > 0
        if valid_mask is not None:
            valid &= valid_mask
        safe = np.where(valid, self.weight, 1.0)
        probs = np.divide(self.probs, safe, out=self.probs)
        onset = np.divide(self.onset, safe, out=self.onset)
        probs[:, ~valid] = NODATA
        onset[~valid] = NODATA
        return probs, onset, valid


def write_risk_cog(
    path: Path,
    probs: np.ndarray,
    onset: np.ndarray,
    crs: Any,
    transform: Any,
    extra_tags: dict[str, str] | None = None,
) -> Path:
    """Write the 5-band risk COG (bands 1-4 class probabilities, band 5 days_to_onset)."""
    n_classes, height, width = probs.shape
    profile = {
        "driver": "COG",
        "dtype": "float32",
        "count": n_classes + 1,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "ZSTD",
        "blocksize": 256,
        "BIGTIFF": "IF_SAFER",
    }
    tmp = path.with_suffix(".tif.part")
    with rasterio.open(tmp, "w", **profile) as dst:
        for b in range(n_classes):
            dst.write(probs[b].astype(np.float32, copy=False), b + 1)
            dst.set_band_description(b + 1, f"p_{CLASS_NAMES[b]}")
        dst.write(onset.astype(np.float32, copy=False), n_classes + 1)
        dst.set_band_description(n_classes + 1, "days_to_onset")
        dst.update_tags(contract_version=CONTRACT_VERSION, **(extra_tags or {}))
    tmp.replace(path)
    return path
