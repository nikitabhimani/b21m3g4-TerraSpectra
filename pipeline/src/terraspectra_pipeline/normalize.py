"""Per-band robust (p2-p98) scaling with statistics sampled from random windows."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from terraspectra_contracts.constants import NODATA


@dataclass
class BandStats:
    """Robust per-band statistics (stored as lists for JSON)."""

    low: list[float]
    high: list[float]
    percentiles: tuple[float, float] = (2.0, 98.0)
    n_pixels: int = 0
    wavelengths_nm: list[float] | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> BandStats:
        data = json.loads(Path(path).read_text())
        data["percentiles"] = tuple(data["percentiles"])
        return cls(**data)


def sample_windows(
    height: int, width: int, size: int = 256, n: int = 32, seed: int = 0
) -> Iterator[Window]:
    """Random windows (clipped to the raster) for statistics sampling."""
    rng = np.random.default_rng(seed)
    h, w = min(size, height), min(size, width)
    for _ in range(n):
        r = int(rng.integers(0, height - h + 1))
        c = int(rng.integers(0, width - w + 1))
        yield Window(c, r, w, h)


def compute_band_stats(
    blocks: Iterable[np.ndarray],
    nodata: float = NODATA,
    percentiles: tuple[float, float] = (2.0, 98.0),
    max_pixels: int = 500_000,
    seed: int = 0,
) -> BandStats:
    """Percentiles per band over valid pixels of ``(B, h, w)`` blocks (subsampled)."""
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for block in blocks:
        flat = block.reshape(block.shape[0], -1)
        valid = np.isfinite(flat).all(axis=0) & (flat != nodata).all(axis=0)
        chunks.append(flat[:, valid])
    if not chunks or sum(c.shape[1] for c in chunks) == 0:
        raise ValueError("no valid pixels to compute statistics from")
    samples = np.concatenate(chunks, axis=1)
    if samples.shape[1] > max_pixels:
        samples = samples[:, rng.choice(samples.shape[1], max_pixels, replace=False)]
    lo, hi = np.percentile(samples, percentiles, axis=1)
    return BandStats(
        low=lo.astype(float).tolist(),
        high=hi.astype(float).tolist(),
        percentiles=percentiles,
        n_pixels=int(samples.shape[1]),
    )


def compute_file_stats(
    path: str | Path, n_windows: int = 32, window_size: int = 256, seed: int = 0
) -> BandStats:
    """Sample windows of a C1 cube and compute robust stats."""
    with rasterio.open(path) as ds:
        nodata = ds.nodata if ds.nodata is not None else NODATA
        wins = sample_windows(ds.height, ds.width, window_size, n_windows, seed)
        stats = compute_band_stats(
            (ds.read(window=w, out_dtype="float32") for w in wins), nodata=nodata, seed=seed
        )
        tags = [ds.tags(i).get("wavelength_nm") for i in range(1, ds.count + 1)]
        if all(tags):
            stats.wavelengths_nm = [float(t) for t in tags if t is not None]
    return stats


def apply_scaling(
    cube: np.ndarray, stats: BandStats, nodata: float = NODATA, clip: bool = True
) -> np.ndarray:
    """``(x - low) / (high - low)`` per band, clipped to [0, 1]; nodata pixels preserved."""
    lo = np.asarray(stats.low, dtype=np.float32)[:, None, None]
    hi = np.asarray(stats.high, dtype=np.float32)[:, None, None]
    if lo.shape[0] != cube.shape[0]:
        raise ValueError(f"stats have {lo.shape[0]} bands, cube has {cube.shape[0]}")
    span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
    out = (cube - lo) / span
    if clip:
        out = np.clip(out, 0.0, 1.0)
    invalid = (cube == nodata).any(axis=0)
    out[:, invalid] = nodata
    return out.astype(np.float32)
