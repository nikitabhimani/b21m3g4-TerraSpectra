"""Bounded-memory windowed reader that turns a C1 cube into batches of model windows.

Windows are ``window_size`` squares placed on a regular grid with stride
``window_size - overlap``; the last window on each axis is shifted to touch the
raster edge, so every pixel is covered. Rasters smaller than a window are
padded (padding is marked invalid).
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import rasterio
from rasterio.windows import Window

from terraspectra_contracts import NODATA, WINDOW_SIZE

T = TypeVar("T")


@dataclass(frozen=True)
class Batch:
    """A batch of windows ready for the model.

    ``data``: float32 ``[N, B, S, S]`` with invalid pixels set to 0.
    ``valid``: bool ``[N, S, S]`` (False for nodata, padding or outside the AOI).
    ``rows``/``cols``: int ``[N]`` top-left raster offsets of each window.
    """

    data: np.ndarray
    valid: np.ndarray
    rows: np.ndarray
    cols: np.ndarray

    def __len__(self) -> int:
        return int(self.data.shape[0])


def window_offsets(size: int, window: int = WINDOW_SIZE, overlap: int = 0) -> list[int]:
    """Top-left offsets along one axis so that ``[0, size)`` is fully covered."""
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    if size <= window:
        return [0]
    stride = window - overlap
    offsets = list(range(0, size - window + 1, stride))
    if offsets[-1] != size - window:
        offsets.append(size - window)
    return offsets


def iter_windows(
    height: int, width: int, window: int = WINDOW_SIZE, overlap: int = 0
) -> Iterator[tuple[int, int]]:
    """Yield ``(row, col)`` offsets of every window, row-major."""
    cols = window_offsets(width, window, overlap)
    for row in window_offsets(height, window, overlap):
        for col in cols:
            yield row, col


class RasterChunker:
    """Iterate a raster as model-ready batches without loading it whole.

    Reads one horizontal strip per group of up to ``batch_size`` windows, so peak
    memory is roughly ``2 * batch_size * bands * window**2 * 4`` bytes.
    """

    def __init__(
        self,
        path: str | Path,
        window_size: int = WINDOW_SIZE,
        overlap: int = 16,
        batch_size: int = 32,
        aoi_mask: np.ndarray | None = None,
        prefetch: int = 0,
    ) -> None:
        self.path = Path(path)
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.prefetch = prefetch
        with rasterio.open(self.path) as src:
            self.height, self.width, self.count = src.height, src.width, src.count
            self.nodata = NODATA if src.nodata is None else float(src.nodata)
        if aoi_mask is not None and aoi_mask.shape != (self.height, self.width):
            raise ValueError("aoi_mask shape must match the raster")
        self.aoi_mask = aoi_mask
        self.row_offsets = window_offsets(self.height, window_size, overlap)
        self.col_offsets = window_offsets(self.width, window_size, overlap)

    def __len__(self) -> int:
        """Total number of windows (including ones later skipped as fully invalid)."""
        return len(self.row_offsets) * len(self.col_offsets)

    def batches(self) -> Iterator[Batch]:
        """Yield batches; windows with no valid pixel are skipped (but still counted)."""
        gen = self._batches()
        return _prefetch(gen, self.prefetch) if self.prefetch > 0 else gen

    def _batches(self) -> Iterator[Batch]:
        s = self.window_size
        pending: list[tuple[np.ndarray, np.ndarray, int, int]] = []
        with rasterio.open(self.path) as src:
            for row in self.row_offsets:
                for i in range(0, len(self.col_offsets), self.batch_size):
                    group = self.col_offsets[i : i + self.batch_size]
                    c0, c1 = group[0], group[-1] + s
                    strip = self._read(src, row, c0, c1 - c0)
                    strip_valid = np.isfinite(strip).all(axis=0) & (strip != self.nodata).all(
                        axis=0
                    )
                    if self.aoi_mask is not None:
                        aoi = np.zeros((s, c1 - c0), dtype=bool)
                        part = self.aoi_mask[row : row + s, c0:c1]
                        aoi[: part.shape[0], : part.shape[1]] = part
                        strip_valid &= aoi
                    for col in group:
                        valid = strip_valid[:, col - c0 : col - c0 + s]
                        if not valid.any():
                            continue
                        data = np.where(valid, strip[:, :, col - c0 : col - c0 + s], 0.0)
                        pending.append((data.astype(np.float32, copy=False), valid, row, col))
                        if len(pending) == self.batch_size:
                            yield _collate(pending)
                            pending = []
        if pending:
            yield _collate(pending)

    def _read(self, src: rasterio.DatasetReader, row: int, col: int, width: int) -> np.ndarray:
        """Read a ``window_size``-tall strip; pad with nodata only past the raster edge.

        (``boundless=True`` goes through a slow VRT path, so it is avoided when possible.)
        """
        s = self.window_size
        h = min(s, self.height - row)
        w = min(width, self.width - col)
        window = Window(col, row, w, h)
        data = src.read(window=window).astype(np.float32, copy=False)
        if (h, w) == (s, width):
            return data
        out = np.full((data.shape[0], s, width), self.nodata, dtype=np.float32)
        out[:, :h, :w] = data
        return out


def _collate(items: list[tuple[np.ndarray, np.ndarray, int, int]]) -> Batch:
    return Batch(
        data=np.stack([it[0] for it in items]),
        valid=np.stack([it[1] for it in items]),
        rows=np.array([it[2] for it in items], dtype=np.int64),
        cols=np.array([it[3] for it in items], dtype=np.int64),
    )


_SENTINEL = object()


def _prefetch(it: Iterator[T], depth: int) -> Iterator[T]:
    """Run ``it`` in a background thread so disk I/O overlaps with GPU compute.

    TODO(Day 10): swap for a multiprocessing reader pool for very large scenes.
    """
    q: queue.Queue[object] = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def worker() -> None:
        try:
            for item in it:
                while not stop.is_set():
                    try:
                        q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:
            q.put(exc)
            return
        q.put(_SENTINEL)

    thread = threading.Thread(target=worker, name="chunk-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is _SENTINEL:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]
    finally:
        stop.set()
        thread.join(timeout=5)
