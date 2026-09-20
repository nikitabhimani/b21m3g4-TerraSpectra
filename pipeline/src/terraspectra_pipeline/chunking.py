"""Window tiling and feathered stitching (self-contained: numpy + rasterio Window only)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from rasterio.windows import Window


def _offsets(length: int, size: int, stride: int, snap: bool) -> list[int]:
    if length <= size:
        return [0]
    if not snap:
        return list(range(0, length, stride))
    offs = list(range(0, length - size + 1, stride))
    if offs[-1] != length - size:
        offs.append(length - size)  # snap a full-size final window to the edge
    return offs


def iter_windows(
    height: int, width: int, size: int = 64, overlap: int = 16, snap_edges: bool = True
) -> Iterator[Window]:
    """Yield windows covering every pixel.

    With ``snap_edges`` (model inference) every window is ``size`` px (smaller only if the raster
    is) and the last one is shifted back to the edge. Without it (block I/O) the last window is
    truncated instead, so ``overlap=0`` gives a disjoint tiling.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require size > 0 and 0 <= overlap < size")
    if height <= 0 or width <= 0:
        return
    stride = size - overlap
    for row in _offsets(height, size, stride, snap_edges):
        for col in _offsets(width, size, stride, snap_edges):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def iter_blocks(height: int, width: int, size: int = 512) -> Iterator[Window]:
    """Disjoint block tiling (edge blocks truncated) for windowed I/O."""
    return iter_windows(height, width, size=size, overlap=0, snap_edges=False)


def _ramp(n: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones(n, dtype=np.float32)
    i = np.arange(n, dtype=np.float32)
    return np.minimum(1.0, np.minimum((i + 0.5) / overlap, (n - i - 0.5) / overlap))


def feather_weights(size: int | tuple[int, int], overlap: int) -> np.ndarray:
    """Strictly positive (h, w) weights ramping linearly to the window border over ``overlap``."""
    h, w = (size, size) if isinstance(size, int) else size
    return np.outer(_ramp(h, overlap), _ramp(w, overlap)).astype(np.float32)


class Stitcher:
    """Accumulate weighted per-window predictions ``(C, h, w)`` into a blended ``(C, H, W)``."""

    def __init__(
        self, channels: int, height: int, width: int, overlap: int = 16, dtype: type = np.float32
    ) -> None:
        self.shape = (channels, height, width)
        self.overlap = overlap
        self._acc: np.ndarray = np.zeros(self.shape, dtype=dtype)
        self._wsum: np.ndarray = np.zeros((height, width), dtype=dtype)

    def add(self, window: Window, pred: np.ndarray, valid: np.ndarray | None = None) -> None:
        """Add a prediction for ``window``; ``valid`` (h, w) excludes pixels from blending."""
        r, c = int(window.row_off), int(window.col_off)
        h, w = pred.shape[-2:]
        if (h, w) != (int(window.height), int(window.width)):
            raise ValueError(f"prediction {pred.shape} does not match window {window}")
        wts = feather_weights((h, w), self.overlap)
        if valid is not None:
            wts = wts * valid.astype(wts.dtype)
        self._acc[:, r : r + h, c : c + w] += pred * wts
        self._wsum[r : r + h, c : c + w] += wts

    @property
    def coverage(self) -> np.ndarray:
        """Boolean (H, W) mask of pixels that received any weight."""
        return self._wsum > 0

    def result(self, fill: float = np.nan) -> np.ndarray:
        """Return the blended array; uncovered pixels get ``fill``."""
        out = np.full(self.shape, fill, dtype=self._acc.dtype)
        cov = self.coverage
        np.divide(self._acc, self._wsum, out=out, where=cov[None])
        return out
