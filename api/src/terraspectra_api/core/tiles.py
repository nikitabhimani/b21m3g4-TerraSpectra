"""XYZ heatmap tiles (256x256 RGBA PNG) rendered from a job's risk COG."""

from __future__ import annotations

import threading
from collections import OrderedDict
from functools import cache
from pathlib import Path

import numpy as np
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.io import Reader
from rio_tiler.utils import render

TILE_SIZE = 256
ALPHA = 210
# Risk score (1 - p_healthy) stops: green → yellow → red.
_STOPS = np.array([0.0, 0.5, 1.0], dtype=np.float32)
_COLORS = np.array([[26, 152, 80], [254, 224, 54], [215, 48, 39]], dtype=np.float32)


def colorize(risk: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map risk in [0, 1] to ``(rgb uint8[3,H,W], alpha uint8[H,W])``."""
    r = np.clip(np.nan_to_num(risk), 0.0, 1.0)
    rgb = np.stack([np.interp(r, _STOPS, _COLORS[:, i]) for i in range(3)]).astype(np.uint8)
    alpha = np.where(valid, ALPHA, 0).astype(np.uint8)
    return rgb, alpha


@cache
def transparent_tile() -> bytes:
    """A fully transparent 256x256 PNG."""
    zeros = np.zeros((3, TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    return render(zeros, mask=np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8), img_format="PNG")


class TileRenderer:
    """Renders and LRU-caches tiles keyed by ``(path, mtime, z, x, y)``.

    TODO(Day 9): shared cache (Redis) / CDN in front for multi-replica deployments.
    """

    def __init__(self, cache_size: int = 1024) -> None:
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int, int, int, int], bytes] = OrderedDict()
        self._lock = threading.Lock()

    def render(self, path: Path, z: int, x: int, y: int) -> bytes:
        key = (str(path), path.stat().st_mtime_ns, z, x, y)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        data = self._render(path, z, x, y)
        if self.cache_size > 0:
            with self._lock:
                self._cache[key] = data
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return data

    @staticmethod
    def _render(path: Path, z: int, x: int, y: int) -> bytes:
        with Reader(str(path)) as src:
            if not src.tile_exists(x, y, z):
                return transparent_tile()
            try:
                img = src.tile(x, y, z, tilesize=TILE_SIZE, indexes=1)
            except TileOutsideBounds:
                return transparent_tile()
        band = img.array[0]
        p_healthy = np.ma.getdata(band)
        valid = ~np.ma.getmaskarray(band) & (p_healthy >= 0.0)
        rgb, alpha = colorize(1.0 - p_healthy, valid)
        return render(rgb, mask=alpha, img_format="PNG")
