"""Plain data containers shared between the data, synth and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LabelledScene:
    """A reflectance scene plus dense targets.

    ``class_map`` uses ``-1`` for "ignore"; ``onset_map`` is days-to-visible-symptoms (30 = none
    within the horizon); ``valid_mask`` is False for nodata pixels (their reflectance is 0).
    """

    cube: np.ndarray  # [200, H, W] float32
    class_map: np.ndarray  # [H, W] int64
    onset_map: np.ndarray  # [H, W] float32
    valid_mask: np.ndarray  # [H, W] bool

    def __post_init__(self) -> None:
        hw = self.cube.shape[1:]
        for name in ("class_map", "onset_map", "valid_mask"):
            if getattr(self, name).shape != hw:
                raise ValueError(f"{name} shape {getattr(self, name).shape} != cube {hw}")

    @property
    def shape(self) -> tuple[int, int]:
        """Spatial ``(H, W)``."""
        return int(self.cube.shape[1]), int(self.cube.shape[2])
