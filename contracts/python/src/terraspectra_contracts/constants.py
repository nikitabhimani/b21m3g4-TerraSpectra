"""Contract C1 (cube) and C2 (model I/O) constants."""

from __future__ import annotations

import json
from enum import IntEnum
from importlib import resources

CONTRACT_VERSION = "1.0.0"

# C1 - cube format
N_BANDS = 200
NODATA = -1.0
REFLECTANCE_RANGE = (0.0, 1.0)
WAVELENGTH_TAG = "wavelength_nm"

# C2 - model I/O
WINDOW_SIZE = 64
N_CLASSES = 4
MAX_ONSET_DAYS = 30.0


class RiskClass(IntEnum):
    HEALTHY = 0
    EARLY_STRESS = 1
    HIGH_BLIGHT_RISK = 2
    VISIBLE_DISEASE = 3


CLASS_NAMES: dict[int, str] = {c.value: c.name.lower() for c in RiskClass}


def _load_wavelengths() -> tuple[float, ...]:
    data = json.loads(resources.files(__package__).joinpath("wavelengths.json").read_text())
    wl = tuple(float(w) for w in data["wavelengths"])
    if len(wl) != N_BANDS:
        raise RuntimeError(f"wavelengths.json has {len(wl)} bands, expected {N_BANDS}")
    return wl


WAVELENGTHS_NM: tuple[float, ...] = _load_wavelengths()
