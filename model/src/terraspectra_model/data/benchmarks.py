"""Loaders for the classic public hyperspectral benchmarks (Indian Pines, Salinas, Pavia U).

Download the ``.mat`` files from the EHU GIC page (see ``BENCHMARKS[...].urls`` or the README)
into ``data_dir``. Wavelengths are approximated from the sensor specifications; the cubes are
scaled to reflectance (/10000), clipped to [0, 1] and resampled to the canonical 200-band grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from terraspectra_model.data.spectral import resample_to_canonical

log = logging.getLogger(__name__)

_EHU = "https://www.ehu.eus/ccwintco/uploads"


def _aviris_wavelengths(n_total: int, removed_1based: list[int]) -> np.ndarray:
    """AVIRIS nominal grid (``n_total`` bands over ~400-2500 nm) minus the removed bands."""
    wl = np.linspace(400.0, 2500.0, n_total)
    keep = np.setdiff1d(np.arange(n_total), np.asarray(removed_1based) - 1)
    return np.asarray(wl[keep])


def _names(spaced: str) -> dict[int, str]:
    return dict(enumerate(spaced.split()))


def _ranges(*spans: tuple[int, int]) -> list[int]:
    return [i for a, b in spans for i in range(a, b + 1)]


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static description of one benchmark scene."""

    name: str
    cube_file: str
    cube_key: str
    gt_file: str
    gt_key: str
    wavelengths: np.ndarray
    class_names: dict[int, str]
    vegetation_classes: frozenset[int]
    scale: float = 10000.0
    urls: tuple[str, ...] = field(default_factory=tuple)


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "indian_pines": BenchmarkSpec(
        name="indian_pines",
        cube_file="Indian_pines_corrected.mat",
        cube_key="indian_pines_corrected",
        gt_file="Indian_pines_gt.mat",
        gt_key="indian_pines_gt",
        # Indian Pines was distributed with 220 bands; 20 water-absorption bands removed -> 200.
        wavelengths=_aviris_wavelengths(220, _ranges((104, 108), (150, 163), (220, 220))),
        class_names=_names(
            "background alfalfa corn_notill corn_mintill corn grass_pasture "
            "grass_trees grass_pasture_mowed hay_windrowed oats soybean_notill "
            "soybean_mintill soybean_clean wheat woods "
            "buildings_grass_trees_drives stone_steel_towers"
        ),
        vegetation_classes=frozenset(range(1, 15)),
        urls=(
            f"{_EHU}/6/67/Indian_pines_corrected.mat",
            f"{_EHU}/c/c4/Indian_pines_gt.mat",
        ),
    ),
    "salinas": BenchmarkSpec(
        name="salinas",
        cube_file="Salinas_corrected.mat",
        cube_key="salinas_corrected",
        gt_file="Salinas_gt.mat",
        gt_key="salinas_gt",
        # Salinas: 224 AVIRIS bands, 20 water-absorption bands removed -> 204.
        wavelengths=_aviris_wavelengths(224, _ranges((108, 112), (154, 167), (224, 224))),
        class_names=_names(
            "background brocoli_green_weeds_1 brocoli_green_weeds_2 fallow "
            "fallow_rough_plow fallow_smooth stubble celery grapes_untrained "
            "soil_vinyard_develop corn_senesced_green_weeds lettuce_romaine_4wk "
            "lettuce_romaine_5wk lettuce_romaine_6wk lettuce_romaine_7wk "
            "vinyard_untrained vinyard_vertical_trellis"
        ),
        vegetation_classes=frozenset({1, 2, 7, 8, 10, 11, 12, 13, 14, 15, 16}),
        urls=(f"{_EHU}/a/a3/Salinas_corrected.mat", f"{_EHU}/f/fa/Salinas_gt.mat"),
    ),
    "pavia_u": BenchmarkSpec(
        name="pavia_u",
        cube_file="PaviaU.mat",
        cube_key="paviaU",
        gt_file="PaviaU_gt.mat",
        gt_key="paviaU_gt",
        # ROSIS: 103 bands kept of 115 over 430-860 nm (approximation).
        wavelengths=np.linspace(430.0, 860.0, 103),
        class_names=_names(
            "background asphalt meadows gravel trees painted_metal_sheets "
            "bare_soil bitumen self_blocking_bricks shadows"
        ),
        vegetation_classes=frozenset({2, 4}),
        urls=(f"{_EHU}/e/ee/PaviaU.mat", f"{_EHU}/5/50/PaviaU_gt.mat"),
    ),
}


@dataclass
class BenchmarkScene:
    """A benchmark cube on the canonical grid with its original labels."""

    name: str
    cube: np.ndarray  # [200, H, W] float32 reflectance
    labels: np.ndarray  # [H, W] int64 benchmark class ids (0 = unlabelled)
    band_covered: np.ndarray  # [200] bool: canonical bands inside the sensor range

    @property
    def vegetation_mask(self) -> np.ndarray:
        """Pixels whose benchmark class is vegetation."""
        return vegetation_mask(self.name, self.labels)


def _load_mat_var(path: Path, key: str) -> np.ndarray:
    from scipy.io import loadmat

    mat = loadmat(str(path))
    if key not in mat:
        # Some mirrors use different capitalisation; fall back to the first array variable.
        candidates = [k for k in mat if not k.startswith("__")]
        match = [k for k in candidates if k.lower() == key.lower()] or candidates
        log.warning("%s: variable %r not found, using %r", path.name, key, match[0])
        key = match[0]
    return np.asarray(mat[key])


def vegetation_mask(name: str, labels: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels whose benchmark class is vegetation.

    TODO(Day 3): refine with an NDVI threshold so unlabelled (class 0) vegetated pixels can
    also receive simulated stress, and drop senescent classes if they confuse training.
    """
    return np.isin(labels, sorted(BENCHMARKS[name].vegetation_classes))


def load_benchmark(name: str, data_dir: str | Path, method: str = "linear") -> BenchmarkScene:
    """Load one benchmark from ``data_dir`` and resample it to the canonical grid."""
    if name not in BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; choose from {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[name]
    root = Path(data_dir)
    cube_path, gt_path = root / spec.cube_file, root / spec.gt_file
    for p in (cube_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing - download it from {' or '.join(spec.urls)} (see model/README.md)"
            )
    raw = _load_mat_var(cube_path, spec.cube_key).astype(np.float32)  # [H, W, B]
    labels = _load_mat_var(gt_path, spec.gt_key).astype(np.int64)
    if raw.ndim != 3 or raw.shape[:2] != labels.shape:
        raise ValueError(f"{name}: unexpected shapes cube={raw.shape} gt={labels.shape}")
    if raw.shape[2] != spec.wavelengths.size:
        raise ValueError(f"{name}: expected {spec.wavelengths.size} bands, file has {raw.shape[2]}")
    cube = np.clip(np.moveaxis(raw, -1, 0) / spec.scale, 0.0, 1.0)
    resampled, covered = resample_to_canonical(
        cube, spec.wavelengths, method="gaussian" if method == "gaussian" else "linear"
    )
    log.info("loaded %s: %s -> %s", name, raw.shape, resampled.shape)
    return BenchmarkScene(name=name, cube=resampled, labels=labels, band_covered=covered)


def available_benchmarks(data_dir: str | Path) -> list[str]:
    """Names of benchmarks whose files are present in ``data_dir``."""
    root = Path(data_dir)
    return [
        n
        for n, s in BENCHMARKS.items()
        if (root / s.cube_file).exists() and (root / s.gt_file).exists()
    ]
