"""Simulated fungal-stress progression on the canonical 200-band grid.

A built-in, simplified leaf + canopy reflectance model (PROSPECT/SAIL-flavoured, not physically
exact) turns chlorophyll ``cab`` (ug/cm2), equivalent water thickness ``cw`` (cm), leaf area index
``lai``, brown pigments ``cbrown`` and a xanthophyll de-epoxidation proxy ``xanth`` into canopy
reflectance. It reproduces the qualitative signatures of early stress: red-edge blue-shift and a
rising green peak as chlorophyll falls, PRI decline, and deeper/less masked SWIR water features.
When the optional ``prosail`` package is installed it can be used instead (``backend="prosail"``).

Label rules (days ``d`` until visible symptoms, integer days):

* not infected (or ``d > max_infected_days``)  -> 0 healthy, onset target 30
* infected, ``early_stress_min_days < d <= max_infected_days`` -> 1 early_stress
* infected, ``0 < d <= early_stress_min_days`` -> 2 high_blight_risk
* infected, ``d == 0`` -> 3 visible_disease
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    def gaussian_filter(input_arr: np.ndarray, sigma: float, mode: str = "wrap") -> np.ndarray:  # type: ignore[no-redef]
        """Pure numpy fallback for gaussian_filter."""
        radius = int(max(1, round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        pad_width = radius
        pad_mode = mode if mode in ("wrap", "reflect", "edge") else "edge"
        padded = np.pad(input_arr, pad_width, mode=pad_mode)
        res = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
        )
        res = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=res
        )
        return res[:input_arr.shape[0], :input_arr.shape[1]]

from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS, WAVELENGTHS_NM, RiskClass

from terraspectra_model.config import SynthConfig
from terraspectra_model.structures import LabelledScene

log = logging.getLogger(__name__)

WL = np.asarray(WAVELENGTHS_NM, dtype=np.float64)


def _g(mu: float, sigma: float) -> np.ndarray:
    return np.asarray(np.exp(-(((WL - mu) / sigma) ** 2)))


def _sig(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


# Specific absorption spectra (arbitrary but calibrated so typical parameters give plausible
# reflectance: green ~0.1, red ~0.04, NIR ~0.45, SWIR water features at 1450/1940 nm).
_RED_TAIL = np.where(WL < 670, _g(670, 25), _g(670, 37))
K_CAB = 0.30 * _g(435, 30) + 0.10 * _g(480, 30) + 0.03 * _sig((700 - WL) / 10) + 0.15 * _RED_TAIL
K_XANTH = _g(531, 12)
K_WATER = 3.4 * (
    0.1 * _g(970, 40)
    + 1.0 * _g(1200, 50)
    + 25.0 * _g(1450, 60)
    + 100.0 * _g(1940, 80)
    + 7.0 * _sig((WL - 1350) / 30)
    + 15.0 * _sig((WL - 2000) / 60)
)
K_DRY = 40.0 * _sig((WL - 1100) / 200)
K_BROWN = np.where(WL < 1300, np.exp(-(WL - 400) / 250), 0.0)
SOIL = (0.08 + 0.22 * (WL - 400) / 2100) * (1 - 0.3 * _g(1450, 50) - 0.4 * _g(1940, 60))


class ReflectanceBackend(Protocol):
    """Anything mapping per-pixel biophysical parameters ``[P]`` to reflectance ``[P, 200]``."""

    def __call__(
        self,
        cab: np.ndarray,
        cw: np.ndarray,
        lai: np.ndarray,
        cbrown: np.ndarray,
        xanth: np.ndarray,
        soil_brightness: np.ndarray,
    ) -> np.ndarray: ...


def builtin_reflectance(
    cab: np.ndarray,
    cw: np.ndarray,
    lai: np.ndarray,
    cbrown: np.ndarray,
    xanth: np.ndarray,
    soil_brightness: np.ndarray,
    cm: float = 0.005,
) -> np.ndarray:
    """Vectorised simplified leaf + canopy model; all inputs broadcast to ``[P]``."""
    cab, cw, lai, cbrown, xanth, soil_b = (
        np.atleast_1d(np.asarray(a, dtype=np.float64)).reshape(-1, 1)
        for a in (cab, cw, lai, cbrown, xanth, soil_brightness)
    )
    depth = cab * K_CAB + 0.3 * xanth * K_XANTH + cw * K_WATER + cm * K_DRY + 2.0 * cbrown * K_BROWN
    leaf = 0.04 + 0.46 * np.exp(-depth)
    leaf_canopy = leaf / (1.0 - 0.25 * leaf)  # crude multiple-scattering boost
    cover = 1.0 - np.exp(-0.5 * lai)
    out = soil_b * SOIL[None, :] * (1.0 - cover) + leaf_canopy * cover
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


class ProsailBackend:
    """PROSAIL (``prosail`` package) with a quantised-parameter cache; slower than the built-in."""

    def __init__(self) -> None:
        import prosail  # noqa: F401  - fail fast if missing

        self._cache: dict[tuple[float, ...], np.ndarray] = {}
        self._idx = WL - 400.0

    def _one(self, cab: float, cw: float, lai: float, cbrown: float) -> np.ndarray:
        key = (round(cab / 2) * 2, round(cw, 3), round(lai * 4) / 4, round(cbrown, 1))
        if key not in self._cache:
            import prosail

            r = prosail.run_prosail(
                1.5, key[0], 8.0, key[3], max(key[1], 1e-4), 0.009, lai=max(key[2], 0.0),
                lidfa=-0.35, hspot=0.01, tts=30.0, tto=10.0, psi=0.0, typelidf=2,
                rsoil=1.0, psoil=1.0,
            )  # fmt: skip
            self._cache[key] = np.interp(self._idx, np.arange(r.size), r)
        return self._cache[key]

    def __call__(
        self,
        cab: np.ndarray,
        cw: np.ndarray,
        lai: np.ndarray,
        cbrown: np.ndarray,
        xanth: np.ndarray,
        soil_brightness: np.ndarray,
    ) -> np.ndarray:
        """Reflectance ``[P, 200]``; xanthophyll (PRI) is applied as a multiplicative feature."""
        params = np.broadcast_arrays(*(np.atleast_1d(a) for a in (cab, cw, lai, cbrown)))
        out = np.stack([self._one(*map(float, p)) for p in zip(*params, strict=True)])
        out *= np.exp(-0.3 * np.atleast_1d(xanth)[:, None] * K_XANTH * 0.2)
        # TODO(Day 3): pass a real soil spectrum (rsoil0) scaled by soil_brightness.
        return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float32)


def get_backend(name: str = "builtin") -> ReflectanceBackend:
    """Resolve ``"builtin" | "prosail" | "auto"`` to a callable backend."""
    if name in ("prosail", "auto"):
        try:
            return ProsailBackend()
        except ImportError:
            if name == "prosail":
                raise
            log.info("prosail not installed; using built-in reflectance model")
    return builtin_reflectance


# --------------------------------------------------------------------------- progression rules
def severity(days: np.ndarray | float, horizon: float = MAX_ONSET_DAYS) -> np.ndarray:
    """Hidden stress severity in [0, 1] for an infected pixel ``days`` before visible symptoms."""
    d = np.clip(np.asarray(days, dtype=np.float64), 0.0, horizon)
    return np.asarray(((horizon - d) / horizon) ** 1.5)


def stressed_parameters(
    cab0: np.ndarray, cw0: np.ndarray, lai0: np.ndarray, sev: np.ndarray
) -> dict[str, np.ndarray]:
    """Degrade healthy parameters by severity (chlorophyll, water, structure, pigments)."""
    return {
        "cab": cab0 * (1.0 - 0.55 * sev),
        "cw": cw0 * (1.0 - 0.35 * sev),
        "lai": lai0 * (1.0 - 0.35 * sev**2),
        "cbrown": 1.5 * np.clip((sev - 0.85) / 0.15, 0.0, 1.0),
        "xanth": sev,
    }


def label_for_days(
    days: np.ndarray,
    infected: np.ndarray,
    early_stress_min_days: float = 7.0,
    max_infected_days: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the module-level label rules; returns ``(classes int64, onset_target float32)``."""
    d = np.asarray(days, dtype=np.float64)
    inf = np.asarray(infected, dtype=bool) & (d <= max_infected_days)
    cls = np.full(d.shape, int(RiskClass.HEALTHY), dtype=np.int64)
    cls[inf & (d > early_stress_min_days)] = RiskClass.EARLY_STRESS
    cls[inf & (d > 0) & (d <= early_stress_min_days)] = RiskClass.HIGH_BLIGHT_RISK
    cls[inf & (d <= 0)] = RiskClass.VISIBLE_DISEASE
    onset = np.where(inf, np.clip(d, 0.0, MAX_ONSET_DAYS), MAX_ONSET_DAYS).astype(np.float32)
    return cls, onset


def simulate_progression(
    days: np.ndarray | list[int] | range = range(int(MAX_ONSET_DAYS) + 1),
    cab0: float = 45.0,
    cw0: float = 0.015,
    lai0: float = 3.5,
    backend: str = "builtin",
    cfg: SynthConfig | None = None,
) -> dict[str, np.ndarray]:
    """Spectra of one infected canopy observed ``d`` days before visible symptoms.

    Returns ``days [T]``, ``spectra [T, 200]``, ``labels [T]``, ``onset [T]`` and the
    un-stressed ``healthy [200]`` reference.
    """
    cfg = cfg or SynthConfig()
    d = np.asarray(list(days), dtype=np.float64)
    fn = get_backend(backend)
    p = stressed_parameters(np.full_like(d, cab0), np.full_like(d, cw0), np.full_like(d, lai0),
                            severity(d))  # fmt: skip
    spectra = fn(p["cab"], p["cw"], p["lai"], p["cbrown"], p["xanth"], np.ones_like(d))
    healthy = fn(*(np.array([v]) for v in (cab0, cw0, lai0, 0.0, 0.0, 1.0)))[0]
    labels, onset = label_for_days(
        d, np.ones_like(d, dtype=bool), cfg.early_stress_min_days, cfg.max_infected_days
    )
    return {"days": d, "spectra": spectra, "labels": labels, "onset": onset, "healthy": healthy}


# --------------------------------------------------------------------------- spatial synthesis
def _smooth_field(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    f = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return np.asarray(f / (f.std() + 1e-8))


def _uniform_field(
    rng: np.random.Generator, shape: tuple[int, int], lo_hi: tuple[float, float]
) -> np.ndarray:
    lo, hi = lo_hi
    t = _sig(_smooth_field(rng, shape, sigma=shape[0] / 6) * 1.2 + rng.normal(0, 0.5))
    return lo + (hi - lo) * t


def infection_days_map(
    rng: np.random.Generator,
    shape: tuple[int, int],
    cfg: SynthConfig,
    fixed_days: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Blob-shaped infection foci: ``(days[H,W], infected[H,W])``.

    Each focus has a centre that is ``d0`` days from symptoms; days grow with distance
    (``days_per_pixel``), producing concentric risk rings. ``fixed_days`` makes every infected
    pixel exactly that many days out (used for lead-time curves).
    """
    H, W = shape
    days = np.full(shape, np.inf)
    n_blobs = int(rng.integers(1, cfg.max_blobs + 1)) if cfg.max_blobs > 0 else 0
    yy, xx = np.mgrid[0:H, 0:W]
    scale = max(H, W) / 64.0
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        radius = rng.uniform(*cfg.blob_radius) * scale
        # Anisotropic, slightly irregular blob.
        ang = rng.uniform(0, np.pi)
        ax = rng.uniform(0.6, 1.0)
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = (-dx * np.sin(ang) + dy * np.cos(ang)) / ax
        dist = np.sqrt(u**2 + v**2) + 1.5 * _smooth_field(rng, shape, 3.0)
        if fixed_days is not None:
            blob = np.where(dist <= radius, float(fixed_days), np.inf)
        else:
            # Multi-stage biophysical epidemic progression:
            # mature (necrotic core d=0 -> high risk halo -> pre-visual edge)
            # developing (high risk core 1<=d<=7 -> pre-visual edge)
            # early (pure pre-visual incubation 8<=d<=25)
            mat = rng.choice(["mature", "developing", "early"], p=[0.45, 0.35, 0.20])
            t = np.clip(dist / max(radius, 1e-4), 0.0, 1.0)
            if mat == "mature":
                t_core = rng.uniform(0.25, 0.40)
                t_high = t_core + rng.uniform(0.25, 0.35)
                d_val = np.where(
                    t <= t_core,
                    0.0,
                    np.where(
                        t <= t_high,
                        1.0 + 6.0 * (t - t_core) / max(t_high - t_core, 1e-4),
                        8.0
                        + (cfg.max_infected_days - 8.0) * (t - t_high) / max(1.0 - t_high, 1e-4),
                    ),
                )
            elif mat == "developing":
                t_high = rng.uniform(0.35, 0.55)
                d_val = np.where(
                    t <= t_high,
                    1.0 + 6.0 * t / max(t_high, 1e-4),
                    8.0
                    + (cfg.max_infected_days - 8.0) * (t - t_high) / max(1.0 - t_high, 1e-4),
                )
            else:
                d0 = rng.uniform(cfg.early_stress_min_days + 1.0, 18.0)
                d_val = d0 + (cfg.max_infected_days - d0) * t

            blob = np.where(dist <= radius, np.floor(d_val), np.inf)
        days = np.minimum(days, blob)
    infected = np.isfinite(days) & (days <= cfg.max_infected_days)
    return np.where(infected, days, MAX_ONSET_DAYS), infected


def _render(
    backend: ReflectanceBackend,
    rng: np.random.Generator,
    shape: tuple[int, int],
    cfg: SynthConfig,
    days: np.ndarray,
    infected: np.ndarray,
    soil: np.ndarray,
) -> np.ndarray:
    cab0 = _uniform_field(rng, shape, cfg.healthy_cab)
    cw0 = _uniform_field(rng, shape, cfg.healthy_cw)
    lai0 = np.where(soil, 0.0, _uniform_field(rng, shape, cfg.healthy_lai))
    sev = np.where(infected, severity(days), 0.0)
    p = stressed_parameters(cab0, cw0, lai0, sev)
    soil_b = np.full(shape, rng.uniform(0.7, 1.4))
    flat = backend(
        np.ravel(p["cab"]),
        np.ravel(p["cw"]),
        np.ravel(p["lai"]),
        np.ravel(p["cbrown"]),
        np.ravel(p["xanth"]),
        soil_b.ravel(),
    )
    cube = flat.T.reshape(N_BANDS, *shape)
    illum = 1.0 + 0.03 * _smooth_field(rng, shape, 2.0)
    cube = cube * illum[None] + rng.normal(0.0, cfg.noise_std, size=cube.shape)
    return np.asarray(np.clip(cube, 0.0, 1.0), dtype=np.float32)


def make_field_patch(
    rng: np.random.Generator,
    cfg: SynthConfig | None = None,
    size: int = 64,
    fixed_days: int | None = None,
    backend: ReflectanceBackend | None = None,
) -> LabelledScene:
    """One synthetic crop-field window with blended infected foci, soil strips and nodata."""
    cfg = cfg or SynthConfig()
    backend = backend or get_backend(cfg.backend)
    shape = (size, size)
    if fixed_days is not None or rng.random() < cfg.patch_infected_p:
        days, infected = infection_days_map(rng, shape, cfg, fixed_days)
    else:
        days, infected = np.full(shape, MAX_ONSET_DAYS), np.zeros(shape, dtype=bool)

    soil = np.zeros(shape, dtype=bool)
    if rng.random() < cfg.soil_fraction_p:  # bare-soil track / field edge
        w = int(rng.integers(2, max(3, size // 6)))
        start = int(rng.integers(0, size - w))
        if rng.random() < 0.5:
            soil[:, start : start + w] = True
        else:
            soil[start : start + w, :] = True
    infected &= ~soil

    cube = _render(backend, rng, shape, cfg, days, infected, soil)
    classes, onset = label_for_days(
        days, infected, cfg.early_stress_min_days, cfg.max_infected_days
    )

    valid = np.ones(shape, dtype=bool)
    if rng.random() < cfg.nodata_p:  # cloud-ish nodata region, zero-filled per C2
        cloud = _smooth_field(rng, shape, 6.0) > rng.uniform(1.0, 2.0)
        valid &= ~cloud
        cube[:, ~valid] = 0.0
    return LabelledScene(cube=cube, class_map=classes, onset_map=onset, valid_mask=valid)


def blend_stress_into_scene(
    cube: np.ndarray,
    vegetation: np.ndarray,
    rng: np.random.Generator,
    cfg: SynthConfig | None = None,
    foci_per_megapixel: float = 400.0,
) -> LabelledScene:
    """Inject simulated stress into real vegetation pixels of a benchmark scene.

    The stressed/healthy reflectance *ratio* from the reflectance model multiplies the real
    spectra, preserving each pixel's texture. Non-vegetation pixels get class ``-1`` (ignored).
    """
    cfg = cfg or SynthConfig()
    backend = get_backend(cfg.backend)
    _, H, W = cube.shape
    n = max(1, int(foci_per_megapixel * H * W / 1e6))
    local = cfg.model_copy(update={"max_blobs": n})
    days, infected = infection_days_map(rng, (H, W), local)
    infected &= vegetation
    idx = np.flatnonzero(infected)
    out = cube.copy()
    if idx.size:
        sev = severity(days.ravel()[idx])
        base = {"cab": np.full(idx.size, 45.0), "cw": np.full(idx.size, 0.015),
                "lai": np.full(idx.size, 3.5)}  # fmt: skip
        p = stressed_parameters(base["cab"], base["cw"], base["lai"], sev)
        ones = np.ones(idx.size)
        stressed = backend(p["cab"], p["cw"], p["lai"], p["cbrown"], p["xanth"], ones)
        healthy = backend(base["cab"], base["cw"], base["lai"], 0 * ones, 0 * ones, ones)
        ratio = stressed / np.maximum(healthy, 1e-4)
        flat = out.reshape(cube.shape[0], -1)
        flat[:, idx] = np.clip(flat[:, idx] * ratio.T, 0.0, 1.0)
    classes, onset = label_for_days(
        days, infected, cfg.early_stress_min_days, cfg.max_infected_days
    )
    classes[~vegetation] = -1
    valid = np.ones((H, W), dtype=bool)
    return LabelledScene(cube=out, class_map=classes, onset_map=onset, valid_mask=valid)


def generate_dataset(
    n: int, seed: int = 0, cfg: SynthConfig | None = None, size: int = 64
) -> dict[str, np.ndarray]:
    """``n`` synthetic windows as stacked arrays (for ``terraspectra-model synth``)."""
    cfg = cfg or SynthConfig()
    backend = get_backend(cfg.backend)
    rng = np.random.default_rng(seed)
    patches = [make_field_patch(rng, cfg, size, backend=backend) for _ in range(n)]
    return {
        "x": np.stack([p.cube for p in patches]),
        "y": np.stack([p.class_map for p in patches]),
        "onset": np.stack([p.onset_map for p in patches]),
        "mask": np.stack([p.valid_mask for p in patches]),
    }
