"""Proxy labels for unlabelled real scenes from pre-visual stress indices.

Each pixel's indices are compared against a field baseline (a high percentile over vegetated
pixels, i.e. "the healthy part of this field"). Deficits are voted against early / high
thresholds from :class:`ProxyLabelConfig` (weighted votes, ``min_score`` to fire);
``confidence`` is the weighted fraction of indicators that agree with the assigned class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from terraspectra_contracts import RiskClass

from terraspectra_model import indices
from terraspectra_model.config import ProxyLabelConfig

IGNORE = -1

# Nominal days-to-onset per proxy class (a demo-level mapping, not agronomically validated).
NOMINAL_ONSET = {0: 30.0, 1: 16.0, 2: 4.0, 3: 0.0}


@dataclass
class ProxyLabels:
    """Proxy label output."""

    class_map: np.ndarray  # [H, W] int64, -1 where not vegetation / nodata
    confidence: np.ndarray  # [H, W] float32 in [0, 1]
    onset_map: np.ndarray  # [H, W] float32 nominal days
    deficits: dict[str, np.ndarray]  # per-indicator deficit maps (positive = more stressed)


def index_deficits(
    cube: np.ndarray, vegetation: np.ndarray, cfg: ProxyLabelConfig
) -> dict[str, np.ndarray]:
    """Baseline-relative deficits of REP (nm), NDRE, PRI, CCI and NDVI."""
    values = {
        "rep_shift_nm": indices.red_edge_position(cube),
        "ndre_drop": indices.ndre(cube),
        "pri_drop": indices.pri(cube),
        "cci_drop": indices.cci(cube),
        "ndvi_drop": indices.ndvi(cube),
    }
    out: dict[str, np.ndarray] = {}
    for name, v in values.items():
        if cfg.smooth_sigma > 0:
            v = gaussian_filter(np.nan_to_num(v), cfg.smooth_sigma, mode="nearest")
        ref = np.percentile(v[vegetation], cfg.baseline_percentile) if vegetation.any() else 0.0
        out[name] = np.asarray(ref - v, dtype=np.float32)
    return out


def proxy_labels(
    cube: np.ndarray,
    cfg: ProxyLabelConfig | None = None,
    valid_mask: np.ndarray | None = None,
) -> ProxyLabels:
    """Compute proxy class, confidence and nominal onset maps for ``cube[200, H, W]``."""
    cfg = cfg or ProxyLabelConfig()
    valid = np.all(cube >= 0, axis=0) if valid_mask is None else valid_mask.astype(bool)
    vegetation = valid & (indices.ndvi(cube) >= cfg.ndvi_vegetation_min)
    deficits = index_deficits(cube, vegetation, cfg)

    thresholds = {
        "rep_shift_nm": cfg.rep_shift_nm,
        "ndre_drop": cfg.ndre_drop,
        "pri_drop": cfg.pri_drop,
        "cci_drop": cfg.cci_drop,
    }
    w = {k: float(cfg.weights.get(k, 1.0)) for k in thresholds}
    total_w = sum(w.values())
    early = sum(w[k] * (deficits[k] >= t[0]) for k, t in thresholds.items())
    high = sum(w[k] * (deficits[k] >= t[1]) for k, t in thresholds.items())
    early_score, high_score = np.asarray(early, float), np.asarray(high, float)
    visible = deficits["ndvi_drop"] >= cfg.visible_ndvi_drop

    cls = np.full(cube.shape[1:], int(RiskClass.HEALTHY), dtype=np.int64)
    cls[early_score >= cfg.min_score] = RiskClass.EARLY_STRESS
    cls[high_score >= cfg.min_score] = RiskClass.HIGH_BLIGHT_RISK
    cls[visible] = RiskClass.VISIBLE_DISEASE

    agree = np.select(
        [cls == RiskClass.VISIBLE_DISEASE, cls == RiskClass.HIGH_BLIGHT_RISK,
         cls == RiskClass.EARLY_STRESS],
        [np.full(cls.shape, total_w), high_score, early_score],
        default=total_w - early_score,
    )  # fmt: skip
    conf = (agree / total_w).astype(np.float32)
    # Pixels still counted as vegetated after visible damage keep their label.
    cls[~(vegetation | (valid & visible))] = IGNORE
    conf[cls == IGNORE] = 0.0
    onset = np.vectorize(lambda c: NOMINAL_ONSET.get(int(c), 30.0), otypes=[np.float32])(cls)
    # TODO(Day 4): calibrate thresholds on simulated series (evaluate.lead_time_curve) and
    # replace the nominal onset lookup with a regression on the stress score.
    return ProxyLabels(class_map=cls, confidence=conf, onset_map=onset, deficits=deficits)
