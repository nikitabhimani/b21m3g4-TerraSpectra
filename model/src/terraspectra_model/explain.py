"""Explainability: integrated-gradients band importance -> contract C4 ``dominant_indicator``."""

from __future__ import annotations

import numpy as np
import torch
from terraspectra_contracts import N_BANDS, WAVELENGTHS_NM, RiskClass
from torch import nn

# Spectral regions (nm) associated with each indicator named in C4 zones.
INDICATOR_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "red_edge_shift": ((690.0, 760.0),),
    "pri_decline": ((520.0, 580.0),),
    "chlorophyll_loss": ((400.0, 515.0), (585.0, 690.0)),
    "water_stress": ((940.0, 1000.0), (1150.0, 1250.0), (1350.0, 2500.0)),
}
INDICATORS: tuple[str, ...] = tuple(INDICATOR_RANGES)

_WL = np.asarray(WAVELENGTHS_NM)


def indicator_band_masks() -> dict[str, np.ndarray]:
    """Boolean ``[200]`` band mask per indicator."""
    return {
        name: np.any([(lo <= _WL) & (hi >= _WL) for lo, hi in spans], axis=0)
        for name, spans in INDICATOR_RANGES.items()
    }


def _region_mean(values: torch.Tensor, region: torch.Tensor | None) -> torch.Tensor:
    """Per-sample mean of ``values [N, H, W]`` over ``region [N, H, W]`` (all pixels if None)."""
    if region is None:
        return values.flatten(1).mean(1)
    m = region.to(values.dtype)
    return (values * m).flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1.0)


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int | None = None,
    region: torch.Tensor | None = None,
    baseline: torch.Tensor | None = None,
    steps: int = 32,
    batch_size: int = 8,
) -> torch.Tensor:
    """Integrated gradients of a region score w.r.t. the input; returns ``[N, 200, H, W]``.

    The score is the mean probability of ``target_class`` (default: any non-healthy class, i.e.
    ``1 - p(healthy)``) over ``region [N, H, W]`` (default: whole window), integrated along the
    straight path from ``baseline`` (default zeros) with the midpoint rule.
    """
    model = model.eval()
    baseline = torch.zeros_like(x) if baseline is None else baseline.expand_as(x)
    alphas = (torch.arange(steps, dtype=x.dtype, device=x.device) + 0.5) / steps
    total = torch.zeros_like(x)
    for start in range(0, steps, batch_size):
        a = alphas[start : start + batch_size]
        n_steps = len(a)
        # [S, N, B, H, W] -> [S*N, B, H, W]; sample order is s * N + n.
        path = baseline[None] + a[:, None, None, None, None] * (x - baseline)[None]
        path = path.flatten(0, 1).detach().requires_grad_(True)
        probs, _ = model(path)
        if target_class is None:
            values = 1.0 - probs[:, int(RiskClass.HEALTHY)]
        else:
            values = probs[:, target_class]
        reg = None if region is None else region.repeat(n_steps, 1, 1)
        (grad,) = torch.autograd.grad(_region_mean(values, reg).sum(), path)
        total += grad.unflatten(0, (n_steps, -1)).sum(0)
    return (x - baseline) * total / steps


def band_importance(
    model: nn.Module,
    x: torch.Tensor,
    region: torch.Tensor | None = None,
    steps: int = 32,
    target_class: int | None = None,
) -> np.ndarray:
    """Normalised non-negative per-band importance ``[200]`` (sums to 1)."""
    if x.shape[1] != N_BANDS:
        raise ValueError(f"expected {N_BANDS} bands")
    attr = integrated_gradients(model, x, target_class, region, steps=steps)
    if region is not None:
        attr = attr * region[:, None].to(attr.dtype)
    imp = attr.abs().sum(dim=(0, 2, 3)).detach().cpu().numpy().astype(np.float64)
    s = imp.sum()
    return imp / s if s > 0 else np.full(N_BANDS, 1.0 / N_BANDS)


def indicator_scores(importance: np.ndarray) -> dict[str, float]:
    """Mean importance per band inside each indicator's spectral region (density, not mass)."""
    return {name: float(importance[m].mean()) for name, m in indicator_band_masks().items()}


def dominant_indicator(importance: np.ndarray) -> str:
    """C4 ``dominant_indicator`` value for a band-importance vector."""
    scores = indicator_scores(importance)
    return max(scores, key=lambda k: scores[k])


def top_bands(importance: np.ndarray, k: int = 10) -> list[tuple[float, float]]:
    """``[(wavelength_nm, importance), ...]`` for the ``k`` most important bands."""
    idx = np.argsort(importance)[::-1][:k]
    return [(float(_WL[i]), float(importance[i])) for i in idx]
