"""Evaluation: metrics report, lead-time curve and temperature-scaling calibration."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from terraspectra_contracts import CLASS_NAMES, N_CLASSES, RiskClass
from torch import nn

from terraspectra_model.config import SynthConfig
from terraspectra_model.structures import LabelledScene

log = logging.getLogger(__name__)


def _logits_and_onset(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use ``forward_logits`` when available, otherwise log-probs of the C2 forward."""
    forward_logits = getattr(model, "forward_logits", None)
    if callable(forward_logits):
        logits, onset = forward_logits(x)
        return cast(torch.Tensor, logits), cast(torch.Tensor, onset)
    probs, onset = model(x)
    return probs.clamp_min(1e-8).log(), onset


def confusion_matrix(pred: np.ndarray, target: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    """``[n, n]`` counts, rows = truth, columns = prediction (``target < 0`` ignored)."""
    sel = target >= 0
    return np.bincount(n * target[sel] + pred[sel], minlength=n * n).reshape(n, n)


def metrics_from_confusion(cm: np.ndarray) -> dict[str, Any]:
    """Overall accuracy, per-class / macro F1 and IoU from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
    present = cm.sum(1) > 0
    return {
        "overall_accuracy": float(tp.sum() / max(1, cm.sum())),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "miou": float(iou[present].mean()) if present.any() else 0.0,
        "per_class_f1": {CLASS_NAMES[i]: float(f1[i]) for i in range(len(f1))},
        "per_class_iou": {CLASS_NAMES[i]: float(iou[i]) for i in range(len(iou))},
        "support": {CLASS_NAMES[i]: int(cm[i].sum()) for i in range(len(f1))},
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: str | torch.device = "cpu",
    collect_logits: bool = False,
) -> dict[str, Any]:
    """Aggregate classification metrics and onset MAE over ``batches``."""
    model = model.eval().to(device)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    abs_err, n_px = 0.0, 0
    all_logits, all_targets = [], []
    for b in batches:
        x = b["x"].to(device)
        logits, onset = _logits_and_onset(model, x)
        y = b["y"].clone()
        y[~b["mask"]] = -1
        pred = logits.argmax(1).cpu()
        cm += confusion_matrix(pred.numpy().ravel(), y.numpy().ravel())
        sel = y >= 0
        abs_err += float((onset.squeeze(1).cpu()[sel] - b["onset"][sel]).abs().sum())
        n_px += int(sel.sum())
        if collect_logits:
            all_logits.append(logits.permute(0, 2, 3, 1).reshape(-1, N_CLASSES)[sel.ravel()].cpu())
            all_targets.append(y[sel])
    report = metrics_from_confusion(cm)
    report["onset_mae_days"] = abs_err / max(1, n_px)
    report["n_pixels"] = n_px
    if collect_logits and all_logits:
        report["_logits"] = torch.cat(all_logits)
        report["_targets"] = torch.cat(all_targets)
    return report


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write the JSON-serialisable part of ``report`` (keys starting with ``_`` dropped)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    path.write_text(json.dumps(clean, indent=2))
    return path


@torch.no_grad()
def lead_time_curve(
    model: nn.Module,
    days: Iterable[int] = range(0, 31, 2),
    n_per_day: int = 8,
    cfg: SynthConfig | None = None,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Detection vs days-before-symptoms on synthetic windows with fixed infection age.

    For each ``d``: ``detection_rate`` = fraction of infected pixels predicted non-healthy,
    ``class_accuracy`` = agreement with the rule-based label, ``false_alarm_rate`` on healthy
    pixels, and ``onset_mae``.
    """
    from terraspectra_model.synth.stress import get_backend, make_field_patch

    cfg = (cfg or SynthConfig()).model_copy(update={"max_infected_days": 30.0, "nodata_p": 0.0})
    backend = get_backend(cfg.backend)
    model = model.eval().to(device)
    rows = []
    for d in days:
        rng = np.random.default_rng((seed, d))
        patches: list[LabelledScene] = [
            make_field_patch(rng, cfg, fixed_days=d, backend=backend) for _ in range(n_per_day)
        ]
        x = torch.from_numpy(np.stack([p.cube for p in patches])).to(device)
        probs, onset = model(x)
        pred = probs.argmax(1).cpu().numpy()
        infected = np.stack([p.class_map != RiskClass.HEALTHY for p in patches])
        # Pixels 26-30 days out are infected but labelled healthy by the rules; count them too.
        infected |= np.stack([p.onset_map < 30 for p in patches])
        target = np.stack([p.class_map for p in patches])
        on = onset.squeeze(1).cpu().numpy()
        n_inf = max(1, int(infected.sum()))
        rows.append(
            {
                "days_before_symptoms": float(d),
                "detection_rate": float((pred[infected] != RiskClass.HEALTHY).sum() / n_inf),
                "class_accuracy": float((pred[infected] == target[infected]).sum() / n_inf),
                "false_alarm_rate": float(
                    (pred[~infected] != RiskClass.HEALTHY).sum() / max(1, int((~infected).sum()))
                ),
                "onset_mae": float(np.abs(on[infected] - d).mean()) if infected.any() else 0.0,
                "n_infected_pixels": float(infected.sum()),
            }
        )
    return rows


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 100,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Temperature scaling (Guo et al. 2017): minimise NLL of ``logits / T``, T in ``bounds``."""
    logits, targets = logits.detach().float(), targets.detach().long()
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().clamp(*bounds))
    log.info("fitted temperature %.3f", t)
    return t


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """ECE of ``probs [P, C]`` against ``targets [P]``."""
    conf, pred = probs.max(1)
    acc = (pred == targets).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    for lo, hi in pairwise(edges):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.float().mean() * (conf[sel].mean() - acc[sel].mean()).abs()
    return float(ece)
