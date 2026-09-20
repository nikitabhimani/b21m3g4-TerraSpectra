"""Loss functions: masked focal loss (classes) + masked Huber (days-to-onset)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    ignore_index: int = -1,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean focal loss over valid pixels; equals weighted cross-entropy when ``gamma == 0``.

    ``logits [N, C, H, W]``, ``target [N, H, W]`` (``ignore_index`` skipped),
    ``valid_mask [N, H, W]`` bool (False skipped).
    """
    valid = target != ignore_index
    if valid_mask is not None:
        valid = valid & valid_mask.bool()
    tgt = target.clamp(min=0)
    logp = F.log_softmax(logits.float(), dim=1)
    logpt = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    loss = -logpt if gamma == 0 else -((1.0 - logpt.exp()).clamp(min=0) ** gamma) * logpt
    w = valid.to(loss.dtype)
    if weight is not None:
        w = w * weight.to(loss)[tgt]
    return (loss * w).sum() / w.sum().clamp(min=1e-8)


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    delta: float = 2.0,
) -> torch.Tensor:
    """Huber loss averaged over ``mask`` pixels; ``pred`` may carry a channel dim of 1."""
    if pred.dim() == target.dim() + 1:
        pred = pred.squeeze(1)
    loss = F.huber_loss(pred.float(), target.float(), reduction="none", delta=delta)
    if mask is None:
        return loss.mean()
    m = mask.to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp(min=1e-8)


class TerraSpectraLoss(nn.Module):
    """``focal(classes) + onset_weight * huber(onset)``; returns ``(total, parts)``."""

    class_weights: torch.Tensor | None

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: list[float] | None = None,
        ignore_index: int = -1,
        onset_weight: float = 0.1,
        huber_delta: float = 2.0,
    ) -> None:
        super().__init__()
        self.gamma, self.ignore_index = gamma, ignore_index
        self.onset_weight, self.huber_delta = onset_weight, huber_delta
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", w)

    def forward(
        self,
        logits: torch.Tensor,
        onset: torch.Tensor,
        target: torch.Tensor,
        onset_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the combined loss."""
        cls = focal_loss(logits, target, self.gamma, self.class_weights, self.ignore_index, mask)
        reg = masked_huber(onset, onset_target, mask & (target != self.ignore_index),
                           self.huber_delta)  # fmt: skip
        total = cls + self.onset_weight * reg
        return total, {"loss_cls": cls.detach(), "loss_onset": reg.detach()}
