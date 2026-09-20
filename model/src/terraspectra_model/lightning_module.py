"""LightningModule wrapping :class:`TerraSpectraNet` with losses, metrics and schedules."""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics import MeanAbsoluteError, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

from terraspectra_model.arch.hybrid import TerraSpectraNet
from terraspectra_model.config import Config
from terraspectra_model.losses import TerraSpectraLoss


def warmup_cosine(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """LR multiplier: linear warmup then cosine decay to ``min_ratio``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _class_metrics(n: int, ignore: int) -> MetricCollection:
    return MetricCollection(
        {
            "oa": MulticlassAccuracy(n, average="micro", ignore_index=ignore),
            "macro_f1": MulticlassF1Score(n, average="macro", ignore_index=ignore),
            "miou": MulticlassJaccardIndex(n, average="macro", ignore_index=ignore),
        }
    )


class TerraSpectraLitModule(L.LightningModule):
    """Training / validation logic. Batches are dicts ``x, y, onset, mask``."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.save_hyperparameters({"cfg": cfg.model_dump(mode="json")})
        self.cfg = cfg
        self.model = TerraSpectraNet(cfg.model)
        t = cfg.train
        self.criterion = TerraSpectraLoss(
            t.focal_gamma, t.class_weights, t.ignore_index, t.onset_weight, t.huber_delta
        )
        n = cfg.model.n_classes
        self.train_metrics = _class_metrics(n, t.ignore_index).clone(prefix="train/")
        self.val_metrics = _class_metrics(n, t.ignore_index).clone(prefix="val/")
        self.val_onset_mae = MeanAbsoluteError()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """C2 forward (probs, onset)."""
        out: tuple[torch.Tensor, torch.Tensor] = self.model(x)
        return out

    def _masked_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        y = batch["y"].clone()
        y[~batch["mask"]] = self.cfg.train.ignore_index
        return y

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits, onset = self.model.forward_logits(batch["x"])
        loss, parts = self.criterion(logits, onset, batch["y"], batch["onset"], batch["mask"])
        bs = batch["x"].shape[0]
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=bs, sync_dist=stage != "train")
        for k, v in parts.items():
            self.log(f"{stage}/{k}", v, batch_size=bs, sync_dist=stage != "train")
        target = self._masked_target(batch)
        preds = logits.detach().argmax(1)
        metrics = self.train_metrics if stage == "train" else self.val_metrics
        metrics.update(preds, target)
        if stage != "train":
            sel = target != self.cfg.train.ignore_index
            if sel.any():
                self.val_onset_mae.update(onset.detach().squeeze(1)[sel], batch["onset"][sel])
        return loss

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """One optimisation step."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """One validation step."""
        self._step(batch, "val")

    def on_train_epoch_end(self) -> None:
        """Log and reset training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics."""
        self.log_dict(self.val_metrics.compute(), prog_bar=True, sync_dist=True)
        if self.val_onset_mae.update_count:
            self.log("val/onset_mae", self.val_onset_mae.compute(), sync_dist=True)
        self.val_metrics.reset()
        self.val_onset_mae.reset()

    def configure_optimizers(self) -> Any:
        """AdamW (no decay on norms/biases/embeddings) + per-step warmup-cosine schedule."""
        t = self.cfg.train
        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 or name.endswith(("pos", "spec_pos")) else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": t.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=t.lr,
        )  # fmt: skip
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warmup = int(t.warmup_fraction * total)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: warmup_cosine(s, total, warmup, t.min_lr_ratio)
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    @property
    def net(self) -> nn.Module:
        """The wrapped network (for export)."""
        return self.model
