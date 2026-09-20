"""Training entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger, Logger

from terraspectra_model.config import Config
from terraspectra_model.data.windows import TerraSpectraDataModule
from terraspectra_model.lightning_module import TerraSpectraLitModule

log = logging.getLogger(__name__)


def _loggers(cfg: Config) -> list[Logger]:
    t = cfg.train
    loggers: list[Logger] = [CSVLogger(str(t.output_dir), name="logs")]
    if t.mlflow:
        try:
            from lightning.pytorch.loggers import MLFlowLogger

            loggers.append(
                MLFlowLogger(experiment_name=t.experiment_name, tracking_uri=t.mlflow_tracking_uri)
            )
        except (ImportError, ModuleNotFoundError):
            log.warning("mlflow requested but not installed (uv sync --extra tracking); skipping")
    return loggers


def resolve_precision(requested: str) -> str:
    """Fall back to full precision where mixed precision is unavailable (CPU fp16)."""
    if "16" in requested and not torch.cuda.is_available():
        if requested.startswith("bf16"):
            return requested
        log.info("no CUDA: precision %s -> 32-true", requested)
        return "32-true"
    return requested


def build_trainer(cfg: Config, extra_callbacks: list[Callback] | None = None) -> L.Trainer:
    """Configure a Lightning Trainer (checkpointing, early stopping, loggers)."""
    t = cfg.train
    ckpt = ModelCheckpoint(
        dirpath=Path(t.output_dir) / "checkpoints",
        filename="epoch{epoch:03d}-step{step}",
        auto_insert_metric_name=False,
        monitor=t.monitor,
        mode="max" if "loss" not in t.monitor and "mae" not in t.monitor else "min",
        save_last=True,
        save_top_k=3,
    )
    callbacks: list[Callback] = [ckpt, *(extra_callbacks or [])]
    if t.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor=t.monitor, mode=ckpt.mode, patience=t.early_stopping_patience)
        )
    loggers: list[Logger] | bool = False if t.fast_dev_run else _loggers(cfg)
    if loggers:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    kwargs: dict[str, Any] = {}
    if t.limit_train_batches is not None:
        kwargs["limit_train_batches"] = t.limit_train_batches
    if t.limit_val_batches is not None:
        kwargs["limit_val_batches"] = t.limit_val_batches
    return L.Trainer(
        max_epochs=t.max_epochs,
        accelerator=t.accelerator,
        devices=t.devices,
        precision=resolve_precision(t.precision),  # type: ignore[arg-type]
        gradient_clip_val=t.gradient_clip_val,
        accumulate_grad_batches=t.accumulate_grad_batches,
        log_every_n_steps=t.log_every_n_steps,
        fast_dev_run=t.fast_dev_run,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(t.output_dir),
        enable_model_summary=not t.fast_dev_run,
        **kwargs,
    )


def train(cfg: Config) -> tuple[TerraSpectraLitModule, str | None]:
    """Train and return ``(module, best_checkpoint_path or None)``."""
    L.seed_everything(cfg.train.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    dm = TerraSpectraDataModule(cfg.data)
    module = TerraSpectraLitModule(cfg)
    trainer = build_trainer(cfg)
    trainer.fit(module, datamodule=dm)
    best = None
    if trainer.checkpoint_callback is not None:
        best = getattr(trainer.checkpoint_callback, "best_model_path", None) or None
    log.info("training finished; best checkpoint: %s", best)
    return module, best


def load_module(checkpoint: str | Path, cfg: Config | None = None) -> TerraSpectraLitModule:
    """Load a trained module from a Lightning checkpoint (config from the checkpoint if omitted)."""
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if cfg is None:
        cfg = Config.model_validate(ckpt["hyper_parameters"]["cfg"])
    module = TerraSpectraLitModule(cfg)
    module.load_state_dict(ckpt["state_dict"])
    return module.eval()
