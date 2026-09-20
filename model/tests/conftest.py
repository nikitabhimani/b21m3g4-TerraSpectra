"""Shared fixtures: tiny CPU config and model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from terraspectra_model.arch.hybrid import TerraSpectraNet
from terraspectra_model.config import Config, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture
def tiny_cfg(tmp_path: Path) -> Config:
    return load_config(
        CONFIG_DIR / "tiny.yaml",
        [f"train.output_dir={tmp_path / 'run'}", f"export.output_path={tmp_path / 'model.pt'}"],
    )


@pytest.fixture
def tiny_model(tiny_cfg: Config) -> TerraSpectraNet:
    return TerraSpectraNet(tiny_cfg.model).eval()
