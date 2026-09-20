from pathlib import Path

import pytest
from pydantic import ValidationError
from terraspectra_contracts import N_BANDS

from terraspectra_model.config import Config, load_config, parse_overrides

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["default.yaml", "tiny.yaml"])
def test_yaml_configs_load(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert isinstance(cfg, Config)
    assert cfg.model.in_bands == N_BANDS
    assert cfg.data.window_size == 64


def test_default_yaml_matches_code_defaults_for_architecture() -> None:
    assert load_config(CONFIG_DIR / "default.yaml").model == Config().model


def test_overrides() -> None:
    cfg = load_config(CONFIG_DIR / "tiny.yaml", ["train.lr=0.01", "data.batch_size=3"])
    assert cfg.train.lr == 0.01
    assert cfg.data.batch_size == 3
    assert parse_overrides(["a.b.c=[1, 2]"]) == {"a": {"b": {"c": [1, 2]}}}


@pytest.mark.parametrize(
    "override",
    ["model.in_bands=100", "data.window_size=32", "model.patch_size=6", "train.nope=1"],
)
def test_invalid_configs_rejected(override: str) -> None:
    with pytest.raises((ValidationError, ValueError)):
        load_config(CONFIG_DIR / "tiny.yaml", [override])
