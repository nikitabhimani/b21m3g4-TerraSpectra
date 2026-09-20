from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from terraspectra_model.cli import app
from terraspectra_model.config import Config
from terraspectra_model.lightning_module import warmup_cosine
from terraspectra_model.train import train


def test_warmup_cosine_schedule() -> None:
    assert warmup_cosine(0, 100, 10, 0.1) == 0.1
    assert warmup_cosine(10, 100, 10, 0.1) == 1.0
    assert abs(warmup_cosine(100, 100, 10, 0.1) - 0.1) < 1e-9


def test_fast_dev_run_smoke(tiny_cfg: Config) -> None:
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.train.fast_dev_run = 2
    module, _ = train(cfg)
    assert module.trainer.state.finished


def test_cli_synth_and_npz_training(tiny_cfg: Config, tmp_path: Path) -> None:
    out = tmp_path / "syn.npz"
    res = CliRunner().invoke(app, ["synth", "--n", "3", "--output", str(out)])
    assert res.exit_code == 0, res.output
    with np.load(out) as d:
        assert d["x"].shape == (3, 200, 64, 64)
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.data.source = "npz"
    cfg.data.npz_paths = [out]
    cfg.train.fast_dev_run = 1
    module, _ = train(cfg)
    assert module.trainer.state.finished
