import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from terraspectra_pipeline import __version__
from terraspectra_pipeline.cli import app

from .helpers import write_multiband

runner = CliRunner()


def test_synthetic_then_validate(tmp_path: Path) -> None:
    cube = tmp_path / "cube.tif"
    result = runner.invoke(app, ["synthetic", str(cube), "--size", "64"])
    assert result.exit_code == 0, result.output
    assert cube.exists()
    result = runner.invoke(app, ["validate", str(cube)])
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_fails_on_bad_file(tmp_path: Path) -> None:
    bad = write_multiband(tmp_path / "rgb.tif", np.zeros((3, 4, 4), np.float32))
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "FAIL" in result.output and "band count" in result.output


def test_info_process_indices_stats(tmp_path: Path, synthetic_cog: Path) -> None:
    result = runner.invoke(app, ["--plain-logs", "info", str(synthetic_cog)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["bands"] == 200

    stats = tmp_path / "stats.json"
    stats_cmd = ["stats", str(synthetic_cog), str(stats), "--windows", "2"]
    assert runner.invoke(app, stats_cmd).exit_code == 0

    out = tmp_path / "processed.tif"
    zarr_out = tmp_path / "processed.zarr"
    result = runner.invoke(
        app,
        [
            "--log-level",
            "WARNING",
            "process",
            str(synthetic_cog),
            str(out),
            "--no-cloud-mask",
            "--stats",
            str(stats),
            "--zarr",
            str(zarr_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output.strip().splitlines()[-1])["size"] == [64, 64]
    assert zarr_out.exists()

    idx = tmp_path / "indices.tif"
    result = runner.invoke(app, ["indices", str(out), str(idx), "--names", "ndvi,rep"])
    assert result.exit_code == 0, result.output
    assert idx.exists()
    assert runner.invoke(app, ["indices", str(out), str(idx), "--names", "foo"]).exit_code != 0


def test_ingest_local_and_version(tmp_path: Path, synthetic_cog: Path) -> None:
    result = runner.invoke(app, ["ingest", str(synthetic_cog), "--dest", str(tmp_path / "raw")])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["sensor"] == "generic"
    assert runner.invoke(app, ["ingest", "usgs"]).exit_code != 0
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output
