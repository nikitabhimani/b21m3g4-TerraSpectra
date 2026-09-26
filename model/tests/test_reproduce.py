"""Unit tests for Day 15 Training Reproducibility and Demo Rehearsal."""

# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure model root is in search path
model_dir = Path(__file__).resolve().parent.parent
if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

from demo import run_demo
from reproduce import run_reproducible_pipeline


def test_reproducibility_pipeline_quick(tmp_path: Path) -> None:
    """Verify that the quick reproducibility pipeline executes end-to-end and passes."""
    report_dir = tmp_path / "repro"
    export_path = report_dir / "test_model.pt"

    report = run_reproducible_pipeline(
        quick=True,
        seed=123,
        output_dir=report_dir,
        export_path=export_path,
    )

    assert report["status"] == "PASSED"
    assert report["contract_c2_verification"]["contract_c2_passed"] is True
    assert report["contract_c2_verification"]["numerical_parity_passed"] is True
    assert export_path.is_file()

    # Check generated JSON report
    report_file = report_dir / "reproducibility_report.json"
    assert report_file.is_file()
    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved["status"] == "PASSED"


def test_demo_rehearsal_run(tmp_path: Path) -> None:
    """Verify demo rehearsal script runs cleanly and generates structured outputs."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    model_path = repo_root / "models" / "model.pt"
    demo_out = tmp_path / "demo_output.json"

    run_demo(
        model_path=model_path,
        seed=42,
        steps=5,
        output_path=demo_out,
    )

    assert demo_out.is_file()
    demo_data = json.loads(demo_out.read_text(encoding="utf-8"))
    assert "dominant_indicator" in demo_data
    assert "forecast_lead_time_days" in demo_data
    assert "risk_breakdown_percent" in demo_data
