import json

import numpy as np
import torch
from terraspectra_contracts import N_BANDS
from terraspectra_contracts.fixtures import make_synthetic_cube
from typer.testing import CliRunner

from terraspectra_model.arch.hybrid import TerraSpectraNet
from terraspectra_model.cli import app
from terraspectra_model.config import SynthConfig
from terraspectra_model.evaluate import (
    confusion_matrix,
    expected_calibration_error,
    fit_temperature,
    lead_time_curve,
    metrics_from_confusion,
)
from terraspectra_model.explain import (
    INDICATORS,
    band_importance,
    dominant_indicator,
    integrated_gradients,
    top_bands,
)


def test_integrated_gradients_band_importance(tiny_model: TerraSpectraNet) -> None:
    x = torch.rand(2, N_BANDS, 64, 64)
    region = torch.zeros(2, 64, 64, dtype=torch.bool)
    region[:, 10:30, 10:30] = True
    imp = band_importance(tiny_model, x, region=region, steps=4)
    assert imp.shape == (N_BANDS,)
    assert (imp >= 0).all() and np.isclose(imp.sum(), 1.0)
    assert dominant_indicator(imp) in INDICATORS
    assert len(top_bands(imp, 5)) == 5


def test_integrated_gradients_completeness(tiny_model: TerraSpectraNet) -> None:
    """Attributions sum approximately to f(x) - f(baseline)."""
    x = torch.from_numpy(np.clip(make_synthetic_cube(64, 64)[0], 0, 1))[None]
    attr = integrated_gradients(tiny_model, x, target_class=2, steps=16, batch_size=16)
    with torch.no_grad():
        fx = tiny_model(x)[0][:, 2].mean()
        f0 = tiny_model(torch.zeros_like(x))[0][:, 2].mean()
    assert abs(float(attr.sum()) - float(fx - f0)) < 0.1 * abs(float(fx - f0)) + 1e-3


def test_dominant_indicator_mapping() -> None:
    wl = np.linspace(400, 2500, N_BANDS)
    for nm, name in [(720, "red_edge_shift"), (531, "pri_decline"), (670, "chlorophyll_loss"),
                     (1650, "water_stress")]:  # fmt: skip
        imp = np.exp(-(((wl - nm) / 8) ** 2))
        assert dominant_indicator(imp / imp.sum()) == name


def test_metrics_and_calibration() -> None:
    pred = np.array([0, 1, 2, 3, 0])
    target = np.array([0, 1, 2, 0, -1])
    m = metrics_from_confusion(confusion_matrix(pred, target))
    assert m["overall_accuracy"] == 0.75
    logits = torch.randn(500, 4) * 5
    targets = torch.randint(0, 4, (500,))
    t = fit_temperature(logits, targets)
    assert t > 1.0  # random labels -> overconfident logits get softened
    ece = expected_calibration_error(torch.softmax(logits, 1), targets)
    assert 0 <= ece <= 1


def test_lead_time_curve(tiny_model: TerraSpectraNet) -> None:
    rows = lead_time_curve(tiny_model, days=[0, 20], n_per_day=1, cfg=SynthConfig())
    assert [r["days_before_symptoms"] for r in rows] == [0.0, 20.0]
    assert all(0 <= r["detection_rate"] <= 1 for r in rows)


def test_explain_cli() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["explain", "--steps", "4", "--k-bands", "3"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["dominant_indicator"] in INDICATORS
    assert len(payload["top_contributing_wavelengths_nm"]) == 3
    assert "mean_risk_probabilities" in payload
