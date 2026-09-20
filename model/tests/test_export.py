import os
from pathlib import Path

import pytest
import torch
from terraspectra_contracts import N_BANDS

from terraspectra_model.arch.hybrid import TerraSpectraNet
from terraspectra_model.export import (
    benchmark,
    check_contract,
    export_onnx,
    export_torchscript,
    verify_parity,
)


def test_torchscript_dynamic_batch(tiny_model: TerraSpectraNet, tmp_path: Path) -> None:
    path = export_torchscript(tiny_model, tmp_path / "model.pt")
    loaded = torch.jit.load(str(path), map_location="cpu")
    with torch.no_grad():
        probs, onset = loaded(torch.rand(3, N_BANDS, 64, 64))  # traced with batch 1
    check_contract(probs, onset, 3)


def test_export_parity(tiny_model: TerraSpectraNet, tmp_path: Path) -> None:
    tiny_model.set_temperature(1.7)
    path = export_torchscript(tiny_model, tmp_path / "model.pt")
    diffs = verify_parity(tiny_model, path, batch=2)
    assert diffs["probs_max_abs_diff"] < 1e-5


def test_fp16_export_keeps_float32_io(tiny_model: TerraSpectraNet, tmp_path: Path) -> None:
    try:
        path = export_torchscript(tiny_model, tmp_path / "model_fp16.pt", fp16=True)
    except RuntimeError as exc:  # pragma: no cover - depends on CPU fp16 kernel support
        pytest.skip(f"fp16 not supported here: {exc}")
    loaded = torch.jit.load(str(path))
    probs, onset = loaded(torch.rand(2, N_BANDS, 64, 64))
    check_contract(probs, onset, 2)


def test_benchmark_reports_throughput(tiny_model: TerraSpectraNet, tmp_path: Path) -> None:
    path = export_torchscript(tiny_model, tmp_path / "model.pt")
    res = benchmark(path, batch_size=2, n_batches=1, device="cpu", warmup=0)
    assert res["windows_per_sec"] > 0


@pytest.mark.skipif(not os.environ.get("TS_RUN_SLOW"), reason="slow (~20 s); set TS_RUN_SLOW=1")
def test_onnx_export_parity(tiny_model: TerraSpectraNet, tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    path = export_onnx(tiny_model, tmp_path / "model.onnx")
    assert path is not None
    verify_parity(tiny_model, path, batch=3)
