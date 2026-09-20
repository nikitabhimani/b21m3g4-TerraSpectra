"""Export to the contract-C2 TorchScript artifact (plus optional FP16 / ONNX twins)."""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS, N_CLASSES, WINDOW_SIZE
from torch import nn

from terraspectra_model.arch.hybrid import TerraSpectraNet

log = logging.getLogger(__name__)

EXAMPLE_SHAPE = (1, N_BANDS, WINDOW_SIZE, WINDOW_SIZE)


class HalfPrecisionWrapper(nn.Module):
    """Runs a :class:`TerraSpectraNet` in FP16 but keeps the C2 float32 input/output dtypes.

    The softmax is computed in float32 so probabilities still sum to 1 within 1e-4.
    """

    def __init__(self, net: TerraSpectraNet) -> None:
        super().__init__()
        self.temperature = float(net.temperature)
        self.net = net.half()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """FP16 inference with float32 I/O."""
        logits, onset = self.net.forward_logits(x.half())
        return torch.softmax(logits.float() / self.temperature, dim=1), onset.float()


def _trace(module: nn.Module, device: torch.device) -> torch.jit.ScriptModule:
    example = torch.zeros(EXAMPLE_SHAPE, device=device)
    with torch.no_grad(), warnings.catch_warnings():
        # Input-shape checks are Python branches; they are evaluated once at trace time.
        warnings.simplefilter("ignore", torch.jit.TracerWarning)
        traced = torch.jit.trace(module.eval().to(device), example, check_trace=False)
    # Not frozen: freezing would bake the trace device in and break .to(device) at load time.
    return traced


def export_torchscript(model: nn.Module, path: str | Path, fp16: bool = False) -> Path:
    """Trace ``model`` on CPU with ``[1,200,64,64]`` and save it. Batch size stays dynamic.

    ``fp16=True`` wraps the model in :class:`HalfPrecisionWrapper` (traced on CUDA when available;
    intended for CUDA inference per C2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    if fp16:
        import copy

        if not isinstance(model, TerraSpectraNet):
            raise TypeError("fp16 export needs a TerraSpectraNet")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        traced = _trace(HalfPrecisionWrapper(copy.deepcopy(model)), device)
    else:
        traced = _trace(model, torch.device("cpu"))
    traced.save(str(path))
    log.info("saved TorchScript model to %s", path)
    return path


def export_onnx(model: nn.Module, path: str | Path, opset: int = 18) -> Path | None:
    """Export ONNX with a dynamic batch axis; returns ``None`` if ``onnx`` is not installed."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        log.warning("onnx not installed (uv sync --extra onnx); skipping ONNX export")
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = (torch.zeros(2, *EXAMPLE_SHAPE[1:]),)
    batch = torch.export.Dim("batch", min=1, max=4096)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=["x"],
            output_names=["probs", "onset"],
            opset_version=opset,
            dynamic_shapes={"x": {0: batch}},
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    log.info("saved ONNX model to %s", path)
    return path


def check_contract(probs: torch.Tensor, onset: torch.Tensor, batch: int) -> None:
    """Raise ``AssertionError`` unless outputs satisfy C2."""
    assert probs.shape == (batch, N_CLASSES, WINDOW_SIZE, WINDOW_SIZE), probs.shape
    assert onset.shape == (batch, 1, WINDOW_SIZE, WINDOW_SIZE), onset.shape
    assert probs.dtype == torch.float32 and onset.dtype == torch.float32
    assert torch.allclose(probs.sum(1), torch.ones_like(probs[:, 0]), atol=1e-4)
    assert float(onset.min()) >= 0.0 and float(onset.max()) <= MAX_ONSET_DAYS


@torch.no_grad()
def verify_parity(
    model: nn.Module,
    exported: str | Path,
    batch: int = 3,
    atol: float = 1e-4,
    seed: int = 0,
) -> dict[str, float]:
    """Compare eager vs TorchScript (``.pt``) or ONNX (``.onnx``) outputs on random input."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, *EXAMPLE_SHAPE[1:], generator=gen)
    model = model.eval().cpu()
    ref_p, ref_o = model(x)
    exported = Path(exported)
    if exported.suffix == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        p_np, o_np = sess.run(["probs", "onset"], {"x": x.numpy()})
        p, o = torch.from_numpy(p_np), torch.from_numpy(o_np)
    else:
        loaded = torch.jit.load(str(exported), map_location="cpu")
        p, o = loaded(x)
    check_contract(p, o, batch)
    diffs = {
        "probs_max_abs_diff": float((p - ref_p).abs().max()),
        "onset_max_abs_diff": float((o - ref_o).abs().max()),
    }
    # Onset is in days (0-30), so allow a proportionally larger tolerance.
    if diffs["probs_max_abs_diff"] > atol or diffs["onset_max_abs_diff"] > atol * MAX_ONSET_DAYS:
        raise AssertionError(f"parity check failed for {exported}: {diffs}")
    log.info("parity OK for %s: %s", exported, diffs)
    return diffs


@torch.no_grad()
def benchmark(
    model: nn.Module | str | Path,
    batch_size: int = 32,
    n_batches: int = 10,
    device: str = "auto",
    warmup: int = 2,
) -> dict[str, Any]:
    """Throughput in windows/sec for an eager module or a TorchScript file."""
    dev = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    if isinstance(model, str | Path):
        net: Any = torch.jit.load(str(model), map_location=dev)
    else:
        net = model.eval().to(dev)
    x = torch.rand(batch_size, *EXAMPLE_SHAPE[1:], device=dev)
    for _ in range(warmup):
        net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        net(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "device": str(dev),
        "batch_size": batch_size,
        "n_batches": n_batches,
        "seconds": dt,
        "windows_per_sec": batch_size * n_batches / dt,
    }
