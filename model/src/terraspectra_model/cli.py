"""``terraspectra-model`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from terraspectra_model.config import Config, load_config

if TYPE_CHECKING:
    from terraspectra_model.arch.hybrid import TerraSpectraNet

app = typer.Typer(
    name="terraspectra-model",
    help="TerraSpectra P2: train, evaluate and export the 3D-CNN + ViT crop-risk model.",
    no_args_is_help=True,
    add_completion=False,
)
log = logging.getLogger("terraspectra_model")

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML config (default: built-in defaults).")
]
OverrideOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Dotted override, e.g. -s train.lr=1e-3 (repeatable)."),
]
CheckpointOpt = Annotated[
    Path | None, typer.Option("--checkpoint", help="Lightning .ckpt (random weights if omitted).")
]


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Lightning installs its own console handler; avoid duplicated lines.
    logging.getLogger("lightning.pytorch").propagate = False


def _load(config: Path | None, overrides: list[str] | None) -> Config:
    return load_config(config, overrides)


def _model(cfg: Config, checkpoint: Path | None) -> TerraSpectraNet:
    from terraspectra_model.arch.hybrid import TerraSpectraNet
    from terraspectra_model.train import load_module

    if checkpoint is None:
        log.warning("no --checkpoint given: using randomly initialised weights")
        return TerraSpectraNet(cfg.model).eval()
    return load_module(checkpoint, cfg).model.eval()


@app.command()
def train(config: ConfigOpt = None, overrides: OverrideOpt = None) -> None:
    """Train the model (checkpoints under ``train.output_dir``)."""
    from terraspectra_model.train import train as run_train

    cfg = _load(config, overrides)
    _, best = run_train(cfg)
    typer.echo(f"best checkpoint: {best}")


@app.command()
def evaluate(
    config: ConfigOpt = None,
    overrides: OverrideOpt = None,
    checkpoint: CheckpointOpt = None,
    output: Annotated[Path, typer.Option(help="JSON report path.")] = Path("reports/metrics.json"),
    lead_time: Annotated[bool, typer.Option(help="Also compute the lead-time curve.")] = True,
    calibrate: Annotated[bool, typer.Option(help="Fit a softmax temperature.")] = False,
) -> None:
    """Write a metrics report (OA, macro-F1, mIoU, onset MAE, lead-time curve) as JSON."""
    from terraspectra_model.data.windows import TerraSpectraDataModule
    from terraspectra_model.evaluate import (
        evaluate_model,
        fit_temperature,
        lead_time_curve,
        write_report,
    )

    cfg = _load(config, overrides)
    model = _model(cfg, checkpoint)
    dm = TerraSpectraDataModule(cfg.data)
    dm.setup("validate")
    report = evaluate_model(model, dm.val_dataloader(), collect_logits=calibrate)
    if calibrate and "_logits" in report:
        t = fit_temperature(report["_logits"], report["_targets"])
        report["temperature"] = t
        model.set_temperature(t)
    if lead_time:
        report["lead_time_curve"] = lead_time_curve(model, cfg=cfg.data.synth)
    report["checkpoint"] = str(checkpoint) if checkpoint else None
    path = write_report(report, output)
    typer.echo(f"report written to {path}")
    typer.echo(
        json.dumps({k: report[k] for k in ("overall_accuracy", "macro_f1", "miou",
                                           "onset_mae_days")}, indent=2)
    )  # fmt: skip


@app.command()
def export(
    config: ConfigOpt = None,
    overrides: OverrideOpt = None,
    checkpoint: CheckpointOpt = None,
    output: Annotated[
        Path | None, typer.Option(help="model.pt path (default export.output_path).")
    ] = None,
    fp16: Annotated[bool | None, typer.Option(help="Also write an FP16 twin.")] = None,
    onnx: Annotated[bool | None, typer.Option(help="Also write model.onnx.")] = None,
    temperature: Annotated[float | None, typer.Option(help="Calibration temperature.")] = None,
) -> None:
    """Export the C2 TorchScript artifact (and optional FP16 / ONNX twins), verifying parity."""
    from terraspectra_model.export import export_onnx, export_torchscript, verify_parity

    cfg = _load(config, overrides)
    e = cfg.export
    out = output or e.output_path
    model = _model(cfg, checkpoint)
    model.set_temperature(temperature if temperature is not None else e.temperature)
    verify_parity(model, export_torchscript(model, out), atol=e.parity_atol)
    typer.echo(f"wrote {out}")
    if fp16 if fp16 is not None else e.fp16:
        p16 = out.with_name(out.stem + "_fp16" + out.suffix)
        verify_parity(model, export_torchscript(model, p16, fp16=True), atol=1e-2)
        typer.echo(f"wrote {p16}")
    if onnx if onnx is not None else e.onnx:
        p = export_onnx(model, out.with_suffix(".onnx"), e.onnx_opset)
        if p is not None:
            verify_parity(model, p, atol=e.parity_atol)
            typer.echo(f"wrote {p}")


@app.command()
def synth(
    output: Annotated[Path, typer.Option(help="Output .npz path.")] = Path("data/synth/train.npz"),
    n: Annotated[int, typer.Option(help="Number of 64x64 windows.")] = 256,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    config: ConfigOpt = None,
    overrides: OverrideOpt = None,
) -> None:
    """Dump a synthetic stress dataset (keys x, y, onset, mask) to .npz."""
    import numpy as np

    from terraspectra_model.synth.stress import generate_dataset

    cfg = _load(config, overrides)
    arrays = generate_dataset(n, seed, cfg.data.synth)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=arrays["x"],
        y=arrays["y"],
        onset=arrays["onset"],
        mask=arrays["mask"],
    )
    counts = np.bincount(arrays["y"].ravel(), minlength=4).tolist()
    typer.echo(f"wrote {output}: x{arrays['x'].shape} class pixel counts={counts}")


@app.command()
def benchmark(
    model_path: Annotated[
        Path | None, typer.Option("--model", help="TorchScript file (default: eager model).")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Windows per batch.")] = 32,
    n_batches: Annotated[int, typer.Option(help="Timed batches.")] = 10,
    device: Annotated[str, typer.Option(help="auto | cpu | cuda")] = "auto",
    config: ConfigOpt = None,
    overrides: OverrideOpt = None,
) -> None:
    """Measure inference throughput (windows/sec)."""
    from terraspectra_model.export import benchmark as run_benchmark

    target = model_path if model_path is not None else _model(_load(config, overrides), None)
    typer.echo(json.dumps(run_benchmark(target, batch_size, n_batches, device), indent=2))


@app.command()
def explain(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input", "-i",
            help="Path to window .npz or .npy. If omitted, a synthetic sample is generated.",
        ),
    ] = None,
    checkpoint: CheckpointOpt = None,
    config: ConfigOpt = None,
    overrides: OverrideOpt = None,
    steps: Annotated[int, typer.Option(help="Integrated gradients approximation steps.")] = 16,
    k_bands: Annotated[int, typer.Option(help="Number of top wavelengths to report.")] = 5,
) -> None:
    """Explain spectral stress predictions using Integrated Gradients & C4 dominant indicator."""
    import numpy as np
    import torch
    from terraspectra_contracts import N_BANDS, WINDOW_SIZE

    from terraspectra_model.explain import (
        band_importance,
        dominant_indicator,
        indicator_scores,
        top_bands,
    )
    from terraspectra_model.synth.stress import make_field_patch

    cfg = _load(config, overrides)
    model = _model(cfg, checkpoint)

    if input_path is not None and input_path.is_file():
        if input_path.suffix == ".npz":
            data = np.load(input_path)
            x = data["x"][0] if data["x"].ndim == 4 else data["x"]
        else:
            x = np.load(input_path)
    else:
        patch = make_field_patch(
            rng=np.random.default_rng(42), cfg=cfg.data.synth, size=WINDOW_SIZE
        )
        x = patch.cube

    if x.ndim == 3 and x.shape[0] == N_BANDS:
        x_tensor = torch.from_numpy(x[None, ...]).float()
    elif x.ndim == 4:
        x_tensor = torch.from_numpy(x).float()
    else:
        raise typer.BadParameter(
            f"Expected array with shape (200, 64, 64) or (N, 200, 64, 64), got {x.shape}"
        )

    with torch.no_grad():
        probs, onset = model(x_tensor)

    imp = band_importance(model, x_tensor, steps=steps)
    dom = dominant_indicator(imp)
    scores = indicator_scores(imp)
    tops = top_bands(imp, k=k_bands)

    result = {
        "dominant_indicator": dom,
        "indicator_scores": {k: round(v, 4) for k, v in scores.items()},
        "top_contributing_wavelengths_nm": [
            {"wavelength_nm": round(wl, 1), "relative_importance": round(imp_val, 4)}
            for wl, imp_val in tops
        ],
        "mean_risk_probabilities": {
            "healthy": round(float(probs[:, 0].mean()), 4),
            "early_stress": round(float(probs[:, 1].mean()), 4),
            "high_blight_risk": round(float(probs[:, 2].mean()), 4),
            "visible_disease": round(float(probs[:, 3].mean()), 4),
        },
        "mean_predicted_onset_days": round(float(onset.mean()), 1),
    }
    typer.echo(json.dumps(result, indent=2))


@app.command(name="download-benchmarks")
def download_benchmarks_cmd(
    benchmark: Annotated[
        str,
        typer.Option(
            "--name", "-n", help="Benchmark name (indian_pines, salinas, pavia_u, or all)."
        ),
    ] = "all",
    data_dir: Annotated[
        Path, typer.Option("--data-dir", "-d", help="Directory to save benchmark datasets.")
    ] = Path("data/benchmarks"),
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Force redownload existing files.")
    ] = False,
) -> None:
    """Download public benchmark datasets (Indian Pines, Salinas, Pavia U)."""
    from terraspectra_model.data.benchmarks import (
        BENCHMARKS,
        download_all_benchmarks,
        download_benchmark,
    )

    if benchmark == "all":
        typer.echo(f"Downloading all benchmarks to {data_dir}...")
        results = download_all_benchmarks(data_dir, force=force)
        for name, (cube, gt) in results.items():
            typer.echo(f"  [OK] {name}: {cube.name}, {gt.name}")
    elif benchmark in BENCHMARKS:
        typer.echo(f"Downloading {benchmark} to {data_dir}...")
        cube, gt = download_benchmark(benchmark, data_dir, force=force)
        typer.echo(f"  [OK] {benchmark}: {cube.name}, {gt.name}")
    else:
        valid = ", ".join(sorted(BENCHMARKS.keys()))
        raise typer.BadParameter(f"Unknown benchmark {benchmark!r}. Choose from: all, {valid}")


if __name__ == "__main__":  # pragma: no cover
    app()
