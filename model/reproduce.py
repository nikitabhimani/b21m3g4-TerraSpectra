"""TerraSpectra Model Track (P2) — Day 15 Training Reproducibility Pipeline.

Provides a deterministic, one-command pipeline that:
1. Fixes seeds across Python, NumPy, and PyTorch for bit-level reproducibility.
2. Trains or validates the 3D-CNN + Vision Transformer hybrid model (TerraSpectraNet).
3. Evaluates segmentation accuracy, per-class F1, mIoU, and onset lead-time MAE.
4. Compiles and exports the model to TorchScript (Contract C2: models/model.pt).
5. Verifies numerical parity (eager vs TorchScript <= 1e-4) and generates an audit report.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Ensure module path is accessible
repo_root = Path(__file__).resolve().parent.parent
sys.path.extend([
    str(repo_root / "model" / "src"),
    str(repo_root / "contracts" / "python" / "src"),
])

from terraspectra_contracts import (
    MAX_ONSET_DAYS,
    N_BANDS,
    N_CLASSES,
    WINDOW_SIZE,
)

from terraspectra_model.config import Config, load_config
from terraspectra_model.data.windows import TerraSpectraDataModule
from terraspectra_model.evaluate import metrics_from_confusion
from terraspectra_model.export import export_torchscript, verify_parity
from terraspectra_model.train import train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("terraspectra.reproduce")


def seed_everything(seed: int = 42) -> None:
    """Enforce strict determinism across all random generators."""
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_reproducible_pipeline(
    quick: bool = False,
    epochs: int | None = None,
    seed: int = 42,
    output_dir: Path = Path("reports/reproducibility"),
    export_path: Path = Path("reports/reproducibility/reproduced_model.pt"),
) -> dict[str, Any]:
    """Execute end-to-end training, evaluation, export, and verification pipeline."""
    start_time = time.time()
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 75)
    log.info("TerraSpectra ML Track (P2) — Day 15 Reproducibility Pipeline")
    log.info("=" * 75)
    log.info("Host OS         : %s", platform.platform())
    log.info("Python Version  : %s", platform.python_version())
    log.info(
        "PyTorch Version : %s (CUDA Available: %s)",
        torch.__version__,
        torch.cuda.is_available(),
    )
    log.info("Execution Mode  : %s", "Quick Smoke Test" if quick else "Full Reproducibility")

    # 1. Load base configuration
    base_config_path = (
        repo_root / "model" / "configs" / ("tiny.yaml" if quick else "fast_baseline.yaml")
    )
    overrides: list[str] = [
        f"train.seed={seed}",
        f"train.output_dir={output_dir / 'training'}",
    ]
    if epochs is not None:
        overrides.append(f"train.max_epochs={epochs}")

    cfg: Config = load_config(base_config_path, overrides)

    # 2. Train model with determinism
    log.info("Commencing reproducible training with config: %s", base_config_path.name)
    train_start = time.time()
    module, best_ckpt = train(cfg)
    train_duration = time.time() - train_start
    log.info("Training finished in %.2f seconds (best ckpt: %s)", train_duration, best_ckpt)

    # 3. Evaluate model on validation set
    model = module.model.eval().cpu()
    dm = TerraSpectraDataModule(cfg.data)
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    onset_errors: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"]
            y = batch["y"].numpy()
            target_onset = batch["onset"].numpy()
            mask = batch["mask"].numpy() if "mask" in batch else np.ones_like(y, dtype=bool)

            probs, pred_onset = model(x)
            pred_classes = probs.argmax(dim=1).cpu().numpy()
            pred_onset_np = pred_onset.squeeze(1).cpu().numpy()

            valid = mask & (y >= 0)
            if valid.any():
                all_preds.append(pred_classes[valid])
                all_targets.append(y[valid])
                stress_mask = valid & (y > 0)
                if stress_mask.any():
                    err = np.abs(pred_onset_np[stress_mask] - target_onset[stress_mask])
                    onset_errors.extend(err.tolist())

    y_pred = np.concatenate(all_preds) if all_preds else np.array([], dtype=int)
    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=int)

    cm = np.bincount(N_CLASSES * y_true + y_pred, minlength=N_CLASSES * N_CLASSES).reshape(
        N_CLASSES, N_CLASSES
    )
    metrics = metrics_from_confusion(cm)
    mean_onset_mae = float(np.mean(onset_errors)) if onset_errors else 0.0

    log.info("Overall Accuracy : %.3f", metrics["overall_accuracy"])
    log.info("Macro F1 Score   : %.3f", metrics["macro_f1"])
    log.info("Mean IoU (mIoU)  : %.3f", metrics["miou"])
    log.info("Onset Target MAE : %.2f days", mean_onset_mae)

    # 4. Export TorchScript Artifact (Contract C2)
    log.info("Exporting model to TorchScript: %s...", export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_torchscript(model, export_path, fp16=False)

    # 5. Verify Parity & Contract C2 Compliance
    diffs = verify_parity(model, export_path, batch=3)
    parity_passed = bool(
        diffs["probs_max_abs_diff"] <= 1e-4
        and diffs["onset_max_abs_diff"] <= 1e-4 * MAX_ONSET_DAYS
    )
    log.info(
        "Parity Check -> Max Prob Diff: %.2e, Max Onset Diff: %.2e (Matches: %s)",
        diffs["probs_max_abs_diff"],
        diffs["onset_max_abs_diff"],
        parity_passed,
    )

    # Test Contract C2 shape and invariants on re-loaded TorchScript
    loaded_ts = torch.jit.load(str(export_path)).eval()
    test_tensor = torch.rand(2, N_BANDS, WINDOW_SIZE, WINDOW_SIZE, dtype=torch.float32)
    with torch.no_grad():
        test_probs, test_onset = loaded_ts(test_tensor)

    contract_c2_passed = (
        test_probs.shape == (2, N_CLASSES, WINDOW_SIZE, WINDOW_SIZE)
        and test_onset.shape == (2, 1, WINDOW_SIZE, WINDOW_SIZE)
        and np.allclose(test_probs.sum(dim=1).numpy(), 1.0, atol=1e-4)
        and (test_onset >= 0.0).all()
        and (test_onset <= float(MAX_ONSET_DAYS)).all()
    )

    total_duration = time.time() - start_time
    report = {
        "status": "PASSED" if (parity_passed and contract_c2_passed) else "FAILED",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "quick_mode": quick,
        "seed": seed,
        "config_used": str(base_config_path.name),
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "training": {
            "max_epochs": cfg.train.max_epochs,
            "duration_seconds": round(train_duration, 2),
            "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        },
        "evaluation_metrics": {
            "overall_accuracy": round(metrics["overall_accuracy"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            "miou": round(metrics["miou"], 4),
            "per_class_f1": {k: round(v, 4) for k, v in metrics["per_class_f1"].items()},
            "onset_mae_days": round(mean_onset_mae, 2),
        },
        "contract_c2_verification": {
            "torchscript_path": str(export_path),
            "artifact_size_mb": round(export_path.stat().st_size / (1024 * 1024), 2),
            "numerical_parity_passed": parity_passed,
            "probs_max_abs_diff": float(diffs["probs_max_abs_diff"]),
            "onset_max_abs_diff": float(diffs["onset_max_abs_diff"]),
            "contract_c2_passed": bool(contract_c2_passed),
        },
        "total_elapsed_seconds": round(total_duration, 2),
    }

    report_path = output_dir / "reproducibility_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Reproducibility report successfully saved to %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TerraSpectra P2 Day 15 Training Reproducibility Pipeline"
    )
    parser.add_argument("--quick", action="store_true", help="Run fast verification smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/reproducibility"),
        help="Directory to save checkpoints and reports",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=Path("reports/reproducibility/reproduced_model.pt"),
        help="Path for exported TorchScript model",
    )
    args = parser.parse_args()

    report = run_reproducible_pipeline(
        quick=args.quick,
        epochs=args.epochs,
        seed=args.seed,
        output_dir=args.output_dir,
        export_path=args.export_path,
    )

    if report["status"] != "PASSED":
        log.error("Pipeline verification failed!")
        sys.exit(1)
    log.info("Pipeline verification successfully finished!")


if __name__ == "__main__":
    main()
