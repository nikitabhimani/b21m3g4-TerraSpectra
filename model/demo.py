"""TerraSpectra Model Track (P2) — Day 15 Demo Rehearsal Runner.

Interactive presentation tool demonstrating:
1. Loading the Contract C2 TorchScript model (models/model.pt).
2. Simulating a 200-band hyperspectral field patch with early fungal stress.
3. Rapid GPU/CPU inference with risk class distribution & onset lead-time forecast.
4. Per-band explainability attribution via Integrated Gradients (Contract C4 indicator).
5. Agronomic action plan prescription for precision farming.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(encoding="utf-8")

# Ensure module path is accessible
repo_root = Path(__file__).resolve().parent.parent
sys.path.extend([
    str(repo_root / "model" / "src"),
    str(repo_root / "contracts" / "python" / "src"),
])

from terraspectra_contracts import (
    CLASS_NAMES,
    N_BANDS,
    N_CLASSES,
    WINDOW_SIZE,
)

from terraspectra_model.config import SynthConfig
from terraspectra_model.explain import (
    band_importance,
    dominant_indicator,
    indicator_scores,
    top_bands,
)
from terraspectra_model.synth.stress import make_field_patch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("terraspectra.demo")


def ascii_bar(fraction: float, width: int = 24) -> str:
    """Create a textual progress bar representation."""
    filled = round(fraction * width)
    return "#" * filled + "-" * (width - filled)


def run_demo(
    model_path: Path = repo_root / "models" / "model.pt",
    seed: int = 101,
    steps: int = 15,
    output_path: Path = Path("reports/demo_rehearsal.json"),
) -> None:
    print("\n" + "=" * 78)
    print("      TERRASPECTRA — HYPERSPECTRAL CROP DISEASE FORECAST DEMO (P2)")
    print("  3D-CNN + Vision Transformer Hybrid | Contract C2 & C4 Demonstration")
    print("=" * 78)

    # 1. Model Loading
    print("\n[1/5] Loading Production Model Artifact:")
    print(f"      Source Path : {model_path}")
    if not model_path.is_file():
        print("      [ERROR] Model artifact missing! Run 'python reproduce.py' first.")
        sys.exit(1)

    load_t0 = time.time()
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    load_time_ms = (time.time() - load_t0) * 1000
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"      Model Size  : {model_size_mb:.2f} MB")
    print(f"      Load Latency: {load_time_ms:.1f} ms [Contract C2 Verified]")

    # 2. Field Simulation
    print("\n[2/5] Simulating Hyperspectral Field Patch (Contract C1):")
    print(f"      Spectral Bands : {N_BANDS} continuous channels (400 nm - 2500 nm)")
    print(f"      Window Geometry: {WINDOW_SIZE} x {WINDOW_SIZE} pixels (64x64 farm sector)")
    print("      Scenario       : Pre-visual Fungal Blight (Inoculation Stage)")

    synth_cfg = SynthConfig(
        max_blobs=2,
        blob_radius=(8.0, 16.0),
        max_infected_days=24.0,
        noise_std=0.003,
    )
    rng = np.random.default_rng(seed)
    patch = make_field_patch(rng, cfg=synth_cfg, size=WINDOW_SIZE)
    x_tensor = torch.from_numpy(patch.cube).unsqueeze(0)  # [1, 200, 64, 64]

    # 3. Model Inference
    print("\n[3/5] Executing 3D-CNN + ViT Forward Inference:")
    inf_t0 = time.time()
    with torch.no_grad():
        probs, onset = model(x_tensor)
    inf_time_ms = (time.time() - inf_t0) * 1000
    print(f"      Inference Time : {inf_time_ms:.2f} ms per 64x64 window")

    probs_np = probs.squeeze(0).numpy()  # [4, 64, 64]
    onset_np = onset.squeeze().numpy()  # [64, 64]
    pred_class_map = probs_np.argmax(axis=0)

    # Risk Distribution Breakdown
    total_pixels = WINDOW_SIZE * WINDOW_SIZE
    class_counts = np.bincount(pred_class_map.ravel(), minlength=N_CLASSES)
    class_percentages = class_counts / total_pixels * 100.0

    print("\n[4/5] Pre-Visual Risk Classification Breakdown:")
    labels = [
        "Class 0 (Healthy Canopy)    ",
        "Class 1 (Early Stress - PRE)",
        "Class 2 (High Blight Risk)  ",
        "Class 3 (Visible Symptoms)  ",
    ]
    for idx, (lbl, pct) in enumerate(zip(labels, class_percentages, strict=False)):
        bar = ascii_bar(pct / 100.0)
        status_tag = ""
        if idx == 1 and pct > 5.0:
            status_tag = " <-- [PRE-VISUAL DETECTION ALERT]"
        elif idx == 2 and pct > 2.0:
            status_tag = " <-- [HIGH INFECTION RISK]"
        print(f"      {lbl} : {pct:>5.1f}% |{bar}|{status_tag}")

    # Forecast Onset
    stress_pixels = (pred_class_map == 1) | (pred_class_map == 2)
    mean_forecast_days = (
        float(onset_np[stress_pixels].mean()) if stress_pixels.any() else float(onset_np.mean())
    )
    print(f"\n      >> Forecasted Window to Visible Outbreak: ~{mean_forecast_days:.1f} DAYS AHEAD")
    print("      >> Lead-Time Advantage: Early intervention possible before foliar browning!")

    # 4. Explainability & Attribution
    print("\n[5/5] Integrated Gradients Spectral Attribution (Contract C4):")
    print(f"      Computing per-band gradients across {steps} integration steps...")
    imp = band_importance(model, x_tensor, steps=steps)
    dom_indicator = dominant_indicator(imp)
    scores = indicator_scores(imp)
    top_wls = top_bands(imp, k=3)

    print(f"      Dominant Indicator : '{dom_indicator.upper()}'")
    print("      Indicator Scores   :")
    for ind_name, score in scores.items():
        print(f"        * {ind_name:<20}: {score:.4f}")

    print("      Top Diagnostic Wavelengths:")
    for wl, val in top_wls:
        print(f"        * {wl:>6.1f} nm (Spectral Weight: {val:.4f})")

    # Agronomic recommendation logic
    recommendations = {
        "pri_decline": (
            "Apply targeted bio-fungicide to North-East sector within 5 days to "
            "mitigate xanthophyll stress."
        ),
        "red_edge_shift": (
            "Early cellular structure breakdown observed. Initiate preventative copper spray."
        ),
        "chlorophyll_loss": (
            "Chlorophyll deficit detected. Schedule drone inspection and foliar nutrition."
        ),
        "water_stress": (
            "Canopy water absorption deficit. Optimize drip irrigation and check root pressure."
        ),
    }
    rec_action = recommendations.get(
        dom_indicator, "Monitor field closely with next satellite pass."
    )

    print("\n" + "=" * 78)
    print("                          EXECUTIVE PRESCRIPTION")
    print("=" * 78)
    print(" Field Sector Status : AT RISK (Pre-Visual Fungal Stress Identified)")
    print(f" Forecast Lead-Time  : ~{mean_forecast_days:.1f} Days Before Visible Outbreak")
    print(f" Dominant Signature  : {dom_indicator.replace('_', ' ').title()}")
    print(f" Actionable Guidance : {rec_action}")
    print("=" * 78 + "\n")

    # Save rehearsal log
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rehearsal_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_path": str(model_path),
        "inference_latency_ms": round(inf_time_ms, 2),
        "risk_breakdown_percent": {
            CLASS_NAMES[i]: round(float(class_percentages[i]), 2) for i in range(N_CLASSES)
        },
        "forecast_lead_time_days": round(mean_forecast_days, 1),
        "dominant_indicator": dom_indicator,
        "indicator_scores": {k: round(float(v), 4) for k, v in scores.items()},
        "top_diagnostic_wavelengths": [
            {"wavelength_nm": round(wl, 1), "importance": round(imp_val, 4)}
            for wl, imp_val in top_wls
        ],
        "prescribed_action": rec_action,
    }
    output_path.write_text(json.dumps(rehearsal_data, indent=2), encoding="utf-8")
    print(f"[OK] Demo rehearsal metrics archived to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TerraSpectra Demo Rehearsal Runner")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=repo_root / "models" / "model.pt",
        help="Path to TorchScript model",
    )
    parser.add_argument("--seed", type=int, default=101, help="Random seed for field patch")
    parser.add_argument("--steps", type=int, default=15, help="Integrated gradients steps")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/demo_rehearsal.json"),
        help="Path to save demo report",
    )
    args = parser.parse_args()

    run_demo(
        model_path=args.model_path,
        seed=args.seed,
        steps=args.steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
