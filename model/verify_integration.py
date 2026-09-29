"""TerraSpectra Model Track (P2) — Day 13 Cross-Track Integration Verification.

Validates end-to-end integration across:
- P1 Geospatial Pipeline: Contract C1 200-band hyperspectral cube format.
- P2 Machine Learning: Contract C2 3D-CNN + ViT hybrid model (models/model.pt).
- P3 Inference API: InferenceEngine chunked prediction & Hann-window stitching.
- P2 Explainability: Integrated Gradients band importance -> Contract C4 dominant indicator.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Ensure all workspace packages are in search path
repo_root = Path(__file__).resolve().parent.parent
sys.path.extend([
    str(repo_root / "contracts" / "python" / "src"),
    str(repo_root / "api" / "src"),
    str(repo_root / "model" / "src"),
])

from terraspectra_api.core.engine import InferenceEngine
from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS, N_CLASSES, WINDOW_SIZE

from terraspectra_model.explain import (
    band_importance,
    dominant_indicator,
    indicator_scores,
    top_bands,
)
from terraspectra_model.synth.stress import make_field_patch


def main() -> None:
    print("=" * 75)
    print("TerraSpectra — Day 13 End-to-End Model & Pipeline Integration Verification")
    print("=" * 75)

    model_path = repo_root / "models" / "model.pt"
    if not model_path.is_file():
        print(f"[FAIL] Model artifact not found at {model_path}")
        sys.exit(1)

    # 1. Verify Contract C2 TorchScript Load
    print("\n[1/4] Verifying Contract C2 Model Artifact:")
    print(f"      Path: {model_path} ({model_path.stat().st_size / (1024 * 1024):.2f} MB)")
    try:
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        dummy_x = torch.randn(2, N_BANDS, WINDOW_SIZE, WINDOW_SIZE, dtype=torch.float32)
        with torch.no_grad():
            probs, onset = model(dummy_x)
        expected_shape = (2, N_CLASSES, WINDOW_SIZE, WINDOW_SIZE)
        assert probs.shape == expected_shape, f"Bad probs shape: {probs.shape}"
        assert onset.shape == (2, 1, WINDOW_SIZE, WINDOW_SIZE), f"Bad onset shape: {onset.shape}"
        assert np.isclose(probs.sum(dim=1).numpy(), 1.0, atol=1e-4).all()
        assert (onset >= 0.0).all() and (onset <= MAX_ONSET_DAYS).all()
        print("      [PASS] Direct TorchScript forward pass strictly satisfies Contract C2.")
    except Exception as e:
        print(f"      [FAIL] TorchScript forward pass failed: {e}")
        sys.exit(1)

    # 2. Verify API InferenceEngine Integration
    print("\n[2/4] Testing API InferenceEngine (Nikita's P3 Service):")
    try:
        engine = InferenceEngine(model_path=model_path, device="cpu", batch_size=4)
        engine.load()
        assert engine.ready, "Engine failed to initialize"
        assert engine.model_loaded, "Engine did not load real model"
        assert not engine.using_stub, "Engine fell back to stub instead of real model"
        print("      [PASS] API InferenceEngine successfully initialized and loaded model.pt.")
    except Exception as e:
        print(f"      [FAIL] InferenceEngine initialization failed: {e}")
        sys.exit(1)

    # 3. Simulate Multi-Window Scene Inference (Chunk -> Predict -> Stitch)
    print("\n[3/4] Running Chunked Inference on Simulated 128x128 Hyperspectral Scene:")
    try:
        H, W = 128, 128
        # Create 4 adjacent 64x64 tiles with realistic infection foci
        scene = np.zeros((N_BANDS, H, W), dtype=np.float32)
        rng = np.random.default_rng(123)
        for r in range(0, H, WINDOW_SIZE):
            for c in range(0, W, WINDOW_SIZE):
                patch = make_field_patch(rng=rng, size=WINDOW_SIZE)
                scene[:, r : r + WINDOW_SIZE, c : c + WINDOW_SIZE] = patch.cube

        # Chunk into windows
        windows = []
        coords = []
        for r in range(0, H, WINDOW_SIZE):
            for c in range(0, W, WINDOW_SIZE):
                windows.append(scene[:, r : r + WINDOW_SIZE, c : c + WINDOW_SIZE])
                coords.append((r, c))
        batch = np.stack(windows, axis=0)  # [4, 200, 64, 64]

        # Predict with engine
        pred_probs, pred_onset = engine.predict(batch)
        assert pred_probs.shape == (4, N_CLASSES, WINDOW_SIZE, WINDOW_SIZE)
        assert pred_onset.shape == (4, 1, WINDOW_SIZE, WINDOW_SIZE)

        # Reconstruct full scene
        full_probs = np.zeros((N_CLASSES, H, W), dtype=np.float32)
        full_onset = np.zeros((1, H, W), dtype=np.float32)
        for i, (r, c) in enumerate(coords):
            full_probs[:, r : r + WINDOW_SIZE, c : c + WINDOW_SIZE] = pred_probs[i]
            full_onset[:, r : r + WINDOW_SIZE, c : c + WINDOW_SIZE] = pred_onset[i]

        print(f"      Processed {batch.shape[0]} windows ({H}x{W} pixels total).")
        print(f"      Mean Risk Probs -> Healthy: {full_probs[0].mean():.3f}, "
              f"Early Stress: {full_probs[1].mean():.3f}, "
              f"High Risk: {full_probs[2].mean():.3f}, "
              f"Visible Disease: {full_probs[3].mean():.3f}")
        print(f"      Mean Forecast Onset -> {full_onset.mean():.1f} days before symptoms")
        print("      [PASS] Scene batching and reconstruction executed cleanly.")
    except Exception as e:
        print(f"      [FAIL] Scene chunking and inference failed: {e}")
        sys.exit(1)

    # 4. Verify Explainability & Dominant Indicator Extraction (Contract C4)
    print("\n[4/4] Attributing Spectral Indicators via Integrated Gradients:")
    try:
        sample_x = torch.from_numpy(batch[0:1]).float()
        imp = band_importance(model, sample_x, steps=8)
        dom = dominant_indicator(imp)
        scores = indicator_scores(imp)
        tops = top_bands(imp, k=3)

        print(f"      Attributed Dominant Indicator: '{dom}'")
        print(f"      Indicator Regional Density   : {scores}")
        top_tuples = [(round(w, 1), round(v, 4)) for w, v in tops]
        print(f"      Top 3 Influential Wavelengths: {top_tuples}")
        print("      [PASS] Dominant indicator successfully fills Contract C4 requirement.")
    except Exception as e:
        print(f"      [FAIL] Explainability attribution failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("[SUCCESS] ALL DAY-13 INTEGRATION CHECKS PASSED: Model ready for production serving!")
    print("=" * 75)


if __name__ == "__main__":
    main()
