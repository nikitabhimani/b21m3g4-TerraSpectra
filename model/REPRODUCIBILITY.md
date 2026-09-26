# TerraSpectra ML Track (P2) — Model Training & Evaluation Reproducibility Guide

This document specifies the exact instructions, seeds, and protocols required to deterministically reproduce the **TerraSpectra 3D-CNN + Vision Transformer Hybrid Model** and its **Contract C2** production artifact (`models/model.pt`).

---

## 1. System Specifications & Dependencies

- **Python Version**: 3.10+ (tested on Python 3.11.9)
- **PyTorch**: >= 2.1.0 (tested on PyTorch 2.14.0 CPU & CUDA)
- **PyTorch Lightning**: >= 2.2.0
- **Package Manager**: `uv` (recommended) or standard `pip`

```bash
cd model
uv sync --all-extras
```

---

## 2. One-Command Quick Smoke Test

To verify complete end-to-end execution (data generation -> model training -> evaluation -> TorchScript export -> numerical parity verification -> Contract C2 compliance):

```bash
uv run python reproduce.py --quick
```

**Expected Runtime**: ~3 seconds  
**Output Artifact**: `reports/reproducibility/reproducibility_report.json`

---

## 3. Full Reproducible Training Pipeline

To run the full reproducible training process with fixed seeds across NumPy, Python, and PyTorch:

```bash
uv run python reproduce.py --epochs 10 --seed 42 --output-dir reports/reproducibility
```

### Determinism Controls:
- **Seed**: `42` (controls dataset simulation, weights initialization, augmentation, and data loader shuffling).
- **Environment Flags**:
  - `PYTHONHASHSEED=42`
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.set_float32_matmul_precision("high")`

### Generated Deliverables:
1. **Checkpoint**: `reports/reproducibility/training/checkpoints/best.ckpt`
2. **Production Artifact**: `reports/reproducibility/reproduced_model.pt` (TorchScript, Contract C2)
3. **Audit Log**: `reports/reproducibility/reproducibility_report.json`

---

## 4. Cross-Track Integration Verification (Day 13)

To confirm drop-in compatibility between the P2 ML model and Nikita's P3 FastAPI inference engine (`api/core/engine.py`):

```bash
uv run python verify_integration.py
```

### Checks Performed:
- [x] Contract C2 TorchScript I/O shape verification: `[N, 200, 64, 64]` -> `[N, 4, 64, 64]` probs, `[N, 1, 64, 64]` onset.
- [x] API `InferenceEngine` initialization and weight loading.
- [x] Chunked inference and Hann-window blending across 128x128 simulated scene.
- [x] Integrated Gradients spectral attribution -> Contract C4 `dominant_indicator` generation.

---

## 5. Interactive Demo Rehearsal (Day 15 Presentation)

To run the terminal-based interactive demonstration for presentation and project review:

```bash
uv run python demo.py --seed 101 --steps 15
```

### Demo Output Highlights:
- **Model Load Latency**: Benchmarked in milliseconds.
- **Hyperspectral Scene Simulation**: 200 bands (400–2500 nm) with biophysical stress progression.
- **Forward Inference**: Risk class breakdown with ASCII visual bars.
- **Forecast Window**: Days-to-onset estimate (e.g. ~18–21 days ahead of visible symptoms).
- **Explainability**: Dominant indicator (`pri_decline`, `red_edge_shift`, `water_stress`) and top diagnostic wavelengths.
- **Prescription**: Automated agronomic recommendations for field application.

---

## 6. Unit Test Suite

Run the full automated test suite to ensure 100% component stability:

```bash
uv run pytest -v
```
All 52 tests covering architecture, losses, synthetic stress generator, proxy labels, benchmark loaders, export parity, and reproducibility should pass cleanly.
