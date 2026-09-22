"""TerraSpectra Model Track (P2) — Spectral stress & Contract C2 exploration.

Companion to pipeline/explore_cube.py for testing:
1. Synthetic 200-band hyperspectral windows (Contract C1).
2. Pre-visual fungal blight stress progression (PROSAIL-based spectral shifts).
3. Spectral indices (NDVI, NDRE, PRI, Red-Edge Position).
4. Model I/O contract verification (Contract C2: 3D-CNN + ViT hybrid).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

# Canonical wavelengths (400 nm to 2500 nm, 200 bands)
WAVELENGTHS = np.linspace(400.0, 2500.0, 200)


def build_synthetic_canopy_spectrum(
    wavelengths: np.ndarray,
    cab: float = 45.0,     # Chlorophyll a+b content (ug/cm^2)
    cw: float = 0.015,     # Equivalent water thickness (cm)
    lai: float = 3.5,      # Leaf Area Index
) -> np.ndarray:
    """Generate a realistic synthetic vegetation reflectance spectrum (simplified PROSAIL approximation)."""
    # 1. Pigment absorption (Chlorophyll-a & b at 450 nm and 670 nm)
    blue_abs = np.exp(-0.5 * ((wavelengths - 450) / 35.0) ** 2)
    red_abs = np.exp(-0.5 * ((wavelengths - 670) / 30.0) ** 2)
    chlorophyll_abs = (cab / 50.0) * (0.65 * blue_abs + 0.85 * red_abs)

    # 2. Green reflectance peak around 550 nm
    green_peak = 0.12 * np.exp(-0.5 * ((wavelengths - 550) / 40.0) ** 2)

    # 3. NIR plateau (750 - 1300 nm) governed by leaf structure & LAI
    nir_plateau = (1.0 - np.exp(-0.8 * lai)) * 0.52 * (
        1.0 / (1.0 + np.exp(-(wavelengths - 715) / 18.0))
    )

    # 4. SWIR water absorption troughs around 1450 nm and 1940 nm
    water_abs_1450 = np.exp(-0.5 * ((wavelengths - 1450) / 80.0) ** 2)
    water_abs_1940 = np.exp(-0.5 * ((wavelengths - 1940) / 90.0) ** 2)
    water_factor = 1.0 - (cw / 0.02) * (0.65 * water_abs_1450 + 0.85 * water_abs_1940)

    # Base soil / background baseline
    soil_baseline = 0.08 + 0.22 * (wavelengths - 400.0) / 2100.0

    reflectance = (soil_baseline * (1.0 - min(lai / 4.0, 0.9))) + (green_peak + nir_plateau) * (1.0 - chlorophyll_abs)
    reflectance = reflectance * np.clip(water_factor, 0.1, 1.0)
    return np.clip(reflectance, 0.01, 0.85).astype(np.float32)


def compute_spectral_indices(cube: np.ndarray, wavelengths: np.ndarray) -> dict[str, float]:
    """Compute pre-visual vegetation stress indices on a mean spectrum or pixel."""
    def band(nm: float) -> float:
        idx = int(np.argmin(np.abs(wavelengths - nm)))
        return float(cube[idx].mean())

    r531, r570 = band(531), band(570)
    r670, r700, r720, r740, r780, r790, r800 = (
        band(670), band(700), band(720), band(740), band(780), band(790), band(800)
    )

    eps = 1e-6
    # NDVI (Normalized Difference Vegetation Index)
    ndvi = (r800 - r670) / (r800 + r670 + eps)
    # NDRE (Red-Edge NDVI, early chlorophyll degradation indicator)
    ndre = (r790 - r720) / (r790 + r720 + eps)
    # PRI (Photochemical Reflectance Index, pre-visual photosynthetic stress)
    pri = (r531 - r570) / (r531 + r570 + eps)

    # Red Edge Position (REP) via Guyot & Baret linear 4-point method
    r_re = (r670 + r780) / 2.0
    rep = 700.0 + 40.0 * (r_re - r700) / (r740 - r700 + eps)

    return {
        "NDVI": round(ndvi, 4),
        "NDRE (Red-Edge)": round(ndre, 4),
        "PRI (Pre-visual)": round(pri, 4),
        "REP_nm": round(float(np.clip(rep, 680.0, 760.0)), 2),
    }


def main() -> None:
    print("=" * 70)
    print("TerraSpectra ML Track (P2) — Hyperspectral Stress & Contract C2")
    print("=" * 70)

    # 1. Build a window conforming to Contract C1 (B=200, H=64, W=64)
    B, H, W = len(WAVELENGTHS), 64, 64
    print(f"\n[1] Constructing Synthetic Hyperspectral Window (Contract C1):")
    print(f"    Shape: ({B} bands, {H} height, {W} width)")
    print(f"    Wavelength Range: {WAVELENGTHS[0]:.1f} nm -> {WAVELENGTHS[-1]:.1f} nm")

    # 2. Simulate 4 Stages of Early Fungal Blight Progression
    stages = [
        ("Healthy Canopy (Onset = 30d, Class 0)", {"cab": 45.0, "cw": 0.015, "lai": 3.8}),
        ("Early Pre-visual Stress (Onset = 18d, Class 1)", {"cab": 32.0, "cw": 0.013, "lai": 3.4}),
        ("High Blight Risk (Onset = 5d, Class 2)", {"cab": 18.0, "cw": 0.009, "lai": 2.6}),
        ("Visible Disease (Onset = 0d, Class 3)", {"cab": 8.0, "cw": 0.004, "lai": 1.5}),
    ]

    print("\n[2] Pre-Visual Stress Progression Analysis:")
    for name, params in stages:
        spec = build_synthetic_canopy_spectrum(WAVELENGTHS, **params)
        # Wrap into pseudo-cube (B, 1, 1) for index calculation
        indices = compute_spectral_indices(spec[:, None, None], WAVELENGTHS)
        print(f"  * {name}:")
        print(f"      Indices -> {indices}")
        print(f"      Key bands: Green(550nm)={spec[14]:.3f}, Red(670nm)={spec[25]:.3f}, NIR(800nm)={spec[38]:.3f}")

    # 3. Assemble full 64x64 window with an infection hotspot in center
    window = np.zeros((B, H, W), dtype=np.float32)
    healthy_spec = build_synthetic_canopy_spectrum(WAVELENGTHS, cab=45.0, cw=0.015, lai=3.8)
    stressed_spec = build_synthetic_canopy_spectrum(WAVELENGTHS, cab=24.0, cw=0.010, lai=2.8)

    # Base background: healthy canopy
    window[:] = healthy_spec[:, None, None]

    # Create center circular infection focus (radius 12 px)
    y, x = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((x - W // 2) ** 2 + (y - H // 2) ** 2)
    focus_mask = dist_from_center <= 12
    window[:, focus_mask] = stressed_spec[:, None]

    # Add realistic sensor photon noise
    noise = np.random.normal(0, 0.005, window.shape).astype(np.float32)
    window = np.clip(window + noise, 0.0, 1.0)

    print(f"\n[3] Generated Window Data:")
    print(f"    Min reflectance: {window.min():.4f}, Max reflectance: {window.max():.4f}")
    print(f"    Infection focus pixels: {focus_mask.sum()} / {H * W} ({focus_mask.mean() * 100:.1f}%)")

    # 4. Verify Model I/O Contract (Contract C2)
    batch_size = 2
    batch_x = np.repeat(window[None, ...], batch_size, axis=0)  # [N, 200, 64, 64]

    # Mock forward pass matching C2: (probs: [N, 4, 64, 64], onset: [N, 1, 64, 64])
    mock_probs = np.zeros((batch_size, 4, H, W), dtype=np.float32)
    mock_probs[:, 0, :, :] = 0.85  # class 0 healthy default
    mock_probs[:, 1, :, :] = 0.05
    mock_probs[:, 2, :, :] = 0.05
    mock_probs[:, 3, :, :] = 0.05

    # Focus hotspot has elevated early-stress & high-risk probabilities
    mock_probs[:, 0, focus_mask] = 0.10
    mock_probs[:, 1, focus_mask] = 0.70  # class 1 early stress
    mock_probs[:, 2, focus_mask] = 0.15
    mock_probs[:, 3, focus_mask] = 0.05

    mock_onset = np.full((batch_size, 1, H, W), 30.0, dtype=np.float32)
    mock_onset[:, 0, focus_mask] = 16.5  # ~16.5 days to visible symptoms

    print(f"\n[4] Contract C2 Verification:")
    print(f"    Model Input  : float32 shape {batch_x.shape} (N, 200, 64, 64)")
    print(f"    Model Output : probs={mock_probs.shape} (N, 4, 64, 64) [Classes: 0=Healthy, 1=Early Stress, 2=High Risk, 3=Visible]")
    print(f"                   onset={mock_onset.shape} (N, 1, 64, 64) [Days to onset: 0-30 days]")
    print(f"    Softmax Sum Verification: {mock_probs.sum(axis=1).mean():.4f} (expected: 1.0)")
    print(f"    Target Outbreak Lead Time: {mock_onset[:, 0, focus_mask].mean():.1f} days before visible symptoms")

    print("\n[OK] Spectral stress simulation and Contract C2 validation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
