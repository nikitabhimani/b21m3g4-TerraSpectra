"""Generate synthetic hyperspectral training and validation datasets (.npz).

Simulates pre-visual fungal blight stress on 200-band hyperspectral windows
conforming to Contract C1 (shape: N, 200, 64, 64) and Contract C2 target labels.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure module is in search path
repo_root = Path(__file__).resolve().parent.parent
sys.path.extend([str(repo_root / "model" / "src"), str(repo_root / "contracts" / "python" / "src")])

from terraspectra_model.config import SynthConfig
from terraspectra_model.synth.stress import generate_dataset


def generate_and_save(
    output_path: Path,
    n_samples: int,
    seed: int,
    cfg: SynthConfig,
) -> None:
    print(f"\nGenerating {n_samples} synthetic windows (seed={seed}) -> {output_path} ...")
    arrays = generate_dataset(n_samples, seed=seed, cfg=cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=arrays["x"],
        y=arrays["y"],
        onset=arrays["onset"],
        mask=arrays["mask"],
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    counts = np.bincount(arrays["y"].ravel(), minlength=4)
    total_pixels = arrays["y"].size

    print(f"  [OK] Successfully saved to {output_path} ({size_mb:.2f} MB)")
    print(f"       Cube Tensor Shape : {arrays['x'].shape} (float32, [0.0, 1.0])")
    print(f"       Class Label Shape : {arrays['y'].shape} (int64, [0..3])")
    print(f"       Onset Days Shape  : {arrays['onset'].shape} (float32, [0.0, 30.0])")
    print(f"       Class Breakdown   :")
    print(f"         - Class 0 (Healthy)          : {counts[0]:>8} px ({counts[0] / total_pixels * 100:>5.1f}%)")
    print(f"         - Class 1 (Early Pre-visual) : {counts[1]:>8} px ({counts[1] / total_pixels * 100:>5.1f}%)")
    print(f"         - Class 2 (High Blight Risk) : {counts[2]:>8} px ({counts[2] / total_pixels * 100:>5.1f}%)")
    print(f"         - Class 3 (Visible Disease)  : {counts[3]:>8} px ({counts[3] / total_pixels * 100:>5.1f}%)")
    print(f"       Mean Onset in Infection Foci   : {arrays['onset'][arrays['y'] > 0].mean():.1f} days")


def main() -> None:
    print("=" * 70)
    print("TerraSpectra ML Track (P2) — Synthetic Dataset Generation")
    print("=" * 70)

    cfg = SynthConfig(
        backend="builtin",
        patch_infected_p=0.8,
        max_blobs=3,
        blob_radius=(4.0, 18.0),
        early_stress_min_days=7.0,
        max_infected_days=25.0,
    )

    data_dir = repo_root / "data" / "synth"

    # 1. Training dataset (128 windows = 524,288 pixels)
    train_path = data_dir / "train.npz"
    generate_and_save(train_path, n_samples=128, seed=42, cfg=cfg)

    # 2. Validation dataset (32 windows = 131,072 pixels)
    val_path = data_dir / "val.npz"
    generate_and_save(val_path, n_samples=32, seed=999, cfg=cfg)

    print("\n" + "=" * 70)
    print("Synthetic dataset generation complete!")
    print(f"Saved to: {data_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
