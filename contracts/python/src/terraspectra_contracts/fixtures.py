"""Contract C5 - shared fixtures so every module can work without the others.

- ``make_synthetic_cube``: a C1-valid cube (numpy) with a planted stressed patch.
- ``write_synthetic_cube``: the same cube as a Cloud-Optimized GeoTIFF (needs ``[raster]``).
- ``StubModel`` / ``export_stub_model``: a C2-valid TorchScript model (needs ``[torch]``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from terraspectra_contracts.constants import (
    N_BANDS,
    N_CLASSES,
    NODATA,
    WAVELENGTH_TAG,
    WAVELENGTHS_NM,
)


def _vegetation_spectrum(wl: np.ndarray, chlorophyll: float) -> np.ndarray:
    """Crude healthy-vegetation reflectance; lower chlorophyll -> red-edge shifts blue-ward."""
    red_edge = 700.0 + 25.0 * chlorophyll
    visible = 0.05 + 0.04 * np.exp(-((wl - 550.0) ** 2) / (2 * 30.0**2)) * (1.5 - chlorophyll)
    nir = 0.45 * chlorophyll + 0.15
    edge = 1.0 / (1.0 + np.exp(-(wl - red_edge) / 12.0))
    swir_decay = np.where(wl > 1300, np.exp(-(wl - 1300) / 900.0), 1.0)
    return (visible + (nir - visible) * edge) * swir_decay


def make_synthetic_cube(
    height: int = 256,
    width: int = 256,
    seed: int = 0,
    stressed_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cube[B,H,W] float32, stress_mask[H,W] bool)``."""
    rng = np.random.default_rng(seed)
    wl = np.asarray(WAVELENGTHS_NM, dtype=np.float32)
    healthy = _vegetation_spectrum(wl, 1.0)
    stressed = _vegetation_spectrum(wl, 0.6)

    mask = np.zeros((height, width), dtype=bool)
    side = max(1, int(np.sqrt(stressed_fraction) * min(height, width)))
    y0, x0 = height // 2, width // 2
    mask[y0 : y0 + side, x0 : x0 + side] = True

    cube = np.where(mask[None], stressed[:, None, None], healthy[:, None, None])
    cube = cube * rng.normal(1.0, 0.03, size=(1, height, width))
    cube = cube + rng.normal(0.0, 0.005, size=(N_BANDS, height, width))
    cube = np.clip(cube, 0.0, 1.0).astype(np.float32)
    cube[:, :2, :2] = NODATA  # a nodata corner, to exercise masking
    return cube, mask


def write_synthetic_cube(
    path: str | Path,
    height: int = 256,
    width: int = 256,
    seed: int = 0,
    crs: str = "EPSG:32643",
    origin: tuple[float, float] = (500000.0, 3420000.0),
    pixel_size: float = 30.0,
) -> Path:
    """Write a C1-valid Cloud-Optimized GeoTIFF and return its path."""
    import rasterio
    from rasterio.transform import from_origin

    cube, _ = make_synthetic_cube(height, width, seed)
    path = Path(path)
    profile = {
        "driver": "COG",
        "dtype": "float32",
        "count": N_BANDS,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": from_origin(origin[0], origin[1], pixel_size, pixel_size),
        "nodata": NODATA,
        "compress": "ZSTD",
        "blocksize": 256,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cube)
        for i, w in enumerate(WAVELENGTHS_NM, start=1):
            dst.update_tags(i, **{WAVELENGTH_TAG: f"{w:.2f}"})
        dst.update_tags(contract_version="1.0.0", source="synthetic")
    return path


def export_stub_model(path: str | Path) -> Path:
    """Export a random-weight TorchScript model honouring contract C2."""
    import torch

    class StubModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Conv2d(N_BANDS, N_CLASSES + 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.proj(x)
            probs = torch.softmax(out[:, :-1], dim=1)
            onset = torch.sigmoid(out[:, -1:]) * 30.0
            return probs, onset

    torch.manual_seed(0)
    example = torch.zeros(1, N_BANDS, 64, 64)
    with torch.no_grad():
        scripted = torch.jit.trace(StubModel().eval(), example)
    path = Path(path)
    scripted.save(str(path))
    return path
