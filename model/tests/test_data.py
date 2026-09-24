from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat
from terraspectra_contracts import N_BANDS, WAVELENGTHS_NM

from terraspectra_model.config import AugmentConfig, Config, SynthConfig
from terraspectra_model.data import benchmarks
from terraspectra_model.data.spectral import resample_to_canonical
from terraspectra_model.data.windows import (
    SyntheticWindowDataset,
    TerraSpectraDataModule,
    WindowDataset,
    augment_window,
)
from terraspectra_model.structures import LabelledScene


def test_resample_identity_and_linear() -> None:
    wl = np.asarray(WAVELENGTHS_NM)
    cube = np.random.default_rng(0).random((N_BANDS, 3, 3)).astype(np.float32)
    out, covered = resample_to_canonical(cube, wl)
    np.testing.assert_allclose(out, cube, atol=1e-6)
    assert covered.all()

    src = np.linspace(400, 2500, 50)
    lin = (src / 2500.0)[:, None] * np.ones((1, 4))
    out, _ = resample_to_canonical(lin, src)
    np.testing.assert_allclose(out[:, 0], wl / 2500.0, atol=1e-5)
    g, _ = resample_to_canonical(lin, src, method="gaussian")
    np.testing.assert_allclose(g[5:-5, 0], wl[5:-5] / 2500.0, atol=5e-3)


def test_resample_partial_coverage_edge_fill() -> None:
    src = np.linspace(430, 860, 103)
    out, covered = resample_to_canonical(np.linspace(0, 1, 103)[:, None], src)
    assert not covered.all() and covered.sum() > 30
    first, last = np.argmax(covered), len(covered) - 1 - np.argmax(covered[::-1])
    assert (out[:first, 0] == out[first, 0]).all() and out[first, 0] < 0.01
    assert (out[last:, 0] == out[last, 0]).all() and out[last, 0] > 0.99


def test_load_benchmark_from_fake_mat(tmp_path: Path) -> None:
    spec = benchmarks.BENCHMARKS["pavia_u"]
    rng = np.random.default_rng(0)
    savemat(tmp_path / spec.cube_file, {spec.cube_key: rng.integers(0, 8000, (10, 12, 103))})
    savemat(tmp_path / spec.gt_file, {spec.gt_key: rng.integers(0, 10, (10, 12))})
    assert benchmarks.available_benchmarks(tmp_path) == ["pavia_u"]
    scene = benchmarks.load_benchmark("pavia_u", tmp_path)
    assert scene.cube.shape == (N_BANDS, 10, 12)
    assert scene.cube.min() >= 0 and scene.cube.max() <= 1
    assert scene.vegetation_mask.sum() == np.isin(scene.labels, [2, 4]).sum()


def test_benchmark_wavelength_counts() -> None:
    assert benchmarks.BENCHMARKS["indian_pines"].wavelengths.size == 200
    assert benchmarks.BENCHMARKS["salinas"].wavelengths.size == 204
    with pytest.raises(FileNotFoundError):
        benchmarks.load_benchmark("salinas", "/nonexistent")


def test_download_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def fake_urlopen(req, timeout=60):
        return FakeResponse(b"MATLAB 5.0 MAT-file test dummy payload")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    cube_path, gt_path = benchmarks.download_benchmark("pavia_u", tmp_path)
    assert cube_path.exists() and cube_path.name == "PaviaU.mat"
    assert gt_path.exists() and gt_path.name == "PaviaU_gt.mat"
    assert cube_path.read_bytes().startswith(b"MATLAB")


def test_augment_keeps_alignment() -> None:
    rng = np.random.default_rng(0)
    x = np.zeros((N_BANDS, 64, 64), dtype=np.float32)
    x[:, :10, :5] = 0.5
    y = np.zeros((64, 64), dtype=np.int64)
    y[:10, :5] = 2
    mask = np.ones((64, 64), dtype=bool)
    cfg = AugmentConfig(spectral_jitter_std=0.0, spectral_scale=(1.0, 1.0), band_dropout_p=0.0)
    for _ in range(5):
        xa, ya, _, _ = augment_window(x, y, y.astype(np.float32), mask, rng, cfg)
        assert ((xa[0] > 0) == (ya == 2)).all()


def test_window_dataset_samples() -> None:
    rng = np.random.default_rng(0)
    scene = LabelledScene(
        cube=rng.random((N_BANDS, 50, 80)).astype(np.float32),
        class_map=rng.integers(0, 4, (50, 80)),
        onset_map=np.full((50, 80), 30.0, dtype=np.float32),
        valid_mask=np.ones((50, 80), dtype=bool),
    )
    ds = WindowDataset([scene], samples=3, augment=AugmentConfig())
    item = ds[1]
    assert item["x"].shape == (N_BANDS, 64, 64)  # padded in height
    assert item["y"].shape == (64, 64) and item["y"].dtype == torch.int64
    assert torch.equal(ds[1]["x"], item["x"])  # deterministic per index


def test_synthetic_dataset_and_datamodule(tiny_cfg: Config) -> None:
    ds = SyntheticWindowDataset(2, SynthConfig(), AugmentConfig())
    assert ds[0]["x"].shape == (N_BANDS, 64, 64)
    dm = TerraSpectraDataModule(tiny_cfg.data)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch["x"].shape == (2, N_BANDS, 64, 64)
    assert batch["mask"].dtype == torch.bool
