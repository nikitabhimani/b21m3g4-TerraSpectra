"""64x64 window datasets (real scenes and on-the-fly synthesis) and the LightningDataModule."""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from terraspectra_model.config import AugmentConfig, DataConfig, SynthConfig
from terraspectra_model.structures import LabelledScene

log = logging.getLogger(__name__)

Batch = dict[str, torch.Tensor]


def augment_window(
    x: np.ndarray,
    y: np.ndarray,
    onset: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    cfg: AugmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geometric (flip / rot90) and spectral (scale, jitter, band dropout) augmentation."""
    if not cfg.enabled:
        return x, y, onset, mask
    if rng.random() < cfg.hflip_p:
        x, y, onset, mask = x[..., ::-1], y[..., ::-1], onset[..., ::-1], mask[..., ::-1]
    if rng.random() < cfg.vflip_p:
        x, y, onset, mask = x[..., ::-1, :], y[::-1], onset[::-1], mask[::-1]
    if cfg.rot90:
        k = int(rng.integers(0, 4))
        if k:
            x = np.rot90(x, k, axes=(1, 2))
            y, onset, mask = (np.rot90(a, k) for a in (y, onset, mask))
    x = np.array(x, dtype=np.float32)  # contiguous copy
    valid = mask[None]
    lo, hi = cfg.spectral_scale
    if hi > lo or lo != 1.0:
        x *= np.float32(rng.uniform(lo, hi))
    if cfg.spectral_jitter_std > 0:
        # Smooth per-band gain wobble (calibration drift) plus pixel noise.
        wobble = np.asarray(
            rng.normal(0.0, cfg.spectral_jitter_std * 4, size=x.shape[0]), dtype=np.float64
        )
        gain = 1.0 + np.convolve(wobble, np.ones(9) / 9.0, mode="same")
        x = x * gain[:, None, None].astype(np.float32)
        x += rng.normal(0, cfg.spectral_jitter_std, x.shape).astype(np.float32)
    if cfg.band_dropout_max_bands > 0 and rng.random() < cfg.band_dropout_p:
        n = int(rng.integers(1, cfg.band_dropout_max_bands + 1))
        x[rng.choice(x.shape[0], n, replace=False)] = 0.0
    x = np.where(valid, np.clip(x, 0.0, 1.0), 0.0).astype(np.float32)
    return x, np.ascontiguousarray(y), np.ascontiguousarray(onset), np.ascontiguousarray(mask)


def to_batch_item(
    x: np.ndarray, y: np.ndarray, onset: np.ndarray, mask: np.ndarray
) -> dict[str, torch.Tensor]:
    """Convert one window to the tensor dict consumed by the LightningModule."""
    return {
        "x": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        "y": torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)),
        "onset": torch.from_numpy(np.ascontiguousarray(onset, dtype=np.float32)),
        "mask": torch.from_numpy(np.ascontiguousarray(mask, dtype=bool)),
    }


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Random 64x64 windows from labelled scenes (deterministic per ``(seed, index)``)."""

    def __init__(
        self,
        scenes: Sequence[LabelledScene],
        samples: int,
        window: int = 64,
        augment: AugmentConfig | None = None,
        min_valid_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        if not scenes:
            raise ValueError("WindowDataset needs at least one scene")
        self.scenes = [self._pad(s, window) for s in scenes]
        self.samples = samples
        self.window = window
        self.augment = augment or AugmentConfig(enabled=False)
        self.min_valid_fraction = min_valid_fraction
        self.seed = seed
        areas = np.array([s.shape[0] * s.shape[1] for s in self.scenes], dtype=np.float64)
        self._p = areas / areas.sum()

    @staticmethod
    def _pad(s: LabelledScene, window: int) -> LabelledScene:
        H, W = s.shape
        ph, pw = max(0, window - H), max(0, window - W)
        if not (ph or pw):
            return s
        pad2 = ((0, ph), (0, pw))
        return LabelledScene(
            cube=np.pad(s.cube, ((0, 0), *pad2)),
            class_map=np.pad(s.class_map, pad2, constant_values=-1),
            onset_map=np.pad(s.onset_map, pad2, constant_values=30.0),
            valid_mask=np.pad(s.valid_mask, pad2, constant_values=False),
        )

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, index))
        w = self.window
        for _attempt in range(10):
            s = self.scenes[int(rng.choice(len(self.scenes), p=self._p))]
            H, W = s.shape
            i, j = int(rng.integers(0, H - w + 1)), int(rng.integers(0, W - w + 1))
            sl = (slice(i, i + w), slice(j, j + w))
            labelled = (s.class_map[sl] >= 0) & s.valid_mask[sl]
            if labelled.mean() >= self.min_valid_fraction:
                break
        x = s.cube[(slice(None), *sl)]
        return to_batch_item(
            *augment_window(x, s.class_map[sl], s.onset_map[sl], s.valid_mask[sl], rng,
                            self.augment)
        )  # fmt: skip


class SyntheticWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Windows generated on the fly by :func:`synth.stress.make_field_patch`."""

    def __init__(
        self,
        samples: int,
        synth: SynthConfig | None = None,
        augment: AugmentConfig | None = None,
        seed: int = 0,
        window: int = 64,
    ) -> None:
        self.samples = samples
        self.synth = synth or SynthConfig()
        self.augment = augment or AugmentConfig(enabled=False)
        self.seed = seed
        self.window = window

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        from terraspectra_model.synth.stress import make_field_patch

        rng = np.random.default_rng((self.seed, index))
        p = make_field_patch(rng, self.synth, self.window)
        return to_batch_item(
            *augment_window(p.cube, p.class_map, p.onset_map, p.valid_mask, rng, self.augment)
        )


class NpzWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-generated windows saved by ``terraspectra-model synth`` (keys x, y, onset, mask)."""

    def __init__(self, paths: Sequence[str | Path], augment: AugmentConfig | None = None) -> None:
        arrays = [np.load(p) for p in paths]
        self.x = np.concatenate([a["x"] for a in arrays])
        self.y = np.concatenate([a["y"] for a in arrays])
        self.onset = np.concatenate([a["onset"] for a in arrays])
        self.mask = np.concatenate([a["mask"] for a in arrays])
        self.augment = augment or AugmentConfig(enabled=False)

    def with_augment(self, augment: AugmentConfig | None) -> NpzWindowDataset:
        """Shallow copy sharing the arrays but with different augmentation."""
        other = copy.copy(self)
        other.augment = augment or AugmentConfig(enabled=False)
        return other

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(index + np.random.randint(1 << 30))
        return to_batch_item(
            *augment_window(self.x[index], self.y[index], self.onset[index], self.mask[index],
                            rng, self.augment)
        )  # fmt: skip


def build_benchmark_scenes(cfg: DataConfig, seed: int = 0) -> list[LabelledScene]:
    """Load available benchmarks and blend simulated stress into their vegetation pixels."""
    from terraspectra_model.data.benchmarks import available_benchmarks, load_benchmark
    from terraspectra_model.synth.stress import blend_stress_into_scene

    names = [n for n in cfg.benchmarks if n in available_benchmarks(cfg.data_dir)]
    if not names:
        raise FileNotFoundError(
            f"no benchmark .mat files in {cfg.data_dir}; see model/README.md (Datasets)"
        )
    rng = np.random.default_rng(seed)
    scenes = []
    for name in names:
        bench = load_benchmark(name, cfg.data_dir)
        scenes.append(blend_stress_into_scene(bench.cube, bench.vegetation_mask, rng, cfg.synth))
    # TODO(Day 4): add proxy-labelled real scenes (synth.proxy_labels) from P1's COGs here.
    return scenes


class TerraSpectraDataModule(L.LightningDataModule):
    """Train/val loaders for the configured data source."""

    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Dataset[Any] | None = None
        self.val_ds: Dataset[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Instantiate datasets (idempotent)."""
        if self.train_ds is not None:
            return
        c = self.cfg
        if c.source == "synthetic":
            self.train_ds = SyntheticWindowDataset(
                c.train_samples, c.synth, c.augment, seed=c.seed, window=c.window_size
            )
            self.val_ds = SyntheticWindowDataset(
                c.val_samples, c.synth, None, seed=c.seed + 10_000, window=c.window_size
            )
        elif c.source == "benchmarks":
            scenes = build_benchmark_scenes(c, c.seed)
            # TODO(Day 8): spatially disjoint train/val split per scene instead of seeds.
            self.train_ds = WindowDataset(
                scenes, c.train_samples, c.window_size, c.augment, c.min_valid_fraction, c.seed
            )
            self.val_ds = WindowDataset(
                scenes, c.val_samples, c.window_size, None, c.min_valid_fraction, c.seed + 10_000
            )
        elif c.source == "npz":
            if not c.npz_paths:
                raise ValueError("data.npz_paths is empty")
            full = NpzWindowDataset(c.npz_paths)
            n_val = max(1, min(c.val_samples, len(full) // 10))
            perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(c.seed))
            val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
            self.train_ds = Subset(full.with_augment(c.augment), train_idx)
            self.val_ds = Subset(full, val_idx)
        log.info("data source=%s train=%s val=%s", c.source, c.train_samples, c.val_samples)

    def _loader(self, ds: Dataset[Any] | None, shuffle: bool) -> DataLoader[Any]:
        if ds is None:
            raise RuntimeError("call setup() first")
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.cfg.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Training loader."""
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader[Any]:
        """Validation loader."""
        return self._loader(self.val_ds, shuffle=False)
