"""Pydantic configuration models, loaded from YAML (see ``model/configs/``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS, N_CLASSES, WINDOW_SIZE


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AugmentConfig(_Strict):
    """Window augmentation (applied to training windows only)."""

    enabled: bool = True
    hflip_p: float = Field(0.5, ge=0, le=1)
    vflip_p: float = Field(0.5, ge=0, le=1)
    rot90: bool = True
    spectral_scale: tuple[float, float] = (0.9, 1.1)
    spectral_jitter_std: float = Field(0.005, ge=0)
    band_dropout_p: float = Field(0.2, ge=0, le=1)
    band_dropout_max_bands: int = Field(10, ge=0)


class SynthConfig(_Strict):
    """Simulated stress progression (see :mod:`terraspectra_model.synth.stress`)."""

    backend: Literal["auto", "builtin", "prosail"] = "builtin"
    patch_infected_p: float = Field(0.7, ge=0, le=1)
    max_blobs: int = Field(3, ge=0)
    blob_radius: tuple[float, float] = (4.0, 20.0)
    max_infected_days: float = Field(25.0, gt=0, le=MAX_ONSET_DAYS)
    early_stress_min_days: float = 7.0
    days_per_pixel: tuple[float, float] = (0.5, 2.0)
    soil_fraction_p: float = Field(0.2, ge=0, le=1)
    nodata_p: float = Field(0.1, ge=0, le=1)
    noise_std: float = Field(0.004, ge=0)
    healthy_cab: tuple[float, float] = (35.0, 60.0)
    healthy_cw: tuple[float, float] = (0.010, 0.020)
    healthy_lai: tuple[float, float] = (2.0, 5.0)


class ProxyLabelConfig(_Strict):
    """Thresholds for proxy labels from pre-visual stress indices (deficits vs field baseline)."""

    ndvi_vegetation_min: float = 0.4
    baseline_percentile: float = Field(50.0, gt=0, lt=100)
    smooth_sigma: float = Field(1.0, ge=0, description="Gaussian smoothing of index maps (px)")
    rep_shift_nm: tuple[float, float] = (3.0, 8.0)
    ndre_drop: tuple[float, float] = (0.03, 0.08)
    pri_drop: tuple[float, float] = (0.02, 0.05)
    cci_drop: tuple[float, float] = (0.03, 0.08)
    visible_ndvi_drop: float = 0.15
    # PRI and CCI share the noisy 531 nm band, so they get half a vote each by default.
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "rep_shift_nm": 1.0, "ndre_drop": 1.0, "pri_drop": 0.5, "cci_drop": 0.5,
        }
    )  # fmt: skip
    min_score: float = Field(1.5, gt=0)


class DataConfig(_Strict):
    """Where training windows come from and how they are batched."""

    source: Literal["synthetic", "benchmarks", "npz"] = "synthetic"
    data_dir: Path = Path("data/raw")
    benchmarks: list[Literal["indian_pines", "salinas", "pavia_u"]] = Field(
        default_factory=lambda: ["indian_pines", "salinas", "pavia_u"]  # type: ignore[arg-type]
    )
    npz_paths: list[Path] = Field(default_factory=list)
    window_size: int = WINDOW_SIZE
    train_samples: int = Field(4000, ge=1)
    val_samples: int = Field(400, ge=1)
    min_valid_fraction: float = Field(0.5, ge=0, le=1)
    batch_size: int = Field(16, ge=1)
    num_workers: int = Field(4, ge=0)
    seed: int = 0
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    synth: SynthConfig = Field(default_factory=SynthConfig)
    proxy: ProxyLabelConfig = Field(default_factory=ProxyLabelConfig)

    @field_validator("window_size")
    @classmethod
    def _window_matches_contract(cls, v: int) -> int:
        if v != WINDOW_SIZE:
            raise ValueError(f"window_size must be {WINDOW_SIZE} (contract C2)")
        return v


class ModelConfig(_Strict):
    """Architecture hyper-parameters for :class:`TerraSpectraNet`."""

    variant: Literal["hybrid", "cnn_only", "vit_only"] = "hybrid"
    in_bands: int = N_BANDS
    n_classes: int = N_CLASSES
    max_onset_days: float = MAX_ONSET_DAYS
    # 3D-CNN encoder
    cnn_channels: list[int] = Field(default_factory=lambda: [16, 32, 64])
    cnn_spectral_kernels: list[int] = Field(default_factory=lambda: [7, 5, 3])
    cnn_spectral_strides: list[int] = Field(default_factory=lambda: [2, 2, 2])
    cnn_out_channels: int = 96
    # ViT
    patch_size: int = 4
    embed_dim: int = 256
    depth: int = Field(6, ge=0)
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = Field(0.1, ge=0, lt=1)
    attn_dropout: float = Field(0.0, ge=0, lt=1)
    pos_embed: Literal["learned", "sincos"] = "learned"
    spectral_tokens: bool = True
    spectral_groups: int = 20
    # Decoder
    decoder_channels: int = 64

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.in_bands != N_BANDS or self.n_classes != N_CLASSES:
            raise ValueError("in_bands/n_classes are fixed by contract C2")
        if not (
            len(self.cnn_channels)
            == len(self.cnn_spectral_kernels)
            == len(self.cnn_spectral_strides)
            > 0
        ):
            raise ValueError("cnn_channels, cnn_spectral_kernels, cnn_spectral_strides must align")
        if WINDOW_SIZE % self.patch_size or self.patch_size & (self.patch_size - 1):
            raise ValueError("patch_size must be a power of two dividing the window size")
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if self.in_bands % self.spectral_groups:
            raise ValueError("in_bands must be divisible by spectral_groups")
        if self.pos_embed == "sincos" and self.embed_dim % 4:
            raise ValueError("sincos position embedding needs embed_dim divisible by 4")
        return self


class TrainConfig(_Strict):
    """Optimisation, loss and trainer settings."""

    max_epochs: int = Field(50, ge=1)
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_fraction: float = Field(0.05, ge=0, lt=1)
    min_lr_ratio: float = Field(0.01, ge=0, le=1)
    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: int | str = "auto"
    gradient_clip_val: float | None = 1.0
    accumulate_grad_batches: int = 1
    focal_gamma: float = Field(2.0, ge=0)
    class_weights: list[float] | None = None
    ignore_index: int = -1
    onset_weight: float = 0.1
    huber_delta: float = 2.0
    monitor: str = "val/macro_f1"
    early_stopping_patience: int = 10
    output_dir: Path = Path("runs/default")
    seed: int = 42
    log_every_n_steps: int = 20
    fast_dev_run: bool | int = False
    limit_train_batches: float | int | None = None
    limit_val_batches: float | int | None = None
    mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    experiment_name: str = "terraspectra-model"

    @field_validator("class_weights")
    @classmethod
    def _weights_len(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and len(v) != N_CLASSES:
            raise ValueError(f"class_weights needs {N_CLASSES} values")
        return v


class ExportConfig(_Strict):
    """TorchScript / ONNX export settings (C2 artifact)."""

    output_path: Path = Path("../models/model.pt")
    fp16: bool = False
    onnx: bool = False
    onnx_opset: int = 18
    parity_atol: float = 1e-4
    temperature: float = Field(1.0, gt=0)


class Config(_Strict):
    """Top-level config."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)


def _deep_update(base: dict[str, Any], upd: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        out[k] = (
            _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Turn ``["train.lr=1e-3", "data.batch_size=4"]`` into a nested dict (YAML-parsed values)."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like section.key=value")
        node = result
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a :class:`Config` from YAML (``None`` -> defaults) with optional dotted overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        data = loaded
    return Config.model_validate(_deep_update(data, parse_overrides(overrides)))
