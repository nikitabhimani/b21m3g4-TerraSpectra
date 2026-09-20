"""TerraSpectraNet: 3D-CNN encoder -> ViT -> upsampling decoder -> class / onset heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from terraspectra_model.arch.cnn3d import SpectralSpatialEncoder3D
from terraspectra_model.arch.vit import ViTEncoder, num_upsamples
from terraspectra_model.config import ModelConfig


def _conv_bn(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.GELU()
    )


class Decoder(nn.Module):
    """Progressive x2 upsampling of ViT features, fused with full-resolution CNN skip features."""

    def __init__(self, in_dim: int, skip_dim: int, ch: int, n_up: int) -> None:
        super().__init__()
        self.inp = _conv_bn(in_dim, ch, 1)
        self.ups = nn.ModuleList(_conv_bn(ch, ch) for _ in range(n_up))
        self.skip = _conv_bn(skip_dim, ch, 1)
        self.fuse = nn.Sequential(_conv_bn(2 * ch, ch), _conv_bn(ch, ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """``x [N, D, h, w]``, ``skip [N, S, H, W]`` -> ``[N, ch, H, W]``."""
        x = self.inp(x)
        for up in self.ups:
            x = up(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat([x, self.skip(skip)], dim=1))


class TerraSpectraNet(nn.Module):
    """Hybrid 3D-CNN + ViT segmentation model.

    ``forward(x) -> (probs[N,4,64,64], onset[N,1,64,64])`` is the contract-C2 entry point used for
    export; ``forward_logits`` returns raw class logits and onset for training.
    ``variant`` supports the Day-9 ablations: ``hybrid``, ``cnn_only`` (no transformer blocks /
    spectral tokens) and ``vit_only`` (a 1x1 spectral projection replaces the 3D-CNN).
    """

    temperature: torch.Tensor

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.in_bands = cfg.in_bands
        self.max_onset = float(cfg.max_onset_days)
        if cfg.variant == "vit_only":
            self.encoder: nn.Module = _conv_bn(cfg.in_bands, cfg.cnn_out_channels, 1)
        else:
            self.encoder = SpectralSpatialEncoder3D(
                cfg.in_bands,
                cfg.cnn_channels,
                cfg.cnn_spectral_kernels,
                cfg.cnn_spectral_strides,
                cfg.cnn_out_channels,
            )
        cnn_only = cfg.variant == "cnn_only"
        self.vit = ViTEncoder(
            in_channels=cfg.cnn_out_channels,
            grid_size=64,
            patch=cfg.patch_size,
            dim=cfg.embed_dim,
            depth=0 if cnn_only else cfg.depth,
            heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            attn_dropout=cfg.attn_dropout,
            pos_embed=cfg.pos_embed,
            spectral_bands=cfg.in_bands if cfg.spectral_tokens and not cnn_only else None,
            spectral_groups=cfg.spectral_groups,
        )
        self.decoder = Decoder(
            cfg.embed_dim, cfg.cnn_out_channels, cfg.decoder_channels, num_upsamples(cfg.patch_size)
        )
        self.class_head = nn.Conv2d(cfg.decoder_channels, cfg.n_classes, kernel_size=1)
        self.onset_head = nn.Conv2d(cfg.decoder_channels, 1, kernel_size=1)
        # Temperature for calibrated softmax (set by evaluate.fit_temperature); exported with model.
        self.register_buffer("temperature", torch.ones(()))

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4 or x.shape[1] != self.in_bands:
            raise ValueError(f"expected [N, {self.in_bands}, H, W], got {tuple(x.shape)}")
        if x.shape[-2] % self.cfg.patch_size or x.shape[-1] != x.shape[-2]:
            raise ValueError(f"expected square windows divisible by patch, got {tuple(x.shape)}")

    def forward_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(class_logits [N,4,H,W], onset_days [N,1,H,W])``."""
        self._check_input(x)
        feats = self.encoder(x)
        tokens = self.vit(feats, x)
        dec = self.decoder(tokens, feats)
        onset = torch.sigmoid(self.onset_head(dec)) * self.max_onset
        return self.class_head(dec), onset

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Contract C2: ``(softmax probs [N,4,64,64], onset [N,1,64,64] in [0,30])``."""
        logits, onset = self.forward_logits(x)
        return torch.softmax(logits / self.temperature, dim=1), onset

    def set_temperature(self, t: float) -> None:
        """Set the calibration temperature used by :meth:`forward`."""
        self.temperature.fill_(float(t))


def build_model(cfg: ModelConfig | None = None) -> TerraSpectraNet:
    """Factory used by the training / export code."""
    return TerraSpectraNet(cfg)
