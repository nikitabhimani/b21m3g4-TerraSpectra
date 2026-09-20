"""3D spectral-spatial convolutional encoder."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def spectral_out_len(length: int, kernels: Sequence[int], strides: Sequence[int]) -> int:
    """Band-axis length after the Conv3d stack (``padding = k // 2``)."""
    for k, s in zip(kernels, strides, strict=True):
        length = (length + 2 * (k // 2) - k) // s + 1
    return length


class Conv3dBlock(nn.Module):
    """Conv3d (k_spec x 3 x 3, spectral stride) -> BatchNorm3d -> GELU."""

    def __init__(self, cin: int, cout: int, k_spec: int, s_spec: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            cin, cout, kernel_size=(k_spec, 3, 3), stride=(s_spec, 1, 1),
            padding=(k_spec // 2, 1, 1), bias=False,
        )  # fmt: skip
        self.bn = nn.BatchNorm3d(cout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, C, B, H, W] -> [N, C', B', H, W]``."""
        return self.act(self.bn(self.conv(x)))


class SpectralSpatialEncoder3D(nn.Module):
    """``[N, B, H, W]`` reflectance -> ``[N, out_channels, H, W]`` features.

    The cube is treated as a single-channel volume; stacked Conv3d blocks with spectral striding
    shrink the band axis, which is then folded into channels and mixed by a 1x1 Conv2d.
    """

    def __init__(
        self,
        in_bands: int,
        channels: Sequence[int] = (16, 32, 64),
        spectral_kernels: Sequence[int] = (7, 5, 3),
        spectral_strides: Sequence[int] = (2, 2, 2),
        out_channels: int = 96,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        cin = 1
        for cout, k, s in zip(channels, spectral_kernels, spectral_strides, strict=True):
            blocks.append(Conv3dBlock(cin, cout, k, s))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.out_bands = spectral_out_len(in_bands, spectral_kernels, spectral_strides)
        self.fold = nn.Sequential(
            nn.Conv2d(cin * self.out_bands, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode; spatial size is preserved."""
        f = self.blocks(x.unsqueeze(1))  # [N, C, B', H, W]
        f = f.flatten(1, 2)  # fold spectral axis into channels: [N, C*B', H, W]
        return self.fold(f)
