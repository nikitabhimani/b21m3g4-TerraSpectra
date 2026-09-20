"""Vision Transformer over patch tokens, with an optional spectral-group token branch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2-D sine-cosine position embedding ``[1, grid*grid, dim]`` (``dim % 4 == 0``)."""
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    out = []
    for coord in (y, x):
        ang = coord.flatten().float()[:, None] * omega[None, :]
        out += [ang.sin(), ang.cos()]
    return torch.cat(out, dim=1)[None]


class PatchEmbed(nn.Module):
    """Non-overlapping ``patch x patch`` projection of a feature map to tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, C, H, W] -> [N, L, D]``."""
        return self.proj(x).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class Attention(nn.Module):
    """Multi-head self-attention via ``scaled_dot_product_attention`` (trace-friendly)."""

    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, L, D] -> [N, L, D]``."""
        q, k, v = (
            t.unflatten(-1, (self.heads, -1)).transpose(1, 2) for t in self.qkv(x).chunk(3, -1)
        )
        p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return self.proj(out.transpose(1, 2).flatten(2))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(
        self, dim: int, heads: int, mlp_ratio: float, dropout: float, attn_dropout: float
    ) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, attn_dropout)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual attention + MLP."""
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.mlp(self.norm2(x))


class SpectralGroupTokens(nn.Module):
    """Tokenise band groups of each patch's mean raw spectrum and summarise them per patch.

    ``[N, B, H, W]`` -> ``[N, L, D]``: patch-average the spectrum, split its ``B`` bands into
    ``G`` contiguous groups, embed each group (+ learned spectral position), run one transformer
    block along the group axis and mean-pool to one vector per spatial patch.
    """

    def __init__(
        self, in_bands: int, groups: int, dim: int, patch: int, heads: int, dropout: float
    ) -> None:
        super().__init__()
        if in_bands % groups:
            raise ValueError("in_bands must be divisible by groups")
        self.patch, self.groups = patch, groups
        self.embed = nn.Linear(in_bands // groups, dim)
        self.spec_pos = nn.Parameter(torch.zeros(1, groups, dim))
        self.block = TransformerBlock(dim, heads, 2.0, dropout, 0.0)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.spec_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Per-patch spectral summary tokens."""
        s = F.avg_pool2d(x, self.patch).flatten(2).transpose(1, 2)  # [N, L, B]
        n_tokens = s.shape[1]
        g = s.unflatten(-1, (self.groups, -1)).flatten(0, 1)  # [N*L, G, B/G]
        g = self.block(self.embed(g) + self.spec_pos)
        g = self.norm(g.mean(1))  # [N*L, D]
        return g.unflatten(0, (-1, n_tokens))


class ViTEncoder(nn.Module):
    """Patch embedding + position embedding + ``depth`` transformer blocks."""

    def __init__(
        self,
        in_channels: int,
        grid_size: int,
        patch: int,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        pos_embed: str = "learned",
        spectral_bands: int | None = None,
        spectral_groups: int = 20,
    ) -> None:
        super().__init__()
        self.patch = patch
        self.grid = grid_size // patch
        self.patch_embed = PatchEmbed(in_channels, dim, patch)
        n = self.grid * self.grid
        if pos_embed == "learned":
            self.pos = nn.Parameter(torch.zeros(1, n, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            self.register_buffer("pos", sincos_2d(self.grid, dim), persistent=False)
        self.spectral: SpectralGroupTokens | None = None
        if spectral_bands:
            self.spectral = SpectralGroupTokens(
                spectral_bands, spectral_groups, dim, patch, heads, dropout
            )
            self.fuse = nn.Linear(2 * dim, dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(dim, heads, mlp_ratio, dropout, attn_dropout) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, raw: torch.Tensor | None = None) -> torch.Tensor:
        """``feats [N, C, H, W]`` (+ raw spectrum) -> feature map ``[N, D, H/p, W/p]``."""
        tok = self.patch_embed(feats) + self.pos
        if self.spectral is not None and raw is not None:
            tok = tok + self.fuse(torch.cat([tok, self.spectral(raw)], dim=-1))
        tok = self.drop(tok)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return tok.transpose(1, 2).unflatten(2, (self.grid, self.grid))


def num_upsamples(patch: int) -> int:
    """Number of x2 stages needed to undo a power-of-two patch size."""
    return int(math.log2(patch))
