# Stage 3a: Temporal Path — Large-Kernel Depthwise Separable Convolutions

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import BandMambaConfig


class DepthwiseSeparableConv1d(nn.Module):
    """
    Single depthwise separable conv block with residual.
    (B, T, D) → (B, T, D)

    Architecture:
      LayerNorm → Linear (expand) → GELU → Depthwise Conv1d → GELU → Linear (project) → + residual
    """

    def __init__(self, dim: int, kernel_size: int = 31, expand_ratio: float = 2.0):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel for symmetric padding"
        padding = kernel_size // 2
        inner_dim = int(dim * expand_ratio)

        self.norm = nn.LayerNorm(dim)
        self.pw_in = nn.Linear(dim, inner_dim)  # pointwise expansion
        self.dw_conv = nn.Conv1d(  # depthwise convolution
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=inner_dim,  # groups=inner_dim makes it depthwise
        )
        self.pw_out = nn.Linear(inner_dim, dim)  # pointwise projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, T, D)"""
        residual = x

        x = self.norm(x)
        x = F.gelu(self.pw_in(x))  # (B, T, inner_dim)
        x = x.permute(0, 2, 1)  # (B, inner_dim, T) — Conv1d expects channels-first
        x = F.gelu(self.dw_conv(x))  # (B, inner_dim, T)
        x = x.permute(0, 2, 1)  # (B, T, inner_dim) — back to channels-last
        x = self.pw_out(x)  # (B, T, D)

        return x + residual


class TemporalBlock(nn.Module):
    """
    Temporal path: stack of depthwise sep convs, applied independently per band.
    (B, K, T, D) → (B, K, T, D)

    All K bands are processed in parallel by merging into the batch dimension.
    """

    def __init__(self, config: BandMambaConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(
                    dim=config.hidden_dim,
                    kernel_size=config.temporal_kernel_size,
                    expand_ratio=config.temporal_expand,
                )
                for _ in range(config.temporal_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, T, D) → (B, K, T, D)"""
        B, K, T, D = x.shape

        # Merge bands into batch: (B*K, T, D)
        x = x.reshape(B * K, T, D)

        for layer in self.layers:
            x = layer(x)

        # Split back: (B, K, T, D)
        return x.reshape(B, K, T, D)


# === TEST ===
if __name__ == "__main__":
    from configs import BandMambaConfig

    cfg = BandMambaConfig()

    # Test single conv block
    block = DepthwiseSeparableConv1d(dim=128, kernel_size=31)
    x = torch.randn(4, 100, 128)
    y = block(x)
    assert y.shape == x.shape
    print(f"DepthwiseSeparableConv1d: {x.shape} → {y.shape} ✓")

    # Test residual (output should differ from input)
    assert not torch.allclose(x, y)
    print(f"Residual connection working ✓")

    # Test temporal block
    temporal = TemporalBlock(cfg)
    x = torch.randn(2, 60, 44, 128)  # B=2, K=60 bands, T=44 frames, D=128
    y = temporal(x)
    assert y.shape == x.shape
    print(f"TemporalBlock: {x.shape} → {y.shape} ✓")

    # Parameter count
    params = sum(p.numel() for p in temporal.parameters())
    print(f"\nTemporal params: {params:,d} ({params/1e6:.3f}M)")
    print("\ntemporal.py: ALL TESTS PASSED")
