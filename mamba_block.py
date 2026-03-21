# Stage 3b: Frequency Path — Bidirectional Mamba
# CUDA KERNEL VERSION: uses mamba-ssm package for 10-20x faster SSM scan

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from configs import BandMambaConfig


class BidirectionalMambaBlock(nn.Module):
    """
    Bidirectional Mamba using the official CUDA kernel.
    (B, L, D) → (B, L, D)

    Uses mamba_ssm.Mamba which fuses the entire SSM scan into a single
    CUDA kernel call — no Python for-loop, no sequential GPU launches.
    """

    def __init__(
        self, dim, state_dim=16, expand_ratio=2.0, dt_rank=None, conv_kernel=4
    ):
        super().__init__()
        self.forward_mamba = Mamba(
            d_model=dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=int(expand_ratio),
        )
        self.backward_mamba = Mamba(
            d_model=dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=int(expand_ratio),
        )
        self.combine = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """(B, L, D) → (B, L, D)"""
        h_fwd = self.forward_mamba(x)
        h_bwd = self.backward_mamba(torch.flip(x, [1]))
        h_bwd = torch.flip(h_bwd, [1])
        h_combined = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.norm(self.combine(h_combined))


class FrequencyBlock(nn.Module):
    """
    Frequency path: BiMamba layers across bands.
    (B, K, T, D) → (B, K, T, D)

    With the CUDA kernel, we can process all time frames at once
    without chunking — the kernel is memory-efficient internally.
    chunk_size is kept for backward compatibility but can be set large.
    """

    def __init__(self, config: BandMambaConfig, chunk_size: int = 64):
        super().__init__()
        self.chunk_size = chunk_size
        self.layers = nn.ModuleList(
            [
                BidirectionalMambaBlock(
                    dim=config.hidden_dim,
                    state_dim=config.mamba_state_dim,
                    expand_ratio=config.mamba_expand,
                    dt_rank=config.dt_rank,
                    conv_kernel=config.mamba_conv_kernel,
                )
                for _ in range(config.mamba_layers)
            ]
        )

    def forward(self, x):
        """(B, K, T, D) → (B, K, T, D)"""
        B, K, T, D = x.shape

        x = x.permute(0, 2, 1, 3)  # (B, T, K, D)

        chunks = []
        for t_start in range(0, T, self.chunk_size):
            t_end = min(t_start + self.chunk_size, T)
            chunk = x[:, t_start:t_end].reshape(-1, K, D)  # (B*chunk_len, K, D)

            for layer in self.layers:
                chunk = layer(chunk)

            chunks.append(chunk.reshape(B, t_end - t_start, K, D))

        x = torch.cat(chunks, dim=1)  # (B, T, K, D)
        return x.permute(0, 2, 1, 3)  # (B, K, T, D)


# === TEST ===
if __name__ == "__main__":
    from configs import BandMambaConfig

    cfg = BandMambaConfig()

    print("Testing BidirectionalMambaBlock (CUDA kernel)...")
    bidir = BidirectionalMambaBlock(
        dim=128, state_dim=16, expand_ratio=2.0, conv_kernel=4
    ).cuda()
    x = torch.randn(4, 60, 128).cuda()
    y = bidir(x)
    assert y.shape == x.shape
    print(f"  BiMamba: {x.shape} → {y.shape} ✓")

    print("Testing FrequencyBlock (CUDA kernel)...")
    freq = FrequencyBlock(cfg, chunk_size=64).cuda()
    x = torch.randn(2, 60, 44, 128).cuda()
    y = freq(x)
    assert y.shape == x.shape
    print(f"  FrequencyBlock: {x.shape} → {y.shape} ✓")

    params = sum(p.numel() for p in freq.parameters())
    print(f"\nFrequency path params: {params:,d} ({params/1e6:.3f}M)")

    # Speed test
    import time

    freq.train()
    x = torch.randn(4, 60, 216, 128).cuda()  # full 5-sec batch
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        y = freq(x)
        loss = y.sum()
        loss.backward()
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 10
    print(f"Avg forward+backward time: {elapsed*1000:.0f}ms")

    print("\nmamba_block.py (CUDA kernel): ALL TESTS PASSED")
