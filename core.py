# Stage 3d: DecoupledBlock — wires temporal + frequency + fusion gate

import torch
import torch.nn as nn

from configs import BandMambaConfig
from temporal import TemporalBlock
from mamba_block import FrequencyBlock
from fusion import AdaptiveFusionGate


class DecoupledBlock(nn.Module):
    """
    One decoupled processing block:
      temporal conv → frequency Mamba → adaptive fusion.
    (B, K, T, D) → (B, K, T, D)
    """

    def __init__(self, config: BandMambaConfig):
        super().__init__()
        self.temporal = TemporalBlock(config)
        self.frequency = FrequencyBlock(config)
        self.gate = AdaptiveFusionGate(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, K, T, D) → (B, K, T, D)"""
        h_temporal = self.temporal(x)
        h_frequency = self.frequency(x)
        return self.gate(h_temporal, h_frequency)


# === TEST ===
if __name__ == "__main__":
    from configs import BandMambaConfig

    cfg = BandMambaConfig()
    block = DecoupledBlock(cfg)
    x = torch.randn(2, 60, 44, 128)
    y = block(x)
    assert y.shape == x.shape
    params = sum(p.numel() for p in block.parameters())
    print(f"DecoupledBlock: {x.shape} → {y.shape} ✓")
    print(f"  Temporal:  {sum(p.numel() for p in block.temporal.parameters()):,d}")
    print(f"  Frequency: {sum(p.numel() for p in block.frequency.parameters()):,d}")
    print(f"  Gate:      {sum(p.numel() for p in block.gate.parameters()):,d}")
    print(f"  Total:     {params:,d} ({params/1e6:.3f}M)")
    print("\ncore.py: ALL TESTS PASSED")
