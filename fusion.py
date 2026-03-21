# Stage 3c: Adaptive Fusion Gate (NOVEL CONTRIBUTION)

import torch
import torch.nn as nn

from configs import BandMambaConfig


class AdaptiveFusionGate(nn.Module):
    """
    Learned per-band, per-frame gating between temporal and frequency paths.
    G = σ(W · [h_temporal; h_frequency])
    Output = G ⊙ h_temporal + (1-G) ⊙ h_frequency
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(
        self, h_temporal: torch.Tensor, h_frequency: torch.Tensor
    ) -> torch.Tensor:
        """(B, K, T, D) × (B, K, T, D) → (B, K, T, D)"""
        G = self.gate(torch.cat([h_temporal, h_frequency], dim=-1))
        return G * h_temporal + (1 - G) * h_frequency


# === TEST ===
if __name__ == "__main__":
    gate = AdaptiveFusionGate(dim=128)
    h_t = torch.randn(2, 60, 44, 128)
    h_f = torch.randn(2, 60, 44, 128)
    out = gate(h_t, h_f)
    assert out.shape == (2, 60, 44, 128)
    params = sum(p.numel() for p in gate.parameters())
    print(f"AdaptiveFusionGate: {out.shape} ✓")
    print(f"Params: {params:,d} ({params/1e6:.3f}M)")
    print("\nfusion.py: ALL TESTS PASSED")
