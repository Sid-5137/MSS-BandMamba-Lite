# Stage 4: Mask Estimator MLP

import torch
import torch.nn as nn

from configs import BandMambaConfig


class MaskEstimator(nn.Module):
    """
    2-layer MLP for mask feature refinement before decoding.
    (B, K, T, D) → (B, K, T, D)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# === TEST ===
if __name__ == "__main__":
    mask_est = MaskEstimator(hidden_dim=128)
    x = torch.randn(2, 60, 44, 128)
    y = mask_est(x)
    assert y.shape == x.shape
    params = sum(p.numel() for p in mask_est.parameters())
    print(f"MaskEstimator: {x.shape} → {y.shape} ✓")
    print(f"Params: {params:,d} ({params/1e6:.3f}M)")
    print("\nmask.py: ALL TESTS PASSED")
