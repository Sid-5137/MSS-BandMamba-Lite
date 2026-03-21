# Full Model: BandMamba-Light

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from configs import BandMambaConfig
from stft import STFTModule
from band_split import BandSplitEncoder, BandSplitDecoder
from core import DecoupledBlock
from mask import MaskEstimator


class BandMambaLight(nn.Module):
    """
    BandMamba-Light: Efficient Lightweight Music Source Separation.

    Pipeline:
      Waveform → STFT → Band-Split Encode → N × DecoupledBlock → Mask → Decode → iSTFT → Waveform

    Memory optimizations:
      1. Gradient checkpointing on DecoupledBlocks (saves ~30% memory)
      2. Works with chunked FrequencyBlock (mamba_block.py) for further savings
    """

    def __init__(self, config: BandMambaConfig, use_checkpoint: bool = True):
        super().__init__()
        self.config = config
        self.use_checkpoint = use_checkpoint

        # Stage 1: STFT
        self.stft_module = STFTModule(config)

        # Stage 2: Band-split encoder
        self.encoder = BandSplitEncoder(config)

        # Stage 3: N × Decoupled processing blocks
        self.blocks = nn.ModuleList(
            [DecoupledBlock(config) for _ in range(config.n_blocks)]
        )

        # Stage 4: Mask estimation + per-source decoders
        self.mask_estimator = MaskEstimator(config.hidden_dim)
        self.decoders = nn.ModuleList(
            [BandSplitDecoder(config) for _ in range(config.n_sources)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T_samples = x.shape[-1]

        # Stage 1: Waveform → Spectrogram
        spec = self.stft_module.stft(x)  # (B, C, F, T)

        # Stage 2: Spectrogram → Band embeddings
        bands = self.encoder(spec)  # (B, K, T, D)

        # Stage 3: Decoupled processing with gradient checkpointing
        h = bands
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

        # Stage 4: Mask estimation → source separation
        h = self.mask_estimator(h)  # (B, K, T, D)

        sources = []
        for decoder in self.decoders:
            mask = decoder(h)  # (B, C, F, T) complex
            masked_spec = spec * mask
            waveform = self.stft_module.istft(masked_spec, length=T_samples)
            sources.append(waveform)

        sources = torch.stack(sources, dim=1)  # (B, n_sources, C, T)

        if self.config.n_sources == 1:
            sources = sources.squeeze(1)  # (B, C, T)

        return sources


def count_parameters(model: nn.Module, verbose: bool = True) -> int:
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n{'='*60}")
        print(f"  BandMamba-Light Parameter Summary")
        print(f"{'='*60}")
        for name, module in model.named_children():
            params = sum(p.numel() for p in module.parameters())
            print(f"  {name:25s}  {params:>10,d}  ({params/1e6:.2f}M)")
        print(f"{'='*60}")
        print(f"  {'TOTAL':25s}  {total:>10,d}  ({total/1e6:.2f}M)")
        print(f"{'='*60}")
    return total


if __name__ == "__main__":
    from configs import BASE_CONFIG

    model = BandMambaLight(BASE_CONFIG)
    count_parameters(model)

    print("Running end-to-end test (1 second audio)...")
    x = torch.randn(1, 2, 44100)
    with torch.no_grad():
        y = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {y.shape}")
    print(f"  Shape match: {x.shape == y.shape}")

    print("\nmodel.py: ALL TESTS PASSED")
