import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from configs import BandMambaConfig


def compute_mel_band_edges(
    n_bands: int, n_fft: int, sr: int = 44100
) -> List[Tuple[int, int]]:
    """
    Compute mel-scale frequency band edges.
    Returns list of (start_bin, end_bin) for K mel-scale bands.
    Lower bands are narrower (more resolution where it matters).
    """
    n_freqs = n_fft // 2 + 1
    max_freq = sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    # Equally spaced edges in mel scale
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(max_freq)
    mel_edges = torch.linspace(mel_min, mel_max, n_bands + 1)
    hz_edges = [mel_to_hz(m.item()) for m in mel_edges]

    # Convert Hz to FFT bin indices
    bin_edges = [int(round(hz / max_freq * (n_freqs - 1))) for hz in hz_edges]

    # Build bands, ensuring no empty bands
    bands = []
    for i in range(n_bands):
        start = min(bin_edges[i], n_freqs - 1)
        end = min(bin_edges[i + 1], n_freqs)
        if end <= start:
            end = start + 1
        bands.append((start, end))

    return bands


class BandSplitEncoder(nn.Module):
    """
    Splits complex spectrogram into mel-scale bands with sparse compression,
    then projects each band to hidden_dim.
    (B, C, F, T) complex → (B, K, T, D)
    """

    def __init__(self, config: BandMambaConfig):
        super().__init__()
        self.n_bands = config.n_bands
        self.hidden_dim = config.hidden_dim
        self.channels = config.channels

        # Compute band edges
        self.band_edges = compute_mel_band_edges(
            config.n_bands, config.n_fft, config.sr
        )

        # Compression ratios: linear ramp from min to max
        self.compression_ratios = []
        for i in range(config.n_bands):
            ratio = config.min_compression + (
                (config.max_compression - config.min_compression)
                * (i / max(config.n_bands - 1, 1))
            )
            self.compression_ratios.append(max(1, int(round(ratio))))

        # Per-band projection layers (different input dims per band!)
        self.band_projections = nn.ModuleList()
        self.band_norms = nn.ModuleList()

        for i, (start, end) in enumerate(self.band_edges):
            band_width = end - start
            comp_ratio = self.compression_ratios[i]
            compressed_width = max(1, math.ceil(band_width / comp_ratio))
            # Input: channels * compressed_width * 2 (real + imag)
            input_dim = config.channels * compressed_width * 2
            self.band_projections.append(nn.Linear(input_dim, config.hidden_dim))
            self.band_norms.append(nn.LayerNorm(config.hidden_dim))

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:   spec: (B, C, F, T) complex spectrogram
        Returns: (B, K, T, D) band embeddings
        """
        B, C, _, T = spec.shape
        band_outputs = []

        for i, (start, end) in enumerate(self.band_edges):
            # 1. Slice this band's frequency bins
            band = spec[:, :, start:end, :]  # (B, C, bw, T)
            band_width = end - start
            comp_ratio = self.compression_ratios[i]
            compressed_width = max(1, math.ceil(band_width / comp_ratio))

            # 2. Sparse compression via adaptive avg pool (real and imag separately)
            band_real = F.adaptive_avg_pool2d(
                band.real, (compressed_width, T)
            )  # (B, C, cw, T)
            band_imag = F.adaptive_avg_pool2d(
                band.imag, (compressed_width, T)
            )  # (B, C, cw, T)

            # 3. Flatten: (B, C, cw, T) → (B, T, C * cw)
            band_real = band_real.permute(0, 3, 1, 2).reshape(
                B, T, C * compressed_width
            )
            band_imag = band_imag.permute(0, 3, 1, 2).reshape(
                B, T, C * compressed_width
            )

            # 4. Concat real + imag → (B, T, C * cw * 2)
            band_flat = torch.cat([band_real, band_imag], dim=-1)

            # 5. Project to hidden dim + normalize
            h = self.band_projections[i](band_flat)  # (B, T, D)
            h = self.band_norms[i](h)
            band_outputs.append(h)

        # Stack all bands: (B, K, T, D)
        return torch.stack(band_outputs, dim=1)


class BandSplitDecoder(nn.Module):
    """
    Inverse of encoder: projects band embeddings back to per-band frequency
    bins to produce a complex mask.
    (B, K, T, D) → (B, C, F, T) complex mask
    """

    def __init__(self, config: BandMambaConfig):
        super().__init__()
        self.n_bands = config.n_bands
        self.channels = config.channels
        self.n_freqs = config.n_freqs

        self.band_edges = compute_mel_band_edges(
            config.n_bands, config.n_fft, config.sr
        )

        # Per-band projection: hidden_dim → channels * band_width * 2 (real + imag mask)
        self.band_projections = nn.ModuleList()
        for i, (start, end) in enumerate(self.band_edges):
            band_width = end - start
            output_dim = config.channels * band_width * 2
            self.band_projections.append(nn.Linear(config.hidden_dim, output_dim))

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        """
        Args:   bands: (B, K, T, D)
        Returns: (B, C, F, T) complex mask (tanh-bounded)
        """
        B, K, T, D = bands.shape
        n_freqs = self.n_freqs

        # Initialize full-size mask
        mask_real = torch.zeros(B, self.channels, n_freqs, T, device=bands.device)
        mask_imag = torch.zeros(B, self.channels, n_freqs, T, device=bands.device)

        for i, (start, end) in enumerate(self.band_edges):
            band_width = end - start
            h = bands[:, i, :, :]  # (B, T, D)

            # Project to mask values
            out = self.band_projections[i](h)  # (B, T, C * bw * 2)

            # Reshape: (B, T, 2, C, bw) → split real and imag
            out = out.reshape(B, T, 2, self.channels, band_width)

            # Place into full mask: permute (B, T, C, bw) → (B, C, bw, T)
            mask_real[:, :, start:end, :] = out[:, :, 0, :, :].permute(0, 2, 3, 1)
            mask_imag[:, :, start:end, :] = out[:, :, 1, :, :].permute(0, 2, 3, 1)

        # Tanh bounding to keep mask values in [-1, 1]
        return torch.complex(torch.tanh(mask_real), torch.tanh(mask_imag))


# === TEST ===
if __name__ == "__main__":
    from configs import BandMambaConfig

    cfg = BandMambaConfig()
    enc = BandSplitEncoder(cfg)
    dec = BandSplitDecoder(cfg)

    # Dummy complex spectrogram: (B=2, C=2, F=2049, T=44)
    spec = torch.randn(2, 2, cfg.n_freqs, 44) + 1j * torch.randn(2, 2, cfg.n_freqs, 44)

    # Encoder test
    bands = enc(spec)
    print(f"Encoder output: {bands.shape}")  # expect (2, 60, 44, 128)
    assert bands.shape == (2, cfg.n_bands, 44, cfg.hidden_dim)

    # Decoder test
    mask = dec(bands)
    print(f"Decoder output: {mask.shape}")  # expect (2, 2, 2049, 44)
    print(f"Mask is complex: {mask.is_complex()}")
    print(f"Mask value range: [{mask.real.min():.3f}, {mask.real.max():.3f}]")

    # Parameter counts
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"\nEncoder params: {enc_params:,d} ({enc_params/1e6:.2f}M)")
    print(f"Decoder params: {dec_params:,d} ({dec_params/1e6:.2f}M)")
    print("\nband_split.py: ALL TESTS PASSED")
