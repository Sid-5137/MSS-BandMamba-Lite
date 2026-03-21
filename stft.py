# Stage 1: Waveform ↔ Complex Spectrogram

import torch
import torch.nn as nn
from typing import Optional

from configs import BandMambaConfig


class STFTModule(nn.Module):

    def __init__(
        self,
        config: Optional[BandMambaConfig] = None,
        *,
        n_fft: Optional[int] = None,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        center: Optional[bool] = None,
    ):
        super().__init__()
        if config is None:
            config = BandMambaConfig()

        self.n_fft = config.n_fft if n_fft is None else n_fft
        self.hop_length = config.hop_length if hop_length is None else hop_length
        self.win_length = config.win_length if win_length is None else win_length
        self.center = config.center if center is None else center

        self.register_buffer("window", torch.hann_window(self.win_length))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            return_complex=True,
        )
        F_bins, T_frames = spec.shape[-2], spec.shape[-1]
        spec = spec.reshape(B, C, F_bins, T_frames)
        return spec

    def istft(self, spec: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """
        Args:   spec: (B, C, F, T_frames) complex64
                length: original sample count for trimming
        Returns: (B, C, T_samples)
        """
        B, C, F_bins, T_frames = spec.shape
        spec_flat = spec.reshape(B * C, F_bins, T_frames)
        x_flat = torch.istft(
            spec_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            length=length,
        )
        return x_flat.reshape(B, C, -1)


if __name__ == "__main__":
    x = torch.randn(2, 2, 44100)  # batch=2, stereo, 1 sec
    stft_mod = STFTModule(n_fft=4096, hop_length=1024)
    spec = stft_mod.stft(x)  # expect (2, 2, 2049, ~44) complex
    x_hat = stft_mod.istft(spec, length=44100)
    assert x_hat.shape == x.shape
    print(f"Reconstruction error: {(x - x_hat).abs().max():.6f}")
    # Should be < 1e-5 (perfect reconstruction with hann window + COLA)
