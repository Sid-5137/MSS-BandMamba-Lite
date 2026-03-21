# Loss functions for training BandMamba-Light.
#
# Based on findings from Gusó et al. "On Loss Functions and Evaluation Metrics
# for Music Source Separation" (2022), which benchmarked 18 losses:
#   - Best objective (SDR): SISDRfreq (5.61 dB)
#   - Best subjective (MOS): LOGL1freq and L2freq (tied)
#   - Worst: MRS/multi-resolution STFT (4.80 dB), mask-based losses
#   - Spectral convergence (Lsc) is unstable due to division by target norm
#
# Our strategy:
#   Phase 1 (warm-up):  L1freq + LOGL1freq — stable spectral losses
#   Phase 2 (fine-tune): SI-SDR + L1freq — optimize evaluation metric

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class L1FreqLoss(nn.Module):
    """
    L1 loss on magnitude spectrogram.
    Ranked well in Gusó et al. (4.92 dB SDR) and correlates best with
    human judgment (Fig. 5 in paper).
    """

    def __init__(self, n_fft: int = 4096, hop_length: int = 1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def _stft_mag(self, x):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return spec.abs()

    def forward(self, predicted, target):
        pred_mag = self._stft_mag(predicted)
        target_mag = self._stft_mag(target)
        return F.l1_loss(pred_mag, target_mag)


class LogL1FreqLoss(nn.Module):
    """
    Log-compressed L1 on magnitude spectrogram (LOGL1freq).
    One of the best losses in Gusó et al. (5.53 dB SDR, MOS 58.96).
    Log compression attenuates low-energy artifacts from mask-based models.
    """

    def __init__(self, n_fft: int = 4096, hop_length: int = 1024, eps: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.eps = eps
        self.register_buffer("window", torch.hann_window(n_fft))

    def _stft_mag(self, x):
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        return spec.abs()

    def forward(self, predicted, target):
        pred_mag = self._stft_mag(predicted)
        target_mag = self._stft_mag(target)
        return F.l1_loss(
            torch.log(pred_mag + self.eps),
            torch.log(target_mag + self.eps),
        )


class SISDRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Distortion Ratio loss (time domain).
    Returns NEGATIVE SI-SDR so minimizing loss = maximizing SI-SDR.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predicted, target):
        predicted = predicted.reshape(-1, predicted.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        predicted = predicted - predicted.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        dot = (predicted * target).sum(dim=-1, keepdim=True)
        target_energy = (target * target).sum(dim=-1, keepdim=True) + self.eps
        s_target = (dot / target_energy) * target

        e_noise = predicted - s_target

        si_sdr = 10 * torch.log10(
            (s_target * s_target).sum(dim=-1)
            / ((e_noise * e_noise).sum(dim=-1) + self.eps)
        )

        return -si_sdr.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss for BandMamba-Light training.

    Phase 1 (warm-up):   L = α * L1freq + β * LOGL1freq
    Phase 2 (fine-tune):  L = α * (-SI-SDR) + β * L1freq
    """

    def __init__(
        self,
        phase: int = 1,
        alpha: float = 1.0,
        beta: float = 1.0,
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        super().__init__()
        self.phase = phase
        self.alpha = alpha
        self.beta = beta

        self.l1_freq = L1FreqLoss(n_fft=n_fft, hop_length=hop_length)
        self.logl1_freq = LogL1FreqLoss(n_fft=n_fft, hop_length=hop_length)
        self.sisdr_loss = SISDRLoss()

    def set_phase(self, phase: int):
        assert phase in (1, 2)
        self.phase = phase
        phase_desc = "L1freq + LOGL1freq" if phase == 1 else "SI-SDR + L1freq"
        print(f"Loss switched to Phase {phase}: {phase_desc}")

    def forward(self, predicted, target):
        l1_freq = self.l1_freq(predicted, target)

        if self.phase == 1:
            logl1_freq = self.logl1_freq(predicted, target)
            time_loss = l1_freq
            freq_loss = logl1_freq
            total_loss = self.alpha * l1_freq + self.beta * logl1_freq
        else:
            sisdr = self.sisdr_loss(predicted, target)
            time_loss = sisdr
            freq_loss = l1_freq
            total_loss = self.alpha * sisdr + self.beta * l1_freq

        return {
            "loss": total_loss,
            "time_loss": time_loss.detach(),
            "freq_loss": freq_loss.detach(),
        }


# === TEST ===
if __name__ == "__main__":
    B, C, T = 2, 2, 44100

    target = torch.randn(B, C, T)
    predicted = target + 0.1 * torch.randn(B, C, T)
    bad_pred = torch.randn(B, C, T)

    l1f = L1FreqLoss()
    assert l1f(predicted, target) < l1f(bad_pred, target)
    print(
        f"L1freq:    good={l1f(predicted, target):.4f}  bad={l1f(bad_pred, target):.4f} ✓"
    )

    logl1f = LogL1FreqLoss()
    assert logl1f(predicted, target) < logl1f(bad_pred, target)
    print(
        f"LOGL1freq: good={logl1f(predicted, target):.4f}  bad={logl1f(bad_pred, target):.4f} ✓"
    )

    sisdr = SISDRLoss()
    assert sisdr(predicted, target) < sisdr(bad_pred, target)
    print(
        f"SI-SDR:    good={sisdr(predicted, target):.4f}  bad={sisdr(bad_pred, target):.4f} ✓"
    )

    # Silent target test
    silent_target = torch.zeros(B, C, T) + 1e-6
    silent_pred = torch.randn(B, C, T) * 0.01
    assert torch.isfinite(l1f(silent_pred, silent_target))
    assert torch.isfinite(logl1f(silent_pred, silent_target))
    print("Silent target: no explosion ✓")

    combined = CombinedLoss(phase=1)
    r1 = combined(predicted, target)
    print(f"\nPhase 1: total={r1['loss']:.4f} ✓")

    combined.set_phase(2)
    r2 = combined(predicted, target)
    print(f"Phase 2: total={r2['loss']:.4f} ✓")

    print("\nlosses.py: ALL TESTS PASSED")
