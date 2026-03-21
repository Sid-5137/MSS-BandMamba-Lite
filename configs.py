# bandmamba_light/configs.py
# ==========================

from dataclasses import dataclass


@dataclass
class BandMambaConfig:
    """
    Central config for BandMamba-Light.

    Parameter counts (analytical):
      Small:  hidden_dim=96,  n_blocks=4, temporal_layers=2, mamba_layers=2 → ~2.65M
      Base:   hidden_dim=128, n_blocks=4, temporal_layers=2, mamba_layers=2 → ~4.18M  ★
      Large:  hidden_dim=128, n_blocks=6, temporal_layers=1, mamba_layers=2 → ~5.16M
    """

    # --- Stage 1: STFT ---
    sr: int = 44100
    n_fft: int = 4096
    hop_length: int = 1024
    win_length: int = 4096  # typically = n_fft
    center: bool = True

    # --- Stage 2: Band-Split ---
    n_bands: int = 60  # number of mel-scale subbands
    channels: int = 2  # stereo input
    min_compression: int = 1  # compression ratio for lowest band
    max_compression: int = 8  # compression ratio for highest band

    # --- Stage 3: Processing Core ---
    hidden_dim: int = 128  # D — main hidden dimension

    # 3a. Temporal path
    temporal_kernel_size: int = 31  # large kernel for local patterns (try 31, 65)
    temporal_layers: int = 2  # conv layers per temporal block
    temporal_expand: float = 2.0  # expansion ratio in depthwise sep conv

    # 3b. Frequency path (Mamba)
    mamba_state_dim: int = 16  # N — SSM state dimension
    mamba_layers: int = 2  # Mamba layers per frequency block
    mamba_expand: float = 2.0  # expansion ratio in Mamba block
    mamba_dt_rank: int | None = None  # if None, defaults to hidden_dim // 16
    mamba_conv_kernel: int = 4  # local conv inside Mamba block

    # 3d. Overall
    n_blocks: int = 4  # number of DecoupledBlocks to stack

    # --- Stage 4: Output ---
    n_sources: int = 1  # stems to separate (1 = single-stem model)

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins from STFT."""
        return self.n_fft // 2 + 1

    @property
    def dt_rank(self) -> int:
        """Effective dt_rank for Mamba."""
        if self.mamba_dt_rank is not None:
            return self.mamba_dt_rank
        return max(1, self.hidden_dim // 16)


# Pre-defined configs
SMALL_CONFIG = BandMambaConfig(hidden_dim=96, n_blocks=4)  # ~2.65M
BASE_CONFIG = BandMambaConfig(hidden_dim=128, n_blocks=4)  # ~4.18M
LARGE_CONFIG = BandMambaConfig(hidden_dim=128, n_blocks=6, temporal_layers=1)  # ~5.16M


# === TEST ===
if __name__ == "__main__":
    cfg = BASE_CONFIG
    print(f"Config: hidden_dim={cfg.hidden_dim}, n_blocks={cfg.n_blocks}")
    print(f"  n_freqs={cfg.n_freqs}, dt_rank={cfg.dt_rank}")
    print(f"  n_bands={cfg.n_bands}, n_sources={cfg.n_sources}")
