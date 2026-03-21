# Optimized approach:
#   - CPU workers: load audio, random chunk, augment
#   - GPU: only model forward/backward
#   - pin_memory=True for fast CPU→GPU transfer

import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple


class MUSDB18HQDataset(Dataset):
    """
    MUSDB18-HQ dataset loader.

    Each item returns a random chunk of:
      - mixture: (C, chunk_samples) stereo waveform
      - target:  (C, chunk_samples) stereo waveform of the target stem

    All loading and chunking happens on CPU (DataLoader workers).

    Expected folder structure:
      root/
        train/
          Track Name/
            mixture.wav
            vocals.wav
            drums.wav
            bass.wav
            other.wav
        test/
          Track Name/
            ...
    """

    STEMS = ["vocals", "drums", "bass", "other"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_stem: str = "vocals",
        chunk_duration: float = 5.0,
        sr: int = 44100,
        samples_per_track: int = 10,
        random_gain: bool = True,
        gain_range: Tuple[float, float] = (0.5, 1.5),
    ):
        """
        Args:
            root:             Path to musdb18hq/ folder
            split:            "train" or "test"
            target_stem:      Which stem to separate ("vocals", "drums", "bass", "other")
            chunk_duration:   Length of random chunks in seconds
            sr:               Sample rate (44100 for MUSDB18-HQ)
            samples_per_track: Number of random chunks to draw per track per epoch
                               (increases effective dataset size without loading more tracks)
            random_gain:      Whether to apply random gain augmentation
            gain_range:       Min/max gain multiplier for augmentation
        """
        assert target_stem in self.STEMS, f"target_stem must be one of {self.STEMS}"
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.root = root
        self.split = split
        self.target_stem = target_stem
        self.sr = sr
        self.chunk_samples = int(chunk_duration * sr)
        self.samples_per_track = samples_per_track if split == "train" else 1
        self.random_gain = random_gain and split == "train"
        self.gain_range = gain_range

        # Discover tracks
        split_dir = os.path.join(root, split)
        self.tracks = sorted(
            [
                d
                for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]
        )

        if len(self.tracks) == 0:
            raise RuntimeError(f"No tracks found in {split_dir}")

        # Validate first track has expected stems
        first_track = os.path.join(split_dir, self.tracks[0])
        for stem in ["mixture", target_stem]:
            wav_path = os.path.join(first_track, f"{stem}.wav")
            if not os.path.exists(wav_path):
                raise RuntimeError(f"Missing {wav_path}")

        print(
            f"MUSDB18-HQ [{split}]: {len(self.tracks)} tracks, "
            f"target='{target_stem}', "
            f"chunk={chunk_duration}s, "
            f"samples_per_track={self.samples_per_track}"
        )

    def __len__(self) -> int:
        return len(self.tracks) * self.samples_per_track

    def _load_audio(self, track_name: str, stem: str) -> torch.Tensor:
        """Load a stem WAV file. Returns (C, T_total) tensor on CPU."""
        path = os.path.join(self.root, self.split, track_name, f"{stem}.wav")
        waveform, file_sr = torchaudio.load(path)

        # Resample if needed (MUSDB18-HQ should already be 44100)
        if file_sr != self.sr:
            resampler = torchaudio.transforms.Resample(file_sr, self.sr)
            waveform = resampler(waveform)

        return waveform  # (C, T_total)

    def _random_chunk(self, *waveforms: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Extract the same random chunk from multiple waveforms.
        All waveforms must have the same length.
        """
        total_samples = waveforms[0].shape[-1]

        if total_samples <= self.chunk_samples:
            # Track is shorter than chunk: pad with zeros
            chunks = []
            for w in waveforms:
                padded = torch.zeros(w.shape[0], self.chunk_samples)
                padded[:, :total_samples] = w
                chunks.append(padded)
            return tuple(chunks)

        # Random start position
        start = random.randint(0, total_samples - self.chunk_samples)
        end = start + self.chunk_samples

        return tuple(w[:, start:end] for w in waveforms)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                'mixture': (C, chunk_samples) tensor
                'target':  (C, chunk_samples) tensor
                'track':   track name (for logging)
        """
        track_idx = idx // self.samples_per_track
        track_name = self.tracks[track_idx]

        # Load mixture and target stem (CPU)
        mixture = self._load_audio(track_name, "mixture")
        target = self._load_audio(track_name, self.target_stem)

        # Try up to 10 random chunks to find a good one
        for _attempt in range(10):
            mix_chunk, tgt_chunk = self._random_chunk(mixture, target)

            # Reject near-silent targets (causes gradient spikes in STFT loss)
            if tgt_chunk.abs().max() < 0.01:
                continue

            # Normalize if mixture has extreme values
            mix_peak = mix_chunk.abs().max()
            if mix_peak > 1.0:
                scale = 0.9 / mix_peak
                mix_chunk = mix_chunk * scale
                tgt_chunk = tgt_chunk * scale

            break  # good chunk found

        # Random gain augmentation (training only)
        if self.random_gain:
            gain = random.uniform(*self.gain_range)
            mix_chunk = mix_chunk * gain
            tgt_chunk = tgt_chunk * gain

        return {
            "mixture": mix_chunk,  # (C, chunk_samples)
            "target": tgt_chunk,  # (C, chunk_samples)
            "track": track_name,
        }


def create_dataloaders(
    root: str,
    target_stem: str = "vocals",
    chunk_duration: float = 5.0,
    batch_size: int = 4,
    num_workers: int = 4,
    sr: int = 44100,
    samples_per_track: int = 10,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders with optimized settings.

    Args:
        root:             Path to musdb18hq/ folder
        target_stem:      Which stem to separate
        chunk_duration:   Chunk length in seconds
        batch_size:       Training batch size
        num_workers:      CPU workers for data loading
        sr:               Sample rate
        samples_per_track: Chunks per track per epoch

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = MUSDB18HQDataset(
        root=root,
        split="train",
        target_stem=target_stem,
        chunk_duration=chunk_duration,
        sr=sr,
        samples_per_track=samples_per_track,
        random_gain=True,
    )

    test_dataset = MUSDB18HQDataset(
        root=root,
        split="test",
        target_stem=target_stem,
        chunk_duration=chunk_duration,
        sr=sr,
        samples_per_track=1,  # no oversampling for test
        random_gain=False,  # no augmentation for test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # randomize track order
        num_workers=num_workers,  # parallel CPU loading
        pin_memory=True,  # fast CPU→GPU transfer
        drop_last=True,  # avoid partial batches
        persistent_workers=True,  # keep workers alive between epochs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # full tracks one at a time for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader


# === TEST ===
if __name__ == "__main__":
    import sys

    # Update this path to your dataset location
    DATASET_ROOT = "/home/sid/Desktop/Audio_Source_Separation/Datasets/musdb18hq"

    if not os.path.exists(DATASET_ROOT):
        print(f"Dataset not found at {DATASET_ROOT}")
        print("Update DATASET_ROOT in the test section")
        sys.exit(1)

    # Test dataset
    print("=" * 60)
    print("  Testing MUSDB18HQ Dataset Loader")
    print("=" * 60)

    dataset = MUSDB18HQDataset(
        root=DATASET_ROOT,
        split="train",
        target_stem="vocals",
        chunk_duration=5.0,
        samples_per_track=10,
    )

    print(
        f"\nDataset length: {len(dataset)} "
        f"({len(dataset.tracks)} tracks × {dataset.samples_per_track} chunks)"
    )

    # Load one sample
    sample = dataset[0]
    print(f"\nSample:")
    print(f"  mixture shape: {sample['mixture'].shape}")  # expect (2, 220500)
    print(f"  target shape:  {sample['target'].shape}")  # expect (2, 220500)
    print(f"  track: {sample['track']}")
    print(f"  duration: {sample['mixture'].shape[-1] / 44100:.1f}s")

    assert sample["mixture"].shape == (
        2,
        220500,
    ), f"Unexpected shape: {sample['mixture'].shape}"
    assert sample["target"].shape == (
        2,
        220500,
    ), f"Unexpected shape: {sample['target'].shape}"
    print("  Shapes correct ✓")

    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader, test_loader = create_dataloaders(
        root=DATASET_ROOT,
        target_stem="vocals",
        batch_size=4,
        num_workers=2,
        samples_per_track=10,
    )

    batch = next(iter(train_loader))
    print(f"  Batch mixture: {batch['mixture'].shape}")  # expect (4, 2, 220500)
    print(f"  Batch target:  {batch['target'].shape}")  # expect (4, 2, 220500)
    print(f"  Batch tracks:  {batch['track']}")

    assert batch["mixture"].shape == (4, 2, 220500)
    assert batch["target"].shape == (4, 2, 220500)
    print("  Batch shapes correct ✓")

    # Verify random chunking gives different chunks
    s1 = dataset[0]["mixture"]
    s2 = dataset[0]["mixture"]
    if not torch.allclose(s1, s2):
        print("  Random chunking verified ✓")
    else:
        print("  Warning: same chunk returned (possible if track is very short)")

    print(f"\nTrain: {len(train_loader)} batches per epoch")
    print(f"Test:  {len(test_loader)} batches")
    print("\ndataset.py: ALL TESTS PASSED")
