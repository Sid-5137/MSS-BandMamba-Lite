# inference.py — BandMamba-Light Music Source Separation
#
# Supports two modes:
#   1. Single-stem: vocals model only → outputs vocals + instrumental
#   2. Multi-stem:  all 4 models → outputs vocals, drums, bass, other
#
# Usage:
#   # Single stem (vocals only)
#   python inference.py --input song.wav --vocals_checkpoint best_model_vocals.pt
#
#   # All stems (when all 4 models are trained)
#   python inference.py --input song.wav \
#       --vocals_checkpoint best_model_vocals.pt \
#       --drums_checkpoint best_model_drums.pt \
#       --bass_checkpoint best_model_bass.pt \
#       --other_checkpoint best_model_other.pt

import os
import argparse
import torch
import torchaudio
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from configs import BandMambaConfig, BASE_CONFIG
from model import BandMambaLight


def load_model(
    checkpoint_path: str,
    config: BandMambaConfig = BASE_CONFIG,
    device: torch.device = None,
) -> BandMambaLight:
    """Load a trained model from checkpoint."""
    model = BandMambaLight(config, use_checkpoint=False).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Handle DataParallel-saved checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"  Loaded from epoch {epoch} (val_loss: {val_loss:.4f})")

    return model


def overlap_add_separate(
    mixture: torch.Tensor,
    model: BandMambaLight,
    device: torch.device,
    sr: int = 44100,
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
) -> torch.Tensor:
    """
    Run separation on a full-length waveform using overlap-add.

    The model was trained on short chunks (5s), so for full songs we:
    1. Split into overlapping chunks (10s with 2s overlap)
    2. Process each chunk through the model
    3. Crossfade overlapping regions for smooth transitions

    Args:
        mixture: (C, T) stereo waveform
        model: trained BandMambaLight
        device: torch device
        sr: sample rate
        chunk_seconds: processing chunk length
        overlap_seconds: overlap between chunks

    Returns:
        separated: (C, T) separated stem
    """
    channels, total_samples = mixture.shape

    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    hop_samples = chunk_samples - overlap_samples

    output = torch.zeros_like(mixture)
    weights = torch.zeros(1, total_samples)

    fade_in = torch.linspace(0, 1, overlap_samples)
    fade_out = torch.linspace(1, 0, overlap_samples)

    n_chunks = max(1, (total_samples - overlap_samples + hop_samples - 1) // hop_samples)

    with torch.no_grad():
        for i in range(n_chunks):
            start = i * hop_samples
            end = min(start + chunk_samples, total_samples)
            actual_len = end - start

            # Extract and pad chunk
            chunk = mixture[:, start:end]
            if actual_len < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - actual_len))

            # Model forward pass
            chunk_out = model(chunk.unsqueeze(0).to(device)).squeeze(0).cpu()
            chunk_out = chunk_out[:, :actual_len]

            # Crossfade weights
            w = torch.ones(actual_len)
            if i > 0 and overlap_samples > 0:
                fade_len = min(overlap_samples, actual_len)
                chunk_out[:, :fade_len] *= fade_in[:fade_len]
                w[:fade_len] *= fade_in[:fade_len]
            if i < n_chunks - 1 and overlap_samples > 0:
                fade_len = min(overlap_samples, actual_len)
                chunk_out[:, -fade_len:] *= fade_out[-fade_len:]
                w[-fade_len:] *= fade_out[-fade_len:]

            output[:, start:end] += chunk_out
            weights[0, start:end] += w

            print(f"\r  Processing: {(i+1)/n_chunks*100:.0f}%", end="", flush=True)

    print()
    return output / weights.clamp(min=1e-8)


def separate_single_stem(
    input_path: str,
    checkpoint_path: str,
    output_dir: str = "./separated",
    config: BandMambaConfig = BASE_CONFIG,
    device: str = "auto",
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    Separate vocals only. Outputs vocals + instrumental (residual).

    Args:
        input_path: path to input audio
        checkpoint_path: path to vocals model checkpoint

    Returns:
        dict with "vocals" and "instrumental" tensors
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load audio
    print(f"Loading: {input_path}")
    waveform, file_sr = torchaudio.load(input_path)
    sr = 44100
    if file_sr != sr:
        waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    print(f"Audio: {waveform.shape[1]/sr:.1f}s, {waveform.shape[0]}ch")

    # Load model
    print(f"Loading vocals model:")
    model = load_model(checkpoint_path, config, device)

    # Separate
    print(f"Separating vocals...")
    vocals = overlap_add_separate(waveform, model, device, sr, chunk_seconds, overlap_seconds)
    instrumental = waveform - vocals

    # Save
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    paths = {}
    for name, audio in [("vocals", vocals), ("instrumental", instrumental)]:
        path = os.path.join(output_dir, f"{basename}_{name}.wav")
        torchaudio.save(path, audio, sr)
        paths[name] = path
        print(f"Saved: {path}")

    return {"vocals": vocals, "instrumental": instrumental}


def separate_all_stems(
    input_path: str,
    checkpoints: Dict[str, str],
    output_dir: str = "./separated",
    config: BandMambaConfig = BASE_CONFIG,
    device: str = "auto",
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    Separate all 4 stems using individual models.

    Each model independently estimates its stem from the mixture.
    No Wiener post-filtering — just raw model outputs.

    Args:
        input_path: path to input audio
        checkpoints: dict mapping stem name to checkpoint path
            e.g. {"vocals": "best_vocals.pt", "drums": "best_drums.pt", ...}

    Returns:
        dict with stem name → (C, T) tensor
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load audio
    print(f"Loading: {input_path}")
    waveform, file_sr = torchaudio.load(input_path)
    sr = 44100
    if file_sr != sr:
        waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    print(f"Audio: {waveform.shape[1]/sr:.1f}s, {waveform.shape[0]}ch")

    # Process each stem
    stems = {}
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    for stem_name, ckpt_path in checkpoints.items():
        print(f"\nLoading {stem_name} model:")
        model = load_model(ckpt_path, config, device)

        print(f"Separating {stem_name}...")
        separated = overlap_add_separate(
            waveform, model, device, sr, chunk_seconds, overlap_seconds
        )
        stems[stem_name] = separated

        # Save
        path = os.path.join(output_dir, f"{basename}_{stem_name}.wav")
        torchaudio.save(path, separated, sr)
        print(f"Saved: {path}")

        # Free model memory
        del model
        torch.cuda.empty_cache()

    return stems


# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BandMamba-Light Music Source Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Vocals only (single model)
  python inference.py --input song.wav --vocals_checkpoint best_vocals.pt

  # All 4 stems (when all models are trained)
  python inference.py --input song.wav \\
      --vocals_checkpoint best_vocals.pt \\
      --drums_checkpoint best_drums.pt \\
      --bass_checkpoint best_bass.pt \\
      --other_checkpoint best_other.pt
        """,
    )
    parser.add_argument("--input", type=str, required=True, help="Input audio file")
    parser.add_argument("--vocals_checkpoint", type=str, default=None)
    parser.add_argument("--drums_checkpoint", type=str, default=None)
    parser.add_argument("--bass_checkpoint", type=str, default=None)
    parser.add_argument("--other_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./separated")
    parser.add_argument("--chunk_seconds", type=float, default=10.0)
    parser.add_argument("--overlap_seconds", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Collect available checkpoints
    checkpoints = {}
    if args.vocals_checkpoint:
        checkpoints["vocals"] = args.vocals_checkpoint
    if args.drums_checkpoint:
        checkpoints["drums"] = args.drums_checkpoint
    if args.bass_checkpoint:
        checkpoints["bass"] = args.bass_checkpoint
    if args.other_checkpoint:
        checkpoints["other"] = args.other_checkpoint

    if len(checkpoints) == 0:
        parser.error("Provide at least --vocals_checkpoint")

    if len(checkpoints) == 1 and "vocals" in checkpoints:
        # Single-stem mode: vocals + instrumental
        print("=" * 50)
        print("  Mode: Vocals separation")
        print("=" * 50)
        separate_single_stem(
            input_path=args.input,
            checkpoint_path=checkpoints["vocals"],
            output_dir=args.output_dir,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            device=args.device,
        )
    else:
        # Multi-stem mode
        print("=" * 50)
        print(f"  Mode: Multi-stem separation ({', '.join(checkpoints.keys())})")
        print("=" * 50)
        separate_all_stems(
            input_path=args.input,
            checkpoints=checkpoints,
            output_dir=args.output_dir,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            device=args.device,
        )

    print("\nDone!")
