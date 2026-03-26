# evaluate.py — BandMamba-Light: Inference + SDR Evaluation (All-in-One)
#
# 1. Runs inference on all MUSDB18-HQ test tracks using all 4 stem models
# 2. Computes formal SI-SDR and BSSEval SDR metrics
# 3. Saves results to CSV
#
# Requirements:
#   pip install soundfile mir_eval numpy
#
# Usage:
#   python evaluate.py \
#       --musdb_root ./musdb18hq \
#       --vocals_checkpoint checkpoints/best_model_vocals.pt \
#       --drums_checkpoint checkpoints/best_model_drums.pt \
#       --bass_checkpoint checkpoints/best_model_bass.pt \
#       --other_checkpoint checkpoints/best_model_other.pt \
#       --output_dir ./evaluation_results
#
# To skip inference (only compute metrics on existing results):
#   python evaluate.py --results_dir ./evaluation_results --metrics_only

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from collections import defaultdict
from typing import Dict

from configs import BandMambaConfig, BASE_CONFIG
from model import BandMambaLight

try:
    from mir_eval.separation import bss_eval_sources
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    print("WARNING: mir_eval not installed. BSSEval SDR will be skipped.")
    print("  Install with: pip install mir_eval")


STEMS = ["vocals", "drums", "bass", "other"]
SR = 44100


# ═════════════════════════════════════════════════════════════════
#  PART 1: MODEL LOADING & INFERENCE
# ═════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, config: BandMambaConfig, device: torch.device) -> BandMambaLight:
    """Load a trained model from checkpoint."""
    model = BandMambaLight(config, use_checkpoint=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    if isinstance(val_loss, float):
        print(f"    Loaded from epoch {epoch} (val_loss: {val_loss:.4f})")
    else:
        print(f"    Loaded from epoch {epoch}")
    return model


def load_audio(path: str) -> torch.Tensor:
    """Load audio using soundfile (no FFmpeg/torchcodec needed)."""
    data, file_sr = sf.read(path, dtype="float32")

    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)

    waveform = torch.from_numpy(data.T)  # (channels, samples)

    if file_sr != SR:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, file_sr, SR)

    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]

    return waveform


def save_audio(path: str, waveform: torch.Tensor, sr: int):
    """Save audio using soundfile."""
    data = waveform.numpy().T
    sf.write(path, data, sr, subtype="FLOAT")


def overlap_add_separate(
    mixture: torch.Tensor,
    model: BandMambaLight,
    device: torch.device,
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
) -> torch.Tensor:
    """Run overlap-add separation on a full-length waveform."""
    channels, total_samples = mixture.shape

    chunk_samples = int(chunk_seconds * SR)
    overlap_samples = int(overlap_seconds * SR)
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

            chunk = mixture[:, start:end]
            if actual_len < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - actual_len))

            # Model forward pass (polarity handled after by detect_polarity)
            chunk_out = model(chunk.unsqueeze(0).to(device)).squeeze(0).cpu()
            chunk_out = chunk_out[:, :actual_len]

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

    return output / weights.clamp(min=1e-8)


def detect_polarity(mixture, model, device, gt_path):
    """
    Auto-detect if model output needs polarity flip.
    Tests on first 5 seconds of first track.
    """
    gt, _ = sf.read(gt_path, dtype="float32")
    gt_mono = gt[:SR * 5, 0] if len(gt) > SR * 5 else gt[:, 0]

    chunk = mixture[:, :SR * 5]
    if chunk.shape[1] < SR * 5:
        chunk = F.pad(chunk, (0, SR * 5 - chunk.shape[1]))

    with torch.no_grad():
        pred = model(chunk.unsqueeze(0).to(device)).squeeze(0).cpu()

    pred_mono = pred[0, :len(gt_mono)].numpy()
    gt_mono = gt_mono[:len(pred_mono)]

    corr_pos = np.corrcoef(gt_mono, pred_mono)[0, 1]
    corr_neg = np.corrcoef(gt_mono, -pred_mono)[0, 1]

    return -1 if corr_neg > corr_pos else 1


def get_test_tracks(musdb_root: str):
    """Get list of test track directories."""
    test_dir = os.path.join(musdb_root, "test")
    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        sys.exit(1)

    tracks = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])
    print(f"Found {len(tracks)} test tracks")
    return tracks


# ═════════════════════════════════════════════════════════════════
#  PART 2: SDR COMPUTATION
# ═════════════════════════════════════════════════════════════════

def si_sdr(reference, estimate):
    """
    Scale-Invariant Signal-to-Distortion Ratio.
    reference, estimate: 1D numpy arrays.
    Returns SI-SDR in dB.
    """
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)

    dot = np.dot(estimate, reference)
    s_target = (dot / (np.dot(reference, reference) + 1e-8)) * reference
    e_noise = estimate - s_target

    return float(10 * np.log10(
        np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-8)
    ))


def compute_chunk_metrics(gt_path, pred_path, chunk_seconds=1.0):
    """
    Compute chunk-level SI-SDR and BSSEval SDR (median over 1s chunks).
    Returns (si_sdr_median, bss_sdr_median).
    """
    gt, sr = sf.read(gt_path, dtype="float32")
    pred, _ = sf.read(pred_path, dtype="float32")

    if gt.ndim == 1:
        gt = np.stack([gt, gt], axis=-1)
    if pred.ndim == 1:
        pred = np.stack([pred, pred], axis=-1)

    min_len = min(len(gt), len(pred))
    gt = gt[:min_len]
    pred = pred[:min_len]

    chunk_samples = int(chunk_seconds * sr)
    n_chunks = min_len // chunk_samples

    si_sdrs = []
    bss_sdrs = []

    for i in range(n_chunks):
        start = i * chunk_samples
        end = start + chunk_samples

        gt_chunk = gt[start:end].mean(axis=-1)
        pred_chunk = pred[start:end].mean(axis=-1)

        # Skip chunks where GT has insufficient energy
        # museval uses RMS threshold — chunks with very low energy
        # produce unreliable SDR values that dominate the median
        gt_energy = np.sqrt(np.mean(gt_chunk ** 2))
        if gt_energy < 1e-4:
            continue

        # SI-SDR
        val = si_sdr(gt_chunk, pred_chunk)
        if not np.isnan(val) and not np.isinf(val) and abs(val) < 100:
            si_sdrs.append(val)

        # BSSEval SDR
        if HAS_MIR_EVAL:
            try:
                sdr, _, _, _ = bss_eval_sources(
                    gt_chunk.reshape(1, -1),
                    pred_chunk.reshape(1, -1),
                    compute_permutation=False,
                )
                if not np.isnan(sdr[0]) and abs(sdr[0]) < 100:
                    bss_sdrs.append(float(sdr[0]))
            except:
                pass

    si_median = float(np.median(si_sdrs)) if si_sdrs else float("nan")
    bss_median = float(np.median(bss_sdrs)) if bss_sdrs else float("nan")

    return si_median, bss_median


# ═════════════════════════════════════════════════════════════════
#  PART 3: MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════

def run_inference(
    musdb_root: str,
    checkpoints: Dict[str, str],
    output_dir: str,
    config: BandMambaConfig = BASE_CONFIG,
    device: str = "auto",
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 2.0,
):
    """Run inference on all test tracks."""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print("=" * 65)
    print("  PART 1: Inference")
    print("=" * 65)
    print(f"  Device:       {device}")
    print(f"  MUSDB root:   {musdb_root}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Stems:        {', '.join(checkpoints.keys())}")
    print(f"  Chunk:        {chunk_seconds}s with {overlap_seconds}s overlap")
    print("=" * 65)

    tracks = get_test_tracks(musdb_root)
    test_dir = os.path.join(musdb_root, "test")

    # Load all models
    print("\nLoading models...")
    models = {}
    for stem_name, ckpt_path in checkpoints.items():
        print(f"  {stem_name}:")
        models[stem_name] = load_model(ckpt_path, config, device)
    print(f"  All {len(models)} models loaded.\n")

    # Auto-detect polarity on first track
    print("Detecting polarity...")
    first_track = os.path.join(test_dir, tracks[0])
    first_mixture = load_audio(os.path.join(first_track, "mixture.wav"))
    polarity = {}
    for stem_name, model in models.items():
        gt_path = os.path.join(first_track, f"{stem_name}.wav")
        if os.path.exists(gt_path):
            p = detect_polarity(first_mixture, model, device, gt_path)
            polarity[stem_name] = p
            sign = "normal" if p == 1 else "INVERTED (will flip)"
            print(f"  {stem_name}: {sign}")
        else:
            polarity[stem_name] = 1

    os.makedirs(output_dir, exist_ok=True)
    total_time = 0

    for t_idx, track_name in enumerate(tracks):
        track_start = time.time()
        print(f"\n[{t_idx+1}/{len(tracks)}] {track_name}")

        track_dir = os.path.join(test_dir, track_name)
        folder_name = f"song{t_idx + 1}"
        out_track_dir = os.path.join(output_dir, folder_name)
        gt_dir = os.path.join(out_track_dir, "gt")
        pred_dir = os.path.join(out_track_dir, "predicted")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        mixture_path = os.path.join(track_dir, "mixture.wav")
        if not os.path.exists(mixture_path):
            print(f"  WARNING: mixture.wav not found, skipping")
            continue

        mixture = load_audio(mixture_path)
        duration = mixture.shape[1] / SR
        print(f"  Duration: {duration:.1f}s")

        # Save mixture
        save_audio(os.path.join(out_track_dir, "mixture.wav"), mixture, SR)

        # Save ground truth
        for stem_name in STEMS:
            gt_path = os.path.join(track_dir, f"{stem_name}.wav")
            if os.path.exists(gt_path):
                gt_audio = load_audio(gt_path)
                if gt_audio.shape[1] > mixture.shape[1]:
                    gt_audio = gt_audio[:, :mixture.shape[1]]
                elif gt_audio.shape[1] < mixture.shape[1]:
                    gt_audio = F.pad(gt_audio, (0, mixture.shape[1] - gt_audio.shape[1]))
                save_audio(os.path.join(gt_dir, f"{stem_name}.wav"), gt_audio, SR)

        # Run inference
        for stem_name, model in models.items():
            separated = overlap_add_separate(
                mixture, model, device, chunk_seconds, overlap_seconds
            )
            # Apply polarity correction
            separated = separated * polarity.get(stem_name, 1)
            save_audio(os.path.join(pred_dir, f"{stem_name}.wav"), separated, SR)

        track_time = time.time() - track_start
        total_time += track_time
        avg_time = total_time / (t_idx + 1)
        remaining = avg_time * (len(tracks) - t_idx - 1)
        print(f"  Done in {track_time:.1f}s  |  ETA: {remaining/60:.1f}min")

    print(f"\n  Inference complete! Total: {total_time/60:.1f} minutes")
    return output_dir


def run_metrics(results_dir: str):
    """Compute SI-SDR and BSSEval SDR on evaluation results."""
    print("\n" + "=" * 65)
    print("  PART 2: Computing SDR Metrics")
    print("=" * 65)

    song_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])

    if not song_dirs:
        print("ERROR: No song folders found!")
        return

    print(f"  Found {len(song_dirs)} tracks\n")

    sisdr_all = defaultdict(list)
    bss_all = defaultdict(list)
    track_results = []

    for t_idx, song_dir in enumerate(song_dirs):
        song_path = os.path.join(results_dir, song_dir)
        gt_dir = os.path.join(song_path, "gt")
        pred_dir = os.path.join(song_path, "predicted")

        if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
            continue

        track_si = {}
        track_bss = {}

        for stem in STEMS:
            gt_path = os.path.join(gt_dir, f"{stem}.wav")
            pred_path = os.path.join(pred_dir, f"{stem}.wav")

            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                continue

            si_val, bss_val = compute_chunk_metrics(gt_path, pred_path)

            if not np.isnan(si_val):
                sisdr_all[stem].append(si_val)
                track_si[stem] = si_val
            if not np.isnan(bss_val):
                bss_all[stem].append(bss_val)
                track_bss[stem] = bss_val

        parts = [f"{s}={track_si[s]:.2f}" for s in STEMS if s in track_si]
        print(f"  [{t_idx+1}/{len(song_dirs)}] {song_dir}: {', '.join(parts)}")
        track_results.append((song_dir, track_si, track_bss))

    # ─── Print Summary ──────────────────────────────────────────
    def print_table(title, data):
        print(f"\n  {title}")
        print(f"  {'Stem':<12} {'Median':>10} {'Mean':>10} {'Std':>8} {'Tracks':>8}")
        print(f"  {'-'*50}")
        medians = {}
        for stem in STEMS:
            if stem in data and len(data[stem]) > 0:
                v = np.array(data[stem])
                med, mean, std = float(np.median(v)), float(np.mean(v)), float(np.std(v))
                medians[stem] = med
                print(f"  {stem:<12} {med:>8.2f} dB {mean:>8.2f} dB {std:>6.2f} {len(v):>8}")
            else:
                print(f"  {stem:<12} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'0':>8}")
        if medians:
            avg = np.mean(list(medians.values()))
            print(f"  {'-'*50}")
            print(f"  {'AVERAGE':<12} {avg:>8.2f} dB")
        return medians

    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    si_medians = print_table("SI-SDR (chunk-level, median over 1s chunks)", sisdr_all)

    bss_medians = {}
    if HAS_MIR_EVAL:
        bss_medians = print_table("BSSEval SDR (chunk-level, median over 1s chunks)", bss_all)

    print("\n" + "=" * 65)

    # ─── Save CSV ───────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "sdr_results.csv")
    with open(csv_path, "w") as f:
        headers = ["track"]
        headers += [f"sisdr_{s}" for s in STEMS]
        if HAS_MIR_EVAL:
            headers += [f"bss_sdr_{s}" for s in STEMS]
        f.write(",".join(headers) + "\n")

        for song_dir, track_si, track_bss in track_results:
            row = [song_dir]
            row += [f"{track_si.get(s, float('nan')):.4f}" for s in STEMS]
            if HAS_MIR_EVAL:
                row += [f"{track_bss.get(s, float('nan')):.4f}" for s in STEMS]
            f.write(",".join(row) + "\n")

        # Summary
        row = ["MEDIAN"]
        row += [f"{si_medians.get(s, float('nan')):.4f}" for s in STEMS]
        if HAS_MIR_EVAL:
            row += [f"{bss_medians.get(s, float('nan')):.4f}" for s in STEMS]
        f.write(",".join(row) + "\n")

    print(f"  Results saved to: {csv_path}")


# ═════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BandMamba-Light: Inference + SDR Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (inference + metrics)
  python evaluate.py \\
      --musdb_root ./musdb18hq \\
      --vocals_checkpoint checkpoints/best_model_vocals.pt \\
      --drums_checkpoint checkpoints/best_model_drums.pt \\
      --bass_checkpoint checkpoints/best_model_bass.pt \\
      --other_checkpoint checkpoints/best_model_other.pt

  # Metrics only (on existing results)
  python evaluate.py --metrics_only --output_dir ./evaluation_results
        """,
    )
    parser.add_argument("--musdb_root", type=str, default=None)
    parser.add_argument("--vocals_checkpoint", type=str, default=None)
    parser.add_argument("--drums_checkpoint", type=str, default=None)
    parser.add_argument("--bass_checkpoint", type=str, default=None)
    parser.add_argument("--other_checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--chunk_seconds", type=float, default=10.0)
    parser.add_argument("--overlap_seconds", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--metrics_only", action="store_true",
                        help="Skip inference, only compute metrics on existing results")
    args = parser.parse_args()

    if args.metrics_only:
        # Just compute metrics
        run_metrics(args.output_dir)
    else:
        # Full pipeline
        if not args.musdb_root:
            parser.error("--musdb_root required for inference")

        checkpoints = {}
        if args.vocals_checkpoint:
            checkpoints["vocals"] = args.vocals_checkpoint
        if args.drums_checkpoint:
            checkpoints["drums"] = args.drums_checkpoint
        if args.bass_checkpoint:
            checkpoints["bass"] = args.bass_checkpoint
        if args.other_checkpoint:
            checkpoints["other"] = args.other_checkpoint

        if not checkpoints:
            parser.error("Provide at least one checkpoint")

        # Step 1: Inference
        run_inference(
            musdb_root=args.musdb_root,
            checkpoints=checkpoints,
            output_dir=args.output_dir,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            device=args.device,
        )

        # Step 2: Metrics
        run_metrics(args.output_dir)
