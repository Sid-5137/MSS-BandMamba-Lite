"""
Generate spectrogram & waveform comparison figures for the project report.
Processes multiple songs and organizes outputs per song.

Usage:
    # Auto-pick 5 best songs (most vocal energy)
    python generate_spectrograms.py --results_dir ./evaluation_results_final

    # Specify songs manually
    python generate_spectrograms.py --results_dir ./evaluation_results_final \
        --songs song7 song10 song22 song32 song47

Requires: pip install soundfile matplotlib numpy
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": "#334155",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "font.family": "sans-serif",
        "font.size": 10,
    }
)

STEMS = ["vocals", "drums", "bass", "other"]
SR = 44100


def compute_spectrogram(audio_path, start_sec, duration_sec, n_fft=4096, hop=1024):
    """Load audio segment and compute magnitude spectrogram in dB."""
    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)

    start = int(start_sec * sr)
    end = int((start_sec + duration_sec) * sr)
    if end > len(data):
        end = len(data)
        start = max(0, end - int(duration_sec * sr))

    segment = data[start:end]

    window = np.hanning(n_fft)
    n_frames = (len(segment) - n_fft) // hop + 1
    if n_frames <= 0:
        return np.zeros((n_fft // 2 + 1, 1)), sr

    spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        frame = segment[i * hop : i * hop + n_fft] * window
        fft_result = np.fft.rfft(frame)
        spec[:, i] = np.abs(fft_result)

    spec_db = 20 * np.log10(spec + 1e-8)
    return spec_db, sr


def find_best_start(song_dir, duration_sec=10):
    """Find the time offset with maximum vocal energy (skip silent intros)."""
    vocals_path = os.path.join(song_dir, "gt", "vocals.wav")
    if not os.path.exists(vocals_path):
        return 30  # default

    data, sr = sf.read(vocals_path, dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)

    chunk = int(duration_sec * sr)
    best_start = 0
    best_energy = 0

    for s in range(0, len(data) - chunk, int(5 * sr)):  # step 5 seconds
        segment = data[s : s + chunk]
        energy = np.sqrt(np.mean(segment**2))
        if energy > best_energy:
            best_energy = energy
            best_start = s

    return best_start / sr


def plot_stem_comparison(song_dir, stem, output_path, start_sec, duration_sec=10):
    """Plot Mixture | GT | Predicted spectrograms for one stem."""
    mixture_path = os.path.join(song_dir, "mixture.wav")
    gt_path = os.path.join(song_dir, "gt", f"{stem}.wav")
    pred_path = os.path.join(song_dir, "predicted", f"{stem}.wav")

    if not all(os.path.exists(p) for p in [mixture_path, gt_path, pred_path]):
        return

    mix_spec, sr = compute_spectrogram(mixture_path, start_sec, duration_sec)
    gt_spec, _ = compute_spectrogram(gt_path, start_sec, duration_sec)
    pred_spec, _ = compute_spectrogram(pred_path, start_sec, duration_sec)

    vmin = min(mix_spec.min(), gt_spec.min(), pred_spec.min())
    vmax = max(mix_spec.max(), gt_spec.max(), pred_spec.max())
    vmin = max(vmin, vmax - 80)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    titles = [
        "Mixture",
        f"Ground Truth ({stem.capitalize()})",
        f"Predicted ({stem.capitalize()})",
    ]

    for ax, spec, title in zip(axes, [mix_spec, gt_spec, pred_spec], titles):
        max_bin = int(16000 / (sr / 2) * spec.shape[0])
        im = ax.imshow(
            spec[:max_bin, :],
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            extent=[0, duration_sec, 0, 16],
        )
        ax.set_title(title, fontsize=12, fontweight="bold", color="#1E293B")
        ax.set_xlabel("Time (s)", fontsize=10)

    axes[0].set_ylabel("Frequency (kHz)", fontsize=10)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax).set_label("Magnitude (dB)", fontsize=9)

    fig.suptitle(
        f"Spectrogram Comparison — {stem.capitalize()}",
        fontsize=14,
        fontweight="bold",
        color="#1A1F36",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_all_stems_grid(song_dir, output_path, start_sec, duration_sec=10):
    """4x3 grid: rows=stems, columns=mixture|GT|predicted."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharey=True)

    for row, stem in enumerate(STEMS):
        mixture_path = os.path.join(song_dir, "mixture.wav")
        gt_path = os.path.join(song_dir, "gt", f"{stem}.wav")
        pred_path = os.path.join(song_dir, "predicted", f"{stem}.wav")

        if not all(os.path.exists(p) for p in [mixture_path, gt_path, pred_path]):
            continue

        mix_spec, sr = compute_spectrogram(mixture_path, start_sec, duration_sec)
        gt_spec, _ = compute_spectrogram(gt_path, start_sec, duration_sec)
        pred_spec, _ = compute_spectrogram(pred_path, start_sec, duration_sec)

        vmin = min(mix_spec.min(), gt_spec.min(), pred_spec.min())
        vmax = max(mix_spec.max(), gt_spec.max(), pred_spec.max())
        vmin = max(vmin, vmax - 80)
        max_bin = int(16000 / (sr / 2) * mix_spec.shape[0])

        for col, (spec, _) in enumerate(
            zip([mix_spec, gt_spec, pred_spec], ["Mixture", f"GT", f"Predicted"])
        ):
            im = axes[row, col].imshow(
                spec[:max_bin, :],
                aspect="auto",
                origin="lower",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration_sec, 0, 16],
            )
            if row == 0:
                axes[row, col].set_title(
                    ["Mixture", "Ground Truth", "Predicted"][col],
                    fontsize=13,
                    fontweight="bold",
                    color="#1E293B",
                )
            if row == 3:
                axes[row, col].set_xlabel("Time (s)", fontsize=10)
            if col == 0:
                axes[row, col].set_ylabel(
                    f"{stem.capitalize()}\nFreq (kHz)", fontsize=10
                )

    fig.suptitle(
        "BandMamba-Light — Separation Results",
        fontsize=16,
        fontweight="bold",
        color="#1A1F36",
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_waveform_comparison(song_dir, stem, output_path, start_sec, duration_sec=5):
    """Plot GT vs Predicted waveforms + residual for one stem."""
    gt_path = os.path.join(song_dir, "gt", f"{stem}.wav")
    pred_path = os.path.join(song_dir, "predicted", f"{stem}.wav")

    if not all(os.path.exists(p) for p in [gt_path, pred_path]):
        return

    gt, sr = sf.read(gt_path, dtype="float32")
    pred, _ = sf.read(pred_path, dtype="float32")
    if gt.ndim == 2:
        gt = gt.mean(axis=1)
    if pred.ndim == 2:
        pred = pred.mean(axis=1)

    start = int(start_sec * sr)
    end = int((start_sec + duration_sec) * sr)
    gt_seg = gt[start:end]
    pred_seg = pred[start:end]
    min_len = min(len(gt_seg), len(pred_seg))
    t = np.linspace(start_sec, start_sec + duration_sec, min_len)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

    ax1.plot(t, gt_seg[:min_len], color="#22C55E", linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Ground Truth — {stem.capitalize()}", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, pred_seg[:min_len], color="#7C5CFC", linewidth=0.5, alpha=0.8)
    ax2.set_ylabel("Amplitude")
    ax2.set_title(f"Predicted — {stem.capitalize()}", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(True, alpha=0.3)

    diff = gt_seg[:min_len] - pred_seg[:min_len]
    ax3.plot(t, diff, color="#EF4444", linewidth=0.5, alpha=0.8)
    ax3.set_ylabel("Amplitude")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Residual Error", fontsize=11, fontweight="bold")
    ax3.set_ylim(-0.5, 0.5)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Waveform Comparison — {stem.capitalize()}",
        fontsize=14,
        fontweight="bold",
        color="#1A1F36",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def auto_select_songs(results_dir, n=5):
    """Pick the N songs with highest vocal energy (most interesting spectrograms)."""
    song_dirs = sorted(
        [
            d
            for d in os.listdir(results_dir)
            if os.path.isdir(os.path.join(results_dir, d))
        ]
    )

    energies = []
    for song in song_dirs:
        vocals_path = os.path.join(results_dir, song, "gt", "vocals.wav")
        if not os.path.exists(vocals_path):
            continue
        data, sr = sf.read(vocals_path, dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        energy = np.sqrt(np.mean(data**2))
        energies.append((song, energy))

    energies.sort(key=lambda x: -x[1])
    selected = [s[0] for s in energies[:n]]
    print(f"Auto-selected {len(selected)} songs with highest vocal energy:")
    for s, e in energies[:n]:
        print(f"  {s}: RMS energy = {e:.4f}")
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectrogram figures for report"
    )
    parser.add_argument("--results_dir", type=str, default="./evaluation_results")
    parser.add_argument(
        "--songs",
        nargs="+",
        default=None,
        help="Song folder names (e.g., song7 song10 song22). Auto-selects 5 if not specified.",
    )
    parser.add_argument("--output_dir", type=str, default="./figures")
    parser.add_argument(
        "--duration", type=float, default=10, help="Duration in seconds"
    )
    args = parser.parse_args()

    if args.songs is None:
        songs = auto_select_songs(args.results_dir, n=5)
    else:
        songs = args.songs

    os.makedirs(args.output_dir, exist_ok=True)

    for song in songs:
        song_dir = os.path.join(args.results_dir, song)
        if not os.path.isdir(song_dir):
            print(f"WARNING: {song_dir} not found, skipping")
            continue

        # Create per-song output folder
        song_out = os.path.join(args.output_dir, song)
        os.makedirs(song_out, exist_ok=True)

        # Auto-find best start time (where vocals are loudest)
        start_sec = find_best_start(song_dir, args.duration)
        print(f"\n{'='*50}")
        print(f"  {song} — best segment starts at {start_sec:.1f}s")
        print(f"{'='*50}")

        # 1. Individual stem spectrograms
        for stem in STEMS:
            plot_stem_comparison(
                song_dir,
                stem,
                os.path.join(song_out, f"spec_{stem}.png"),
                start_sec,
                args.duration,
            )
            print(f"  spec_{stem}.png")

        # 2. All stems grid
        plot_all_stems_grid(
            song_dir,
            os.path.join(song_out, f"spec_all_stems_grid.png"),
            start_sec,
            args.duration,
        )
        print(f"  spec_all_stems_grid.png")

        # 3. Waveform comparisons
        for stem in STEMS:
            plot_waveform_comparison(
                song_dir,
                stem,
                os.path.join(song_out, f"waveform_{stem}.png"),
                start_sec,
                min(args.duration, 5),
            )
            print(f"  waveform_{stem}.png")

    # Summary
    print(f"\n{'='*50}")
    print(f"  All figures saved to: {args.output_dir}/")
    print(f"{'='*50}")
    print(f"\nFolder structure:")
    print(f"  {args.output_dir}/")
    for song in songs:
        print(f"  ├── {song}/")
        print(f"  │   ├── spec_vocals.png")
        print(f"  │   ├── spec_drums.png")
        print(f"  │   ├── spec_bass.png")
        print(f"  │   ├── spec_other.png")
        print(f"  │   ├── spec_all_stems_grid.png   ← best for report")
        print(f"  │   ├── waveform_vocals.png")
        print(f"  │   ├── waveform_drums.png")
        print(f"  │   ├── waveform_bass.png")
        print(f"  │   └── waveform_other.png")

    print(f"\nFor the report, use the spec_all_stems_grid.png from each song.")
    print(f"Pick the 2-3 that look best visually.")


if __name__ == "__main__":
    main()
