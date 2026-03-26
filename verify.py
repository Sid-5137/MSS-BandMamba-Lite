import soundfile as sf
import numpy as np

gt_vocals, _ = sf.read("evaluation_results_final/song1/gt/vocals.wav", dtype="float32")
pred_vocals, _ = sf.read(
    "evaluation_results_final/song1/predicted/vocals.wav", dtype="float32"
)
mix, _ = sf.read("evaluation_results_final/song1/mixture.wav", dtype="float32")

# Find where vocals are loud (skip intro)
sr = 44100
for start_sec in [0, 15, 30, 45, 60, 90, 120]:
    s = start_sec * sr
    e = s + 5 * sr
    if e > len(gt_vocals):
        break
    gt_energy = np.sqrt(np.mean(gt_vocals[s:e, 0] ** 2))
    pred_energy = np.sqrt(np.mean(pred_vocals[s:e, 0] ** 2))
    corr = np.corrcoef(gt_vocals[s:e, 0], pred_vocals[s:e, 0])[0, 1]
    print(
        f"  {start_sec:3d}s: GT_energy={gt_energy:.4f}  Pred_energy={pred_energy:.4f}  Corr={corr:.4f}"
    )
