"""
Extension: Which frequencies carry the seizure-discriminative signal?

Bandpass-filters each raw EEG segment to isolate a single clinical EEG band,
then saves the band-limited segments for training. Training the same CNN on
each isolated band reveals where the discriminative information lives.

Bands (Hz):
    delta: 0.5-4    theta: 4-8    alpha: 8-13
    beta:  13-30    gamma: 30-70

Usage:
    python band_preprocessing.py --band delta
    python band_preprocessing.py --band all      # generate every band at once
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

from config import SAMPLING_RATE, CLASS_NAMES

# Clinical EEG bands (Hz). Upper edge of gamma capped below Nyquist (~86.8 Hz).
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 70)
}


def bandpass(segment, fs, low, high, order=4):
    """Zero-phase Butterworth bandpass."""
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, segment)


def process_band(segments, band_name):
    """Bandpass every segment to the given band, then z-score normalize."""
    low, high = BANDS[band_name]
    out = []
    for seg in segments:
        filtered = bandpass(seg, SAMPLING_RATE, low, high)
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
        out.append(filtered.astype(np.float32))
    return np.array(out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", choices=list(BANDS.keys()) + ["all"],
                        required=True)
    args = parser.parse_args()

    raw_dir = Path("data/processed")
    segments = np.load(raw_dir / "segments.npy")
    labels = np.load(raw_dir / "labels.npy")
    print(f"Loaded {len(labels)} raw segments")

    bands_to_do = list(BANDS.keys()) if args.band == "all" else [args.band]

    for band_name in bands_to_do:
        low, high = BANDS[band_name]
        print(f"  Isolating {band_name} ({low}-{high} Hz)...")
        band_segments = process_band(segments, band_name)

        out_dir = Path(f"data/processed_band_{band_name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "segments.npy", band_segments)
        np.save(out_dir / "labels.npy", labels)
        print(f"    Saved to {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
