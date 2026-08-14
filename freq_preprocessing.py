"""
Extension 1: Frequency-domain input representations.

Transforms the raw Bonn EEG segments into frequency-domain representations
and saves them for training the same CNN architecture.

The representation that is produced:
    - fft:    Magnitude of the real FFT, 1D input

Usage:
    python freq_preprocessing.py
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import signal as sp_signal

from config import SAMPLING_RATE, SEGMENT_LENGTH, CLASS_NAMES


def to_fft(segment, target_len):
    """Magnitude of the real FFT, resampled to target_len points."""
    mag = np.abs(np.fft.fft(segment))
    mag = np.log(mag + 1e-12)
    mag_resampled = np.interp(
        np.linspace(0, len(mag) - 1, target_len),
        np.arange(len(mag)),
        mag,
    )

    return mag_resampled.astype(np.float32)

def transform_all(segments):
    """Apply the transform to all segments."""
    out = []
    for seg in segments:
        f = to_fft(seg, SEGMENT_LENGTH)
        # Z-score normalize the frequency representation
        f = (f - np.mean(f)) / (np.std(f) + 1e-8)
        out.append(f)
    return np.array(out, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Load the raw (time-domain) processed data produced by data_loading.py
    raw_dir = Path("data/processed")
    segments = np.load(raw_dir / "segments.npy")
    labels = np.load(raw_dir / "labels.npy")
    print(f"Loaded {len(labels)} raw segments")

    freq_segments = transform_all(segments)
    print(f"Output shape: {freq_segments.shape}")

    out_dir = Path("data/processed_fft")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "segments.npy", freq_segments)
    np.save(out_dir / "labels.npy", labels)
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()
