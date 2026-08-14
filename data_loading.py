"""
Load the Bonn University EEG dataset from .txt files.

Each .txt file contains SEGMENT_LENGTH integer samples, one per line,
representing a single 23.6-second single-channel EEG segment.

Usage:
    python data_loading.py    # loads, z-score normalizes, saves .npy
"""

import numpy as np
from pathlib import Path

from config import (
    DATA_DIR, SET_FOLDERS, SEGMENT_LENGTH, CLASS_NAMES, SEED
)


def read_txt_segment(filepath):
    """Read one Bonn .txt file into a 1D numpy array."""
    values = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(float(line))
    return np.array(values, dtype=np.float32)


def find_class_files(class_name):
    """Locate all .txt files for a given class, trying several folder/prefix options."""
    spec = SET_FOLDERS[class_name]
    folder_candidates = [
        DATA_DIR / spec["folder"],
        DATA_DIR / spec["folder"].upper(),
        DATA_DIR / spec["folder"].capitalize(),
        DATA_DIR / class_name,
        DATA_DIR / class_name.lower(),
    ]

    # Also try prefix-named folders (e.g. a folder literally called "F")
    for prefix in spec["prefixes"]:
        folder_candidates.append(DATA_DIR / prefix)
        folder_candidates.append(DATA_DIR / prefix.upper())

    for folder in folder_candidates:
        if folder.exists() and folder.is_dir():
            files = sorted(folder.glob("*.txt"))
            if files:
                return files

    # Last resort: search recursively for files matching the prefixes
    all_txt = list(DATA_DIR.rglob("*.txt"))
    for prefix in spec["prefixes"]:
        matched = sorted([f for f in all_txt if f.name.upper().startswith(prefix.upper())])
        if matched:
            return matched

    return []


def load_bonn():
    """Load all three classes, return (segments, labels)."""
    all_segments = []
    all_labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        files = find_class_files(class_name)
        if not files:
            print(f"  WARNING: no files found for {class_name} "
                  f"(folder '{SET_FOLDERS[class_name]['folder']}', "
                  f"prefixes {SET_FOLDERS[class_name]['prefixes']})")
            continue

        print(f"  {class_name}: {len(files)} files from {files[0].parent}")

        for filepath in files:
            seg = read_txt_segment(filepath)

            if len(seg) != SEGMENT_LENGTH:
                print(f"    WARNING: {filepath.name} has {len(seg)} samples "
                      f"(expected {SEGMENT_LENGTH}), skipping")
                continue

            # Z-score normalization (paper's only preprocessing)
            seg = (seg - np.mean(seg)) / (np.std(seg) + 1e-8)

            all_segments.append(seg)
            all_labels.append(label)

    segments = np.array(all_segments, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)
    return segments, labels


def main():
    print("Loading Bonn EEG dataset...")
    segments, labels = load_bonn()

    if len(labels) == 0:
        print("\nNo data loaded. Check DATA_DIR and folder structure in config.py.")
        print(f"DATA_DIR is currently: {DATA_DIR.resolve()}")
        return

    print(f"\nTotal segments: {len(labels)}")
    for label, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {np.sum(labels == label)}")

    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "segments.npy", segments)
    np.save(out_dir / "labels.npy", labels)
    print(f"\nSaved to {out_dir}/segments.npy and labels.npy")


if __name__ == "__main__":
    main()
