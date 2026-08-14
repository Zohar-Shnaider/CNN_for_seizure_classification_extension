"""
Plot accuracy per isolated EEG band.

Usage:
    python plot_bands.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, CLASS_NAMES

# Ordered low to high frequency; broadband and raw as references
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_RANGES = {
    "delta": "0.5-4 Hz", "theta": "4-8 Hz", "alpha": "8-13 Hz",
    "beta": "13-30 Hz", "gamma": "30-70 Hz"}


def main():
    results = {}
    for band in BAND_ORDER:
        path = RESULTS_DIR / f"band_{band}_results.json"
        if path.exists():
            with open(path) as f:
                results[band] = json.load(f)

    # Optionally include the raw replication as reference
    raw_path = RESULTS_DIR / "bonn_results.json"
    raw_acc = None
    if raw_path.exists():
        with open(raw_path) as f:
            raw_acc = json.load(f)["accuracy_mean"] * 100

    if not results:
        print("No band results found. Run train_band.py first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    bands = [b for b in BAND_ORDER if b in results]
    accs = [results[b]["accuracy_mean"] * 100 for b in bands]
    stds = [results[b]["accuracy_std"] * 100 for b in bands]

    # Color the single bands differently from broadband
    colors = []
    for b in bands:
        if b == "broadband":
            colors.append("#455A64")
        else:
            colors.append("#42A5F5")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    xlabels = [f"{b.capitalize()}\n{BAND_RANGES[b]}" for b in bands]
    bars = ax.bar(xlabels, accs, yerr=stds, capsize=5, color=colors)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f"{acc:.1f}%", ha="center", fontsize=10, fontweight="bold")

    if raw_acc:
        ax.axhline(raw_acc, color="red", linestyle="--", lw=1,
                   label=f"Raw full signal ({raw_acc:.1f}%)")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy by Isolated EEG Band")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "band_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved band_comparison.png")

    # Summary table
    print(f"\n{'Band':<14s} {'Range':<12s} {'Accuracy':>18s}")
    print("-" * 46)
    for b in bands:
        print(f"{b.capitalize():<14s} {BAND_RANGES[b]:<12s} "
              f"{results[b]['accuracy_mean']*100:>7.2f}% "
              f"± {results[b]['accuracy_std']*100:.2f}%")

    # Identify the most informative single band
    single_bands = [b for b in bands if b != "broadband"]
    if single_bands:
        best = max(single_bands, key=lambda b: results[b]["accuracy_mean"])
        worst = min(single_bands, key=lambda b: results[b]["accuracy_mean"])
        print(f"\nMost informative single band: {best.capitalize()} "
              f"({results[best]['accuracy_mean']*100:.1f}%)")
        print(f"Least informative single band: {worst.capitalize()} "
              f"({results[worst]['accuracy_mean']*100:.1f}%)")


if __name__ == "__main__":
    main()
