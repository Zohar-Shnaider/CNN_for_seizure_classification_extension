"""
Compare Extension 1 representations against the raw baseline.

Usage:
    python plot_ext1.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, CLASS_NAMES

REPRS = ["raw", "fft"]
LABELS = {"raw": "Raw (time)", "fft": "FFT"}


def main():
    results = {}
    for r in REPRS:
        path = RESULTS_DIR / f"ext1_{r}_results.json"
        if r == "raw":
            # allow using the replication result as the raw baseline
            alt = RESULTS_DIR / "bonn_results.json"
            if not path.exists() and alt.exists():
                path = alt
        if path.exists():
            with open(path) as f:
                results[r] = json.load(f)

    if not results:
        print("No results found. Run train_ext1.py first.")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Bar chart of accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    reprs = [r for r in REPRS if r in results]
    accs = [results[r]["accuracy_mean"] * 100 for r in reprs]
    stds = [results[r]["accuracy_std"] * 100 for r in reprs]
    colors = ["#90A4AE", "#42A5F5"]

    bars = ax.bar([LABELS[r] for r in reprs], accs,
                  yerr=stds, capsize=5,
                  color=[colors[REPRS.index(r)] for r in reprs])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.1f}%", ha="center", fontsize=10)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Input Representation Comparison (10-fold CV)")
    ax.set_ylim(0, 105)
    ax.axhline(88.67, color="red", linestyle="--", lw=1, label="Paper (88.67%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ext1_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved ext1_comparison.png")

    # Print summary
    print(f"\n{'Representation':<15s} {'Accuracy':>18s}")
    print("-" * 35)
    for r in reprs:
        print(f"{LABELS[r]:<15s} {results[r]['accuracy_mean']*100:>7.2f}% "
              f"± {results[r]['accuracy_std']*100:.2f}%")


if __name__ == "__main__":
    main()
