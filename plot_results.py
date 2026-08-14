"""
Plot confusion matrix and ROC curves for the Bonn replication.

Usage:
    python plot_results.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import RESULTS_DIR, FIGURES_DIR, CLASS_NAMES, NUM_CLASSES


def main():
    results_path = RESULTS_DIR / "bonn_results.json"
    if not results_path.exists():
        print("No results found. Run train.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # === Confusion matrix ===
    cm = np.array(results["confusion_matrix"])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                    ha="center", va="center", color=color, fontsize=10)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Acc: {results['accuracy_mean']*100:.2f}%)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("Saved confusion_matrix.png")

    # === ROC curves ===
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    for i, name in enumerate(CLASS_NAMES):
        roc = results["roc"][name]
        ax.plot(roc["fpr"], roc["tpr"], color=colors[i], lw=2,
                label=f"{name} (AUC={roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (one-vs-rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print("Saved roc_curves.png")

    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
