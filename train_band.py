"""
Train the CNN on a single isolated EEG band.

Same architecture, hyperparameters, and 10-fold CV as the replication.
Reads from a band-limited data directory produced by band_preprocessing.py.

Usage:
    python train_band.py --band delta
    python train_band.py --band theta
    ... etc
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from train import set_seed, train_model, evaluate, compute_fold_metrics
from config import (
    RESULTS_DIR, NUM_FOLDS, NUM_CLASSES, CLASS_NAMES, SEED, BATCH_SIZE
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset, Subset

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", choices=BANDS, required=True)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Band: {args.band}")

    data_dir = Path(f"data/processed_band_{args.band}")
    segments = np.load(data_dir / "segments.npy")
    labels = np.load(data_dir / "labels.npy")
    print(f"Loaded {len(labels)} segments")

    X = torch.from_numpy(segments).float().unsqueeze(1)
    y = torch.from_numpy(labels).long()
    dataset = TensorDataset(X, y)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_accs, fold_sens, fold_specs = [], [], []
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    all_labels, all_probs = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(segments)):
        print(f"\n  Fold {fold+1}/{NUM_FOLDS}")
        train_sub, val_idx = train_test_split(
            train_idx, test_size=0.3, random_state=SEED + fold
        )

        train_loader = DataLoader(Subset(dataset, train_sub),
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=BATCH_SIZE)

        model = train_model(train_loader, val_loader, device, len(train_sub))
        test_labels, test_preds, test_probs = evaluate(model, test_loader, device)

        acc = accuracy_score(test_labels, test_preds)
        sens, spec, cm = compute_fold_metrics(test_labels, test_preds)

        fold_accs.append(acc)
        fold_sens.append(sens)
        fold_specs.append(spec)
        total_cm += cm
        all_labels.extend(test_labels)
        all_probs.extend(test_probs)

        print(f"    acc={acc:.4f}")

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    roc_data = {}
    for c in range(NUM_CLASSES):
        binary = (all_labels == c).astype(int)
        fpr, tpr, _ = roc_curve(binary, all_probs[:, c])
        roc_data[CLASS_NAMES[c]] = {"auc": auc(fpr, tpr)}

    results = {
        "band": args.band,
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "sensitivity_mean": float(np.mean(fold_sens)),
        "specificity_mean": float(np.mean(fold_specs)),
        "fold_accuracies": fold_accs,
        "confusion_matrix": total_cm.tolist(),
        "roc": roc_data,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"band_{args.band}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  BAND: {args.band.upper()}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {results['accuracy_mean']*100:.2f}% "
          f"± {results['accuracy_std']*100:.2f}%")
    print(f"  Sensitivity: {results['sensitivity_mean']*100:.2f}%")
    print(f"  Specificity: {results['specificity_mean']*100:.2f}%")
    for c in CLASS_NAMES:
        print(f"  AUC {c}: {roc_data[c]['auc']:.4f}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
