"""
Reproduce Acharya et al. (2018) results on the Bonn dataset.

10-fold cross-validation with the paper's exact hyperparameters.

Usage:
    python train.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, f1_score
from pathlib import Path

from config import (
    RESULTS_DIR, FIGURES_DIR,
    LEARNING_RATE, LAMBDA_REG, MOMENTUM, BATCH_SIZE, NUM_EPOCHS, NUM_FOLDS,
    NUM_CLASSES, CLASS_NAMES, SEED
)
from model import EEG_CNN


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        # L2 regularization is applied via the optimizer's weight_decay,
        # matching the paper's weight-update equation exactly (see train_model).
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_model(train_loader, val_loader, device, train_set_size):
    model = EEG_CNN().to(device)
    criterion = nn.CrossEntropyLoss()

    # Paper's weight update:
    #   ΔW = -(x·λ/r)·W - (x/n)·∂C/∂W + m·ΔW_prev
    # The regularization term -(x·λ/r)·W is standard L2 weight decay with
    # coefficient λ/r, where r is the number of training samples.
    effective_weight_decay = LAMBDA_REG / train_set_size

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=effective_weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        model.eval()
        v_loss = 0.0
        n = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                v_loss += criterion(model(X_b), y_b).item()
                n += 1
        v_loss /= max(n, 1)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            print(f"      epoch {epoch+1}/{NUM_EPOCHS} "
                  f"train={train_loss:.4f} val={v_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, loader, device):
    model.eval()
    labels, preds, probs = [], [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b = X_b.to(device)
            p = torch.softmax(model(X_b), dim=1)
            labels.extend(y_b.numpy())
            preds.extend(p.argmax(dim=1).cpu().numpy())
            probs.extend(p.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def compute_fold_metrics(labels, preds):
    """Sensitivity and specificity per the paper's definitions."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    sens, spec = [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        fp = np.sum(cm[:, c]) - tp
        tn = np.sum(cm) - tp - fn - fp
        sens.append(tp / max(tp + fn, 1))
        spec.append(tn / max(tn + fp, 1))
    return np.mean(sens), np.mean(spec), cm


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path("data/processed")
    segments = np.load(data_dir / "segments.npy")
    labels = np.load(data_dir / "labels.npy")
    print(f"Loaded {len(labels)} segments")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]}: {np.sum(labels == c)}")

    X = torch.from_numpy(segments).float().unsqueeze(1)
    y = torch.from_numpy(labels).long()
    dataset = TensorDataset(X, y)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_accs, fold_sens, fold_specs = [], [], []
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    all_labels, all_probs = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(segments)):
        print(f"\n  Fold {fold+1}/{NUM_FOLDS}")

        # 30% of training as validation (paper)
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

        print(f"    acc={acc:.4f} sens={sens:.4f} spec={spec:.4f}")

    # ROC (one-vs-rest, macro)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    roc_data = {}
    for c in range(NUM_CLASSES):
        binary = (all_labels == c).astype(int)
        fpr, tpr, _ = roc_curve(binary, all_probs[:, c])
        roc_data[CLASS_NAMES[c]] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc(fpr, tpr)
        }

    # Results
    results = {
        "accuracy_mean": float(np.mean(fold_accs)),
        "accuracy_std": float(np.std(fold_accs)),
        "sensitivity_mean": float(np.mean(fold_sens)),
        "sensitivity_std": float(np.std(fold_sens)),
        "specificity_mean": float(np.mean(fold_specs)),
        "specificity_std": float(np.std(fold_specs)),
        "fold_accuracies": fold_accs,
        "confusion_matrix": total_cm.tolist(),
        "roc": roc_data,
    }

    with open(RESULTS_DIR / "bonn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*55}")
    print(f"  REPLICATION RESULTS (Bonn, 10-fold CV)")
    print(f"{'='*55}")
    print(f"  Accuracy:    {results['accuracy_mean']*100:.2f}% "
          f"± {results['accuracy_std']*100:.2f}%")
    print(f"  Sensitivity: {results['sensitivity_mean']*100:.2f}% "
          f"± {results['sensitivity_std']*100:.2f}%")
    print(f"  Specificity: {results['specificity_mean']*100:.2f}% "
          f"± {results['specificity_std']*100:.2f}%")
    print(f"\n  Paper reported: 88.67% / 95.00% / 90.00%")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "           " + "  ".join(f"{n[:4]:>6s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{total_cm[i,j]:6d}" for j in range(NUM_CLASSES))
        print(f"  {name:>8s}  {row}")

    print(f"\n  ROC AUC:")
    for c in CLASS_NAMES:
        print(f"    {c}: {roc_data[c]['auc']:.4f}")

    print(f"\nResults saved to {RESULTS_DIR / 'bonn_results.json'}")


if __name__ == "__main__":
    main()
