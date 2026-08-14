"""
Configuration for exact reproduction of Acharya et al. (2018)
on the Bonn University EEG database.
"""

from pathlib import Path

# === Paths ===
# Point DATA_DIR at the folder containing the Bonn set subfolders.
# Expected structure (folder names can be adjusted in SET_FOLDERS below):
#   data/setB/  (normal,   files O001.txt ... O100.txt)
#   data/setD/  (preictal, files F001.txt ... F100.txt)
#   data/setE/  (seizure,  files S001.txt ... S100.txt)
DATA_DIR = Path("data")

# Map each class to its folder name and expected file prefix.
# The paper uses Set B (normal), Set D (preictal), Set E (seizure), but the
# original Andrzejak (2001) naming — used by this dataset mirror — is Z/O/N/F/S:
#   Set Z = healthy, eyes open        Set O = healthy, eyes closed   (paper's "B")
#   Set N = interictal, hippocampal   Set F = interictal, epileptog. (paper's "D")
#   Set S = ictal / seizure                                          (paper's "E")
SET_FOLDERS = {
    "Normal":   {"folder": "Set_O", "prefixes": ["O"]},
    "Preictal": {"folder": "Set_F", "prefixes": ["F"]},
    "Seizure":  {"folder": "Set_S", "prefixes": ["S"]},
}

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")

# === Signal properties (fixed by the Bonn dataset) ===
SEGMENT_LENGTH = 4097      # samples per file
SAMPLING_RATE = 173.61     # Hz

# === CNN architecture ===
# (type, out_channels, kernel_size, stride)
CNN_LAYERS = [
    ("conv",    4,  6, 1),
    ("pool",    4,  2, 2),
    ("conv",    4,  5, 1),
    ("pool",    4,  2, 2),
    ("conv",   10,  4, 1),
    ("pool",   10,  2, 2),
    ("conv",   10,  4, 1),
    ("pool",   10,  2, 2),
    ("conv",   15,  4, 1),
    ("pool",   15,  2, 2),
]
FC_SIZES = [50, 20, 3]
LEAKY_RELU_SLOPE = 0.01

# === Training (exact paper values) ===
LEARNING_RATE = 1e-3
LAMBDA_REG = 0.7       # L2 regularization
MOMENTUM = 0.3
BATCH_SIZE = 3
NUM_EPOCHS = 150
NUM_FOLDS = 10

# === Classes ===
CLASS_NAMES = ["Normal", "Preictal", "Seizure"]
NUM_CLASSES = 3

# === Random seed ===
SEED = 42
