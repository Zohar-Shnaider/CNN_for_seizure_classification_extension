# Acharya et al. (2018) Replication and Extension 

Exact reproduction of the 13-layer CNN for 3-class EEG classification
(normal / preictal / seizure) on the Bonn University database, in addition to two extensions.

## Expected results

The paper reports: **88.67% accuracy, 95.00% sensitivity, 90.00% specificity**.

## Data placement

The Bonn dataset comes as five sets of 100 `.txt` files each, one integer
sample per line (4097 samples = 23.6s at 173.61 Hz). This mirror uses the
original Andrzejak (2001) Z/O/N/F/S naming. The paper uses three of the five
sets:

- **Normal** = Set **O** (healthy volunteers, eyes closed) — paper calls this "Set B"
- **Preictal** = Set **F** (interictal, hippocampal formation) — paper calls this "Set D"
- **Seizure** = Set **S** (ictal / seizure activity) — paper calls this "Set E"

Expected structure (as produced by the mirror):

```
data/
  Set_O/   O001.txt ... O100.txt
  Set_F/   F001.txt ... F100.txt
  Set_S/   S001.txt ... S100.txt
```

(The other two sets, `Set_Z/` and `Set_N/`, are not used.) If your folders
are named differently, edit `SET_FOLDERS` in `config.py`.

## Replication Run

```bash
# 1. Load and normalize the .txt files -> data/processed/*.npy
python data_loading.py

# 2. Train with 10-fold CV (paper's exact hyperparameters)
python train.py

# 3. Plot confusion matrix and ROC curves
python plot_results.py
```

On GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Architecture (exact paper replication)

13 layers: 5 conv + 5 max-pool + 3 fully connected. Kernel sizes 6/5/4/4/4,
filter counts 4/4/10/10/15, LeakyReLU (α=0.01), softmax output.

## Hyperparameters (exact paper values)

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 1e-3 |
| L2 regularization (λ) | 0.7 |
| Momentum | 0.3 |
| Batch size | 3 |
| Epochs | 150 |
| Cross-validation | 10-fold (30% of train as validation) |

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings and hyperparameters |
| `data_loading.py` | Read Bonn `.txt` files, z-score normalize |
| `model.py` | 13-layer CNN |
| `train.py` | 10-fold CV training and evaluation |
| `plot_results.py` | Confusion matrix + ROC figures |


# Extensions to the Bonn Replication

Two extensions that interrogate what the CNN actually learns in the frequency
domain, building on the validated time-domain replication.

## Extension 1: Frequency-Domain Input

Tests whether the CNN's learned time-domain convolutions are an efficient
substitute for an explicit frequency transform.

```bash
# 1. Make sure the raw replication data exists
python data_loading.py

# 2. Generate frequency-domain representations
python freq_preprocessing.py

# 3. Train the SAME CNN on each representation
python train_ext1.py --repr raw     # time-domain baseline
python train_ext1.py --repr fft

# 4. Compare
python plot_ext1.py
```

- **FFT** — magnitude of the real FFT (log scale)


## Extension 2: Which Frequencies Carry the Signal?

Isolates each clinical EEG band via bandpass filtering, trains the same CNN
on each band separately, and measures accuracy per band. This localizes
*where* the seizure-discriminative information lives in the spectrum.

```bash
# 1. Generate band-isolated versions of the data
python band_preprocessing.py --band all

# 2. Train the CNN on each band
python train_band.py --band delta
python train_band.py --band theta
python train_band.py --band alpha
python train_band.py --band beta
python train_band.py --band gamma

# 3. Plot accuracy per band
python plot_bands.py
```

Bands: delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz),
gamma (30-70 Hz). Each band-isolated
segment is bandpass-filtered then z-score normalized, keeping the input
length at 4097 so the identical CNN architecture is used throughout.

## Files

| File | Purpose |
|------|---------|
| `freq_preprocessing.py` | Transform raw segments to frequency domain |
| `train_ext1.py` | Train CNN on any representation |
| `plot_ext1.py` | Compare representations |
| `band_preprocessing.py` | Bandpass-isolate each EEG band |
| `train_band.py` | Train CNN on a single isolated band |
| `plot_bands.py` | Compare accuracy across bands |
