# Handwriten-Text-Recognition
A deep learning system for recognizing handwritten words in images. The project compares classical (SVM + HOG) and deep learning (CRNN) approaches to optical character recognition. The model combines a ResNet-inspired convolutional backbone with a bidirectional GRU and CTC loss — a CRNN architecture adapted for CPU training.

---

## Data

The dataset used can be found here: https://www.kaggle.com/datasets/ssarkar445/handwriting-recognitionocr
---

## Architecture

```
Input Image (1 × 64 × 256)
        │
        ▼
Conv + BN + ReLU + MaxPool
        │
  ResidualBlock(32 → 64)  + MaxPool
        │
  ResidualBlock(64 → 128) + MaxPool (height only)
        │
  ResidualBlock(128 → 256)+ MaxPool (height only)
        │
  Conv(256 → 512) + BN + ReLU
        │
  AdaptiveAvgPool → (Batch, TimeSteps, 512)
        │
  Linear(512 → 256)
        │
  BiGRU(256, num_layers=2) → (Batch, TimeSteps, 512)
        │
  Linear(512 → num_classes)
        │
  CTC Loss / Greedy Decode
```

- **Feature extraction**: ResNet-style residual blocks with asymmetric max-pooling to preserve the horizontal timeline width
- **Sequence modeling**: 2-layer bidirectional GRU
- **Decoding**: CTC greedy decode with blank token filtering and repetition collapse
- **Input**: grayscale images resized to 256 × 64 with CLAHE contrast enhancement

---

## Project Structure

```
├── NN.py          # Full training pipeline (dataset, model, training loop, evaluation)
├── ML.py          # CUDA availability check / environment probe
├── ImRec.pptx     # Project presentation
└── README.md
```

---

## Dataset

The pipeline expects a Kaggle-style OCR archive with the following layout:

```
archive/
├── *.csv           # Index file(s) with columns: FILENAME, IDENTITY
└── **/*.jpg/png    # Word-crop images (any subdirectory depth)
```

The CSV `IDENTITY` column holds the ground-truth transcription. Only alphanumeric characters and spaces are kept (`A–Z`, `0–9`, ` `). Labels marked `UNREADABLE`, `NAN`, or `N/A` are dropped.

Update the dataset path at the top of `__main__` in `NN.py`:

```python
path = r"C:\path\to\your\archive"
```

---

## Preprocessing

| Step | Detail |
|---|---|
| Grayscale read | `cv2.IMREAD_GRAYSCALE` |
| Contrast enhancement | CLAHE (`clipLimit=2.0`, `tileGridSize=8×8`) |
| Resize | 256 × 64, bicubic interpolation |
| Normalization | `[0, 1]` float32 |
| Augmentation (train only) | Gaussian noise (`σ=0.02`, applied with 30% probability) |

Labels longer than 24 characters are removed to stay within the CTC timeline budget (~63 slots after downsampling).

---

## Training

The dataset is capped at **12,000 samples** for CPU efficiency, then split 85/15 into train and validation sets.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=1e-3`) |
| LR scheduler | Cosine annealing (`T_max=10`, `η_min=1e-5`) |
| Batch size | 64 |
| Epochs | 30 |
| Gradient clipping | `max_norm=1.0` |
| Device | CPU (auto-detected) |

---

## Evaluation Metrics

After training, the model is evaluated on the validation split:

- **CER** — Character Error Rate (via `torchmetrics`)
- **WER** — Word Error Rate (via `torchmetrics`)
- **Word Recognition Accuracy** — `1 - WER`
- **Perfect Match Accuracy** — exact string equality rate

A loss curve plot (train vs. validation) is displayed at the end of training.

---

## Requirements

```
torch
torchvision
torchmetrics
opencv-python
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install torch torchvision torchmetrics opencv-python numpy pandas matplotlib scikit-learn
```

> **GPU note:** The training loop is pinned to CPU (`torch.device("cpu")`). To use a GPU, change that line and verify your CUDA setup with `ML.py`:
> ```bash
> python ML.py
> ```

---

## Running

```bash
python NN.py
```

Progress is printed every 20 steps. After all epochs, evaluation results are printed and the loss curve is shown.
