import os
import glob
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Advanced Evaluation Metrics
from torchmetrics.text import CharErrorRate, WordErrorRate


# ------------------------------------------------------------------
# 1. DATASET DEFINITION & RIGOROUS PREPROCESSING
# ------------------------------------------------------------------
class RobustOCRDataset(Dataset):
    def __init__(self, df, char_to_num, img_width=256, img_height=64, Phase="train"):
        self.df = df
        self.char_to_num = char_to_num
        self.img_width = img_width
        self.img_height = img_height
        self.phase = Phase

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['IMAGE_PATH']
        label_str = row['IDENTITY']

        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Fallback for unreadable files
        if img is None:
            img = np.ones((self.img_height, self.img_width), dtype=np.float32) * 255.0

        # Accuracy Booster: Contrast Enhancement (CLAHE)
        # Fixes faded handwriting lines before resizing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        # High-quality interpolation resizing
        img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_CUBIC)

        # Normalize image intensities to [0, 1] range
        img = img.astype(np.float32) / 255.0

        # Standard Data Augmentation for training phase (Improves Generalization Accuracy)
        if self.phase == "train":
            # Add subtle random pixel noise
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.02, img.shape)
                img = np.clip(img + noise, 0.0, 1.0)

        # Convert to Channel-First Tensor shape: (1, Height, Width)
        img = np.expand_dims(img, axis=0)

        # Map string characters to numbers
        label = [self.char_to_num[char] for char in label_str if char in self.char_to_num]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long), label_str


def collate_fn(batch):
    imgs, labels, label_strs = zip(*batch)
    imgs = torch.stack(imgs, 0)

    label_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    labels_flattened = torch.cat(labels)

    return imgs, labels_flattened, label_lengths, label_strs


# ------------------------------------------------------------------
# 2. HIGH-ACCURACY RESNET + BGRU ARCHITECTURE (CRNN)
# ------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Residual connection block to retain spatial handwriting features without degradation."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ResNetBGRU(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(ResNetBGRU, self).__init__()

        # Feature Extraction Backbone (ResNet-inspired)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 256x64 -> 128x32

            ResidualBlock(32, 64, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128x32 -> 64x16

            ResidualBlock(64, 128, stride=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 64x16 -> 64x8 (Preserves timeline width)

            ResidualBlock(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 64x8 -> 64x4

            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0, bias=False),  # Height pool alternative
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)  # Yields final width dimension steps ~ 63
        )

        # Adaptive pooling strictly ensures height is compressed to 1 regardless of inputs
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))

        # Linear layer mapping features to recurrent network dimensions
        self.linear = nn.Linear(512, hidden_size)

        # Recurrent Sequence modeling layers (Using GRU over LSTM for faster CPU iteration)
        self.gru = nn.Sequential(
            nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True, num_layers=2)
        )

        # Final fully-connected classification head
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 1. Spatial convolutions
        features = self.feature_extractor(x)
        features = self.adaptive_pool(features)

        # 2. Reshape tensors cleanly for sequence processing
        features = features.squeeze(2)  # Remove explicit vertical channel dimension
        features = features.permute(0, 2, 1)  # Format to (Batch, TimeSteps, Channels)

        # 3. Recurrent processing
        rnn_input = self.linear(features)
        rnn_output, _ = self.gru(rnn_input)

        # 4. Generate prediction classifications
        logits = self.fc(rnn_output)
        return logits.permute(1, 0, 2)  # Return standard CTC representation: (TimeSteps, Batch, Classes)


# ------------------------------------------------------------------
# 3. CTC DECODING LOGIC
# ------------------------------------------------------------------
def ctc_decode(log_probs, num_to_char):
    arg_maxes = torch.argmax(log_probs, dim=2).permute(1, 0).cpu().numpy()
    decodes = []
    for sequence in arg_maxes:
        decode = []
        prev_char = None
        for idx in sequence:
            if idx != 0:  # Filter out the CTC blank placeholder token
                if idx != prev_char:  # Compress repetitions
                    decode.append(num_to_char[idx])
            prev_char = idx
        decodes.append("".join(decode))
    return decodes


# ------------------------------------------------------------------
# MAIN EXECUTION LOOP BLOCK (Windows Multiprocessing Protection)
# ------------------------------------------------------------------
if __name__ == '__main__':

    # --- STEP 1: PARSE LOCAL DATA PATHS ---
    path = r"C:\Users\vladg\Downloads\archive"
    print(f"Targeting local dataset path: {path}")

    csv_files = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"Could not find any CSV data indexes inside: {path}")

    dfs = []
    for csv_f in csv_files:
        dfs.append(pd.read_csv(csv_f))
    df = pd.concat(dfs, ignore_index=True)

    df.columns = [col.upper() for col in df.columns]
    print(f"Initial raw dataset footprint: {len(df)} rows.")

    # --- STEP 2: AGGRESIVE DATA QUALITY STRIPPING (CRITICAL ACCURACY FIX) ---
    df = df.dropna(subset=['IDENTITY'])
    df['IDENTITY'] = df['IDENTITY'].astype(str).str.strip().str.upper()

    # Strip bad texts completely so they never poison our network loss curves
    df = df[~df['IDENTITY'].isin(['UNREADABLE', 'NAN', 'N/A', '', ' '])]

    # Accuracy Protection: Remove words containing weird punctuation characters
    # This prevents the model from wasting capacity trying to learn chaotic noise
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
    df = df[df['IDENTITY'].apply(lambda x: all(c in allowed_chars for c in x))]

    # --- STEP 3: HIGH-SPEED INDEXING SYSTEM ---
    print("Executing fast parallel directory indexing mapping...")
    start_time = time.time()
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))

    image_lookup = {os.path.basename(p): p for p in all_image_paths}
    print(f"Successfully mapped {len(image_lookup)} disk paths in {time.time() - start_time:.2f}s.")

    df['BASE_NAME'] = df['FILENAME'].apply(os.path.basename)
    df['IMAGE_PATH'] = df['BASE_NAME'].map(image_lookup)
    df = df.drop(columns=['BASE_NAME']).dropna(subset=['IMAGE_PATH']).reset_index(drop=True)

    # --- STEP 4: STRICT MATHEMATICAL BOUNDS FILTERING ---
    # Our model shrinks 256 horizontal pixels down to 63 sequence timeline slots.
    # To maintain exceptional accuracy, we remove text strings that exceed 24 tokens.
    # This leaves a safe, roomy timeline buffer for CTC spacing.
    df = df[df['IDENTITY'].str.len() <= 24]
    df = df[df['IDENTITY'].str.len() > 0]
    df = df.reset_index(drop=True)

    # --- STEP 5: CAP DATASET FOR OPTIMIZED CPU PROCESSING ---
    print("CPU Execution Mode Configured: Downsampling dataset.")
    df = df.sample(n=12000, random_state=42).reset_index(drop=True)
    print(f"Optimized clean subset size: {len(df)} rows.")

    # --- STEP 6: VOCABULARY GENERATION ---
    characters = sorted(list(set("".join(df['IDENTITY'].values))))
    char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}
    num_to_char = {idx + 1: char for idx, char in enumerate(characters)}
    char_to_num['-'] = 0  # Standard CTC blank token allocation
    num_to_char[0] = '-'

    NUM_CLASSES = len(char_to_num)

    # Train-Validation Partitioning
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    train_dataset = RobustOCRDataset(train_df, char_to_num, Phase="train")
    val_dataset = RobustOCRDataset(val_df, char_to_num, Phase="val")

    num_cpus = min(4, os.cpu_count() or 1)
    print(f"Allocating {num_cpus} multi-threaded CPU parallel processing workers...")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, drop_last=True,
                              num_workers=num_cpus)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=num_cpus)

    # --- STEP 7: OPTIMIZATION & INITIALIZATION ---
    device = torch.device("cpu")
    model = ResNetBGRU(num_classes=NUM_CLASSES).to(device)

    # Using strict standard CTC criteria
    criterion = nn.CTCLoss(blank=0, zero_infinity=False)

    # AdamW with robust L2 regularization to stop overfitting on smaller datasets
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # Dynamic Cosine Annealing learning rate policy (High accuracy industry benchmark)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    NUM_EPOCHS = 30
    train_losses, val_losses = [], []

    print(f"\n--- Launching Training Sequence Engine on Device: [{device.type.upper()}] ---")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (imgs, labels_flat, label_lengths, _) in enumerate(train_loader):
            imgs, labels_flat = imgs.to(device), labels_flat.to(device)

            optimizer.zero_grad()

            # Forward pass sequence
            logits = model(imgs)
            seq_len = logits.size(0)
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=seq_len, dtype=torch.long).to(device)
            log_probs = logits.log_softmax(2)

            loss = criterion(log_probs, labels_flat, input_lengths, label_lengths)

            # Catch stray arithmetic overflows instantly
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            if batch_idx % 20 == 0:
                print(
                    f"   [Epoch {epoch:02d}] Step {batch_idx:03d}/{len(train_loader)} | Running Step Loss: {loss.item():.4f}",
                    flush=True)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation Assessment
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for imgs, labels_flat, label_lengths, _ in val_loader:
                imgs, labels_flat = imgs.to(device), labels_flat.to(device)
                logits = model(imgs)
                seq_len = logits.size(0)
                input_lengths = torch.full(size=(imgs.size(0),), fill_value=seq_len, dtype=torch.long).to(device)
                log_probs = logits.log_softmax(2)

                loss = criterion(log_probs, labels_flat, input_lengths, label_lengths)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    epoch_val_loss += loss.item() * imgs.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Update scheduling curves
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS:02d} Finished | Duration: {elapsed:.1f}s | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}\n")

    # --- STEP 8: LOSS VISUALIZATION ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='#1f77b4', lw=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='#ff7f0e', lw=2)
    plt.xlabel('Epoch Cycles')
    plt.ylabel('CTC Log Loss Metrics')
    plt.title('ResNet-BGRU Loss Progression Trends (Stable Math)')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.show()

    # --- STEP 9: ADVANCED EVALUATION & ACCURACY METRICS ---
    print("\nRunning precision evaluation across validation split data...")
    model.eval()
    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for imgs, _, _, target_strs in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            log_probs = logits.log_softmax(2)

            preds = ctc_decode(log_probs, num_to_char)
            all_predictions.extend(preds)
            all_targets.extend(target_strs)

    # Compile exact performance scores
    cer_score = cer_metric(all_predictions, all_targets).item()
    wer_score = wer_metric(all_predictions, all_targets).item()

    exact_matches = sum(1 for p, t in zip(all_predictions, all_targets) if p == t)
    sequence_accuracy = exact_matches / len(all_targets)

    print("\n" + "=" * 50)
    print("             PRODUCTION ACCURACY REPORT           ")
    print("==================================================")
    print(f" Character Error Rate (CER)     : {cer_score * 100:.2f}%")
    print(f" Word Error Rate (WER)          : {wer_score * 100:.2f}%")
    print(f" Word Recognition Accuracy Rate : {(1.0 - wer_score) * 100:.2f}%")
    print(f" Perfect String Match Accuracy  : {sequence_accuracy * 100:.2f}%")
    print("=" * 50)

    print("\nVisual Model Prediction Showcase Logs:")
    for i in range(min(7, len(all_targets))):
        print(f"  True Label: [{all_targets[i]:<15}] --------> Model Prediction: [{all_predictions[i]}]")