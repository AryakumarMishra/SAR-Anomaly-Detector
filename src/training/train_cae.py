import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.src_data.background_dataset import SARBackgroundDataset
from src.models.cae import ConvolutionalAutoencoder


# Configurations
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 7
NUM_WORKERS = 2


# Reproduceability
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# File/Folder Path
PATCH_DIR = PROJECT_ROOT / "data/SSDD/background_patches"
SPLIT_DIR = PROJECT_ROOT / "data/SSDD/splits"

TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "models/checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs/training"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_cae.pth"



# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")




# Dataset and DataLoaders
train_dataset = SARBackgroundDataset(
    metadata_file=TRAIN_CSV,
    patch_dir=PATCH_DIR,
)

val_dataset = SARBackgroundDataset(
    metadata_file=VAL_CSV,
    patch_dir=PATCH_DIR,
)

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise RuntimeError("Training or validation dataset is empty. Check the split CSV files and paths.")

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": device.type == "cuda",
}

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **loader_kwargs,
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **loader_kwargs,
)

print(f"Training patches: {len(train_dataset)}")
print(f"Validation patches: {len(val_dataset)}")


# Model, Loss, Optimizer
model = ConvolutionalAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# Training and Validation Loop
history = {
    "train_loss": [],
    "val_loss": [],
}

best_val_loss = float("inf")
epochs_without_improvement = 0


def run_epoch(loader, training):
    """Run one training or validation epoch."""

    model.train(training)

    total_squared_error = 0.0
    total_pixels = 0

    for images in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            reconstructions = model(images)
            loss = criterion(reconstructions, images)

            if training:
                loss.backward()
                optimizer.step()

        pixels = images.numel()
        total_squared_error += loss.item() * pixels
        total_pixels += pixels

    return total_squared_error / total_pixels


for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    improved = val_loss < best_val_loss

    if improved:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        # Save the best model and enough information to reproduce this training run.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "patch_size": 64,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "seed": RANDOM_SEED,
                },
            },
            BEST_MODEL_PATH,
        )
    else:
        epochs_without_improvement += 1

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"train_loss={train_loss:.6f} "
        f"val_loss={val_loss:.6f}"
        + ("  [best]" if improved else "")
    )

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break



# Saving Loss History
history_path = OUTPUT_DIR / "loss_history.csv"

with open(history_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

    for i, (train_loss, val_loss) in enumerate(
        zip(history["train_loss"], history["val_loss"]),
        start=1,
    ):
        writer.writerow([i, train_loss, val_loss])



# Plotting the Learning Curves
plt.figure(figsize=(9, 5))
plt.plot(history["train_loss"], label="Train MSE")
plt.plot(history["val_loss"], label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.title("CAE Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

loss_plot = OUTPUT_DIR / "loss_curve.png"
plt.savefig(loss_plot, dpi=150)
plt.close()

print("\nTraining complete.")
print(f"Best validation MSE: {best_val_loss:.6f}")
print(f"Best checkpoint: {BEST_MODEL_PATH}")
print(f"Loss history: {history_path}")
print(f"Loss plot: {loss_plot}")