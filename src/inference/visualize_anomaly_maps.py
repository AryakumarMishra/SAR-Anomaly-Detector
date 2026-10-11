import csv
import json
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cae import ConvolutionalAutoencoder


# Configurations
PATCH_SIZE = 64
STRIDE = 16
NUM_IMAGES = 6
INFERENCE_BATCH_SIZE = 128
RANDOM_SEED = 42


# File/ Folder paths
IMAGE_DIR = PROJECT_ROOT / "data/SSDD/images/train"
ANNOTATION_FILE = PROJECT_ROOT / "data/SSDD/annotations/train.json"
VAL_CSV = PROJECT_ROOT / "data/SSDD/splits/val.csv"
CHECKPOINT = PROJECT_ROOT / "trained_models/checkpoints/best_cae.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs/anomaly_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Device and checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CHECKPOINT, map_location=device,)

model = ConvolutionalAutoencoder().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Device: {device}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Checkpoint validation MSE: {checkpoint['val_loss']:.6f}")



# Loading validation source images
with open(VAL_CSV, newline="") as f:
    val_rows = list(csv.DictReader(f))

val_filenames = sorted({row["source_image"] for row in val_rows})

with open(ANNOTATION_FILE) as f:
    annotation_data = json.load(f)

image_info_by_name = {
    item["file_name"]: item
    for item in annotation_data["images"]
}

annotations_by_id = {}

for ann in annotation_data["annotations"]:
    annotations_by_id.setdefault(
        ann["image_id"], []
    ).append(ann)

# Selecting validation images with at least one ship.
candidates = [
    name for name in val_filenames
    if name in image_info_by_name
    and annotations_by_id.get(
        image_info_by_name[name]["id"], []
    )
]

if not candidates:
    raise RuntimeError(
        "No annotated validation images found. "
        "Check val.csv and train.json."
    )

random.seed(RANDOM_SEED)
selected_images = random.sample(
    candidates, min(NUM_IMAGES, len(candidates))
)

print(f"Selected validation images: {len(selected_images)}")



# Sliding window utilities
def window_starts(length, patch_size, stride):
    """Include the final edge even if stride doesn't divide evenly."""
    if length < patch_size:
        raise ValueError(
            f"Image dimension {length} is smaller than {patch_size}"
        )

    starts = list(range(0, length - patch_size + 1, stride))
    final_start = length - patch_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


@torch.no_grad()
def make_anomaly_map(image):
    """
    Return a full-resolution map of mean squared
    reconstruction error, averaged across overlapping windows.
    """

    height, width = image.shape

    ys = window_starts(height, PATCH_SIZE, STRIDE)
    xs = window_starts(width, PATCH_SIZE, STRIDE)

    patches = []
    positions = []

    for y in ys:
        for x in xs:
            patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]

            # Match training preprocessing: grayscale / 255.
            patch = patch.astype(np.float32) / 255.0

            patches.append(patch)
            positions.append((x, y))

    error_sum = np.zeros(
        (height, width), dtype=np.float32
    )
    coverage = np.zeros(
        (height, width), dtype=np.float32
    )

    for start in range(0, len(patches), INFERENCE_BATCH_SIZE):
        batch_patches = patches[start:start + INFERENCE_BATCH_SIZE]
        batch_positions = positions[start:start + INFERENCE_BATCH_SIZE]

        batch = torch.from_numpy(
            np.stack(batch_patches)[:, None, :, :]
        ).to(device)

        reconstructed = model(batch)

        # Per-pixel squared reconstruction error.
        errors = (
            (reconstructed - batch) ** 2
        ).squeeze(1).cpu().numpy()

        for error, (x, y) in zip(errors, batch_positions):
            error_sum[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += error
            coverage[y:y + PATCH_SIZE, x:x + PATCH_SIZE] += 1.0

    return error_sum / np.maximum(coverage, 1.0)


# Generating and visualizing the maps
for filename in selected_images:
    info = image_info_by_name[filename]
    image_id = info["id"]

    image_path = IMAGE_DIR / filename
    image = cv2.imread(
        str(image_path), cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(f"Skipping unreadable image: {filename}")
        continue

    annotations = annotations_by_id.get(image_id, [])
    error_map = make_anomaly_map(image)

    vmax = float(np.percentile(error_map, 99.5))
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Original image with ground truth.
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        rect = plt.Rectangle(
            (x, y), w, h,
            fill=False, edgecolor="lime", linewidth=1.5
        )
        axes[0].add_patch(rect)

    axes[0].set_title("Original + GT ship boxes")

    # Reconstruction error.
    im = axes[1].imshow(
        error_map, cmap="hot", vmin=0, vmax=vmax
    )
    axes[1].set_title("Reconstruction-error map")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Error overlay, retaining ground-truth boxes.
    axes[2].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(
        error_map, cmap="hot",
        vmin=0, vmax=vmax, alpha=0.55
    )

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        rect = plt.Rectangle(
            (x, y), w, h,
            fill=False, edgecolor="lime", linewidth=1.5
        )
        axes[2].add_patch(rect)

    axes[2].set_title("Error overlay + GT boxes")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"{filename} | Ground-truth ships: {len(annotations)}"
    )
    fig.tight_layout()

    output_path = OUTPUT_DIR / f"{Path(filename).stem}_anomaly.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(
        f"{filename}: "
        f"mean error={error_map.mean():.6f}, "
        f"max error={error_map.max():.6f}, "
        f"saved={output_path.name}"
    )

print(f"\nAll visualizations saved to: {OUTPUT_DIR}")