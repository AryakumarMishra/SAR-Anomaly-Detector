import csv
import random
from pathlib import Path


# Configurations
VAL_RATIO = 0.20
RANDOM_SEED = 42


# File/Folder Paths
PROJECT_ROOT = Path(__file__).resolve().parents[0]
PATCH_DIR = (PROJECT_ROOT/"data"/"SSDD"/"background_patches")
METADATA_FILE = PATCH_DIR/"metadata.csv"
SPLIT_DIR = (PROJECT_ROOT/"data"/"SSDD"/"splits")
TRAIN_METADATA = SPLIT_DIR / "train.csv"
VAL_METADATA = SPLIT_DIR / "val.csv"


# Loading metadata
with open(METADATA_FILE, "r") as f:
    reader = csv.DictReader(f)
    metadata = list(reader)


# Finding unique source images
source_images = sorted({row["source_image"] for row in metadata})
print(f"Total source images: {len(source_images)}")
print(f"Total patches: {len(metadata)}")


# Shuffling source images
random.seed(RANDOM_SEED)
random.shuffle(source_images)


# Splitting source images
num_val = int(len(source_images) * VAL_RATIO)
val_images = set(source_images[:num_val])
train_images = set(source_images[num_val:])


# Splitting the patches as per source image
train_rows = []
val_rows = []

for row in metadata:
    if row["source_image"] in train_images:
        train_rows.append(row)

    elif row["source_image"] in val_images:
        val_rows.append(row)



# Saving the split data
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = list(metadata[0].keys())

with open(TRAIN_METADATA, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train_rows)


with open(VAL_METADATA, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(val_rows)



# Summary
print("BACKGROUND DATASET SPLIT:\n")

print(f"Training source images: {len(train_images)}")
print(f"Validation source images: {len(val_images)}\n")

print(f"Training patches: {len(train_rows)}")
print(f"Validation patches: {len(val_rows)}\n")

print(f"Training metadata:")
print(TRAIN_METADATA)

print(f"Validation metadata:")
print(VAL_METADATA)