import csv
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class SARBackgroundDataset(Dataset):
    def __init__(self, metadata_file, patch_dir, transform=None):
        self.patch_dir = Path(patch_dir)
        self.transform = transform

        with open(metadata_file, "r") as f:
            reader = csv.DictReader(f)
            self.metadata = list(reader)


    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, index):
        row = self.metadata[index]
        patch_path = (self.patch_dir/row["patch_filename"])

        image = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not load image: {patch_path}")

        # Converting uint8 [0,255] to float32 [0,1]
        image = image.astype("float32") / 255.0

        # Adding channel dimension: H × W to 1 × H × W
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        if self.transform is not None:
            image = self.transform(image)

        return image