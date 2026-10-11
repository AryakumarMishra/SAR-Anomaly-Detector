# SAR Anomaly Detector

An experimental **unsupervised ship-anomaly localization** project using a convolutional autoencoder (CAE) trained on background patches from the SAR Ship Detection Dataset (SSDD).


## Motivation

Ships can appear as bright, structured targets against the textured ocean background in synthetic aperture radar (SAR) imagery. This project
explores whether a model trained only on ocean/background patches can highlight unusual regions using reconstruction error, without being
directly trained on ship labels.

## Current pipeline

1.  Inspect SSDD's COCO-style JSON annotations.
2.  Extract 64 × 64 background patches, avoiding annotated ship boxes.
3.  Split patches by **source image**, not randomly by patch, to reduce leakage.
4.  Train a convolutional autoencoder using reconstruction loss.
5.  Run overlapping sliding-window inference on SAR images.
6.  Aggregate pixel-wise squared reconstruction error into anomaly maps.
7.  Compare high-error regions with ground-truth ship annotations.

## Baseline configuration and results

  Model: Convolutional autoencoder (CAE)
  Input/output: 64 × 64 grayscale
  Parameters: 775,425
  Normalization: Pixel values divided by 255
  Loss: Mean squared error (MSE)
  Optimizer: Adam
  Learning rate: `1e-3`
  Batch size: `32`
  Maximum epochs: `50`
  Early-stopping patience: `7` epochs
  Training hardware         NVIDIA RTX 3050 Laptop GPU, 6 GB VRAM

The first baseline run reached a best validation MSE of **0.002792 at epoch 50**, with training MSE **0.002707** at that epoch.
This suggests the model reconstructs held-out background patches reasonably consistently.

## Dataset

The current implementation uses the SSDD copy downloaded from Kaggle. The inspected training annotation file contained 928 image records and
2,041 ship annotations. The background-patch collection contained 13,920 patches.

Reference repository: https://github.com/TianwenZhang0825/Official-SSDD

## Project structure

``` text
SAR_Anomaly_Detection/
├── data/
│   └── SSDD/                         # Local data; not commited
│       ├── annotations/
│       ├── images/
│       ├── background_patches/
│       └── splits/
├── src/
│   ├── data/
│   │   ├── background_dataset.py
│   │   └── create_background_split.py
│   ├── models/
│   │   └── cae.py
│   ├── training/
│   │   └── train_cae.py
│   └── inference/
│       └── visualize_anomaly_maps.py
├── models/
│   └── checkpoints/                  # Generated; usually not committed
├── outputs/
│   ├── training/
│   └── anomaly_maps/
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment setup

Python 3.10 or newer is recommended.

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


