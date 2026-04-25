"""
monai_dataset.py
----------------
MONAI-based dataset and DataLoader construction for the lung nodule pipeline.

Loads raw HU-calibrated .npy patches produced by preprocessing.py and applies
medical-imaging-specific transforms via MONAI (intensity windowing, spatial
augmentations, normalisation) — all at training time with full float32 precision.

Expects a directory layout produced by preprocessing.py:

  <data_dir>/
    Benign_0/      ← class 0
      *.npy
    Malignant_1/   ← class 1
      *.npy

Usage:
  from monai_dataset import build_dataloaders
  train_loader, val_loader, test_loader, class_weights = build_dataloaders(
      data_dir="/path/to/processed_images"
  )
"""

import os
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from monai.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RepeatChanneld,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)


# ──────────────────────────────────────────────
#  Lung windowing constants (must match training expectations)
# ──────────────────────────────────────────────

LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500
LUNG_HU_MIN = LUNG_WINDOW_CENTER - LUNG_WINDOW_WIDTH // 2   # -1350
LUNG_HU_MAX = LUNG_WINDOW_CENTER + LUNG_WINDOW_WIDTH // 2   #  150


# ──────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────

def get_train_transform() -> Compose:
    """
    MONAI augmentation + normalisation for training.

    Pipeline:
      1. Load .npy → add channel dim → (1, H, W)
      2. ScaleIntensityRange: lung windowing  HU → [0, 1]
      3. Resize to 224×224
      4. Replicate 1-ch → 3-ch  (for ImageNet-pretrained backbones)
      5. Data augmentation (flip, rotate, intensity jitter)
      6. Normalise per-channel
      7. Convert to tensor
    """
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN,
            a_max=LUNG_HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(keys="image", spatial_size=(224, 224), mode="bilinear"),
        RepeatChanneld(keys="image", repeats=3),
        # --- Augmentations ---
        RandFlipd(keys="image", spatial_axis=0, prob=0.5),  # vertical
        RandFlipd(keys="image", spatial_axis=1, prob=0.5),  # horizontal
        RandRotated(
            keys="image",
            range_x=np.deg2rad(15),
            prob=0.5,
            mode="bilinear",
            padding_mode="zeros",
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        # --- Normalisation ---
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


def get_val_transform() -> Compose:
    """Deterministic transform for validation and test sets."""
    return Compose([
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN,
            a_max=LUNG_HU_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        Resized(keys="image", spatial_size=(224, 224), mode="bilinear"),
        RepeatChanneld(keys="image", repeats=3),
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


# ──────────────────────────────────────────────
#  Data list builder
# ──────────────────────────────────────────────

def _build_data_list(data_dir: str) -> Tuple[List[Dict], List[int]]:
    """
    Walk Benign_0/ and Malignant_1/ directories and return
    a MONAI-style list of dicts + flat label list.
    """
    class_dirs = {
        "Benign_0": 0,
        "Malignant_1": 1,
    }

    data_list: List[Dict] = []
    labels: List[int] = []

    for folder, label in class_dirs.items():
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"Expected class folder not found: {folder_path}\n"
                f"Run preprocessing.py first to generate .npy patches."
            )

        npy_files = sorted(glob(os.path.join(folder_path, "*.npy")))
        for f in npy_files:
            data_list.append({"image": f, "label": label})
            labels.append(label)

    if not data_list:
        raise RuntimeError(
            f"No .npy files found in {data_dir}/Benign_0 or Malignant_1. "
            f"Run preprocessing.py first."
        )

    return data_list, labels


# ──────────────────────────────────────────────
#  DataLoader factory
# ──────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    batch_size:  int   = 32,
    num_workers: int   = 2,
    seed:        int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders from .npy patch directories.

    Returns
    -------
    train_loader, val_loader, test_loader, pos_weight

    pos_weight  – tensor suitable for BCEWithLogitsLoss to handle class imbalance
                  (benign_count / malignant_count)
    """
    data_list, all_labels = _build_data_list(data_dir)

    n       = len(data_list)
    train_n = int(train_ratio * n)
    val_n   = int(val_ratio   * n)
    test_n  = n - train_n - val_n

    # Reproducible shuffle + split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    train_indices = indices[:train_n]
    val_indices   = indices[train_n:train_n + val_n]
    test_indices  = indices[train_n + val_n:]

    train_data = [data_list[i] for i in train_indices]
    val_data   = [data_list[i] for i in val_indices]
    test_data  = [data_list[i] for i in test_indices]

    # Create MONAI Datasets with split-specific transforms
    train_set = Dataset(data=train_data, transform=get_train_transform())
    val_set   = Dataset(data=val_data,   transform=get_val_transform())
    test_set  = Dataset(data=test_data,  transform=get_val_transform())

    # Weighted sampler to handle class imbalance in training batches
    train_labels  = [all_labels[i] for i in train_indices]
    class_counts  = [train_labels.count(0), train_labels.count(1)]

    if class_counts[1] == 0:
        raise RuntimeError(
            "No malignant samples in training split. "
            "Check your data or adjust split ratios."
        )

    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    pos_weight = torch.tensor([class_counts[0] / class_counts[1]])

    print(f"Dataset split  — total: {n} | "
          f"train: {train_n} | val: {val_n} | test: {test_n}")
    print(f"Class counts   — Benign: {class_counts[0]} | "
          f"Malignant: {class_counts[1]}")
    print(f"pos_weight     — {pos_weight.item():.3f}")
    print(f"Transforms     — MONAI (HU windowing [{LUNG_HU_MIN}, {LUNG_HU_MAX}])")

    return train_loader, val_loader, test_loader, pos_weight


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    train_loader, val_loader, test_loader, pos_weight = build_dataloaders(data_dir)

    batch = next(iter(train_loader))
    imgs   = batch["image"]
    labels = batch["label"]
    print(f"Batch shape: {imgs.shape}  dtype: {imgs.dtype}")
    print(f"Value range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"Labels: {labels[:8].tolist()}")
