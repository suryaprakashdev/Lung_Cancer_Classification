"""
dataset.py
----------
Dataset construction and DataLoader creation for the lung nodule pipeline.

Expects a directory layout produced by preprocessing.py:

  <data_dir>/
    Benign_0/      ← class 0
      *.png
    Malignant_1/   ← class 1
      *.png

Usage:
  from dataset import build_dataloaders
  train_loader, val_loader, test_loader, class_weights = build_dataloaders(
      data_dir="/path/to/processed_images"
  )
"""

import torch
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from typing import Tuple


# ──────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """Augmentation + normalisation for training."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),   # replicate to 3ch for pretrained norms
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_val_transform() -> transforms.Compose:
    """Deterministic transform for validation and test sets."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


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
    Build train / val / test DataLoaders from an ImageFolder directory.

    Returns
    -------
    train_loader, val_loader, test_loader, pos_weight

    pos_weight  – tensor suitable for BCEWithLogitsLoss to handle class imbalance
                  (benign_count / malignant_count)
    """
    # Load without transforms first so we can split indices reproducibly
    full_dataset = datasets.ImageFolder(data_dir)

    n       = len(full_dataset)
    train_n = int(train_ratio * n)
    val_n   = int(val_ratio   * n)
    test_n  = n - train_n - val_n

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_n, val_n, test_n], generator=generator
    )

    # Apply split-specific transforms via a thin wrapper
    train_set = _TransformSubset(train_set, get_train_transform())
    val_set   = _TransformSubset(val_set,   get_val_transform())
    test_set  = _TransformSubset(test_set,  get_val_transform())

    # Weighted sampler to handle class imbalance in training batches
    train_labels  = [full_dataset.targets[i] for i in train_set.subset.indices]
    class_counts  = [train_labels.count(0), train_labels.count(1)]
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    pos_weight = torch.tensor([class_counts[0] / class_counts[1]])

    print(f"Dataset split  — total: {n} | "
          f"train: {train_n} | val: {val_n} | test: {test_n}")
    print(f"Class counts   — Benign: {class_counts[0]} | "
          f"Malignant: {class_counts[1]}")
    print(f"pos_weight     — {pos_weight.item():.3f}")

    return train_loader, val_loader, test_loader, pos_weight


# ──────────────────────────────────────────────
#  Helper: per-split transform wrapper
# ──────────────────────────────────────────────

class _TransformSubset(torch.utils.data.Dataset):
    """Apply a different transform to a Subset without mutating the parent."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        # img is a PIL Image at this stage (ImageFolder with no transform set)
        return self.transform(img), label


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    train_loader, val_loader, test_loader, pos_weight = build_dataloaders(data_dir)

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}  Labels: {labels[:8].tolist()}")
