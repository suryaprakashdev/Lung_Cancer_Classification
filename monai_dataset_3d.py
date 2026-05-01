"""
monai_dataset_3d.py
-------------------
MONAI-based 3D dataset and DataLoader construction for the lung nodule pipeline.

Loads 3D HU-calibrated .npy volumes and segmentation masks produced by
preprocessing.py. Supports both segmentation (U-Net) and classification
(ResNet) tasks with appropriate transforms.

Expects the directory layout produced by preprocessing.py:

  <data_dir>/
    patient_splits.json
    volumes/
      <patient_id>/
        *_vol.npy      (64,64,64) float32 HU
        *_mask.npy     (64,64,64) bool
        *_meta.json    {...}

Usage:
  from monai_dataset_3d import build_seg_dataloaders, build_cls_dataloaders
"""

import os
import json
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from monai.data import Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    ToTensord,
)


# ──────────────────────────────────────────────
#  Lung windowing constants
# ──────────────────────────────────────────────

LUNG_HU_MIN = -1350
LUNG_HU_MAX = 150


# ──────────────────────────────────────────────
#  Transforms — Segmentation (U-Net)
# ──────────────────────────────────────────────

def get_seg_train_transform() -> Compose:
    """
    3D MONAI transforms for segmentation training.

    Pipeline: Load (already numpy) → channel → window → augment → normalize → tensor
    """
    return Compose([
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # 3D spatial augmentations
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["image", "mask"], prob=0.5, spatial_axes=(0, 1)),
        # Intensity augmentations (image only)
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.01),
        # Normalise
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys=["image", "mask"]),
    ])


def get_seg_val_transform() -> Compose:
    """Deterministic transform for segmentation validation/test."""
    return Compose([
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys=["image", "mask"]),
    ])


# ──────────────────────────────────────────────
#  Transforms — Classification (ResNet)
# ──────────────────────────────────────────────

def get_cls_train_transform() -> Compose:
    """
    3D MONAI transforms for classification training.

    Input is a 64³ HU volume → windowed → augmented → normalised tensor.
    """
    return Compose([
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        RandFlipd(keys="image", spatial_axis=0, prob=0.5),
        RandFlipd(keys="image", spatial_axis=1, prob=0.5),
        RandFlipd(keys="image", spatial_axis=2, prob=0.5),
        RandRotate90d(keys="image", prob=0.5, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.01),
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


def get_cls_val_transform() -> Compose:
    """Deterministic transform for classification validation/test."""
    return Compose([
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys=["image", "label"]),
    ])


# ──────────────────────────────────────────────
#  Data list builders
# ──────────────────────────────────────────────

def _load_patient_splits(data_dir: str) -> dict:
    """Load patient_splits.json produced by preprocessing.py."""
    splits_path = os.path.join(data_dir, "patient_splits.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"patient_splits.json not found in {data_dir}. "
            f"Run preprocessing.py first."
        )
    with open(splits_path) as f:
        return json.load(f)


def _build_seg_data_lists(data_dir: str) -> Dict[str, List[Dict]]:
    """
    Build MONAI-style data dicts for segmentation, split by patient.

    Returns {"train": [...], "val": [...], "test": [...]}
    """
    splits = _load_patient_splits(data_dir)
    volumes_dir = os.path.join(data_dir, "volumes")

    data_lists = {"train": [], "val": [], "test": []}

    for split_name, patient_ids in splits.items():
        for pid in patient_ids:
            patient_dir = os.path.join(volumes_dir, pid)
            if not os.path.isdir(patient_dir):
                continue

            vol_files = sorted(glob(os.path.join(patient_dir, "*_vol.npy")))
            for vol_path in vol_files:
                prefix = vol_path.replace("_vol.npy", "")
                mask_path = f"{prefix}_mask.npy"
                meta_path = f"{prefix}_meta.json"

                if not os.path.exists(mask_path):
                    continue

                # Load the numpy arrays directly (no LoadImaged needed)
                vol = np.load(vol_path).astype(np.float32)
                mask = np.load(mask_path).astype(np.float32)

                entry = {
                    "image": vol,
                    "mask": mask,
                }

                # Load metadata for label
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    entry["label"] = meta.get("label", 0)
                    entry["malignancy"] = meta.get("avg_malignancy", 0.0)
                    entry["patient_id"] = pid
                else:
                    entry["label"] = 0
                    entry["patient_id"] = pid

                data_lists[split_name].append(entry)

    return data_lists


def _build_cls_data_lists(data_dir: str) -> Dict[str, List[Dict]]:
    """
    Build MONAI-style data dicts for classification, split by patient.

    Returns {"train": [...], "val": [...], "test": [...]}
    """
    splits = _load_patient_splits(data_dir)
    volumes_dir = os.path.join(data_dir, "volumes")

    data_lists = {"train": [], "val": [], "test": []}

    for split_name, patient_ids in splits.items():
        for pid in patient_ids:
            patient_dir = os.path.join(volumes_dir, pid)
            if not os.path.isdir(patient_dir):
                continue

            vol_files = sorted(glob(os.path.join(patient_dir, "*_vol.npy")))
            for vol_path in vol_files:
                prefix = vol_path.replace("_vol.npy", "")
                meta_path = f"{prefix}_meta.json"

                # Load volume directly
                vol = np.load(vol_path).astype(np.float32)

                entry = {"image": vol}

                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    entry["label"] = meta.get("label", 0)
                    entry["patient_id"] = pid
                else:
                    entry["label"] = 0
                    entry["patient_id"] = pid

                data_lists[split_name].append(entry)

    return data_lists


# ──────────────────────────────────────────────
#  DataLoader factories
# ──────────────────────────────────────────────

def build_seg_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders for 3D segmentation.

    Returns (train_loader, val_loader, test_loader)
    """
    data_lists = _build_seg_data_lists(data_dir)

    train_set = Dataset(data=data_lists["train"], transform=get_seg_train_transform())
    val_set   = Dataset(data=data_lists["val"],   transform=get_seg_val_transform())
    test_set  = Dataset(data=data_lists["test"],  transform=get_seg_val_transform())

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    for name, dl in [("train", data_lists["train"]),
                     ("val", data_lists["val"]),
                     ("test", data_lists["test"])]:
        print(f"  {name:6s}: {len(dl)} volumes")

    return train_loader, val_loader, test_loader


def build_cls_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train/val/test DataLoaders for 3D classification.

    Returns (train_loader, val_loader, test_loader, pos_weight)
    """
    data_lists = _build_cls_data_lists(data_dir)

    train_set = Dataset(data=data_lists["train"], transform=get_cls_train_transform())
    val_set   = Dataset(data=data_lists["val"],   transform=get_cls_val_transform())
    test_set  = Dataset(data=data_lists["test"],  transform=get_cls_val_transform())

    # Compute pos_weight for class imbalance
    train_labels = [d["label"] for d in data_lists["train"]]
    n_benign = train_labels.count(0)
    n_malignant = train_labels.count(1)

    if n_malignant == 0:
        raise RuntimeError("No malignant samples in training split!")

    pos_weight = torch.tensor([n_benign / n_malignant])

    # Weighted sampler for training
    sample_weights = [1.0 / n_benign if l == 0 else 1.0 / n_malignant
                      for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Dataset split (patient-level):")
    for name, dl in [("train", data_lists["train"]),
                     ("val", data_lists["val"]),
                     ("test", data_lists["test"])]:
        labels = [d["label"] for d in dl]
        print(f"  {name:6s}: {len(dl)} volumes | "
              f"benign={labels.count(0)} | malignant={labels.count(1)}")
    print(f"pos_weight: {pos_weight.item():.3f}")

    return train_loader, val_loader, test_loader, pos_weight


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed_3d"

    print("=== Segmentation DataLoaders ===")
    train_l, val_l, test_l = build_seg_dataloaders(data_dir, batch_size=2)
    batch = next(iter(train_l))
    print(f"Image shape: {batch['image'].shape}  dtype: {batch['image'].dtype}")
    print(f"Mask  shape: {batch['mask'].shape}   dtype: {batch['mask'].dtype}")

    print("\n=== Classification DataLoaders ===")
    train_l, val_l, test_l, pw = build_cls_dataloaders(data_dir, batch_size=2)
    batch = next(iter(train_l))
    print(f"Image shape: {batch['image'].shape}  dtype: {batch['image'].dtype}")
    print(f"Labels: {batch['label'][:4].tolist()}")
