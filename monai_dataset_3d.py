"""
monai_dataset_3d.py
-------------------
MONAI-native 3D dataset and DataLoader construction for the lung nodule pipeline.

Key performance features:
  ▸ CacheDataset    — deterministic transforms run once, cached in RAM
  ▸ PersistentDataset — disk-backed cache for large datasets
  ▸ ThreadDataLoader — MONAI's multi-threaded loader (no GIL for numpy/torch ops)
  ▸ Lazy file-path loading via LoadImaged (no upfront np.load of all data)
  ▸ num_workers auto-tuned to available CPU cores

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

import multiprocessing as mp
import os
import json
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from monai.data import (
    CacheDataset,
    DataLoader,
    PersistentDataset,
    ThreadDataLoader,
)
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
)


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

LUNG_HU_MIN     = -1350
LUNG_HU_MAX     = 150
DEFAULT_WORKERS = max(1, mp.cpu_count() - 1)


# ──────────────────────────────────────────────
#  Custom numpy loader for .npy files
# ──────────────────────────────────────────────

class NumpyLoader:
    """
    MONAI-compatible reader for .npy files.
    LoadImaged doesn't natively support .npy, so we register this.
    """
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, path):
        data = np.load(path).astype(self.dtype)
        return data


def _load_npy_image(path):
    """Load a .npy file and return as float32."""
    return np.load(path).astype(np.float32)


def _load_npy_mask(path):
    """Load a .npy mask file and return as float32 for MONAI transforms."""
    return np.load(path).astype(np.float32)


# ──────────────────────────────────────────────
#  Transforms — Segmentation (U-Net)
# ──────────────────────────────────────────────

def get_seg_train_transform() -> Compose:
    """
    3D MONAI transforms for segmentation training.

    Pipeline: lazy-load .npy → channel → window → augment → normalize → tensor
    """
    return Compose([
        # Lazy load: path string → numpy array
        Lambdad(keys="image", func=_load_npy_image),
        Lambdad(keys="mask",  func=_load_npy_mask),
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
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
    ])


def get_seg_val_transform() -> Compose:
    """Deterministic transform for segmentation validation/test (cacheable)."""
    return Compose([
        Lambdad(keys="image", func=_load_npy_image),
        Lambdad(keys="mask",  func=_load_npy_mask),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys="image", channel_wise=True),
        EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
    ])


# ──────────────────────────────────────────────
#  Transforms — Classification (ResNet)
# ──────────────────────────────────────────────

def get_cls_train_transform() -> Compose:
    """
    3D MONAI transforms for classification training.
    Input is a path to a 64³ HU .npy file.
    """
    return Compose([
        Lambdad(keys="image", func=_load_npy_image),
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
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_cls_val_transform() -> Compose:
    """Deterministic transform for classification validation/test."""
    return Compose([
        Lambdad(keys="image", func=_load_npy_image),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys="image", channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


# ──────────────────────────────────────────────
#  Data list builders (lazy: store paths, not arrays)
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

    Stores FILE PATHS (not loaded arrays) for lazy loading.
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

                # Store PATHS — loaded lazily by Lambdad in transforms
                entry = {
                    "image": vol_path,
                    "mask":  mask_path,
                }

                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    entry["label"]      = meta.get("label", 0)
                    entry["malignancy"] = meta.get("avg_malignancy", 0.0)
                    entry["patient_id"] = pid
                else:
                    entry["label"]      = 0
                    entry["patient_id"] = pid

                data_lists[split_name].append(entry)

    return data_lists


def _build_cls_data_lists(data_dir: str) -> Dict[str, List[Dict]]:
    """
    Build MONAI-style data dicts for classification, split by patient.
    Stores FILE PATHS for lazy loading.
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

                entry = {"image": vol_path}

                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    entry["label"]      = meta.get("label", 0)
                    entry["patient_id"] = pid
                else:
                    entry["label"]      = 0
                    entry["patient_id"] = pid

                data_lists[split_name].append(entry)

    return data_lists


# ──────────────────────────────────────────────
#  DataLoader factories
# ──────────────────────────────────────────────

def build_seg_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = DEFAULT_WORKERS,
    cache_rate: float = 1.0,
    persistent_cache_dir: Optional[str] = None,
) -> Tuple:
    """
    Build train/val/test DataLoaders for 3D segmentation.

    Caching strategy (choose one):
      ▸ cache_rate=1.0 (default)  — CacheDataset: all data cached in RAM
      ▸ cache_rate<1.0            — CacheDataset: partial cache
      ▸ persistent_cache_dir set  — PersistentDataset: disk-backed cache

    ThreadDataLoader is used for val/test (deterministic → thread-safe).

    Returns (train_loader, val_loader, test_loader)
    """
    data_lists = _build_seg_data_lists(data_dir)

    # ── Training: CacheDataset for random augmentations ──
    # Only deterministic (pre-augmentation) transforms benefit from caching,
    # but CacheDataset caches up to the first randomized transform automatically.
    train_set = CacheDataset(
        data=data_lists["train"],
        transform=get_seg_train_transform(),
        cache_rate=cache_rate,
        num_workers=num_workers,   # parallel caching threads
    )

    # ── Val/Test: full caching (no randomness) ──
    if persistent_cache_dir:
        # Disk-backed cache — survives restarts, great for Colab
        val_cache = os.path.join(persistent_cache_dir, "seg_val_cache")
        test_cache = os.path.join(persistent_cache_dir, "seg_test_cache")
        os.makedirs(val_cache, exist_ok=True)
        os.makedirs(test_cache, exist_ok=True)

        val_set = PersistentDataset(
            data=data_lists["val"],
            transform=get_seg_val_transform(),
            cache_dir=val_cache,
        )
        test_set = PersistentDataset(
            data=data_lists["test"],
            transform=get_seg_val_transform(),
            cache_dir=test_cache,
        )
    else:
        val_set = CacheDataset(
            data=data_lists["val"],
            transform=get_seg_val_transform(),
            cache_rate=1.0,
            num_workers=num_workers,
        )
        test_set = CacheDataset(
            data=data_lists["test"],
            transform=get_seg_val_transform(),
            cache_rate=1.0,
            num_workers=num_workers,
        )

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    # ThreadDataLoader for val/test: lower overhead for cached data
    val_loader = ThreadDataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0,  # ThreadDataLoader handles parallelism internally
    )
    test_loader = ThreadDataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    _print_split_info("Segmentation", data_lists)
    return train_loader, val_loader, test_loader


def build_cls_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = DEFAULT_WORKERS,
    cache_rate: float = 1.0,
    persistent_cache_dir: Optional[str] = None,
) -> Tuple:
    """
    Build train/val/test DataLoaders for 3D classification.

    Returns (train_loader, val_loader, test_loader, pos_weight)
    """
    data_lists = _build_cls_data_lists(data_dir)

    # ── Compute class weights ──
    train_labels = [d["label"] for d in data_lists["train"]]
    n_benign    = train_labels.count(0)
    n_malignant = train_labels.count(1)

    if n_malignant == 0:
        raise RuntimeError("No malignant samples in training split!")

    pos_weight = torch.tensor([n_benign / n_malignant])

    # Weighted sampler for class-balanced batches
    sample_weights = [
        1.0 / n_benign if l == 0 else 1.0 / n_malignant
        for l in train_labels
    ]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # ── Training: CacheDataset ──
    train_set = CacheDataset(
        data=data_lists["train"],
        transform=get_cls_train_transform(),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    # ── Val/Test: full caching ──
    if persistent_cache_dir:
        val_cache  = os.path.join(persistent_cache_dir, "cls_val_cache")
        test_cache = os.path.join(persistent_cache_dir, "cls_test_cache")
        os.makedirs(val_cache, exist_ok=True)
        os.makedirs(test_cache, exist_ok=True)

        val_set = PersistentDataset(
            data=data_lists["val"],
            transform=get_cls_val_transform(),
            cache_dir=val_cache,
        )
        test_set = PersistentDataset(
            data=data_lists["test"],
            transform=get_cls_val_transform(),
            cache_dir=test_cache,
        )
    else:
        val_set = CacheDataset(
            data=data_lists["val"],
            transform=get_cls_val_transform(),
            cache_rate=1.0,
            num_workers=num_workers,
        )
        test_set = CacheDataset(
            data=data_lists["test"],
            transform=get_cls_val_transform(),
            cache_rate=1.0,
            num_workers=num_workers,
        )

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = ThreadDataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )
    test_loader = ThreadDataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    _print_split_info("Classification", data_lists, pos_weight)
    return train_loader, val_loader, test_loader, pos_weight


def _print_split_info(task: str, data_lists: dict,
                      pos_weight: Optional[torch.Tensor] = None) -> None:
    """Pretty-print dataset split statistics."""
    print(f"\n  {task} dataset (patient-level, lazy file-path loading):")
    for name in ["train", "val", "test"]:
        dl = data_lists[name]
        labels = [d.get("label", -1) for d in dl]
        benign    = labels.count(0)
        malignant = labels.count(1)
        print(f"    {name:6s}: {len(dl):4d} volumes | "
              f"benign={benign} | malignant={malignant}")
    if pos_weight is not None:
        print(f"  pos_weight: {pos_weight.item():.3f}")
    print()


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed_3d"

    print("=" * 50)
    print("  MONAI Dataset Sanity Check")
    print("=" * 50)

    # ── Segmentation ──
    print("\n=== Segmentation DataLoaders ===")
    t0 = time.perf_counter()
    train_l, val_l, test_l = build_seg_dataloaders(
        data_dir, batch_size=2, num_workers=2, cache_rate=1.0,
    )
    cache_time = time.perf_counter() - t0
    print(f"  Cache build time: {cache_time:.1f}s")

    t0 = time.perf_counter()
    batch = next(iter(train_l))
    load_time = time.perf_counter() - t0
    print(f"  Image shape: {batch['image'].shape}  dtype: {batch['image'].dtype}")
    print(f"  Mask  shape: {batch['mask'].shape}   dtype: {batch['mask'].dtype}")
    print(f"  First batch time: {load_time*1000:.0f}ms")

    # ── Classification ──
    print("\n=== Classification DataLoaders ===")
    t0 = time.perf_counter()
    train_l, val_l, test_l, pw = build_cls_dataloaders(
        data_dir, batch_size=2, num_workers=2, cache_rate=1.0,
    )
    cache_time = time.perf_counter() - t0
    print(f"  Cache build time: {cache_time:.1f}s")

    t0 = time.perf_counter()
    batch = next(iter(train_l))
    load_time = time.perf_counter() - t0
    print(f"  Image shape: {batch['image'].shape}  dtype: {batch['image'].dtype}")
    print(f"  Labels: {batch['label'][:4].tolist()}")
    print(f"  First batch time: {load_time*1000:.0f}ms")
