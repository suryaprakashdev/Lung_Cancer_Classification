"""
preprocessing.py
----------------
Phase 1: MONAI-native 3D preprocessing pipeline for the LIDC-IDRI dataset.

Converts raw DICOM scans → labelled 3D .npy volumes + segmentation masks,
with patient-level train/val/test splits (no data leakage).

All resampling uses MONAI transforms (Spacing) instead of scipy.ndimage.zoom.
Scan-level extraction is fully parallelised via ProcessPoolExecutor.

Steps:
  1. Organise raw DICOMs into pylidc-compatible folder structure
  2. Configure pylidc to locate the data
  3. Patient-level split (70/15/15 by LIDC-IDRI-XXXX ID)
  4. Parallel: for every scan → cluster annotations, build volume+mask,
     MONAI Spacing resample to 1mm isotropic, extract 64³ crops, save

Output layout:
  <out_dir>/
    patient_splits.json
    volumes/
      <patient_id>/
        <patient_id>_nodule_<i>_vol.npy      (64,64,64) float32 HU
        <patient_id>_nodule_<i>_mask.npy     (64,64,64) bool
        <patient_id>_nodule_<i>_meta.json    {malignancy, patient_id, ...}

Run:
  python preprocessing.py --raw_dir /path/to/LIDC-IDRI \\
                           --out_dir /path/to/processed_3d \\
                           --num_workers 8
"""

import os
import json
import shutil
import argparse
import configparser
import importlib
import multiprocessing as mp
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import pylidc as pl
import torch
from glob import glob

# ── MONAI-native resampling ──
from monai.transforms import Spacing, SpatialCrop, ResizeWithPadOrCrop

# ──────────────────────────────────────────────
#  Monkey-patches required by older pylidc
# ──────────────────────────────────────────────
np.int   = int
np.float = float
np.bool  = bool
configparser.SafeConfigParser = configparser.ConfigParser


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

CROP_SIZE       = 64                   # target 3D crop size per axis
PAD_VOXELS      = 10                   # padding around nodule bbox
TARGET_SPACING  = (1.0, 1.0, 1.0)     # isotropic 1mm³ voxels
DEFAULT_WORKERS = max(1, mp.cpu_count() - 1)   # leave 1 core free


# ──────────────────────────────────────────────
#  MONAI resampling wrappers
# ──────────────────────────────────────────────

def _monai_resample_volume(
    vol: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = TARGET_SPACING,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Resample a (H, W, D) volume to target spacing using MONAI Spacing.

    MONAI Spacing expects:
      input  : (C, H, W, D) tensor with affine/pixdim metadata
      pixdim : target spacing in each dimension

    Returns
    -------
    resampled : (H', W', D') float32 volume
    scale     : (sy, sx, sz) scale factors applied
    """
    # Add channel dim: (1, H, W, D)
    tensor = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)

    resampler = Spacing(
        pixdim=target_spacing,
        mode="bilinear",        # smooth interpolation for HU data
        padding_mode="border",  # replicate edge values
    )

    # Spacing needs the source pixdim via affine or explicit pixdim arg.
    # We pass src_affine as a simple diagonal.
    src_affine = np.diag([*original_spacing, 1.0]).astype(np.float64)
    resampled = resampler(tensor, affine=src_affine)

    # resampled is a MetaTensor (C, H', W', D')
    out = resampled.squeeze(0).numpy()

    scale = tuple(o / t for o, t in zip(original_spacing, target_spacing))
    return out, scale


def _monai_resample_mask(
    mask: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = TARGET_SPACING,
) -> np.ndarray:
    """
    Resample a binary mask using MONAI Spacing (nearest-neighbour).
    """
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

    resampler = Spacing(
        pixdim=target_spacing,
        mode="nearest",         # preserve binary values
        padding_mode="zeros",
    )

    src_affine = np.diag([*original_spacing, 1.0]).astype(np.float64)
    resampled = resampler(tensor, affine=src_affine)

    return (resampled.squeeze(0).numpy() > 0.5).astype(np.bool_)


def _monai_crop_and_pad(
    vol: np.ndarray,
    center: Tuple[int, int, int],
    crop_size: int = CROP_SIZE,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Extract a centred 3D crop using MONAI transforms.

    Uses SpatialCrop for extraction and ResizeWithPadOrCrop to guarantee
    the output is exactly (crop_size, crop_size, crop_size).

    Returns (crop, start_coords).
    """
    half = crop_size // 2
    # Compute ROI start/end
    starts = [max(0, int(c) - half) for c in center]
    ends   = [min(vol.shape[d], starts[d] + crop_size) for d in range(3)]
    # Re-adjust starts if we hit the edge
    starts = [max(0, ends[d] - crop_size) for d in range(3)]

    # Add channel dim for MONAI: (1, H, W, D)
    tensor = torch.from_numpy(vol.astype(np.float32)).unsqueeze(0)

    cropper = SpatialCrop(roi_start=starts, roi_end=ends)
    cropped = cropper(tensor)

    # Guarantee exact size via pad-or-crop
    padder = ResizeWithPadOrCrop(spatial_size=(crop_size, crop_size, crop_size))
    result = padder(cropped)

    return result.squeeze(0).numpy(), tuple(starts)


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def organise_dicom_folders(raw_dir: str) -> None:
    """
    Move unorganised DICOM series folders under their PatientID parent so
    pylidc can resolve paths ( root/<PatientID>/<SeriesUID>/*.dcm ).
    """
    skip = {"Benign_0", "Malignant_1", "processed_images", "processed_3d"}
    orphans = [
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
        and not d.startswith("LIDC-IDRI")
        and d not in skip
    ]
    print(f"Found {len(orphans)} orphan folders to reorganise …")

    moved, failed = 0, 0
    for folder in orphans:
        folder_path = os.path.join(raw_dir, folder)
        dcm_files   = glob(os.path.join(folder_path, "**/*.dcm"), recursive=True)
        if not dcm_files:
            continue
        try:
            ds         = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
            patient_id = str(ds.PatientID).strip()
            dest       = os.path.join(raw_dir, patient_id)
            os.makedirs(dest, exist_ok=True)
            shutil.move(folder_path, dest)
            moved += 1
        except Exception as e:
            print(f"  Failed {folder[:40]}: {e}")
            failed += 1

    print(f"Done: {moved} moved, {failed} failed.\n")


def configure_pylidc(raw_dir: str) -> None:
    """Write ~/.pylidcrc so pylidc can find the database."""
    config = configparser.ConfigParser()
    config["DEFAULT"] = {"path": raw_dir}
    rc_path = os.path.expanduser("~/.pylidcrc")
    with open(rc_path, "w") as f:
        config.write(f)
    print(f"pylidcrc written → {rc_path}  (path = {raw_dir})")


def load_volume_directly(scan, root: str) -> Tuple[np.ndarray, Tuple]:
    """
    Bypass pylidc's internal path resolution and build a (H, W, Z) HU volume
    directly from DICOMs on disk.  Returns (volume, (row_sp, col_sp, slice_sp)).
    """
    series_path = os.path.join(root, scan.patient_id, scan.series_instance_uid)
    dcm_files   = sorted(glob(os.path.join(series_path, "*.dcm")))

    if not dcm_files:
        raise FileNotFoundError(f"No DICOMs found in {series_path}")

    slices = []
    pixel_spacings = []
    slice_positions = []

    for f in dcm_files:
        ds        = pydicom.dcmread(f)
        img       = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu_slice  = img * slope + intercept

        ps = getattr(ds, "PixelSpacing", [1.0, 1.0])
        pixel_spacings.append((float(ps[0]), float(ps[1])))

        try:
            z = float(ds.ImagePositionPatient[2])
        except (AttributeError, IndexError):
            z = float(getattr(ds, "InstanceNumber", len(slices)))
        slice_positions.append(z)

        slices.append((z, hu_slice))

    slices.sort(key=lambda x: x[0])
    vol = np.stack([s[1] for s in slices], axis=-1)  # (H, W, Z)

    ps = pixel_spacings[0]
    sorted_z = sorted(slice_positions)
    if len(sorted_z) > 1:
        slice_thickness = float(np.median(np.diff(sorted_z)))
    else:
        slice_thickness = float(getattr(pydicom.dcmread(dcm_files[0]),
                                         "SliceThickness", 1.0))

    original_spacing = (ps[0], ps[1], abs(slice_thickness))
    return vol, original_spacing


def build_union_mask(nodule_annotations, vol_shape) -> np.ndarray:
    """
    Build a union segmentation mask from all annotations for a nodule.
    Each annotation's boolean_mask() is placed at its bbox position,
    then all are OR-combined.
    """
    union_mask = np.zeros(vol_shape, dtype=np.bool_)

    for ann in nodule_annotations:
        try:
            bbox   = ann.bbox()
            mask_3d = ann.boolean_mask()

            if mask_3d.sum() == 0:
                continue

            y_s, y_e = bbox[0].start, min(bbox[0].start + mask_3d.shape[0], vol_shape[0])
            x_s, x_e = bbox[1].start, min(bbox[1].start + mask_3d.shape[1], vol_shape[1])
            z_s, z_e = bbox[2].start, min(bbox[2].start + mask_3d.shape[2], vol_shape[2])

            my, mx, mz = y_e - y_s, x_e - x_s, z_e - z_s
            union_mask[y_s:y_e, x_s:x_e, z_s:z_e] |= mask_3d[:my, :mx, :mz]

        except Exception as e:
            # Silently skip bad annotations in workers
            continue

    return union_mask


# ──────────────────────────────────────────────
#  Patient-level split
# ──────────────────────────────────────────────

def build_patient_splits(scans, out_dir: str,
                         train_ratio: float = 0.70,
                         val_ratio: float = 0.15,
                         seed: int = 42) -> dict:
    """
    Split patients (not patches) into train/val/test sets.
    Returns dict: {"train": [ids], "val": [ids], "test": [ids]}
    """
    splits_path = os.path.join(out_dir, "patient_splits.json")

    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        print(f"Loaded existing patient splits from {splits_path}")
        print(f"  Train: {len(splits['train'])} | "
              f"Val: {len(splits['val'])} | "
              f"Test: {len(splits['test'])}")
        return splits

    patient_ids = sorted(set(scan.patient_id for scan in scans))
    n = len(patient_ids)
    print(f"Total unique patients: {n}")

    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    train_n = int(train_ratio * n)
    val_n   = int(val_ratio * n)

    splits = {
        "train": patient_ids[:train_n],
        "val":   patient_ids[train_n:train_n + val_n],
        "test":  patient_ids[train_n + val_n:],
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Patient splits saved → {splits_path}")
    print(f"  Train: {len(splits['train'])} | "
          f"Val: {len(splits['val'])} | "
          f"Test: {len(splits['test'])}")
    return splits


# ──────────────────────────────────────────────
#  Single-scan worker function (runs in parallel)
# ──────────────────────────────────────────────

def _process_single_scan(
    scan_patient_id: str,
    scan_series_uid: str,
    raw_dir: str,
    out_dir: str,
    patient_to_split: Dict[str, str],
    min_annotators: int,
) -> Dict:
    """
    Process one scan: load DICOM → MONAI resample → extract crops → save.

    Designed to run inside a ProcessPoolExecutor worker.
    Returns a stats dict with counts of saved / skipped nodules.
    """
    # Re-import inside worker (fork safety)
    import configparser
    configparser.SafeConfigParser = configparser.ConfigParser
    np.int   = int
    np.float = float
    np.bool  = bool

    import pylidc as pl

    volumes_dir = os.path.join(out_dir, "volumes")
    stats = {"saved": 0, "skipped": 0, "benign": 0, "malignant": 0,
             "patient_id": scan_patient_id, "error": None}

    try:
        # Re-query this specific scan (each worker gets its own DB session)
        scans = pl.query(pl.Scan).filter(
            pl.Scan.patient_id == scan_patient_id,
            pl.Scan.series_instance_uid == scan_series_uid,
        ).all()

        if not scans:
            stats["error"] = "Scan not found in DB"
            return stats

        scan = scans[0]
        nodules = scan.cluster_annotations()
        if not nodules:
            return stats

        # Load full volume with spacing info
        vol, original_spacing = load_volume_directly(scan, raw_dir)

        # ── MONAI Spacing resample to isotropic 1mm³ ──
        vol_resampled, scale_factors = _monai_resample_volume(
            vol, original_spacing, TARGET_SPACING
        )

        patient_dir = os.path.join(volumes_dir, scan.patient_id)
        os.makedirs(patient_dir, exist_ok=True)

        for i, nodule in enumerate(nodules):

            if len(nodule) < min_annotators:
                stats["skipped"] += 1
                continue

            scores = [ann.malignancy for ann in nodule]
            avg_malignancy = sum(scores) / len(scores)

            if avg_malignancy == 3.0:
                stats["skipped"] += 1
                continue

            is_malignant = avg_malignancy > 3.0

            # Build union segmentation mask in original space
            union_mask_orig = build_union_mask(nodule, vol.shape)

            if union_mask_orig.sum() == 0:
                stats["skipped"] += 1
                continue

            # ── MONAI Spacing resample mask (nearest) ──
            union_mask = _monai_resample_mask(
                union_mask_orig, original_spacing, TARGET_SPACING
            )

            # Align shapes (MONAI can differ by ±1 voxel)
            min_shape = tuple(
                min(v, m) for v, m in
                zip(vol_resampled.shape, union_mask.shape)
            )
            vol_aligned  = vol_resampled[:min_shape[0], :min_shape[1], :min_shape[2]]
            mask_aligned = union_mask[:min_shape[0], :min_shape[1], :min_shape[2]]

            # Find nodule centroid in resampled space
            coords = np.argwhere(mask_aligned)
            if len(coords) == 0:
                stats["skipped"] += 1
                continue
            centroid = tuple(coords.mean(axis=0).astype(int))

            # ── MONAI SpatialCrop + ResizeWithPadOrCrop ──
            vol_crop, crop_start = _monai_crop_and_pad(
                vol_aligned, centroid, CROP_SIZE
            )
            mask_crop, _ = _monai_crop_and_pad(
                mask_aligned.astype(np.float32), centroid, CROP_SIZE
            )
            mask_crop = mask_crop > 0.5

            # Save files
            prefix = f"{scan.patient_id}_nodule_{i}"
            np.save(os.path.join(patient_dir, f"{prefix}_vol.npy"),
                    vol_crop.astype(np.float32))
            np.save(os.path.join(patient_dir, f"{prefix}_mask.npy"),
                    mask_crop.astype(np.bool_))

            split = patient_to_split.get(scan.patient_id, "unknown")
            meta = {
                "patient_id":        scan.patient_id,
                "nodule_index":      i,
                "malignancy_scores": scores,
                "avg_malignancy":    avg_malignancy,
                "is_malignant":      is_malignant,
                "label":             int(is_malignant),
                "split":             split,
                "crop_size":         CROP_SIZE,
                "original_spacing":  list(original_spacing),
                "target_spacing":    list(TARGET_SPACING),
                "centroid_resampled": list(centroid),
                "crop_start":        list(crop_start),
                "mask_voxels":       int(mask_crop.sum()),
                "n_annotators":      len(nodule),
                "resampler":         "monai.transforms.Spacing",
            }
            with open(os.path.join(patient_dir, f"{prefix}_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

            stats["saved"] += 1
            if is_malignant:
                stats["malignant"] += 1
            else:
                stats["benign"] += 1

    except Exception as e:
        stats["error"] = f"{type(e).__name__}: {e}"

    return stats


def _checkpoint(path: str, patient_id: str) -> None:
    """Append patient_id to checkpoint file (thread-safe via append mode)."""
    with open(path, "a") as f:
        f.write(patient_id + "\n")


# ──────────────────────────────────────────────
#  Main extraction pipeline (parallelised)
# ──────────────────────────────────────────────

def extract_3d_nodule_volumes(
    raw_dir: str,
    out_dir: str,
    min_annotators: int = 2,
    num_workers: int = DEFAULT_WORKERS,
) -> None:
    """
    Extract 3D nodule volumes and segmentation masks from every scan,
    running up to `num_workers` scans in parallel.

    Uses ProcessPoolExecutor for CPU-bound DICOM loading + MONAI resampling.
    """
    volumes_dir     = os.path.join(out_dir, "volumes")
    checkpoint_file = os.path.join(out_dir, "processed_scans_3d_checkpoint.txt")

    os.makedirs(volumes_dir, exist_ok=True)

    # Resume support
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            processed_ids = set(f.read().splitlines())
        print(f"Resuming: {len(processed_ids)} scans already processed.")
    else:
        processed_ids = set()
        print("Starting fresh.")

    importlib.reload(pl)
    scans = pl.query(pl.Scan).all()
    print(f"Total scans found: {len(scans)}")

    # Build patient-level splits
    splits = build_patient_splits(scans, out_dir)
    patient_to_split = {}
    for split_name, ids in splits.items():
        for pid in ids:
            patient_to_split[pid] = split_name

    # Filter to unprocessed scans
    work_items = [
        (scan.patient_id, scan.series_instance_uid)
        for scan in scans
        if scan.patient_id not in processed_ids
    ]
    print(f"Scans to process: {len(work_items)} (skipping {len(processed_ids)} done)")
    print(f"Workers: {num_workers}\n")

    if not work_items:
        print("Nothing to do — all scans already processed.")
        return

    # ── Parallel extraction ───────────────────
    t0 = time.perf_counter()
    total_saved   = 0
    total_skipped = 0
    errors        = []
    split_stats   = defaultdict(lambda: {"benign": 0, "malignant": 0})

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for pid, suid in work_items:
            future = executor.submit(
                _process_single_scan,
                scan_patient_id=pid,
                scan_series_uid=suid,
                raw_dir=raw_dir,
                out_dir=out_dir,
                patient_to_split=patient_to_split,
                min_annotators=min_annotators,
            )
            futures[future] = pid

        done_count = 0
        for future in as_completed(futures):
            pid = futures[future]
            done_count += 1

            try:
                stats = future.result(timeout=600)  # 10 min timeout per scan
            except Exception as e:
                errors.append(f"{pid}: {e}")
                _checkpoint(checkpoint_file, pid)
                continue

            if stats["error"]:
                errors.append(f"{pid}: {stats['error']}")
            else:
                total_saved   += stats["saved"]
                total_skipped += stats["skipped"]
                split = patient_to_split.get(pid, "unknown")
                split_stats[split]["benign"]    += stats["benign"]
                split_stats[split]["malignant"] += stats["malignant"]

            _checkpoint(checkpoint_file, pid)

            # Progress update every 25 scans
            if done_count % 25 == 0 or done_count == len(work_items):
                elapsed = time.perf_counter() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(work_items) - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{done_count:4d}/{len(work_items)}] "
                    f"saved={total_saved} | skipped={total_skipped} | "
                    f"errors={len(errors)} | "
                    f"{rate:.1f} scans/s | ETA {eta/60:.1f} min"
                )

    elapsed_total = time.perf_counter() - t0
    print(f"\n{'='*55}")
    print(f"  Preprocessing complete")
    print(f"{'='*55}")
    print(f"  Extracted : {total_saved}")
    print(f"  Skipped   : {total_skipped}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Time      : {elapsed_total/60:.1f} min")
    print(f"  Throughput: {len(work_items)/elapsed_total:.1f} scans/s")
    print(f"  Workers   : {num_workers}")
    print(f"  Resampler : monai.transforms.Spacing (bilinear vol / nearest mask)")

    print(f"\n  Per-split class counts:")
    for split in ["train", "val", "test"]:
        s = split_stats[split]
        print(f"    {split:6s}: benign={s['benign']:4d} | "
              f"malignant={s['malignant']:4d} | "
              f"total={s['benign']+s['malignant']:4d}")

    if errors:
        print(f"\n  ⚠  {len(errors)} scan errors:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LIDC-IDRI 3D preprocessing: extract 64³ volumes + seg masks "
                    "(MONAI-native, parallelised)"
    )
    p.add_argument("--raw_dir",  required=True,
                   help="Root directory containing raw LIDC-IDRI DICOM data")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for processed 3D volumes")
    p.add_argument("--min_ann",  type=int, default=2,
                   help="Minimum radiologist annotations required (default: 2)")
    p.add_argument("--num_workers", type=int, default=DEFAULT_WORKERS,
                   help=f"Parallel workers for scan processing (default: {DEFAULT_WORKERS})")
    p.add_argument("--skip_organise", action="store_true",
                   help="Skip the folder reorganisation step")
    return p.parse_args()


if __name__ == "__main__":
    # Force 'spawn' so MONAI + CUDA play nicely in workers
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    if not args.skip_organise:
        organise_dicom_folders(args.raw_dir)

    configure_pylidc(args.raw_dir)
    extract_3d_nodule_volumes(
        args.raw_dir,
        args.out_dir,
        min_annotators=args.min_ann,
        num_workers=args.num_workers,
    )
