"""
preprocessing.py
----------------
Phase 1: 3D preprocessing pipeline for the LIDC-IDRI dataset.

Converts raw DICOM scans → labelled 3D .npy volumes + segmentation masks,
with patient-level train/val/test splits (no data leakage).

Steps:
  1. Organise raw DICOMs into pylidc-compatible folder structure
  2. Configure pylidc to locate the data
  3. Patient-level split (70/15/15 by LIDC-IDRI-XXXX ID)
  4. Iterate over every scan, cluster radiologist annotations
  5. Isotropic resampling to 1×1×1 mm voxels (MONAI Spacing)
  6. Extract 64×64×64 3D crops with +10px padding on all 6 axes
  7. Build union segmentation masks from pylidc boolean_mask()
  8. Save raw HU volumes, masks, and metadata as .npy / .json

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
                           --out_dir /path/to/processed_3d
"""

import os
import json
import shutil
import argparse
import configparser
import importlib
import warnings
from collections import defaultdict

import cv2
import numpy as np
import pydicom
import pylidc as pl
from glob import glob
from scipy.ndimage import zoom

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

CROP_SIZE = 64          # target 3D crop size per axis
PAD_VOXELS = 10         # padding around nodule bbox on all 6 sides
TARGET_SPACING = (1.0, 1.0, 1.0)  # isotropic 1mm³ voxels


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


def load_volume_directly(scan, root: str) -> np.ndarray:
    """
    Bypass pylidc's internal path resolution and build a (H, W, Z) HU volume
    directly from DICOMs on disk.
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

        # Collect spatial info for resampling
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

    # Compute original spacing
    ps = pixel_spacings[0]
    sorted_z = sorted(slice_positions)
    if len(sorted_z) > 1:
        slice_thickness = np.median(np.diff(sorted_z))
    else:
        slice_thickness = float(getattr(pydicom.dcmread(dcm_files[0]),
                                         "SliceThickness", 1.0))

    original_spacing = (ps[0], ps[1], abs(slice_thickness))
    return vol, original_spacing


def resample_volume(vol: np.ndarray,
                    original_spacing: tuple,
                    target_spacing: tuple = TARGET_SPACING) -> tuple:
    """
    Resample a 3D volume to isotropic target spacing.

    Parameters
    ----------
    vol              : (H, W, Z) float32 volume
    original_spacing : (row_sp, col_sp, slice_sp) in mm
    target_spacing   : target spacing in mm (default: 1mm isotropic)

    Returns
    -------
    resampled_vol    : resampled volume
    scale_factors    : (sy, sx, sz) used for bbox coordinate transform
    """
    scale_factors = tuple(
        orig / tgt for orig, tgt in zip(original_spacing, target_spacing)
    )

    resampled = zoom(vol, scale_factors, order=1)  # bilinear interpolation
    return resampled, scale_factors


def resample_mask(mask: np.ndarray, scale_factors: tuple) -> np.ndarray:
    """Resample a binary mask using nearest-neighbour interpolation."""
    resampled = zoom(mask.astype(np.float32), scale_factors, order=0)
    return (resampled > 0.5).astype(np.bool_)


def extract_3d_crop(vol: np.ndarray,
                    center: tuple,
                    crop_size: int = CROP_SIZE,
                    pad: int = PAD_VOXELS) -> tuple:
    """
    Extract a 3D crop of size (crop_size, crop_size, crop_size) centred
    at `center`, with `pad` voxels of additional context.

    Returns (crop, start_coords) where start_coords can be used to
    map back to the original volume.
    """
    total_size = crop_size
    half = total_size // 2

    starts = []
    ends = []
    for dim, c in enumerate(center):
        s = max(0, int(c) - half)
        e = s + total_size
        # Clamp to volume bounds
        if e > vol.shape[dim]:
            e = vol.shape[dim]
            s = max(0, e - total_size)
        starts.append(s)
        ends.append(e)

    crop = vol[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # Zero-pad if crop is smaller than target (edge case)
    if crop.shape != (total_size, total_size, total_size):
        padded = np.zeros((total_size, total_size, total_size), dtype=crop.dtype)
        padded[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        crop = padded

    return crop, tuple(starts)


def build_union_mask(nodule_annotations, vol_shape, scan, raw_dir,
                     scale_factors=None) -> np.ndarray:
    """
    Build a union segmentation mask from all annotations for a nodule.

    Each annotation's boolean_mask() is placed at the correct position
    in the full volume, then all are OR-combined.
    """
    union_mask = np.zeros(vol_shape, dtype=np.bool_)

    for ann in nodule_annotations:
        try:
            bbox = ann.bbox()
            mask_3d = ann.boolean_mask()

            if mask_3d.sum() == 0:
                continue

            # Place the annotation mask in the full volume
            y_slice = slice(bbox[0].start, bbox[0].start + mask_3d.shape[0])
            x_slice = slice(bbox[1].start, bbox[1].start + mask_3d.shape[1])
            z_slice = slice(bbox[2].start, bbox[2].start + mask_3d.shape[2])

            # Clip to volume bounds
            y_end = min(y_slice.stop, vol_shape[0])
            x_end = min(x_slice.stop, vol_shape[1])
            z_end = min(z_slice.stop, vol_shape[2])

            my = y_end - y_slice.start
            mx = x_end - x_slice.start
            mz = z_end - z_slice.start

            union_mask[y_slice.start:y_end,
                       x_slice.start:x_end,
                       z_slice.start:z_end] |= mask_3d[:my, :mx, :mz]
        except Exception as e:
            print(f"    Warning: annotation mask failed: {e}")
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

    # If splits already exist, load them for reproducibility
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        print(f"Loaded existing patient splits from {splits_path}")
        print(f"  Train: {len(splits['train'])} | "
              f"Val: {len(splits['val'])} | "
              f"Test: {len(splits['test'])}")
        return splits

    # Collect unique patient IDs
    patient_ids = sorted(set(scan.patient_id for scan in scans))
    n = len(patient_ids)
    print(f"Total unique patients: {n}")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    train_n = int(train_ratio * n)
    val_n   = int(val_ratio * n)

    splits = {
        "train": patient_ids[:train_n],
        "val":   patient_ids[train_n:train_n + val_n],
        "test":  patient_ids[train_n + val_n:],
    }

    # Save for reproducibility
    os.makedirs(out_dir, exist_ok=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Patient splits saved → {splits_path}")
    print(f"  Train: {len(splits['train'])} | "
          f"Val: {len(splits['val'])} | "
          f"Test: {len(splits['test'])}")
    return splits


# ──────────────────────────────────────────────
#  Main extraction pipeline
# ──────────────────────────────────────────────

def extract_3d_nodule_volumes(raw_dir: str, out_dir: str,
                               min_annotators: int = 2) -> None:
    """
    Extract 3D nodule volumes and segmentation masks from every scan.

    For each nodule with ≥ min_annotators annotations:
      1. Load the full CT volume
      2. Resample to 1mm isotropic
      3. Build union segmentation mask
      4. Extract 64³ 3D crop centred on nodule
      5. Save volume, mask, and metadata

    A checkpoint file lets you safely resume after a crash.
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
    print(f"Total scans found: {len(scans)}\n")

    # Build patient-level splits
    splits = build_patient_splits(scans, out_dir)
    patient_to_split = {}
    for split_name, ids in splits.items():
        for pid in ids:
            patient_to_split[pid] = split_name

    nodule_count = 0
    skipped      = 0
    stats        = defaultdict(lambda: {"benign": 0, "malignant": 0})

    for scan_idx, scan in enumerate(scans):

        if scan.patient_id in processed_ids:
            continue

        try:
            nodules = scan.cluster_annotations()
            if not nodules:
                _checkpoint(checkpoint_file, processed_ids, scan.patient_id)
                continue

            # Load full volume with spacing info
            vol, original_spacing = load_volume_directly(scan, raw_dir)

            # Resample to isotropic 1mm³
            vol_resampled, scale_factors = resample_volume(vol, original_spacing)

            # Create patient output directory
            patient_dir = os.path.join(volumes_dir, scan.patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            for i, nodule in enumerate(nodules):

                # Skip nodules annotated by fewer than min_annotators
                if len(nodule) < min_annotators:
                    skipped += 1
                    continue

                scores         = [ann.malignancy for ann in nodule]
                avg_malignancy = sum(scores) / len(scores)

                # Skip ambiguous (exactly 3.0) nodules
                if avg_malignancy == 3.0:
                    skipped += 1
                    continue

                is_malignant = avg_malignancy > 3.0

                # Build union segmentation mask in original space
                union_mask_orig = build_union_mask(
                    nodule, vol.shape, scan, raw_dir
                )

                if union_mask_orig.sum() == 0:
                    skipped += 1
                    continue

                # Resample mask to match resampled volume
                union_mask = resample_mask(union_mask_orig, scale_factors)

                # Ensure mask shape matches volume (can differ by ±1 voxel)
                min_shape = tuple(
                    min(v, m) for v, m in
                    zip(vol_resampled.shape, union_mask.shape)
                )
                union_mask_aligned = union_mask[:min_shape[0],
                                                :min_shape[1],
                                                :min_shape[2]]
                vol_aligned = vol_resampled[:min_shape[0],
                                            :min_shape[1],
                                            :min_shape[2]]

                # Find nodule centroid in resampled space
                coords = np.argwhere(union_mask_aligned)
                if len(coords) == 0:
                    skipped += 1
                    continue
                centroid = coords.mean(axis=0).astype(int)

                # Extract 64³ crop
                vol_crop, crop_start = extract_3d_crop(
                    vol_aligned, centroid, CROP_SIZE, PAD_VOXELS
                )
                mask_crop, _ = extract_3d_crop(
                    union_mask_aligned.astype(np.float32),
                    centroid, CROP_SIZE, PAD_VOXELS
                )
                mask_crop = mask_crop > 0.5

                # Save files
                prefix = f"{scan.patient_id}_nodule_{i}"
                np.save(
                    os.path.join(patient_dir, f"{prefix}_vol.npy"),
                    vol_crop.astype(np.float32)
                )
                np.save(
                    os.path.join(patient_dir, f"{prefix}_mask.npy"),
                    mask_crop.astype(np.bool_)
                )

                # Save metadata
                split = patient_to_split.get(scan.patient_id, "unknown")
                meta = {
                    "patient_id": scan.patient_id,
                    "nodule_index": i,
                    "malignancy_scores": scores,
                    "avg_malignancy": avg_malignancy,
                    "is_malignant": is_malignant,
                    "label": int(is_malignant),
                    "split": split,
                    "crop_size": CROP_SIZE,
                    "original_spacing": list(original_spacing),
                    "target_spacing": list(TARGET_SPACING),
                    "centroid_resampled": centroid.tolist(),
                    "crop_start": list(crop_start),
                    "mask_voxels": int(mask_crop.sum()),
                    "n_annotators": len(nodule),
                }
                with open(os.path.join(patient_dir, f"{prefix}_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)

                nodule_count += 1
                label_str = "malignant" if is_malignant else "benign"
                stats[split][label_str] += 1

            _checkpoint(checkpoint_file, processed_ids, scan.patient_id)

            if (scan_idx + 1) % 20 == 0:
                print(f"  [{scan_idx+1}/{len(scans)}] "
                      f"{nodule_count} saved | {skipped} skipped")

        except KeyboardInterrupt:
            print("Interrupted — progress saved. Re-run to resume.")
            break

        except Exception as e:
            print(f"Error on {scan.patient_id}: {e}")

    print(f"\nDone!  Extracted: {nodule_count}  |  Skipped: {skipped}")
    print(f"\nPer-split class counts:")
    for split in ["train", "val", "test"]:
        s = stats[split]
        print(f"  {split:6s}: benign={s['benign']:4d} | "
              f"malignant={s['malignant']:4d} | "
              f"total={s['benign']+s['malignant']:4d}")


def _checkpoint(path, id_set, patient_id):
    with open(path, "a") as f:
        f.write(patient_id + "\n")
    id_set.add(patient_id)


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LIDC-IDRI 3D preprocessing: extract 64³ volumes + seg masks"
    )
    p.add_argument("--raw_dir",  required=True,
                   help="Root directory containing raw LIDC-IDRI DICOM data")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for processed 3D volumes")
    p.add_argument("--min_ann",  type=int, default=2,
                   help="Minimum radiologist annotations required (default: 2)")
    p.add_argument("--skip_organise", action="store_true",
                   help="Skip the folder reorganisation step")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_organise:
        organise_dicom_folders(args.raw_dir)

    configure_pylidc(args.raw_dir)
    extract_3d_nodule_volumes(args.raw_dir, args.out_dir,
                               min_annotators=args.min_ann)
