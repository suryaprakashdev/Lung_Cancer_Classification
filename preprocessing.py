"""
preprocessing.py
----------------
Stage 1: Convert raw LIDC-IDRI DICOM files → labelled .npy patches.

Saves raw float32 HU-calibrated patches so that windowing / normalisation
can happen at training time via MONAI transforms (no quantisation loss).

Steps:
  1. Organise raw DICOMs into pylidc-compatible folder structure
  2. Configure pylidc to locate the data
  3. Iterate over every scan, cluster radiologist annotations into nodules
  4. Crop a 2-D patch, save raw HU values to
     Benign_0/ or Malignant_1/ directories as .npy files

Run:
  python preprocessing.py --raw_dir /path/to/LIDC-IDRI \
                           --out_dir /path/to/processed_images
"""

import os
import shutil
import argparse
import configparser
import importlib
import warnings

import cv2
import numpy as np
import pydicom
import pylidc as pl
from glob import glob


# ──────────────────────────────────────────────
#  Monkey-patches required by older pylidc
# ──────────────────────────────────────────────
np.int   = int
np.float = float
np.bool  = bool
configparser.SafeConfigParser = configparser.ConfigParser


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def apply_lung_window(image: np.ndarray,
                      window_center: int = -600,
                      window_width:  int = 1500) -> np.ndarray:
    """Clip to lung HU range then normalise to 0-255."""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed = np.clip(image, img_min, img_max)
    normalised = (windowed - img_min) / (img_max - img_min) * 255.0
    return normalised.astype(np.uint8)


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
    for f in dcm_files:
        ds       = pydicom.dcmread(f)
        img      = ds.pixel_array.astype(np.float32)
        slope    = float(getattr(ds, "RescaleSlope",     1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slices.append((float(ds.InstanceNumber), img * slope + intercept))

    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=-1)   # (H, W, Z)


def organise_dicom_folders(raw_dir: str) -> None:
    """
    Move unorganised DICOM series folders under their PatientID parent so
    pylidc can resolve paths ( root/<PatientID>/<SeriesUID>/*.dcm ).
    """
    skip = {"Benign_0", "Malignant_1", "processed_images"}
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


# ──────────────────────────────────────────────
#  Main extraction pipeline
# ──────────────────────────────────────────────

def extract_nodule_patches(raw_dir: str, out_dir: str,
                            pad: int = 10,
                            min_annotators: int = 2) -> None:
    """
    Extract 2-D nodule patches from every scan and save them under
    out_dir/Benign_0/  and  out_dir/Malignant_1/.

    A checkpoint file lets you safely resume after a crash.
    """
    benign_dir      = os.path.join(out_dir, "Benign_0")
    malignant_dir   = os.path.join(out_dir, "Malignant_1")
    checkpoint_file = os.path.join(out_dir, "processed_scans_checkpoint.txt")

    os.makedirs(benign_dir,    exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

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

    nodule_count = 0
    skipped      = 0

    for scan_idx, scan in enumerate(scans):

        if scan.patient_id in processed_ids:
            continue

        try:
            nodules = scan.cluster_annotations()
            if not nodules:
                _checkpoint(checkpoint_file, processed_ids, scan.patient_id)
                continue

            vol = load_volume_directly(scan, raw_dir)

            for i, nodule in enumerate(nodules):

                # Skip nodules annotated by fewer than `min_annotators` radiologists
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

                # Use the annotation with the largest bounding box
                annotation = max(nodule, key=lambda a: a.bbox_matrix().size)
                bbox       = annotation.bbox()
                mask3d     = annotation.boolean_mask()

                if mask3d.sum() == 0:
                    skipped += 1
                    continue

                z_center          = int(mask3d.shape[2] / 2)
                z_index_in_volume = int(
                    np.clip(bbox[2].start + z_center, 0, vol.shape[2] - 1)
                )
                image_2d = vol[:, :, z_index_in_volume]

                y_min = max(0, bbox[0].start - pad)
                y_max = min(image_2d.shape[0], bbox[0].stop  + pad)
                x_min = max(0, bbox[1].start - pad)
                x_max = min(image_2d.shape[1], bbox[1].stop  + pad)

                cropped = image_2d[y_min:y_max, x_min:x_max]
                if cropped.size == 0:
                    skipped += 1
                    continue

                # Save raw HU patch — windowing happens at training
                # time via MONAI transforms (no quantisation loss)
                folder   = malignant_dir if is_malignant else benign_dir
                filename = f"{scan.patient_id}_nodule_{i}_malig_{avg_malignancy:.1f}.npy"
                np.save(os.path.join(folder, filename), cropped.astype(np.float32))
                nodule_count += 1

            _checkpoint(checkpoint_file, processed_ids, scan.patient_id)

            if (scan_idx + 1) % 50 == 0:
                print(f"  [{scan_idx+1}/{len(scans)}] "
                      f"{nodule_count} saved | {skipped} skipped")

        except KeyboardInterrupt:
            print("Interrupted — progress saved. Re-run to resume.")
            break

        except Exception as e:
            print(f"Error on {scan.patient_id}: {e}")

    print(f"\nDone!  Extracted: {nodule_count}  |  Skipped: {skipped}")
    print(f"  Benign    → {benign_dir}")
    print(f"  Malignant → {malignant_dir}")

    _print_class_balance(benign_dir, malignant_dir)


def _checkpoint(path, id_set, patient_id):
    with open(path, "a") as f:
        f.write(patient_id + "\n")
    id_set.add(patient_id)


def _print_class_balance(benign_dir, malignant_dir):
    b = len(os.listdir(benign_dir))
    m = len(os.listdir(malignant_dir))
    t = b + m
    print(f"\nClass balance:")
    print(f"  Benign:    {b}  ({100*b/t:.1f}%)")
    print(f"  Malignant: {m}  ({100*m/t:.1f}%)")
    print(f"  Ratio:     {b/m:.2f}:1")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LIDC-IDRI preprocessing")
    p.add_argument("--raw_dir",  required=True,
                   help="Root directory containing raw LIDC-IDRI DICOM data")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for processed PNG patches")
    p.add_argument("--pad",      type=int, default=10,
                   help="Pixel padding around nodule bounding box (default: 10)")
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
    extract_nodule_patches(args.raw_dir, args.out_dir,
                           pad=args.pad,
                           min_annotators=args.min_ann)
