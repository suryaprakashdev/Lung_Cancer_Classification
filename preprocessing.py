"""
preprocessing_3d.py
-------------------
Stage 1: Convert raw LIDC-IDRI DICOM series → 3D nodule crops (.npy)

What this script does
---------------------
  1. Queries pylidc for all scans and serialises nodule metadata to plain
     Python dicts (fixes multiprocessing pickle crash)
  2. Loads each patient's DICOM volume — handles all known LIDC-IDRI folder
     structures (flat, nested, extensionless)
  3. Resamples to 1×1×1 mm isotropic spacing via scipy.ndimage.zoom
     (fixes MONAI Spacing API incompatibility)
  4. Extracts 64×64×64 nodule crops + binary segmentation masks
  5. Saves under a patient-level directory structure ready for
     patient-level train/test splitting

Output layout
-------------
  <out_dir>/
    volumes/
      LIDC-IDRI-0001/
        Benign_0/
          LIDC-IDRI-0001_nodule0_malig2.0_vol.npy    ← (64,64,64) HU crop
          LIDC-IDRI-0001_nodule0_malig2.0_mask.npy   ← (64,64,64) bool mask
        Malignant_1/
          LIDC-IDRI-0001_nodule1_malig4.5_vol.npy
          LIDC-IDRI-0001_nodule1_malig4.5_mask.npy
      LIDC-IDRI-0002/
        ...
    dataset_summary.json    ← per-patient stats, class balance, spacing info
    checkpoint.txt          ← processed patient IDs (resume support)

Key fixes vs previous version
------------------------------
  ✓ Pickle fix      — all pylidc ORM objects serialised to dicts before
                       passing to worker processes
  ✓ Scipy resampling — replaces MONAI Spacing (affine API changed in MONAI>=1.3)
  ✓ Robust DICOM loader — handles flat / nested / extensionless file structures
  ✓ Malignancy label — ambiguous (avg==3.0) nodules are skipped
  ✓ Min annotators  — nodules with <2 radiologist annotations are skipped
  ✓ Resume support  — re-run safely after crash; already-done patients skipped
  ✓ Error surfacing — first 5 errors printed in full, all errors logged to JSON

Usage
-----
  python preprocessing_3d.py \
      --raw_dir  /content/drive/MyDrive/LIDC-IDRI \
      --out_dir  /content/drive/MyDrive/LIDC-3D-Processed \
      --workers  4

  # Resume after crash (same command — checkpoint.txt handles it)
  python preprocessing_3d.py \
      --raw_dir  /content/drive/MyDrive/LIDC-IDRI \
      --out_dir  /content/drive/MyDrive/LIDC-3D-Processed \
      --workers  4
"""

import os
import sys
import json
import argparse
import configparser
import multiprocessing as mp
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
import pylidc as pl
from scipy.ndimage import zoom as scipy_zoom
from monai.transforms import SpatialCrop, ResizeWithPadOrCrop
import torch


# ──────────────────────────────────────────────
#  Pylidc compatibility patches
#  (required for pylidc < 0.2.3 on NumPy >= 1.24)
# ──────────────────────────────────────────────
np.int   = int
np.float = float
np.bool  = bool
configparser.SafeConfigParser = configparser.ConfigParser


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
CROP_SIZE       = 64                  # output voxels per axis
TARGET_SPACING  = (1.0, 1.0, 1.0)    # mm — isotropic target
MIN_ANNOTATORS  = 2                   # skip nodules with fewer radiologists
IGNORE_DIRS     = {                   # preprocessing outputs — never scan these
    "Benign_0", "Malignant_1", "volumes",
    "processed_images", "checkpoints", "results",
    "__pycache__", ".ipynb_checkpoints",
}


# ──────────────────────────────────────────────
#  Resampling  (scipy — no MONAI Spacing API issues)
# ──────────────────────────────────────────────

def resample_volume(vol: np.ndarray, spacing: Tuple) -> np.ndarray:
    """
    Resample a float32 HU volume to 1×1×1 mm isotropic.

    Parameters
    ----------
    vol     : (H, W, D) float32 HU array
    spacing : (x_mm, y_mm, z_mm) original voxel size

    Returns
    -------
    (H', W', D') float32 resampled array
    """
    sx, sy, sz = spacing
    tx, ty, tz = TARGET_SPACING

    # zoom_factors must match (H, W, D) axis order
    zoom_factors = (sy / ty, sx / tx, sz / tz)

    return scipy_zoom(
        vol.astype(np.float32),
        zoom=zoom_factors,
        order=1,          # bilinear interpolation for continuous HU values
        prefilter=False,  # skip Gaussian pre-smoothing for speed
    ).astype(np.float32)


def resample_mask(mask: np.ndarray, spacing: Tuple) -> np.ndarray:
    """
    Resample a binary segmentation mask to 1×1×1 mm.
    Uses nearest-neighbour (order=0) to preserve binary values.
    """
    sx, sy, sz = spacing
    tx, ty, tz = TARGET_SPACING
    zoom_factors = (sy / ty, sx / tx, sz / tz)

    return scipy_zoom(
        mask.astype(np.float32),
        zoom=zoom_factors,
        order=0,          # nearest-neighbour — critical for binary masks
        prefilter=False,
    ) > 0.5


# ──────────────────────────────────────────────
#  3-D crop  (MONAI SpatialCrop + ResizeWithPadOrCrop)
# ──────────────────────────────────────────────

def crop_3d(vol: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Extract a CROP_SIZE³ region centred on `center` from `vol`.
    Uses ResizeWithPadOrCrop to guarantee exact output size even when
    the centre is near the volume boundary.

    Parameters
    ----------
    vol    : (H, W, D) float32 numpy array
    center : (3,) int array  [y, x, z]

    Returns
    -------
    (CROP_SIZE, CROP_SIZE, CROP_SIZE) float32 array
    """
    half   = CROP_SIZE // 2
    starts = [max(0, int(c) - half) for c in center]
    ends   = [min(vol.shape[i], starts[i] + CROP_SIZE) for i in range(3)]
    # Re-anchor start if end hit boundary
    starts = [max(0, ends[i] - CROP_SIZE) for i in range(3)]

    tensor  = torch.from_numpy(vol).unsqueeze(0)   # (1, H, W, D)
    cropped = SpatialCrop(roi_start=starts, roi_end=ends)(tensor)
    padded  = ResizeWithPadOrCrop(
        (CROP_SIZE, CROP_SIZE, CROP_SIZE)
    )(cropped)

    return padded.squeeze(0).numpy()


# ──────────────────────────────────────────────
#  DICOM loader  (handles all LIDC-IDRI structures)
# ──────────────────────────────────────────────

def load_volume(
    patient_id: str,
    series_uid: str,
    root: str,
) -> Tuple[np.ndarray, Tuple]:
    """
    Load a DICOM CT series and return a (H, W, D) HU volume + spacing.

    Handles three known LIDC-IDRI folder structures:
      A  root/PatientID/SeriesUID/*.dcm            (TCIA direct download)
      B  root/PatientID/StudyUID/SeriesUID/*.dcm   (TCIA nested)
      C  root/PatientID/**/*.dcm                   (any depth, filter by UID tag)

    Returns
    -------
    vol     : (H, W, D) float32 HU array, sorted inferior→superior
    spacing : (x_mm, y_mm, z_mm) tuple of physical voxel size
    """

    # ── Structure A: direct ───────────────────
    direct_path = os.path.join(root, patient_id, series_uid)
    files = sorted(glob(os.path.join(direct_path, "*.dcm")))

    # ── Structure B: one extra nesting level ──
    if not files:
        files = sorted(glob(
            os.path.join(root, patient_id, "*", series_uid, "*.dcm")
        ))

    # ── Structure C: recursive + UID tag filter ──
    if not files:
        all_dcms = sorted(glob(
            os.path.join(root, patient_id, "**", "*.dcm"),
            recursive=True,
        ))

        # Some LIDC-IDRI files have no extension
        if not all_dcms:
            for dirpath, dirnames, fnames in os.walk(
                os.path.join(root, patient_id)
            ):
                dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
                for fname in fnames:
                    if "." not in fname:
                        all_dcms.append(os.path.join(dirpath, fname))

        # Filter to correct series by tag
        if all_dcms:
            matched_dir = None
            for f in all_dcms[:10]:
                try:
                    ds = pydicom.dcmread(f, stop_before_pixels=True)
                    if str(getattr(ds, "SeriesInstanceUID", "")) == series_uid:
                        matched_dir = os.path.dirname(f)
                        break
                except Exception:
                    continue

            if matched_dir:
                files = sorted([
                    f for f in all_dcms
                    if os.path.dirname(f) == matched_dir
                ])
            else:
                # Last resort — use all DICOMs found under patient folder
                files = all_dcms

    if not files:
        raise FileNotFoundError(
            f"No DICOM files found for {patient_id} / ...{series_uid[-20:]}. "
            f"Searched under: {os.path.join(root, patient_id)}"
        )

    # ── Decode slices + convert to HU ─────────
    slices: List[Tuple[float, np.ndarray]] = []
    z_positions: List[float] = []

    for f in files:
        try:
            ds  = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            hu  = (img
                   * float(getattr(ds, "RescaleSlope",     1))
                   + float(getattr(ds, "RescaleIntercept", 0)))

            # Prefer ImagePositionPatient z; fall back to InstanceNumber
            if hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
            else:
                z = float(getattr(ds, "InstanceNumber", len(slices)))

            slices.append((z, hu))
            z_positions.append(z)
        except Exception:
            continue  # skip unreadable / non-image DICOM files

    if not slices:
        raise ValueError(
            f"Could not decode any DICOM slices for {patient_id}. "
            f"Files found: {len(files)}"
        )

    # Sort inferior → superior
    slices.sort(key=lambda t: t[0])
    vol = np.stack([s[1] for s in slices], axis=-1)   # (H, W, D)

    # ── Compute physical spacing ───────────────
    sorted_z = sorted(z_positions)
    if len(sorted_z) > 1:
        diffs     = np.abs(np.diff(sorted_z))
        z_spacing = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
    else:
        z_spacing = 1.0

    xy_spacing = (1.0, 1.0)
    for f in files[:5]:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            ps = getattr(ds, "PixelSpacing", None)
            if ps and len(ps) >= 2:
                xy_spacing = (float(ps[0]), float(ps[1]))
                break
        except Exception:
            continue

    spacing = (*xy_spacing, z_spacing)   # (x_mm, y_mm, z_mm)
    return vol, spacing


# ──────────────────────────────────────────────
#  Segmentation mask builder  (plain dicts — no ORM)
# ──────────────────────────────────────────────

def build_mask_from_dict(
    annotations: List[Dict],
    vol_shape: Tuple,
) -> np.ndarray:
    """
    Reconstruct a 3D binary mask from serialised annotation dicts.
    Takes the union of all radiologist masks (consensus).

    Parameters
    ----------
    annotations : list of dicts with keys "bbox" and "mask"
    vol_shape   : (H, W, D) shape of the full unresampled volume

    Returns
    -------
    (H, W, D) bool array
    """
    mask = np.zeros(vol_shape, dtype=np.bool_)

    for ann in annotations:
        (y0, y1), (x0, x1), (z0, z1) = ann["bbox"]
        m = ann["mask"]   # (mH, mW, mD) bool

        # Clip destination region to volume bounds
        y1c = min(y0 + m.shape[0], vol_shape[0])
        x1c = min(x0 + m.shape[1], vol_shape[1])
        z1c = min(z0 + m.shape[2], vol_shape[2])

        mask[y0:y1c, x0:x1c, z0:z1c] |= m[
            : y1c - y0,
            : x1c - x0,
            : z1c - z0,
        ]

    return mask


# ──────────────────────────────────────────────
#  Metadata preparation  (main process only)
# ──────────────────────────────────────────────

def prepare_metadata(
    scans: List,
    checkpoint_ids: set,
) -> List[Dict]:
    """
    Iterate every pylidc Scan, cluster annotations, and serialise
    all nodule data to plain Python dicts safe for multiprocessing.

    This must run in the MAIN process where pylidc SQLAlchemy is live.
    Workers receive only serialisable primitives (dicts + numpy arrays).

    Parameters
    ----------
    scans          : list of pylidc.Scan objects
    checkpoint_ids : patient IDs already processed (resume support)

    Returns
    -------
    List of patient dicts ready to pass to worker()
    """
    prepared      = []
    total_nodules = 0

    print(f"Preparing metadata for {len(scans)} scans...")
    print("(Warnings about >4 annotations are normal — pylidc handles them)\n")

    for i, scan in enumerate(scans):

        pid = scan.patient_id

        # Resume: skip already-processed patients
        if pid in checkpoint_ids:
            continue

        try:
            nodule_groups = scan.cluster_annotations()
        except Exception as e:
            print(f"  [WARN] cluster_annotations failed for {pid}: {e}")
            continue

        if not nodule_groups:
            continue

        serialised_nodules = []

        for nodule in nodule_groups:

            # Skip under-annotated nodules
            if len(nodule) < MIN_ANNOTATORS:
                continue

            annotations    = []
            mal_scores     = []

            for ann in nodule:
                try:
                    bbox = ann.bbox()
                    mask = ann.boolean_mask()  # (mH, mW, mD) bool
                    mal_scores.append(int(ann.malignancy))

                    annotations.append({
                        # Store bbox as plain int pairs — slice objects not picklable
                        "bbox": [
                            [int(bbox[0].start), int(bbox[0].stop)],
                            [int(bbox[1].start), int(bbox[1].stop)],
                            [int(bbox[2].start), int(bbox[2].stop)],
                        ],
                        "mask": mask.astype(np.bool_),  # numpy array — picklable
                    })
                except Exception:
                    continue

            if not annotations or not mal_scores:
                continue

            avg_malignancy = sum(mal_scores) / len(mal_scores)

            # Skip ambiguous nodules (radiologists evenly split 3.0)
            if avg_malignancy == 3.0:
                continue

            serialised_nodules.append({
                "annotations":     annotations,
                "avg_malignancy":  round(avg_malignancy, 2),
                "is_malignant":    avg_malignancy > 3.0,
                "n_annotators":    len(nodule),
            })

        if not serialised_nodules:
            continue

        prepared.append({
            "patient_id": pid,
            "series_uid": scan.series_instance_uid,
            "nodules":    serialised_nodules,
        })
        total_nodules += len(serialised_nodules)

        if (i + 1) % 100 == 0:
            print(
                f"  {i+1}/{len(scans)} scans | "
                f"{len(prepared)} with nodules | "
                f"{total_nodules} nodules serialised"
            )

    print(
        f"\nReady: {len(prepared)} patients | "
        f"{total_nodules} nodules to process\n"
    )
    return prepared


# ──────────────────────────────────────────────
#  Worker  (runs in subprocess — no pylidc)
# ──────────────────────────────────────────────

def worker(item: Dict, raw_dir: str, out_dir: str) -> Dict:
    """
    Process one patient: load DICOM → resample → extract crops → save.

    Receives only plain-Python/numpy data — no SQLAlchemy objects.
    All exceptions are caught and returned in the result dict so the
    main process can log them without crashing the pool.
    """
    pid     = item["patient_id"]
    uid     = item["series_uid"]
    nodules = item["nodules"]

    result = {
        "patient_id": pid,
        "saved":      0,
        "skipped":    0,
        "error":      None,
        "spacing":    None,
        "vol_shape":  None,
    }

    try:
        # ── 1. Load DICOM volume ──────────────────────────────
        vol, spacing = load_volume(pid, uid, raw_dir)
        result["spacing"]   = spacing
        result["vol_shape"] = vol.shape

        # ── 2. Resample full volume to isotropic spacing ──────
        vol_resampled = resample_volume(vol, spacing)

        # ── 3. Process each nodule ────────────────────────────
        for i, nodule in enumerate(nodules):

            annotations    = nodule["annotations"]
            is_malignant   = nodule["is_malignant"]
            avg_malignancy = nodule["avg_malignancy"]

            # Build union mask in ORIGINAL voxel space
            mask_orig = build_mask_from_dict(annotations, vol.shape)
            if mask_orig.sum() == 0:
                result["skipped"] += 1
                continue

            # Resample mask to isotropic space
            mask_resampled = resample_mask(mask_orig, spacing)

            # Find nodule centre in resampled space
            coords = np.argwhere(mask_resampled)
            if len(coords) == 0:
                result["skipped"] += 1
                continue

            center = coords.mean(axis=0).astype(int)   # [y, x, z]

            # ── 4. Extract 64³ crops ──────────────────────────
            vol_crop  = crop_3d(vol_resampled,                       center)
            mask_crop = crop_3d(mask_resampled.astype(np.float32),   center) > 0.5

            # ── 5. Save ───────────────────────────────────────
            label     = "Malignant_1" if is_malignant else "Benign_0"
            prefix    = f"{pid}_nodule{i}_malig{avg_malignancy:.1f}"
            label_dir = os.path.join(out_dir, "volumes", pid, label)
            os.makedirs(label_dir, exist_ok=True)

            np.save(os.path.join(label_dir, f"{prefix}_vol.npy"),  vol_crop)
            np.save(os.path.join(label_dir, f"{prefix}_mask.npy"), mask_crop.astype(np.bool_))

            result["saved"] += 1

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    return result


# ──────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────

def load_checkpoint(out_dir: str) -> set:
    path = os.path.join(out_dir, "checkpoint.txt")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"Resuming — {len(ids)} patients already done (loaded from checkpoint.txt)")
    return ids


def save_checkpoint(out_dir: str, patient_id: str):
    path = os.path.join(out_dir, "checkpoint.txt")
    with open(path, "a") as f:
        f.write(patient_id + "\n")


# ──────────────────────────────────────────────
#  Summary + class balance report
# ──────────────────────────────────────────────

def print_class_balance(out_dir: str):
    """Walk output tree and count Benign_0 / Malignant_1 crops."""
    benign    = 0
    malignant = 0
    volumes_dir = os.path.join(out_dir, "volumes")

    for pid in os.listdir(volumes_dir):
        pid_path = os.path.join(volumes_dir, pid)
        if not os.path.isdir(pid_path):
            continue
        for label in ["Benign_0", "Malignant_1"]:
            label_path = os.path.join(pid_path, label)
            if not os.path.isdir(label_path):
                continue
            count = len([
                f for f in os.listdir(label_path)
                if f.endswith("_vol.npy")
            ])
            if label == "Benign_0":
                benign    += count
            else:
                malignant += count

    total = benign + malignant
    if total == 0:
        print("No crops found.")
        return

    print("\n" + "=" * 45)
    print("  Dataset class balance")
    print("=" * 45)
    print(f"  Benign    : {benign:5d}  ({100*benign/total:.1f}%)")
    print(f"  Malignant : {malignant:5d}  ({100*malignant/total:.1f}%)")
    print(f"  Total     : {total:5d}")
    if malignant > 0:
        print(f"  Ratio     : {benign/malignant:.2f} : 1  (benign:malignant)")
    print("=" * 45)


def write_summary(out_dir: str, all_results: List[Dict]):
    """Write per-patient stats to dataset_summary.json."""
    summary = {
        "total_patients":   len(all_results),
        "total_saved":      sum(r["saved"]   for r in all_results),
        "total_skipped":    sum(r["skipped"] for r in all_results),
        "total_errors":     sum(1 for r in all_results if r["error"]),
        "patients":         [
            {
                "patient_id": r["patient_id"],
                "saved":      r["saved"],
                "skipped":    r["skipped"],
                "spacing":    r["spacing"],
                "vol_shape":  r["vol_shape"],
                "error":      r["error"] is not None,
            }
            for r in all_results
        ],
    }
    path = os.path.join(out_dir, "dataset_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {path}")


# ──────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────

def run(raw_dir: str, out_dir: str, num_workers: int):
    os.makedirs(out_dir, exist_ok=True)

    # Load resume checkpoint
    checkpoint_ids = load_checkpoint(out_dir)

    # Query pylidc — must happen in main process
    print("Querying pylidc database...")
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans in database\n")

    # Serialise all pylidc data to plain dicts (main process, SQLAlchemy live)
    prepared = prepare_metadata(scans, checkpoint_ids)

    if not prepared:
        print("Nothing to process — all patients already in checkpoint.")
        print_class_balance(out_dir)
        return

    print(f"Launching {num_workers} worker processes...\n")

    all_results  = []
    total_saved  = 0
    total_skipped = 0
    errors       = 0
    error_log    = []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:

        futures = {
            pool.submit(worker, item, raw_dir, out_dir): item["patient_id"]
            for item in prepared
        }

        for i, future in enumerate(as_completed(futures)):
            pid = futures[future]

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "patient_id": pid,
                    "saved":      0,
                    "skipped":    0,
                    "error":      f"Future crashed: {e}\n{traceback.format_exc()}",
                    "spacing":    None,
                    "vol_shape":  None,
                }

            all_results.append(result)

            if result["error"]:
                errors += 1
                error_log.append({
                    "patient_id": pid,
                    "error":      result["error"],
                })
                # Print first 5 errors in full so you can see what's wrong
                if errors <= 5:
                    print(f"\n[ERROR #{errors} — {pid}]")
                    print(result["error"])
                    print()
            else:
                total_saved   += result["saved"]
                total_skipped += result["skipped"]
                # Mark patient as done in checkpoint
                save_checkpoint(out_dir, pid)

            # Progress report every 25 patients
            if (i + 1) % 25 == 0 or (i + 1) == len(prepared):
                print(
                    f"[{i+1:4d}/{len(prepared)}] "
                    f"saved={total_saved:5d} | "
                    f"skipped={total_skipped:4d} | "
                    f"errors={errors:3d}"
                )

    # ── Final report ──────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  PREPROCESSING COMPLETE")
    print("=" * 45)
    print(f"  Patients processed : {len(prepared)}")
    print(f"  Nodule crops saved : {total_saved}")
    print(f"  Nodules skipped    : {total_skipped}")
    print(f"  Patients errored   : {errors}")

    print_class_balance(out_dir)
    write_summary(out_dir, all_results)

    # Save error log
    if error_log:
        err_path = os.path.join(out_dir, "error_log.json")
        with open(err_path, "w") as f:
            json.dump(error_log, f, indent=2)
        print(f"  Error log saved    → {err_path}")
        print(f"\n  To retry failed patients, re-run the same command.")
        print(f"  Successful patients are checkpointed and will be skipped.")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LIDC-IDRI 3D preprocessing — DICOM → 64³ isotropic nodule crops"
    )
    p.add_argument(
        "--raw_dir", required=True,
        help="Root directory of raw LIDC-IDRI DICOM files",
    )
    p.add_argument(
        "--out_dir", required=True,
        help="Output directory for processed .npy crops",
    )
    p.add_argument(
        "--workers", type=int,
        default=max(1, mp.cpu_count() - 1),
        help="Number of parallel worker processes (default: cpu_count - 1)",
    )
    return p.parse_args()


if __name__ == "__main__":
    # Required for ProcessPoolExecutor on macOS/Windows with spawn start method
    mp.set_start_method("spawn", force=True)

    args = parse_args()

    print("=" * 55)
    print("  LIDC-IDRI 3D Preprocessing Pipeline")
    print("=" * 55)
    print(f"  Raw DICOM dir : {args.raw_dir}")
    print(f"  Output dir    : {args.out_dir}")
    print(f"  Workers       : {args.workers}")
    print(f"  Crop size     : {CROP_SIZE}³ voxels")
    print(f"  Target spacing: {TARGET_SPACING} mm (isotropic)")
    print(f"  Min annotators: {MIN_ANNOTATORS}")
    print("=" * 55 + "\n")

    run(args.raw_dir, args.out_dir, args.workers)