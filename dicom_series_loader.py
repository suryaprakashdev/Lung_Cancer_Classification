"""
dicom_series_loader.py
----------------------
Load, sort, and selectively sample slices from a DICOM CT series folder.

A CT series is a 3-D volume stored as N individual .dcm files — one per axial
slice.  This module normalises that into an ordered stack and exposes pluggable
slice-selection strategies so the downstream 2-D inference pipeline only sees
the slices it actually needs.

Design principles
-----------------
* Zero coupling to the inference pipeline — this module knows nothing about
  models or predictions.
* Sorting priority: ImagePositionPatient[2] (z-coordinate) > InstanceNumber
  > filename.  The first guarantees physical ordering even when acquisition
  order differs from instance numbering.
* Memory-efficient: pixel arrays are loaded lazily (only selected slices are
  fully decoded).

Typical usage
-------------
    loader = DICOMSeriesLoader()
    series = loader.load_series("/path/to/patient_folder")

    # Select 10 evenly-spaced slices from the middle third of the volume
    slices = series.select_slices("middle", n=10)

    for s in slices:
        hu = s.hu_array       # (H, W) float32 in Hounsfield Units
        meta = s.metadata
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Data containers
# ──────────────────────────────────────────────

@dataclass
class DICOMSlice:
    """
    A single decoded CT slice with HU pixel data and DICOM metadata.

    Attributes
    ----------
    hu_array        : (H, W) float32 array in Hounsfield Units
    slice_index     : position in the sorted stack (0 = most inferior)
    z_position      : physical z-coordinate (mm) from ImagePositionPatient
    instance_number : DICOM InstanceNumber tag (used as fallback sort key)
    file_path       : absolute path to the source .dcm file
    metadata        : dictionary of DICOM tags useful downstream
    """

    hu_array: np.ndarray
    slice_index: int
    z_position: float
    instance_number: int
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.hu_array.shape  # type: ignore[return-value]


@dataclass
class DICOMSeries:
    """
    An ordered stack of DICOMSlice objects representing a single CT volume.

    The stack is sorted inferior → superior (increasing z).

    Attributes
    ----------
    patient_id      : PatientID tag (empty string if absent)
    study_date      : StudyDate tag
    series_uid      : SeriesInstanceUID
    folder_path     : source directory
    slices          : sorted list of DICOMSlice objects (already decoded)
    pixel_spacing   : (row_spacing, col_spacing) in mm
    slice_thickness : mm
    """

    patient_id: str
    study_date: str
    series_uid: str
    folder_path: str
    slices: List[DICOMSlice]
    pixel_spacing: Tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0

    # ── Slice selection ───────────────────────

    def select_slices(
        self,
        strategy: str = "all",
        n: Optional[int] = None,
        roi_mask: Optional[np.ndarray] = None,
    ) -> List[DICOMSlice]:
        """
        Return a subset of slices according to the chosen strategy.

        Parameters
        ----------
        strategy : one of {"all", "middle", "uniform", "roi"}
            all     — every slice in the volume
            middle  — the central ``n`` slices (middle third if n is None)
            uniform — ``n`` evenly-spaced slices across the full stack
            roi     — slices whose mean HU within ``roi_mask`` exceeds
                      a density threshold; falls back to "uniform" if no mask

        n        : number of slices to select (strategy-dependent)
        roi_mask : (H, W) uint8 binary mask used by the "roi" strategy

        Returns
        -------
        Ordered list of DICOMSlice objects (subset of self.slices)
        """
        total = len(self.slices)
        if total == 0:
            return []

        strategy = strategy.lower()

        if strategy == "all":
            selected = self.slices

        elif strategy == "middle":
            selected = self._select_middle(n)

        elif strategy == "uniform":
            k = n or max(1, total // 5)
            selected = self._select_uniform(k)

        elif strategy == "roi":
            if roi_mask is not None:
                selected = self._select_roi(roi_mask, n)
            else:
                logger.warning(
                    "strategy='roi' requested but no roi_mask supplied. "
                    "Falling back to 'uniform'."
                )
                k = n or max(1, total // 5)
                selected = self._select_uniform(k)

        else:
            raise ValueError(
                f"Unknown slice selection strategy '{strategy}'. "
                "Choose from: all | middle | uniform | roi"
            )

        logger.info(
            "Slice selection | strategy=%s | total=%d | selected=%d",
            strategy,
            total,
            len(selected),
        )
        return selected

    # ── Internal selection methods ─────────────

    def _select_middle(self, n: Optional[int]) -> List[DICOMSlice]:
        """Return the central n slices.  Defaults to the middle third."""
        total = len(self.slices)
        if n is None:
            # Middle third of the volume
            start = total // 3
            end = 2 * total // 3
        else:
            center = total // 2
            half = n // 2
            start = max(0, center - half)
            end = min(total, start + n)
            # Shift left if we hit the end
            if end == total:
                start = max(0, total - n)
        return self.slices[start:end]

    def _select_uniform(self, k: int) -> List[DICOMSlice]:
        """Return k evenly-spaced slices across the full stack."""
        total = len(self.slices)
        k = min(k, total)
        indices = np.linspace(0, total - 1, k, dtype=int)
        return [self.slices[i] for i in indices]

    def _select_roi(
        self,
        roi_mask: np.ndarray,
        n: Optional[int],
    ) -> List[DICOMSlice]:
        """
        Select slices where the mean HU inside ``roi_mask`` is highest
        (most dense tissue — likely nodule-bearing slices).

        The mask is expected at the same (H, W) as each slice.
        If dimensions differ, the mask is resized via nearest-neighbour.
        """
        import cv2  # lazy import — not needed for other strategies

        density_threshold = -100.0  # HU: soft tissue / nodule threshold

        scores: List[Tuple[float, DICOMSlice]] = []
        for sl in self.slices:
            h, w = sl.hu_array.shape
            # Resize mask if necessary
            if roi_mask.shape != (h, w):
                mask_resized = cv2.resize(
                    roi_mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask_resized = roi_mask.astype(np.uint8)

            masked_hu = sl.hu_array[mask_resized > 0]
            if masked_hu.size == 0:
                continue
            # Score = fraction of voxels denser than threshold
            score = float(np.mean(masked_hu > density_threshold))
            scores.append((score, sl))

        if not scores:
            logger.warning("ROI strategy produced 0 candidates; using uniform.")
            k = n or max(1, len(self.slices) // 5)
            return self._select_uniform(k)

        # Sort by score descending, then take top-n
        scores.sort(key=lambda t: t[0], reverse=True)
        k = n or len(scores)
        selected = [sl for _, sl in scores[:k]]
        # Re-sort by slice_index to restore spatial order
        selected.sort(key=lambda s: s.slice_index)
        return selected

    # ── Convenience ────────────────────────────

    def __len__(self) -> int:
        return len(self.slices)

    def __repr__(self) -> str:
        return (
            f"DICOMSeries(patient={self.patient_id!r}, "
            f"slices={len(self.slices)}, "
            f"shape={self.slices[0].shape if self.slices else 'n/a'})"
        )


# ──────────────────────────────────────────────
#  Loader
# ──────────────────────────────────────────────

class DICOMSeriesLoader:
    """
    Load all .dcm files in a directory as a single sorted DICOMSeries.

    The loader:
      1. Scans the directory for .dcm files
      2. Reads each header (fast — pydicom deferred pixel decoding)
      3. Sorts slices by z-position → InstanceNumber → filename
      4. Decodes pixel arrays and converts to HU (RescaleSlope/Intercept)
      5. Returns a DICOMSeries with all metadata populated

    Parameters
    ----------
    accepted_modalities : CT is the only supported modality for now
    min_slices          : minimum number of slices to constitute a valid series
    """

    ACCEPTED_MODALITIES = {"CT"}

    def __init__(
        self,
        accepted_modalities: Optional[set] = None,
        min_slices: int = 1,
    ):
        self.accepted_modalities = accepted_modalities or self.ACCEPTED_MODALITIES
        self.min_slices = min_slices

    # ── public API ────────────────────────────

    def load_series(self, folder_path: str) -> DICOMSeries:
        """
        Load and sort all DICOM slices in ``folder_path``.

        Parameters
        ----------
        folder_path : directory containing one CT series (.dcm files)

        Returns
        -------
        DICOMSeries — sorted, HU-calibrated, metadata-populated

        Raises
        ------
        FileNotFoundError : folder does not exist
        ValueError        : no valid DICOM files found, or wrong modality
        """
        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Series folder not found: {folder_path}")

        dcm_files = self._discover_dicom_files(folder_path)
        if not dcm_files:
            raise ValueError(f"No .dcm files found in: {folder_path}")

        logger.info(
            "Loading DICOM series from '%s' | files found: %d",
            folder_path,
            len(dcm_files),
        )

        # Step 1: Read headers (fast path — pixel data not yet decoded)
        records = self._read_headers(dcm_files)

        # Step 2: Filter to a single modality
        records = self._filter_by_modality(records)
        if not records:
            raise ValueError(
                f"No CT slices found in {folder_path}. "
                f"Accepted modalities: {self.accepted_modalities}"
            )

        # Step 3: Sort spatially
        records = self._sort_slices(records)

        # Step 4: Decode pixels + convert to HU
        slices = self._decode_slices(records)

        if len(slices) < self.min_slices:
            raise ValueError(
                f"Series has only {len(slices)} slice(s); "
                f"minimum required: {self.min_slices}"
            )

        # Step 5: Extract series-level metadata from first slice record
        first_ds = records[0]["dataset"]
        series = DICOMSeries(
            patient_id=str(getattr(first_ds, "PatientID", "")),
            study_date=str(getattr(first_ds, "StudyDate", "")),
            series_uid=str(getattr(first_ds, "SeriesInstanceUID", "")),
            folder_path=folder_path,
            slices=slices,
            pixel_spacing=self._get_pixel_spacing(first_ds),
            slice_thickness=float(getattr(first_ds, "SliceThickness", 1.0)),
        )

        logger.info(
            "Series loaded | patient=%s | slices=%d | shape=%s | "
            "spacing=(%.2f, %.2f, %.2f) mm",
            series.patient_id,
            len(series.slices),
            series.slices[0].shape,
            *series.pixel_spacing,
            series.slice_thickness,
        )
        return series

    # ── File discovery ─────────────────────────

    def _discover_dicom_files(self, folder: str) -> List[str]:
        """Return all files with .dcm / .dicom extension in the folder."""
        found = []
        for name in os.listdir(folder):
            if name.lower().endswith((".dcm", ".dicom")):
                found.append(os.path.join(folder, name))
        # Fallback: try every file (some DICOM series have no extension)
        if not found:
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                if os.path.isfile(full):
                    found.append(full)
        return sorted(found)  # deterministic initial order

    # ── Header reading ─────────────────────────

    def _read_headers(self, files: List[str]) -> List[Dict]:
        """
        Read each DICOM file with deferred pixel loading for speed.
        Returns list of dicts with 'path', 'dataset', 'z', 'instance'.
        Invalid or non-DICOM files are silently skipped.
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required. Install with: pip install pydicom"
            )

        records = []
        for path in files:
            try:
                # stop_before_pixels avoids decoding large pixel arrays now
                ds = pydicom.dcmread(path, stop_before_pixels=False)
                records.append({
                    "path": path,
                    "dataset": ds,
                    "z": self._get_z_position(ds),
                    "instance": int(getattr(ds, "InstanceNumber", 0)),
                })
            except Exception as exc:
                logger.debug("Skipping %s — not readable: %s", path, exc)
        return records

    # ── Modality filter ────────────────────────

    def _filter_by_modality(self, records: List[Dict]) -> List[Dict]:
        valid = []
        for r in records:
            modality = str(getattr(r["dataset"], "Modality", "UNKNOWN"))
            if modality in self.accepted_modalities:
                valid.append(r)
            else:
                logger.debug("Skipping %s — modality=%s", r["path"], modality)
        return valid

    # ── Spatial sorting ────────────────────────

    def _sort_slices(self, records: List[Dict]) -> List[Dict]:
        """
        Sort records inferior → superior.

        Priority:
          1. ImagePositionPatient z-coordinate (physical position in mm)
          2. InstanceNumber
          3. Filename (lexicographic, for DICOM-lite series)
        """
        has_z = any(r["z"] is not None for r in records)

        if has_z:
            # Fill None z-values with the mean so they don't break sorting
            z_vals = [r["z"] for r in records if r["z"] is not None]
            z_mean = float(np.mean(z_vals)) if z_vals else 0.0
            records.sort(key=lambda r: (
                r["z"] if r["z"] is not None else z_mean,
                r["instance"],
                os.path.basename(r["path"]),
            ))
        else:
            # No spatial metadata — fall back to InstanceNumber then filename
            records.sort(key=lambda r: (r["instance"], os.path.basename(r["path"])))
            logger.warning(
                "ImagePositionPatient not found in this series; "
                "sorting by InstanceNumber."
            )

        return records

    # ── Pixel decoding ─────────────────────────

    def _decode_slices(self, records: List[Dict]) -> List[DICOMSlice]:
        """Convert pixel arrays to HU and wrap in DICOMSlice objects."""
        slices = []
        for idx, record in enumerate(records):
            ds = record["dataset"]
            try:
                pixel = ds.pixel_array.astype(np.float32)
            except Exception as exc:
                logger.warning(
                    "Cannot decode pixels from %s: %s — skipping.",
                    record["path"],
                    exc,
                )
                continue

            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            hu_array = pixel * slope + intercept

            metadata = {
                "patient_id": str(getattr(ds, "PatientID", "")),
                "study_date": str(getattr(ds, "StudyDate", "")),
                "modality": str(getattr(ds, "Modality", "CT")),
                "slice_thickness": float(getattr(ds, "SliceThickness", 0)),
                "pixel_spacing": self._get_pixel_spacing(ds),
                "rows": int(getattr(ds, "Rows", hu_array.shape[0])),
                "columns": int(getattr(ds, "Columns", hu_array.shape[1])),
                "instance_number": record["instance"],
                "z_position": record["z"],
                "rescale_slope": slope,
                "rescale_intercept": intercept,
                "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            }

            slices.append(DICOMSlice(
                hu_array=hu_array,
                slice_index=idx,
                z_position=record["z"] or float(idx),
                instance_number=record["instance"],
                file_path=record["path"],
                metadata=metadata,
            ))

        return slices

    # ── Metadata helpers ───────────────────────

    @staticmethod
    def _get_z_position(ds) -> Optional[float]:
        """Extract z-coordinate from ImagePositionPatient (most reliable)."""
        try:
            ipp = ds.ImagePositionPatient
            return float(ipp[2])
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _get_pixel_spacing(ds) -> Tuple[float, float]:
        try:
            ps = ds.PixelSpacing
            return (float(ps[0]), float(ps[1]))
        except (AttributeError, IndexError, TypeError):
            return (1.0, 1.0)
