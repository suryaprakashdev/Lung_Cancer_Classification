"""
input_handler.py
----------------
Load and validate DICOM / JPEG / PNG inputs into a standardised
``InputImage`` dataclass that downstream modules consume.

Handles:
  • DICOM → HU-calibrated 2-D pixel array + metadata extraction
  • JPEG / PNG → 0–255 uint8 array (assumed pre-windowed)
  • Input validation (dimensions, modality, file integrity)
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .config import InputConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Data container
# ──────────────────────────────────────────────

@dataclass
class InputImage:
    """Standardised representation consumed by the rest of the pipeline."""

    pixel_data: np.ndarray             # 2-D array (H, W), float32
    is_hu_calibrated: bool             # True for DICOM, False for JPEG/PNG
    source_type: str                   # "dicom" | "jpeg" | "png"
    original_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
#  Loader
# ──────────────────────────────────────────────

class ImageLoader:
    """Load and validate medical images from disk."""

    def __init__(self, config: Optional[InputConfig] = None):
        self.cfg = config or InputConfig()

    # ── public API ────────────────────────────

    def load(self, path: str) -> InputImage:
        """
        Load a single image file and return an ``InputImage``.

        Raises
        ------
        FileNotFoundError  – path does not exist
        ValueError         – unsupported extension / invalid image
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext not in self.cfg.accepted_extensions:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Accepted: {self.cfg.accepted_extensions}"
            )

        if ext in (".dcm", ".dicom"):
            return self._load_dicom(path)
        else:
            return self._load_image(path, ext)

    # ── DICOM ─────────────────────────────────

    def _load_dicom(self, path: str) -> InputImage:
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required for DICOM input. "
                "Install with: pip install pydicom"
            )

        try:
            ds = pydicom.dcmread(path)
        except Exception as exc:
            raise ValueError(f"Failed to read DICOM file: {exc}") from exc

        # Modality check
        modality = getattr(ds, "Modality", "UNKNOWN")
        if modality not in self.cfg.accepted_modalities:
            raise ValueError(
                f"Unsupported DICOM modality '{modality}'. "
                f"Accepted: {self.cfg.accepted_modalities}"
            )

        if not hasattr(ds, "pixel_array"):
            raise ValueError("DICOM file has no pixel data.")

        # Convert to Hounsfield Units
        pixel = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu_image = pixel * slope + intercept

        self._validate_dimensions(hu_image, path)

        # Extract clinically relevant metadata
        metadata = {
            "patient_id": str(getattr(ds, "PatientID", "unknown")),
            "study_date": str(getattr(ds, "StudyDate", "")),
            "modality": modality,
            "slice_thickness": float(getattr(ds, "SliceThickness", 0)),
            "pixel_spacing": [
                float(s) for s in getattr(ds, "PixelSpacing", [1.0, 1.0])
            ],
            "rows": int(getattr(ds, "Rows", hu_image.shape[0])),
            "columns": int(getattr(ds, "Columns", hu_image.shape[1])),
            "instance_number": int(getattr(ds, "InstanceNumber", 0)),
            "rescale_slope": slope,
            "rescale_intercept": intercept,
        }

        logger.info(
            "Loaded DICOM | patient=%s | shape=%s | HU range=[%.0f, %.0f]",
            metadata["patient_id"],
            hu_image.shape,
            hu_image.min(),
            hu_image.max(),
        )

        return InputImage(
            pixel_data=hu_image,
            is_hu_calibrated=True,
            source_type="dicom",
            original_path=path,
            metadata=metadata,
        )

    # ── JPEG / PNG ────────────────────────────

    def _load_image(self, path: str, ext: str) -> InputImage:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image (corrupt?): {path}")

        self._validate_dimensions(img, path)

        source_type = "jpeg" if ext in (".jpg", ".jpeg") else "png"

        logger.info(
            "Loaded %s | shape=%s | value range=[%d, %d]",
            source_type.upper(),
            img.shape,
            img.min(),
            img.max(),
        )

        if source_type == "jpeg":
            logger.warning(
                "JPEG input is not HU-calibrated. "
                "Lung windowing cannot be faithfully reproduced. "
                "Assuming the image is already pre-windowed."
            )

        return InputImage(
            pixel_data=img.astype(np.float32),
            is_hu_calibrated=False,
            source_type=source_type,
            original_path=path,
            metadata={"original_shape": img.shape},
        )

    # ── Validation ────────────────────────────

    def _validate_dimensions(self, img: np.ndarray, path: str) -> None:
        h, w = img.shape[:2]
        if h < self.cfg.min_dimension or w < self.cfg.min_dimension:
            raise ValueError(
                f"Image too small ({w}×{h}). "
                f"Minimum: {self.cfg.min_dimension}×{self.cfg.min_dimension}. "
                f"File: {path}"
            )
        if h > self.cfg.max_dimension or w > self.cfg.max_dimension:
            raise ValueError(
                f"Image too large ({w}×{h}). "
                f"Maximum: {self.cfg.max_dimension}×{self.cfg.max_dimension}. "
                f"File: {path}"
            )
