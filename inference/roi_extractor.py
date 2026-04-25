"""
roi_extractor.py
----------------
Annotation-free Region-of-Interest extraction for full CT slices.

Two tracks:
  Track A — Pre-cropped patches: pass through unchanged
  Track B — Full slices: threshold-based lung segmentation → candidate detection

This is a screening filter, NOT a replacement for a dedicated detection model.
For production at scale, integrate an upstream detection network.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import PreprocessingConfig, ROIConfig

logger = logging.getLogger(__name__)


@dataclass
class CandidatePatch:
    """A single candidate nodule region cropped from a full CT slice."""

    patch: np.ndarray                    # (H, W) float32 — cropped region
    bbox: Tuple[int, int, int, int]      # (x, y, w, h) in original coords
    center: Tuple[int, int]              # (cx, cy) in original coords
    area_px: int
    circularity: float
    index: int                           # candidate index within the slice


class ROIExtractor:
    """Extract candidate nodule regions from full CT slices without annotations."""

    def __init__(
        self,
        roi_config: Optional[ROIConfig] = None,
        preproc_config: Optional[PreprocessingConfig] = None,
    ):
        self.roi = roi_config or ROIConfig()
        self.preproc = preproc_config or PreprocessingConfig()

    # ── public API ────────────────────────────

    def is_full_slice(self, pixel_data: np.ndarray) -> bool:
        """
        Heuristic: a full CT slice is typically ≥ 256×256.
        Pre-cropped nodule patches from your training pipeline are usually
        much smaller (variable size, then resized to 224×224).
        """
        h, w = pixel_data.shape[:2]
        return h >= 256 and w >= 256

    def extract(
        self,
        pixel_data: np.ndarray,
        is_hu_calibrated: bool,
    ) -> List[CandidatePatch]:
        """
        Extract candidate nodule patches from a full CT slice.

        Parameters
        ----------
        pixel_data       : (H, W) float32 — HU values (DICOM) or 0-255 (image)
        is_hu_calibrated : True if pixel_data is in Hounsfield Units

        Returns
        -------
        List of CandidatePatch objects, sorted by area (largest first).
        """
        if not self.is_full_slice(pixel_data):
            logger.info(
                "Image is small (%s) — treating as pre-cropped nodule patch. "
                "Skipping ROI extraction.",
                pixel_data.shape,
            )
            return [CandidatePatch(
                patch=pixel_data,
                bbox=(0, 0, pixel_data.shape[1], pixel_data.shape[0]),
                center=(pixel_data.shape[1] // 2, pixel_data.shape[0] // 2),
                area_px=pixel_data.size,
                circularity=1.0,
                index=0,
            )]

        if is_hu_calibrated:
            return self._extract_hu_based(pixel_data)
        else:
            return self._extract_intensity_based(pixel_data)

    # ── HU-based extraction (DICOM) ──────────

    def _extract_hu_based(self, hu_image: np.ndarray) -> List[CandidatePatch]:
        """Threshold-based lung segmentation + nodule candidate detection."""

        # Step 1: Segment the lung field
        lung_mask = self._segment_lungs(hu_image)
        if lung_mask.sum() == 0:
            logger.warning("Lung segmentation produced an empty mask.")
            return []

        # Step 2: Find dense structures within lungs (potential nodules)
        dense_mask = (hu_image > self.roi.nodule_threshold_hu).astype(np.uint8)
        candidate_mask = cv2.bitwise_and(dense_mask, dense_mask, mask=lung_mask)

        # Step 3: Find and filter contours
        candidates = self._extract_candidates(candidate_mask, hu_image)

        logger.info(
            "ROI extraction found %d candidate(s) from %s slice",
            len(candidates),
            hu_image.shape,
        )
        return candidates

    def _segment_lungs(self, hu_image: np.ndarray) -> np.ndarray:
        """Binary lung mask via HU thresholding + morphological cleanup."""
        # Air and lung tissue are below the threshold
        binary = (hu_image < self.roi.lung_threshold_hu).astype(np.uint8)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.roi.morph_kernel_size, self.roi.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Keep the two largest connected components (left + right lungs)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return binary

        # Sort components by area (skip background at index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1] + 1  # +1 for background offset
        top_n = min(2, len(sorted_indices))

        lung_mask = np.zeros_like(binary)
        for i in range(top_n):
            lung_mask[labels == sorted_indices[i]] = 1

        # Fill holes within the lung mask
        contours, _ = cv2.findContours(
            lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(lung_mask, contours, -1, 1, cv2.FILLED)

        return lung_mask

    # ── Intensity-based extraction (JPEG/PNG) ─

    def _extract_intensity_based(self, image: np.ndarray) -> List[CandidatePatch]:
        """
        Fallback for non-HU images: use adaptive thresholding to find
        bright regions that may correspond to nodules.
        """
        img_u8 = np.clip(image, 0, 255).astype(np.uint8)

        # Adaptive threshold to find bright dense regions
        blurred = cv2.GaussianBlur(img_u8, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        candidates = self._extract_candidates(thresh, image)

        logger.info(
            "Intensity-based extraction found %d candidate(s)",
            len(candidates),
        )
        return candidates

    # ── Shared candidate extraction ───────────

    def _extract_candidates(
        self,
        binary_mask: np.ndarray,
        source_image: np.ndarray,
    ) -> List[CandidatePatch]:
        """Extract, filter, and crop candidate regions from a binary mask."""
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        pad = self.preproc.bbox_padding

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # Area filter
            if area < self.roi.min_candidate_area:
                continue
            if area > self.roi.max_candidate_area:
                continue

            # Circularity filter
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.roi.min_circularity:
                continue

            # Bounding box with padding (matches training-time padding)
            x, y, w, h = cv2.boundingRect(contour)
            x_min = max(0, x - pad)
            y_min = max(0, y - pad)
            x_max = min(source_image.shape[1], x + w + pad)
            y_max = min(source_image.shape[0], y + h + pad)

            patch = source_image[y_min:y_max, x_min:x_max]
            if patch.size == 0:
                continue

            cx = x + w // 2
            cy = y + h // 2

            candidates.append(CandidatePatch(
                patch=patch.astype(np.float32),
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                center=(cx, cy),
                area_px=int(area),
                circularity=float(circularity),
                index=len(candidates),
            ))

        # Sort by area (largest first — larger nodules are more clinically relevant)
        candidates.sort(key=lambda c: c.area_px, reverse=True)
        return candidates
