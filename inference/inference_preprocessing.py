"""
inference_preprocessing.py
--------------------------
Transform raw pixel data into model-ready tensors using MONAI transforms,
exactly matching the deterministic validation transform used during training.

Two paths:
  HU-calibrated (DICOM)  → MONAI: intensity windowing → resize → normalize → tensor
  Pre-windowed  (JPEG/PNG) → MONAI: scale → resize → normalize → tensor
"""

import logging
from typing import Optional

import numpy as np
import torch

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    RepeatChannel,
    Resize,
    ScaleIntensityRange,
    ToTensor,
)

from .config import PreprocessingConfig

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Converts a 2-D pixel array into a normalised (1, 3, 224, 224) tensor
    consistent with the MONAI-based training-time validation transform.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.cfg = config or PreprocessingConfig()

        # Build the exact MONAI transform chain matching training val transforms
        self.hu_transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensityRange(
                a_min=self.cfg.lung_hu_min,
                a_max=self.cfg.lung_hu_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resize(spatial_size=self.cfg.input_size, mode="bilinear"),
            RepeatChannel(repeats=3),
            NormalizeIntensity(channel_wise=True),
            ToTensor(),
        ])

        # For pre-windowed images (0-255 range), scale to [0, 1] first
        self.image_transform = Compose([
            EnsureChannelFirst(channel_dim="no_channel"),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resize(spatial_size=self.cfg.input_size, mode="bilinear"),
            RepeatChannel(repeats=3),
            NormalizeIntensity(channel_wise=True),
            ToTensor(),
        ])

    # ── public API ────────────────────────────

    def preprocess(
        self,
        pixel_data: np.ndarray,
        is_hu_calibrated: bool,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Convert a 2-D pixel array → batched tensor (1, 3, 224, 224).

        Parameters
        ----------
        pixel_data       : (H, W) float32 array — HU values or 0-255 uint8
        is_hu_calibrated : if True, apply lung windowing via MONAI
        device           : target torch device
        """
        pixel_data = pixel_data.astype(np.float32)

        if is_hu_calibrated:
            tensor = self.hu_transform(pixel_data)
            logger.debug("Applied HU-calibrated MONAI transform pipeline")
        else:
            tensor = self.image_transform(pixel_data)
            logger.debug("Applied pre-windowed MONAI transform pipeline")

        tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)

        logger.debug(
            "Preprocessed tensor | shape=%s | device=%s | "
            "min=%.3f | max=%.3f",
            tensor.shape, tensor.device, tensor.min().item(), tensor.max().item(),
        )
        return tensor

    # ── Convenience: preprocess a file path directly ──────────

    def preprocess_from_file(
        self,
        image_path: str,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Load and preprocess a .npy, JPEG, or PNG nodule patch."""
        import os

        ext = os.path.splitext(image_path)[1].lower()

        if ext == ".npy":
            img = np.load(image_path).astype(np.float32)
            return self.preprocess(img, is_hu_calibrated=True, device=device)
        else:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            return self.preprocess(
                img.astype(np.float32), is_hu_calibrated=False, device=device
            )
