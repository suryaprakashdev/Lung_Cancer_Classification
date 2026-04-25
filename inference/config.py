"""
config.py
---------
Centralised configuration for the inference pipeline.

Every magic number lives here so the pipeline stays in sync with training.
Two preset profiles are provided — one for the git-repo checkpoints
(BCEWithLogitsLoss, 1 logit, ImageNet norms) and one for the Colab
checkpoints (CrossEntropyLoss, 2 logits, [0.5] norms).
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────────────────────────
#  Preprocessing constants (must match training)
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class PreprocessingConfig:
    """Mirrors the exact transform chain used during training (MONAI-based)."""

    input_size: Tuple[int, int] = (224, 224)

    # Lung HU windowing (from preprocessing.py)
    lung_window_center: int = -600
    lung_window_width: int = 1500

    # Derived HU range for ScaleIntensityRange
    @property
    def lung_hu_min(self) -> int:
        return self.lung_window_center - self.lung_window_width // 2

    @property
    def lung_hu_max(self) -> int:
        return self.lung_window_center + self.lung_window_width // 2

    # MONAI normalisation: channel-wise computed (no hardcoded mean/std)
    # These are kept only for backward compatibility with legacy pipelines
    normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)

    # Padding added around nodule bounding boxes during training
    bbox_padding: int = 10


# ──────────────────────────────────────────────
#  Inference constants
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class InferenceConfig:
    """Model loading and prediction parameters."""

    # Output head format
    #   "bce"  → single raw logit  + sigmoid  (git_repo checkpoints)
    #   "ce"   → 2-class logits    + softmax  (Colab checkpoints)
    output_mode: str = "bce"

    classification_threshold: float = 0.5

    # Confidence tiers  (distance from threshold)
    confidence_high: float = 0.3    # |p − 0.5| > 0.3  → High
    confidence_medium: float = 0.15  # |p − 0.5| > 0.15 → Medium

    # Default model (best performer on your dataset)
    default_model: str = "efficientnet_b2"

    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"

    # Use mixed-precision (FP16) on CUDA — ~2× faster, negligible accuracy loss
    use_amp: bool = True


# ──────────────────────────────────────────────
#  ROI extraction constants
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class ROIConfig:
    """Parameters for annotation-free nodule candidate detection."""

    # Lung segmentation threshold (HU)
    lung_threshold_hu: float = -400.0

    # Dense-structure threshold within lung field (HU)
    nodule_threshold_hu: float = -100.0

    # Candidate area filters (in pixels — adjust based on PixelSpacing)
    min_candidate_area: int = 9       # ~3×3 px
    max_candidate_area: int = 10000   # ~100×100 px

    # Minimum circularity (4π·area / perimeter²)
    min_circularity: float = 0.3

    # Morphological kernel size for lung mask cleanup
    morph_kernel_size: int = 5


# ──────────────────────────────────────────────
#  Input validation
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class InputConfig:
    """Constraints for accepted input images."""

    min_dimension: int = 32
    max_dimension: int = 4096
    accepted_extensions: Tuple[str, ...] = (".dcm", ".dicom", ".jpg", ".jpeg", ".png")
    accepted_modalities: Tuple[str, ...] = ("CT",)


# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class LogConfig:
    log_level: str = "INFO"
    log_format: str = (
        "%(asctime)s | %(name)-24s | %(levelname)-5s | %(message)s"
    )


# ──────────────────────────────────────────────
#  Preset profiles
# ──────────────────────────────────────────────

# Profile for git_repo checkpoints (BCEWithLogitsLoss, 1 logit, ImageNet norms)
GIT_REPO_PROFILE = {
    "preprocessing": PreprocessingConfig(),
    "inference": InferenceConfig(output_mode="bce"),
}

# Profile for Colab checkpoints (CrossEntropyLoss, 2 logits, [0.5] norms)
COLAB_PROFILE = {
    "preprocessing": PreprocessingConfig(
        normalize_mean=(0.5, 0.5, 0.5),
        normalize_std=(0.5, 0.5, 0.5),
    ),
    "inference": InferenceConfig(output_mode="ce"),
}
