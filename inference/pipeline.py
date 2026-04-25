"""
pipeline.py
-----------
End-to-end orchestrator that chains all inference modules together.

Usage
-----
    from inference import InferencePipeline

    pipe = InferencePipeline(checkpoint_dir="/path/to/checkpoints")
    result = pipe.run("patient_scan.dcm", model_name="efficientnet_b2")

    print(result.prediction)           # "Malignant"
    print(result.probability)          # 0.8723
    print(result.confidence)           # "High"
    result.save_gradcam("output.png")  # Grad-CAM overlay
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from .config import (
    InputConfig,
    InferenceConfig,
    LogConfig,
    PreprocessingConfig,
    ROIConfig,
    GIT_REPO_PROFILE,
    COLAB_PROFILE,
)
from .input_handler import ImageLoader, InputImage
from .inference_preprocessing import Preprocessor
from .roi_extractor import ROIExtractor, CandidatePatch
from .inference_engine import ModelInferenceEngine, InferenceResult
from .explainability import GradCAM, create_overlay, draw_bbox_on_overlay

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Pipeline result container
# ──────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Complete inference output including predictions and Grad-CAM data."""

    request_id: str
    input_path: str
    source_type: str

    # Per-candidate results (usually 1 for pre-cropped, N for full slices)
    predictions: List[InferenceResult] = field(default_factory=list)
    candidate_patches: List[CandidatePatch] = field(default_factory=list)

    # Grad-CAM outputs (populated if generate_gradcam=True)
    gradcam_heatmaps: List[np.ndarray] = field(default_factory=list)
    gradcam_overlays: List[np.ndarray] = field(default_factory=list)

    # Timing
    total_time_ms: float = 0.0
    input_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_prediction(self) -> Optional[InferenceResult]:
        """The prediction for the first (or only) candidate."""
        return self.predictions[0] if self.predictions else None

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"{'='*55}",
            f"  Inference Result — {self.request_id[:8]}",
            f"{'='*55}",
            f"  Input:    {os.path.basename(self.input_path)}",
            f"  Type:     {self.source_type}",
            f"  Regions:  {len(self.predictions)} candidate(s)",
        ]
        for i, pred in enumerate(self.predictions):
            lines.append(
                f"  [{i}] {pred.prediction:10s} | "
                f"prob={pred.probability:.4f} | "
                f"confidence={pred.confidence:6s} | "
                f"model={pred.model_name}"
            )
        lines.append(f"  Total:    {self.total_time_ms:.1f} ms")
        lines.append(f"{'='*55}")
        return "\n".join(lines)

    def save_gradcam(
        self,
        output_path: str,
        candidate_index: int = 0,
    ) -> str:
        """Save Grad-CAM overlay to disk."""
        if candidate_index >= len(self.gradcam_overlays):
            raise IndexError(
                f"No Grad-CAM overlay for candidate {candidate_index}. "
                f"Available: {len(self.gradcam_overlays)}"
            )
        overlay = self.gradcam_overlays[candidate_index]
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_bgr)
        logger.info("Saved Grad-CAM overlay → %s", output_path)
        return output_path

    def to_dict(self) -> Dict:
        """Serialisable summary (for JSON logging / API responses)."""
        return {
            "request_id": self.request_id,
            "input_path": self.input_path,
            "source_type": self.source_type,
            "num_candidates": len(self.predictions),
            "predictions": [p.to_dict() for p in self.predictions],
            "total_time_ms": round(self.total_time_ms, 2),
            "input_metadata": self.input_metadata,
        }


# ──────────────────────────────────────────────
#  Pipeline orchestrator
# ──────────────────────────────────────────────

class InferencePipeline:
    """
    End-to-end inference pipeline: input → preprocess → ROI → predict → Grad-CAM.

    Parameters
    ----------
    checkpoint_dir   : directory containing *_best.pth files
    profile          : "git_repo" (default) or "colab" — selects the correct
                       normalisation and output-head interpretation
    model_name       : default model to use for predictions
    device           : torch device (auto-detected if None)
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        profile: str = "git_repo",
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        # Select configuration profile
        if profile == "colab":
            cfg = COLAB_PROFILE
        else:
            cfg = GIT_REPO_PROFILE

        preproc_cfg = cfg["preprocessing"]
        infer_cfg = cfg["inference"]

        # Override checkpoint dir
        infer_cfg = InferenceConfig(
            output_mode=infer_cfg.output_mode,
            classification_threshold=infer_cfg.classification_threshold,
            confidence_high=infer_cfg.confidence_high,
            confidence_medium=infer_cfg.confidence_medium,
            default_model=model_name or infer_cfg.default_model,
            checkpoint_dir=checkpoint_dir,
            use_amp=infer_cfg.use_amp,
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialise sub-modules
        self.loader = ImageLoader(InputConfig())
        self.preprocessor = Preprocessor(preproc_cfg)
        self.roi_extractor = ROIExtractor(ROIConfig(), preproc_cfg)
        self.engine = ModelInferenceEngine(infer_cfg, self.device)

        self._profile = profile
        self._default_model = infer_cfg.default_model

        # Configure logging
        log_cfg = LogConfig()
        logging.basicConfig(level=log_cfg.log_level, format=log_cfg.log_format)

        logger.info(
            "Pipeline initialised | profile=%s | model=%s | device=%s",
            profile,
            self._default_model,
            self.device,
        )

    # ── Setup ─────────────────────────────────

    def setup(self, model_names: Optional[List[str]] = None) -> None:
        """
        Load model(s) and run a warmup forward pass.
        Call this once at startup before processing requests.
        """
        names = model_names or [self._default_model]
        for name in names:
            self.engine.load_model(name)
            self.engine.warmup(name)
        logger.info("Pipeline ready. Loaded models: %s", names)

    # ── Main entry point ──────────────────────

    def run(
        self,
        input_path: str,
        model_name: Optional[str] = None,
        generate_gradcam: bool = True,
        max_candidates: int = 5,
    ) -> PipelineResult:
        """
        Run the full inference pipeline on a single input file.

        Parameters
        ----------
        input_path      : path to a DICOM, JPEG, or PNG file
        model_name      : override the default model
        generate_gradcam: if True, produce Grad-CAM overlays
        max_candidates  : limit the number of ROI candidates processed

        Returns
        -------
        PipelineResult with predictions, Grad-CAM data, and metadata
        """
        request_id = str(uuid.uuid4())
        model = model_name or self._default_model
        t0 = time.perf_counter()

        logger.info(
            "Pipeline.run | request=%s | input=%s | model=%s",
            request_id[:8],
            os.path.basename(input_path),
            model,
        )

        result = PipelineResult(
            request_id=request_id,
            input_path=input_path,
            source_type="unknown",
        )

        try:
            # Step 1: Load and validate input
            input_image = self.loader.load(input_path)
            result.source_type = input_image.source_type
            result.input_metadata = input_image.metadata

            # Step 2: ROI extraction (or pass-through for small patches)
            candidates = self.roi_extractor.extract(
                input_image.pixel_data,
                input_image.is_hu_calibrated,
            )
            candidates = candidates[:max_candidates]
            result.candidate_patches = candidates

            if not candidates:
                logger.warning("No candidate regions found in %s", input_path)
                result.total_time_ms = (time.perf_counter() - t0) * 1000
                return result

            # Step 3 & 4: Preprocess + Predict each candidate
            for candidate in candidates:
                tensor = self.preprocessor.preprocess(
                    candidate.patch,
                    is_hu_calibrated=input_image.is_hu_calibrated,
                    device=self.device,
                )

                prediction = self.engine.predict(tensor, model_name=model)
                result.predictions.append(prediction)

                # Step 5: Grad-CAM (optional)
                if generate_gradcam:
                    heatmap, overlay = self._generate_gradcam(
                        tensor, candidate.patch, model
                    )
                    result.gradcam_heatmaps.append(heatmap)
                    result.gradcam_overlays.append(overlay)

        except Exception as exc:
            logger.error(
                "Pipeline error | request=%s | error=%s",
                request_id[:8],
                str(exc),
                exc_info=True,
            )
            raise

        result.total_time_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "Pipeline complete | request=%s | candidates=%d | time=%.1fms",
            request_id[:8],
            len(result.predictions),
            result.total_time_ms,
        )
        return result

    # ── Grad-CAM helper ───────────────────────

    def _generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        original_patch: np.ndarray,
        model_name: str,
    ) -> tuple:
        """Generate heatmap + overlay for a single candidate."""
        model = self.engine._model_cache.get(model_name)
        if model is None:
            logger.warning("Cannot generate Grad-CAM — model not loaded.")
            empty = np.zeros((224, 224), dtype=np.float32)
            return empty, np.zeros((224, 224, 3), dtype=np.uint8)

        with GradCAM(model, model_name) as gradcam:
            heatmap = gradcam.generate(input_tensor)

        # Prepare the original patch for overlay
        if original_patch.max() > 1.0:
            display_img = np.clip(original_patch, 0, 255).astype(np.uint8)
        else:
            display_img = (original_patch * 255).astype(np.uint8)

        overlay = create_overlay(display_img, heatmap)
        overlay = draw_bbox_on_overlay(overlay, heatmap)

        return heatmap, overlay

    # ── Batch inference ───────────────────────

    def run_batch(
        self,
        input_paths: List[str],
        model_name: Optional[str] = None,
        generate_gradcam: bool = False,
    ) -> List[PipelineResult]:
        """
        Process multiple inputs sequentially.
        For high-throughput, consider parallelising with a DataLoader.
        """
        results = []
        for i, path in enumerate(input_paths):
            logger.info("Batch progress: %d/%d", i + 1, len(input_paths))
            try:
                result = self.run(
                    path,
                    model_name=model_name,
                    generate_gradcam=generate_gradcam,
                )
                results.append(result)
            except Exception as exc:
                logger.error("Failed on %s: %s", path, exc)
                # Create error result rather than crashing the batch
                results.append(PipelineResult(
                    request_id=str(uuid.uuid4()),
                    input_path=path,
                    source_type="error",
                    input_metadata={"error": str(exc)},
                ))
        return results

    # ── Cleanup ───────────────────────────────

    def shutdown(self) -> None:
        """Release all GPU resources."""
        self.engine.unload_all()
        logger.info("Pipeline shutdown complete.")
