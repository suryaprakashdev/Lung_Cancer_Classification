"""
volume_inference.py
-------------------
End-to-end wrapper that bridges 3D DICOM CT volumes and the existing 2D
inference pipeline.

The existing ``InferencePipeline`` is designed for single-image inputs.
This module wraps it to handle volumetric data:

    DICOM folder → DICOMSeries → slice selection → per-slice pipeline.run()
                → aggregation → VolumeInferenceResult

The 2D pipeline is called once per selected slice and its internal logic
(preprocessing, ROI extraction, model inference, Grad-CAM) is left entirely
untouched.

Design decisions
----------------
* Aggregation is decoupled from selection — swap either without touching the other.
* Grad-CAM is only generated for top-k suspicious slices (configurable) to
  avoid GPU/memory pressure when running over 200+ slice volumes.
* A ``FutureHooks`` namespace provides stubs for future 3D model integration
  (e.g. MONAI SlidingWindowInferer) without altering current call sites.
* CPU-safe: no CUDA-specific code paths are added beyond what the existing
  engine already handles.

Usage
-----
    from volume_inference import VolumeInferenceEngine

    engine = VolumeInferenceEngine(
        checkpoint_dir="checkpoints",
        model_name="efficientnet_b2",
    )
    result = engine.run_volume(
        "/path/to/patient_series_folder",
        slice_strategy="middle",
        n_slices=20,
        aggregation="top_k",
        k=5,
        gradcam_top_k=3,
    )

    print(result.summary())
    result.save_top_gradcams("output/")
"""

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from inference.pipeline import InferencePipeline, PipelineResult
from inference.inference_engine import InferenceResult

from dicom_series_loader import DICOMSeriesLoader, DICOMSeries, DICOMSlice

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Result containers
# ──────────────────────────────────────────────

@dataclass
class SliceInferenceRecord:
    """
    Inference output for a single CT slice, with enough metadata to
    reconstruct provenance (which file, which z-position, which model).
    """

    slice_index: int
    z_position: float
    file_path: str
    pipeline_result: PipelineResult      # full output from the 2D pipeline
    primary_prediction: InferenceResult  # shortcut to the main candidate
    has_gradcam: bool = False


@dataclass
class VolumeInferenceResult:
    """
    Aggregated patient-level result built from per-slice InferenceResults.

    Attributes
    ----------
    request_id          : unique run identifier
    folder_path         : input DICOM series directory
    patient_id          : PatientID from DICOM header
    model_name          : model used for inference
    aggregation_method  : how slice probabilities were combined
    volume_probability  : final patient-level malignancy probability [0, 1]
    volume_prediction   : "Malignant" | "Benign"
    volume_confidence   : "High" | "Medium" | "Low"
    classification_threshold : decision boundary (default 0.5)
    n_slices_total      : slices in the full volume
    n_slices_processed  : slices actually run through inference
    slice_records       : per-slice detail (sorted by z_position)
    top_suspicious      : top-k most suspicious SliceInferenceRecords
    total_time_ms       : wall-clock inference time
    """

    request_id: str
    folder_path: str
    patient_id: str
    model_name: str
    aggregation_method: str

    volume_probability: float
    volume_prediction: str
    volume_confidence: str
    classification_threshold: float = 0.5

    n_slices_total: int = 0
    n_slices_processed: int = 0

    slice_records: List[SliceInferenceRecord] = field(default_factory=list)
    top_suspicious: List[SliceInferenceRecord] = field(default_factory=list)

    total_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Human-readable summary ─────────────────

    def summary(self) -> str:
        sep = "=" * 60
        lines = [
            sep,
            f"  Volume Inference Result — {self.request_id[:8]}",
            sep,
            f"  Patient ID  : {self.patient_id}",
            f"  Series      : {os.path.basename(self.folder_path)}",
            f"  Model       : {self.model_name}",
            f"  Aggregation : {self.aggregation_method}",
            f"  Slices      : {self.n_slices_processed} / {self.n_slices_total} processed",
            "",
            f"  ► PREDICTION  : {self.volume_prediction}",
            f"  ► PROBABILITY : {self.volume_probability:.4f}",
            f"  ► CONFIDENCE  : {self.volume_confidence}",
            "",
            "  Top suspicious slices:",
        ]
        for rec in self.top_suspicious:
            p = rec.primary_prediction
            lines.append(
                f"    [z={rec.z_position:+.1f}mm | slice={rec.slice_index:03d}] "
                f"{p.prediction:10s} | prob={p.probability:.4f} | "
                f"GradCAM={'✓' if rec.has_gradcam else '—'}"
            )
        lines += [f"  Total time  : {self.total_time_ms:.1f} ms", sep]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """JSON-serialisable output for API responses / audit logs."""
        return {
            "request_id": self.request_id,
            "patient_id": self.patient_id,
            "folder_path": self.folder_path,
            "model_name": self.model_name,
            "aggregation_method": self.aggregation_method,
            "volume_prediction": self.volume_prediction,
            "volume_probability": round(self.volume_probability, 4),
            "volume_confidence": self.volume_confidence,
            "n_slices_total": self.n_slices_total,
            "n_slices_processed": self.n_slices_processed,
            "total_time_ms": round(self.total_time_ms, 2),
            "top_suspicious": [
                {
                    "slice_index": r.slice_index,
                    "z_position": round(r.z_position, 2),
                    "file": os.path.basename(r.file_path),
                    "probability": round(r.primary_prediction.probability, 4),
                    "prediction": r.primary_prediction.prediction,
                    "has_gradcam": r.has_gradcam,
                }
                for r in self.top_suspicious
            ],
            "all_slices": [
                {
                    "slice_index": r.slice_index,
                    "z_position": round(r.z_position, 2),
                    "probability": round(r.primary_prediction.probability, 4),
                    "prediction": r.primary_prediction.prediction,
                    "confidence": r.primary_prediction.confidence,
                }
                for r in self.slice_records
            ],
        }

    def save_top_gradcams(
        self,
        output_dir: str,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Save Grad-CAM overlays for all top suspicious slices that have them.

        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        prefix = prefix or self.patient_id or self.request_id[:8]
        saved = []

        for rec in self.top_suspicious:
            if not rec.has_gradcam:
                continue
            pr = rec.pipeline_result
            if not pr.gradcam_overlays:
                continue

            fname = (
                f"{prefix}_slice{rec.slice_index:03d}"
                f"_z{rec.z_position:+.0f}"
                f"_prob{rec.primary_prediction.probability:.2f}.png"
            )
            out_path = os.path.join(output_dir, fname)
            try:
                pr.save_gradcam(out_path, candidate_index=0)
                saved.append(out_path)
            except Exception as exc:
                logger.warning("Failed to save Grad-CAM for slice %d: %s", rec.slice_index, exc)

        logger.info("Saved %d Grad-CAM overlays → %s", len(saved), output_dir)
        return saved


# ──────────────────────────────────────────────
#  Aggregation strategies
# ──────────────────────────────────────────────

class ProbabilityAggregator:
    """
    Combines per-slice malignancy probabilities into a scalar patient-level score.

    Strategies
    ----------
    max      : probability of the most suspicious slice (clinical worst-case)
    mean     : average over all processed slices (population-level estimate)
    top_k    : mean of the k highest-probability slices (balances outliers)
    """

    SUPPORTED = ("max", "mean", "top_k")

    def aggregate(
        self,
        probabilities: List[float],
        strategy: str = "top_k",
        k: int = 5,
    ) -> Tuple[float, str]:
        """
        Parameters
        ----------
        probabilities : list of per-slice malignancy probabilities [0, 1]
        strategy      : one of {"max", "mean", "top_k"}
        k             : number of top slices used by "top_k"

        Returns
        -------
        (aggregated_probability, strategy_label)
        """
        if not probabilities:
            return 0.0, strategy

        probs = np.asarray(probabilities, dtype=np.float32)

        if strategy == "max":
            result = float(probs.max())

        elif strategy == "mean":
            result = float(probs.mean())

        elif strategy == "top_k":
            k_actual = min(k, len(probs))
            top = np.sort(probs)[::-1][:k_actual]
            strong = top[top > 0.7]

            if len(strong) > 0:
                result = float(strong.max()) 
            else:
                result = float(top.mean())     

        else:
            raise ValueError(
                f"Unknown aggregation strategy '{strategy}'. "
                f"Choose from: {self.SUPPORTED}"
            )

        logger.debug(
            "Aggregation | strategy=%s | n=%d | result=%.4f",
            strategy,
            len(probabilities),
            result,
        )
        return result, f"{strategy}(k={k})" if strategy == "top_k" else strategy

    def confidence_tier(
        self,
        probability: float,
        threshold: float = 0.5,
        high: float = 0.3,
        medium: float = 0.15,
    ) -> str:
        distance = abs(probability - threshold)
        if distance > high:
            return "High"
        elif distance > medium:
            return "Medium"
        return "Low"


# ──────────────────────────────────────────────
#  Future 3D integration stubs
# ──────────────────────────────────────────────

class FutureHooks:
    """
    Namespace of no-op stubs that can be overridden when 3D models become
    available (e.g. MONAI SlidingWindowInferer or a nnU-Net detector).

    Example override for a MONAI 3D volume model::

        class MONAI3DInferer(FutureHooks):
            def run_3d_inference(self, volume: np.ndarray, **kwargs):
                from monai.inferers import SlidingWindowInferer
                # ... return VolumeInferenceResult directly

    By returning ``None`` from all stubs, the VolumeInferenceEngine
    falls through to the 2D slice-wise path — backward compatible.
    """

    def preprocess_volume(
        self,
        volume: np.ndarray,          # (D, H, W) float32 HU array
        pixel_spacing: Tuple,
        slice_thickness: float,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Override to apply 3D resampling / intensity normalisation."""
        return None  # signal: "not implemented, use 2D path"

    def run_3d_inference(
        self,
        volume: np.ndarray,
        **kwargs,
    ) -> Optional["VolumeInferenceResult"]:
        """Override to run a native 3D model and bypass slice-wise inference."""
        return None  # signal: "not implemented, use 2D path"

    def postprocess_segmentation(
        self,
        segmentation: np.ndarray,    # (D, H, W) predicted mask
        series: DICOMSeries,
        **kwargs,
    ) -> Optional[np.ndarray]:
        """Override to convert a 3D segmentation mask into an ROI mask per slice."""
        return None


# ──────────────────────────────────────────────
#  Volume inference engine
# ──────────────────────────────────────────────

class VolumeInferenceEngine:
    """
    Patient-level inference engine for volumetric CT data.

    Wraps the existing 2D ``InferencePipeline`` and orchestrates:
      1. DICOM series loading
      2. Slice selection
      3. Per-slice inference (via existing pipeline)
      4. Probability aggregation
      5. Grad-CAM generation for top-k slices

    Parameters
    ----------
    checkpoint_dir   : directory containing *_best.pth checkpoints
    model_name       : model to run (default: efficientnet_b2)
    profile          : "git_repo" or "colab" — must match training
    device           : torch.device or None (auto-detected)
    hooks            : optional FutureHooks subclass for 3D extension
    classification_threshold : decision boundary (default 0.5)
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        model_name: str = "efficientnet_b2",
        profile: str = "git_repo",
        device=None,
        hooks: Optional[FutureHooks] = None,
        classification_threshold: float = 0.5,
    ):
        self._pipeline = InferencePipeline(
            checkpoint_dir=checkpoint_dir,
            profile=profile,
            model_name=model_name,
            device=device,
        )
        self._loader = DICOMSeriesLoader()
        self._aggregator = ProbabilityAggregator()
        self._hooks = hooks or FutureHooks()
        self._model_name = model_name
        self._threshold = classification_threshold

        # Warm up the model on startup
        self._pipeline.setup([model_name])

        logger.info(
            "VolumeInferenceEngine ready | model=%s | profile=%s",
            model_name,
            profile,
        )

    # ── Main entry point ──────────────────────

    def run_volume(
        self,
        folder_path: str,
        slice_strategy: str = "middle",
        n_slices: Optional[int] = None,
        aggregation: str = "top_k",
        k: int = 5,
        gradcam_top_k: int = 3,
        roi_mask: Optional[np.ndarray] = None,
    ) -> VolumeInferenceResult:
        """
        Run end-to-end inference on a DICOM CT series folder.

        Parameters
        ----------
        folder_path     : path to directory containing .dcm slices
        slice_strategy  : "all" | "middle" | "uniform" | "roi"
        n_slices        : number of slices to process (strategy-dependent)
        aggregation     : "max" | "mean" | "top_k" probability combination
        k               : top-k slices for "top_k" aggregation and reporting
        gradcam_top_k   : generate Grad-CAM for this many top suspicious slices
        roi_mask        : (H, W) binary mask for the "roi" slice strategy

        Returns
        -------
        VolumeInferenceResult
        """
        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()

        logger.info(
            "VolumeInference | request=%s | folder=%s | strategy=%s | agg=%s",
            request_id[:8],
            os.path.basename(folder_path),
            slice_strategy,
            aggregation,
        )

        # ── 1. Load DICOM series ──────────────
        series = self._loader.load_series(folder_path)

        # ── 2. Optional 3D hook (no-op by default) ──
        vol_array = np.stack([s.hu_array for s in series.slices], axis=0)
        vol_array = self._hooks.preprocess_volume(
            vol_array,
            series.pixel_spacing,
            series.slice_thickness,
        ) or vol_array

        hook_result = self._hooks.run_3d_inference(vol_array)
        if hook_result is not None:
            # 3D model took over — return its result directly
            hook_result.total_time_ms = (time.perf_counter() - t0) * 1000
            return hook_result

        # ── 3. Slice selection ────────────────
        selected_slices = series.select_slices(
            strategy=slice_strategy,
            n=n_slices,
            roi_mask=roi_mask,
        )

        if not selected_slices:
            logger.error("Slice selection produced 0 slices. Aborting.")
            return self._empty_result(request_id, series, folder_path, aggregation)

        # ── 4. Per-slice inference ────────────
        slice_records = self._run_slice_inference(selected_slices)

        if not slice_records:
            logger.error("All slice inferences failed. Aborting.")
            return self._empty_result(request_id, series, folder_path, aggregation)

        # ── 5. Aggregate ──────────────────────
        probabilities = [r.primary_prediction.probability for r in slice_records]
        volume_prob, agg_label = self._aggregator.aggregate(
            probabilities, strategy=aggregation, k=k
        )

        volume_pred = "Malignant" if volume_prob > self._threshold else "Benign"
        volume_conf = self._aggregator.confidence_tier(
            volume_prob, threshold=self._threshold
        )

        # ── 6. Identify top-k suspicious slices ──
        sorted_by_prob = sorted(
            slice_records,
            key=lambda r: r.primary_prediction.probability,
            reverse=True,
        )
        top_k_records = sorted_by_prob[:k]

        # ── 7. Grad-CAM for top suspicious slices ──
        if gradcam_top_k > 0:
            top_k_records = self._attach_gradcam(
                top_k_records, n=gradcam_top_k
            )

        total_ms = (time.perf_counter() - t0) * 1000

        result = VolumeInferenceResult(
            request_id=request_id,
            folder_path=folder_path,
            patient_id=series.patient_id,
            model_name=self._model_name,
            aggregation_method=agg_label,
            volume_probability=volume_prob,
            volume_prediction=volume_pred,
            volume_confidence=volume_conf,
            classification_threshold=self._threshold,
            n_slices_total=len(series.slices),
            n_slices_processed=len(slice_records),
            slice_records=slice_records,
            top_suspicious=top_k_records,
            total_time_ms=total_ms,
            metadata={
                "study_date": series.study_date,
                "series_uid": series.series_uid,
                "pixel_spacing": series.pixel_spacing,
                "slice_thickness": series.slice_thickness,
            },
        )

        logger.info(
            "Volume inference complete | request=%s | prediction=%s | "
            "prob=%.4f | confidence=%s | time=%.1fms",
            request_id[:8],
            volume_pred,
            volume_prob,
            volume_conf,
            total_ms,
        )
        return result

    # ── Slice-level inference ─────────────────

    def _run_slice_inference(
        self,
        slices: List[DICOMSlice],
    ) -> List[SliceInferenceRecord]:
        """
        Call the existing 2D pipeline for each slice.

        Each DICOMSlice already holds the path to its original .dcm file,
        so we pass that directly — no temp files needed.
        InputHandler natively handles DICOM loading and HU conversion.
        """
        records: List[SliceInferenceRecord] = []

        for sl in slices:
            try:
                pr = self._pipeline.run(
                    sl.file_path,              # original .dcm — accepted natively
                    model_name=self._model_name,
                    generate_gradcam=False,    # deferred to top-k only
                    max_candidates=1,          # one nodule candidate per slice
                )

                if not pr.predictions:
                    logger.debug(
                        "No prediction for slice %d — skipped.", sl.slice_index
                    )
                    continue

                records.append(SliceInferenceRecord(
                    slice_index=sl.slice_index,
                    z_position=sl.z_position,
                    file_path=sl.file_path,
                    pipeline_result=pr,
                    primary_prediction=pr.primary_prediction,
                    has_gradcam=False,
                ))

                logger.debug(
                    "Slice %03d | z=%+.1f | %s | prob=%.4f",
                    sl.slice_index,
                    sl.z_position,
                    pr.primary_prediction.prediction,
                    pr.primary_prediction.probability,
                )

            except Exception as exc:
                logger.warning(
                    "Inference failed for slice %d: %s", sl.slice_index, exc
                )

        return records

    # ── Deferred Grad-CAM ─────────────────────

    def _attach_gradcam(
        self,
        records: List[SliceInferenceRecord],
        n: int,
    ) -> List[SliceInferenceRecord]:
        """
        Re-run the pipeline on the top-n slices with Grad-CAM enabled.

        We re-run rather than caching intermediates to avoid keeping
        large tensors alive during the full slice loop.
        The original .dcm path is passed directly — no temp files needed.
        """
        gradcam_targets = records[:n]

        for rec in gradcam_targets:
            try:
                pr = self._pipeline.run(
                    rec.file_path,             # original .dcm — accepted natively
                    model_name=self._model_name,
                    generate_gradcam=True,
                    max_candidates=1,
                )
                rec.pipeline_result = pr
                rec.has_gradcam = bool(pr.gradcam_overlays)
                logger.debug("Grad-CAM generated for slice %d.", rec.slice_index)
            except Exception as exc:
                logger.warning(
                    "Grad-CAM failed for slice %d: %s", rec.slice_index, exc
                )

        return records

    # ── Batch volume inference ─────────────────

    def run_batch_volumes(
        self,
        folder_paths: List[str],
        **kwargs,
    ) -> List[VolumeInferenceResult]:
        """
        Process multiple patient CT series sequentially.

        All kwargs are forwarded to ``run_volume()``.
        Failed volumes are logged but do not crash the batch.
        """
        results = []
        for i, folder in enumerate(folder_paths):
            logger.info(
                "Batch progress: %d / %d | %s",
                i + 1,
                len(folder_paths),
                os.path.basename(folder),
            )
            try:
                result = self.run_volume(folder, **kwargs)
                results.append(result)
            except Exception as exc:
                logger.error("Failed on %s: %s", folder, exc)
                results.append(
                    self._empty_result(
                        str(uuid.uuid4()), None, folder, kwargs.get("aggregation", "top_k")
                    )
                )
        return results

    # ── Cleanup ───────────────────────────────

    def shutdown(self) -> None:
        self._pipeline.shutdown()
        logger.info("VolumeInferenceEngine shut down.")

    # ── Helpers ───────────────────────────────

    def _empty_result(
        self,
        request_id: str,
        series: Optional[DICOMSeries],
        folder_path: str,
        aggregation: str,
    ) -> VolumeInferenceResult:
        return VolumeInferenceResult(
            request_id=request_id,
            folder_path=folder_path,
            patient_id=series.patient_id if series else "",
            model_name=self._model_name,
            aggregation_method=aggregation,
            volume_probability=0.0,
            volume_prediction="Benign",
            volume_confidence="Low",
            n_slices_total=len(series.slices) if series else 0,
            n_slices_processed=0,
            metadata={"error": "inference_failed"},
        )