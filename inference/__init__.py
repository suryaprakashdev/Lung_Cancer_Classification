"""
inference — Production-grade lung nodule classification pipeline.

Modules
───────
  config                  Centralised constants (must match training)
  input_handler           DICOM / JPEG / PNG loading and validation
  inference_preprocessing MONAI-based HU windowing + tensor transforms
  roi_extractor           Annotation-free ROI extraction from full CT slices
  inference_engine        Model loading, caching and prediction
  explainability          Grad-CAM heatmap generation
  pipeline                End-to-end orchestrator
"""

from .pipeline import InferencePipeline, PipelineResult  # noqa: F401
