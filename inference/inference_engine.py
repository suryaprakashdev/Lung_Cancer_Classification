"""
inference_engine.py
-------------------
Model loading, caching, and prediction with structured output.

Supports both checkpoint formats:
  • git_repo  — 1 logit + sigmoid  (BCEWithLogitsLoss)
  • Colab     — 2 logits + softmax (CrossEntropyLoss)
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

# Import from the parent package (models.py in git_repo root)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import get_model

from .config import InferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Structured prediction output."""

    prediction: str           # "Benign" | "Malignant"
    probability: float        # 0.0–1.0 (malignancy probability)
    model_name: str
    confidence: str           # "High" | "Medium" | "Low"
    processing_time_ms: float
    raw_logit: float
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        return {
            "prediction": self.prediction,
            "probability": round(self.probability, 4),
            "confidence": self.confidence,
            "model_name": self.model_name,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "raw_logit": round(self.raw_logit, 4),
            **self.metadata,
        }


class ModelInferenceEngine:
    """
    Load trained models and run predictions with structured output.

    Models are cached in memory after first load — subsequent calls
    reuse the same model without disk I/O.
    """

    def __init__(
        self,
        config: Optional[InferenceConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config or InferenceConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model_cache: Dict[str, nn.Module] = {}
        self._warmup_done = False

        logger.info("InferenceEngine initialised | device=%s", self.device)

    # ── Model loading ─────────────────────────

    def load_model(self, name: str, checkpoint_path: Optional[str] = None) -> nn.Module:
        """
        Load a model by name. Uses cache if already loaded.

        Parameters
        ----------
        name             : model registry key (e.g. "efficientnet_b2")
        checkpoint_path  : explicit path to .pth file, or None to auto-discover
                           from ``self.cfg.checkpoint_dir``
        """
        if name in self._model_cache:
            logger.debug("Model '%s' loaded from cache.", name)
            return self._model_cache[name]

        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.cfg.checkpoint_dir, f"{name}_best.pth"
            )

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        model = get_model(name).to(self.device)
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        self._model_cache[name] = model

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Loaded model '%s' | params=%s | checkpoint=%s",
            name,
            f"{param_count:,}",
            checkpoint_path,
        )
        return model

    def warmup(self, name: Optional[str] = None) -> None:
        """
        Run a dummy forward pass to pre-allocate GPU memory and trigger
        JIT compilation of CUDA kernels.
        """
        model_name = name or self.cfg.default_model
        model = self._model_cache.get(model_name)
        if model is None:
            logger.warning("Cannot warm up '%s' — model not loaded.", model_name)
            return

        dummy = torch.randn(1, 3, 224, 224, device=self.device)
        with torch.no_grad():
            model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self._warmup_done = True
        logger.info("Warmup complete for '%s'.", model_name)

    # ── Prediction ────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        input_tensor: torch.Tensor,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run inference on a preprocessed tensor.

        Parameters
        ----------
        input_tensor : (1, 3, 224, 224) float tensor on the correct device
        model_name   : which model to use (default: cfg.default_model)

        Returns
        -------
        InferenceResult with prediction, probability, and confidence tier
        """
        name = model_name or self.cfg.default_model
        model = self._model_cache.get(name)
        if model is None:
            raise RuntimeError(
                f"Model '{name}' not loaded. Call load_model() first."
            )

        t0 = time.perf_counter()

        # Mixed-precision inference
        if self.cfg.use_amp and self.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                raw_output = model(input_tensor)
        else:
            raw_output = model(input_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Interpret output based on checkpoint format
        probability, raw_logit = self._interpret_output(raw_output)

        # Classification
        threshold = self.cfg.classification_threshold
        prediction = "Malignant" if probability > threshold else "Benign"

        # Confidence tier
        distance = abs(probability - threshold)
        if distance > self.cfg.confidence_high:
            confidence = "High"
        elif distance > self.cfg.confidence_medium:
            confidence = "Medium"
        else:
            confidence = "Low"

        result = InferenceResult(
            prediction=prediction,
            probability=float(probability),
            model_name=name,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
            raw_logit=float(raw_logit),
        )

        logger.info(
            "Prediction | model=%s | result=%s | prob=%.4f | "
            "confidence=%s | time=%.1fms",
            name,
            prediction,
            probability,
            confidence,
            elapsed_ms,
        )
        return result

    def _interpret_output(self, raw_output: torch.Tensor):
        """
        Convert raw model output to malignancy probability.

        BCE mode (git_repo): single logit → sigmoid
        CE mode  (Colab):    2 logits → softmax → class-1 prob
        """
        if self.cfg.output_mode == "bce":
            raw_logit = raw_output.squeeze().item()
            probability = torch.sigmoid(raw_output).squeeze().item()
        elif self.cfg.output_mode == "ce":
            raw_logit = raw_output[0, 1].item()  # malignant logit
            probs = torch.softmax(raw_output, dim=1)
            probability = probs[0, 1].item()
        else:
            raise ValueError(f"Unknown output_mode: {self.cfg.output_mode}")

        return probability, raw_logit

    # ── Cleanup ───────────────────────────────

    def unload_model(self, name: str) -> None:
        """Remove a model from cache and free GPU memory."""
        if name in self._model_cache:
            del self._model_cache[name]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            logger.info("Unloaded model '%s'.", name)

    def unload_all(self) -> None:
        """Remove all cached models."""
        self._model_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("All models unloaded.")
