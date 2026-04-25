"""
explainability.py
-----------------
Grad-CAM heatmap generation for model interpretability.

Refactored from postprocessing.py for production use with:
  • Proper hook cleanup via context manager
  • Thread-safe design (new wrapper per request)
  • Fixed model.train() bug — uses torch.enable_grad() instead
  • Separate raw heatmap and overlay outputs
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .config import PreprocessingConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Grad-CAM target layer registry
# ──────────────────────────────────────────────

GRADCAM_LAYERS: Dict[str, callable] = {
    "custom_cnn":      lambda m: m.block4,
    "resnet18":        lambda m: m.layer4,
    "efficientnet_b2": lambda m: m.features[-1],
    "densenet121":     lambda m: m.features.denseblock4,
}


# ──────────────────────────────────────────────
#  Grad-CAM engine
# ──────────────────────────────────────────────

class GradCAM:
    """
    Lightweight, production-safe Grad-CAM implementation.

    Creates forward/backward hooks on the target layer, computes
    the class activation map, then cleans up hooks automatically.

    Usage
    -----
        gradcam = GradCAM(model, model_name="efficientnet_b2")
        heatmap = gradcam.generate(input_tensor)   # (H, W) float [0, 1]
        gradcam.cleanup()  # or use as context manager
    """

    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name

        if model_name not in GRADCAM_LAYERS:
            raise ValueError(
                f"No Grad-CAM target layer defined for '{model_name}'. "
                f"Available: {list(GRADCAM_LAYERS)}"
            )

        self.target_layer = GRADCAM_LAYERS[model_name](model)
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()

    # ── Hook management ───────────────────────

    def _register_hooks(self):
        def save_activation(_, __, output):
            self._activations = output.detach()

        def save_gradient(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(save_gradient)
        )

    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._activations = None
        self._gradients = None
        logger.debug("Grad-CAM hooks cleaned up for '%s'.", self.model_name)

    # ── Heatmap generation ────────────────────

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute a Grad-CAM heatmap for the given input.

        Parameters
        ----------
        input_tensor : (1, 3, H, W) tensor on the model's device

        Returns
        -------
        cam : (H_feat, W_feat) numpy array normalised to [0, 1]

        Note
        ----
        Uses ``torch.enable_grad()`` instead of ``model.train()``
        to allow gradient computation without enabling dropout
        or changing batch-norm statistics.
        """
        self.model.zero_grad()

        # Forward + backward with gradients enabled (model stays in eval mode)
        with torch.enable_grad():
            input_tensor = input_tensor.requires_grad_(False)
            # Need a fresh tensor that allows gradients through the graph
            x = input_tensor.detach().requires_grad_(True)
            output = self.model(x)
            output.backward(torch.ones_like(output))

        if self._activations is None or self._gradients is None:
            logger.warning("Grad-CAM hooks did not fire. Returning empty map.")
            return np.zeros((7, 7), dtype=np.float32)

        # GAP of gradients over spatial dimensions → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        logger.debug(
            "Generated Grad-CAM | model=%s | cam_shape=%s",
            self.model_name,
            cam.shape,
        )
        return cam


# ──────────────────────────────────────────────
#  Visualisation helpers
# ──────────────────────────────────────────────

def create_overlay(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto the original image.

    Returns
    -------
    overlay : (H, W, 3) uint8 RGB image
    """
    # Resize CAM to match image
    img_resized = cv2.resize(original_image, target_size)
    cam_resized = cv2.resize(cam, target_size)

    # Convert grayscale to 3-channel if needed
    if img_resized.ndim == 2:
        img_rgb = cv2.cvtColor(img_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif img_resized.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_resized.astype(np.uint8)

    # Apply colormap to heatmap
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), colormap
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (alpha * heatmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def get_activation_bbox(
    cam: np.ndarray,
    threshold: int = 100,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract the bounding box of the highest-activation region.

    Returns (x, y, w, h) or None if no region found.
    """
    cam_u8 = (cam * 255).astype(np.uint8)
    _, binary = cv2.threshold(cam_u8, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)


def draw_bbox_on_overlay(
    overlay: np.ndarray,
    cam: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw activation bounding box on the overlay image."""
    cam_resized = cv2.resize(cam, target_size)
    bbox = get_activation_bbox(cam_resized)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
    return overlay
