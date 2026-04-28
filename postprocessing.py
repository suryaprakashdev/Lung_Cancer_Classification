"""
postprocessing.py
-----------------
Stage 3: Evaluate trained models on the held-out test set, generate
         Grad-CAM heatmaps, and produce publication-ready plots.

Outputs (all saved to --out_dir)
─────────────────────────────────
  <model>_metrics.json       – test AUC, acc, F1, sensitivity, specificity
  <model>_confusion.png      – confusion matrix heat-map
  roc_all_models.png         – overlaid ROC curves
  gradcam_<model>_<img>.png  – Grad-CAM overlay + bounding box

Usage:
  python postprocessing.py \
      --data_dir  /path/to/processed_images \
      --ckpt_dir  /path/to/checkpoints \
      --out_dir   /path/to/results \
      --gradcam_image /path/to/any_nodule.png
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, f1_score,
                             confusion_matrix, roc_curve)

from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    NormalizeIntensity,
    RepeatChannel,
    Resize,
    ScaleIntensityRange,
    ToTensor,
)

from monai_dataset import build_dataloaders, LUNG_HU_MIN, LUNG_HU_MAX
from models import get_model, MODEL_REGISTRY


# ──────────────────────────────────────────────
#  Shared inference transform (same as val — MONAI)
# ──────────────────────────────────────────────

def _build_infer_transform_dict():
    """Dict-based MONAI transform (skip EnsureChannelFirstd — we'll reshape manually)."""
    from monai.transforms import (
        Compose, ScaleIntensityRanged,
        Resized, RepeatChanneld, NormalizeIntensityd, ToTensord,
    )
    from monai_dataset import LUNG_HU_MIN, LUNG_HU_MAX
    
    return Compose([
        # NO EnsureChannelFirstd — manually reshape array instead
        ScaleIntensityRanged(
            keys="image",
            a_min=LUNG_HU_MIN, a_max=LUNG_HU_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        Resized(keys="image", spatial_size=(224, 224), mode="bilinear"),
        RepeatChanneld(keys="image", repeats=3),
        NormalizeIntensityd(keys="image", channel_wise=True),
        ToTensord(keys="image"),
    ])

 


# ──────────────────────────────────────────────
#  Model loading helper
# ──────────────────────────────────────────────

def load_model(name: str, ckpt_dir: str, device: torch.device) -> nn.Module:
    path  = os.path.join(ckpt_dir, f"{name}_best.pth")
    model = get_model(name).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"  Loaded {name} ← {path}")
    return model


# ──────────────────────────────────────────────
#  Test evaluation
# ──────────────────────────────────────────────

def test_model(model: nn.Module,
               test_loader,
               device: torch.device) -> Dict:
    """Return test-set metrics and raw arrays for further plotting."""
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"]
            probs  = torch.sigmoid(model(imgs)).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds      = (all_probs > 0.5).astype(int)

    cm         = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc":         roc_auc_score(all_labels, all_probs),
        "acc":         float((preds == all_labels).mean()),
        "f1":          f1_score(all_labels, preds),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "conf_matrix": cm,
        "labels":      all_labels,
        "probs":       all_probs,
    }


# ──────────────────────────────────────────────
#  Confusion matrix plot
# ──────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, name: str, out_dir: str) -> None:
    class_names = ["Benign", "Malignant"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(xticks=range(2), yticks=range(2),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="Actual", xlabel="Predicted",
           title=f"Confusion Matrix — {name}")

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{name}_confusion.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  ROC curve overlay
# ──────────────────────────────────────────────

def plot_roc_curves(results: Dict[str, Dict], out_dir: str) -> None:
    plt.figure(figsize=(7, 6))

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        plt.plot(fpr, tpr, lw=2,
                 label=f"{name}  (AUC = {res['auc']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curves — all models")
    plt.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(out_dir, "roc_all_models.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  Grad-CAM
# ──────────────────────────────────────────────

_GRADCAM_LAYERS = {
    "custom_cnn":      lambda m: m.block4,
    "resnet18":        lambda m: m.layer4,
    "efficientnet_b2": lambda m: m.features[-1],
    "densenet121":     lambda m: m.features.denseblock4,
}


class GradCAMWrapper:
    """
    Lightweight Grad-CAM implementation (no extra dependencies beyond PyTorch).
    Uses pytorch-grad-cam under the hood if installed, otherwise falls back
    to a manual hook-based implementation.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(_, __, output):
            self.activations = output.detach()

        def save_gradient(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(save_activation))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(save_gradient))

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output.backward(torch.ones_like(output))

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def load_image_for_gradcam(image_path: str, device):
    """Load and preprocess image for Grad-CAM."""
    import os
    import cv2
    import numpy as np
    import torch
    from monai_dataset import LUNG_HU_MIN, LUNG_HU_MAX
    
    ext = os.path.splitext(image_path)[1].lower()

    if ext == ".npy":
        img_raw = np.load(image_path).astype(np.float32)
        
        # Display version
        img_display = np.clip(
            (img_raw - LUNG_HU_MIN) / (LUNG_HU_MAX - LUNG_HU_MIN) * 255,
            0, 255,
        ).astype(np.uint8)
        img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_display_rgb, (224, 224))
        
        # ✓ Manually add channel dimension: (H, W) → (1, H, W)
        img_raw_ch = np.expand_dims(img_raw, axis=0)
        
        transform = _build_infer_transform_dict()
        data = transform({"image": img_raw_ch})
        tensor = data["image"]
        
    else:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img_display_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_display_rgb, (224, 224))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # ✓ Manually add channel dimension
        gray_ch = np.expand_dims(gray, axis=0)
        
        transform = _build_infer_transform_dict()
        data = transform({"image": gray_ch})
        tensor = data["image"]

    input_tensor = tensor.unsqueeze(0).to(device)
    return img_display_rgb, img_resized, input_tensor


def get_bbox_from_cam(cam: np.ndarray,
                      threshold: int = 100) -> Tuple[int, int, int, int] | None:
    """Return (x, y, w, h) bounding box of the highest-activation region."""
    cam_u8   = (cam * 255).astype(np.uint8)
    _, thresh = cv2.threshold(cam_u8, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)


def run_gradcam(name:       str,
                model:      nn.Module,
                image_path: str,
                device:     torch.device,
                out_dir:    str) -> None:
    """Produce a side-by-side Original | Grad-CAM figure and save it."""
    original, resized, input_tensor = load_image_for_gradcam(image_path, device)

    with torch.no_grad():
        logit = model(input_tensor)
    prob  = torch.sigmoid(logit).item()
    label = "Malignant" if prob > 0.5 else "Benign"
    print(f"  [{name}] Prediction: {label}  (prob = {prob:.3f})")

    target_layer = _GRADCAM_LAYERS[name](model)
    gradcam      = GradCAMWrapper(model, target_layer)

    model.train()  # enable gradients temporarily
    cam = gradcam.generate(input_tensor)
    model.eval()
    gradcam.remove_hooks()

    cam_resized = cv2.resize(cam, (224, 224))

    # Overlay heatmap on image
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.4 * heatmap_rgb + 0.6 * resized).astype(np.uint8)

    # Draw bounding box
    bbox = get_bbox_from_cam(cam_resized)
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{name} — {label} ({prob:.3f})", fontsize=13)

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis("off")

    plt.tight_layout()
    stem  = Path(image_path).stem
    path  = os.path.join(out_dir, f"gradcam_{name}_{stem}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  Summary table
# ──────────────────────────────────────────────

def print_summary_table(results: Dict[str, Dict]) -> None:
    header = f"{'Model':<20} {'AUC':>6} {'Acc':>6} {'F1':>6} {'Sens':>6} {'Spec':>6}"
    print("\n" + "=" * len(header))
    print("  Test-set results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, r in sorted(results.items(), key=lambda x: -x[1]["auc"]):
        print(f"{name:<20} {r['auc']:>6.3f} {r['acc']:>6.3f} "
              f"{r['f1']:>6.3f} {r['sensitivity']:>6.3f} {r['specificity']:>6.3f}")
    print("=" * len(header) + "\n")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    # Build test loader (we only need test split here)
    _, _, test_loader, _ = build_dataloaders(
        data_dir    = args.data_dir,
        num_workers = args.num_workers,
    )

    models_to_eval = list(MODEL_REGISTRY.keys())
    if args.model != "all":
        models_to_eval = [args.model]

    all_results = {}

    for name in models_to_eval:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {name}")
        print(f"{'='*50}")

        ckpt_path = os.path.join(args.ckpt_dir, f"{name}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠ Checkpoint not found: {ckpt_path}  — skipping.")
            continue

        model = load_model(name, args.ckpt_dir, device)
        res   = test_model(model, test_loader, device)
        all_results[name] = res

        # Save numeric metrics
        metrics = {k: float(v) for k, v in res.items()
                   if k not in ("conf_matrix", "labels", "probs")}
        with open(os.path.join(args.out_dir, f"{name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        plot_confusion_matrix(res["conf_matrix"], name, args.out_dir)

        # Optional Grad-CAM
        if args.gradcam_image:
            run_gradcam(name, model, args.gradcam_image, device, args.out_dir)

    if all_results:
        plot_roc_curves(all_results, args.out_dir)
        print_summary_table(all_results)


def parse_args():
    p = argparse.ArgumentParser(description="Lung nodule postprocessing & evaluation")
    p.add_argument("--data_dir",      required=True,
                   help="Processed image directory (same as used in train.py)")
    p.add_argument("--ckpt_dir",      required=True,
                   help="Directory containing *_best.pth checkpoint files")
    p.add_argument("--out_dir",       default="results",
                   help="Output directory for plots and metrics (default: results/)")
    p.add_argument("--model",         default="all",
                   choices=["all"] + list(MODEL_REGISTRY.keys()))
    p.add_argument("--gradcam_image", default=None,
                   help="Optional: path to a PNG patch for Grad-CAM visualisation")
    p.add_argument("--num_workers",   type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
