"""
evaluation.py
-------------
Comprehensive medical imaging evaluation metrics.

Metrics (from plan diagram):
  - FROC: Sensitivity at various FP/scan rates
  - AUC-ROC: Patient-level, target ≥ 0.92
  - Sensitivity: Target ≥ 0.90, no missed cancer
  - Dice score: Segmentation quality, target ≥ 0.75
  - ECE: Expected Calibration Error, target < 0.05

Usage:
  python evaluation.py --data_dir /path/to/processed_3d \\
                       --ckpt_dir /path/to/checkpoints \\
                       --out_dir /path/to/results
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    confusion_matrix, precision_recall_curve,
)
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

from unet3d import UNet3D
from resnet3d import ResNet3D10
from monai_dataset_3d import build_seg_dataloaders, build_cls_dataloaders


# ──────────────────────────────────────────────
#  1. FROC — Free-Response ROC
# ──────────────────────────────────────────────

def compute_froc(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    fp_rates: List[float] = [0.125, 0.25, 0.5, 1, 2, 4, 8],
) -> Dict:
    """
    Compute FROC: sensitivity at specified FP/scan rates.

    For binary classification at patient level, this approximates
    the FROC by computing sensitivity at various FPR operating points.
    """
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Interpolate sensitivity at each target FP rate
    froc_points = {}
    n_negatives = (all_labels == 0).sum()

    for fp_target in fp_rates:
        target_fpr = fp_target / max(n_negatives, 1)
        target_fpr = min(target_fpr, 1.0)

        # Find closest FPR point
        idx = np.searchsorted(fpr, target_fpr)
        idx = min(idx, len(tpr) - 1)
        sensitivity = tpr[idx]
        froc_points[f"sens@{fp_target}FP/scan"] = float(sensitivity)

    return froc_points


def plot_froc(all_labels: np.ndarray, all_probs: np.ndarray,
              out_dir: str) -> None:
    """Plot FROC curve and save."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    n_neg = (all_labels == 0).sum()

    # Convert FPR to FP/scan
    fp_per_scan = fpr * n_neg

    plt.figure(figsize=(8, 6))
    plt.plot(fp_per_scan, tpr, 'b-', linewidth=2)
    plt.xlabel("Average FP per scan")
    plt.ylabel("Sensitivity (TPR)")
    plt.title("FROC Curve — Patient-level Detection")
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 10])
    plt.ylim([0, 1.05])

    # Mark key operating points
    for fp_target in [0.125, 0.25, 0.5, 1, 2, 4]:
        idx = np.searchsorted(fp_per_scan, fp_target)
        if idx < len(tpr):
            plt.plot(fp_per_scan[idx], tpr[idx], 'ro', markersize=6)
            plt.annotate(f"{tpr[idx]:.2f}",
                        (fp_per_scan[idx], tpr[idx]),
                        textcoords="offset points", xytext=(10, -5),
                        fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "froc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  2. AUC-ROC
# ──────────────────────────────────────────────

def compute_auc(all_labels: np.ndarray, all_probs: np.ndarray) -> float:
    """Compute patient-level AUC-ROC. Target ≥ 0.92."""
    try:
        return roc_auc_score(all_labels, all_probs)
    except ValueError:
        return 0.0


def plot_roc(all_labels: np.ndarray, all_probs: np.ndarray,
             out_dir: str) -> None:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = compute_auc(all_labels, all_probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2,
             label=f"3D ResNet-10 (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")

    # Target line
    plt.axhline(y=0.90, color='r', linestyle=':', alpha=0.5,
                label="Sensitivity target (0.90)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC Curve — 3D ResNet-10 Classifier")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, "roc_curve_3d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  3. Sensitivity analysis
# ──────────────────────────────────────────────

def compute_sensitivity_specificity(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute sensitivity and specificity. Target sensitivity ≥ 0.90."""
    preds = (all_probs > threshold).astype(int)
    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "missed_cancers": int(fn),
        "threshold": threshold,
    }


# ──────────────────────────────────────────────
#  4. Dice score
# ──────────────────────────────────────────────

def evaluate_dice(model: nn.Module, test_loader,
                  device: torch.device) -> float:
    """
    Compute mean Dice score on the test set. Target ≥ 0.75.
    """
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice_metric(y_pred=preds, y=masks)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    return dice_score


def plot_dice_distribution(model: nn.Module, test_loader,
                           device: torch.device, out_dir: str) -> None:
    """Plot per-sample Dice score distribution."""
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Per-sample Dice
            for i in range(preds.shape[0]):
                pred_flat = preds[i].flatten()
                mask_flat = masks[i].flatten()
                intersection = (pred_flat * mask_flat).sum()
                dice = (2 * intersection) / (pred_flat.sum() + mask_flat.sum() + 1e-8)
                dice_scores.append(dice.item())

    plt.figure(figsize=(8, 5))
    plt.hist(dice_scores, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    plt.axvline(x=np.mean(dice_scores), color='red', linestyle='--',
                label=f"Mean = {np.mean(dice_scores):.3f}")
    plt.axvline(x=0.75, color='orange', linestyle=':',
                label="Target = 0.75")
    plt.xlabel("Dice Score")
    plt.ylabel("Count")
    plt.title("Dice Score Distribution — U-Net Segmentation")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "dice_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  5. ECE — Expected Calibration Error
# ──────────────────────────────────────────────

def compute_ece(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error. Target < 0.05.

    ECE = Σ (|bin_count| / N) * |accuracy_in_bin - confidence_in_bin|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(all_labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (all_probs > lo) & (all_probs <= hi)
        count = mask.sum()

        if count == 0:
            continue

        bin_accuracy = all_labels[mask].mean()
        bin_confidence = all_probs[mask].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_calibration(all_labels: np.ndarray, all_probs: np.ndarray,
                     out_dir: str, n_bins: int = 15) -> None:
    """Plot reliability diagram (calibration curve)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (all_probs > lo) & (all_probs <= hi)
        count = mask.sum()

        if count == 0:
            continue

        bin_centers.append((lo + hi) / 2)
        bin_accuracies.append(all_labels[mask].mean())
        bin_counts.append(count)

    ece = compute_ece(all_labels, all_probs, n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.6,
            color='steelblue', edgecolor='white', label="Model")
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Perfect calibration")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title(f"Calibration Plot — ECE = {ece:.4f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    ax2.hist(all_probs, bins=n_bins, color='gray', edgecolor='white', alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    path = os.path.join(out_dir, "calibration_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  Summary table
# ──────────────────────────────────────────────

def print_evaluation_summary(metrics: Dict) -> None:
    """Print a formatted evaluation summary with target comparisons."""
    targets = {
        "auc": ("≥ 0.92", 0.92),
        "sensitivity": ("≥ 0.90", 0.90),
        "dice": ("≥ 0.75", 0.75),
        "ece": ("< 0.05", 0.05),
    }

    print("\n" + "=" * 60)
    print("  Evaluation Summary — Medical Imaging Metrics")
    print("=" * 60)

    for metric, (target_str, target_val) in targets.items():
        if metric not in metrics:
            continue
        val = metrics[metric]
        if metric == "ece":
            passed = val < target_val
        else:
            passed = val >= target_val
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {metric:15s}: {val:.4f}  (target {target_str})  [{status}]")

    # Additional metrics
    for key in ["specificity", "f1", "missed_cancers"]:
        if key in metrics:
            print(f"  {key:15s}: {metrics[key]}")

    print("=" * 60)


# ──────────────────────────────────────────────
#  Main evaluation pipeline
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}")

    metrics = {}

    # ── Segmentation evaluation (Dice) ────────
    print("\n" + "=" * 50)
    print("  Phase 2: U-Net Segmentation Evaluation")
    print("=" * 50)

    unet_path = os.path.join(args.ckpt_dir, "unet3d_best.pth")
    if os.path.exists(unet_path):
        _, _, seg_test_loader = build_seg_dataloaders(
            args.data_dir, batch_size=4, num_workers=args.num_workers
        )

        unet = UNet3D().to(device)
        unet.load_state_dict(torch.load(unet_path, map_location=device))
        unet.eval()

        dice = evaluate_dice(unet, seg_test_loader, device)
        metrics["dice"] = dice
        print(f"  Test Dice: {dice:.4f}  (target ≥ 0.75)")

        plot_dice_distribution(unet, seg_test_loader, device, args.out_dir)
        del unet
    else:
        print(f"  ⚠ U-Net checkpoint not found: {unet_path}")

    # ── Classification evaluation ─────────────
    print("\n" + "=" * 50)
    print("  Phase 3: ResNet-10 Classification Evaluation")
    print("=" * 50)

    resnet_path = os.path.join(args.ckpt_dir, "resnet3d_calibrated.pth")
    if not os.path.exists(resnet_path):
        resnet_path = os.path.join(args.ckpt_dir, "resnet3d_best.pth")

    if os.path.exists(resnet_path):
        _, _, cls_test_loader, _ = build_cls_dataloaders(
            args.data_dir, batch_size=16, num_workers=args.num_workers
        )

        resnet = ResNet3D10().to(device)
        resnet.load_state_dict(torch.load(resnet_path, map_location=device))
        resnet.eval()

        # Collect predictions
        all_labels, all_probs = [], []
        with torch.no_grad():
            for batch in cls_test_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"]
                probs  = torch.sigmoid(resnet(imgs)).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_labels.extend(labels.numpy())

        all_labels = np.array(all_labels)
        all_probs  = np.array(all_probs)

        # AUC-ROC
        auc = compute_auc(all_labels, all_probs)
        metrics["auc"] = auc
        print(f"  AUC-ROC: {auc:.4f}  (target ≥ 0.92)")

        # Sensitivity & Specificity
        sens_spec = compute_sensitivity_specificity(all_labels, all_probs)
        metrics.update(sens_spec)
        print(f"  Sensitivity: {sens_spec['sensitivity']:.4f}  (target ≥ 0.90)")
        print(f"  Specificity: {sens_spec['specificity']:.4f}")
        print(f"  Missed cancers: {sens_spec['missed_cancers']}")

        # F1
        preds = (all_probs > 0.5).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        metrics["f1"] = f1
        print(f"  F1 score: {f1:.4f}")

        # ECE
        ece = compute_ece(all_labels, all_probs)
        metrics["ece"] = ece
        print(f"  ECE: {ece:.4f}  (target < 0.05)")

        # FROC
        froc = compute_froc(all_labels, all_probs)
        metrics.update(froc)
        print(f"  FROC points: {froc}")

        # Plots
        plot_roc(all_labels, all_probs, args.out_dir)
        plot_froc(all_labels, all_probs, args.out_dir)
        plot_calibration(all_labels, all_probs, args.out_dir)

        del resnet
    else:
        print(f"  ⚠ ResNet checkpoint not found: {resnet_path}")

    # ── Summary ────────────────────────────────
    print_evaluation_summary(metrics)

    # Save all metrics
    # Convert numpy types for JSON serialization
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            json_metrics[k] = float(v)
        elif isinstance(v, np.ndarray):
            json_metrics[k] = v.tolist()
        else:
            json_metrics[k] = v

    metrics_path = os.path.join(args.out_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"\n  All metrics saved → {metrics_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Medical imaging evaluation")
    p.add_argument("--data_dir",    required=True,
                   help="Processed 3D volume directory")
    p.add_argument("--ckpt_dir",    required=True,
                   help="Directory containing trained checkpoints")
    p.add_argument("--out_dir",     default="results",
                   help="Output directory for plots and metrics")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
