"""
running.py
----------
Main entry point for the 3D lung nodule detection & classification pipeline.

This script orchestrates the full pipeline on Google Colab with A100 GPU:
  Phase 1: Preprocessing (run separately via preprocessing.py)
  Phase 2: Train U-Net segmentation
  Phase 3: Train ResNet-10 classifier
  Phase 4: Run 3D inference on a DICOM series
  Eval:    Comprehensive medical evaluation metrics

Usage on Colab:
  1. Mount Google Drive
  2. Set paths below
  3. Run cells sequentially
"""

# ──────────────────────────────────────────────
#  Configuration — Update these paths for your setup
# ──────────────────────────────────────────────

# Google Drive paths (update for your folder structure)
GDRIVE_ROOT    = "/content/drive/MyDrive"
RAW_DICOM_DIR  = f"{GDRIVE_ROOT}/LIDC-IDRI"                    # raw DICOM data
PROCESSED_DIR  = f"{GDRIVE_ROOT}/Lung_Cancer_Classification/processed_3d"  # preprocessed 3D volumes
CHECKPOINT_DIR = f"{GDRIVE_ROOT}/Lung_Cancer_Classification/Checkpoints_3D"
RESULTS_DIR    = f"{GDRIVE_ROOT}/Lung_Cancer_Classification/Results_3D"

# Example DICOM series for inference test
TEST_SERIES    = f"{RAW_DICOM_DIR}/LIDC-IDRI-0065/1.3.6.1.4.1.14519.5.2.1.6279.6001.163217526257871051722166468085"


# ══════════════════════════════════════════════
#  Phase 1: Preprocessing (uncomment to run)
# ══════════════════════════════════════════════

# Run this once to extract 3D volumes + seg masks + patient splits:
#
# !python preprocessing.py \
#     --raw_dir "{RAW_DICOM_DIR}" \
#     --out_dir "{PROCESSED_DIR}" \
#     --min_ann 2 \
#     --skip_organise


# ══════════════════════════════════════════════
#  Phase 2: Train U-Net (uncomment to run)
# ══════════════════════════════════════════════

# !python train_unet.py \
#     --data_dir "{PROCESSED_DIR}" \
#     --save_dir "{CHECKPOINT_DIR}" \
#     --epochs 100 \
#     --batch_size 8 \
#     --patience 15


# ══════════════════════════════════════════════
#  Phase 3: Train ResNet Classifier (uncomment to run)
# ══════════════════════════════════════════════

# !python train_classifier.py \
#     --data_dir "{PROCESSED_DIR}" \
#     --save_dir "{CHECKPOINT_DIR}" \
#     --epochs 50 \
#     --batch_size 16 \
#     --patience 10


# ══════════════════════════════════════════════
#  Phase 4: Run 3D Inference
# ══════════════════════════════════════════════

import logging
logging.basicConfig(level=logging.INFO)

from inference_3d import InferencePipeline3D

# Initialize the 3D pipeline
pipeline = InferencePipeline3D(
    unet_checkpoint=f"{CHECKPOINT_DIR}/unet3d_best.pth",
    resnet_checkpoint=f"{CHECKPOINT_DIR}/resnet3d_calibrated.pth",
)

# Run inference on a test volume
result = pipeline.run_volume(
    folder_path=TEST_SERIES,
    aggregation="top_k",
    k=5,
)

print(result.summary())


# ══════════════════════════════════════════════
#  Evaluation: Medical Imaging Metrics
# ══════════════════════════════════════════════

# Run comprehensive evaluation:
#
# !python evaluation.py \
#     --data_dir "{PROCESSED_DIR}" \
#     --ckpt_dir "{CHECKPOINT_DIR}" \
#     --out_dir "{RESULTS_DIR}"


# ══════════════════════════════════════════════
#  Visualisation
# ══════════════════════════════════════════════

import matplotlib.pyplot as plt
import numpy as np


def visualize_3d_results(result, n_candidates=3):
    """Visualize segmentation mask and top candidates from 3D inference."""

    if result.segmentation_mask is None:
        print("No segmentation mask available.")
        return

    mask = result.segmentation_mask
    candidates = result.candidates[:n_candidates]

    if not candidates:
        print("No candidates found.")
        return

    fig, axes = plt.subplots(len(candidates), 3, figsize=(15, 5 * len(candidates)))
    if len(candidates) == 1:
        axes = [axes]

    for row, cand in enumerate(candidates):
        z, y, x = cand.centroid

        # Col 0: Axial slice through centroid
        if z < mask.shape[0]:
            axes[row][0].imshow(mask[z], cmap="hot", alpha=0.7)
            axes[row][0].set_title(
                f"Seg Mask — Axial (z={z})\n"
                f"Candidate {cand.candidate_index}", fontsize=9
            )
            axes[row][0].plot(x, y, 'g+', markersize=15, markeredgewidth=2)
        axes[row][0].axis("off")

        # Col 1: Coronal slice
        if y < mask.shape[1]:
            axes[row][1].imshow(mask[:, y, :], cmap="hot", alpha=0.7)
            axes[row][1].set_title(f"Coronal (y={y})", fontsize=9)
            axes[row][1].plot(x, z, 'g+', markersize=15, markeredgewidth=2)
        axes[row][1].axis("off")

        # Col 2: Prediction info
        info_text = (
            f"Candidate {cand.candidate_index}\n"
            f"─────────────\n"
            f"Centroid: ({z}, {y}, {x})\n"
            f"Volume: {cand.volume_voxels} voxels\n"
            f"Prediction: {cand.prediction}\n"
            f"Probability: {cand.probability:.4f}"
        )
        axes[row][2].text(0.1, 0.5, info_text, transform=axes[row][2].transAxes,
                          fontsize=11, verticalalignment='center',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[row][2].axis("off")

    plt.suptitle(
        f"Patient: {result.patient_id} | "
        f"{result.patient_prediction} | "
        f"prob={result.patient_probability:.4f} | "
        f"{result.patient_confidence} confidence\n"
        f"Candidates: {result.n_candidates_found} | "
        f"Aggregation: {result.aggregation_method}",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig("inference_3d_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → inference_3d_visualization.png")


visualize_3d_results(result, n_candidates=3)