#!/usr/bin/env python3
"""
demo_inference.py
-----------------
Quick demonstration of the inference pipeline.

Can be run locally or in Google Colab.

Usage (local):
    python demo_inference.py \
        --input  /path/to/image.dcm \
        --ckpt   /path/to/checkpoints \
        --model  efficientnet_b2

Usage (Colab):
    See the "Google Colab Setup" section below for step-by-step instructions.
"""

import argparse
import json
import os
import sys

# Add the project root to sys.path so `models.py` can be found
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from inference import InferencePipeline


def main(args):
    # ── 1. Initialise the pipeline ────────────────────────────
    print("\n[1/4] Initialising pipeline...")
    pipeline = InferencePipeline(
        checkpoint_dir=args.ckpt,
        profile=args.profile,
        model_name=args.model,
    )

    # ── 2. Load model and warm up GPU ─────────────────────────
    print("[2/4] Loading model and warming up...")
    pipeline.setup([args.model])

    # ── 3. Run inference ──────────────────────────────────────
    print(f"[3/4] Running inference on: {args.input}")
    result = pipeline.run(
        input_path=args.input,
        model_name=args.model,
        generate_gradcam=not args.no_gradcam,
    )

    # ── 4. Display results ────────────────────────────────────
    print(f"\n[4/4] Results:")
    print(result.summary())

    # Save Grad-CAM overlay if generated
    if result.gradcam_overlays:
        out_path = os.path.join(
            args.output_dir,
            f"gradcam_{args.model}_{os.path.splitext(os.path.basename(args.input))[0]}.png",
        )
        os.makedirs(args.output_dir, exist_ok=True)
        result.save_gradcam(out_path)
        print(f"\n  Grad-CAM saved → {out_path}")

    # Save JSON report
    json_path = os.path.join(args.output_dir, "inference_result.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"  JSON report  → {json_path}")

    # Cleanup
    pipeline.shutdown()
    print("\nDone.\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="Lung nodule classification inference demo"
    )
    p.add_argument(
        "--input", required=True,
        help="Path to a DICOM (.dcm) or image (.jpg/.png) file",
    )
    p.add_argument(
        "--ckpt", default="checkpoints",
        help="Directory containing *_best.pth checkpoint files",
    )
    p.add_argument(
        "--model", default="efficientnet_b2",
        choices=["custom_cnn", "resnet18", "efficientnet_b2", "densenet121"],
        help="Model architecture to use (default: efficientnet_b2)",
    )
    p.add_argument(
        "--profile", default="git_repo",
        choices=["git_repo", "colab"],
        help="Config profile: 'git_repo' (BCE, ImageNet norms) or "
             "'colab' (CE, [0.5] norms). Must match your checkpoint.",
    )
    p.add_argument(
        "--output_dir", default="inference_results",
        help="Directory to save outputs (default: inference_results/)",
    )
    p.add_argument(
        "--no_gradcam", action="store_true",
        help="Skip Grad-CAM generation (faster inference)",
    )
    return p.parse_args()


# ──────────────────────────────────────────────
#  Google Colab Quick Start
# ──────────────────────────────────────────────
#
#  Step 1: Enable GPU
#    Runtime → Change runtime type → T4 GPU → Save
#
#  Step 2: Mount Google Drive
#    from google.colab import drive
#    drive.mount('/content/drive')
#
#  Step 3: Clone your repo
#    !git clone https://github.com/suryaprakashdev/Lung_Cancer_Classification.git
#    %cd Lung_Cancer_Classification
#
#  Step 4: Install deps
#    !pip install -q pydicom opencv-python-headless
#
#  Step 5: Verify GPU
#    import torch
#    print(f"CUDA: {torch.cuda.is_available()}")
#    print(f"GPU:  {torch.cuda.get_device_name(0)}")
#
#  Step 6: Run inference
#    from inference import InferencePipeline
#
#    pipe = InferencePipeline(
#        checkpoint_dir="/content/drive/MyDrive/LIDC-IDRI-Processed",
#        profile="colab",          # ← use "colab" if trained with training_model.py
#        model_name="efficientnet_b2",
#    )
#    pipe.setup()
#
#    result = pipe.run("/content/drive/MyDrive/test_image.dcm")
#    print(result.summary())
#
#    # Display Grad-CAM in notebook
#    from IPython.display import display
#    from PIL import Image as PILImage
#    if result.gradcam_overlays:
#        display(PILImage.fromarray(result.gradcam_overlays[0]))
#
# ──────────────────────────────────────────────

if __name__ == "__main__":
    main(parse_args())
