"""
train.py
--------
Stage 2: Train and validate all four model architectures.

Models trained
──────────────
  custom_cnn        LungNoduleCNNv2  (from scratch)
  resnet18           ImageNet pre-trained
  efficientnet_b2    ImageNet pre-trained
  densenet121        ImageNet pre-trained

Each model gets:
  • AdamW optimiser  (lr=1e-4, weight_decay=1e-3)
  • CosineAnnealingLR scheduler
  • BCEWithLogitsLoss with pos_weight for class imbalance
  • Early stopping (patience=10 on val AUC)
  • Best and last checkpoints saved under --save_dir

Usage:
  python train.py --data_dir /path/to/processed_images \
                  --save_dir /path/to/checkpoints \
                  --epochs 30 \
                  --patience 10 \
                  --batch_size 32
"""

import os
import json
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

from monai_dataset import build_dataloaders
from models  import get_model, MODEL_REGISTRY


# ──────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────

def evaluate(model: nn.Module,
             loader,
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    Run one full pass over `loader` and return a metrics dict:
      loss, auc, acc, f1, sensitivity, specificity
    """
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].float().unsqueeze(1).to(device)
            out    = model(imgs)
            total_loss += criterion(out, labels).item()
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds      = (all_probs > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    return {
        "loss":        total_loss / len(loader),
        "auc":         roc_auc_score(all_labels, all_probs),
        "acc":         float((preds == all_labels).mean()),
        "f1":          f1_score(all_labels, preds),
        "sensitivity": tp / (tp + fn),   # recall for malignant class
        "specificity": tn / (tn + fp),
    }


# ──────────────────────────────────────────────
#  Training loop (single model)
# ──────────────────────────────────────────────

def train_model(name:         str,
                model:        nn.Module,
                train_loader,
                val_loader,
                pos_weight:   torch.Tensor,
                device:       torch.device,
                save_dir:     str,
                epochs:       int = 30,
                patience:     int = 10) -> float:
    """
    Train `model` and return the best validation AUC achieved.

    Checkpoints saved to:
      <save_dir>/<name>_best.pth   – weights at best val AUC
      <save_dir>/<name>_last.pth   – weights at final epoch
      <save_dir>/<name>_history.json
    """
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc         = 0.0
    patience_counter = 0
    history          = []

    best_path = os.path.join(save_dir, f"{name}_best.pth")
    last_path = os.path.join(save_dir, f"{name}_last.pth")

    print(f"\n{'='*55}")
    print(f"  Training: {name}")
    print(f"{'='*55}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader,
                    desc=f"[{name}] Epoch {epoch+1:02d}/{epochs}",
                    leave=False)

        for batch in loop:
            imgs   = batch["image"].to(device)
            labels = batch["label"].float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device)

        log = {
            "epoch":       epoch + 1,
            "train_loss":  epoch_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(log)

        print(
            f"[{name}] {epoch+1:02d}/{epochs} | "
            f"loss: {log['train_loss']:.4f} | "
            f"AUC: {val_metrics['auc']:.3f} | "
            f"acc: {val_metrics['acc']:.3f} | "
            f"sens: {val_metrics['sensitivity']:.3f} | "
            f"spec: {val_metrics['specificity']:.3f}"
        )

        # Always save last checkpoint
        torch.save(model.state_dict(), last_path)

        # Save best checkpoint and reset patience
        if val_metrics["auc"] > best_auc:
            best_auc         = val_metrics["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best saved  (AUC = {best_auc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ⚑ Early stopping at epoch {epoch+1}")
            break

        # Persist history after every epoch (safe against crashes)
        with open(os.path.join(save_dir, f"{name}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"  Best AUC: {best_auc:.3f}\n")
    return best_auc


# ──────────────────────────────────────────────
#  Main: train all models
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, pos_weight = build_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # Save test_loader indices for reproducible evaluation in postprocessing.py
    torch.save(test_loader, os.path.join(args.save_dir, "test_loader.pth"))

    models_to_train = list(MODEL_REGISTRY.keys())
    if args.model != "all":
        models_to_train = [args.model]

    results = {}
    for name in models_to_train:
        model = get_model(name).to(device)
        auc   = train_model(
            name         = name,
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            pos_weight   = pos_weight,
            device       = device,
            save_dir     = args.save_dir,
            epochs       = args.epochs,
            patience     = args.patience,
        )
        results[name] = auc

    print("\n" + "="*40)
    print("  Final validation AUC comparison")
    print("="*40)
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {auc:.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="Lung nodule classifier training")
    p.add_argument("--data_dir",    required=True,
                   help="Processed image directory (output of preprocessing.py)")
    p.add_argument("--save_dir",    default="checkpoints",
                   help="Where to save model weights and history (default: checkpoints/)")
    p.add_argument("--model",       default="all",
                   choices=["all"] + list(MODEL_REGISTRY.keys()),
                   help="Which model to train (default: all)")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
