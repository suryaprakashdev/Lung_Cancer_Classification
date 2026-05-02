"""
train_classifier.py
-------------------
Phase 3: Train the 3D ResNet-10 classifier for malignancy prediction.

Training configuration (from plan diagram):
  - Loss: BCEWithLogitsLoss + pos_weight
  - Optimizer: AdamW lr=1e-4
  - Scheduler: CosineAnnealingLR
  - Early stopping on validation AUC, patience=10
  - Temperature calibration after training

Optimized for A100 GPU on Colab Pro with:
  ▸ MONAI CacheDataset / PersistentDataset for fast data loading
  ▸ Mixed precision (AMP) training for 2× speedup
  ▸ TF32 matmuls on A100

Usage:
  python train_classifier.py --data_dir /path/to/processed_3d \\
                              --save_dir /path/to/checkpoints \\
                              --epochs 50 \\
                              --batch_size 16
"""

import multiprocessing as mp
import os
import json
import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

from resnet3d import ResNet3D10
from monai_dataset_3d import build_cls_dataloaders


# ──────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────

def evaluate_cls(model: nn.Module,
                 loader,
                 criterion: nn.Module,
                 device: torch.device) -> Dict[str, float]:
    """
    Run one full pass and return metrics:
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

    # Handle edge case where only one class is present
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "loss":        total_loss / max(len(loader), 1),
        "auc":         auc,
        "acc":         float((preds == all_labels).mean()),
        "f1":          f1_score(all_labels, preds, zero_division=0),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
    }


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────

def train_classifier(model:        nn.Module,
                     train_loader,
                     val_loader,
                     pos_weight:   torch.Tensor,
                     device:       torch.device,
                     save_dir:     str,
                     epochs:       int = 50,
                     patience:     int = 10,
                     lr:           float = 1e-4,
                     use_amp:      bool = True) -> float:
    """
    Train the 3D ResNet classifier. Returns best val AUC.

    Checkpoints:
      <save_dir>/resnet3d_best.pth
      <save_dir>/resnet3d_last.pth
      <save_dir>/resnet3d_history.json
    """
    os.makedirs(save_dir, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    amp_dtype = torch.float16 if use_amp and device.type == "cuda" else torch.float32

    best_auc         = 0.0
    patience_counter = 0
    history          = []

    best_path = os.path.join(save_dir, "resnet3d_best.pth")
    last_path = os.path.join(save_dir, "resnet3d_last.pth")

    print(f"\n{'='*55}")
    print(f"  Training: 3D ResNet-10 (Classification)")
    print(f"{'='*55}")
    print(f"  Loss: BCE + pos_wt={pos_weight.item():.2f}")
    print(f"  Optimizer: AdamW lr={lr}")
    print(f"  Patience: {patience}  |  Epochs: {epochs}")
    print(f"  AMP: {use_amp and device.type == 'cuda'}")
    print(f"{'='*55}\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader,
                    desc=f"[ResNet3D] Epoch {epoch+1:02d}/{epochs}",
                    leave=False)

        for batch in loop:
            imgs   = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                loss = criterion(model(imgs), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        val_metrics = evaluate_cls(model, val_loader, criterion, device)

        log = {
            "epoch":       epoch + 1,
            "train_loss":  epoch_loss / max(len(train_loader), 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr":          optimizer.param_groups[0]["lr"],
        }
        history.append(log)

        print(
            f"[ResNet3D] {epoch+1:02d}/{epochs} | "
            f"loss: {log['train_loss']:.4f} | "
            f"AUC: {val_metrics['auc']:.3f} | "
            f"acc: {val_metrics['acc']:.3f} | "
            f"sens: {val_metrics['sensitivity']:.3f} | "
            f"spec: {val_metrics['specificity']:.3f}"
        )

        # Save last checkpoint
        torch.save(model.state_dict(), last_path)

        # Save best checkpoint (on val AUC)
        if val_metrics["auc"] > best_auc:
            best_auc         = val_metrics["auc"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best saved (AUC = {best_auc:.3f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ⚑ Early stopping at epoch {epoch+1}")
            break

        # Persist history
        with open(os.path.join(save_dir, "resnet3d_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n  Best AUC: {best_auc:.3f}")
    return best_auc


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        # Enable TF32 on A100 for faster matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, pos_weight = build_cls_dataloaders(
        data_dir              = args.data_dir,
        batch_size            = args.batch_size,
        num_workers           = args.num_workers,
        cache_rate            = args.cache_rate,
        persistent_cache_dir  = args.cache_dir,
    )

    model = ResNet3D10().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet3D10 parameters: {params:,}")

    best_auc = train_classifier(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        pos_weight   = pos_weight,
        device       = device,
        save_dir     = args.save_dir,
        epochs       = args.epochs,
        patience     = args.patience,
        lr           = args.lr,
        use_amp      = not args.no_amp,
    )

    # ── Temperature calibration ──
    print("\n" + "="*40)
    print("  Temperature calibration")
    print("="*40)
    model.load_state_dict(torch.load(best_path := os.path.join(
        args.save_dir, "resnet3d_best.pth"), map_location=device))
    temp = model.calibrate_temperature(val_loader, device)

    # Save calibrated model
    calibrated_path = os.path.join(args.save_dir, "resnet3d_calibrated.pth")
    torch.save(model.state_dict(), calibrated_path)
    print(f"  Saved calibrated model → {calibrated_path}")

    # ── Test evaluation ──
    print("\n" + "="*40)
    print("  Test-set evaluation")
    print("="*40)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    test_metrics = evaluate_cls(model, test_loader, criterion, device)
    print(f"  AUC:         {test_metrics['auc']:.3f}")
    print(f"  Accuracy:    {test_metrics['acc']:.3f}")
    print(f"  F1:          {test_metrics['f1']:.3f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.3f}")
    print(f"  Specificity: {test_metrics['specificity']:.3f}")

    # Save test metrics
    with open(os.path.join(args.save_dir, "resnet3d_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)


def parse_args():
    auto_workers = max(1, mp.cpu_count() - 1)
    p = argparse.ArgumentParser(description="3D ResNet classifier training")
    p.add_argument("--data_dir",    required=True,
                   help="Processed 3D volume directory (output of preprocessing.py)")
    p.add_argument("--save_dir",    default="checkpoints",
                   help="Where to save model weights (default: checkpoints/)")
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--patience",    type=int, default=10)
    p.add_argument("--batch_size",  type=int, default=16,
                   help="Batch size (default: 16 for A100)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=auto_workers,
                   help=f"DataLoader workers (default: {auto_workers})")
    p.add_argument("--cache_rate",  type=float, default=1.0,
                   help="CacheDataset rate: 1.0=all in RAM (default: 1.0)")
    p.add_argument("--cache_dir",   type=str, default=None,
                   help="PersistentDataset cache dir (survives restarts)")
    p.add_argument("--no_amp",      action="store_true",
                   help="Disable mixed precision training")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
