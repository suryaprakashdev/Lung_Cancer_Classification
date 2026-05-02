"""
train_unet.py
-------------
Phase 2: Train the 3D U-Net for nodule segmentation.

Uses DiceCELoss (Dice + CrossEntropy) as specified in the plan.
Optimized for A100 GPU on Colab Pro with:
  ▸ MONAI CacheDataset / PersistentDataset for fast data loading
  ▸ Mixed precision (AMP) training for 2× speedup on A100
  ▸ Auto num_workers tuned to CPU cores

Usage:
  python train_unet.py --data_dir /path/to/processed_3d \\
                       --save_dir /path/to/checkpoints \\
                       --epochs 100 \\
                       --batch_size 8
"""

import multiprocessing as mp
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from tqdm import tqdm

from unet3d import UNet3D
from monai_dataset_3d import build_seg_dataloaders


# ──────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────

def evaluate_seg(model: nn.Module,
                 loader,
                 criterion,
                 device: torch.device) -> dict:
    """Run one full pass and return loss + Dice score."""
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks  = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            # Dice metric expects binary predictions
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice_metric(y_pred=preds, y=masks)

    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()

    return {
        "loss": total_loss / max(len(loader), 1),
        "dice": dice_score,
    }


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────

def train_unet(model: nn.Module,
               train_loader,
               val_loader,
               device: torch.device,
               save_dir: str,
               epochs: int = 100,
               patience: int = 15,
               lr: float = 1e-3,
               use_amp: bool = True) -> float:
    """
    Train the 3D U-Net and return the best validation Dice score.

    Checkpoints:
      <save_dir>/unet3d_best.pth
      <save_dir>/unet3d_last.pth
      <save_dir>/unet3d_history.json
    """
    os.makedirs(save_dir, exist_ok=True)

    # DiceCELoss: combined Dice + CrossEntropy (from plan)
    criterion = DiceCELoss(
        to_onehot_y=False,
        sigmoid=True,       # apply sigmoid inside the loss
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP scaler for mixed precision (2× speedup on A100)
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    amp_dtype = torch.float16 if use_amp and device.type == "cuda" else torch.float32

    best_dice        = 0.0
    patience_counter = 0
    history          = []

    best_path = os.path.join(save_dir, "unet3d_best.pth")
    last_path = os.path.join(save_dir, "unet3d_last.pth")

    print(f"\n{'='*55}")
    print(f"  Training: 3D U-Net (Segmentation)")
    print(f"{'='*55}")
    print(f"  Loss: DiceCELoss  |  Optimizer: AdamW lr={lr}")
    print(f"  Patience: {patience}  |  Epochs: {epochs}")
    print(f"  AMP: {use_amp and device.type == 'cuda'}")
    print(f"{'='*55}\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader,
                    desc=f"[U-Net] Epoch {epoch+1:03d}/{epochs}",
                    leave=False)

        for batch in loop:
            images = batch["image"].to(device, non_blocking=True)
            masks  = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        val_metrics = evaluate_seg(model, val_loader, criterion, device)

        log = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(log)

        print(
            f"[U-Net] {epoch+1:03d}/{epochs} | "
            f"train_loss: {log['train_loss']:.4f} | "
            f"val_loss: {val_metrics['loss']:.4f} | "
            f"val_dice: {val_metrics['dice']:.4f}"
        )

        # Save last checkpoint
        torch.save(model.state_dict(), last_path)

        # Save best checkpoint
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best saved (Dice = {best_dice:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ⚑ Early stopping at epoch {epoch+1}")
            break

        # Save history after every epoch
        with open(os.path.join(save_dir, "unet3d_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n  Best Dice: {best_dice:.4f}")
    return best_dice


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

    train_loader, val_loader, test_loader = build_seg_dataloaders(
        data_dir              = args.data_dir,
        batch_size            = args.batch_size,
        num_workers           = args.num_workers,
        cache_rate            = args.cache_rate,
        persistent_cache_dir  = args.cache_dir,
    )

    model = UNet3D().to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet3D parameters: {params:,}")

    best_dice = train_unet(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        save_dir     = args.save_dir,
        epochs       = args.epochs,
        patience     = args.patience,
        lr           = args.lr,
        use_amp      = not args.no_amp,
    )

    # Evaluate on test set
    print("\n" + "="*40)
    print("  Test-set evaluation")
    print("="*40)
    criterion = DiceCELoss(to_onehot_y=False, sigmoid=True)
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, "unet3d_best.pth"),
        map_location=device,
    ))
    test_metrics = evaluate_seg(model, test_loader, criterion, device)
    print(f"  Test Dice:  {test_metrics['dice']:.4f}")
    print(f"  Test Loss:  {test_metrics['loss']:.4f}")


def parse_args():
    auto_workers = max(1, mp.cpu_count() - 1)
    p = argparse.ArgumentParser(description="3D U-Net segmentation training")
    p.add_argument("--data_dir",    required=True,
                   help="Processed 3D volume directory (output of preprocessing.py)")
    p.add_argument("--save_dir",    default="checkpoints",
                   help="Where to save model weights (default: checkpoints/)")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=8,
                   help="Batch size (default: 8 for A100)")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=auto_workers,
                   help=f"DataLoader workers (default: {auto_workers})")
    p.add_argument("--cache_rate",  type=float, default=1.0,
                   help="CacheDataset rate: 1.0=all in RAM, 0.5=half (default: 1.0)")
    p.add_argument("--cache_dir",   type=str, default=None,
                   help="PersistentDataset cache dir (survives restarts)")
    p.add_argument("--no_amp",      action="store_true",
                   help="Disable mixed precision training")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
