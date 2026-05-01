"""
resnet3d.py
-----------
Phase 3: 3D ResNet-10 classifier for malignancy prediction.

Architecture (from plan diagram)
─────────────────────────────────
  Input stem:    Conv3d 7×7×7, stride=2, BN3d + ReLU, MaxPool3d
  Body:          4 × ResBlock3d, channels 16→32→64→128
                 BN3d + ReLU, skip connection via addition (not cat)
  Head:          GlobalAvgPool3d → 128-dim vector
                 FC(128→64) → Dropout(0.5) → FC(64→1)
                 Raw logit output + temperature scaling

  ~1.8M parameters

Usage:
  from resnet3d import ResNet3D10
  model = ResNet3D10()
  # Input:  (B, 1, 64, 64, 64)
  # Output: (B, 1) — raw logits (apply sigmoid for probabilities)
"""

import torch
import torch.nn as nn


class ResBlock3D(nn.Module):
    """
    3D residual block with skip connection via addition.

    Conv3d 3×3×3 → BN3d → ReLU → Conv3d 3×3×3 → BN3d → (+skip) → ReLU

    If in_ch ≠ out_ch, a 1×1×1 projection is used for the skip path.
    Downsampling (stride=2) can be applied on the first conv.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_ch)

        # Skip connection: 1×1×1 projection if dimensions change
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # skip add (not concatenation)
        out = self.relu(out)
        return out


class ResNet3D10(nn.Module):
    """
    3D ResNet-10 for binary malignancy classification.

    Architecture matches the plan diagram:
      - Conv3d 7×7×7 input stem
      - 4 × ResBlock3d stages: 16→32→64→128 channels
      - GlobalAvgPool3d → 128-dim → FC(128→64) → Dropout → FC(64→1)
      - ~1.8M trainable parameters

    Input:  (B, 1, 64, 64, 64) — single-channel HU-windowed volume
    Output: (B, 1) — raw logit (apply sigmoid externally)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1,
                 dropout_p: float = 0.5):
        super().__init__()

        # Input stem: Conv3d 7×7×7, stride=2
        # 64→32 spatial
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # 32→16 spatial
        )

        # 4 residual stages (2 blocks per stage → ~1.8M total params)
        # After stem: (B, 16, 16, 16, 16)
        self.layer1 = nn.Sequential(
            ResBlock3D(16, 16, stride=1),
            ResBlock3D(16, 16, stride=1),
        )  # (B, 16, 16, 16, 16)
        self.layer2 = nn.Sequential(
            ResBlock3D(16, 32, stride=2),
            ResBlock3D(32, 32, stride=1),
        )  # (B, 32, 8, 8, 8)
        self.layer3 = nn.Sequential(
            ResBlock3D(32, 64, stride=2),
            ResBlock3D(64, 64, stride=1),
        )  # (B, 64, 4, 4, 4)
        self.layer4 = nn.Sequential(
            ResBlock3D(64, 128, stride=2),
            ResBlock3D(128, 128, stride=1),
        )  # (B, 128, 2, 2, 2)

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool3d(1)   # (B, 128, 1, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(64, num_classes),  # raw logit
        )

        # Temperature scaling parameter (learned during calibration)
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return self.classifier(x)

    def forward_scaled(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling for calibrated probabilities."""
        logits = self.forward(x)
        return logits / self.temperature

    def calibrate_temperature(self, val_loader, device: torch.device,
                               max_iter: int = 50, lr: float = 0.01):
        """
        Learn the temperature parameter on the validation set
        to improve probability calibration (minimize NLL).
        """
        self.eval()
        nll_criterion = nn.BCEWithLogitsLoss()

        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].float().unsqueeze(1).to(device)
                logits = self.forward(images)
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Optimize temperature
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(all_logits / self.temperature, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad_(False)

        print(f"  Calibrated temperature: {self.temperature.item():.4f}")
        return self.temperature.item()


# ──────────────────────────────────────────────
#  Model registry (replaces the old models.py)
# ──────────────────────────────────────────────

MODEL_REGISTRY_3D = {
    "resnet3d_10": ResNet3D10,
}


def get_model_3d(name: str = "resnet3d_10", **kwargs) -> nn.Module:
    """Instantiate a 3D model by name."""
    if name not in MODEL_REGISTRY_3D:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(MODEL_REGISTRY_3D)}")
    return MODEL_REGISTRY_3D[name](**kwargs)


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet3D10().to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet3D10 | params: {params:,}")

    # Forward pass test
    dummy = torch.randn(4, 1, 64, 64, 64).to(device)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Logit range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # Probabilities
    probs = torch.sigmoid(out)
    print(f"Prob range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    # Temperature-scaled
    out_scaled = model.forward_scaled(dummy)
    print(f"Temp-scaled range: [{out_scaled.min().item():.4f}, "
          f"{out_scaled.max().item():.4f}]")
