"""
unet3d.py
---------
Phase 2: 3D U-Net for nodule detection and segmentation.

Architecture (from plan diagram)
─────────────────────────────────
  Encoder:    Input (1,64,64,64), Conv3d 3×3×3, BN3d + ReLU,
              MaxPool3d ×4, channels 16→32→64→128, skip connections
  Bottleneck: 256 channels, 4×4×4 spatial, Conv3d ×2, Dropout3d 0.5
  Decoder:    ConvTranspose3d, concatenate skip features,
              channels 128→64→32→16, output (1,64,64,64),
              sigmoid voxel mask
  Loss:       DiceCELoss (Dice + CrossEntropy)

Usage:
  from unet3d import UNet3D
  model = UNet3D()
  # Input:  (B, 1, 64, 64, 64)
  # Output: (B, 1, 64, 64, 64) — sigmoid probabilities
"""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two consecutive Conv3d 3×3×3 → BN3d → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock3D followed by MaxPool3d (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)   # skip connection output
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """ConvTranspose3d (upsample) → concatenate skip → ConvBlock3D."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle potential size mismatch (±1 voxel from rounding)
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = nn.functional.pad(x, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2,
            ])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for voxel-level nodule segmentation.

    Architecture:
      Encoder:    1 → 16 → 32 → 64 → 128  (4 downsampling levels)
      Bottleneck: 128 → 256 → 256           (2× Conv3d, Dropout3d 0.5)
      Decoder:    256+128 → 128 → 64 → 32 → 16  (4 upsampling levels)
      Output:     16 → 1 (sigmoid)

    Input:  (B, 1, 64, 64, 64)
    Output: (B, 1, 64, 64, 64) — voxel-level probabilities

    Spatial dimensions through encoder:
      64³ → 32³ → 16³ → 8³ → 4³ (bottleneck)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 features: tuple = (16, 32, 64, 128)):
        super().__init__()

        # Encoder path
        self.enc1 = EncoderBlock(in_channels, features[0])  # 1→16,   64→32
        self.enc2 = EncoderBlock(features[0], features[1])  # 16→32,  32→16
        self.enc3 = EncoderBlock(features[1], features[2])  # 32→64,  16→8
        self.enc4 = EncoderBlock(features[2], features[3])  # 64→128, 8→4

        # Bottleneck: 256 channels at 4×4×4 spatial
        self.bottleneck = nn.Sequential(
            ConvBlock3D(features[3], features[3] * 2),  # 128→256
            nn.Dropout3d(0.5),
        )

        # Decoder path
        self.dec4 = DecoderBlock(features[3] * 2, features[3], features[3])  # 256→128
        self.dec3 = DecoderBlock(features[3], features[2], features[2])       # 128→64
        self.dec2 = DecoderBlock(features[2], features[1], features[1])       # 64→32
        self.dec1 = DecoderBlock(features[1], features[0], features[0])       # 32→16

        # Final 1×1×1 conv → single channel output
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip1, x = self.enc1(x)   # skip: (B,16,64,64,64), x: (B,16,32,32,32)
        skip2, x = self.enc2(x)   # skip: (B,32,32,32,32), x: (B,32,16,16,16)
        skip3, x = self.enc3(x)   # skip: (B,64,16,16,16), x: (B,64, 8, 8, 8)
        skip4, x = self.enc4(x)   # skip: (B,128,8,8,8),   x: (B,128,4,4,4)

        # Bottleneck
        x = self.bottleneck(x)     # (B, 256, 4, 4, 4)

        # Decoder
        x = self.dec4(x, skip4)    # (B, 128, 8, 8, 8)
        x = self.dec3(x, skip3)    # (B, 64, 16, 16, 16)
        x = self.dec2(x, skip2)    # (B, 32, 32, 32, 32)
        x = self.dec1(x, skip1)    # (B, 16, 64, 64, 64)

        # Output: raw logits (sigmoid applied in loss or at inference)
        return self.final_conv(x)  # (B, 1, 64, 64, 64)


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D().to(device)

    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"UNet3D | params: {params:,}")

    # Forward pass test
    dummy = torch.randn(2, 1, 64, 64, 64).to(device)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # Apply sigmoid for probabilities
    probs = torch.sigmoid(out)
    print(f"Prob range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
