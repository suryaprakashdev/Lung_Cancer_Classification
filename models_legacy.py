"""
models.py
---------
All model architectures used in the lung nodule malignancy classification
pipeline.

Architectures
─────────────
  LungNoduleCNNv2   – Custom 4-block CNN with GAP + Grad-CAM support
  build_resnet18     – ImageNet-pretrained ResNet-18 fine-tuned for binary output
  build_efficientnet_b2 – ImageNet-pretrained EfficientNet-B2
  build_densenet121  – ImageNet-pretrained DenseNet-121

All heads output a single raw logit (no sigmoid) for use with
BCEWithLogitsLoss.  Apply torch.sigmoid() at inference time.

Grad-CAM target layers
──────────────────────
  LungNoduleCNNv2   → model.block4
  ResNet-18         → model.layer4
  EfficientNet-B2   → model.features[-1]
  DenseNet-121      → model.features.denseblock4
"""

import torch
import torch.nn as nn
from torchvision import models


# ──────────────────────────────────────────────
#  Custom CNN
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool (halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LungNoduleCNNv2(nn.Module):
    """
    4-block CNN with Global Average Pooling.

    Input:  (B, 3, 224, 224)
    Spatial after 4 MaxPools: 224 → 112 → 56 → 28 → 14
    GAP collapses 256×14×14 → 256 (vs 50,176 with Flatten → less overfitting)

    Grad-CAM target: self.block4
    """

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()

        self.block1 = ConvBlock(3,   32)   # 224 → 112
        self.block2 = ConvBlock(32,  64)   # 112 → 56
        self.block3 = ConvBlock(64,  128)  # 56  → 28
        self.block4 = ConvBlock(128, 256)  # 28  → 14  ← Grad-CAM here

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(128, 1),             # raw logit
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)    # (B, 256, 14, 14)
        x = self.gap(x)
        return self.classifier(x)


# ──────────────────────────────────────────────
#  Pretrained model factories
# ──────────────────────────────────────────────

def build_resnet18(dropout_p: float = 0.5) -> nn.Module:
    """
    ResNet-18 fine-tuned for binary classification.
    Grad-CAM target: model.layer4
    """
    m = models.resnet18(weights="IMAGENET1K_V1")
    m.fc = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(m.fc.in_features, 1),
    )
    return m


def build_efficientnet_b2(dropout_p: float = 0.5) -> nn.Module:
    """
    EfficientNet-B2 fine-tuned for binary classification.
    Grad-CAM target: model.features[-1]
    """
    m = models.efficientnet_b2(weights="IMAGENET1K_V1")
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(in_features, 1),
    )
    return m


def build_densenet121(dropout_p: float = 0.5) -> nn.Module:
    """
    DenseNet-121 fine-tuned for binary classification.
    Grad-CAM target: model.features.denseblock4
    """
    m = models.densenet121(weights="IMAGENET1K_V1")
    m.classifier = nn.Sequential(
        nn.Dropout(dropout_p),
        nn.Linear(m.classifier.in_features, 1),
    )
    return m


# ──────────────────────────────────────────────
#  Registry – easy access by name string
# ──────────────────────────────────────────────

MODEL_REGISTRY = {
    "custom_cnn":      LungNoduleCNNv2,
    "resnet18":        build_resnet18,
    "efficientnet_b2": build_efficientnet_b2,
    "densenet121":     build_densenet121,
}


def get_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by name from MODEL_REGISTRY."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


# ──────────────────────────────────────────────
#  Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy  = torch.randn(2, 3, 224, 224).to(device)

    for name, fn in MODEL_REGISTRY.items():
        m   = fn().to(device)
        out = m(dummy)
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"{name:20s} | output: {out.shape} | params: {params:,}")
