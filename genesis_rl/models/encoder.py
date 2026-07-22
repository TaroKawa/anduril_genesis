"""凍結ResNet18エンコーダ(Spakona FrozenEncoder互換・バッチ対応)。

ImageNet事前学習・fc=Identity・512次元。常に凍結(BN統計もeval固定)なので
特徴は時不変 → replayには画像でなく512次元特徴を保存できる。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenResNet18(nn.Module):
    FEAT_DIM = 512

    def __init__(self, pretrained: bool = True, bf16: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        m.fc = nn.Identity()
        self.net = m
        self.bf16 = bf16
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True):
        # BNを常にevalに保つ(特徴の時不変性)
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N,3,224,224) [0,1] → (N,512) float32。"""
        x = (x - self.mean) / self.std
        if self.bf16 and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = self.net(x)
            return f.float()
        return self.net(x)
