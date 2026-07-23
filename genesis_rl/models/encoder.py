"""凍結視覚エンコーダ。

FrozenDINOv2  — 自己教師あり事前学習ViT-S/14(384次元)。現行の既定エンコーダ。
FrozenResNet18 — ImageNet事前学習(512次元)。旧構成・比較用に残す。

いずれも常に凍結なので特徴は時不変 → replayには画像でなく特徴を保存できる。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenDINOv2(nn.Module):
    """DINOv2 ViT-S/14(凍結)。(N,3,224,224) [0,1] → (N,384)。

    torch.hub経由でロード(初回はGitHub+重みをDL、torch-cacheボリュームに永続化)。
    224は14の倍数なのでリサイズ不要。CLSトークン埋め込みを返す。
    """

    FEAT_DIM = 384

    def __init__(self, bf16: bool = True):
        super().__init__()
        self.net = self._load_hub()
        self.bf16 = bf16
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.net.eval()

    @staticmethod
    def _load_hub():
        """torch.hubのGitHub API検証(_validate_not_a_forked_repo)はレート制限(403)で
        落ちることがあるため常にskip_validation。キャッシュ済みならsource="local"で
        ネットワークを一切使わずロードする(torch-cacheボリュームに永続化される)。

        複数collectorが同時に初回ロードするとzip展開が競合する(展開途中に他プロセスの
        renameで展開先が消えFileNotFoundError)ため、flockで直列化する。"""
        import fcntl
        import os

        repo, model = "facebookresearch/dinov2", "dinov2_vits14"
        hub_dir = torch.hub.get_dir()
        local = os.path.join(hub_dir, "facebookresearch_dinov2_main")
        os.makedirs(hub_dir, exist_ok=True)
        with open(os.path.join(hub_dir, ".dinov2.lock"), "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)   # withを抜ける(close)と自動解放
            if os.path.isdir(local):
                try:
                    return torch.hub.load(local, model, source="local")
                except Exception as e:
                    print(f"[encoder] local hub load failed ({e}); falling back to github",
                          flush=True)
            return torch.hub.load(repo, model, skip_validation=True, trust_repo=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.bf16 and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = self.net(x)
            return f.float()
        return self.net(x)


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
