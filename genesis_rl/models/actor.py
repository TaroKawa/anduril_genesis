"""SAC Actor: tanh-Gaussian(Spakona SACActor互換、hidden 512)。

入力 = concat[凍結ResNet特徴(512), vec(55)]。出力 a∈[-1,1]^4。
mean headはゼロ初期化(t=0でアクション0 = 推力0.3325の緩上昇 = 望ましいburn-in挙動)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..contracts import ACTION_DIM, VEC_DIM
from .encoder import FrozenResNet18

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class SACActor(nn.Module):
    def __init__(self, hidden: int = 512, feat_dim: int = FrozenResNet18.FEAT_DIM, vec_dim: int = VEC_DIM):
        super().__init__()
        in_dim = feat_dim + vec_dim
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, ACTION_DIM)
        self.log_std = nn.Linear(hidden, ACTION_DIM)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, feat: torch.Tensor, vec: torch.Tensor):
        h = self.body(torch.cat([feat, vec], dim=1))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, feat: torch.Tensor, vec: torch.Tensor):
        """returns (action, log_prob, mean_action)"""
        mean, log_std = self(feat, vec)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        a = torch.tanh(pre_tanh)
        # tanh-squashed Gaussianのlog_prob(ヤコビアン補正)
        log_prob = (-0.5 * (noise.pow(2) + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))).sum(dim=1)
        log_prob = log_prob - torch.log((1 - a.pow(2)).clamp(min=1e-6)).sum(dim=1)
        return a, log_prob, torch.tanh(mean)

    @torch.no_grad()
    def act(self, feat: torch.Tensor, vec: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(feat, vec)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
