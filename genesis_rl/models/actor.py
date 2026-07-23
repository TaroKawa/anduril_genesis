"""SAC Actor: 時系列トランク + tanh-Gaussian。

入力 = 観測履歴 feat_hist (B,K,384) + vec_hist (B,K,55)。出力 a∈[-1,1]^4。
トランク(TemporalTrunk)が履歴を統合し、通過ゲートone-hotはヘッドで合流。
mean headはゼロ初期化(t=0でアクション0 = 推力0.2694=ホバー = 望ましいburn-in挙動)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..contracts import ACTION_DIM, MAX_GATES
from .temporal import TemporalTrunk, static_part

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class SACActor(nn.Module):
    def __init__(self, hidden: int = 512):
        super().__init__()
        self.trunk = TemporalTrunk()
        self.body = nn.Sequential(
            nn.Linear(self.trunk.out_dim + MAX_GATES, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, ACTION_DIM)
        self.log_std = nn.Linear(hidden, ACTION_DIM)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)

    def forward(self, feat_hist: torch.Tensor, vec_hist: torch.Tensor):
        h = self.trunk(feat_hist, vec_hist)
        h = self.body(torch.cat([h, static_part(vec_hist)], dim=1))
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, feat_hist: torch.Tensor, vec_hist: torch.Tensor):
        """returns (action, log_prob, mean_action)"""
        mean, log_std = self(feat_hist, vec_hist)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        pre_tanh = mean + std * noise
        a = torch.tanh(pre_tanh)
        # tanh-squashed Gaussianのlog_prob(ヤコビアン補正)
        log_prob = (-0.5 * (noise.pow(2) + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))).sum(dim=1)
        log_prob = log_prob - torch.log((1 - a.pow(2)).clamp(min=1e-6)).sum(dim=1)
        return a, log_prob, torch.tanh(mean)

    @torch.no_grad()
    def act(self, feat_hist: torch.Tensor, vec_hist: torch.Tensor,
            deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(feat_hist, vec_hist)
        if deterministic:
            return torch.tanh(mean)
        return torch.tanh(mean + log_std.exp() * torch.randn_like(mean))
