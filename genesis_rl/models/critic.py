"""二重の Twin-Q:

TwinQPriv — 特権観測(priv 39)で学習しactorを駆動する(非対称SACの核)。
            真値はマルコフ的なので履歴不要 → MLPのまま。
TwinQObs  — 実観測(履歴feat+vec)の補助critic。actorと同構造の時系列トランクを
            専有し、Phase 2(本番シム・特権テレメトリ遮断)へ持ち込む。

DroQ構成: Dropout(0.01) + LayerNorm、targetもtrainモード維持(dropout有効)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..contracts import ACTION_DIM, MAX_GATES, PRIV_DIM
from .temporal import TemporalTrunk, static_part


def _q_mlp(in_dim: int, hidden: int, dropout: float, layernorm: bool) -> nn.Sequential:
    def block(i, o):
        layers = [nn.Linear(i, o)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        if layernorm:
            layers.append(nn.LayerNorm(o))
        layers.append(nn.ReLU())
        return layers

    return nn.Sequential(*block(in_dim, hidden), *block(hidden, hidden), nn.Linear(hidden, 1))


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 512, dropout: float = 0.01, layernorm: bool = True):
        super().__init__()
        in_dim = obs_dim + ACTION_DIM
        self.q1 = _q_mlp(in_dim, hidden, dropout, layernorm)
        self.q2 = _q_mlp(in_dim, hidden, dropout, layernorm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat([obs, action], dim=1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(obs, action)
        return torch.minimum(q1, q2)


class TemporalTwinQ(nn.Module):
    """実観測(履歴)用Twin-Q: 時系列トランク + DroQヘッド×2。"""

    def __init__(self, hidden: int = 512, dropout: float = 0.01, layernorm: bool = True):
        super().__init__()
        self.trunk = TemporalTrunk()
        in_dim = self.trunk.out_dim + MAX_GATES + ACTION_DIM
        self.q1 = _q_mlp(in_dim, hidden, dropout, layernorm)
        self.q2 = _q_mlp(in_dim, hidden, dropout, layernorm)

    def forward(self, feat_hist: torch.Tensor, vec_hist: torch.Tensor, action: torch.Tensor):
        h = self.trunk(feat_hist, vec_hist)
        x = torch.cat([h, static_part(vec_hist), action], dim=1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def min_q(self, feat_hist: torch.Tensor, vec_hist: torch.Tensor,
              action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(feat_hist, vec_hist, action)
        return torch.minimum(q1, q2)


def make_priv_critic(hidden: int = 512, dropout: float = 0.01, layernorm: bool = True) -> TwinQ:
    return TwinQ(PRIV_DIM, hidden, dropout, layernorm)


def make_obs_critic(hidden: int = 512, dropout: float = 0.01, layernorm: bool = True) -> TemporalTwinQ:
    return TemporalTwinQ(hidden, dropout, layernorm)
