"""二重の Twin-Q:

TwinQPriv — 特権観測(priv 39)で学習しactorを駆動する(非対称SACの核)。
TwinQObs  — 実観測(feat512+vec55)の補助critic。Phase 1で同じバッチにより常時学習し、
            Phase 2(本番シム・特権テレメトリ遮断)へ自己整合なcriticとして持ち込む。

DroQ構成: Dropout(0.01) + LayerNorm、targetもtrainモード維持(dropout有効)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..contracts import ACTION_DIM, PRIV_DIM, VEC_DIM
from .encoder import FrozenResNet18


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


def make_priv_critic(hidden: int = 512, dropout: float = 0.01, layernorm: bool = True) -> TwinQ:
    return TwinQ(PRIV_DIM, hidden, dropout, layernorm)


def make_obs_critic(hidden: int = 512, dropout: float = 0.01, layernorm: bool = True) -> TwinQ:
    return TwinQ(FrozenResNet18.FEAT_DIM + VEC_DIM, hidden, dropout, layernorm)
