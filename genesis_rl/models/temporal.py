"""時系列トランク: 観測履歴(K決定ステップ)を小型Transformerで統合する。

トークン/ステップ = Linear(feat(384) ⊕ vec動的成分(15) → d)。one-hot(通過ゲート)は
窓内で不変なのでトークンに入れず、呼び出し側がトランク出力に連結する。
エピソード開始前のスロットは全ゼロ(collectorが境界でゼロ埋め)→ ここで
key_padding_maskとして検出し、注意から除外する。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import contracts as C


class TemporalTrunk(nn.Module):
    """(B,K,FEAT_DIM) + (B,K,VEC_DIM) → (B, d_model)。最終トークンの出力を返す。"""

    def __init__(self, d_model: int = 256, n_layers: int = 2, n_heads: int = 4, ffn: int = 512,
                 feat_dim: int = C.FEAT_DIM, k: int = C.HIST_K):
        super().__init__()
        self.k = k
        self.out_dim = d_model
        self.register_buffer("dyn_idx", torch.tensor(C.VEC_DYN_IDX, dtype=torch.long),
                             persistent=False)
        self.proj = nn.Linear(feat_dim + len(C.VEC_DYN_IDX), d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.zeros(1, k, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=ffn,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, feat_hist: torch.Tensor, vec_hist: torch.Tensor) -> torch.Tensor:
        dyn = vec_hist.index_select(-1, self.dyn_idx)                  # (B,K,15)
        x = self.in_norm(self.proj(torch.cat([feat_hist, dyn], dim=-1))) + self.pos
        # パディング検出: エピソード開始前のスロットはfeatが厳密にゼロ
        pad = feat_hist.abs().sum(dim=-1) == 0                          # (B,K) True=無視
        pad = pad & ~pad.all(dim=1, keepdim=True)                       # 全パディング行の保護
        h = self.enc(x, src_key_padding_mask=pad)
        return h[:, -1]


def static_part(vec_hist: torch.Tensor) -> torch.Tensor:
    """現在ステップのone-hot(通過ゲート)を取り出す。(B,K,VEC_DIM) → (B,MAX_GATES)。"""
    return vec_hist[:, -1, C.VEC_ONEHOT]
