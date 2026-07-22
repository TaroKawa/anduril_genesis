"""GPU常駐の遅延キュー。センサー遅延(検出17.3ms・画像1フレーム・アクション遅延)の実装。

プロデューサのtickごとに push し、read() は per-env の遅延ステップ分だけ過去の
要素を返す。ウォームアップ前(エピソード先頭)は最古の有効値を返す。
"""

from __future__ import annotations

import torch


class DelayQueue:
    def __init__(self, num_envs: int, shape: tuple[int, ...], max_delay: int,
                 device: torch.device, dtype=torch.float32):
        self.num_envs = num_envs
        self.capacity = max_delay + 1
        self.buf = torch.zeros((self.capacity, num_envs, *shape), device=device, dtype=dtype)
        self.delay = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.head = 0
        # per-env: エピソード内で何回pushされたか(ウォームアップ処理用)
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)

    def set_delay(self, delay_steps: torch.Tensor, envs_idx: torch.Tensor | None = None):
        """per-env遅延を設定(エピソードごとのDRジッタ)。"""
        if envs_idx is None:
            self.delay.copy_(delay_steps.clamp(0, self.capacity - 1))
        else:
            self.delay[envs_idx] = delay_steps.clamp(0, self.capacity - 1)

    def reset_idx(self, envs_idx: torch.Tensor, fill: torch.Tensor | None = None):
        self.count[envs_idx] = 0
        if fill is not None:
            self.buf[:, envs_idx] = fill
        else:
            self.buf[:, envs_idx] = 0

    def push(self, x: torch.Tensor):
        self.head = (self.head + 1) % self.capacity
        self.buf[self.head] = x
        self.count += 1

    def read(self) -> torch.Tensor:
        """(num_envs, *shape) — per-env遅延分だけ過去の値。"""
        eff = torch.minimum(self.delay, (self.count - 1).clamp(min=0))
        idx = (self.head - eff) % self.capacity
        env_ar = torch.arange(self.num_envs, device=self.buf.device)
        return self.buf[idx, env_ar]

    def age(self) -> torch.Tensor:
        """(num_envs,) 読み出し値の実効エイジ(pushステップ単位)。"""
        return torch.minimum(self.delay, (self.count - 1).clamp(min=0))
