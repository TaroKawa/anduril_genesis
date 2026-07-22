"""GPU常駐リングReplayバッファ + 成功バッファ(RLPD混合)。

画像はなく凍結ResNetの512次元特徴(fp16)を保存する — 1Mで約2.5GB。
n-step済みの遷移 (feat, vec, priv, act, R_n, gpow=γ^k, done, nfeat, nvec, npriv) を格納。
done=真の終端(衝突/完走)のみ。タイムアウトはgpowでブートストラップ継続。
"""

from __future__ import annotations

import torch


FIELDS = ("feat", "vec", "priv", "act", "rew", "gpow", "done", "nfeat", "nvec", "npriv")


class ReplayBuffer:
    def __init__(self, capacity: int, vec_dim: int, priv_dim: int, act_dim: int,
                 feat_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.feat = torch.zeros(capacity, feat_dim, dtype=torch.float16, device=device)
        self.nfeat = torch.zeros(capacity, feat_dim, dtype=torch.float16, device=device)
        self.vec = torch.zeros(capacity, vec_dim, device=device)
        self.nvec = torch.zeros(capacity, vec_dim, device=device)
        self.priv = torch.zeros(capacity, priv_dim, device=device)
        self.npriv = torch.zeros(capacity, priv_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.gpow = torch.zeros(capacity, device=device)   # γ^k(1-done)相当の割引係数
        self.done = torch.zeros(capacity, device=device)
        self.ptr = 0
        self.size = 0

    def add_batch(self, batch: dict[str, torch.Tensor]):
        n = batch["feat"].shape[0]
        if n == 0:
            return
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        self.feat[idx] = batch["feat"].to(self.device, dtype=torch.float16)
        self.nfeat[idx] = batch["nfeat"].to(self.device, dtype=torch.float16)
        for k in ("vec", "nvec", "priv", "npriv", "act", "rew", "gpow", "done"):
            getattr(self, k)[idx] = batch[k].to(self.device)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        out = {k: getattr(self, k)[idx] for k in FIELDS}
        out["feat"] = out["feat"].float()
        out["nfeat"] = out["nfeat"].float()
        return out

    def state_size_bytes(self) -> int:
        return sum(getattr(self, k).element_size() * getattr(self, k).numel() for k in FIELDS)


class MixedSampler:
    """通常バッファ + 成功バッファの混合サンプリング(RLPD)。"""

    def __init__(self, main: ReplayBuffer, success: ReplayBuffer):
        self.main = main
        self.success = success
        self.success_ratio = 0.5

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n_succ = int(batch_size * self.success_ratio) if self.success.size > 1000 else 0
        n_main = batch_size - n_succ
        b = self.main.sample(n_main)
        if n_succ > 0:
            s = self.success.sample(n_succ)
            b = {k: torch.cat([b[k], s[k]], dim=0) for k in b}
        return b
