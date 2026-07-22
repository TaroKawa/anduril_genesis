"""ベクトル化 n-step アセンブラ(Spakona NStepAssemblerのテンソル化)。

各決定ステップ (s_t, a_t, r_t, s_{t+1}, done, terminal) を受け取り、
n-step遷移 (s_t, a_t, R=Σγ^i r, s_{t+k}, gpow=γ^k·(1-terminal)) を吐き出す。
  - 通常: k=n(熟成)
  - エピソード終端: 全pending行をk<nでflush(terminalのみgpow=0にする)
タイムアウト(done但しterminalでない)はgpow>0でブートストラップが続く。
"""

from __future__ import annotations

import torch


class NStepAssembler:
    def __init__(self, num_envs: int, n_step: int, gamma: float,
                 vec_shape: tuple | int, priv_dim: int, act_dim: int,
                 feat_shape: tuple | int, device: torch.device):
        vec_shape = (vec_shape,) if isinstance(vec_shape, int) else tuple(vec_shape)
        feat_shape = (feat_shape,) if isinstance(feat_shape, int) else tuple(feat_shape)
        self.N = num_envs
        self.n = n_step
        self.gamma = gamma
        self.device = device
        n = n_step
        self.feat = torch.zeros(n, num_envs, *feat_shape, device=device)
        self.vec = torch.zeros(n, num_envs, *vec_shape, device=device)
        self.priv = torch.zeros(n, num_envs, priv_dim, device=device)
        self.act = torch.zeros(n, num_envs, act_dim, device=device)
        self.rew_acc = torch.zeros(n, num_envs, device=device)
        self.age = torch.zeros(num_envs, dtype=torch.long, device=device)  # pending行数(0..n)
        self.head = 0

    def push(self, feat, vec, priv, act, rew, done, terminal, nfeat, nvec, npriv) -> dict | None:
        n, dev = self.n, self.device
        h = self.head
        # 1) 新しい行を書き込み
        self.feat[h], self.vec[h], self.priv[h], self.act[h] = feat, vec, priv, act
        self.rew_acc[h] = 0.0
        age_new = (self.age + 1).clamp(max=n)

        # 2) pending全行に割引報酬を加算(行の経過k=0..n-1、γ^k·r)
        for k in range(n):
            slot = (h - k) % n
            mask = age_new > k
            if mask.any():
                self.rew_acc[slot, mask] += (self.gamma ** k) * rew[mask]

        out = {key: [] for key in ("feat", "vec", "priv", "act", "rew", "gpow", "done",
                                   "nfeat", "nvec", "npriv")}

        def flush(slot_idx: torch.Tensor, envs: torch.Tensor, gpow: torch.Tensor, done_flag: torch.Tensor):
            out["feat"].append(self.feat[slot_idx, envs])
            out["vec"].append(self.vec[slot_idx, envs])
            out["priv"].append(self.priv[slot_idx, envs])
            out["act"].append(self.act[slot_idx, envs])
            out["rew"].append(self.rew_acc[slot_idx, envs])
            out["gpow"].append(gpow)
            out["done"].append(done_flag)
            out["nfeat"].append(nfeat[envs])
            out["nvec"].append(nvec[envs])
            out["npriv"].append(npriv[envs])

        # 3) エピソード終端: 全pending行をflush
        if done.any():
            d_envs = done.nonzero(as_tuple=False).squeeze(1)
            for k in range(n):
                envs_k = d_envs[age_new[d_envs] > k]
                if len(envs_k) == 0:
                    continue
                slot = torch.full_like(envs_k, (h - k) % n)
                a = k + 1  # この行の経過ステップ数
                term = terminal[envs_k].float()
                gpow = (self.gamma ** a) * (1.0 - term)
                flush(slot, envs_k, gpow, term)

        # 4) 熟成flush(done以外でage==n)
        mature = (age_new == n) & ~done
        if mature.any():
            m_envs = mature.nonzero(as_tuple=False).squeeze(1)
            slot = torch.full_like(m_envs, (h - (n - 1)) % n)
            gpow = torch.full((len(m_envs),), self.gamma ** n, device=dev)
            flush(slot, m_envs, gpow, torch.zeros(len(m_envs), device=dev))

        # 5) age更新・head前進
        self.age = torch.where(done, torch.zeros_like(age_new),
                               torch.where(mature, age_new - 1, age_new))
        self.head = (h + 1) % n

        if not out["feat"]:
            return None
        return {k: torch.cat(v, dim=0) for k, v in out.items()}
