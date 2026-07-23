"""非対称SACの更新ステップ。

- TwinQPriv(特権) : TD学習。actor損失はmin(Q1_priv, Q2_priv)を最大化 → 学習を駆動
- TwinQObs(実観測): 同じバッチで独自のTD損失+targetを学習(Phase2への持ち出し用)。
                    Phase1ではactorを一切駆動しない
- auto-α(target entropy -4)
- DroQ: criticのdropout+LayerNorm、targetもtrainモード維持
- n-step: replayに入っているR_n/gpowを使う(gpow=γ^k·(1-terminal))
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..config import SacConfig
from ..models.actor import SACActor
from ..models.critic import make_obs_critic, make_priv_critic


@dataclass
class SacLosses:
    # 値はGPUテンソル(detach済み)のまま返す。float化は200更新に1回のログ時だけ行い、
    # 毎更新のGPU→CPU同期(連続バースト更新の直列化)を避ける。
    q_priv: "torch.Tensor"
    q_obs: "torch.Tensor"
    actor: "torch.Tensor"
    alpha: "torch.Tensor"
    alpha_value: "torch.Tensor"
    entropy: "torch.Tensor"
    q_mean: "torch.Tensor"


class SacAgent:
    def __init__(self, cfg: SacConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        h, do, ln = cfg.hidden, cfg.critic_dropout, cfg.critic_layernorm
        self.actor = SACActor(hidden=h).to(device)
        self.q_priv = make_priv_critic(h, do, ln).to(device)
        self.q_priv_target = copy.deepcopy(self.q_priv)
        self.q_obs = make_obs_critic(h, do, ln).to(device)
        self.q_obs_target = copy.deepcopy(self.q_obs)
        # DroQ: targetもtrainモード(dropout有効)に保つ
        self.q_priv_target.train()
        self.q_obs_target.train()
        for p in self.q_priv_target.parameters():
            p.requires_grad_(False)
        for p in self.q_obs_target.parameters():
            p.requires_grad_(False)

        self.log_alpha = torch.tensor(float(torch.log(torch.tensor(cfg.alpha_init))),
                                      device=device, requires_grad=True)
        self.target_entropy = cfg.target_entropy
        self.alpha_floor = 0.0  # カリキュラム進級時に一時的に>0へ

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.opt_q_priv = torch.optim.Adam(self.q_priv.parameters(), lr=cfg.lr)
        self.opt_q_obs = torch.optim.Adam(self.q_obs.parameters(), lr=cfg.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

        self._update = self._update_impl
        if cfg.compile:
            try:
                self._update = torch.compile(self._update_impl, mode="reduce-overhead")
            except Exception as e:  # pragma: no cover
                print(f"[sac] torch.compile failed, falling back to eager: {e}")

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(min=self.alpha_floor)

    def update(self, batch: dict[str, torch.Tensor]) -> SacLosses:
        return self._update(batch)

    def _update_impl(self, batch: dict[str, torch.Tensor]) -> SacLosses:
        cfg = self.cfg
        feat, vec, priv, act = batch["feat"], batch["vec"], batch["priv"], batch["act"]
        rew, gpow = batch["rew"], batch["gpow"]
        nfeat, nvec, npriv = batch["nfeat"], batch["nvec"], batch["npriv"]

        alpha = self.alpha.detach()

        # --- critic targets ---
        with torch.no_grad():
            na, nlogp, _ = self.actor.sample(nfeat, nvec)
            tq_priv = self.q_priv_target.min_q(npriv, na) - alpha * nlogp
            y_priv = rew + gpow * tq_priv
            tq_obs = self.q_obs_target.min_q(nfeat, nvec, na) - alpha * nlogp
            y_obs = rew + gpow * tq_obs

        # --- Q_priv ---
        q1, q2 = self.q_priv(priv, act)
        loss_q_priv = F.mse_loss(q1, y_priv) + F.mse_loss(q2, y_priv)
        self.opt_q_priv.zero_grad(set_to_none=True)
        loss_q_priv.backward()
        self.opt_q_priv.step()

        # --- Q_obs(補助・actorへは影響しない) ---
        o1, o2 = self.q_obs(feat, vec, act)
        loss_q_obs = F.mse_loss(o1, y_obs) + F.mse_loss(o2, y_obs)
        self.opt_q_obs.zero_grad(set_to_none=True)
        loss_q_obs.backward()
        self.opt_q_obs.step()

        # --- actor(特権criticで駆動) ---
        a, logp, _ = self.actor.sample(feat, vec)
        q_pi = self.q_priv.min_q(priv, a)
        loss_actor = (self.alpha.detach() * logp - q_pi).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        self.opt_actor.step()

        # --- alpha ---
        loss_alpha = (-self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        # --- soft update ---
        with torch.no_grad():
            tau = cfg.tau
            for t, s in zip(self.q_priv_target.parameters(), self.q_priv.parameters()):
                t.lerp_(s, tau)
            for t, s in zip(self.q_obs_target.parameters(), self.q_obs.parameters()):
                t.lerp_(s, tau)

        return SacLosses(
            q_priv=loss_q_priv.detach(),
            q_obs=loss_q_obs.detach(),
            actor=loss_actor.detach(),
            alpha=loss_alpha.detach(),
            alpha_value=self.alpha.detach(),
            entropy=-logp.detach().mean(),
            q_mean=q_pi.detach().mean(),
        )

    # --- checkpoint ---

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "q_priv": self.q_priv.state_dict(),
            "q_priv_target": self.q_priv_target.state_dict(),
            "q_obs": self.q_obs.state_dict(),
            "q_obs_target": self.q_obs_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_q_priv": self.opt_q_priv.state_dict(),
            "opt_q_obs": self.opt_q_obs.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.q_priv.load_state_dict(sd["q_priv"])
        self.q_priv_target.load_state_dict(sd["q_priv_target"])
        self.q_obs.load_state_dict(sd["q_obs"])
        self.q_obs_target.load_state_dict(sd["q_obs_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.opt_actor.load_state_dict(sd["opt_actor"])
        self.opt_q_priv.load_state_dict(sd["opt_q_priv"])
        self.opt_q_obs.load_state_dict(sd["opt_q_obs"])
        self.opt_alpha.load_state_dict(sd["opt_alpha"])
