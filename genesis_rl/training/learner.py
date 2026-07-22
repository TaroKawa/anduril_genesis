"""ラーナー: GPU常駐replay + SAC更新 + ckpt + TensorBoard。

replay-ratioガバナー: 累積 R = batch×updates / 収集遷移数 が cap を超えたら更新を待つ
(並列収集ではUTD≪1が正常。新鮮なデータより速く回して過学習するのを防ぐ)。
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .. import contracts as C
from ..config import TrainConfig
from ..models.encoder import FrozenResNet18
from .checkpoint import save_checkpoint
from .loggers import TrainLogger
from .replay import MixedSampler, ReplayBuffer
from .sac import SacAgent


class Learner:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.agent = SacAgent(cfg.sac, device)
        self.replay = ReplayBuffer(cfg.sac.replay_capacity, C.VEC_DIM, C.PRIV_DIM, C.ACTION_DIM,
                                   FrozenResNet18.FEAT_DIM, device)
        self.success = ReplayBuffer(cfg.sac.success_capacity, C.VEC_DIM, C.PRIV_DIM, C.ACTION_DIM,
                                    FrozenResNet18.FEAT_DIM, device)
        self.sampler = MixedSampler(self.replay, self.success)
        self.sampler.success_ratio = cfg.sac.success_ratio
        self.logger = TrainLogger(cfg.run.ckpt_dir)
        self.updates = 0
        self.transitions = 0        # 収集された遷移数(コレクター報告)
        self.stage = 0
        self.best_gates = -1.0
        self.best_return = -1e18
        self._last_ckpt = time.time()

    def add_transitions(self, batch: dict, success: bool = False):
        (self.success if success else self.replay).add_batch(batch)
        if not success:
            self.transitions += batch["feat"].shape[0]

    def can_update(self) -> bool:
        if self.replay.size < self.cfg.sac.learn_start:
            return False
        ratio = (self.updates * self.cfg.sac.batch_size) / max(self.transitions, 1)
        return ratio < self.cfg.sac.replay_ratio_cap

    def update_once(self) -> dict | None:
        if not self.can_update():
            return None
        batch = self.sampler.sample(self.cfg.sac.batch_size)
        losses = self.agent.update(batch)
        self.updates += 1
        if self.updates % 200 == 0:
            self.logger.log_scalars(self.transitions, {
                "loss/q_priv": losses.q_priv, "loss/q_obs": losses.q_obs,
                "loss/actor": losses.actor, "loss/alpha": losses.alpha,
                "sac/alpha": losses.alpha_value, "sac/entropy": losses.entropy,
                "sac/q_mean": losses.q_mean, "sac/updates": self.updates,
                "sac/replay_size": self.replay.size, "sac/success_size": self.success.size,
                "sac/replay_ratio": (self.updates * self.cfg.sac.batch_size) / max(self.transitions, 1),
            })
        return losses.__dict__

    def actor_weights_cpu(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.agent.actor.state_dict().items()}

    def maybe_checkpoint(self, ep_stats: dict | None = None, force: bool = False):
        cfg = self.cfg
        now = time.time()
        ckpt_dir = Path(cfg.run.ckpt_dir)
        if force or now - self._last_ckpt > cfg.run.ckpt_interval_s:
            save_checkpoint(ckpt_dir / "latest.pt", self.agent, learner_step=self.updates,
                            env_transitions=self.transitions, stage=self.stage)
            self.logger.save_plot()
            self._last_ckpt = now
        if ep_stats:
            g = ep_stats.get("episode/gates_mean", -1)
            r = ep_stats.get("episode/return_mean", -1e18)
            if g > self.best_gates:
                self.best_gates = g
                save_checkpoint(ckpt_dir / "best_gates.pt", self.agent, learner_step=self.updates,
                                env_transitions=self.transitions, stage=self.stage)
            if r > self.best_return:
                self.best_return = r
                save_checkpoint(ckpt_dir / "best_return.pt", self.agent, learner_step=self.updates,
                                env_transitions=self.transitions, stage=self.stage)

    def update_success_ratio(self, stage: int):
        # Stage2以降(ゲート3+を安定通過)は成功バッファ依存を下げる
        self.stage = stage
        self.sampler.success_ratio = 0.25 if stage >= 2 else self.cfg.sac.success_ratio
