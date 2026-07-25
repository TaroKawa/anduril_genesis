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
from .checkpoint import save_checkpoint
from .loggers import TrainLogger
from .replay import MixedSampler, ReplayBuffer
from .sac import SacAgent


class Learner:
    def __init__(self, cfg: TrainConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.agent = SacAgent(cfg.sac, device)
        vec_shape = (C.HIST_K, C.VEC_DIM)
        feat_shape = (C.HIST_K, C.FEAT_DIM)
        self.replay = ReplayBuffer(cfg.sac.replay_capacity, vec_shape, C.PRIV_DIM, C.ACTION_DIM,
                                   feat_shape, device)
        self.success = ReplayBuffer(cfg.sac.success_capacity, vec_shape, C.PRIV_DIM, C.ACTION_DIM,
                                    feat_shape, device)
        self.sampler = MixedSampler(self.replay, self.success)
        self.sampler.success_ratio = cfg.sac.success_ratio
        self.logger = TrainLogger(cfg.run.ckpt_dir)
        self.updates = 0
        self.transitions = 0        # 収集された遷移数(コレクター報告)
        self.stage = 0
        self.best_gates = -1.0
        self.best_return = -1e18
        self._best_return_loaded = False  # best_return.pt の記録値をstage5で一度だけ読み戻す
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

    def update_once(self) -> bool | None:
        if not self.can_update():
            return None
        batch = self.sampler.sample(self.cfg.sac.batch_size)
        losses = self.agent.update(batch)
        self.updates += 1
        if self.updates % 200 == 0:
            # ここで初めてGPU→CPU同期(200更新に1回)。損失はテンソルのまま来るのでfloat化する
            self.logger.log_scalars(self.transitions, {
                "loss/q_priv": float(losses.q_priv), "loss/q_obs": float(losses.q_obs),
                "loss/actor": float(losses.actor), "loss/alpha": float(losses.alpha),
                "sac/alpha": float(losses.alpha_value), "sac/entropy": float(losses.entropy),
                "sac/q_mean": float(losses.q_mean), "sac/updates": self.updates,
                "sac/replay_size": self.replay.size, "sac/success_size": self.success.size,
                "sac/replay_ratio": (self.updates * self.cfg.sac.batch_size) / max(self.transitions, 1),
            })
        return True

    def actor_weights_cpu(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.agent.actor.state_dict().items()}

    def maybe_checkpoint(self, ep_stats: dict | None = None, force: bool = False):
        import dataclasses

        cfg = self.cfg
        now = time.time()
        ckpt_dir = Path(cfg.run.ckpt_dir)
        # 学習時設定を残す(dcl/client.pyがdrone.cmd_gain模擬の有無をここから自動判別する)
        snap = dataclasses.asdict(cfg)
        if force or now - self._last_ckpt > cfg.run.ckpt_interval_s:
            save_checkpoint(ckpt_dir / "latest.pt", self.agent, learner_step=self.updates,
                            env_transitions=self.transitions, stage=self.stage, cfg_snapshot=snap)
            self.logger.save_plot()
            self._last_ckpt = now
        if ep_stats:
            g = ep_stats.get("episode/gates_mean", -1)
            r = ep_stats.get("episode/return_mean", -1e18)
            if g > self.best_gates:
                self.best_gates = g
                save_checkpoint(ckpt_dir / "best_gates.pt", self.agent, learner_step=self.updates,
                                env_transitions=self.transitions, stage=self.stage, cfg_snapshot=snap)
            # best_return は最終堅牢化ステージ(stage5)でのみ更新・保存する。return_mean は
            # コース/報酬がステージごとに違うため比較不能で、旧stageの高リターンが残ると
            # stage5では二度と更新されず陳腐化する。stage5内のベストだけを残す。
            # コンテナ再起動(カリキュラム再構築/ストール)をまたいでも、既存の
            # best_return.pt が stage5 で記録した値を読み戻して基準にし、再起動直後の
            # 低リターンで上書きしないようにする。
            if self.stage >= 5:
                if not self._best_return_loaded:
                    self._load_best_return(ckpt_dir / "best_return.pt")
                if r > self.best_return:
                    self.best_return = r
                    save_checkpoint(ckpt_dir / "best_return.pt", self.agent, learner_step=self.updates,
                                    env_transitions=self.transitions, stage=self.stage,
                                    cfg_snapshot=snap, extra={"best_return": r})

    def _load_best_return(self, path: Path):
        """stage5で記録済みの best_return.pt があればその基準値を読み戻す。
        旧stage由来(best_return未記録 or stage<5)の値は比較不能なので採用しない。"""
        self._best_return_loaded = True
        if not path.exists():
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return
        if int(payload.get("stage", -1)) >= 5 and "best_return" in payload:
            self.best_return = float(payload["best_return"])

    def update_success_ratio(self, stage: int):
        # Stage3以降(フルコースを安定通過)は成功バッファ依存を下げる
        self.stage = stage
        self.sampler.success_ratio = 0.25 if stage >= 3 else self.cfg.sac.success_ratio
