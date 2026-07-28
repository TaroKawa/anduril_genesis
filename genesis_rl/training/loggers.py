"""TensorBoard + 進捗PNG(Spakonaのsave_plot踏襲)。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


class TrainLogger:
    def __init__(self, ckpt_dir: str | Path):
        from torch.utils.tensorboard import SummaryWriter

        self.dir = Path(ckpt_dir)
        self.tb = SummaryWriter(log_dir=str(self.dir / "tb"))
        self.ep_gates = deque(maxlen=2000)
        self.ep_success = deque(maxlen=2000)
        self.ep_return = deque(maxlen=2000)
        self.ep_spawn_gate = deque(maxlen=2000)
        self.ep_spawn_dist = deque(maxlen=2000)
        # 正規スタート(spawn_gate<=1)だけの成功率/通過ゲート数。逆カリキュラム中の
        # episode/success_rate は途中スポーン(残りゲートが少なくfinishしやすい)を含むので
        # 実力より甘く出る。進級を決めるのは curriculum 側の正規スタート限定の値なので
        # (curriculum.record_episodes)、同じ母集団の系列をここでも出して可視化する。
        self.ep_fs_success = deque(maxlen=2000)
        self.ep_fs_gates = deque(maxlen=2000)
        # 失敗モードの内訳(なぜ/どこでエピソードが終わるか)。done=collision|finish|timeout。
        self.ep_collision = deque(maxlen=2000)
        self.ep_finish = deque(maxlen=2000)
        self._resume_prob = None
        self._stage = None
        self._history = {"transitions": [], "gates": [], "success": [], "success_fs": [],
                         "return": []}

    def log_episode(self, transitions: int, info: dict):
        self.ep_gates.append(info["gates"])
        self.ep_success.append(1.0 if info["success"] else 0.0)
        self.ep_return.append(info.get("episode_sums", {}).get("total", 0.0))
        if "spawn_gate" in info:
            self.ep_spawn_gate.append(info["spawn_gate"])
            if int(info["spawn_gate"]) <= 1:
                self.ep_fs_success.append(1.0 if info["success"] else 0.0)
                self.ep_fs_gates.append(info["gates"])
        if "spawn_dist_g1" in info:
            self.ep_spawn_dist.append(info["spawn_dist_g1"])
        if "collision" in info:
            self.ep_collision.append(1.0 if info["collision"] else 0.0)
        if "finish" in info:
            self.ep_finish.append(1.0 if info["finish"] else 0.0)
        self._resume_prob = info.get("resume_prob", self._resume_prob)
        self._stage = info.get("stage", self._stage)

    def log_scalars(self, step: int, scalars: dict, prefix: str = ""):
        for k, v in scalars.items():
            self.tb.add_scalar(f"{prefix}{k}", v, step)

    def flush_episode_stats(self, transitions: int):
        if not self.ep_gates:
            return {}
        stats = {
            "episode/gates_mean": sum(self.ep_gates) / len(self.ep_gates),
            "episode/gates_max": max(self.ep_gates),
            "episode/success_rate": sum(self.ep_success) / len(self.ep_success),
            "episode/return_mean": sum(self.ep_return) / len(self.ep_return),
        }
        # 逆カリキュラムの現在地(どこからスポーンしているか・正規スタートへの移行度)
        if self.ep_spawn_gate:
            stats["curriculum/spawn_gate_mean"] = sum(self.ep_spawn_gate) / len(self.ep_spawn_gate)
        if self.ep_spawn_dist:
            stats["curriculum/spawn_dist_gate1_mean"] = sum(self.ep_spawn_dist) / len(self.ep_spawn_dist)
        if self._resume_prob is not None:
            stats["curriculum/resume_prob"] = self._resume_prob
        # 正規スタート限定(=進級判定と同じ母集団)。episode/success_rate と併記することで
        # 「途中スポーン込みの見かけ」と「フルコースの実力」を分けて追える。
        if self.ep_fs_success:
            stats["episode/success_rate_full_start"] = sum(self.ep_fs_success) / len(self.ep_fs_success)
            stats["episode/gates_mean_full_start"] = sum(self.ep_fs_gates) / len(self.ep_fs_gates)
            stats["curriculum/full_start_frac"] = len(self.ep_fs_success) / len(self.ep_success)
        if self._stage is not None:
            stats["curriculum/stage"] = self._stage
        # 失敗モードの内訳: なぜエピソードが終わったか(衝突/完走/タイムアウト)。
        # done = collision | finish | timeout なので timeout = ~collision & ~finish。
        if self.ep_collision and self.ep_finish:
            col = sum(self.ep_collision) / len(self.ep_collision)
            fin = sum(self.ep_finish) / len(self.ep_finish)
            stats["episode/collision_rate"] = col
            stats["episode/finish_rate"] = fin
            stats["episode/timeout_rate"] = max(0.0, 1.0 - col - fin)
        # どこで落ちたか: 通過ゲート数の分布(p10/p50=中央値)。gates_meanだけだと分からない。
        if self.ep_gates:
            sg = sorted(self.ep_gates)
            stats["episode/gates_p50"] = sg[len(sg) // 2]
            stats["episode/gates_p10"] = sg[max(0, len(sg) // 10)]
            try:
                self.tb.add_histogram("episode/done_gates",
                                      np.array(self.ep_gates, dtype=float), transitions)
            except Exception:
                pass
        self.log_scalars(transitions, stats)
        h = self._history
        h["transitions"].append(transitions)
        h["gates"].append(stats["episode/gates_mean"])
        h["success"].append(stats["episode/success_rate"])
        h["success_fs"].append(stats.get("episode/success_rate_full_start", float("nan")))
        h["return"].append(stats["episode/return_mean"])
        return stats

    def save_plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            h = self._history
            if len(h["transitions"]) < 2:
                return
            fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
            axes[0].plot(h["transitions"], h["gates"]); axes[0].set_ylabel("gates/ep")
            # ラベルはASCII固定(コンテナにCJKフォントが無く豆腐になるため)
            axes[1].plot(h["transitions"], h["success"], label="all (incl. resume spawn)")
            axes[1].plot(h["transitions"], h["success_fs"], label="full start (advance metric)")
            axes[1].set_ylabel("success rate"); axes[1].legend(fontsize=8)
            axes[2].plot(h["transitions"], h["return"]); axes[2].set_ylabel("return")
            axes[2].set_xlabel("transitions")
            fig.tight_layout()
            fig.savefig(self.dir / "progress.png", dpi=100)
            plt.close(fig)
        except Exception:
            pass
