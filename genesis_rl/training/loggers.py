"""TensorBoard + 進捗PNG(Spakonaのsave_plot踏襲)。"""

from __future__ import annotations

from collections import deque
from pathlib import Path


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
        self._resume_prob = None
        self._stage = None
        self._history = {"transitions": [], "gates": [], "success": [], "return": []}

    def log_episode(self, transitions: int, info: dict):
        self.ep_gates.append(info["gates"])
        self.ep_success.append(1.0 if info["success"] else 0.0)
        self.ep_return.append(info.get("episode_sums", {}).get("total", 0.0))
        if "spawn_gate" in info:
            self.ep_spawn_gate.append(info["spawn_gate"])
        if "spawn_dist_g1" in info:
            self.ep_spawn_dist.append(info["spawn_dist_g1"])
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
        if self._stage is not None:
            stats["curriculum/stage"] = self._stage
        self.log_scalars(transitions, stats)
        h = self._history
        h["transitions"].append(transitions)
        h["gates"].append(stats["episode/gates_mean"])
        h["success"].append(stats["episode/success_rate"])
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
            axes[1].plot(h["transitions"], h["success"]); axes[1].set_ylabel("success rate")
            axes[2].plot(h["transitions"], h["return"]); axes[2].set_ylabel("return")
            axes[2].set_xlabel("transitions")
            fig.tight_layout()
            fig.savefig(self.dir / "progress.png", dpi=100)
            plt.close(fig)
        except Exception:
            pass
