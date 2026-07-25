"""報酬計算(Genesis非依存・torch)。重みはSpakona rl_config.yamlの実績値ベース。"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch


@dataclass
class RewardWeights:
    gate: float = 50.0        # ゲート通過ボーナス
    finish: float = 100.0     # 完走(最終ゲート)ボーナス
    collision: float = -20.0  # 衝突ペナルティ(終端)
    approach: float = 1.0     # 接近報酬 [m^-1](d_prev - d_now)
    closeness: float = 0.05   # 視覚closeness(ゲートが見えている間の密報酬)
    path: float = 0.08        # 青パス追従(特権): 進路先読み点を視野中心に保つ密報酬 ∈[0,1]。
                              # ゲートが柱裏/軸外で見えない旋回中でも航法手掛かりを与える
    smooth: float = -0.02     # アクション平滑化 ‖Δa‖²
    rate: float = -0.01       # レートペナルティ (‖ω‖/6)²(rate_max=6 に整合)
    wrong_way: float = -5.0   # 逆走(非終端)
    speed_finish: float = 0.0 # Stage4: 完走時間ボーナス w*(60-T)/60(カリキュラムが設定)
    approach_clip: float = 3.0  # 1決定あたりの接近クリップ [m]


class RewardComputer:
    def __init__(self, num_envs: int, device: torch.device, weights: RewardWeights | None = None):
        self.w = weights or RewardWeights()
        self.device = device
        self.num_envs = num_envs
        self.episode_sums = {k: torch.zeros(num_envs, device=device) for k in asdict(self.w)}
        self.episode_sums["total"] = torch.zeros(num_envs, device=device)

    def reset_idx(self, envs_idx: torch.Tensor):
        for v in self.episode_sums.values():
            v[envs_idx] = 0.0

    def compute(
        self,
        *,
        gate_pass: torch.Tensor,      # (N,) bool このステップで正規ゲートを通過
        finish: torch.Tensor,         # (N,) bool 最終ゲートを通過
        collision: torch.Tensor,      # (N,) bool
        d_prev: torch.Tensor,         # (N,) 前決定時のアクティブゲートまでの距離 [m]
        d_now: torch.Tensor,          # (N,)
        closeness: torch.Tensor,      # (N,) (1 - rel_dist_true) * visible ∈ [0,1]
        path_view: torch.Tensor,      # (N,) 進路先読み点(青パス)の視野内中心度 ∈ [0,1]
        action: torch.Tensor,         # (N,4) [-1,1]
        last_action: torch.Tensor,    # (N,4)
        omega_norm: torch.Tensor,     # (N,) ‖ω‖ [rad/s]
        wrong_way: torch.Tensor,      # (N,) bool
        episode_t: torch.Tensor,      # (N,) エピソード経過 [s]
        max_episode_s: float = 60.0,
    ) -> torch.Tensor:
        w = self.w
        terms = {
            "gate": w.gate * gate_pass.float(),
            "finish": w.finish * finish.float(),
            "collision": w.collision * collision.float(),
            # ゲート通過直後はアクティブゲートが切り替わり距離が跳ぶのでスキップ
            "approach": w.approach
            * torch.where(gate_pass, torch.zeros_like(d_now), (d_prev - d_now).clamp(-w.approach_clip, w.approach_clip)),
            "closeness": w.closeness * closeness,
            # 青パス追従: 進路先読み点が視野中心にあるほど密に加点(通過後はゲート同様スキップしない
            # ＝旋回中も継続的に手掛かりを与える)。finish後は0にして完走の速度志向を邪魔しない。
            "path": w.path * path_view * (~finish).float(),
            "smooth": w.smooth * (action - last_action).pow(2).sum(dim=1),
            "rate": w.rate * (omega_norm / 6.0).pow(2),
            "wrong_way": w.wrong_way * wrong_way.float(),
            "speed_finish": w.speed_finish
            * finish.float()
            * ((max_episode_s - episode_t).clamp(min=0.0) / max_episode_s),
        }
        total = torch.zeros_like(d_now)
        for k, v in terms.items():
            total += v
            self.episode_sums[k] += v
        self.episode_sums["total"] += total
        return total
