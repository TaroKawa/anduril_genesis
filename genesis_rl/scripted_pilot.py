"""特権スクリプトパイロット: リボン純追跡(pure pursuit)で署名付きアクションを生成。

previewの飛行デモ・符号再現のend-to-end検証・(必要なら)デモ収集に使う。
本物のポリシーと同じ [-1,1]^4 アクション(ActionMap経由)を出すので、
これが正しく飛ぶ = 左手系レート指令の変換(ProductionSigns)が正しい。

制御則(本番AttitudeHoldを模したP姿勢制御):
  1. リボン上の最近点からlookahead先の目標点へ向かう所望速度を計算
  2. 所望加速度 → 所望姿勢(tilt) + 推力(比力sysidの逆算)
  3. 姿勢誤差×kp → ボディレート指令(rate clamp)、ヨーは進行方向へ向ける
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import contracts as C
from .course import CourseSpec
from .frames import quat_rotate, quat_rotate_inv

G = 9.81


class ScriptedPilot:
    def __init__(self, course: CourseSpec, device: torch.device,
                 lookahead: float = 3.0, v_des: float = 3.0, kp_att: float = 5.0,
                 drone_cfg=None):
        self.ribbon = torch.tensor(course.ribbon_pts, device=device, dtype=torch.float32)
        self.device = device
        self.lookahead = lookahead
        self.v_des = v_des
        self.kp_att = kp_att
        self.action_map = C.ActionMap()
        # プラント較正の補償: DroneModelはcmd_gain倍のレートを達成し、推力曲線は
        # A=g*(t/hover)^alpha。逆算をここで合わせる(drone_cfg未指定なら旧挙動)。
        cg = getattr(drone_cfg, "cmd_gain", (1.0, 1.0, 1.0)) if drone_cfg else (1.0, 1.0, 1.0)
        self.inv_cmd_gain = torch.tensor([1.0 / g for g in cg], device=device)
        self.hover = getattr(drone_cfg, "hover_thrust", C.HOVER_THRUST) if drone_cfg else C.HOVER_THRUST
        self.alpha = getattr(drone_cfg, "thrust_alpha", 2.0) if drone_cfg else 2.0
        # 弧長テーブル
        seg = torch.linalg.norm(self.ribbon[1:] - self.ribbon[:-1], dim=1)
        self.cum = torch.cat([torch.zeros(1, device=device), torch.cumsum(seg, 0)])

    def act(self, pos_ned: torch.Tensor, vel_ned: torch.Tensor, quat_ned: torch.Tensor,
            cmd_signs) -> torch.Tensor:
        """(N,3),(N,3),(N,4) → (N,4) アクション[-1,1](本番規約レート+推力)。"""
        N = pos_ned.shape[0]
        # 最近点 + lookahead
        d = torch.cdist(pos_ned, self.ribbon)          # (N,M)
        i_near = d.argmin(dim=1)
        s_target = self.cum[i_near] + self.lookahead
        idx = torch.searchsorted(self.cum, s_target.clamp(max=self.cum[-1] - 1e-3))
        target = self.ribbon[idx.clamp(max=len(self.ribbon) - 1)]
        # さらに先を見てカーブ度合いを推定 → カーブでは減速(レート上限±0.4で曲がり切るため)
        idx2 = torch.searchsorted(self.cum, (s_target + self.lookahead).clamp(max=self.cum[-1] - 1e-3))
        target2 = self.ribbon[idx2.clamp(max=len(self.ribbon) - 1)]

        # 所望速度・加速度
        to_t = target - pos_ned
        dir_t = to_t / to_t.norm(dim=1, keepdim=True).clamp(min=1e-6)
        dir_ahead = target2 - target
        dir_ahead = dir_ahead / dir_ahead.norm(dim=1, keepdim=True).clamp(min=1e-6)
        turn = 1.0 - (dir_t * dir_ahead).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)  # 0=直線, 2=Uターン
        v_eff = (self.v_des * (1.0 - 0.85 * (turn / 0.5).clamp(max=1.0))).clamp(min=1.3)
        v_cmd = dir_t * v_eff
        # P制御 + ドラッグフィードフォワード(sysid: a_drag = -0.72 v)
        a_cmd = 1.6 * (v_cmd - vel_ned) + 0.72 * vel_ned
        a_cmd[:, :2] = a_cmd[:, :2].clamp(-4.0, 4.0)
        a_cmd[:, 2] = a_cmd[:, 2].clamp(-3.0, 3.0)

        # 力学: a = g + A·u(u=機体上向きのNED表現)→ A·u = a_cmd - g
        f = a_cmd - torch.tensor([0.0, 0.0, G], device=self.device)
        f_norm = f.norm(dim=1, keepdim=True).clamp(min=1.0)
        u_des = f / f_norm                              # 機体上向き(ホバーで(0,0,-1))
        yaw_des = torch.atan2(dir_t[:, 1], dir_t[:, 0])
        # body z軸(FRD、下向き)の所望NED方向 = -u_des
        z_b_des = -u_des
        # 現在姿勢
        x_b = quat_rotate(quat_ned, torch.tensor([1.0, 0, 0], device=self.device).expand(N, 3))
        y_b = quat_rotate(quat_ned, torch.tensor([0.0, 1, 0], device=self.device).expand(N, 3))
        z_b = quat_rotate(quat_ned, torch.tensor([0.0, 0, 1], device=self.device).expand(N, 3))

        # 姿勢誤差(小角近似): e = z_b × z_b_des をbodyへ射影 → roll/pitchレート
        e = torch.cross(z_b, z_b_des, dim=1)
        e_body_roll = (e * x_b).sum(dim=1)
        e_body_pitch = (e * y_b).sum(dim=1)
        # 誤差ベクトルの回転方向: roll軸まわり = x_b成分、pitch軸まわり = y_b成分
        roll_rate = self.kp_att * e_body_roll
        pitch_rate = self.kp_att * e_body_pitch

        yaw_now = torch.atan2(x_b[:, 1], x_b[:, 0])
        dyaw = (yaw_des - yaw_now + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = (1.5 * dyaw).clamp(-0.8, 0.8)

        # 推力: A = |f| → thrust = hover * (A/g)^(1/alpha)
        A = f_norm.squeeze(1)
        thrust = self.hover * (A / G).clamp(min=0.1).pow(1.0 / self.alpha)

        # 内部FRDレート(=達成したいレート) → 指令へ: プラントゲイン分を割り、
        # 本番指令規約へ(ProductionSigns.command_to_frdの逆 = 同じ符号積)
        rates_frd = torch.stack([roll_rate, pitch_rate, yaw_rate], dim=1) * self.inv_cmd_gain
        sign = torch.tensor(cmd_signs, device=self.device, dtype=rates_frd.dtype)
        rates_cmd = rates_frd * sign  # sign^2=1 なので逆変換も同じ
        cmd = torch.cat([rates_cmd, thrust.unsqueeze(1)], dim=1)
        return self.action_map.from_command(cmd)
