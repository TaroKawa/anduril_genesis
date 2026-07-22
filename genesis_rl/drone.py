"""ドローン動力学: sysidベースの力・トルク直接印加 + ボディレート追従ループ。

`gs.morphs.Drone`(cf2x 27g)は質量が30倍違うため使わず、Box剛体に
apply_links_external_force/torque(local=True)で以下を印加する:
  推力(比力sysid): A(thrust) = g * (thrust / hover_thrust)^2、F = m*A*(body上向き)
  ドラッグ(sysid): F = -m * drag_c * v_world(線形)
  レート追従:      τ = I * k_rate * (ω_sp - ω)(一次遅れτ≈1/k_rate)

内部は右手系(Genesis world / body FLU)。指令・観測の左手系変換はframes.ProductionSigns。
"""

from __future__ import annotations

import torch

from .config import DroneConfig
from .contracts import HOVER_THRUST
from .frames import (
    ProductionSigns,
    flu_to_frd,
    frd_to_flu,
    ned_to_world,
    quat_rotate_inv,
    quat_world_flu_to_ned_frd,
    world_to_ned,
)

G = 9.81


class DroneModel:
    def __init__(self, entity, solver, cfg: DroneConfig, signs: ProductionSigns,
                 num_envs: int, device: torch.device):
        self.entity = entity
        self.solver = solver
        self.cfg = cfg
        self.signs = signs
        self.num_envs = num_envs
        self.device = device
        self.base_link_idx = entity.base_link.idx

        c = cfg
        self.inertia = torch.tensor(c.inertia, device=device).expand(num_envs, 3).clone()
        # エピソードごとのDR倍率
        self.mass_mul = torch.ones(num_envs, 1, device=device)
        self.k_rate_mul = torch.ones(num_envs, 1, device=device)
        self.drag_mul = torch.ones(num_envs, 1, device=device)
        self.hover_mul = torch.ones(num_envs, 1, device=device)
        self.inertia_mul = torch.ones(num_envs, 1, device=device)
        # 直近の印加(IMU比力の解析計算に使う)
        self.last_specific_force_frd = torch.zeros(num_envs, 3, device=device)

    def reset_idx(self, envs_idx: torch.Tensor, dr: bool = True):
        n = len(envs_idx)
        c = self.cfg

        def u(lo, hi):
            return torch.rand(n, 1, device=self.device) * (hi - lo) + lo

        if dr:
            self.mass_mul[envs_idx] = u(*c.dr_mass)
            self.k_rate_mul[envs_idx] = u(*c.dr_k_rate)
            self.drag_mul[envs_idx] = u(*c.dr_drag)
            self.hover_mul[envs_idx] = u(*c.dr_hover)
            self.inertia_mul[envs_idx] = u(*c.dr_inertia)
        else:
            for t in (self.mass_mul, self.k_rate_mul, self.drag_mul, self.hover_mul, self.inertia_mul):
                t[envs_idx] = 1.0
        self.last_specific_force_frd[envs_idx] = 0.0

    # --- 状態取得(NED/FRD規約) ---

    def state(self):
        pos_w = self.entity.get_pos()
        quat_w = self.entity.get_quat()
        vel_w = self.entity.get_vel()
        ang_w = self.entity.get_ang()  # world frame角速度
        omega_flu = quat_rotate_inv(quat_w, ang_w)
        return {
            "pos_ned": world_to_ned(pos_w),
            "quat_ned": quat_world_flu_to_ned_frd(quat_w),
            "vel_ned": world_to_ned(vel_w),
            "omega_frd": flu_to_frd(omega_flu),
            "pos_w": pos_w,
            "quat_w": quat_w,
            "vel_w": vel_w,
        }

    # --- 物理ステップごとの印加 ---

    def apply(self, cmd: torch.Tensor, state: dict):
        """cmd (N,4) 本番規約 (roll_rate, pitch_rate, yaw_rate [rad/s], thrust 0..1)。"""
        c = self.cfg
        m = c.mass * self.mass_mul

        omega_sp_frd = self.signs.command_to_frd(cmd[:, :3]).clamp(-c.rate_max, c.rate_max)
        thrust = cmd[:, 3:4].clamp(0.0, 1.0)

        # 推力(比力モデル): A = g * (t/hover)^2、body上向き(FLU +z)
        hover = HOVER_THRUST * self.hover_mul
        A = G * (thrust / hover).pow(2)
        f_thrust_local = torch.cat([torch.zeros_like(A), torch.zeros_like(A), m * A], dim=1)

        # ドラッグ(world系)
        f_drag_world = -m * c.drag_c * self.drag_mul * state["vel_w"]

        # レート追従トルク(body FRD → FLU、localで印加)
        omega_frd = state["omega_frd"]
        I = self.inertia * self.inertia_mul
        alpha_sp = c.k_rate * self.k_rate_mul * (omega_sp_frd - omega_frd)
        tau_frd = I * alpha_sp
        tau_max = I * 40.0
        tau_frd = tau_frd.clamp(-tau_max, tau_max)
        tau_local = frd_to_flu(tau_frd)

        self.solver.apply_links_external_force(
            f_thrust_local, links_idx=[self.base_link_idx], ref="link_com", local=True)
        self.solver.apply_links_external_force(
            f_drag_world, links_idx=[self.base_link_idx], ref="link_com", local=False)
        self.solver.apply_links_external_torque(
            tau_local, links_idx=[self.base_link_idx], ref="link_com", local=True)

        # IMU用の解析比力(重力を除く印加加速度をbodyへ): f_b = R^T (F_thrust + F_drag)/m
        # 推力はbody FLU +z = FRD -z
        f_frd = torch.cat([torch.zeros_like(A), torch.zeros_like(A), -A], dim=1)
        drag_ned = world_to_ned(f_drag_world) / m
        f_frd = f_frd + quat_rotate_inv(state["quat_ned"], drag_ned)
        self.last_specific_force_frd = f_frd

    # --- リセット ---

    def set_state(self, pos_ned: torch.Tensor, quat_ned: torch.Tensor, envs_idx: torch.Tensor):
        from .frames import quat_ned_frd_to_world_flu

        pos_w = ned_to_world(pos_ned)
        quat_w = quat_ned_frd_to_world_flu(quat_ned)
        self.entity.set_pos(pos_w, envs_idx=envs_idx, zero_velocity=True)
        self.entity.set_quat(quat_w, envs_idx=envs_idx, zero_velocity=True)
