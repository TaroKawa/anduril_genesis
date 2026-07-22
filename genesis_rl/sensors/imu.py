"""IMUシミュレーション: 40Hz実効レート(3物理ステップ毎)、ノイズ+バイアスwalk+接触スパイク。

比力(accel)は本番HIGHRES_IMUと同じ規約: f_b = R^T (a_world - g)。
静止+水平で (0,0,-9.81)、ピッチ-17.8°ピンで (-3.0, 0, -9.34)。
"""

from __future__ import annotations

import torch

from ..config import SensorConfig
from ..frames import ProductionSigns, quat_rotate_inv

G_NED = 9.81
IMU_DECIMATION = 3  # 120Hz / 3 = 40Hz


class ImuSim:
    def __init__(self, num_envs: int, cfg: SensorConfig, signs: ProductionSigns, device: torch.device):
        self.cfg = cfg
        self.signs = signs
        self.device = device
        self.num_envs = num_envs
        self.accel_bias = torch.zeros(num_envs, 3, device=device)
        self.gyro_bias = torch.zeros(num_envs, 3, device=device)
        self.latest = torch.zeros(num_envs, 6, device=device)  # [gyro(3), accel(3)]
        self.tick_count = 0

    def reset_idx(self, envs_idx: torch.Tensor):
        n = len(envs_idx)
        c = self.cfg
        self.accel_bias[envs_idx] = (torch.rand(n, 3, device=self.device) * 2 - 1) * c.accel_bias_init
        self.gyro_bias[envs_idx] = (torch.rand(n, 3, device=self.device) * 2 - 1) * c.gyro_bias_init
        self.latest[envs_idx] = 0.0

    def tick_analytic(self, quat_ned: torch.Tensor, specific_force_frd: torch.Tensor,
                      omega_frd: torch.Tensor, dt: float):
        """物理ステップごとに呼ぶ。3回に1回だけ観測を更新(40Hz)。

        比力はDroneModelが印加力から解析計算した値(f_b = R^T(F_thrust+F_drag)/m)。
        静止時(印加ゼロ・接触支持)は -R^T g を用いる(ピン留め/着地状態の再現)。
        """
        self.tick_count += 1
        if self.tick_count % IMU_DECIMATION != 0:
            return
        c, s = self.cfg, self.cfg.noise_scale
        dt_s = dt * IMU_DECIMATION

        # 印加比力がほぼゼロ(=支持されている静止状態)なら重力反力を観測する
        g_ned = torch.tensor([0.0, 0.0, G_NED], device=self.device).expand(self.num_envs, 3)
        f_rest = -quat_rotate_inv(quat_ned, g_ned)
        near_zero = specific_force_frd.norm(dim=1, keepdim=True) < 0.5
        f_b = torch.where(near_zero, f_rest, specific_force_frd)

        # バイアスwalk
        self.accel_bias += torch.randn_like(self.accel_bias) * c.accel_bias_walk * (dt_s**0.5) * s
        self.gyro_bias += torch.randn_like(self.gyro_bias) * c.gyro_bias_walk * (dt_s**0.5) * s

        accel = f_b + self.accel_bias + torch.randn_like(f_b) * c.accel_sigma * s
        # 接触スパイク注入(本番実測: 25-1300 m/s^2 のスパイクが混入する)
        spike_mask = torch.rand(self.num_envs, 1, device=self.device) < c.accel_spike_p * s
        spike = (torch.rand_like(f_b) * 2 - 1) * (15.0 + torch.rand_like(f_b) * 45.0)
        accel = torch.where(spike_mask, accel + spike, accel)

        gyro = omega_frd + self.gyro_bias + torch.randn_like(omega_frd) * c.gyro_sigma * s

        self.latest = torch.cat(
            [self.signs.gyro_to_obs(gyro), self.signs.accel_to_obs(accel)], dim=1
        )

    def read(self) -> torch.Tensor:
        """(N,6) [gyro(3), accel(3)] 本番規約の最新40Hzサンプル。"""
        return self.latest
