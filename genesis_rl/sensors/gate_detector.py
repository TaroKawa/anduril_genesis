"""ゲート検出のシミュレーション(本番YOLOXパイプラインの代替)。

アクティブゲート中心を本番intrinsics(fx=fy=320, 640x360, 20°上チルト)で投影し、
YOLOXのbbox jitterを模したノイズを付与する:
  - ピクセルjitter: σ_px(d) = base + gain/max(d,1)  ← 近いほど大きい
  - ドロップアウト: 至近で増える(bboxが画面からはみ出す状況)
  - 偽検出: 低確率で画面内の一様乱数(ゲート上の白ロゴ等の誤検出の模擬)
出力: [u_n, v_n, visible, rel_dist] (u_n,v_n は中心正規化 [-1,1]相当)
rel_dist = clip(1 - s_px^2/150000, 0, 1)(Spakona GATE_AREA_MAX互換、0=至近, 1=遠方/未検出)
"""

from __future__ import annotations

import math

import torch

from ..config import SensorConfig
from ..contracts import FX, CX, CY, IMG_W, IMG_H, GATE_AREA_MAX, CAM_TILT_DEG
from ..course import GATE_OUTER
from ..frames import quat_rotate_inv

DET_DIM = 4  # [u_n, v_n, visible, rel_dist]


class SimGateDetector:
    def __init__(self, num_envs: int, cfg: SensorConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        t = math.radians(CAM_TILT_DEG)
        # body FRD → camera(x右・y下・z前方=光軸)。カメラは20°上向きチルト。
        # z_cam = (cos t, 0, -sin t)(前+上)、x_cam = body右 (0,1,0)、
        # y_cam = z_cam × x_cam = (sin t, 0, cos t)(≈body下)。
        self.R_cb = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [math.sin(t), 0.0, math.cos(t)],
                [math.cos(t), 0.0, -math.sin(t)],
            ],
            device=device,
        )

    def detect(
        self,
        drone_pos_ned: torch.Tensor,   # (N,3)
        drone_quat_ned: torch.Tensor,  # (N,4) body FRD → NED
        gate_pos_ned: torch.Tensor,    # (N,3) アクティブゲート中心
        gate_yaw: torch.Tensor,        # (N,) ゲート法線方位
        noise: bool = True,
    ) -> torch.Tensor:
        """(N,4) [u_n, v_n, visible, rel_dist]。"""
        c, s = self.cfg, self.cfg.noise_scale
        rel_ned = gate_pos_ned - drone_pos_ned
        rel_body = quat_rotate_inv(drone_quat_ned, rel_ned)          # FRD
        rel_cam = rel_body @ self.R_cb.T                              # camera frame
        d = rel_ned.norm(dim=1)

        z = rel_cam[:, 2].clamp(min=1e-6)
        u = CX + FX * rel_cam[:, 0] / z
        v = CY + FX * rel_cam[:, 1] / z

        # 可視判定: 前方0.3m以上、画面内(10pxマージン)、45m以内、視線とゲート法線の角度<75°
        in_front = rel_cam[:, 2] > 0.3
        in_frame = (u > 10) & (u < IMG_W - 10) & (v > 10) & (v < IMG_H - 10)
        in_range = d < 45.0
        gate_normal = torch.stack([torch.cos(gate_yaw), torch.sin(gate_yaw), torch.zeros_like(gate_yaw)], dim=1)
        view_dir = rel_ned / d.unsqueeze(1).clamp(min=1e-6)
        cos_ang = (view_dir * gate_normal).sum(dim=1).abs()
        angle_ok = cos_ang > math.cos(math.radians(75.0))
        visible = in_front & in_frame & in_range & angle_ok

        if noise:
            sigma_px = c.det_px_base + c.det_px_gain / d.clamp(min=1.0)
            u = u + torch.randn_like(u) * sigma_px * s
            v = v + torch.randn_like(v) * sigma_px * s
            # 至近ドロップアウト
            p_drop = c.det_dropout_base + c.det_dropout_close * torch.sigmoid((2.0 - d) / 0.5)
            visible = visible & (torch.rand_like(d) > p_drop * s)
            # 偽検出
            outlier = torch.rand_like(d) < c.det_outlier_p * s
            u = torch.where(outlier, torch.rand_like(u) * IMG_W, u)
            v = torch.where(outlier, torch.rand_like(v) * IMG_H, v)

        # rel_dist: 投影された外形サイズ s_px = FX * 2.7 / d(画面内にクランプ)
        s_px = (FX * GATE_OUTER / d.clamp(min=0.5)).clamp(max=float(IMG_W))
        rel_dist = (1.0 - s_px.pow(2) / GATE_AREA_MAX).clamp(0.0, 1.0)
        if noise:
            rel_dist = (rel_dist * (1.0 + torch.randn_like(rel_dist) * (0.03 + 0.06 / d.clamp(min=1.0)) * s)).clamp(0.0, 1.0)

        u_n = ((u - CX) / CX).clamp(-1.5, 1.5)
        v_n = ((v - CY) / CY).clamp(-1.5, 1.5)
        vis_f = visible.float()
        # 未検出は中立値(Spakona: center=(0.5,0.5)=画面中央, rel_dist=1)
        u_n = u_n * vis_f
        v_n = v_n * vis_f
        rel_dist = torch.where(visible, rel_dist, torch.ones_like(rel_dist))
        return torch.stack([u_n, v_n, vis_f, rel_dist], dim=1)
