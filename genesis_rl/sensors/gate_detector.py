"""ゲート検出のシミュレーション(本番YOLOXパイプラインの模擬)。

本番(DCL)側の実装は andu_ddrnet 準拠の YOLOX-x で、**視野内の全ゲートを検出してから
bbox面積が最大の1個だけを採用する**。どのゲートが「次に通るべきゲート」かは知らない。
学習側もこれに合わせ、全ゲートを本番intrinsics(fx=fy=320, 640x360, 20°上チルト)で投影し、
可視なものの中から見かけ面積が最大の1個を返す(旧実装はアクティブゲートだけを投影しており、
隣のゲートが大きく映る配置での「別ゲートを追う」挙動が学習で一度も経験されなかった)。

bboxモデル(deployの実測に合わせた):
  w_px = FX·GATE_OUTER·|cosθ|/d   (θ = 視線とゲート法線の角、斜めに見ると横が縮む)
  h_px = FX·GATE_OUTER/d
  area = det_bbox_gain · w_px · h_px     det_bbox_gain=1.10(既知距離レンダでの実測)
可視条件(deploy GateYOLOX と同じ規約):
  前方0.3m以上 / bbox中心が画面内(10pxマージン) / area >= det_min_area(=500px², 約39m)
ノイズ: ピクセルjitter σ_px(d)=base+gain/max(d,1)、至近ドロップアウト、偽検出。

出力: [u_n, v_n, visible, rel_dist]
  rel_dist = clip(1 - area/GATE_AREA_MAX, 0, 1)  (0=至近, 1=遠方/未検出)
"""

from __future__ import annotations

import math

import torch

from ..config import SensorConfig
from ..contracts import CX, CY, FX, GATE_AREA_MAX, IMG_H, IMG_W, CAM_TILT_DEG
from ..course import GATE_OUTER
from ..frames import quat_rotate_inv

DET_DIM = 4  # [u_n, v_n, visible, rel_dist]

_FRAME_MARGIN_PX = 10.0
_MAX_VIEW_ANGLE_DEG = 80.0   # これ以上の斜めは実質検出されない(横幅が潰れる)


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
        self._cos_max = math.cos(math.radians(_MAX_VIEW_ANGLE_DEG))

    # --- 幾何(ノイズなし) ---

    def _project(self, drone_pos_ned, drone_quat_ned, gate_pos_ned, gate_normal):
        """(N,G,·) へブロードキャストして投影する。returns (u, v, d, area, visible_geom)。

        gate_pos_ned/gate_normal は (N,G,3)。単一ゲートは G=1 で渡す。
        """
        rel_ned = gate_pos_ned - drone_pos_ned.unsqueeze(1)                  # (N,G,3)
        q = drone_quat_ned.unsqueeze(1).expand(-1, rel_ned.shape[1], -1)     # (N,G,4)
        rel_body = quat_rotate_inv(q, rel_ned)
        rel_cam = rel_body @ self.R_cb.T
        d = rel_ned.norm(dim=2).clamp(min=1e-6)

        z = rel_cam[:, :, 2].clamp(min=1e-6)
        u = CX + FX * rel_cam[:, :, 0] / z
        v = CY + FX * rel_cam[:, :, 1] / z

        view_dir = rel_ned / d.unsqueeze(2)
        cos_ang = (view_dir * gate_normal).sum(dim=2).abs()
        # 実bboxモデル: 斜めに見ると横幅だけ|cosθ|で縮む(高さはyaw回転で変わらない)
        w_px = FX * GATE_OUTER * cos_ang / d
        h_px = FX * GATE_OUTER / d
        area = self.cfg.det_bbox_gain * w_px * h_px

        in_front = rel_cam[:, :, 2] > 0.3
        m = _FRAME_MARGIN_PX
        in_frame = (u > m) & (u < IMG_W - m) & (v > m) & (v < IMG_H - m)
        big_enough = area >= self.cfg.det_min_area
        angle_ok = cos_ang > self._cos_max
        return u, v, d, area, in_front & in_frame & big_enough & angle_ok

    def _occluded(self, drone_pos_ned, gate_pos_ned, pillar_xy, pillar_valid):
        """柱に隠れて見えないゲートを (N,G) bool で返す。

        柱は床から天井まで通っているので水平面の線分×円の判定でよい。
        実機のYOLOXは柱裏のゲートを当然検出しないが、解析投影はすり抜けてしまうため、
        これが無いと「旋回中にゲートを見失う」状況を学習で一度も経験しない。
        """
        if pillar_xy is None or pillar_xy.shape[1] == 0:
            return torch.zeros(gate_pos_ned.shape[:2], dtype=torch.bool,
                               device=gate_pos_ned.device)
        a = drone_pos_ned[:, None, None, :2]          # (N,1,1,2) 視点
        b = gate_pos_ned[:, :, None, :2]              # (N,G,1,2) ゲート
        p = pillar_xy[:, None, :, :]                  # (N,1,P,2) 柱
        ab = b - a                                    # (N,G,1,2)
        denom = ab.pow(2).sum(-1).clamp(min=1e-6)     # (N,G,1)
        t = ((p - a) * ab).sum(-1) / denom            # (N,G,P) 線分上の射影パラメータ
        closest = a + t.clamp(0.0, 1.0).unsqueeze(-1) * ab
        dist = (p - closest).norm(dim=-1)             # (N,G,P)
        # 柱は1.5m角 → 半径0.75m。機体直近/ゲート直近は遮蔽扱いしない(t端を除外)
        hit = (dist < 0.85) & (t > 0.03) & (t < 0.97)
        if pillar_valid is not None:
            hit = hit & pillar_valid[:, None, :]
        return hit.any(dim=2)

    # --- 本番YOLOX相当(全ゲート → 面積最大の1個) ---

    def detect_scene(
        self,
        drone_pos_ned: torch.Tensor,    # (N,3)
        drone_quat_ned: torch.Tensor,   # (N,4) body FRD → NED
        gate_pos_ned: torch.Tensor,     # (N,G,3) 全ゲート中心
        gate_normal: torch.Tensor,      # (N,G,3) 全ゲート法線
        noise: bool = True,
        pillar_xy: torch.Tensor | None = None,     # (N,P,2) 柱の水平位置(遮蔽判定用)
        pillar_valid: torch.Tensor | None = None,  # (N,P) 有効スロット
    ) -> torch.Tensor:
        """(N,4) [u_n, v_n, visible, rel_dist]。視野内で最も大きく映るゲートを1つ返す。"""
        c, s = self.cfg, self.cfg.noise_scale
        u, v, d, area, visible = self._project(drone_pos_ned, drone_quat_ned,
                                               gate_pos_ned, gate_normal)
        if pillar_xy is not None:
            visible = visible & ~self._occluded(drone_pos_ned, gate_pos_ned,
                                                pillar_xy, pillar_valid)

        if noise:
            sigma_px = c.det_px_base + c.det_px_gain / d.clamp(min=1.0)
            u = u + torch.randn_like(u) * sigma_px * s
            v = v + torch.randn_like(v) * sigma_px * s
            # 至近ドロップアウト(枠が画面外へはみ出しYOLOXが割れる/色検証に落ちる)
            p_drop = c.det_dropout_base + c.det_dropout_close * torch.sigmoid((2.0 - d) / 0.5)
            visible = visible & (torch.rand_like(d) > p_drop * s)

        # 面積最大の可視ゲートを選ぶ(deployの np.argmax(areas) と同じ規約)
        score = torch.where(visible, area, torch.full_like(area, -1.0))
        k = score.argmax(dim=1)                                    # (N,)
        env = torch.arange(u.shape[0], device=u.device)
        any_vis = visible.any(dim=1)
        u_s, v_s, a_s = u[env, k], v[env, k], area[env, k]

        if noise:
            # 偽検出: 画面内のランダム位置にbboxが立つ(ロゴ/リボン等の誤検出)
            outlier = torch.rand_like(a_s) < c.det_outlier_p * s
            u_s = torch.where(outlier, torch.rand_like(u_s) * IMG_W, u_s)
            v_s = torch.where(outlier, torch.rand_like(v_s) * IMG_H, v_s)
            any_vis = any_vis | outlier

        rel_dist = (1.0 - a_s / GATE_AREA_MAX).clamp(0.0, 1.0)
        if noise:
            d_s = d[env, k]
            jit = 1.0 + torch.randn_like(rel_dist) * (0.03 + 0.06 / d_s.clamp(min=1.0)) * s
            rel_dist = (rel_dist * jit).clamp(0.0, 1.0)
        return self._pack(u_s, v_s, any_vis, rel_dist)

    # --- 単一ゲート(報酬用のclosenessなど、アクティブゲートを直接見たいとき) ---

    def detect(
        self,
        drone_pos_ned: torch.Tensor,   # (N,3)
        drone_quat_ned: torch.Tensor,  # (N,4)
        gate_pos_ned: torch.Tensor,    # (N,3) アクティブゲート中心
        gate_normal: torch.Tensor,     # (N,3)
        noise: bool = False,
        pillar_xy: torch.Tensor | None = None,
        pillar_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(N,4)。指定した1ゲートだけを投影する(既定はノイズなし=真値)。"""
        return self.detect_scene(drone_pos_ned, drone_quat_ned,
                                 gate_pos_ned.unsqueeze(1), gate_normal.unsqueeze(1), noise,
                                 pillar_xy=pillar_xy, pillar_valid=pillar_valid)

    @staticmethod
    def _pack(u, v, visible, rel_dist):
        u_n = ((u - CX) / CX).clamp(-1.5, 1.5)
        v_n = ((v - CY) / CY).clamp(-1.5, 1.5)
        vis_f = visible.float()
        # 未検出は中立値(deploy client._build_vec と同じ: center=(0.5,0.5)→(0,0), rel_dist=1)
        return torch.stack([u_n * vis_f, v_n * vis_f, vis_f,
                            torch.where(visible, rel_dist, torch.ones_like(rel_dist))], dim=1)
