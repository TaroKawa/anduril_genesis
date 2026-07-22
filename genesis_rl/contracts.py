"""観測・アクション契約 — Phase 1 (Genesis) / Phase 2 (本番DCLシム) でバイト互換。

ここを変えたら学習済みcheckpointは転移できない。contract_hash() でckptと照合する。
本番側の対応物:
  - アクション: src_anduril/controller.py::update_attitude_flight_control
      SET_ATTITUDE_TARGET (roll_rate, pitch_rate, yaw_rate [rad/s], thrust 0..1)
  - vec のゲート検出: Spakona vision_rx の obs_gate (YOLOX bbox center / rel_dist)
  - 画像: 30fps 640x360 JPEG → to_resnet() で 224x224
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

# --- タイミング(本番シム仕様 §3.2 / §4.6) ---
PHYS_HZ = 120
DECIMATION = 4          # 1決定 = 4物理ステップ = 30Hz(カメラフレームと同期)
POLICY_HZ = PHYS_HZ / DECIMATION  # 30.0
DT_PHYS = 1.0 / PHYS_HZ
DT_POLICY = DECIMATION * DT_PHYS

# --- カメラ(仕様 §3.8。fx=fy=320 が実飛行で検証済みのintrinsics) ---
IMG_W, IMG_H = 640, 360
FX = FY = 320.0
CX, CY = 320.0, 180.0
CAM_TILT_DEG = 20.0     # ボディから上向き20°(実飛行でfit済み)
VFOV_DEG = float(2.0 * np.degrees(np.arctan2(CY, FY)))  # 58.715°(specの"90"はfyと矛盾)
RESNET_RES = 224

# --- 検出(Spakona YOLOX パイプライン互換) ---
DETECT_LATENCY_S = 0.0173     # 物体検出 17.3 ms → 決定時に見えるのは前フレームの検出
GATE_AREA_MAX = 150000.0      # rel_dist = 1 - bbox_area/GATE_AREA_MAX (Spakona互換)
GATE_OBS_MAX_AGE_S = 0.5      # これより古い検出は未検出扱い

# --- 観測スケール(Spakona rl_config.yaml 互換) ---
RATE_SCALE = 4.0        # gyro [rad/s]
ACCEL_SCALE = 25.0      # accel [m/s^2](accel_max)
V_SCALE = 15.0          # 特権velocity [m/s]
POS_SCALE = 60.0        # 特権position [m]
GATE_REL_SCALE = 30.0   # 特権ゲート相対位置 [m]
MAX_GATES = 40          # 通過ゲートone-hot長(Spakona max_gates)

# --- vec レイアウト(55次元) ---
VEC_GYRO = slice(0, 3)
VEC_ACCEL = slice(3, 6)
VEC_GATE = slice(6, 11)         # [u_n, v_n, visible, rel_dist, age_n]
VEC_ONEHOT = slice(11, 11 + MAX_GATES)
VEC_LAST_ACTION = slice(51, 55)
VEC_DIM = 55

# --- priv レイアウト(39次元) ---
PRIV_DIM = 39

ACTION_DIM = 4

# --- 推力・レートのアクションマッピング(env側。ネットは常に[-1,1]^4) ---
HOVER_THRUST = 0.2742           # sysid: A==g となる指令推力
TAKEOFF_THRUST = 0.265          # これ未満では離陸できない(A(0.265)=9.16 < g)
THRUST_CENTER = 0.3325          # レンジ [0.265, 0.40] の中心
THRUST_HALFSPAN = 0.0675
RATE_LIMITS = (0.4, 0.4, 0.1)   # roll, pitch, yaw [rad/s](ユーザー確認済みの控えめ設定)


@dataclass(frozen=True)
class ActionMap:
    """a ∈ [-1,1]^4 → 物理コマンド (roll_rate, pitch_rate, yaw_rate [rad/s], thrust 0..1)。"""

    rate_limits: tuple[float, float, float] = RATE_LIMITS
    thrust_center: float = THRUST_CENTER
    thrust_halfspan: float = THRUST_HALFSPAN

    def to_command(self, a: torch.Tensor) -> torch.Tensor:
        """(N,4) [-1,1] → (N,4) 物理コマンド。"""
        a = a.clamp(-1.0, 1.0)
        lim = torch.tensor(self.rate_limits, device=a.device, dtype=a.dtype)
        rates = a[:, :3] * lim
        thrust = self.thrust_center + a[:, 3:4] * self.thrust_halfspan
        return torch.cat([rates, thrust], dim=1)

    def from_command(self, cmd: torch.Tensor) -> torch.Tensor:
        """物理コマンド → [-1,1]^4(デモ・スクリプトパイロット用の逆写像)。"""
        lim = torch.tensor(self.rate_limits, device=cmd.device, dtype=cmd.dtype)
        a_rates = (cmd[:, :3] / lim).clamp(-1.0, 1.0)
        a_thrust = ((cmd[:, 3:4] - self.thrust_center) / self.thrust_halfspan).clamp(-1.0, 1.0)
        return torch.cat([a_rates, a_thrust], dim=1)


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_resnet(rgb_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) uint8 → (N,3,224,224) float32 [0,1]。

    ImageNet正規化はエンコーダ内部で行う(Spakona FrozenEncoder互換)。
    本番のJPEGフレームにも同じ変換を適用すること。
    """
    x = rgb_u8.permute(0, 3, 1, 2).float() / 255.0
    x = torch.nn.functional.interpolate(x, size=(RESNET_RES, RESNET_RES), mode="bilinear", align_corners=False)
    return x


def contract_hash() -> str:
    """契約のバージョンハッシュ。checkpointに保存しresume/転移時に照合する。"""
    spec = (
        f"vec{VEC_DIM}-priv{PRIV_DIM}-act{ACTION_DIM}"
        f"-rate{RATE_LIMITS}-thr{THRUST_CENTER}+-{THRUST_HALFSPAN}"
        f"-hz{POLICY_HZ}-res{RESNET_RES}-maxg{MAX_GATES}"
    )
    return hashlib.sha256(spec.encode()).hexdigest()[:16]
