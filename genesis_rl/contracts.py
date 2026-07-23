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

from .user_config import uc   # config.yaml のユーザー調整値(無ければ下の既定値)

# --- タイミング(本番シム仕様 §3.2 / §4.6)。config.yaml: timing ---
PHYS_HZ = uc("timing", "phys_hz", 120)
DECIMATION = uc("timing", "decimation", 4)  # 1決定 = 4物理ステップ = 30Hz(カメラ同期)
POLICY_HZ = PHYS_HZ / DECIMATION  # 30.0
DT_PHYS = 1.0 / PHYS_HZ
DT_POLICY = DECIMATION * DT_PHYS

# --- カメラ(fx=fy=320 が実飛行で検証済みのintrinsics)。config.yaml: camera ---
IMG_W = uc("camera", "img_w", 640)
IMG_H = uc("camera", "img_h", 360)
FX = uc("camera", "fx", 320.0)
FY = uc("camera", "fy", 320.0)
CX = uc("camera", "cx", 320.0)
CY = uc("camera", "cy", 180.0)
CAM_TILT_DEG = uc("camera", "tilt_deg", 20.0)  # ボディから上向き(実飛行でfit済み)
VFOV_DEG = float(2.0 * np.degrees(np.arctan2(CY, FY)))  # 58.715°(specの"90"はfyと矛盾)
RESNET_RES = 224

# --- 検出。config.yaml: observation ---
DETECT_LATENCY_S = uc("observation", "detect_latency_s", 0.0173)  # 物体検出遅延
GATE_AREA_MAX = uc("observation", "gate_area_max", 150000.0)  # rel_dist=1-bbox_area/これ
GATE_OBS_MAX_AGE_S = uc("observation", "gate_obs_max_age_s", 0.5)  # 古い検出は未検出扱い

# --- 観測スケール。config.yaml: observation ---
RATE_SCALE = uc("observation", "rate_scale", 4.0)      # gyro [rad/s]
ACCEL_SCALE = uc("observation", "accel_scale", 25.0)   # accel [m/s^2](accel_max)
V_SCALE = uc("observation", "v_scale", 15.0)           # 特権velocity [m/s]
POS_SCALE = uc("observation", "pos_scale", 60.0)       # 特権position [m]
GATE_REL_SCALE = uc("observation", "gate_rel_scale", 30.0)  # 特権ゲート相対位置 [m]
MAX_GATES = 40          # 通過ゲートone-hot長。VEC layout に直結=構造値のためコード固定

# --- vec レイアウト(55次元) ---
VEC_GYRO = slice(0, 3)
VEC_ACCEL = slice(3, 6)
VEC_GATE = slice(6, 11)         # [u_n, v_n, visible, rel_dist, age_n]
VEC_ONEHOT = slice(11, 11 + MAX_GATES)
VEC_LAST_ACTION = slice(51, 55)
VEC_DIM = 55
# 時系列トークンに入れる動的成分(one-hotは窓内で不変なのでヘッド側で合流)
VEC_DYN_IDX = tuple(range(0, 11)) + tuple(range(51, 55))  # gyro+accel+gate検出+last_action = 15

# --- 時系列・視覚特徴の契約 ---
HIST_K = 6                       # 方策が見る観測履歴長 [決定ステップ] ≈0.2s @30Hz
ENCODER_NAME = "dinov2_vits14"   # 凍結視覚エンコーダ(torch.hub facebookresearch/dinov2)
FEAT_DIM = 384                   # ViT-S埋め込み次元

# --- priv レイアウト(39次元) ---
PRIV_DIM = 39

ACTION_DIM = 4

# --- 推力・レートのアクションマッピング(env側。ネットは常に[-1,1]^4)。config.yaml: action ---
# thrust はホバー(0.2694)中心の対称帯 [0.2388,0.30](a3=0でホバー、+1で上昇/-1で下降)。
# スタートのピン解除は別途 client.py が takeoff_thrust(0.265)を出す。詳細は config.yaml。
HOVER_THRUST = uc("action", "hover_thrust", 0.2742)      # A==g 参考値
TAKEOFF_THRUST = uc("action", "takeoff_thrust", 0.265)   # 離陸/ピン解除の閾値=スタート出力
THRUST_CENTER = uc("action", "thrust_center", 0.2694)    # = ホバー
THRUST_HALFSPAN = uc("action", "thrust_halfspan", 0.0306)  # → thrust ∈ [0.2388, 0.30]
RATE_LIMITS = uc("action", "rate_limits", (1.0, 1.0, 1.0))  # roll/pitch/yaw 指令上限 [rad/s]


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
        f"-{ENCODER_NAME}-feat{FEAT_DIM}-hist{HIST_K}"
    )
    return hashlib.sha256(spec.encode()).hexdigest()[:16]
