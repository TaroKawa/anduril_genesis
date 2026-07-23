"""座標系変換と本番シムの左手系挙動の再現。

内部力学は標準右手系 FRD(body) / NED(world)。Genesis world は右手系 Z-up。
  NED (n,e,d) <-> Genesis world (x,y,z):  n=x, e=-y, d=-z   (x軸まわり180°の固有回転)
  Body FRD    <-> Genesis body FLU:       f=x, r=-y, d=-z

本番シムの癖はインターフェース2箇所のみに適用(値はconfig.EnvConfig.signs_*):
  - cmd_rate_sign: 指令レート→内部FRDレート目標の符号
  - gyro_out_sign: 内部FRD角速度→観測(学習vec)の符号
2026-07-23のDCL実機軸別同定(runs/sysid2_*)で (+1,+1,-1)/(+1,+1,-1) が確定:
  +roll指令=右バンク、+pitch指令=機首上げ(※Spakonaドキュメントの「+pitch→前傾」は
  実機と逆)、+yaw指令=機首左。生gyroは(-ω,-ω,+ω)で、deploy側(client.py)の
  GYRO_OBS_SIGN=-1⊙生 と gyro_out_sign⊙ω が一致する。
accelは標準FRD(ピッチ-17.8°ピン時の実測 (-3.0, 0, -9.34) = -R^T g と一致)。

クォータニオンは全て wxyz 順(Genesis / MAVLink と同じ)。
"""

from __future__ import annotations

import torch

# NED→Genesis world の回転(x軸180°)。ベクトル成分の入れ替えは符号flipで済む。
_NED2WORLD_SIGN = (1.0, -1.0, -1.0)


def ned_to_world(v: torch.Tensor) -> torch.Tensor:
    """(...,3) NED → Genesis world。"""
    s = torch.tensor(_NED2WORLD_SIGN, device=v.device, dtype=v.dtype)
    return v * s


def world_to_ned(v: torch.Tensor) -> torch.Tensor:
    """(...,3) Genesis world → NED(対合なので同じ演算)。"""
    return ned_to_world(v)


def frd_to_flu(v: torch.Tensor) -> torch.Tensor:
    """(...,3) body FRD → Genesis body FLU(同じくx軸180°)。"""
    return ned_to_world(v)


def flu_to_frd(v: torch.Tensor) -> torch.Tensor:
    return ned_to_world(v)


# --- クォータニオン(wxyz)ユーティリティ ---

def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q(...,4) で v(...,3) を回転(bodyベクトル→world)。"""
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[..., 1:]


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """worldベクトル→body。"""
    return quat_rotate(quat_conj(q), v)


def quat_from_euler_frd_ned(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """NED/FRD の ZYX オイラー角(rad) → body(FRD)→NED クォータニオン wxyz。"""
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ],
        dim=-1,
    )


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
    """(...,4) → (...,6) 回転行列の最初の2列(連続な姿勢表現)。"""
    w, x, y, z = q.unbind(-1)
    col0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    col1 = torch.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)], dim=-1)
    return torch.cat([col0, col1], dim=-1)


# --- NED body クォータニオン <-> Genesis world FLU クォータニオン ---
# T = x軸180°回転: q_T = (0,1,0,0)。 q_world_flu = q_T ⊗ q_ned_frd ⊗ q_T
_QT = torch.tensor([0.0, 1.0, 0.0, 0.0])


def quat_ned_frd_to_world_flu(q: torch.Tensor) -> torch.Tensor:
    qt = _QT.to(device=q.device, dtype=q.dtype).expand_as(q)
    return quat_mul(quat_mul(qt, q), qt)


def quat_world_flu_to_ned_frd(q: torch.Tensor) -> torch.Tensor:
    return quat_ned_frd_to_world_flu(q)  # 対合


class ProductionSigns:
    """本番シム(左手系挙動)の符号ブロック。configから注入・全軸個別に切替可能。"""

    def __init__(self, cmd_rate_sign=(-1.0, -1.0, -1.0), gyro_out_sign=(-1.0, -1.0, -1.0),
                 accel_out_sign=(1.0, 1.0, 1.0)):
        self.cmd_rate_sign = tuple(cmd_rate_sign)
        self.gyro_out_sign = tuple(gyro_out_sign)
        self.accel_out_sign = tuple(accel_out_sign)

    def _t(self, s, ref: torch.Tensor) -> torch.Tensor:
        return torch.tensor(s, device=ref.device, dtype=ref.dtype)

    def command_to_frd(self, rates_cmd: torch.Tensor) -> torch.Tensor:
        """ポリシーが出す(本番規約の)レート指令 → 内部FRDレート目標。"""
        return rates_cmd * self._t(self.cmd_rate_sign, rates_cmd)

    def gyro_to_obs(self, omega_frd: torch.Tensor) -> torch.Tensor:
        """内部FRD真値角速度 → 観測ジャイロ(本番HIGHRES_IMU規約)。"""
        return omega_frd * self._t(self.gyro_out_sign, omega_frd)

    def accel_to_obs(self, accel_frd: torch.Tensor) -> torch.Tensor:
        return accel_frd * self._t(self.accel_out_sign, accel_frd)
