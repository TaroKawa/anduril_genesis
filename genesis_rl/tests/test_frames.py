import math

import torch

from genesis_rl import frames
from genesis_rl.contracts import (ActionMap, TAKEOFF_THRUST, HOVER_THRUST,
                                  RATE_LIMITS, THRUST_CENTER, THRUST_HALFSPAN)


def test_ned_world_roundtrip():
    v = torch.randn(16, 3)
    assert torch.allclose(frames.world_to_ned(frames.ned_to_world(v)), v)


def test_quat_rotate_identity():
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.randn(1, 3)
    assert torch.allclose(frames.quat_rotate(q, v), v, atol=1e-6)


def test_pinned_accel_matches_production():
    """ピッチ-17.8°前傾で静止(ピン留め)時の比力 = -R^T g = 実測 (-3.0, 0, -9.34)。"""
    pitch = torch.tensor([math.radians(-17.8)])
    q = frames.quat_from_euler_frd_ned(torch.zeros(1), pitch, torch.zeros(1))
    g_ned = torch.tensor([[0.0, 0.0, 9.81]])
    # 静止: 比力 f_b = R^T(a - g) = -R^T g
    f_b = -frames.quat_rotate_inv(q, g_ned)
    expected = torch.tensor([[-3.0, 0.0, -9.34]])
    assert torch.allclose(f_b, expected, atol=0.05), f_b


def test_command_sign_end_to_end():
    """+pitch_rate指令 → 内部FRDでは負(前傾)。ジャイロ観測はそれを+として報告(-1×-1)。"""
    signs = frames.ProductionSigns()
    cmd = torch.tensor([[0.0, 0.3, 0.0]])
    omega_sp = signs.command_to_frd(cmd)
    assert omega_sp[0, 1] < 0  # 内部FRDで機首下げ回転
    gyro_obs = signs.gyro_to_obs(omega_sp)
    assert torch.allclose(gyro_obs, cmd)  # 追従できていれば観測は指令と同符号


def test_quat_ned_world_conversion():
    """NED/FRDのピッチ-17.8°前傾 → world/FLUに変換して機首方向(bodyX)が上を向くこと。"""
    pitch = torch.tensor([math.radians(-17.8)])
    q_ned = frames.quat_from_euler_frd_ned(torch.zeros(1), pitch, torch.zeros(1))
    q_world = frames.quat_ned_frd_to_world_flu(q_ned)
    fwd_world = frames.quat_rotate(q_world, torch.tensor([[1.0, 0.0, 0.0]]))
    # NEDで機首下げ = d成分正 = worldではz負…ではなく上? NED d=下、前傾なら機首は下向き
    # world z = -d なので前傾の機首は world z < 0
    assert fwd_world[0, 2] < 0
    # 大きさ: sin(17.8°)
    assert abs(fwd_world[0, 2].item() + math.sin(math.radians(17.8))) < 1e-4


def test_action_map():
    am = ActionMap()
    a = torch.tensor([[0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0], [1.0, -1.0, 1.0, 0.0]])
    cmd = am.to_command(a)
    assert abs(cmd[0, 3].item() - (THRUST_CENTER - THRUST_HALFSPAN)) < 1e-6   # 下限=下降端
    assert abs(cmd[1, 3].item() - (THRUST_CENTER + THRUST_HALFSPAN)) < 1e-6   # 上限=上昇端0.30
    assert abs(cmd[2, 0].item() - RATE_LIMITS[0]) < 1e-6   # roll
    assert abs(cmd[2, 1].item() + RATE_LIMITS[1]) < 1e-6   # pitch(a=-1)
    assert abs(cmd[2, 2].item() - RATE_LIMITS[2]) < 1e-6   # yaw
    # ホバー推力が可動域内(非飽和)
    a_h = am.from_command(torch.tensor([[0.0, 0.0, 0.0, HOVER_THRUST]]))
    assert -1.0 < a_h[0, 3].item() < 1.0
    # 逆写像の往復
    assert torch.allclose(am.to_command(am.from_command(cmd)), cmd, atol=1e-6)
