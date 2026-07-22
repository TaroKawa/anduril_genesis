import math

import torch

from genesis_rl.config import SensorConfig
from genesis_rl.contracts import CX, CY, FX, CAM_TILT_DEG
from genesis_rl.frames import quat_from_euler_frd_ned
from genesis_rl.sensors.gate_detector import SimGateDetector


def _detector():
    return SimGateDetector(1, SensorConfig(), torch.device("cpu"))


def test_gate_dead_ahead_level():
    """水平姿勢・正面10m先のゲートは、20°上チルトのカメラでは画面中央より下に写る。"""
    det = _detector()
    pos = torch.zeros(1, 3)
    quat = quat_from_euler_frd_ned(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    gate = torch.tensor([[10.0, 0.0, 0.0]])
    out = det.detect(pos, quat, gate, torch.tensor([[1.0, 0.0, 0.0]]), noise=False)
    assert out[0, 2].item() == 1.0  # visible
    assert abs(out[0, 0].item()) < 1e-5  # 水平中央
    # v = CY + FX*tan(20°) → v_n = FX*tan(20°)/CY > 0(画像で下)
    expected_vn = FX * math.tan(math.radians(CAM_TILT_DEG)) / CY
    assert abs(out[0, 1].item() - expected_vn) < 1e-3


def test_gate_centered_when_pitched_forward():
    """機体が-20°前傾するとカメラ光軸が水平になり、正面のゲートが画面中央に写る。"""
    det = _detector()
    pos = torch.zeros(1, 3)
    quat = quat_from_euler_frd_ned(torch.zeros(1), torch.tensor([math.radians(-CAM_TILT_DEG)]), torch.zeros(1))
    gate = torch.tensor([[10.0, 0.0, 0.0]])
    out = det.detect(pos, quat, gate, torch.tensor([[1.0, 0.0, 0.0]]), noise=False)
    assert out[0, 2].item() == 1.0
    assert abs(out[0, 0].item()) < 1e-4
    assert abs(out[0, 1].item()) < 1e-4


def test_gate_right_offset():
    """右にあるゲートは u_n > 0。"""
    det = _detector()
    pos = torch.zeros(1, 3)
    quat = quat_from_euler_frd_ned(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    gate = torch.tensor([[10.0, 3.0, 0.0]])  # NEDでe正=右
    out = det.detect(pos, quat, gate, torch.tensor([[1.0, 0.0, 0.0]]), noise=False)
    assert out[0, 0].item() > 0.05


def test_behind_not_visible():
    det = _detector()
    pos = torch.zeros(1, 3)
    quat = quat_from_euler_frd_ned(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    gate = torch.tensor([[-10.0, 0.0, 0.0]])
    out = det.detect(pos, quat, gate, torch.tensor([[1.0, 0.0, 0.0]]), noise=False)
    assert out[0, 2].item() == 0.0
    assert out[0, 3].item() == 1.0  # 未検出はrel_dist=1


def test_rel_dist_monotonic():
    det = _detector()
    quat = quat_from_euler_frd_ned(torch.zeros(1), torch.tensor([math.radians(-20.0)]), torch.zeros(1))
    rels = []
    for d in [3.0, 6.0, 12.0, 24.0]:
        out = det.detect(torch.zeros(1, 3), quat, torch.tensor([[d, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]), noise=False)
        rels.append(out[0, 3].item())
    assert rels == sorted(rels)  # 遠いほどrel_dist大


def test_noise_grows_when_close():
    torch.manual_seed(0)
    det = SimGateDetector(1, SensorConfig(det_outlier_p=0.0, det_dropout_base=0.0, det_dropout_close=0.0),
                          torch.device("cpu"))
    quat = quat_from_euler_frd_ned(torch.zeros(1), torch.tensor([math.radians(-20.0)]), torch.zeros(1))

    def spread(d):
        us = []
        for _ in range(300):
            out = det.detect(torch.zeros(1, 3), quat, torch.tensor([[d, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]]), noise=True)
            us.append(out[0, 0].item())
        return torch.tensor(us).std().item()

    assert spread(2.0) > 2.0 * spread(20.0)  # 至近のjitterは遠方より明確に大きい
