"""学習側の観測がデプロイ(dcl/client.py)と同じ規約かを守る回帰テスト。

2026-07-26に実測で見つかったズレの再発防止。どれも「学習では成立するが実機で崩れる」
種類なので、環境を起動しない純粋な単体レベルで固定しておく。
"""

from __future__ import annotations

import math

import torch

from genesis_rl import contracts as C
from genesis_rl.config import SensorConfig
from genesis_rl.course import GATE_OUTER
from genesis_rl.frames import quat_from_euler_frd_ned
from genesis_rl.sensors.gate_detector import SimGateDetector


def _det(noise_scale: float = 0.0) -> SimGateDetector:
    cfg = SensorConfig()
    cfg.noise_scale = noise_scale
    return SimGateDetector(1, cfg, torch.device("cpu"))


def _level_quat():
    z = torch.zeros(1)
    return quat_from_euler_frd_ned(z, z, z)


def test_rel_dist_matches_deploy_bbox_convention():
    """rel_dist は deploy(1 - 実bbox面積/GATE_AREA_MAX)と同じ値を返すこと。

    実bbox面積 = det_bbox_gain × (FX·GATE_OUTER/d)²(既知距離レンダでの実測 1.10倍)。
    ここがズレると方策が距離を誤認する(旧: deploy側 gate_area_max=25000 で6倍過小)。
    """
    det = _det()
    for d in (5.0, 8.0, 15.0):
        gate = torch.tensor([[[d, 0.0, 0.0]]])
        out = det.detect_scene(torch.zeros(1, 3), _level_quat(), gate,
                               torch.tensor([[[1.0, 0.0, 0.0]]]), noise=False)
        area = SensorConfig().det_bbox_gain * (C.FX * GATE_OUTER / d) ** 2
        expect = max(0.0, 1.0 - area / C.GATE_AREA_MAX)
        assert abs(float(out[0, 3]) - expect) < 1e-4, (d, float(out[0, 3]), expect)


def test_picks_largest_gate_not_the_active_one():
    """deploy の YOLOX は面積最大の1個を採る。学習側も同じ選択規約であること。"""
    det = _det()
    # 正面12mと、やや右10m。近い方(=大きく映る方)が選ばれる。
    gates = torch.tensor([[[12.0, 0.0, 0.0], [10.0, 1.5, 0.0]]])
    normals = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    out = det.detect_scene(torch.zeros(1, 3), _level_quat(), gates, normals, noise=False)
    near = det.detect_scene(torch.zeros(1, 3), _level_quat(), gates[:, 1:], normals[:, 1:],
                            noise=False)
    assert float(out[0, 2]) == 1.0
    assert abs(float(out[0, 0]) - float(near[0, 0])) < 1e-5   # 近い方の u_n と一致
    assert abs(float(out[0, 3]) - float(near[0, 3])) < 1e-5


def test_far_gate_dropped_like_yolox_min_area():
    """deploy の min_area(=500px²、約39m)より小さいbboxは検出されないこと。"""
    det = _det()
    n = torch.tensor([[[1.0, 0.0, 0.0]]])
    d_edge = math.sqrt(SensorConfig().det_bbox_gain * (C.FX * GATE_OUTER) ** 2
                       / SensorConfig().det_min_area)
    for d, want in ((d_edge * 0.8, 1.0), (d_edge * 1.2, 0.0)):
        out = det.detect_scene(torch.zeros(1, 3), _level_quat(),
                               torch.tensor([[[d, 0.0, 0.0]]]), n, noise=False)
        assert float(out[0, 2]) == want, (d, float(out[0, 2]))


def test_invisible_uses_neutral_values():
    """未検出時は deploy の (0,0,0,1) と同じ中立値であること(age_nは env 側が付ける)。"""
    det = _det()
    behind = torch.tensor([[[-10.0, 0.0, 0.0]]])
    out = det.detect_scene(torch.zeros(1, 3), _level_quat(), behind,
                           torch.tensor([[[1.0, 0.0, 0.0]]]), noise=False)
    assert out[0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_pillar_occludes_gate():
    """柱の裏のゲートは検出されないこと(実機YOLOXは当然見えない)。

    経路上を飛んでいる限り遮蔽はほとんど起きないが、コースを外れた姿勢で
    「ゲートを見失う」状況を学習に経験させるために必要。
    """
    det = _det()
    gate = torch.tensor([[[20.0, 0.0, 0.0]]])
    n = torch.tensor([[[1.0, 0.0, 0.0]]])
    valid = torch.tensor([[True]])
    seen = det.detect_scene(torch.zeros(1, 3), _level_quat(), gate, n, noise=False,
                            pillar_xy=torch.tensor([[[10.0, 5.0]]]), pillar_valid=valid)
    hidden = det.detect_scene(torch.zeros(1, 3), _level_quat(), gate, n, noise=False,
                              pillar_xy=torch.tensor([[[10.0, 0.0]]]), pillar_valid=valid)
    assert float(seen[0, 2]) == 1.0, "視線から外れた柱で遮蔽してはいけない"
    assert float(hidden[0, 2]) == 0.0, "視線上の柱はゲートを隠すこと"


def test_delay_config_matches_measured_deploy_latency():
    """映像/検出遅延が実機実測(85ms ≈ 2-3決定フレーム)相当であること。"""
    s = SensorConfig()
    for lo, hi in ((s.img_delay_frames, s.img_delay_frames + s.img_delay_jitter),
                   (s.det_delay_frames, s.det_delay_frames + s.det_delay_jitter)):
        assert lo >= 2 and hi <= 3, (lo, hi)
    # age_n = 遅延段数 × DT_POLICY / GATE_OBS_MAX_AGE_S が実機の 0.17 付近に入ること
    mid = 0.5 * (s.det_delay_frames + s.det_delay_frames + s.det_delay_jitter)
    age_n = mid * C.DT_POLICY / C.GATE_OBS_MAX_AGE_S
    assert 0.13 <= age_n <= 0.21, age_n
