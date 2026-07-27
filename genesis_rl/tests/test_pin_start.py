"""発進拘束(ピン)の状態機械。実シムVQ1の実測仕様(2026-07-28)を固定する。

発進は **ルールベース**: 拘束中は方策の指令を使わず固定の発進推力(pin_start_thrust)を
出す前提なので、解除は方策の出力に依存しない。実機deploy(dcl/client.py がピン解除まで
START_THRUST を出す)と同じ構造。

  ・拘束中は方策の指令が一切通らない(推力も姿勢も)
  ・pin_start_thrust > pin_release_thrust なら、方策の出力に関係なく遅延後に解除
  ・pin_start_thrust が閾値以下(=設定ミス)なら永久に拘束されたまま
Genesisを起こさずに検証できるよう、必要な属性だけ持つスタブへメソッドを適用する。
"""

from types import SimpleNamespace

import torch

from genesis_rl.envs.genesis_race_env import GenesisRaceEnv

APPLY = GenesisRaceEnv._apply_pin
THR = 0.18


def _stub(n: int, delay_steps: int = 2, start_thrust: float = 0.265):
    e = SimpleNamespace()
    e.cfg = SimpleNamespace(pin_release_thrust=THR)
    e.pinned = torch.ones(n, dtype=torch.bool)
    e.pin_timer = torch.zeros(n, dtype=torch.long) - 1
    e.pin_release_steps = delay_steps
    e.pin_start_thrust = start_thrust
    return e


def _cmd(thrusts):
    return torch.tensor([[0.5, -0.3, 0.2, t] for t in thrusts], dtype=torch.float32)


def test_pinned_zeroes_all_commands():
    e = _stub(3)
    out = APPLY(e, _cmd([0.0, 0.175, 0.30]))
    assert torch.allclose(out, torch.zeros_like(out)), out
    assert e.pinned.all()          # 解除は遅延後なので同一ステップでは外れない


def test_release_is_independent_of_policy_action():
    """a3=0相当(0.175 < 閾値0.18)でも、ルールベース発進推力で必ず解除される。"""
    e = _stub(3, delay_steps=2)
    cmd = _cmd([0.0, 0.175, 0.30])
    for _ in range(3):
        out = APPLY(e, cmd)
    assert not e.pinned.any()
    assert torch.allclose(out, cmd)      # 解除後は方策の指令が素通し


def test_stays_pinned_when_start_thrust_below_threshold():
    """設定ミス(発進推力が閾値以下)は実シム同様に永久拘束として現れる。"""
    e = _stub(2, start_thrust=THR)        # ちょうど閾値は「超えていない」
    cmd = _cmd([0.30, 0.30])
    for _ in range(200):
        out = APPLY(e, cmd)
    assert e.pinned.all()
    assert torch.allclose(out, torch.zeros_like(out))


def test_zero_delay_releases_immediately():
    e = _stub(1, delay_steps=0)
    cmd = _cmd([0.0])
    APPLY(e, cmd)                         # 1回目でカウント開始と同時に解除
    assert not e.pinned.any()
    assert torch.allclose(APPLY(e, cmd), cmd)


def test_release_takes_exactly_delay_steps():
    e = _stub(1, delay_steps=5)
    cmd = _cmd([0.30])
    for i in range(5):
        APPLY(e, cmd)
        assert e.pinned.all(), f"step {i} で早く外れた"
    APPLY(e, cmd)
    assert not e.pinned.any()


def test_noop_when_nothing_pinned():
    e = _stub(2)
    e.pinned[:] = False
    cmd = _cmd([0.0, 0.3])
    assert torch.allclose(APPLY(e, cmd), cmd)
