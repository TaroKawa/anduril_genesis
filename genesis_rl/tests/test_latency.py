import torch

from genesis_rl.latency import DelayQueue


def test_delay_exact():
    q = DelayQueue(num_envs=2, shape=(1,), max_delay=3, device=torch.device("cpu"))
    q.set_delay(torch.tensor([0, 2]))
    for t in range(6):
        q.push(torch.tensor([[float(t)], [float(t)]]))
        out = q.read()
        assert out[0, 0].item() == float(t)               # env0: 遅延0
        assert out[1, 0].item() == float(max(t - 2, 0))   # env1: 遅延2、ウォームアップは最古値


def test_reset_idx():
    q = DelayQueue(num_envs=2, shape=(1,), max_delay=2, device=torch.device("cpu"))
    q.set_delay(torch.tensor([2, 2]))
    for t in range(4):
        q.push(torch.full((2, 1), float(t)))
    q.reset_idx(torch.tensor([0]))
    q.push(torch.full((2, 1), 10.0))
    out = q.read()
    assert out[0, 0].item() == 10.0   # リセット後env0は新しい値のみ
    assert out[1, 0].item() == 2.0    # env1は遅延2のまま(t=4でpushした10の2つ前 = t=2)


def test_age():
    q = DelayQueue(num_envs=1, shape=(1,), max_delay=3, device=torch.device("cpu"))
    q.set_delay(torch.tensor([3]))
    q.push(torch.zeros(1, 1))
    assert q.age()[0].item() == 0     # ウォームアップ中は実効エイジ0
    for _ in range(5):
        q.push(torch.zeros(1, 1))
    assert q.age()[0].item() == 3
