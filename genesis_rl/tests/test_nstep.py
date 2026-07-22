import torch

from genesis_rl.training.nstep import NStepAssembler


def _mk(n_envs=2, n=3, gamma=0.9):
    return NStepAssembler(n_envs, n, gamma, vec_dim=1, priv_dim=1, act_dim=1, feat_dim=1,
                          device=torch.device("cpu"))


def _push(asm, t, rew, done=(0, 0), terminal=(0, 0)):
    N = asm.N
    val = torch.full((N, 1), float(t))
    nval = torch.full((N, 1), float(t + 1))
    return asm.push(val, val, val, val,
                    torch.tensor(rew, dtype=torch.float32),
                    torch.tensor(done, dtype=torch.bool),
                    torch.tensor(terminal, dtype=torch.bool),
                    nval, nval, nval)


def test_mature_flush():
    asm = _mk()
    assert _push(asm, 0, (1.0, 1.0)) is None
    assert _push(asm, 1, (1.0, 1.0)) is None
    out = _push(asm, 2, (1.0, 1.0))
    assert out is not None and out["feat"].shape[0] == 2
    # R = 1 + 0.9 + 0.81
    assert torch.allclose(out["rew"], torch.full((2,), 2.71))
    assert torch.allclose(out["gpow"], torch.full((2,), 0.9**3))
    assert torch.allclose(out["feat"].squeeze(), torch.zeros(2))   # s_0
    assert torch.allclose(out["nfeat"].squeeze(), torch.full((2,), 3.0))  # s_3


def test_terminal_flush_all_pending():
    asm = _mk()
    _push(asm, 0, (1.0, 1.0))
    out = _push(asm, 1, (2.0, 2.0), done=(1, 0), terminal=(1, 0))
    # env0: 2行flush(age2: R=1+0.9*2=2.8, gpow=0 / age1: R=2, gpow=0)
    assert out is not None
    assert out["feat"].shape[0] == 2
    rews = sorted(out["rew"].tolist())
    assert abs(rews[0] - 2.0) < 1e-6 and abs(rews[1] - 2.8) < 1e-6
    assert torch.allclose(out["gpow"], torch.zeros(2))
    assert torch.allclose(out["done"], torch.ones(2))


def test_timeout_keeps_bootstrap():
    asm = _mk()
    out = _push(asm, 0, (1.0, 1.0), done=(1, 0), terminal=(0, 0))  # env0タイムアウト
    assert out is not None and out["feat"].shape[0] == 1
    assert abs(out["gpow"][0].item() - 0.9) < 1e-6  # γ^1、ブートストラップ継続
    assert out["done"][0].item() == 0.0


def test_stream_continues_after_reset():
    asm = _mk()
    _push(asm, 0, (1.0, 1.0), done=(1, 1), terminal=(1, 1))
    assert _push(asm, 1, (1.0, 1.0)) is None
    assert _push(asm, 2, (1.0, 1.0)) is None
    out = _push(asm, 3, (1.0, 1.0))
    assert out is not None and out["feat"].shape[0] == 2
    assert torch.allclose(out["feat"].squeeze(), torch.ones(2))  # リセット後の先頭 s_1
