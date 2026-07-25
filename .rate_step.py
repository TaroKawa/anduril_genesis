# Genesisプラントの開ループ・レート応答: 指令に対する達成角速度と時定数を測る
import numpy as np
import torch

from genesis_rl import contracts as C
from genesis_rl.config import EnvConfig, RenderConfig
from genesis_rl.envs.genesis_race_env import GenesisRaceEnv

cfg = EnvConfig()
cfg.num_envs = 1
cfg.stage = 3
cfg.render = RenderConfig(backend="none", width=320, height=180)
for k in ("dr_mass", "dr_k_rate", "dr_drag", "dr_hover", "dr_inertia", "dr_cmd_gain", "dr_thrust_alpha"):
    setattr(cfg.drone, k, (1.0, 1.0))
cfg.sensors.noise_scale = 0.0
cfg.max_episode_s = 600.0
cfg.no_gate_timeout_s = 600.0
env = GenesisRaceEnv(cfg, num_envs=1)
env.reset()
dev = env.device
print(f"cfg.inertia={cfg.drone.inertia} k_rate={cfg.drone.k_rate} cmd_gain={cfg.drone.cmd_gain}")
print(f"Box実慣性(理論) Ixx=Iyy={0.9/12*(0.28**2+0.16**2):.5f} Izz={0.9/12*(2*0.28**2):.5f}")

theory = np.array([1., 1., -1.]) * np.array([-1., -1., 1.]) * np.array(cfg.drone.cmd_gain)
print(f"\n理論: 観測gyro = {theory.round(2)} × 指令レート")
for ax, name in enumerate(["roll", "pitch", "yaw"]):
    for amp in (0.3, 1.0):
        env.reset_idx(env._all_idx)
        a = torch.zeros(1, 4, device=dev)
        a[0, ax] = amp
        a[0, 3] = 0.0                       # ホバー推力
        trace = []
        for t in range(30):                 # 1.0s
            obs, priv, r, done, info = env.step(a)
            trace.append(float(obs["vec"][0, ax] * C.RATE_SCALE))
        tr = np.array(trace)
        cmd_rate = amp * C.RATE_LIMITS[ax]
        steady = tr[15:].mean()             # 0.5-1.0s平均
        t63 = next((i for i, v in enumerate(tr) if abs(v) >= 0.63 * abs(steady)), -1)
        print(f" {name:5s} 指令{cmd_rate:+.2f} rad/s → 観測gyro定常 {steady:+.3f} "
              f"(slope {steady / cmd_rate:+.2f}, 理論 {theory[ax]:+.2f})  τ≈{(t63 + 1) / C.POLICY_HZ * 1000:.0f}ms")
