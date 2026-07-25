# 改修後シーンのFPVフレーム収集(方策で飛行しつつ obs["rgb"] を保存)
import os
import sys

import cv2
import torch

from genesis_rl import contracts as C
from genesis_rl.config import EnvConfig, RenderConfig
from genesis_rl.envs.genesis_race_env import GenesisRaceEnv
from genesis_rl.scripts.eval_video import LoadedPolicy

out_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/genesis_newlook"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
color_dr = len(sys.argv) > 3 and sys.argv[3] == "dr"

cfg = EnvConfig()
cfg.num_envs = 1
cfg.stage = 3
cfg.course_seed = 0
cfg.color_dr = color_dr
cfg.clutter = True
cfg.render = RenderConfig(backend="sequential", width=320, height=180)
cfg.per_env_courses = (len(sys.argv) > 4 and sys.argv[4] == 'perenv')
cfg.course_pool = 1
for k in ("dr_mass", "dr_k_rate", "dr_drag", "dr_hover", "dr_inertia", "dr_cmd_gain", "dr_thrust_alpha"):
    setattr(cfg.drone, k, (1.0, 1.0))
cfg.sensors.noise_scale = 1.0
cfg.max_episode_s = 120.0

env = GenesisRaceEnv(cfg, num_envs=1)
policy = LoadedPolicy("checkpoints/latest.pt", env.device, num_envs=1)
obs, priv = env.reset()
os.makedirs(out_dir, exist_ok=True)
done_prev = torch.zeros(1, dtype=torch.bool, device=env.device)
gates = 0
ep = 0
for step in range(int(duration * C.POLICY_HZ)):
    a = policy.act(obs["rgb"], obs["vec"], done_prev)
    obs, priv, reward, done, info = env.step(a)
    done_prev = done.clone()
    gates = max(gates, int(info["gates_passed"][0]))
    ep += int(done[0])
    if step % 5 == 0:
        fr = obs["rgb"][0].cpu().numpy()
        cv2.imwrite(f"{out_dir}/frame_{step:05d}.png", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    if step % 300 == 0:
        print(f"t={step / C.POLICY_HZ:5.1f}s ep={ep} best={gates}", flush=True)
print(f"done: episodes={ep} best_gates={gates} -> {out_dir}")
