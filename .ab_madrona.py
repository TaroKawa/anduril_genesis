# Genesis側A/B(確率的方策版): 学習時と同じサンプリングで飛ばして成功率を見る。
import json
import sys

import numpy as np
import torch

from genesis_rl import contracts as C
from genesis_rl.config import EnvConfig, RenderConfig
from genesis_rl.envs.genesis_race_env import GenesisRaceEnv
from genesis_rl.scripts.eval_video import LoadedPolicy

out_path = sys.argv[1] if len(sys.argv) > 1 else "runs/genesis_ab3/steps.jsonl"
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
stochastic = (len(sys.argv) > 3 and sys.argv[3] == "stochastic")

cfg = EnvConfig()
cfg.num_envs = 1
cfg.stage = 3
cfg.course_seed = 0
cfg.color_dr = True
cfg.clutter = True
cfg.render = RenderConfig(backend="batch", width=320, height=180)
for k in ("dr_mass", "dr_k_rate", "dr_drag", "dr_hover", "dr_inertia", "dr_cmd_gain", "dr_thrust_alpha"):
    setattr(cfg.drone, k, (1.0, 1.0))
cfg.sensors.noise_scale = 1.0
cfg.max_episode_s = 120.0

env = GenesisRaceEnv(cfg, num_envs=1)
policy = LoadedPolicy("checkpoints/latest.pt", env.device, num_envs=1)


def act(rgb_u8, vec, done_prev):
    feat = policy.encoder(C.to_resnet(rgb_u8))
    if done_prev is not None and done_prev.any():
        policy.feat_hist[done_prev, :] = 0.0
        policy.vec_hist[done_prev, :] = 0.0
    policy.feat_hist = torch.cat([policy.feat_hist[:, 1:], feat.unsqueeze(1)], dim=1)
    policy.vec_hist = torch.cat([policy.vec_hist[:, 1:], vec.unsqueeze(1)], dim=1)
    return policy.actor.act(policy.feat_hist, policy.vec_hist, deterministic=not stochastic)


obs, priv = env.reset()
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
f = open(out_path, "w")
done_prev = torch.zeros(1, dtype=torch.bool, device=env.device)
n_steps = int(duration * C.POLICY_HZ)
ep = 0
gates_hist = []
gates_best = 0
with torch.no_grad():
    for step in range(n_steps):
        a = act(obs["rgb"], obs["vec"], done_prev)
        obs, priv, reward, done, info = env.step(a)
        done_prev = done.clone()
        gates_best = max(gates_best, int(info["gates_passed"][0]))
        if done[0]:
            ep += 1
            gates_hist.append(int(info["done_gates"][0]))
        if step % 10 == 0:
            import cv2
            fr = obs["rgb"][0].cpu().numpy()
            cv2.imwrite(f"{os.path.dirname(out_path)}/frame_{step:05d}.png",
                        cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        f.write(json.dumps({"t_rel": step / C.POLICY_HZ,
                            "gates_passed": int(info["gates_passed"][0]),
                            "raw_action": [float(x) for x in a[0].cpu().numpy()],
                            "done": bool(done[0]),
                            "collision": bool(info["collision"][0])}) + "\n")
        if step % 300 == 0:
            print(f"t={step / C.POLICY_HZ:5.1f}s gate={int(env.active_gate[0])} ep={ep} "
                  f"best={gates_best} hist={gates_hist[-5:]}", flush=True)
f.close()
print(f"done ({'stochastic' if stochastic else 'deterministic'}): episodes={ep}, "
      f"best={gates_best}, gates/ep={gates_hist}")
