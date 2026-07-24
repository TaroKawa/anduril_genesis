# 固定ポーズ・シーン評価用レンダ: 各ゲート手前(2.5/5/8m)からゲート正対で1枚ずつ。
import os
import sys

import cv2
import torch

from genesis_rl.config import EnvConfig, RenderConfig
from genesis_rl.envs.genesis_race_env import GenesisRaceEnv
from genesis_rl.frames import quat_from_euler_frd_ned

out_dir, seed = sys.argv[1], int(sys.argv[2])
color_dr = len(sys.argv) > 3 and sys.argv[3] == "dr"

cfg = EnvConfig()
cfg.num_envs = 1
cfg.stage = 3
cfg.course_seed = seed
cfg.color_dr = color_dr
cfg.clutter = True
cfg.render = RenderConfig(backend="sequential", width=320, height=180)
env = GenesisRaceEnv(cfg, num_envs=1)
env.reset_idx(env._all_idx)
os.makedirs(out_dir, exist_ok=True)
dev = env.device
zero = torch.zeros(1, device=dev)
n = 0
for k in range(1, env.n_gates):
    for dist in (2.5, 5.0, 8.0):
        gp = env.gate_pos[0, k]
        gn = env.gate_normal[0, k]
        pos = (gp - gn * dist).unsqueeze(0).clone()
        pos[0, 2] -= 0.1
        yaw = env.gate_yaw[0, k].reshape(1)
        quat = quat_from_euler_frd_ned(zero, zero - 0.09, yaw)  # 実飛行の軽い前傾
        env.drone.set_state(pos, quat, torch.tensor([0], device=dev))
        env.active_gate[:] = k
        env._update_ribbon(env._all_idx)
        env._update_glow(env._all_idx)
        env.scene.step()
        rgb = env.rig.render()
        cv2.imwrite(f"{out_dir}/s{seed}_g{k:02d}_d{dist:.0f}.png",
                    cv2.cvtColor(rgb[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
        n += 1
print(f"seed {seed}: {n} frames -> {out_dir}")
