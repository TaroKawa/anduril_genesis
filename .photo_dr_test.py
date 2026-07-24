import torch, cv2, numpy as np
from genesis_rl.config import EnvConfig, RenderConfig
from genesis_rl.envs.genesis_race_env import GenesisRaceEnv
cfg = EnvConfig()
cfg.num_envs = 8
cfg.stage = 3
cfg.color_dr = True
cfg.render = RenderConfig(backend="batch", width=320, height=180, photo_dr=1.0)
env = GenesisRaceEnv(cfg, num_envs=8)
obs, priv = env.reset()
for i in range(3):
    obs, *_ = env.step(torch.zeros(8, 4, device=env.device))
rgb = obs["rgb"].cpu().numpy()
for e in range(8):
    cv2.imwrite(f"runs/photo_dr_test/env{e}.png", cv2.cvtColor(rgb[e], cv2.COLOR_RGB2BGR))
print("photo_dr test frames saved; means:", rgb.reshape(8, -1).mean(1).round(1))
