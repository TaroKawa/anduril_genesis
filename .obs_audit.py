# 学習環境側の観測(vec)分布を実DCLログと同じ指標で計測する
import sys
import numpy as np
import torch

from genesis_rl import contracts as C
from genesis_rl.config import EnvConfig, RenderConfig
from genesis_rl.envs.genesis_race_env import GenesisRaceEnv
from genesis_rl.scripts.eval_video import LoadedPolicy

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200

cfg = EnvConfig()
cfg.num_envs = 1
cfg.stage = 3
cfg.course_seed = 0
cfg.clutter = True
cfg.render = RenderConfig(backend="sequential", width=320, height=180)
cfg.max_episode_s = 60.0
env = GenesisRaceEnv(cfg, num_envs=1)
policy = LoadedPolicy("checkpoints/latest.pt", env.device, num_envs=1)
obs, priv = env.reset()
done_prev = torch.zeros(1, dtype=torch.bool, device=env.device)

V, A, D = [], [], []
for t in range(steps):
    a = policy.act(obs["rgb"], obs["vec"], done_prev)
    V.append(obs["vec"][0].cpu().numpy().copy())
    A.append(a[0].cpu().numpy().copy())
    # 真の距離とノイズなし検出(rel_dist較正の突合せ用)
    st = env.drone.state()
    act = env.active_gate.clamp(max=env.n_gates - 1)
    gp = env.gate_pos[env._env_ar, act]
    gn = env.gate_normal[env._env_ar, act]
    det = env.detector.detect(st["pos_ned"], st["quat_ned"], gp, gn, noise=False)
    D.append([float((st["pos_ned"] - gp).norm(dim=1)[0]), float(det[0, 3]), float(det[0, 2])])
    obs, priv, r, done, info = env.step(a)
    done_prev = done.clone()

V = np.stack(V); A = np.stack(A); D = np.stack(D)
np.savez("runs/obs_audit_genesis.npz", V=V, A=A, D=D)
g = V[:, 6:11]
vis = g[:, 2] > 0.5
print(f"=== Genesis (n={len(V)}) ===")
print(f" gyro  mean {V[:,0:3].mean(0).round(3)} std {V[:,0:3].std(0).round(3)} absmax {np.abs(V[:,0:3]).max(0).round(3)}")
print(f" accel mean {V[:,3:6].mean(0).round(3)} std {V[:,3:6].std(0).round(3)}")
print(f" gate visible率 {vis.mean()*100:.0f}%")
print(f"   u_n {g[vis,0].mean():+.3f}±{g[vis,0].std():.3f}  v_n {g[vis,1].mean():+.3f}±{g[vis,1].std():.3f}")
print(f"   rel_dist mean {g[vis,3].mean():.3f} min {g[vis,3].min():.3f} p10 {np.percentile(g[vis,3],10):.3f} p90 {np.percentile(g[vis,3],90):.3f}")
print(f"   rel=0の割合(visible中) {(g[vis,3] <= 1e-6).mean()*100:.1f}%")
print(f"   age_n visible時 {g[vis,4].mean():.3f} / 不可視時 {g[~vis,4].mean() if (~vis).any() else float('nan'):.3f}")
print(f" onehot idx {np.unique(V[:,11:51].argmax(1), return_counts=True)}")
print(f" action mean {A.mean(0).round(3)} std {A.std(0).round(3)}")
print(f" 真距離[m] mean {D[:,0].mean():.1f} p10 {np.percentile(D[:,0],10):.1f} p90 {np.percentile(D[:,0],90):.1f}")
