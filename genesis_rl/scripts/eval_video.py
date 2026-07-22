"""学習済みチェックポイントをGenesis上で推論し、飛行動画(mp4)を書き出す。

  # 旧ResNet構成のckpt(自動判別)を評価
  uv run python -m genesis_rl.scripts.eval_video \
      --ckpt checkpoints_old_resnet/best_gates.pt \
      --out checkpoints_old_resnet/eval_flight.mp4 --stage 2 --seed 1

  # 新DINOv2+時系列構成のckpt
  uv run python -m genesis_rl.scripts.eval_video --ckpt checkpoints/latest.pt

チェックポイントのactor構造(旧: ResNet18+MLP / 新: DINOv2+時系列Transformer)は
state dictのキーから自動判別する。3画面(FPV+チェイス/俯瞰)合成はpreviewと同じ。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class LegacyActor(nn.Module):
    """旧構成のactor(凍結ResNet18特徴512+vec55 → MLP512×2 → tanh-Gaussian)。

    旧チェックポイント(契約hash 5b93…世代)の再生専用。
    """

    def __init__(self, feat_dim: int = 512, vec_dim: int = 55, hidden: int = 512, act_dim: int = 4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feat_dim + vec_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    @torch.no_grad()
    def act(self, feat, vec, deterministic=True):
        h = self.body(torch.cat([feat, vec], dim=1))
        mean = self.mean(h)
        if deterministic:
            return torch.tanh(mean)
        std = self.log_std(h).clamp(-5.0, 2.0).exp()
        return torch.tanh(mean + std * torch.randn_like(mean))


class LoadedPolicy:
    """ckptからactor+エンコーダを復元し、単一envの推論(履歴管理込み)を提供する。"""

    def __init__(self, ckpt_path: str, device: torch.device, num_envs: int = 1):
        from .. import contracts as C
        from ..models.encoder import FrozenDINOv2, FrozenResNet18

        self.C = C
        self.device = device
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload["agent"]["actor"]
        self.legacy = not any(k.startswith("trunk.") for k in sd)
        if self.legacy:
            self.encoder = FrozenResNet18(bf16=True).to(device).eval()
            self.actor = LegacyActor().to(device).eval()
            self.actor.load_state_dict(sd)
            print(f"[eval] legacy actor (ResNet18+MLP) loaded: {ckpt_path} "
                  f"(updates={payload.get('learner_step', '?')})")
        else:
            from ..models.actor import SACActor
            self.encoder = FrozenDINOv2(bf16=True).to(device).eval()
            self.actor = SACActor().to(device).eval()
            self.actor.load_state_dict(sd)
            self.feat_hist = torch.zeros(num_envs, C.HIST_K, C.FEAT_DIM, device=device)
            self.vec_hist = torch.zeros(num_envs, C.HIST_K, C.VEC_DIM, device=device)
            print(f"[eval] temporal actor (DINOv2+Transformer) loaded: {ckpt_path} "
                  f"(updates={payload.get('learner_step', '?')})")

    @torch.no_grad()
    def act(self, rgb_u8: torch.Tensor, vec: torch.Tensor, done_prev: torch.Tensor | None = None):
        feat = self.encoder(self.C.to_resnet(rgb_u8))
        if self.legacy:
            return self.actor.act(feat, vec, deterministic=True)
        if done_prev is not None and done_prev.any():
            self.feat_hist[done_prev, :] = 0.0
            self.vec_hist[done_prev, :] = 0.0
        self.feat_hist = torch.cat([self.feat_hist[:, 1:], feat.unsqueeze(1)], dim=1)
        self.vec_hist = torch.cat([self.vec_hist[:, 1:], vec.unsqueeze(1)], dim=1)
        return self.actor.act(self.feat_hist, self.vec_hist, deterministic=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--out", type=str, default="checkpoints/eval_flight.mp4")
    ap.add_argument("--stage", type=int, default=2, help="コースstage(旧ckptの緩カーブ=2)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--noise", type=float, default=0.6, help="センサーノイズスケール")
    args = ap.parse_args()

    import cv2
    import imageio

    from .. import contracts as C
    from ..config import EnvConfig, RenderConfig
    from ..envs.genesis_race_env import GenesisRaceEnv
    from .preview import compose_frame

    cfg = EnvConfig()
    cfg.num_envs = 1
    cfg.stage = args.stage
    cfg.course_seed = args.seed
    cfg.render = RenderConfig(backend="sequential", width=320, height=180)
    # 評価は公称ダイナミクス(DRなし)・指定ノイズで
    cfg.drone.dr_mass = cfg.drone.dr_k_rate = cfg.drone.dr_drag = cfg.drone.dr_hover = cfg.drone.dr_inertia = (1.0, 1.0)
    cfg.sensors.noise_scale = args.noise
    cfg.max_episode_s = 120.0

    env = GenesisRaceEnv(cfg, num_envs=1, extra_cameras=True)
    policy = LoadedPolicy(args.ckpt, env.device, num_envs=1)
    obs, priv = env.reset()

    hall = env.course.hall
    frames = []
    n_steps = int(args.duration * C.POLICY_HZ)
    max_gates_seen = 0
    episodes = 0
    done_prev = torch.zeros(1, dtype=torch.bool, device=env.device)
    for step in range(n_steps):
        a = policy.act(obs["rgb"], obs["vec"], done_prev)
        obs, priv, reward, done, info = env.step(a)
        done_prev = done.clone()
        max_gates_seen = max(max_gates_seen, int(info["gates_passed"][0].item()))
        episodes += int(done[0].item())

        # 俯瞰カメラはドローンを追うTVカメラ(preview.pyと同じ)
        st_w = env.drone_entity.get_pos()
        p = st_w[0] if st_w.ndim == 2 else st_w
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        import math as _m
        dcx, dcy = -px, -py
        n = _m.hypot(dcx, dcy) or 1.0
        cx_ = max(min(px + dcx / n * 14.0 - 7.0, hall.length / 2 - 2.0), -hall.length / 2 + 2.0)
        cy_ = max(min(py + dcy / n * 14.0, hall.width / 2 - 2.0), -hall.width / 2 + 2.0)
        env.extra_cams["overview"].set_pose(pos=(cx_, cy_, min(pz + 7.5, hall.height - 1.0)),
                                            lookat=(px, py, pz))

        fpv, _, _, _ = env.extra_cams["fpv_hd"].render(rgb=True)
        chase, _, _, _ = env.extra_cams["chase"].render(rgb=True)
        ov, _, _, _ = env.extra_cams["overview"].render(rgb=True)
        fpv = np.ascontiguousarray(np.asarray(fpv)[..., :3]).astype(np.uint8)
        chase = np.ascontiguousarray(np.asarray(chase)[..., :3]).astype(np.uint8)
        ov = np.ascontiguousarray(np.asarray(ov)[..., :3]).astype(np.uint8)

        det = env.det_queue.read()[0]
        if det[2] > 0.5:
            u = int(det[0].item() * C.CX + C.CX)
            v = int(det[1].item() * C.CY + C.CY)
            cv2.drawMarker(fpv, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)

        st = env.drone.state()
        pos = st["pos_ned"][0].tolist()
        spd = st["vel_ned"][0].norm().item()
        texts = [
            f"POLICY EVAL  t={step / C.POLICY_HZ:5.1f}s  gate {int(env.active_gate[0].item())}"
            f"/{env.n_gates}  best={max_gates_seen}  ep={episodes}",
            f"pos N={pos[0]:.1f} E={pos[1]:.1f} alt={-pos[2]:.1f}m  v={spd:.1f}m/s"
            f"  a=[{a[0,0]:.2f},{a[0,1]:.2f},{a[0,2]:.2f},{a[0,3]:.2f}]",
        ]
        if bool(done[0].item()):
            texts.append("RESET (collision/timeout/finish)")
        frames.append(compose_frame(fpv, chase, ov, texts))

        if step % 90 == 0:
            print(f"  t={step / C.POLICY_HZ:5.1f}s gate={int(env.active_gate[0])} "
                  f"best={max_gates_seen} ep={episodes} v={spd:.1f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out), fps=int(C.POLICY_HZ), codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    print(f"\nsaved: {out}  ({len(frames)} frames, best gates={max_gates_seen}, episodes={episodes})")


if __name__ == "__main__":
    main()
