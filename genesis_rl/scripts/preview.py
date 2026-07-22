"""コースプレビュー動画の生成(学習前の確認用)。

  uv run python -m genesis_rl.scripts.preview --out checkpoints/course_preview.mp4

3画面(FPV 640x360 + チェイス / 俯瞰)を合成したmp4を出力する。
飛行はScriptedPilot(リボン純追跡)が署名付きアクションインターフェース経由で行うため、
ゲートを通過できていれば左手系レート指令の再現が正しいことのend-to-end検証になる。
FPVにはシミュレートされたゲート検出(ノイズ付き)の中心点をオーバーレイする。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def compose_frame(fpv, chase, overview, texts):
    import cv2

    fpv = cv2.resize(np.asarray(fpv), (640, 360))
    chase = cv2.resize(np.asarray(chase), (640, 360))
    ov = cv2.resize(np.asarray(overview), (1280, 720))
    top = np.concatenate([fpv, chase], axis=1)
    frame = np.concatenate([top, ov], axis=0)
    y = 30
    for t in texts:
        cv2.putText(frame, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
    cv2.putText(frame, "FPV (640x360, 20deg tilt)", (12, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "CHASE", (652, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--out", type=str, default="checkpoints/course_preview.mp4")
    ap.add_argument("--duration", type=float, default=45.0, help="動画の長さ [s]")
    ap.add_argument("--policy", choices=["scripted", "random", "hover"], default="scripted")
    ap.add_argument("--v-des", type=float, default=3.0, help="スクリプトパイロットの目標速度 [m/s]")
    args = ap.parse_args()

    import cv2
    import imageio

    from .. import contracts as C
    from ..config import EnvConfig, RenderConfig
    from ..envs.genesis_race_env import GenesisRaceEnv
    from ..scripted_pilot import ScriptedPilot

    cfg = EnvConfig()
    cfg.num_envs = 1
    cfg.stage = args.stage
    cfg.course_seed = args.seed
    cfg.render = RenderConfig(backend="sequential", width=320, height=180)
    # プレビューは公称ダイナミクス・ノイズ控えめで見やすく
    cfg.drone.dr_mass = cfg.drone.dr_k_rate = cfg.drone.dr_drag = cfg.drone.dr_hover = cfg.drone.dr_inertia = (1.0, 1.0)
    cfg.sensors.noise_scale = 0.5
    cfg.sensors.act_delay_jitter = 0
    cfg.no_gate_timeout_s = 12.0  # プレビューはゆっくり飛ぶのでタイムアウトを緩める

    env = GenesisRaceEnv(cfg, num_envs=1, extra_cameras=True)
    pilot = ScriptedPilot(env.course, env.device, v_des=args.v_des)
    env.reset_idx(env._all_idx)

    # コースレポート
    print(f"\n=== course seed={args.seed} stage={args.stage} ===")
    print(f"gates: {env.n_gates}  total arc: {env.course.total_arc:.1f} m  pillars: {len(env.course.pillars)}")
    for i, g in enumerate(env.course.gates):
        c = g.center_ned
        print(f"  gate{i:2d}: N={c[0]:7.1f} E={c[1]:7.1f} alt={-c[2]:5.1f}m yaw={np.degrees(g.yaw):6.1f}deg")

    frames = []
    n_steps = int(args.duration * C.POLICY_HZ)
    max_gates_seen = 0
    for step in range(n_steps):
        state = env.drone.state()
        if args.policy == "scripted":
            action = pilot.act(state["pos_ned"], state["vel_ned"], state["quat_ned"], cfg.signs_cmd)
        elif args.policy == "random":
            action = torch.randn(1, C.ACTION_DIM, device=env.device).clamp(-1, 1) * 0.3
        else:
            action = torch.zeros(1, C.ACTION_DIM, device=env.device)
            action[:, 3] = -0.85  # ≈hover

        obs, priv, reward, done, info = env.step(action)
        max_gates_seen = max(max_gates_seen, int(info["gates_passed"][0].item()))

        # 俯瞰カメラはドローンを横上から追うTVカメラ(固定視点では120mホールが映らない)
        st_w = env.drone_entity.get_pos()
        p = st_w[0] if st_w.ndim == 2 else st_w
        env.extra_cams["overview"].set_pose(
            pos=(float(p[0]) - 8.0, float(p[1]) - 10.0, float(p[2]) + 6.0),
            lookat=(float(p[0]), float(p[1]), float(p[2])))

        fpv, _, _, _ = env.extra_cams["fpv_hd"].render(rgb=True)
        chase, _, _, _ = env.extra_cams["chase"].render(rgb=True)
        ov, _, _, _ = env.extra_cams["overview"].render(rgb=True)
        fpv = np.ascontiguousarray(np.asarray(fpv)[..., :3]).astype(np.uint8)
        chase = np.ascontiguousarray(np.asarray(chase)[..., :3]).astype(np.uint8)
        ov = np.ascontiguousarray(np.asarray(ov)[..., :3]).astype(np.uint8)

        # 検出オーバーレイ(遅延済み観測 = ポリシーが見るもの)
        det = env.det_queue.read()[0]
        if det[2] > 0.5:
            u = int(det[0].item() * C.CX + C.CX)
            v = int(det[1].item() * C.CY + C.CY)
            cv2.drawMarker(fpv, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.putText(fpv, f"rel_dist={det[3].item():.2f}", (u + 14, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        st = env.drone.state()
        pos = st["pos_ned"][0].tolist()
        spd = st["vel_ned"][0].norm().item()
        texts = [
            f"t={step / C.POLICY_HZ:5.1f}s  gate {int(env.active_gate[0].item())}/{env.n_gates}"
            f"  passed(best)={max_gates_seen}",
            f"pos N={pos[0]:.1f} E={pos[1]:.1f} alt={-pos[2]:.1f}m  v={spd:.1f}m/s"
            f"  thrust={env.prev_cmd[0, 3].item():.3f}",
        ]
        if bool(done[0].item()):
            texts.append("RESET (collision/timeout/finish)")
        frames.append(compose_frame(fpv, chase, ov, texts))

        if step % 60 == 0:
            print(f"  t={step / C.POLICY_HZ:5.1f}s gate={int(env.active_gate[0])} best={max_gates_seen} v={spd:.1f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out), fps=int(C.POLICY_HZ), codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    print(f"\nsaved: {out}  ({len(frames)} frames, best gates passed: {max_gates_seen})")


if __name__ == "__main__":
    main()
