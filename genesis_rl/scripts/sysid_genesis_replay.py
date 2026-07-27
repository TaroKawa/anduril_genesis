# -*- coding: utf-8 -*-
"""Genesis側でDCL同定スケジュール(dcl.client.ScriptedRatePilot)を再生し、
実シムと同一フォーマットのログ(steps.jsonl + imu.jsonl)を出力する検証ツール。

  LD_LIBRARY_PATH=/usr/lib/wsl/lib UV_PROJECT_ENVIRONMENT=.venv-host \
      uv run python -m genesis_rl.scripts.sysid_genesis_replay --out-base runs/genesis_sysid

その後、実シムと同じ解析を両方に当ててギャップを数値比較する:
  ... -m genesis_rl.scripts.analyze_sysid --dir runs/genesis_sysid_rate ...

再生条件: コース無しの最小シーン(ドローンBox単体・衝突無効=天井/壁/床に当たらない)、
DRオフ(公称ダイナミクス)・センサノイズなし(gyro/accelは解析値を直記録)・
スポーン姿勢は実機同様の前傾-17.8°。
記録規約はrecorderと同一: gyro生値 = FRD omega(=-達成レート指令規約)、accel = FRD比力。
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch


def make_min_scene(cfg_env):
    """ドローンBoxだけの衝突なしシーン + DroneModel。(scene, model, device)を返す。"""
    from .. import contracts as C
    from ..drone import DroneModel
    from ..envs.genesis_race_env import _ensure_gs_init
    from ..frames import ProductionSigns

    gs = _ensure_gs_init(0)
    device = torch.device(str(gs.device))
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=C.DT_PHYS, substeps=1),
        rigid_options=gs.options.RigidOptions(dt=C.DT_PHYS, enable_collision=False),
        show_viewer=False,
    )
    vol = 0.28 * 0.28 * 0.16                      # scene_builder._add_droneと同一
    ent = scene.add_entity(
        gs.morphs.Box(pos=(0.0, 0.0, 10.0), size=(0.28, 0.28, 0.16), fixed=False),
        material=gs.materials.Rigid(rho=cfg_env.drone.mass / vol),
    )
    scene.build(n_envs=1)
    signs = ProductionSigns(cfg_env.signs_cmd, cfg_env.signs_gyro, cfg_env.signs_accel)
    model = DroneModel(ent, scene.rigid_solver, cfg_env.drone, signs, 1, device)
    return scene, model, device


def replay(scene, model, device, cfg_env, plan: str, duration: float, out_dir: str):
    from .. import contracts as C
    from ..dcl.client import ScriptedRatePilot

    os.makedirs(out_dir, exist_ok=True)
    pilot = ScriptedRatePilot(plan=plan)
    idx = torch.zeros(1, dtype=torch.long, device=device)
    model.reset_idx(idx, dr=False)

    # スポーン: 実機と同じ前傾-17.8°(NED-FRD、East軸まわりpitch)・静止・高度10m
    th = math.radians(cfg_env.spawn_pitch_deg)
    pos = torch.tensor([[0.0, 0.0, -10.0]], device=device)
    quat = torch.tensor([[math.cos(th / 2), 0.0, math.sin(th / 2), 0.0]], device=device)
    model.set_state(pos, quat, idx)

    t_base = 1_000_000_000.0   # 疑似壁時計(解析はt差分しか見ない)
    f_steps = open(os.path.join(out_dir, "steps.jsonl"), "w", encoding="utf-8")
    f_imu = open(os.path.join(out_dir, "imu.jsonl"), "w", encoding="utf-8")
    # 実シムと同じ *配線規約* で真値も出す(=同じ analyze_truth / verify_model に
    # かけて数値を直接比較できる)。DCLは姿勢をy軸鏡像の左手系で配信するので、
    # 標準FRDのクォータニオンに同じ変換を掛けてから書く。
    f_truth = open(os.path.join(out_dir, "truth.jsonl"), "w", encoding="utf-8")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"sysid": True, "sysid_plan": plan, "genesis_replay": True}, f)

    def _write_truth(sim_t, st, omega_frd):
        from .analyze_truth import fix_odom_quat, quat_to_R
        import numpy as np

        q_std = st["quat_ned"][0].detach().cpu().numpy()[None]
        q_dcl = fix_odom_quat(q_std)              # 対合なので同じ変換で実シム規約へ
        R = quat_to_R(q_dcl)[0]
        rpy = [float(np.arctan2(R[2, 1], R[2, 2])),
               float(-np.arcsin(np.clip(R[2, 0], -1, 1))),
               float(np.arctan2(R[1, 0], R[0, 0]))]
        om = omega_frd[0].detach().cpu().numpy()
        om_rep = (-om).tolist()                   # 実測: ATTITUDE角速度 = -標準FRDω
        p = st["pos_ned"][0].detach().cpu().numpy().tolist()
        v = st["vel_ned"][0].detach().cpu().numpy().tolist()
        rec = {"t_rx_wall": t_base + sim_t, "t_sim": sim_t}
        for kind, vals in (("ATT", rpy + om_rep), ("POS", p + v),
                           ("ODOM", p + q_dcl[0].tolist() + v + om_rep)):
            f_truth.write(json.dumps({**rec, "kind": kind, "v": vals}) + "\n")

    n_dec = int(duration * C.POLICY_HZ)
    sim_t = 0.0
    for k in range(n_dec):
        cmd = pilot.cmd_at(k * C.DT_POLICY)
        f_steps.write(json.dumps({"t_wall": t_base + sim_t, "cmd": list(cmd)}) + "\n")
        cmd_t = torch.tensor([cmd], device=device, dtype=torch.float32)
        for _ in range(C.DECIMATION):
            state = model.state()
            model.apply(cmd_t, state)
            scene.step()
            sim_t += C.DT_PHYS
            new = model.state()
            # 生HIGHRES_IMU gyro相当 = signs_gyro⊙ω(2026-07-27の実測で
            # deploy側 gyro_obs_sign=+1 に統一したので、ここも余計な反転を掛けない)。
            gyro = model.signs.gyro_to_obs(new["omega_frd"])[0].tolist()
            accel = model.signs.accel_to_obs(model.last_specific_force_frd)[0].tolist()
            f_imu.write(json.dumps({"t_rx_wall": t_base + sim_t, "t_sim": sim_t,
                                    "gyro": gyro, "accel": accel}) + "\n")
            _write_truth(sim_t, new, new["omega_frd"])
    f_steps.close()
    f_imu.close()
    f_truth.close()
    st = model.state()
    print(f"[replay] plan={plan}: {n_dec} decisions ({duration:.0f}s) -> {out_dir}/ "
          f"(終了高度 {-st['pos_ned'][0, 2].item():.1f}m, "
          f"水平移動 {st['pos_ned'][0, :2].norm().item():.0f}m)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", default="runs/genesis_sysid")
    ap.add_argument("--plans", nargs="+", default=["rate", "thrust", "drag"],
                    choices=["rate", "roll", "yaw", "thrust", "drag"])
    ap.add_argument("--duration", type=float, default=0.0,
                    help=">0で全プラン共通の再生秒数(既定はプランごとの実機と同等)")
    args = ap.parse_args()

    from ..config import TrainConfig

    cfg_env = TrainConfig().env
    scene, model, device = make_min_scene(cfg_env)

    durations = {"rate": 65.0, "roll": 45.0, "yaw": 50.0, "thrust": 45.0, "drag": 35.0}
    for plan in args.plans:
        replay(scene, model, device, cfg_env, plan,
               args.duration or durations[plan], f"{args.out_base}_{plan}")


if __name__ == "__main__":
    main()
