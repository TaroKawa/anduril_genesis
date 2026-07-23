# -*- coding: utf-8 -*-
"""record-dir(fly_dcl --record-dir)のログを読み、sim-to-simギャップの容疑を切り分ける。

  UV_PROJECT_ENVIRONMENT=.venv-host uv run python -m genesis_rl.scripts.analyze_flight \
      --dir runs/dcl_0723

出力する診断:
  1) エピソード分割と墜落タイミング(発進→衝突までの秒数・到達ゲート)
  2) one-hot規約チェック: active_gate_index の実値系列(Genesisは通過ゲート、client.pyは
     active_gate_index をそのまま立てる → 実値の始点/増え方で off-by-one を確定)
  3) ゲート検出: 可視率・rel_dist のヒストグラム(Genesis SimGateDetectorと突き合わせる用)
  4) レート追従sysid: 指令レート cmd[:3] に対する次サンプルの gyro 応答比(≈1で追従良)
  5) アクション統計: raw_action[4] の平均/分散(方策が特定方向に張り付いていないか)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load(dir_: str) -> list[dict]:
    recs = []
    with open(os.path.join(dir_, "steps.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    recs = load(args.dir)
    if not recs:
        print("no records.")
        return
    print(f"loaded {len(recs)} decisions from {args.dir}\n")

    t = np.array([r["t_rel"] for r in recs])
    agi = np.array([r["race"]["active_gate_index"] for r in recs])
    col = np.array([r["collision"] for r in recs], dtype=bool)
    vis = np.array([r["gate"]["visible"] for r in recs], dtype=bool)
    rel = np.array([r["gate"]["rel_dist"] for r in recs], float)
    cmd = np.array([r["cmd"] for r in recs], float)             # (N,4) rates+thrust
    gyro = np.array([r["imu"]["gyro"] for r in recs], float)    # (N,3) 本番規約
    act = np.array([r["raw_action"] for r in recs], float)      # (N,4) [-1,1]

    # 1) 墜落タイミング(衝突フラグの立ち上がりで区切る)
    print("== 1. エピソード/墜落 ==")
    crash_idx = np.where(col & ~np.r_[False, col[:-1]])[0]
    if len(crash_idx) == 0:
        print(f"  衝突マークなし。飛行時間 {t[-1]-t[0]:.1f}s、最終 gate={agi[-1]}")
    else:
        prev = 0.0
        for k, ci in enumerate(crash_idx):
            dur = t[ci] - prev
            print(f"  crash#{k+1}: t={t[ci]:6.1f}s  発進から{dur:4.1f}s  到達gate={int(agi[ci])}")
            prev = t[ci]
        print(f"  → 平均生存 {np.mean(np.diff(np.r_[0.0, t[crash_idx]])):.1f}s "
              f"({'発進直後に崩壊=動力学/視覚の疑い濃厚' if np.median(np.diff(np.r_[0.0, t[crash_idx]]))<4 else '航法の疑い'})")

    # 2) one-hot off-by-one
    print("\n== 2. active_gate_index 系列(one-hot規約) ==")
    uniq, first = np.unique(agi, return_index=True)
    print(f"  観測値: {sorted(set(int(x) for x in uniq))}  始点={int(agi[0])}")
    print("  ※ client.pyは onehot[active_gate_index] を立てる。始点が0でなく1なら、")
    print("    または『まだ1つも通過してないのに1』なら Genesis(通過ゲート基準)とズレる。")

    # 3) 検出統計
    print("\n== 3. ゲート検出 ==")
    print(f"  可視率 {vis.mean()*100:4.1f}%   rel_dist[visible] "
          f"min/med/max = {rel[vis].min():.2f}/{np.median(rel[vis]):.2f}/{rel[vis].max():.2f}"
          if vis.any() else "  可視フレームなし(検出が全く効いていない=視覚ギャップ最有力)")

    # 4) レート追従sysid: cmd rate(t) → 生gyro(t+1)。比 = 生gyro / cmd
    #    注: recorderは生HIGHRES_IMU gyro(観測側の符号反転前)を記録している。
    #    Genesis内部規約では 生ω ≈ cmd_rate_sign*cmd = -cmd なので符号は負が正常。
    #    重要なのは |比| = プラントのレートゲイン(Genesisは追従速くほぼ1.0を想定)。
    #    |比|≫1 → 実シムが指令より過剰回転(k_rate過小/オーバーシュート未モデル)。
    #    符号が正 → 生gyroと指令が同符号 = Genesis内部と逆(command_to_frd符号の疑い)。
    print("\n== 4. レート追従sysid(指令→生gyro。開ループ=--sysidで真値) ==")
    for ax, name in enumerate(["roll", "pitch", "yaw"]):
        c = cmd[:-1, ax]
        g = gyro[1:, ax]
        big = np.abs(c) > 0.05
        if big.sum() > 5:
            ratio = np.median(g[big] / c[big])
            gain = abs(ratio)
            sign_ok = ratio < 0  # 生gyroは負が正常(Genesis内部 -cmd)
            note = f"gain={gain:.2f}({'≈1=追従OK' if 0.7<gain<1.4 else '過剰回転' if gain>=1.4 else '過小'})"
            note += f", 符号{'OK' if sign_ok else '正=command符号不整合の疑い'}"
            print(f"  {name:5s}: n={int(big.sum())}  比 median={ratio:+.2f}  {note}")
        else:
            print(f"  {name:5s}: 有意な指令が少なく評価不可")

    # 5) アクション張り付き
    print("\n== 5. raw_action 統計(張り付き検出) ==")
    for i, name in enumerate(["roll", "pitch", "yaw", "thrust"]):
        print(f"  a[{name:6s}] mean={act[:,i].mean():+.2f} std={act[:,i].std():.2f} "
              f"|>0.9|率={np.mean(np.abs(act[:,i])>0.9)*100:4.1f}%")


if __name__ == "__main__":
    main()
