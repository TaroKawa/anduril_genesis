# -*- coding: utf-8 -*-
"""VQ1(レガシー/テレメトリ有効)で同定した動力学が VQ2(本番)へ持ち込めるかを判定する。

VQ1レガシーは「VQ2ローンチ前と同じ挙動」の旧ビルドなので、真値で測った値を無検証で
VQ2へ持ち込むのは危険。そこで **両ビルドで観測できる信号だけ** を比べる:
  HIGHRES_IMU(gyro/accel) / ACTUATOR_OUTPUT_STATUS / 送った指令
これらが一致するなら、VQ1真値で決めた質量特性・推力曲線・ドラッグ・レート応答は
そのままVQ2の値として使ってよい。ずれるなら、ずれた量だけを補正する。

手順:
  1) VQ1シムを起動して同じプランを録る
       python -m genesis_rl.scripts.fly_dcl --sysid --sysid-plan fit \
           --record-dir runs/vq1_fit --no-video --no-relay --max-sec 170
  2) VQ2シムに差し替えて *同じコマンド* で録る
       ... --record-dir runs/vq2_fit ...
  3) 比較
       python -m genesis_rl.scripts.compare_builds --vq1 runs/vq1_fit --vq2 runs/vq2_fit

比較する2つの視点:
  A. 位相整合した生信号   同じ指令列を同じ初期状態から入れて同じ応答が返るか
  B. パラメータ           指令→gyroの定常ゲイン・時定数、推力水準ごとの -accel_z
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .analyze_truth import _fit_time, _load_jsonl, cmd_at, valid_mask

G = 9.81
AXES = ("roll", "pitch", "yaw")


def load_run(dir_: str) -> dict:
    """IMUと指令だけ読む(VQ2でも成立する最小セット)。"""
    steps = _load_jsonl(os.path.join(dir_, "steps.jsonl"))
    imu = _load_jsonl(os.path.join(dir_, "imu.jsonl"))
    if not steps or not imu:
        raise SystemExit(f"{dir_}: steps.jsonl / imu.jsonl が無い")
    t_cmd = np.array([r["t_wall"] for r in steps], float)
    cmd = np.array([r["cmd"] for r in steps], float)
    ti = np.array([r["t_rx_wall"] for r in imu], float)
    ts = np.array([r["t_sim"] for r in imu], float)
    gy = np.array([r["gyro"] for r in imu], float)
    ac = np.array([r["accel"] for r in imu], float)
    keep = np.r_[True, np.any(np.diff(np.c_[gy, ac], axis=0) != 0, axis=1)]
    t = _fit_time(ts[keep], ti[keep])
    breaks = np.nonzero(np.diff(t_cmd) > 0.2)[0]
    segs = np.split(np.arange(len(t_cmd)), breaks + 1)
    flights = [(t_cmd[s[0]], t_cmd[s[-1]] + 1 / 30.0) for s in segs if len(s) > 30]
    collisions = [e["t_wall"] for e in _load_jsonl(os.path.join(dir_, "events.jsonl"))
                  if e.get("type") == "collision"]
    # 「真値あり」= 状態推定テレメトリ(ATT/POS/ODOM)が来ているかで判定する。
    # truth.jsonl の存在だけで見ると、VQ2でも届く ACTUATOR_OUTPUT_STATUS(kind=ACT)が
    # 入っているせいで VQ2 を VQ1 と誤判定する。
    tp = os.path.join(dir_, "truth.jsonl")
    kinds = set()
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 4000:
                    break
                if line.strip():
                    kinds.add(json.loads(line)["kind"])
    has_truth = bool(kinds & {"ATT", "POS", "ODOM"})
    return {"t": t, "gyro": gy[keep], "accel": ac[keep], "t_cmd": t_cmd, "cmd": cmd,
            "flights": flights, "collisions": collisions,
            "mask": valid_mask(t, flights, collisions),
            "hz": float((keep.sum() - 1) / max(ti[keep][-1] - ti[keep][0], 1e-9)),
            "n_raw": len(imu), "has_truth": has_truth, "truth_kinds": sorted(kinds),
            "dir": dir_}


def phase_curves(r: dict, dur: float = 20.0, dt: float = 0.02):
    """各飛行を「発進からの経過秒」で整列し、飛行間で平均した生信号を返す。"""
    tt = np.arange(0.0, dur, dt)
    gy, ac = [], []
    for (a, b) in r["flights"]:
        if b - a < 0.6 * dur:
            continue
        sel = (r["t"] >= a) & (r["t"] <= b)
        if sel.sum() < 50:
            continue
        loc = r["t"][sel] - a
        gy.append(np.stack([np.interp(tt, loc, r["gyro"][sel, i]) for i in range(3)], axis=1))
        ac.append(np.stack([np.interp(tt, loc, r["accel"][sel, i]) for i in range(3)], axis=1))
    if not gy:
        return None
    return {"t": tt, "gyro": np.mean(gy, axis=0), "accel": np.mean(ac, axis=0),
            "n_flights": len(gy),
            "gyro_sd": np.std(gy, axis=0).mean(), "accel_sd": np.std(ac, axis=0).mean()}


def cmd_agreement(a: dict, b: dict, dur: float, dt: float = 0.02):
    """両runの指令列が「発進からの経過秒」で一致しているかを確認する。

    比較の前提は *同じ指令を入れたら同じ応答が返るか* なので、プランの定数
    (ホバー推力・プラントゲイン)を間で変えていると比較そのものが無効になる。
    """
    tt = np.arange(0.0, dur, dt)
    out = []
    for r in (a, b):
        seqs = []
        for (s, e) in r["flights"]:
            if e - s < 0.6 * dur:
                continue
            seqs.append(cmd_at(r["t_cmd"], r["cmd"], s + tt))
        out.append(np.mean(seqs, axis=0) if seqs else None)
    if out[0] is None or out[1] is None:
        return None
    d = np.abs(out[0] - out[1])
    return {"max_rate": float(d[:, :3].max()), "max_thrust": float(d[:, 3].max()),
            "rms": float(np.sqrt(np.mean(d ** 2)))}


def rate_params(r: dict):
    """指令→生gyro の定常ゲインと時定数(規約に依存しない比較量)。"""
    t, gy, m = r["t"], r["gyro"], r["mask"]
    cmdg = cmd_at(r["t_cmd"], r["cmd"], t)
    changed = np.r_[True, np.any(np.diff(cmdg[:, :3], axis=0) != 0, axis=1)]
    settle = np.ones(len(t), bool)
    for i in np.nonzero(changed)[0]:
        j = np.searchsorted(t, t[i] + 0.15)
        settle[i:j] = False
    out = {}
    dt = np.diff(t)
    k_grid = np.exp(np.linspace(np.log(3.0), np.log(200.0), 60))
    for ax in range(3):
        others = [o for o in range(3) if o != ax]
        single = (np.abs(cmdg[:, others[0]]) < 1e-9) & (np.abs(cmdg[:, others[1]]) < 1e-9)
        sel = m & single & settle & (np.abs(cmdg[:, ax]) > 0.05)
        gain = (float(cmdg[sel, ax] @ gy[sel, ax] / (cmdg[sel, ax] @ cmdg[sel, ax]))
                if sel.sum() > 30 else float("nan"))
        pair = (m & single)[:-1] & (m & single)[1:] & (dt > 0.002) & (dt < 0.05)
        k = float("nan")
        if pair.sum() > 200:
            y, w0, dts = gy[1:, ax][pair], gy[:-1, ax][pair], dt[pair]
            best = None
            for kk in k_grid:
                phi = np.exp(-kk * dts)
                x = (1 - phi) * cmd_at(r["t_cmd"], r["cmd"], t[:-1])[pair, ax]
                rr = y - phi * w0
                g = float(x @ rr / max(x @ x, 1e-12))
                res = float(np.mean((rr - g * x) ** 2))
                if best is None or res < best[0]:
                    best = (res, kk)
            k = float(best[1])
        out[ax] = {"gain": gain, "k": k, "n": int(sel.sum())}
    return out


def thrust_levels(r: dict):
    """推力水準ごとの -accel_z 中央値(=比力A の代理。姿勢を知らなくても比較できる)。"""
    t, m = r["t"], r["mask"]
    cmdg = cmd_at(r["t_cmd"], r["cmd"], t)
    thr = cmdg[:, 3]
    changed = np.r_[True, np.diff(thr) != 0]
    settle = np.ones(len(t), bool)
    for i in np.nonzero(changed)[0]:
        settle[i:np.searchsorted(t, t[i] + 0.15)] = False
    # 回転していない区間に限る(傾いていると -accel_z が推力そのものでなくなる)
    calm = np.all(np.abs(cmdg[:, :3]) < 1e-9, axis=1)
    out = {}
    for lv in sorted(set(np.round(thr[m], 4))):
        sel = m & settle & calm & (np.abs(thr - lv) < 1e-9)
        if sel.sum() < 20:
            continue
        out[float(lv)] = (float(np.median(-r["accel"][sel, 2])), int(sel.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq1", required=True, help="VQ1(テレメトリ有効)で録ったrun")
    ap.add_argument("--vq2", required=True, help="VQ2(本番)で同じプランを録ったrun")
    ap.add_argument("--dur", type=float, default=20.0, help="位相整合で比べる秒数")
    args = ap.parse_args()

    a, b = load_run(args.vq1), load_run(args.vq2)
    print("=" * 78)
    for r, tag in ((a, "VQ1"), (b, "VQ2")):
        print(f"  {tag}: {r['dir']}  IMU {r['hz']:.0f}Hz(生{r['n_raw']}件)  "
              f"飛行{len(r['flights'])}本  衝突{len(r['collisions'])}回  "
              f"真値テレメトリ={'あり' if r['has_truth'] else 'なし'}"
              f"(受信kind: {'/'.join(r['truth_kinds']) or '無'})")
    if a["has_truth"] and b["has_truth"]:
        print("  警告: --vq2 のrunにも真値がある = 両方VQ1で録れている可能性")
    if not a["has_truth"]:
        print("  警告: --vq1 のrunに真値が無い = VQ1レガシーではないビルドで録れている")

    ca_cmd = cmd_agreement(a, b, args.dur)
    if ca_cmd is not None:
        ok = ca_cmd["max_rate"] < 1e-6 and ca_cmd["max_thrust"] < 1e-6
        print(f"\n== 0. 前提確認: 指令列の一致 ==")
        print(f"  レート最大差 {ca_cmd['max_rate']:.6f} rad/s / 推力最大差 "
              f"{ca_cmd['max_thrust']:.6f} → "
              f"{'完全一致(比較可)' if ok else '★不一致: プラン定数が間で変わっている。比較は無効'}")

    print("\n== A. 位相整合した生信号の一致(同じ指令→同じ応答か) ==")
    ca, cb = phase_curves(a, args.dur), phase_curves(b, args.dur)
    if ca and cb:
        print(f"  平均に使った飛行: VQ1 {ca['n_flights']}本 / VQ2 {cb['n_flights']}本 "
              f"(飛行間ばらつき gyro {ca['gyro_sd']:.3f}/{cb['gyro_sd']:.3f} rad/s)")
        for i, nm in enumerate(("gyro_x", "gyro_y", "gyro_z")):
            d = ca["gyro"][:, i] - cb["gyro"][:, i]
            print(f"  {nm}: 差rms {np.sqrt(np.mean(d ** 2)):.4f} rad/s  "
                  f"(VQ1 rms {np.sqrt(np.mean(ca['gyro'][:, i] ** 2)):.3f})")
        for i, nm in enumerate(("accel_x", "accel_y", "accel_z")):
            d = ca["accel"][:, i] - cb["accel"][:, i]
            print(f"  {nm}: 差rms {np.sqrt(np.mean(d ** 2)):.3f} m/s²  "
                  f"(VQ1 rms {np.sqrt(np.mean(ca['accel'][:, i] ** 2)):.2f})")
        # 前半5秒(まだ軌道が発散していない領域)だけの一致も見る
        e = ca["t"] < 5.0
        dg = np.sqrt(np.mean((ca["gyro"][e] - cb["gyro"][e]) ** 2))
        da = np.sqrt(np.mean((ca["accel"][e] - cb["accel"][e]) ** 2))
        print(f"  最初の5秒: gyro差rms {dg:.4f} rad/s / accel差rms {da:.3f} m/s²  "
              f"→ {'同一プラントと見なせる' if dg < 0.05 and da < 0.5 else '差あり(下のパラメータ比較で切り分け)'}")
    else:
        print("  位相整合できる長さの飛行がない")

    print("\n== B. パラメータ比較 ==")
    ra, rb = rate_params(a), rate_params(b)
    print("  指令→生gyro 定常ゲイン / レート時定数k:")
    for ax in range(3):
        ga, gb = ra[ax]["gain"], rb[ax]["gain"]
        ka, kb = ra[ax]["k"], rb[ax]["k"]
        dg = abs(ga - gb) / max(abs(ga), 1e-6) * 100 if np.isfinite(ga * gb) else float("nan")
        print(f"    {AXES[ax]:5s}: gain VQ1 {ga:+.3f} / VQ2 {gb:+.3f} ({dg:4.1f}%差)   "
              f"k VQ1 {ka:5.1f} / VQ2 {kb:5.1f}")
    ta, tb = thrust_levels(a), thrust_levels(b)
    print("  推力水準ごとの -accel_z [m/s²](=比力Aの代理):")
    for lv in sorted(set(ta) | set(tb)):
        va = ta.get(lv, (float("nan"), 0))
        vb = tb.get(lv, (float("nan"), 0))
        d = (va[0] - vb[0]) if np.isfinite(va[0] * vb[0]) else float("nan")
        print(f"    thrust={lv:6.4f}: VQ1 {va[0]:7.3f}(n={va[1]:4d}) / "
              f"VQ2 {vb[0]:7.3f}(n={vb[1]:4d})  差 {d:+.3f}")
    print("\n  判定の目安: gainが5%以内・-accel_zが0.2m/s²以内なら VQ1真値の同定結果を")
    print("  そのままVQ2の値として採用してよい。超える軸/水準だけ補正すること。")


if __name__ == "__main__":
    main()
