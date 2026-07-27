# -*- coding: utf-8 -*-
"""自作シムの動力学モデルを実シム真値と突き合わせて「精度」を数値化する。

  # 現行 config.yaml のパラメータで評価
  python -m genesis_rl.scripts.verify_model --dir runs/vq1_fit_0727

  # 候補パラメータで評価(config.yamlは触らない)
  python -m genesis_rl.scripts.verify_model --dir runs/vq1_fit_0727 \
      --set hover=0.2543 --set alpha=1.84 --set drag_c=0.45 --set k_rate=40 \
      --set cmd_gain=0.93,0.93,0.85 --set lag=0.05

  # 複数runを結合して評価(速度域を跨いだ汎化を見る)
  python -m genesis_rl.scripts.verify_model --dir runs/vq1_* --joint

  # 軌道誤差そのものを最小化してパラメータを詰める(座標降下法・scipy不要)
  #   ※あくまで診断用。遅延↔k_rate などパラメータ間に縮退があるため、中央値だけ下げて
  #     裾(p90)を悪化させる解に落ちることがある。採用値は analyze_truth の直接同定を
  #     一次ソースにし、この最適化は「まだ詰められる余地があるか」の確認に使う。
  python -m genesis_rl.scripts.verify_model --dir runs/vq1_fit_0727 --optimize

測り方(短ホライズン発散):
  実データの各時刻から *真値の状態で初期化* してモデルを H 秒だけ回し、H秒後の
  位置/速度/姿勢の誤差を集計する。長時間積分の発散(カオス)に埋もれず、
  「1秒後にどれだけズレるモデルか」を素直に比較できる。窓は0.5秒ごとにずらす。

モデルは genesis_rl/drone.py と同じ式(NED/FRD・標準右手系):
  ω_sp = cmd_gain ⊙ clamp(u(t-lag), ±rate_max)      cmd_gainは符号込み(=signs_cmd×ゲイン)
  ω̇   = k_rate (ω_sp - ω)
  A    = g (thrust/hover)^alpha
  a    = g_ned + R(q)·(0,0,-A) - (c1 + c2|v|)·v
`--set drag_c=` は線形項のみ(c2=0)の指定と等価。
"""

from __future__ import annotations

import argparse

import numpy as np

from .analyze_truth import (RESAMPLE_HZ, build_grid, cmd_at, concat_grids,
                            om_sign_from_quat, quat_to_R)

G = 9.81


def default_params() -> dict:
    """config.yaml(dynamics)の現行値を初期値として読む。"""
    from ..user_config import uc
    cg = uc("dynamics", "cmd_gain", (1.0, 1.0, 0.89))
    sc = uc("signs", "cmd", (-1.0, -1.0, 1.0))
    k = uc("dynamics", "k_rate", 35.0)
    k = (float(k),) * 3 if isinstance(k, (int, float)) else tuple(float(x) for x in k)
    return {
        # 自作シムの実効ゲイン = signs_cmd × cmd_gain(符号込みで1本にまとめる)
        "cmd_gain": np.array([s * g for s, g in zip(sc, cg)], float),
        "k_rate": np.array(k, float),
        "hover": float(uc("dynamics", "hover_thrust", 0.2694)),
        "alpha": float(uc("dynamics", "thrust_alpha", 1.84)),
        "c1": float(uc("dynamics", "drag_c", 0.64)),
        "c2": float(uc("dynamics", "drag_c2", 0.0)),
        # 鉛直(body z)のドラッグ。未指定なら水平と同じ=等方
        "c1z": float(uc("dynamics", "drag_cz", None) or uc("dynamics", "drag_c", 0.64)),
        "c2z": float(uc("dynamics", "drag_c2z", None) or uc("dynamics", "drag_c2", 0.0)),
        "lag": float(uc("sensor", "act_delay_steps", 0)) / 120.0,
        "rate_max": float(uc("dynamics", "rate_max", 6.0)),
    }


def rollout(p0, v0, q0, w0, u_seq, dt, prm):
    """真値状態から u_seq(各ステップの指令[N,4])を与えて前進積分。最終状態を返す。"""
    p, v, q, w = p0.copy(), v0.copy(), q0.copy(), w0.copy()
    cg, k, rm = prm["cmd_gain"], prm["k_rate"], prm["rate_max"]
    hov, al, c1, c2 = prm["hover"], prm["alpha"], prm["c1"], prm["c2"]
    g_ned = np.array([0.0, 0.0, G])
    for u in u_seq:
        w_sp = cg * np.clip(u[:3], -rm, rm)   # 実シムは指令をクリップ→プラントゲイン
        w = w + k * (w_sp - w) * dt
        # クォータニオン積分(body角速度)
        qd = 0.5 * np.array([
            -q[1] * w[0] - q[2] * w[1] - q[3] * w[2],
            q[0] * w[0] + q[2] * w[2] - q[3] * w[1],
            q[0] * w[1] - q[1] * w[2] + q[3] * w[0],
            q[0] * w[2] + q[1] * w[1] - q[2] * w[0]])
        q = q + qd * dt
        q = q / np.linalg.norm(q)
        R = quat_to_R(q[None])[0]
        A = G * (max(u[3], 1e-6) / hov) ** al
        sp = np.linalg.norm(v)
        vb = R.T @ v                                   # body FRD 速度
        kh, kv = c1 + c2 * sp, prm["c1z"] + prm["c2z"] * sp
        a_drag_b = -np.array([kh * vb[0], kh * vb[1], kv * vb[2]])
        a = g_ned + R @ (np.array([0.0, 0.0, -A]) + a_drag_b)
        v = v + a * dt
        p = p + v * dt
    return p, v, q, w


def quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> float:
    d = abs(float(np.dot(qa, qb)))
    return float(np.degrees(2.0 * np.arccos(min(d, 1.0))))


def evaluate(d: dict, prm: dict, horizon: float = 1.0, stride: float = 0.5,
             om_sign=(1.0, 1.0, 1.0)) -> dict:
    """短ホライズン発散を集計。"""
    dt = 1.0 / RESAMPLE_HZ
    n_h = int(horizon / dt)
    t, m = d["t"], d["mask"]
    om = d["att"]
    # ω を真値の規約(クォータニオン基準)へ揃えてグリッドへ
    om_grid = np.stack([np.interp(t, om["t"], om["om"][:, i] * om_sign[i]) for i in range(3)],
                       axis=1)
    starts = []
    step = int(stride / dt)
    for i in range(0, len(t) - n_h - 1, step):
        if m[i] and m[i + n_h]:
            starts.append(i)
    if not starts:
        raise SystemExit("有効な評価窓がない(飛行区間が短すぎる)")
    lag_n = int(round(prm["lag"] / dt))
    errs = []
    for i in starts:
        u_seq = d["cmd"][max(i - lag_n, 0):i + n_h - lag_n] if lag_n else d["cmd"][i:i + n_h]
        if len(u_seq) < n_h:
            continue
        p, v, q, _ = rollout(d["pos"][i], d["vel"][i], d["q"][i], om_grid[i],
                             u_seq, dt, prm)
        j = i + n_h
        errs.append((np.linalg.norm(p - d["pos"][j]),
                     np.linalg.norm(v - d["vel"][j]),
                     quat_angle_deg(q, d["q"][j])))
    e = np.array(errs)
    return {"n": len(e), "pos_med": float(np.median(e[:, 0])), "pos_p90": float(np.percentile(e[:, 0], 90)),
            "vel_med": float(np.median(e[:, 1])), "att_med": float(np.median(e[:, 2])),
            "att_p90": float(np.percentile(e[:, 2], 90)),
            "score": float(np.median(e[:, 0]) + 0.3 * np.median(e[:, 1])
                           + 0.02 * np.median(e[:, 2]))}


def fmt(r: dict) -> str:
    return (f"位置誤差 中央{r['pos_med']:.3f}m p90{r['pos_p90']:.3f}m | "
            f"速度 {r['vel_med']:.3f}m/s | 姿勢 中央{r['att_med']:.2f}° p90{r['att_p90']:.2f}° "
            f"(窓{r['n']}本)")


def optimize(d: dict, prm: dict, horizon: float, om_sign, stride: float = 0.5) -> dict:
    """座標降下法(倍率グリッド)で軌道誤差を最小化。scipy不要・決定的。"""
    keys = [("hover", 0.02), ("alpha", 0.15), ("c1", 0.4), ("c2", 0.02),
            ("k_rate0", 0.4), ("k_rate1", 0.4), ("k_rate2", 0.4), ("lag", 0.03),
            ("cmd_gain0", 0.10), ("cmd_gain1", 0.10), ("cmd_gain2", 0.10)]
    cur = dict(prm)
    cur["cmd_gain"] = np.array(prm["cmd_gain"], float)
    cur["k_rate"] = np.array(prm["k_rate"], float)
    best = evaluate(d, cur, horizon, stride, om_sign=om_sign)["score"]
    print(f"  初期 score={best:.4f}")
    for it in range(4):
        improved = False
        for key, span in keys:
            for scale in (1.0, 0.5, 0.25):
                delta = span * scale
                for sgn in (+1, -1):
                    trial = dict(cur)
                    trial["cmd_gain"] = np.array(cur["cmd_gain"], float)
                    trial["k_rate"] = np.array(cur["k_rate"], float)
                    if key.startswith("cmd_gain"):
                        ax = int(key[-1])
                        trial["cmd_gain"][ax] = cur["cmd_gain"][ax] * (1.0 + sgn * delta)
                    elif key.startswith("k_rate"):
                        ax = int(key[-1])
                        trial["k_rate"][ax] = cur["k_rate"][ax] * (1.0 + sgn * delta)
                    elif key == "lag":
                        trial["lag"] = max(0.0, cur["lag"] + sgn * delta)
                    elif key == "c2":
                        trial["c2"] = max(0.0, cur["c2"] + sgn * delta)
                    else:
                        trial[key] = cur[key] * (1.0 + sgn * delta)
                    s = evaluate(d, trial, horizon, stride, om_sign=om_sign)["score"]
                    if s < best - 1e-6:
                        best, cur, improved = s, trial, True
        print(f"  反復{it + 1}: score={best:.4f}")
        if not improved:
            break
    return cur


def apply_sets(prm: dict, sets: list[str]) -> dict:
    for s in sets:
        k, _, v = s.partition("=")
        k = k.strip()
        if k in ("cmd_gain", "k_rate"):
            xs = [float(x) for x in v.split(",")]
            prm[k] = np.array(xs * 3 if len(xs) == 1 else xs, float)
        elif k == "drag_c":
            prm["c1"], prm["c2"] = float(v), 0.0
            prm["c1z"], prm["c2z"] = float(v), 0.0   # 等方の線形ドラッグ(旧モデル)
        elif k in prm:
            prm[k] = float(v)
        else:
            raise SystemExit(f"unknown param: {k}(有効: {sorted(prm)} / drag_c)")
    return prm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, nargs="+")
    ap.add_argument("--set", action="append", default=[], help="param=value(複数可)")
    ap.add_argument("--horizon", type=float, default=1.0, help="発散を測る秒数")
    ap.add_argument("--stride", type=float, default=0.5, help="評価窓の間隔[s](大きいほど速い)")
    ap.add_argument("--optimize", action="store_true", help="軌道誤差最小化でパラメータを詰める")
    ap.add_argument("--joint", action="store_true",
                    help="--dir を結合して1つの評価/最適化にする(速度域を跨いで汎化させる)")
    ap.add_argument("--om-sign", type=str, default="",
                    help="ATTITUDE角速度→標準FRDの符号(例 1,-1,1)。既定はrunのクォータニオンから実測")
    args = ap.parse_args()

    dirs, _cache = args.dir, {}
    if args.joint and len(dirs) > 1:
        merged = concat_grids([build_grid(x) for x in dirs])
        dirs = [" + ".join(args.dir)]
        _cache = {dirs[0]: merged}
    for dir_ in dirs:
        d = _cache.get(dir_) or build_grid(dir_)
        om_sign = ([float(x) for x in args.om_sign.split(",")] if args.om_sign
                   else om_sign_from_quat(d))
        print(f"########## {dir_}  (評価サンプル {int(d['mask'].sum())}, "
              f"ATTITUDE→標準FRD符号 {tuple(int(s) for s in om_sign)})")
        base = default_params()
        print("  [現行 config.yaml] " + " ".join(
            f"{k}={np.round(v, 4)}" for k, v in base.items() if k != "rate_max"))
        r0 = evaluate(d, base, args.horizon, args.stride, om_sign=om_sign)
        print(f"  現行モデル:  {fmt(r0)}")

        prm = apply_sets(dict(base), args.set)
        if args.set:
            r1 = evaluate(d, prm, args.horizon, args.stride, om_sign=om_sign)
            print(f"  指定パラメータ: {fmt(r1)}")
        if args.optimize:
            prm = optimize(d, prm, args.horizon, om_sign, args.stride)
            r2 = evaluate(d, prm, args.horizon, args.stride, om_sign=om_sign)
            print(f"  最適化後:    {fmt(r2)}")
            print("\n  ---- config.yaml へ ----")
            cg, kr = prm["cmd_gain"], np.asarray(prm["k_rate"], float)
            print(f"  signs.cmd:             [{np.sign(cg[0]):+.0f}, {np.sign(cg[1]):+.0f}, "
                  f"{np.sign(cg[2]):+.0f}]")
            print(f"  dynamics.cmd_gain:     [{abs(cg[0]):.3f}, {abs(cg[1]):.3f}, {abs(cg[2]):.3f}]")
            print(f"  dynamics.k_rate:       [{kr[0]:.1f}, {kr[1]:.1f}, {kr[2]:.1f}]")
            print(f"  dynamics.hover_thrust: {prm['hover']:.4f}")
            print(f"  dynamics.thrust_alpha: {prm['alpha']:.3f}")
            print(f"  dynamics.drag_c:       {prm['c1']:.3f}"
                  + (f"   # + 2次項 c2={prm['c2']:.4f}(drone.pyに項の追加が必要)"
                     if prm["c2"] > 1e-4 else ""))
            print(f"  sensor.act_delay_steps: {round(prm['lag'] * 120)}   "
                  f"# 実測遅延 {prm['lag'] * 1000:.0f}ms @120Hz物理")
        print()


if __name__ == "__main__":
    main()
