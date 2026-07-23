# -*- coding: utf-8 -*-
"""開ループ同定フライト(fly_dcl --sysid --record-dir DIR)の動力学フィット。

  UV_PROJECT_ENVIRONMENT=.venv-host uv run python -m genesis_rl.scripts.analyze_sysid \
      --dir runs/sysid_rate

入力: DIR/steps.jsonl(30Hz決定=指令の階段) + DIR/imu.jsonl(HIGHRES_IMU全サンプル)。
プラン(rate/thrust/drag)は自動判別不要 — データに存在する種類の区間だけ解析される。

出力する同定:
  1) レート定常ゲイン(軸×振幅) … 達成レート(-生gyro)/指令。線形性・飽和の確認
  2) レートステップ応答       … 遅延(p>0.3)・t63。Genesis k_rate=1/τ の較正値
  3) 推力比力曲線 A(thrust)    … ln A = ln g + α ln(t/h) フィット → α(2乗則?), h(hover)
  4) 線形ドラッグ c            … 水平惰性中の体x/y比力の指数減衰 f_h ∝ e^{-ct}
  5) Genesis較正の推奨値まとめ(config.py DroneConfig / dcl/client.py RATE_CMD_GAIN)

規約: 生gyroは指令規約と逆符号(達成レート = -gyro)。accelは標準FRD比力
(水平静止 ≈ (0,0,-9.81)、f_z = -A + c*v_up、水平時の体x/y比力 = 水平ドラッグ)。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

G = 9.81
AXES = ("roll", "pitch", "yaw")


def _load_jsonl(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_run(dir_: str):
    steps = _load_jsonl(os.path.join(dir_, "steps.jsonl"))
    imu_path = os.path.join(dir_, "imu.jsonl")
    if not os.path.exists(imu_path):
        raise SystemExit("imu.jsonl がありません(旧recorder)。sysid v2で録り直してください。")
    imu = _load_jsonl(imu_path)
    meta = {}
    mp = os.path.join(dir_, "meta.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    return steps, imu, meta


def dedup_imu(imu: list[dict]):
    """メッセージレート > 実効更新レートの場合の重複値を落とし、
    シム時刻→壁時計の線形フィットでジッタの少ない時刻を作る。"""
    g = np.array([r["gyro"] for r in imu], float)
    a = np.array([r["accel"] for r in imu], float)
    tw = np.array([r["t_rx_wall"] for r in imu], float)
    ts = np.array([r["t_sim"] for r in imu], float)
    changed = np.r_[True, np.any(np.diff(g, axis=0) != 0, axis=1)
                    | np.any(np.diff(a, axis=0) != 0, axis=1)]
    g, a, tw, ts = g[changed], a[changed], tw[changed], ts[changed]
    # t_sim が単調ならフィット時刻を使う(UDP受信ジッタ除去。オフセット=平均転送遅延は残る)
    if len(ts) > 10 and np.all(np.diff(ts) > 0):
        A_ = np.vstack([ts, np.ones_like(ts)]).T
        coef, *_ = np.linalg.lstsq(A_, tw, rcond=None)
        t = A_ @ coef
    else:
        t = tw
    return {"t": t, "gyro": g, "accel": a,
            "rate_hz": 1.0 / np.median(np.diff(t)) if len(t) > 2 else 0.0,
            "n_raw": len(imu)}


def load_collisions(dir_: str) -> list[float]:
    """events.jsonl の衝突時刻(steps.jsonl の30Hzサンプリングでは取り漏れる)。"""
    p = os.path.join(dir_, "events.jsonl")
    if not os.path.exists(p):
        return []
    return [e["t_wall"] for e in _load_jsonl(p) if e.get("type") == "collision"]


def build_segments(steps: list[dict], collisions: list[float]):
    """指令一定の区間 [(t0, t1, cmd4)] に分割。

    決定間ギャップ>0.2s(=リセット/非飛行を跨ぐ)で強制分割し、
    衝突時刻の [t-0.7, t+2.0] に重なる区間は捨てる(転倒・リセット直後の暴れ)。
    """
    t = np.array([r["t_wall"] for r in steps], float)
    cmd = np.array([r["cmd"] for r in steps], float)
    segs = []
    i = 0
    while i < len(t):
        j = i
        while (j + 1 < len(t) and np.array_equal(cmd[j + 1], cmd[i])
               and t[j + 1] - t[j] < 0.2):
            j += 1
        t1 = t[j + 1] if (j + 1 < len(t) and t[j + 1] - t[j] < 0.2) else t[j] + 1.0 / 30.0
        segs.append((t[i], t1, cmd[i]))
        i = j + 1
    ok = [s for s in segs
          if not any(s[0] < tc + 2.0 and s[1] > tc - 0.7 for tc in collisions)]
    return ok, len(segs) - len(ok)


def _win(imu, t0, t1):
    m = (imu["t"] >= t0) & (imu["t"] < t1)
    return m


def steady_window(t0, t1):
    """定常値の取得窓: 区間後半(立ち上がりを避ける)。"""
    dur = t1 - t0
    return t0 + max(0.15, 0.4 * dur), t1 - 0.02


# ---------------------------------------------------------------- 1,2) レート

def analyze_rates(segs, imu):
    print("== 1. レート定常ゲイン(達成レート=-生gyro / 指令) ==")
    rows = []   # (axis, amp, cmd, gain, n)
    for (t0, t1, c) in segs:
        if t1 - t0 < 0.3:
            continue
        rates = c[:3]
        nz = np.nonzero(np.abs(rates) > 1e-9)[0]
        if len(nz) != 1:
            continue
        ax = int(nz[0])
        w0, w1 = steady_window(t0, t1)
        m = _win(imu, w0, w1)
        if m.sum() < 3:
            continue
        achieved = np.median(-imu["gyro"][m, ax])
        rows.append((ax, abs(rates[ax]), rates[ax], achieved / rates[ax], int(m.sum())))
    if not rows:
        print("  レート区間なし(このrunはrateプランではない)\n")
        return {}
    gains = {}
    for ax in range(3):
        amps = sorted(set(round(r[1], 3) for r in rows if r[0] == ax))
        per_amp = []
        for amp in amps:
            g = [r[3] for r in rows if r[0] == ax and round(r[1], 3) == amp]
            per_amp.append((amp, float(np.median(g)), len(g)))
        all_g = [r[3] for r in rows if r[0] == ax]
        gains[ax] = float(np.median(all_g))
        detail = "  ".join(f"|{amp:.2f}|→{g:+.2f}(n={n})" for amp, g, n in per_amp)
        spread = (max(g for _, g, _ in per_amp) - min(g for _, g, _ in per_amp)) if per_amp else 0
        note = "線形" if abs(spread) < 0.15 * abs(gains[ax]) else f"非線形/飽和の疑い(振幅間ばらつき{spread:+.2f})"
        print(f"  {AXES[ax]:5s}: median={gains[ax]:+.2f}  {detail}  [{note}]")
    print()

    print("== 2. レートステップ応答(遅延・t63) ==")
    lat, t63s, shape = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}, []
    for k in range(len(segs) - 1):
        (a0, a1, ca), (b0, b1, cb) = segs[k], segs[k + 1]
        d = cb[:3] - ca[:3]
        nz = np.nonzero(np.abs(d) > 0.04)[0]
        if len(nz) != 1 or b1 - b0 < 0.3:
            continue
        ax = int(nz[0])
        wa = _win(imu, *steady_window(a0, a1))
        wb = _win(imu, *steady_window(b0, b1))
        if wa.sum() < 3 or wb.sum() < 3:
            continue
        y0 = np.median(-imu["gyro"][wa, ax])
        y1 = np.median(-imu["gyro"][wb, ax])
        if abs(y1 - y0) < 0.1:
            continue
        m = (imu["t"] >= b0 - 0.05) & (imu["t"] < b0 + 0.4)
        p = (-imu["gyro"][m, ax] - y0) / (y1 - y0)
        dt = imu["t"][m] - b0
        shape += list(zip(dt, p))
        after = dt >= 0
        hit = np.nonzero(after & (p >= 0.3))[0]
        if len(hit):
            lat[ax].append(dt[hit[0]])
        hit = np.nonzero(after & (p >= 0.632))[0]
        if len(hit):
            t63s[ax].append(dt[hit[0]])
    all_lat = sum(lat.values(), [])
    all_t63 = sum(t63s.values(), [])
    if all_t63:
        dt_s = 1.0 / imu["rate_hz"] if imu["rate_hz"] else 0.025
        for ax in range(3):
            if t63s[ax]:
                print(f"  {AXES[ax]:5s}: 遅延(p>0.3) median={np.median(lat[ax])*1e3:5.0f}ms  "
                      f"t63 median={np.median(t63s[ax])*1e3:5.0f}ms  (n={len(t63s[ax])})")
        t63 = float(np.median(all_t63))
        print(f"  全軸: 遅延 {np.median(all_lat)*1e3:.0f}ms / t63 {t63*1e3:.0f}ms "
              f"(IMUサンプル間隔 {dt_s*1e3:.0f}ms → t63がこれ以下なら分解能限界=上限値)")
        # 平均ステップ形状(25msビン)
        sh = np.array(shape)
        print("  正規化ステップ形状(25msビン平均):")
        for lo in np.arange(-0.05, 0.30, 0.025):
            m = (sh[:, 0] >= lo) & (sh[:, 0] < lo + 0.025)
            if m.sum():
                v = sh[m, 1].mean()
                bar = "#" * int(np.clip(v, 0, 1.5) * 40)
                print(f"    {lo*1e3:+4.0f}ms {v:+5.2f} {bar}")
        print()
        return {"gains": gains, "t63": t63, "latency": float(np.median(all_lat))}
    print("  有効なステップエッジなし\n")
    return {"gains": gains} if rows else {}


# ---------------------------------------------------------------- 3) 推力

def analyze_thrust(segs, imu, drag_c: float):
    print("== 3. 推力比力曲線 A(thrust)(rates=0の区間) ==")
    cand = [(t0, t1, c) for (t0, t1, c) in segs
            if t1 - t0 >= 0.25 and np.all(np.abs(c[:3]) < 1e-9)]
    if len(cand) < 4:
        print("  推力階段区間が不足(このrunはthrustプランではない)\n")
        return {}
    # v_up推定: a_up = -f_z - g(水平仮定。比力にはドラッグも含まれるのでこれで完結、
    # ドラッグ減衰項を追加すると二重計上になる)。実IMUのバイアスによる長時間ドリフトは
    # 全飛行平均のバイアス除去 + 緩いリーク(τ=10s ≫ 推力セグメント0.5s)で抑える。
    t, fz = imu["t"], imu["accel"][:, 2]
    v_up = np.zeros_like(t)
    dt = np.r_[0.0, np.diff(t)]
    ok = dt < 0.2
    bias = float(np.mean(-fz[ok] - G))    # 飛行全体で正味v_z≈0の仮定
    v = 0.0
    for i in range(len(t)):
        if dt[i] >= 0.2:      # 受信ギャップ(リセット等)→ 積分をやり直す
            v = 0.0
        else:
            v += (-fz[i] - G - bias) * dt[i]
            v *= max(0.0, 1.0 - 0.1 * dt[i])
        v_up[i] = v

    by_level: dict[float, list[tuple[float, float]]] = {}
    for (t0, t1, c) in cand:
        thr = round(float(c[3]), 4)
        m = _win(imu, *steady_window(t0, t1))
        if m.sum() < 3:
            continue
        A_raw = float(np.median(-imu["accel"][m, 2]))
        A_cor = float(np.median(-imu["accel"][m, 2] + drag_c * v_up[m]))
        by_level.setdefault(thr, []).append((A_raw, A_cor))
    levels = sorted(by_level)
    xs, ys = [], []
    print("  thrust   A_raw   A_corr   A_model(2乗則,h=0.2742)   n")
    for thr in levels:
        arr = np.array(by_level[thr])
        A_raw, A_cor = arr[:, 0].mean(), arr[:, 1].mean()
        A_model = G * (thr / 0.2742) ** 2
        print(f"  {thr:6.3f}  {A_raw:6.2f}  {A_cor:6.2f}   {A_model:6.2f}"
              f"                    {len(arr)}")
        if A_cor > 0.5:
            xs.append(np.log(thr))
            ys.append(np.log(A_cor))
    if len(xs) < 3:
        print("  フィット不能(有効レベル不足)\n")
        return {}
    xs, ys = np.array(xs), np.array(ys)

    def fit(m):
        alpha, b = np.polyfit(xs[m], ys[m], 1)
        hover = float(np.exp((np.log(G) - b) / alpha))
        resid = float(np.sqrt(np.mean((np.polyval([alpha, b], xs[m]) - ys[m]) ** 2)))
        return float(alpha), hover, resid

    alpha, hover, resid = fit(np.ones_like(xs, bool))
    print(f"  フィット(全域):     A = g*(thrust/{hover:.4f})^{alpha:.2f}  (log残差rms={resid:.3f})")
    in_range = xs >= np.log(0.24)   # 方策レンジ[0.265,0.40]近傍(下端は少し余裕)
    if in_range.sum() >= 3:
        alpha, hover, resid = fit(in_range)
        print(f"  フィット(≥0.24):    A = g*(thrust/{hover:.4f})^{alpha:.2f}  (log残差rms={resid:.3f})"
              f"  ← 方策レンジではこちらを採用")
    print(f"  現行想定:           A = g*(thrust/0.2742)^2.00\n")
    return {"alpha": alpha, "hover": hover}


# ---------------------------------------------------------------- 4) ドラッグ

def analyze_drag(segs, imu):
    print("== 4. 線形ドラッグ c(水平惰性: 体x/y比力の指数減衰) ==")
    # 惰性窓 = rates全0・thrust≈hover・3s以上(dragプランの水平コースト)
    coast = [(t0, t1, c) for (t0, t1, c) in segs
             if t1 - t0 >= 2.5 and np.all(np.abs(c[:3]) < 1e-9)
             and abs(c[3] - 0.2742) < 0.005]
    cs = []
    for (t0, t1, c) in coast:
        m = _win(imu, t0 + 0.15, t1 - 0.05)
        if m.sum() < 20:
            continue
        fh = np.sqrt(imu["accel"][m, 0] ** 2 + imu["accel"][m, 1] ** 2)
        tt = imu["t"][m] - t0
        if len(fh) < 20 or fh[0] < 0.8:
            continue   # 初速が小さすぎて減衰が測れない
        head = np.median(fh[tt < 1.0])
        tail = np.median(fh[tt > tt[-1] - 1.0])
        if head < 1.5 * tail:
            continue   # 減衰していない窓(姿勢が水平化に失敗している等)は捨てる
        # 残留傾きがあると f_h は0でなく定数へ収束する → オフセット付き指数でフィット:
        # f_h(t) = b + a*e^{-ct}(cをグリッド、a,bは線形最小二乗。a>0=減衰、bは小さい正)
        best = None
        for c_try in np.arange(0.10, 1.51, 0.01):
            X = np.vstack([np.exp(-c_try * tt), np.ones_like(tt)]).T
            coef, res, *_ = np.linalg.lstsq(X, fh, rcond=None)
            r = float(res[0]) if len(res) else float(np.sum((X @ coef - fh) ** 2))
            if coef[0] > 0 and coef[1] > -0.1 and (best is None or r < best[0]):
                best = (r, c_try, coef[0], coef[1])
        if best is None:
            continue
        _, c_fit, a_fit, b_fit = best
        cs.append(c_fit)
        print(f"  coast t={t0:6.1f}s dur={t1-t0:.1f}s: f_h {fh[0]:.2f}→{fh[-1]:.2f} m/s²"
              f"  c={c_fit:.2f} (offset={b_fit:+.2f})")
    # 前傾定常(tilt-hold)の一致チェック: f_x → -g·tanθ(θ=17.8°なら-3.15)
    tilt = [(t0, t1, c) for (t0, t1, c) in segs
            if t1 - t0 >= 2.5 and np.all(np.abs(c[:3]) < 1e-9)
            and abs(c[3] - 0.281) < 0.004]
    for (t0, t1, c) in tilt:
        m = _win(imu, t1 - 1.5, t1 - 0.05)
        if m.sum() >= 5:
            fx = float(np.median(imu["accel"][m, 0]))
            print(f"  tilt-hold t={t0:6.1f}s: 終端 f_x={fx:+.2f} m/s² "
                  f"(前傾17.8°の予測 -g·tanθ = -3.15; ずれ=実傾き/姿勢保持の差)")
    if not cs:
        print("  惰性減衰窓なし(このrunはdragプランではない)\n")
        return {}
    c_med = float(np.median(cs))
    print(f"  → c = {c_med:.2f}(現行想定 0.72)\n")
    return {"drag_c": c_med}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, nargs="+",
                    help="record-dir(複数可: rate/thrust/dragの各runをまとめて解析)")
    ap.add_argument("--drag-c", type=float, default=0.72,
                    help="推力補正に使うドラッグ係数(dragプラン解析後に判れば差し替え)")
    args = ap.parse_args()

    res = {}
    for d in args.dir:
        steps, imu_raw, meta = load_run(d)
        if not steps:
            print(f"{d}: no records")
            continue
        imu = dedup_imu(imu_raw)
        collisions = load_collisions(d)
        segs, n_dropped = build_segments(steps, collisions)
        print(f"########## {d}  (plan={meta.get('sysid_plan')}, "
              f"steps={len(steps)}, imu {imu['n_raw']}→有効{len(imu['t'])} "
              f"@{imu['rate_hz']:.0f}Hz, 衝突={len(collisions)}, 除外区間={n_dropped})")
        res.update({k: v for k, v in analyze_rates(segs, imu).items() if v})
        res.update(analyze_drag(segs, imu))            # dragを先に→thrust補正に反映
        c_for_thrust = res.get("drag_c", args.drag_c)
        res.update(analyze_thrust(segs, imu, c_for_thrust))

    print("########## 5. Genesis較正の推奨値")
    if "gains" in res:
        g = res["gains"]
        print(f"  dcl/client.py RATE_CMD_GAIN = ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})")
        print(f"  config.py    drone.cmd_gain = ({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})"
              f"   # Genesis内で指令に乗算=実シム模擬(再学習用)")
    if "t63" in res:
        k = 1.0 / max(res["t63"], 1e-3)
        print(f"  config.py    drone.k_rate   = {k:.0f}   # τ63={res['t63']*1e3:.0f}ms"
              f"(IMU分解能以下なら下限値=これ以上速い)")
    if "hover" in res:
        print(f"  config.py    drone.hover_thrust = {res['hover']:.4f}"
              f"  (α={res['alpha']:.2f}; 2から大きく外れるならDroneModelの指数も変更)")
    if "drag_c" in res:
        print(f"  config.py    drone.drag_c   = {res['drag_c']:.2f}")
    if not res:
        print("  (同定結果なし)")


if __name__ == "__main__":
    main()
