# -*- coding: utf-8 -*-
"""VQ1(レガシー=テレメトリ有効)実シム飛行の *真値* 同定。自作シムの動力学較正用。

  python -m genesis_rl.scripts.analyze_truth --dir runs/vq1_fit_0727 [runs/... ]

入力(fly_dcl --sysid --record-dir DIR で取得):
  DIR/steps.jsonl  30Hz決定ごとの指令 cmd[4]=(roll,pitch,yaw[rad/s], thrust)
  DIR/truth.jsonl  ATT/POS/ODOM/ACT の真値テレメトリ(VQ1のみ。無ければVQ2ビルド)
  DIR/imu.jsonl    HIGHRES_IMU 生値(符号監査とVQ2との橋渡しに使う)
  DIR/events.jsonl 衝突時刻

analyze_sysid.py(IMU単独)との違い — 真値があるので推定の連鎖が消える:
  * 速度を加速度の積分で作らない → ドラッグ・推力の分離が一発で決まる
  * 「きれいに水平化できた窓」を待たない → 全サンプルを1つの回帰に使える
  * 指令→物理回転の符号と遅延を推測でなく実測で確定できる
  * ドラッグ構造(線形/2次、等方/異方)、推力の速度依存をデータで選べる

物理規約(自作シム drone.py と同一):
  NED world(z下向き)、body FRD、g_ned=(0,0,+9.81)
  a_ned = g_ned + R·(0,0,-A) + D_ned          A=推力比力[m/s²]、D=ドラッグ
  比力(=加速度計が測る量) f_body = Rᵀ(a_ned - g_ned) = (0,0,-A) + Rᵀ D_ned
  線形worldドラッグ D_ned = -c·v_ned なら Rᵀ D_ned = -c·v_body なので
    f_x = -c·v_body_x,  f_y = -c·v_body_y,  f_z = -A - c·v_body_z
  → 真値の (R, v, a) から c と A を直接最小二乗できる。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

G = 9.81
AXES = ("roll", "pitch", "yaw")
RESAMPLE_HZ = 100.0

# ODOMETRY のクォータニオンは「標準の body(FRD)→NED」ではなく y軸(East/右)を鏡像にした
# 左手系で入ってくる。(w,x,y,z) → (w,-x,y,-z) に直すと、真値の位置・速度から求めた比力
# Rᵀ(a-g) が HIGHRES_IMU の accel と 0.01 m/s² で一致する(2026-07-27 実測、下記)。
#
#   軸   補正なしの一致(比/差rms)     補正後
#   x     +0.998 / 0.036 m/s²        +0.999 / 0.012
#   y     -0.140 / 1.541             +0.998 / 0.008   ← 補正なしでは無相関
#   z     +1.007 / 0.429             +1.001 / 0.204
#
# これが memory の「左手系」の正体。--no-quat-fix で無効化して比較できる。
ODOM_QUAT_YMIRROR = True


def fix_odom_quat(q: np.ndarray) -> np.ndarray:
    """DCLのODOMETRYクォータニオン → 標準 body(FRD)→NED クォータニオン。"""
    q = q * np.array([1.0, -1.0, 1.0, -1.0])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ---------------------------------------------------------------- 読み込み

def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _fit_time(t_sim: np.ndarray, t_wall: np.ndarray) -> np.ndarray:
    """シム時計を時間軸にし、リセットで巻き戻る区間ごとに壁時計へオフセットして貼る。

    ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / HIGHRES_IMU は *同じシム時計* の
    タイムスタンプを持つので、これを軸にすると型どうしが厳密に整列する。
    受信時刻(t_rx_wall)だけで整列すると型ごとの転送遅延差(数ms〜)がそのまま
    「姿勢と速度の時間ずれ」になり、高角速度・高速域で運動学の突き合わせが崩れる。
    区間ごとのオフセットは median(t_wall - t_sim) = その型の平均転送遅延なので、
    型間の遅延差はここで吸収される。
    """
    if len(t_sim) < 20:
        return t_wall
    t = np.empty_like(t_sim, dtype=float)
    starts = np.r_[0, np.nonzero(np.diff(t_sim) < -0.5)[0] + 1, len(t_sim)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 3:
            t[a:b] = t_wall[a:b]
            continue
        t[a:b] = t_sim[a:b] + float(np.median(t_wall[a:b] - t_sim[a:b]))
    return t


def load_truth(dir_: str) -> dict:
    """truth.jsonl を kind別の {t, v[…]} へ。時刻はジッタ除去済みの壁時計。"""
    recs = _load_jsonl(os.path.join(dir_, "truth.jsonl"))
    if not recs:
        return {}
    out = {}
    for kind in ("ATT", "POS", "ODOM", "ACT"):
        sel = [r for r in recs if r["kind"] == kind]
        if not sel:
            continue
        tw = np.array([r["t_rx_wall"] for r in sel], float)
        ts = np.array([r["t_sim"] for r in sel], float)
        v = np.array([r["v"] for r in sel], float)
        # 同一値の連続(メッセージレート>実効更新レート)を落とす
        keep = np.r_[True, np.any(np.diff(v, axis=0) != 0, axis=1)]
        out[kind] = {"t": _fit_time(ts[keep], tw[keep]), "v": v[keep],
                     "hz": float((keep.sum() - 1) / max(tw[keep][-1] - tw[keep][0], 1e-9)),
                     "n_raw": len(sel)}
    return out


def load_cmd(dir_: str):
    """steps.jsonl → (t, cmd[N,4])。飛行中(pin解除後)しか記録されない。"""
    steps = _load_jsonl(os.path.join(dir_, "steps.jsonl"))
    if not steps:
        raise SystemExit(f"{dir_}: steps.jsonl が空(飛行していない)")
    t = np.array([r["t_wall"] for r in steps], float)
    cmd = np.array([r["cmd"] for r in steps], float)
    return t, cmd


# ---------------------------------------------------------------- 整列

def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz(body→NED) → (N,3,3) 回転行列。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def euler_to_R(rpy: np.ndarray) -> np.ndarray:
    """(N,3) ZYX オイラー(roll,pitch,yaw) → body→NED 回転行列。"""
    r, p, y = rpy[:, 0], rpy[:, 1], rpy[:, 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(rpy), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def R_to_quat(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,4) wxyz(w>0に正規化)。"""
    w = np.sqrt(np.clip(1.0 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], 1e-12, None)) / 2.0
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    return np.stack([w, x, y, z], axis=1)


def _interp(t_new: np.ndarray, t: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(v.T).T
    return np.stack([np.interp(t_new, t, v[:, i]) for i in range(v.shape[1])], axis=1)


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """移動平均(端は縮める)。微分前の高周波(通信ジッタ)除去用。"""
    if n <= 1:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, i], k, mode="same")[pad:pad + len(x)]
                     for i in range(x.shape[1])], axis=1)


def valid_mask(t: np.ndarray, flights: list, collisions: list) -> np.ndarray:
    """飛行区間内かつ衝突の前後を除いたサンプルの真偽マスク。"""
    m = np.zeros(len(t), bool)
    for (a, b) in flights:
        m |= (t >= a + 0.25) & (t <= b - 0.10)
    for tc in collisions:
        m &= ~((t > tc - 0.30) & (t < tc + 2.0))
    return m


def build_grid(dir_: str, smooth_n: int = 5, rate_clip: float = 1e9) -> dict:
    """指令・真値・IMUを共通の等間隔グリッドへ載せ、飛行中マスクを作る。"""
    meta = {}
    if os.path.exists(os.path.join(dir_, "meta.json")):
        with open(os.path.join(dir_, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    truth = load_truth(dir_)
    if not truth:
        raise SystemExit(
            f"{dir_}: truth.jsonl が無い/空 → テレメトリ無効ビルド(VQ2)で録ったデータ。\n"
            "  VQ1レガシーシムを起動して録り直すか、IMU単独の analyze_sysid.py を使ってください。")
    if "POS" not in truth or "ATT" not in truth:
        raise SystemExit(f"{dir_}: ATT/POS が足りない(受信: {sorted(truth)})")

    t_cmd, cmd = load_cmd(dir_)
    # 飛行区間: 30Hz決定が続いている範囲(ギャップ>0.2sで分断=リセット/待機)
    breaks = np.nonzero(np.diff(t_cmd) > 0.2)[0]
    seg_bounds = np.split(np.arange(len(t_cmd)), breaks + 1)
    flights = [(t_cmd[s[0]], t_cmd[s[-1]] + 1.0 / 30.0) for s in seg_bounds if len(s) > 30]
    if not flights:
        raise SystemExit(f"{dir_}: 連続飛行区間なし(全部1秒未満)")

    t0 = max(min(truth["POS"]["t"][0], truth["ATT"]["t"][0]), t_cmd[0])
    t1 = min(max(truth["POS"]["t"][-1], truth["ATT"]["t"][-1]), t_cmd[-1])
    t = np.arange(t0, t1, 1.0 / RESAMPLE_HZ)

    att = _interp(t, truth["ATT"]["t"], truth["ATT"]["v"])       # rpy + 角速度
    pos = _interp(t, truth["POS"]["t"], truth["POS"]["v"])       # xyz + v
    rpy, omega_att = att[:, 0:3], att[:, 3:6]
    p_ned, v_ned = pos[:, 0:3], pos[:, 3:6]
    if "ODOM" in truth:
        q_ned = _interp(t, truth["ODOM"]["t"], truth["ODOM"]["v"])[:, 3:7]
        q_ned = (fix_odom_quat(q_ned) if ODOM_QUAT_YMIRROR
                 else q_ned / np.linalg.norm(q_ned, axis=1, keepdims=True))
    else:
        q_ned = R_to_quat(euler_to_R(rpy))
    R = quat_to_R(q_ned)

    # 指令はゼロ次ホールド(30Hz決定の間は一定)
    idx = np.clip(np.searchsorted(t_cmd, t, side="right") - 1, 0, len(t_cmd) - 1)
    cmd_g = cmd[idx]

    # 速度を平滑化して微分 → 真の加速度(a_ned)
    v_s = _smooth(v_ned, smooth_n)
    a_ned = np.gradient(v_s, t, axis=0)
    om_s = _smooth(omega_att, smooth_n)
    om_dot = np.gradient(om_s, t, axis=0)

    collisions = [ev["t_wall"] for ev in _load_jsonl(os.path.join(dir_, "events.jsonl"))
                  if ev.get("type") == "collision"]
    m = valid_mask(t, flights, collisions)
    # 速度の不連続(リセット/瞬間移動/未検出の接触)を除く。微分すると数十〜150m/s²の
    # 巨大スパイクになり、最小二乗の残差を単独で支配してしまう。
    jump = np.zeros(len(t), bool)
    bad = np.nonzero(np.linalg.norm(a_ned, axis=1) > 40.0)[0]
    for i in bad:
        jump[max(i - 8, 0):i + 9] = True
    m &= ~jump
    n_drop = int(jump.sum())
    # 飽和領域を線形モデルの同定に混ぜない。指令が実シムのクリップ(約2.6rad/s)を超えると
    # 達成レートは頭打ちなので、そのまま回帰するとゲインも時定数も一律に小さく出る
    # (satプランを混ぜた4run結合で cmd_gain 0.95→0.82、k_rate 61→20 まで崩れた)。
    sat = np.any(np.abs(cmd_g[:, :3]) > rate_clip * 1.02, axis=1)
    n_sat = int((m & sat).sum())
    m &= ~sat

    imu = {}
    raw_imu = _load_jsonl(os.path.join(dir_, "imu.jsonl"))
    if raw_imu:
        ti = np.array([r["t_rx_wall"] for r in raw_imu], float)
        ts = np.array([r["t_sim"] for r in raw_imu], float)
        gy = np.array([r["gyro"] for r in raw_imu], float)
        ac = np.array([r["accel"] for r in raw_imu], float)
        keep = np.r_[True, np.any(np.diff(np.c_[gy, ac], axis=0) != 0, axis=1)]
        tt = _fit_time(ts[keep], ti[keep])
        imu = {"gyro": _interp(t, tt, gy[keep]), "accel": _interp(t, tt, ac[keep]),
               "hz": float((keep.sum() - 1) / max(ti[keep][-1] - ti[keep][0], 1e-9))}

    # レート同定は補間・平滑を通さない生サンプルで行う(τ≈30msを潰さないため)。
    # 飽和サンプルの除外はここでも必須(グリッド側のマスクは使わないため)。
    att_t = truth["ATT"]["t"]
    att_cmd = cmd_at(t_cmd, cmd, att_t)
    att = {"t": att_t, "om": truth["ATT"]["v"][:, 3:6],
           "mask": (valid_mask(att_t, flights, collisions)
                    & ~np.any(np.abs(att_cmd[:, :3]) > rate_clip * 1.02, axis=1))}

    return {"t": t, "dt": 1.0 / RESAMPLE_HZ, "cmd": cmd_g, "rpy": rpy, "omega": om_s,
            "omega_dot": om_dot, "pos": p_ned, "vel": v_ned, "accel_ned": a_ned,
            "R": R, "q": q_ned, "mask": m, "imu": imu, "truth": truth, "flights": flights,
            "t_cmd": t_cmd, "cmd_raw": cmd, "att": att, "n_drop_jump": n_drop,
            "n_drop_sat": n_sat, "rate_clip": rate_clip, "meta": meta, "dir": dir_}


def cmd_at(t_cmd: np.ndarray, cmd: np.ndarray, t: np.ndarray) -> np.ndarray:
    """30Hz決定の指令をゼロ次ホールドで任意時刻へ引く。"""
    idx = np.clip(np.searchsorted(t_cmd, t, side="right") - 1, 0, len(t_cmd) - 1)
    return cmd[idx]


# ---------------------------------------------------------------- 0) 棚卸し

def report_rates(d: dict, dir_: str):
    print(f"########## {dir_}")
    tr = d["truth"]
    parts = [f"{k}={tr[k]['hz']:.0f}Hz({tr[k]['n_raw']}件)" for k in sorted(tr)]
    if d["imu"]:
        parts.append(f"IMU={d['imu']['hz']:.0f}Hz")
    print("  テレメトリ: " + "  ".join(parts))
    print(f"  飛行区間 {len(d['flights'])}本 / 解析サンプル {int(d['mask'].sum())} "
          f"({d['mask'].sum() / RESAMPLE_HZ:.1f}s @ {RESAMPLE_HZ:.0f}Hz、"
          f"速度不連続で除外 {d.get('n_drop_jump', 0)}、"
          f"レート飽和で除外 {d.get('n_drop_sat', 0)})")
    m = d["mask"]
    if m.sum():
        sp = np.linalg.norm(d["vel"][m], axis=1)
        print(f"  速度 |v|: 中央{np.median(sp):.1f} 最大{sp.max():.1f} m/s   "
              f"高度 -z: {-d['pos'][m, 2].min():.1f}〜{-d['pos'][m, 2].max():.1f} m")
        print(f"  姿勢: roll {np.degrees(d['rpy'][m, 0]).min():+.0f}〜"
              f"{np.degrees(d['rpy'][m, 0]).max():+.0f}°  "
              f"pitch {np.degrees(d['rpy'][m, 1]).min():+.0f}〜"
              f"{np.degrees(d['rpy'][m, 1]).max():+.0f}°")


# ---------------------------------------------------------------- 1) 符号監査

def audit_signs(d: dict):
    """指令→回転、生gyro→真ω、accel→比力 の符号を実測で確定する。"""
    print("\n== 1. 符号・規約の実測(自作シム signs_* の根拠) ==")
    m = d["mask"]
    om, cmd = d["omega"], d["cmd"]

    # (a) ATTITUDE角速度 vs 指令: 軸ごとに |指令|>0.05 の *定常* 区間で回帰。
    # 指令変化から0.15s以内は過渡(τ=13〜30ms + 遅延)なので除く。含めるとゲインが
    # 一律に小さく出る(パルスが短いプランほど過小評価される)。
    changed = np.r_[True, np.any(np.diff(cmd[:, :3], axis=0) != 0, axis=1)]
    settle = np.ones(len(cmd), bool)
    for i in np.nonzero(changed)[0]:
        settle[i:i + int(0.15 * RESAMPLE_HZ)] = False
    m = m & settle
    print("  指令→達成角速度(ATTITUDE rollspeed等との比。過渡0.15s除外):")
    gains = {}
    for ax in range(3):
        sel = m & (np.abs(cmd[:, ax]) > 0.05)
        # 他軸が同時に動いている区間は除く(fitプランはダブレットで単軸のみ)
        for other in range(3):
            if other != ax:
                sel &= np.abs(cmd[:, other]) < 1e-9
        if sel.sum() < 30:
            print(f"    {AXES[ax]:5s}: データ不足({int(sel.sum())}サンプル)")
            continue
        u, w = cmd[sel, ax], om[sel, ax]
        g = float(u @ w / (u @ u))
        r = float(np.corrcoef(u, w)[0, 1])
        gains[ax] = g
        print(f"    {AXES[ax]:5s}: 達成/指令 = {g:+.3f}  (相関 {r:+.3f}, n={int(sel.sum())})"
              f"  {'← 符号反転' if g < 0 else ''}")
    if gains:
        mag = np.mean([abs(v) for v in gains.values()])
        hint = ("bit16(rad/s)解釈が有効 = 達成≈指令"
                if mag < 1.6 else "レガシー解釈(≈2.5倍ゲイン)= bit16未対応ビルド")
        print(f"    → |ゲイン|平均 {mag:.2f} → {hint}")

    # (b) 生gyro vs 真ω
    if d["imu"]:
        gy = d["imu"]["gyro"]
        print("  生HIGHRES_IMU gyro / 真ω(ATTITUDE)の比 = deploy側 gyro_obs_sign の根拠:")
        for ax in range(3):
            sel = m & (np.abs(om[:, ax]) > 0.15)
            if sel.sum() < 30:
                print(f"    {AXES[ax]:5s}: データ不足")
                continue
            a, b = om[sel, ax], gy[sel, ax]
            ratio = float(a @ b / (a @ a))
            print(f"    {AXES[ax]:5s}: {ratio:+.3f} (相関 {np.corrcoef(a, b)[0, 1]:+.3f})")

    # (c) オイラー角の微分 vs ATTITUDE角速度(ATTITUDEのpitch符号が標準かの検証)
    #     小角近似ではなく厳密に: ZYXオイラーレート → body角速度
    r, p = d["rpy"][:, 0], d["rpy"][:, 1]
    rpy_dot = np.gradient(_smooth(d["rpy"], 5), d["t"], axis=0)
    # unwrap の飛びを除去(yawが±πで折り返す)
    ok = m & (np.abs(rpy_dot).max(axis=1) < 20.0)
    p_b = np.stack([
        rpy_dot[:, 0] - np.sin(p) * rpy_dot[:, 2],
        np.cos(r) * rpy_dot[:, 1] + np.cos(p) * np.sin(r) * rpy_dot[:, 2],
        -np.sin(r) * rpy_dot[:, 1] + np.cos(p) * np.cos(r) * rpy_dot[:, 2],
    ], axis=1)
    for ax in range(3):
        sel = ok & (np.abs(d["omega"][:, ax]) > 0.15)
        if sel.sum() < 30:
            continue
        a, b = d["omega"][sel, ax], p_b[sel, ax]
        print(f"    {AXES[ax]:5s}: dEuler由来ω / ATTITUDEω = {a @ b / (a @ a):+.3f}"
              f" (1.0なら標準ZYX・同符号)")


# ---------------------------------------------------------------- 2) レート追従

def report_rate_linearity(d: dict):
    """指令振幅ごとの達成/指令ゲイン = 飽和の有無(方策の指令レンジ設計に直結)。"""
    print("\n== 1c. レート指令の振幅依存(飽和の確認。action.rate_limits の妥当性) ==")
    m, om, cmd = d["mask"], d["omega"], d["cmd"]
    changed = np.r_[True, np.any(np.diff(cmd[:, :3], axis=0) != 0, axis=1)]
    settle = np.ones(len(cmd), bool)
    for i in np.nonzero(changed)[0]:
        settle[i:i + int(0.12 * RESAMPLE_HZ)] = False
    m = m & settle
    amps = sorted({round(float(a), 2) for a in np.abs(cmd[m][:, :3]).reshape(-1)
                   if a > 0.02})
    if not amps:
        print("  レート指令のある区間なし")
        return
    for ax in range(3):
        others = [o for o in range(3) if o != ax]
        single = ((np.abs(cmd[:, others[0]]) < 1e-9) & (np.abs(cmd[:, others[1]]) < 1e-9))
        parts, gains, achieved = [], [], []
        for a in amps:
            sel = m & single & (np.abs(np.abs(cmd[:, ax]) - a) < 1e-6)
            if sel.sum() < 15:
                continue
            u, w = cmd[sel, ax], om[sel, ax]
            g = abs(u @ w / (u @ u))
            parts.append(f"|{a:.2f}|→{g:.2f}")
            gains.append((a, g))
            achieved.append(a * g)
        if not parts:
            print(f"  {AXES[ax]:5s}: データなし")
            continue
        line = f"  {AXES[ax]:5s}: " + "  ".join(parts)
        if len(gains) >= 3 and gains[-1][1] < 0.8 * gains[0][1]:
            g0 = gains[0][1]                       # 小振幅=飽和していない領域のゲイン
            sat = max(achieved)                    # 達成レートの上限
            line += (f"   → 飽和: 達成上限 ≈{sat:.2f} rad/s "
                     f"(指令換算 {sat / max(g0, 1e-6):.2f} rad/s)")
        print(line)
    print("  (小振幅のゲインは維持されるが大振幅で落ちるのは達成レートのクリップ。"
          "学習側 action.rate_limits と dynamics.rate_max をこの上限に合わせる)")


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def omega_from_quat(t_o: np.ndarray, q: np.ndarray):
    """(N,)時刻と(N,4)クォータニオン(body→NED, wxyz)から body角速度(標準FRD)を復元。

    戻り値 (t_mid, omega, valid)。クォータニオンは定義そのものなので、オイラー角の
    符号規約に依存しない絶対基準になる。
    """
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    dt = np.diff(t_o)
    qd = (q[1:] - q[:-1]) / np.maximum(dt, 1e-9)[:, None]
    qc = q[:-1] * np.array([1.0, -1.0, -1.0, -1.0])
    om = 2.0 * _quat_mul(qc, qd)[:, 1:]
    return 0.5 * (t_o[:-1] + t_o[1:]), om, (dt > 0.002) & (dt < 0.10)


def om_sign_from_quat(d: dict) -> tuple:
    """ATTITUDE角速度 → 標準FRD角速度 の軸別符号(±1)を実測で決める。"""
    tr = d["truth"]
    if "ODOM" not in tr:
        return (1.0, 1.0, 1.0)
    q_odom = tr["ODOM"]["v"][:, 3:7]
    tm, om_q, ok = omega_from_quat(tr["ODOM"]["t"],
                                   fix_odom_quat(q_odom) if ODOM_QUAT_YMIRROR else q_odom)
    m = valid_mask(tm, d["flights"], []) & ok
    om_att = _interp(tm, d["att"]["t"], d["att"]["om"])
    out = []
    for ax in range(3):
        sel = m & (np.abs(om_q[:, ax]) > 0.15)
        if sel.sum() < 30:
            out.append(1.0)
            continue
        a, b = om_att[sel, ax], om_q[sel, ax]
        out.append(float(np.sign(a @ b)))
    return tuple(out)


def audit_body_rates(d: dict):
    """ODOMETRYクォータニオンの時間変化から body角速度を復元し、報告値と突き合わせる。

    クォータニオンは「body→NEDの回転」という定義そのものなので、オイラー角の
    符号規約に依存しない唯一の基準になる。自作シムは内部を標準FRD/NEDで持つので、
    ここで得た比が signs_gyro / signs_cmd の確定値になる。
    """
    tr = d["truth"]
    if "ODOM" not in tr:
        print("  (ODOMETRYなし → クォータニオン基準の検証は不可)")
        return {}
    print("\n== 1b. クォータニオン基準の角速度(規約に依存しない絶対基準) ==")
    q_odom = tr["ODOM"]["v"][:, 3:7]
    tm, om_q, ok = omega_from_quat(tr["ODOM"]["t"],
                                   fix_odom_quat(q_odom) if ODOM_QUAT_YMIRROR else q_odom)
    cmd = cmd_at(d["t_cmd"], d["cmd_raw"], tm)
    clip = d.get("rate_clip", 1e9)
    m = (valid_mask(tm, d["flights"], []) & ok
         & ~np.any(np.abs(cmd[:, :3]) > clip * 1.02, axis=1))
    om_att = _interp(tm, d["att"]["t"], d["att"]["om"])
    print("  ATTITUDE角速度 / クォータニオン由来ω(=1.0なら標準FRDと同符号・同定義):")
    signs = []
    for ax in range(3):
        sel = m & (np.abs(om_q[:, ax]) > 0.15)
        if sel.sum() < 30:
            print(f"    {AXES[ax]:5s}: データ不足({int(sel.sum())})")
            signs.append(np.nan)
            continue
        a, b = om_q[sel, ax], om_att[sel, ax]
        r = float(a @ b / (a @ a))
        signs.append(r)
        print(f"    {AXES[ax]:5s}: {r:+.3f}  (相関 {np.corrcoef(a, b)[0, 1]:+.3f}, "
              f"n={int(sel.sum())})")
    print("  指令 / クォータニオン由来ω(=自作シム signs_cmd × cmd_gain の確定値):")
    cmd_sign = []
    for ax in range(3):
        others = [o for o in range(3) if o != ax]
        sel = (m & (np.abs(cmd[:, ax]) > 0.05)
               & (np.abs(cmd[:, others[0]]) < 1e-9) & (np.abs(cmd[:, others[1]]) < 1e-9))
        if sel.sum() < 30:
            print(f"    {AXES[ax]:5s}: データ不足({int(sel.sum())})")
            cmd_sign.append(np.nan)
            continue
        u, w = cmd[sel, ax], om_q[sel, ax]
        g = float(u @ w / (u @ u))
        cmd_sign.append(g)
        print(f"    {AXES[ax]:5s}: 達成ω/指令 = {g:+.3f}"
              f"  → signs_cmd[{ax}]={np.sign(g):+.0f}, |gain|={abs(g):.3f}")
    return {"att_over_quat": signs, "cmd_over_quat": cmd_sign}


def fit_rate_loop(d: dict):
    """ω[n+1] = φω[n] + (1-φ)·g·u(t-遅延)、φ=e^{-k·Δt} を軸ごとに同定。

    微分を取らない1ステップ先予測なので、τ≈30ms級の速い応答でもテレメトリ
    (≈90Hz)から素直に出る。k と 遅延 をグリッド、ゲイン g は各点で線形最小二乗。
    立ち上がり(指令ON)と減衰(指令OFF後)の両方を使うため k が一意に決まる。
    """
    print("\n== 2. レート追従(自作シム drone.py の k_rate / cmd_gain) ==")
    t, om, m = d["att"]["t"], d["att"]["om"], d["att"]["mask"]
    dt = np.diff(t)
    out = {}
    k_grid = np.exp(np.linspace(np.log(3.0), np.log(200.0), 90))
    lags = np.arange(0.0, 0.131, 0.005)
    for ax in range(3):
        # 単軸区間: 他軸の指令が0。指令OFFの減衰も含めるため this軸の値は問わない
        others = [o for o in range(3) if o != ax]
        active = np.zeros(len(t), bool)
        for lag in (0.0,):     # マスク作成は遅延0で判定(区間端は下でトリム)
            c = cmd_at(d["t_cmd"], d["cmd_raw"], t - lag)
            active |= (np.abs(c[:, others[0]]) < 1e-9) & (np.abs(c[:, others[1]]) < 1e-9)
        sel = m & active
        # 1ステップ先予測に使えるのは「次サンプルも有効かつΔtが正常」なペア
        pair = sel[:-1] & sel[1:] & (dt > 0.002) & (dt < 0.05)
        if pair.sum() < 200:
            print(f"  {AXES[ax]:5s}: データ不足({int(pair.sum())}ペア)")
            continue
        y = om[1:, ax][pair]
        w0 = om[:-1, ax][pair]
        dts = dt[pair]
        best = None
        for lag in lags:
            u = cmd_at(d["t_cmd"], d["cmd_raw"], t[:-1] - lag)[pair, ax]
            for k in k_grid:
                phi = np.exp(-k * dts)
                x = (1.0 - phi) * u
                # y - φw0 = g·x を g について解く
                r = y - phi * w0
                g = float(x @ r / max(x @ x, 1e-12))
                resid = float(np.mean((r - g * x) ** 2))
                if best is None or resid < best[0]:
                    best = (resid, k, g, lag)
        resid, k, g, lag = best
        var = float(np.var(y - w0))
        out[ax] = {"k": float(k), "gain": float(g), "lag_s": float(lag)}
        print(f"  {AXES[ax]:5s}: k_rate={k:6.1f} 1/s (τ={1000 / k:5.1f}ms)  "
              f"cmd_gain={g:+.3f}  遅延={lag * 1000:3.0f}ms  "
              f"1step残差={np.sqrt(resid) * 1000:5.1f}mrad/s "
              f"(説明率{1 - resid / max(var, 1e-12):.3f})  n={int(pair.sum())}")
    return out


# ---------------------------------------------------------------- 3) 並進(ドラッグ・推力)

def fit_translation(d: dict):
    """比力 f_body = Rᵀ(a_ned - g) から ドラッグと推力曲線 A(thrust) を同時同定。

    水平(body x,y)成分には推力が入らないので、そこから先にドラッグ構造を決め、
    鉛直成分は α をグリッドして (β=g/h^α, 鉛直ドラッグ) を線形最小二乗で解く。
    定常窓の中央値ではなく全サンプルを使うため、階段の水準数に縛られない。
    """
    print("\n== 3. 並進: ドラッグと推力比力(自作シム drag_c / hover_thrust / thrust_alpha) ==")
    m = d["mask"]
    g_ned = np.array([0.0, 0.0, G])
    Rt = np.transpose(d["R"], (0, 2, 1))
    f_body = np.einsum("nij,nj->ni", Rt, d["accel_ned"] - g_ned)   # 比力(=加速度計相当)
    v_body = np.einsum("nij,nj->ni", Rt, d["vel"])
    speed = np.linalg.norm(d["vel"], axis=1)
    thr = d["cmd"][:, 3]

    # --- 3成分すべてを1つの最小二乗にかける(ベクトル式のまま解く):
    #       a_ned - g = R·(0,0,-A(thrust)) - (c1 + c2|v|)·v
    #     αをグリッドし、A=β·thrust^α とすれば (β, c1, c2) について線形。
    #     ボディ軸で水平/鉛直に分けて解くと、傾いた高速巡航で抗力の向きを取り違えて
    #     推力へ吸わせてしまう(このデータでは高速域のAが2.4m/s²も低く出た)。
    s = m & (thr > 0.05)
    if s.sum() < 200:
        print("  有効サンプル不足")
        return {}
    y3 = (d["accel_ned"][s] - g_ned).reshape(-1)                  # (3N,)
    e_up = np.einsum("nij,j->ni", d["R"][s], np.array([0.0, 0.0, -1.0]))
    v3 = d["vel"][s]
    sp = speed[s]
    t_ = thr[s]
    n_s = int(s.sum())

    def _ls(X, yv):
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        return coef, float(np.sqrt(np.mean((X @ coef - yv) ** 2)))

    def _fit_iso(alpha, use_c1=True, use_c2=True):
        cols = [(e_up * (t_ ** alpha)[:, None]).reshape(-1)]
        if use_c1:
            cols.append((-v3).reshape(-1))
        if use_c2:
            cols.append((-sp[:, None] * v3).reshape(-1))
        return _ls(np.stack(cols, axis=1), y3)

    print(f"  ベクトル同時フィット [n={n_s} サンプル(3成分) |v| 0〜{sp.max():.1f}m/s]:")
    results = {}
    for name, (u1, u2) in (("線形のみ", (True, False)), ("2次のみ", (False, True)),
                           ("線形+2次", (True, True))):
        best = None
        for alpha in np.arange(0.8, 3.51, 0.02):
            coef, r = _fit_iso(alpha, u1, u2)
            if coef[0] > 0 and (best is None or r < best[0]):
                best = (r, alpha, coef)
        r, alpha, coef = best
        hov = float((G / coef[0]) ** (1.0 / alpha))
        c1 = float(coef[1]) if u1 else 0.0
        c2 = float(coef[-1]) if u2 else 0.0
        results[name] = (r, alpha, hov, c1, c2)
        print(f"    {name:8s} hover={hov:.4f} α={alpha:.2f} c1={c1:7.3f} c2={c2:7.4f}"
              f"   残差rms {r:.3f} m/s²")
    struct_name = min(results, key=lambda k: results[k][0])
    r_best, alpha, hover, c1, c2 = results[struct_name]
    struct = {"線形のみ": "linear", "2次のみ": "quad", "線形+2次": "both"}[struct_name]
    print(f"    → 最良: {struct_name}"
          f"{'  (現行drone.pyは線形のみ → 2次項の追加が要る)' if struct != 'linear' else ''}")

    # 推力の関数形(べき乗則 vs 多項式)を比べる。ドラッグは上で選んだ構造に固定。
    drag_cols = []
    if abs(c1) > 1e-6:
        drag_cols.append((-v3).reshape(-1))
    if abs(c2) > 1e-6:
        drag_cols.append((-sp[:, None] * v3).reshape(-1))
    forms = {}
    X = np.stack([(e_up * (t_ ** alpha)[:, None]).reshape(-1)] + drag_cols, axis=1)
    forms[f"べき乗 t^{alpha:.2f}"] = (_ls(X, y3)[1], None)
    for nm, cols in (("1次 a1·t", [t_]),
                     ("2次 a1·t+a2·t²", [t_, t_ ** 2]),
                     ("2次+定数", [np.ones_like(t_), t_, t_ ** 2])):
        X = np.stack([(e_up * c[:, None]).reshape(-1) for c in cols] + drag_cols, axis=1)
        coef, r = _ls(X, y3)
        forms[nm] = (r, coef[:len(cols)])
    print("  推力の関数形(ドラッグ構造は上の最良に固定):")
    for nm, (r, coef) in forms.items():
        extra = ("" if coef is None else
                 "  係数 " + " ".join(f"{c:+.3f}" for c in coef))
        print(f"    {nm:16s} 残差rms {r:.3f} m/s²{extra}")
    if min(forms.values(), key=lambda x: x[0])[0] < 0.9 * forms[f"べき乗 t^{alpha:.2f}"][0]:
        print("    → べき乗則より多項式の方が明確に良い(drone.py の推力モデル変更を検討)")
    else:
        print("    → べき乗則で十分(現行 drone.py の形のまま係数だけ更新すればよい)")

    # ボディ軸別ドラッグ(水平 xy と 鉛直 z で係数を分ける)を試す = 異方性の検定
    Rt_s = np.transpose(d["R"][s], (0, 2, 1))
    vb = np.einsum("nij,nj->ni", Rt_s, v3)
    vb_h = np.stack([vb[:, 0], vb[:, 1], np.zeros_like(vb[:, 2])], axis=1)
    vb_z = np.stack([np.zeros_like(vb[:, 0]), np.zeros_like(vb[:, 1]), vb[:, 2]], axis=1)
    Dh = np.einsum("nij,nj->ni", d["R"][s], vb_h)
    Dz = np.einsum("nij,nj->ni", d["R"][s], vb_z)
    best = None
    for al in np.arange(0.8, 3.51, 0.02):
        X = np.stack([(e_up * (t_ ** al)[:, None]).reshape(-1),
                      (-sp[:, None] * Dh).reshape(-1), (-Dh).reshape(-1),
                      (-sp[:, None] * Dz).reshape(-1), (-Dz).reshape(-1)], axis=1)
        coef, r = _ls(X, y3)
        if coef[0] > 0 and (best is None or r < best[0]):
            best = (r, al, coef)
    r_ani, al_ani, ca = best
    print(f"    body軸別: hover={float((G / ca[0]) ** (1 / al_ani)):.4f} α={al_ani:.2f}  "
          f"水平 c2h={ca[1]:.4f} c1h={ca[2]:+.3f} / 鉛直 c2v={ca[3]:.4f} c1v={ca[4]:+.3f}"
          f"   残差rms {r_ani:.3f}")
    print(f"      → {'異方性あり(body軸別ドラッグの価値あり)' if r_ani < 0.9 * r_best else '等方で十分'}")

    fit = {"hover": hover, "alpha": float(alpha), "drag_c1": c1, "drag_c2": c2,
           "drag_struct": struct, "resid": r_best,
           "aniso": {"c2h": float(ca[1]), "c1h": float(ca[2]),
                     "c2v": float(ca[3]), "c1v": float(ca[4]),
                     "hover": float((G / ca[0]) ** (1 / al_ani)), "alpha": float(al_ani),
                     "resid": r_ani}}

    # --- 水準別のAの表(読み取り用。上のフィットには使わない)
    A = -f_body[:, 2] - ((c1 + c2 * speed)[:, None] * v_body)[:, 2]
    edge = np.r_[True, np.diff(thr) != 0]
    drop = np.zeros(len(thr), bool)
    for i in np.nonzero(edge)[0]:
        drop[i:i + int(0.15 * RESAMPLE_HZ)] = True
    print("  水準別(過渡0.15s除外・中央値) 実測A と フィット値:")
    for lv in sorted(set(np.round(thr[m], 4))):
        sl = m & (np.abs(thr - lv) < 1e-9) & ~drop
        if sl.sum() < 20:
            continue
        a_med = float(np.median(A[sl]))
        a_fit = G * (lv / hover) ** alpha
        print(f"    thrust={lv:6.4f}: 実測A={a_med:6.3f} (A/g={a_med / G:5.3f})  "
              f"フィット={a_fit:6.3f}  差{a_med - a_fit:+5.3f}  "
              f"|v|中央{np.median(speed[sl]):4.1f}  n={int(sl.sum())}")
    fit["drag_c"] = float(c1)

    # --- (d) IMU accel と運動学由来比力の一致(accel符号・IMUモデルの検証)
    if d["imu"]:
        ac = d["imu"]["accel"]
        s = m & (speed > 0.5)
        if s.sum() > 100:
            print("  IMU accel vs 運動学由来比力 Rᵀ(a-g)(accel_obs_sign と IMU忠実度の検証):")
            for i, nm in enumerate("xyz"):
                a, b_ = f_body[s, i], ac[s, i]
                ratio = float(a @ b_ / (a @ a))
                rms = float(np.sqrt(np.mean((b_ - a) ** 2)))
                print(f"    {nm}: 比 {ratio:+.3f}  差rms {rms:.2f} m/s²  "
                      f"(相関 {np.corrcoef(a, b_)[0, 1]:+.3f})")
    return fit


# ---------------------------------------------------------------- 4) スポーン/ピン

def report_spawn(d: dict):
    """発進直前の姿勢・位置(自作シム spawn_* の根拠)。"""
    print("\n== 4. スポーン状態(自作シム spawn_pitch_deg / spawn_below_center) ==")
    t = d["t"]
    for k, (a, b) in enumerate(d["flights"][:3]):
        s = (t >= a - 0.05) & (t < a + 0.05)
        if s.sum() == 0:
            continue
        rpy = np.degrees(d["rpy"][s].mean(axis=0))
        # 補正済みクォータニオンから求めた標準ZYXオイラー角(=自作シムに入れるべき値)
        R = d["R"][s].mean(axis=0)
        std = np.degrees([np.arctan2(R[2, 1], R[2, 2]),
                          -np.arcsin(np.clip(R[2, 0], -1, 1)),
                          np.arctan2(R[1, 0], R[0, 0])])
        pos = d["pos"][s].mean(axis=0)
        v = d["vel"][s].mean(axis=0)
        print(f"  飛行{k + 1}: ATTITUDE報告 rpy=({rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f})°  "
              f"→ 標準FRD rpy=({std[0]:+.1f},{std[1]:+.1f},{std[2]:+.1f})°")
        print(f"          NED=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f})  "
              f"|v|={np.linalg.norm(v):.2f}")
    print("  ※ 標準FRDの pitch が config.yaml env_physics.spawn_pitch_deg(現行 -17.8)に対応")


# ---------------------------------------------------------------- 5) 出力

def print_recommendation(rate: dict, trans: dict, d: dict):
    print("\n########## config.yaml へ反映する較正値")
    if rate:
        k = [rate[a]["k"] for a in sorted(rate)]
        g = [rate[a]["gain"] for a in sorted(rate)]
        lag = [rate[a]["lag_s"] for a in sorted(rate)]
        if len(k) == 3:
            print(f"  dynamics.k_rate:      [{k[0]:.0f}, {k[1]:.0f}, {k[2]:.0f}]"
                  f"   # 軸別。現行は単一スカラなので drone.py の対応が要る")
            print(f"  dynamics.cmd_gain:    [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]")
        else:
            print(f"  dynamics.k_rate / cmd_gain: {k} / {g}")
        print(f"  # 指令→応答遅延 実測 {[round(x * 1000) for x in lag]} ms "
              f"→ sensor.act_delay_steps = {round(np.median(lag) * 120)} (120Hz物理ステップ換算)")
    if "hover" in trans:
        print(f"  dynamics.hover_thrust: {trans['hover']:.4f}")
        print(f"  dynamics.thrust_alpha: {trans['alpha']:.3f}")
    if "drag_struct" in trans:
        if trans["drag_struct"] == "linear":
            print(f"  dynamics.drag_c:       {trans['drag_c1']:.3f}   # 線形で十分")
        else:
            print(f"  dynamics.drag_c:       {trans['drag_c1']:.3f}   # 線形項")
            print(f"  dynamics.drag_c2:      {trans['drag_c2']:.4f}  # 2次項"
                  f"(drone.py に -m·c2·|v|·v の追加が必要)")
        an = trans.get("aniso")
        if an and an["resid"] < 0.9 * trans["resid"]:
            print(f"  # body軸別ドラッグの方が残差が小さい({an['resid']:.3f} < "
                  f"{trans['resid']:.3f}): 水平 c2h={an['c2h']:.4f} c1h={an['c1h']:+.3f} / "
                  f"鉛直 c2v={an['c2v']:.4f} c1v={an['c1v']:+.3f}")
    if d["imu"]:
        dec = max(1, round(120.0 / max(d["imu"]["hz"], 1.0)))
        print(f"  sensor.imu_decimation: {dec}        # 実測IMU {d['imu']['hz']:.0f}Hz "
              f"(120Hz物理 / {dec})")
    print("\n  ※ VQ1で測った値をVQ2へ持ち込む前に scripts/compare_builds.py で"
          "IMU可視な量の一致を必ず確認する(VQ1レガシーは旧ビルド)。")


def concat_grids(ds: list[dict]) -> dict:
    """複数runを1つの回帰に結合する(速度域・推力域が広がるほど同定が安定する)。

    runは時系列で連続しているのでそのまま連結できる。run間のギャップは
    flights/mask に入らないため回帰には使われない。
    """
    if len(ds) == 1:
        return ds[0]
    # 時刻昇順に並べ替えるのは必須。cmd_at() は searchsorted で指令を引くので、
    # 連結後の t_cmd が単調でないと指令の対応が壊れる(CLIの --dir の順で結果が変わる)。
    ds = sorted(ds, key=lambda x: float(x["t"][0]))
    out = {"dt": ds[0]["dt"]}
    for k in ("t", "cmd", "rpy", "omega", "omega_dot", "pos", "vel", "accel_ned",
              "R", "q", "mask"):
        out[k] = np.concatenate([d[k] for d in ds], axis=0)
    out["flights"] = [f for d in ds for f in d["flights"]]
    out["t_cmd"] = np.concatenate([d["t_cmd"] for d in ds])
    out["cmd_raw"] = np.concatenate([d["cmd_raw"] for d in ds], axis=0)
    out["att"] = {k: np.concatenate([d["att"][k] for d in ds], axis=0)
                  for k in ("t", "om", "mask")}
    if all(d["imu"] for d in ds):
        out["imu"] = {"gyro": np.concatenate([d["imu"]["gyro"] for d in ds], axis=0),
                      "accel": np.concatenate([d["imu"]["accel"] for d in ds], axis=0),
                      "hz": float(np.mean([d["imu"]["hz"] for d in ds]))}
    else:
        out["imu"] = {}
    out["meta"] = ds[0]["meta"]
    out["dir"] = " + ".join(d["dir"] for d in ds)
    out["rate_clip"] = min(d["rate_clip"] for d in ds)
    kinds = set.intersection(*[set(d["truth"]) for d in ds])
    out["truth"] = {k: {"t": np.concatenate([d["truth"][k]["t"] for d in ds]),
                        "v": np.concatenate([d["truth"][k]["v"] for d in ds], axis=0),
                        "hz": float(np.mean([d["truth"][k]["hz"] for d in ds])),
                        "n_raw": int(sum(d["truth"][k]["n_raw"] for d in ds))}
                    for k in kinds}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, nargs="+", help="fly_dcl --record-dir で録った run")
    ap.add_argument("--smooth", type=int, default=5, help="微分前の移動平均サンプル数")
    ap.add_argument("--separate", action="store_true",
                    help="runを結合せず個別に解析する(既定は結合して1つの回帰にする)")
    ap.add_argument("--rate-clip", type=float, default=0.0,
                    help="この指令レート[rad/s]を超えるサンプルを同定から除く(飽和領域)。"
                         "0で config.yaml の dynamics.rate_max を使う")
    args = ap.parse_args()

    if args.rate_clip <= 0:
        from ..user_config import uc
        args.rate_clip = float(uc("dynamics", "rate_max", 2.65))

    ds = []
    for dir_ in args.dir:
        d = build_grid(dir_, smooth_n=args.smooth, rate_clip=args.rate_clip)
        report_rates(d, dir_)
        ds.append(d)
        if args.separate:
            audit_signs(d)
            audit_body_rates(d)
            print_recommendation(fit_rate_loop(d), fit_translation(d), d)
            report_spawn(d)
            print()
    if args.separate:
        return
    # satプランは意図的に飽和/大振幅で飛ばすrun。線形モデルの同定に混ぜると
    # 一次モデルから外れた回復挙動(角加速度制限の疑い)にゲインと時定数が引っ張られる
    # (実測: 混ぜると cmd_gain 0.95→1.16、k_rate 61→16 まで崩壊)。飽和上限の測定だけに使う。
    sat_ds = [x for x in ds if x["meta"].get("sysid_plan") == "sat"]
    fit_ds = [x for x in ds if x["meta"].get("sysid_plan") != "sat"] or ds
    for x in sat_ds:
        print(f"\n########## {x['dir']} (plan=sat) は飽和測定専用: "
              f"振幅依存だけ見てパラメータ同定からは除外")
        report_rate_linearity(x)
    d = concat_grids(fit_ds)
    if len(fit_ds) > 1:
        print(f"\n########## {len(fit_ds)}本を結合して同定 "
              f"(有効 {d['mask'].sum() / RESAMPLE_HZ:.1f}s)")
    audit_signs(d)
    if not sat_ds:
        report_rate_linearity(d)
    audit_body_rates(d)
    rate = fit_rate_loop(d)
    trans = fit_translation(d)
    report_spawn(d)
    print_recommendation(rate, trans, d)


if __name__ == "__main__":
    main()
