# -*- coding: utf-8 -*-
"""「何をしないとピンが外れて発進しないのか」を実シムで切り分ける。

  # VQ1(真値テレメトリ有効)を起動した状態で
  python -m genesis_rl.scripts.probe_start --no-relay

自作シム(Genesis)の発進シーケンスを実機に合わせるための一次データを取る。
1試行ごとに: シムリセット → レース予約(start_pending)を待つ → 指定の条件で指令を送りながら
カウントダウンを越え、位置が動き出すかを見る。

判定は真値の位置(LOCAL_POSITION_NED)で行う。race_status のフラグは4Hzしか来ないので
「実際に動き出した時刻」の分解能が足りない。

試行(--trials で選択可):
  none    SET_ATTITUDE_TARGET を一切送らない(ハートビート/TIMESYNCのみ)
  t0.00   推力0を送り続ける
  t0.20   推力0.20(ホバー0.2665未満)
  t0.24   推力0.239(学習アクション帯の旧下端)
  t0.265  推力0.265(deploy START_THRUST = takeoff_thrust)
  t0.30   推力0.30
  noarm   ARM せずに 0.265 を送る
  rate    ピン中に roll 指令を入れて「回転は自由か(並進ピンか)」を見る
"""

from __future__ import annotations

import argparse
import collections
import time

import numpy as np

TRIALS = {
    "none": dict(thrust=None, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.00": dict(thrust=0.00, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.20": dict(thrust=0.20, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.24": dict(thrust=0.239, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.265": dict(thrust=0.265, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.30": dict(thrust=0.30, arm=True, rates=(0.0, 0.0, 0.0)),
    "noarm": dict(thrust=0.265, arm=False, rates=(0.0, 0.0, 0.0)),
    "rate": dict(thrust=0.265, arm=True, rates=(0.5, 0.0, 0.0)),
    # カウントダウン中は何も送らず、開始フラグから delay 秒後に送り始める。
    # 「解除の引き金は時刻か、それとも指令そのものか」を切り分ける。
    "late": dict(thrust=0.265, arm=True, rates=(0.0, 0.0, 0.0), delay_after_flag=2.0),
    # 解除の閾値を挟み込む(t0.00は動かず、t0.20は動くと判明済み)
    "t0.01": dict(thrust=0.01, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.05": dict(thrust=0.05, arm=True, rates=(0.0, 0.0, 0.0)),
    "t0.10": dict(thrust=0.10, arm=True, rates=(0.0, 0.0, 0.0)),
    # 推力0でレートだけ送る: 引き金は「何か送ること」か「推力>0」かの切り分け
    "rate0": dict(thrust=0.0, arm=True, rates=(0.5, 0.0, 0.0)),
    # 解除閾値の直読み: 推力を 0.08→0.24 へ8秒でランプし、動き出した瞬間の指令値を読む。
    # 試行を分けて挟み込むより、リセット1回で済むぶん確実(ランプ率0.02/s)。
    "ramp": dict(thrust=0.08, arm=True, rates=(0.0, 0.0, 0.0), ramp=(0.08, 0.24, 8.0)),
    # 発進後に閾値未満(0.05)へ落とす: 低推力指令が *飛行中も* 無視されるのかを見る。
    # 新しいアクション帯 thrust∈[0,0.35] の下半分(0〜0.18)が実機で死んでいないかの確認。
    "lowthr": dict(thrust=0.265, arm=True, rates=(0.0, 0.0, 0.0),
                   phase2_at_s=2.0, phase2_thrust=0.05, phase2_rates=(0.5, 0.0, 0.0)),
}
MOTION_M = 0.05          # これ以上位置が動いたら「発進した」
ATT_MOVED_RAD = 0.05     # これ以上姿勢が変わったら「回転は拘束されていない」


def wait_reset(mav, shared, timeout=6.0) -> bool:
    """シムをリセットし、機体がスポーンへ戻る(=sim_boot_ms が巻き戻る)まで待つ。

    **レース予約(start_pending)は待たない**。この実験で判ったとおり拘束の解除は
    カウントダウンではなく指令で起きるので、予約状態は物理に関係ない。
    予約を待つ実装にすると「レースが再スケジュールされない」状態で延々ハングする。
    """
    prev = (shared.get("race") or {}).get("sim_boot_ms")
    mav.sim_reset()
    end = time.time() + timeout
    while time.time() < end:
        mav.heartbeat_if_due()
        sb = (shared.get("race") or {}).get("sim_boot_ms")
        if prev is None or (sb is not None and sb < prev - 1000):
            time.sleep(0.4)      # スポーン直後の落ち着き待ち
            return True
        time.sleep(0.02)
    return False


def run_trial(mav, shared, name: str, cfg: dict, tx_hz: float, hold_s: float) -> dict:
    from .. import contracts as C   # noqa: F401  (契約の読み込み確認のみ)

    if not wait_reset(mav, shared):
        return {"name": name, "error": "リセットが反映されない(シムが応答していない)"}
    if cfg["arm"]:
        mav.arm()

    race0 = dict(shared.get("race") or {})
    t0 = time.time()
    # カウントダウン残り [s](race_status の生値から)
    cd = None
    if race0.get("race_start_ms") is not None:
        cd = (race0["race_start_ms"] - race0["sim_boot_ms"]) / 1000.0

    pos, vel, att, acc, flag_t = [], [], [], [], None
    tx_log = []                      # (時刻, 送った推力) 解除閾値の読み取り用
    next_tx = 0.0
    delay = cfg.get("delay_after_flag")
    ramp = cfg.get("ramp")           # (lo, hi, 秒) 指定時は推力を線形に上げる
    t_end = t0 + (ramp[2] if ramp else (cd if cd and cd > 0 else 4.0)) + hold_s + (delay or 0.0)
    while time.time() < t_end:
        now = time.time()
        mav.heartbeat_if_due()
        # delay_after_flag 指定時は「開始フラグ + delay」まで送信しない
        tx_open = cfg["thrust"] is not None and (
            delay is None or (flag_t is not None and now >= flag_t + delay))
        if tx_open and now >= next_tx:
            next_tx = now + 1.0 / tx_hz
            thr, rates = cfg["thrust"], cfg["rates"]
            if ramp:
                lo, hi, dur = ramp
                thr = lo + (hi - lo) * min((now - t0) / dur, 1.0)
            p2 = cfg.get("phase2_at_s")
            if p2 is not None and now - t0 >= p2:
                thr, rates = cfg["phase2_thrust"], cfg["phase2_rates"]
            mav.send_rates(*rates, thr)
            tx_log.append((now, thr))
        tr = shared.get("truth") or {}
        if "POS" in tr:
            pos.append((now, tr["POS"]["v"][:3]))
            vel.append((now, tr["POS"]["v"][3:6]))
        if "ATT" in tr:
            att.append((now, tr["ATT"]["v"][:3]))
        im = shared.get("imu") or {}
        if im:
            acc.append((now, im["accel"]))
        race = shared.get("race") or {}
        if flag_t is None and race.get("pin_released"):
            flag_t = now
        time.sleep(0.002)

    if not pos:
        return {"name": name, "error": "位置テレメトリが来ない(VQ1で起動しているか)"}

    tp = np.array([p[0] for p in pos])
    xyz = np.array([p[1] for p in pos], float)
    ref = np.median(xyz[tp < tp[0] + 0.5], axis=0)      # ピン中の基準位置
    moved = np.linalg.norm(xyz - ref, axis=1) > MOTION_M
    t_motion = float(tp[np.argmax(moved)]) if moved.any() else None

    out = {"name": name, "countdown_s": cd, "released": bool(moved.any()),
           "t_motion_rel": (t_motion - t0) if t_motion else None,
           "t_flag_rel": (flag_t - t0) if flag_t else None,
           "t_motion_after_flag": ((t_motion - flag_t) if (t_motion and flag_t) else None),
           "pos_ref": ref.tolist(),
           "drift_m": float(np.linalg.norm(xyz[-1] - ref))}
    # 解除直後の立ち上がり: 発進 0.2/0.5s 後の速度と、解除前後の比力ピーク
    if t_motion is not None and vel:
        tv = np.array([v[0] for v in vel])
        vv = np.array([v[1] for v in vel], float)
        for dt_ in (0.2, 0.5, 1.0):
            sel = np.abs(tv - (t_motion + dt_)) < 0.05
            if sel.any():
                out[f"v_at_{dt_}s"] = float(np.linalg.norm(vv[sel].mean(axis=0)))
    if t_motion is not None and acc:
        tc = np.array([a[0] for a in acc])
        aa = np.array([a[1] for a in acc], float)
        sel = (tc > t_motion - 0.3) & (tc < t_motion + 0.5)
        if sel.any():
            out["accel_peak_g"] = float(np.linalg.norm(aa[sel], axis=1).max() / 9.81)
    # ランプ試行: 動き出した瞬間に送っていた推力 = 解除閾値(遅延分を差し引いた値も出す)
    if ramp and t_motion is not None and tx_log:
        tt = np.array([x[0] for x in tx_log])
        th = np.array([x[1] for x in tx_log], float)
        i = int(np.searchsorted(tt, t_motion) - 1)
        out["thrust_at_motion"] = float(th[max(i, 0)])
        j = int(np.searchsorted(tt, t_motion - 0.15) - 1)
        out["thrust_at_motion_minus150ms"] = float(th[max(j, 0)])
        out["ramp_rate"] = float((ramp[1] - ramp[0]) / ramp[2])

    # ピン中(=動き出す前)の姿勢変化と比力
    pre = tp < (t_motion - 0.05 if t_motion else tp[-1])
    if att:
        ta = np.array([a[0] for a in att])
        rpy = np.array([a[1] for a in att], float)
        pre_a = ta < (t_motion - 0.05 if t_motion else ta[-1])
        if pre_a.sum() > 5:
            d = np.abs(rpy[pre_a] - rpy[pre_a][0]).max(axis=0)
            out["att_change_pre_deg"] = np.degrees(d).round(2).tolist()
    if acc:
        tc = np.array([a[0] for a in acc])
        aa = np.array([a[1] for a in acc], float)
        pre_c = tc < (t_motion - 0.05 if t_motion else tc[-1])
        if pre_c.sum() > 5:
            mag = np.linalg.norm(aa[pre_c], axis=1)
            out["accel_pre_g"] = (float(np.median(mag) / 9.81), float(mag.max() / 9.81))
    out["n_pos_pre"] = int(pre.sum())

    # phase2(発進後に低推力+レートへ切替)の応答: 指令が生きているかを比力と角速度で見る
    p2 = cfg.get("phase2_at_s")
    if p2 is not None and att:
        t_p2 = t0 + p2 + 0.25            # 切替の反映を待つ
        ta = np.array([a[0] for a in att])
        rpy = np.array([a[1] for a in att], float)
        sel = (ta > t_p2) & (ta < t_p2 + 1.0)
        if sel.sum() > 5:
            roll = rpy[sel, 0]
            dt_ = ta[sel][-1] - ta[sel][0]
            out["phase2_roll_rate"] = float((roll[-1] - roll[0]) / max(dt_, 1e-6))
            out["phase2_roll_cmd"] = float(cfg["phase2_rates"][0])
        if acc:
            tc = np.array([a[0] for a in acc])
            aa = np.array([a[1] for a in acc], float)
            s2 = (tc > t_p2) & (tc < t_p2 + 1.0)
            if s2.sum() > 5:
                # body z の比力 = -A。低推力が本当に効いていれば |f_z| は小さくなる
                out["phase2_A_over_g"] = float(np.median(-aa[s2, 2]) / 9.81)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mavlink-ip", default="0.0.0.0")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--no-relay", action="store_true")
    ap.add_argument("--trials", nargs="+", default=list(TRIALS),
                    choices=list(TRIALS), help="実行する試行")
    ap.add_argument("--tx-hz", type=float, default=90.0)
    ap.add_argument("--hold", type=float, default=3.0,
                    help="カウントダウン終了後に観測を続ける秒数")
    args = ap.parse_args()

    from ..dcl.client import MavlinkIO, spawn_win_relay

    relay = None
    mav = None
    shared: dict = {}
    results = []
    try:
        if not args.no_relay:
            relay = spawn_win_relay(args.mavlink_port, args.video_port)
        mav = MavlinkIO(shared, args.mavlink_ip, args.mavlink_port)
        shared["imu_log"] = collections.deque(maxlen=10)   # 使わないが実装対称のため
        for name in args.trials:
            print(f"\n---- 試行 {name}: {TRIALS[name]} ----", flush=True)
            r = run_trial(mav, shared, name, TRIALS[name], args.tx_hz, args.hold)
            results.append(r)
            print("  " + str(r), flush=True)
    except KeyboardInterrupt:
        print("\n中断", flush=True)
    finally:
        if mav is not None:
            try:
                mav.sim_reset()
                mav.close()
            except Exception:
                pass
        if relay is not None:
            try:
                relay.terminate()
                relay.wait(timeout=3)
            except Exception:
                pass

    print("\n" + "=" * 78)
    hdr = (f"{'試行':8s} {'CD[s]':>6s} {'発進':>4s} {'flag後':>7s} {'漂流m':>7s} "
           f"{'|a|ピン中':>9s} {'|a|解除時':>9s} {'v@0.2':>6s} {'v@0.5':>6s} {'v@1.0':>6s} "
           f"{'ピン中姿勢変化deg':>18s}")
    print(hdr)
    for r in results:
        if r.get("error"):
            print(f"{r['name']:8s} {r['error']}")
            continue

        def _f(k, fmt="%.2f", na="-"):
            v = r.get(k)
            return (fmt % v) if isinstance(v, (int, float)) else na

        ac = r.get("accel_pre_g")
        print(f"{r['name']:8s} {_f('countdown_s'):>6s} {'○' if r['released'] else '×':>4s} "
              f"{_f('t_motion_after_flag'):>7s} {_f('drift_m'):>7s} "
              f"{(('%.2f' % ac[0]) if ac else '-'):>9s} {_f('accel_peak_g'):>9s} "
              f"{_f('v_at_0.2s'):>6s} {_f('v_at_0.5s'):>6s} {_f('v_at_1.0s'):>6s} "
              f"{str(r.get('att_change_pre_deg', '-')):>18s}")
    print("=" * 78)
    print("読み方:")
    print("  発進× = その条件では機体が動かない(=発進に必要な条件が欠けている)")
    print("  flag後 = レース開始フラグから実際に動き出すまでの秒数(負なら旗より先に動いた)")
    print("  |a|ピン中 が 1.00g のままなら重力を拘束が支えている(=ピン留め)")
    print("  ピン中に姿勢だけ変わるなら並進ピン(回転は自由)")


if __name__ == "__main__":
    main()
