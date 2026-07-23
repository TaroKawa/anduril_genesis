# -*- coding: utf-8 -*-
"""VQ1/VQ2 シミュレータの MAVLink テレメトリ棚卸し(受動 + 任意で飛行時)。

目的:
  1. 「VQ1(レガシー)ではテレメトリインターフェースが復元された」を実測で確認する。
  2. 状態推定系(ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / トラック情報)が
     出ているか、型ごとの受信数・レート・サンプル値で示す。
  3. 「飛行中でないと出ないテレメトリ」があるかを確認する(--fly)。

公式サンプル(check/mavlink_rx.py・controller.py)から判明した重要事実:
  - 「最新版シム(=VQ2)」では ATTITUDE / LOCAL_POSITION_NED / ODOMETRY /
    トラック情報(ゲート位置)は *設定で無効化* されている(コメントに明記)。
    → これらが出れば VQ1レガシー(テレメトリ有効版)である証拠。
  - ENCAPSULATED_DATA は sub-type 1=レース状態 / 2=トラック情報。
  - COLLISION / ACTUATOR_OUTPUT_STATUS / TIMESYNC / DATA_TRANSMISSION_HANDSHAKE
    (トラックデータのハンドシェイク)も流れ得る。

実行(WSL、Windows側でシムを起動した状態で):
  受動のみ(推奨・安全):
    UV_PROJECT_ENVIRONMENT=.venv-host uv run python check/probe_telemetry.py --secs 12
  飛行時テレメトリも確認(ARM してドローンを実際に動かす):
    UV_PROJECT_ENVIRONMENT=.venv-host uv run python check/probe_telemetry.py --fly

  ※ シムは Windows の 127.0.0.1 へ UDP 送信するため、既存 win_relay.py を
    自動起動して WSL へ転送させる。既にリレーが動いていれば --no-relay。
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# VQ2 でも出る基本メッセージ(テレメトリ「無し」でも出る)
BASELINE = {"HEARTBEAT", "HIGHRES_IMU", "ENCAPSULATED_DATA", "TIMESYNC",
            "ACTUATOR_OUTPUT_STATUS", "COLLISION"}
# 状態推定系(VQ2では公式に無効化 → 出れば VQ1レガシー=テレメトリ復元の証拠)
STATE_ESTIMATE = {"ATTITUDE", "ATTITUDE_QUATERNION", "LOCAL_POSITION_NED",
                  "GLOBAL_POSITION_INT", "ODOMETRY", "VFR_HUD", "ALTITUDE"}

SAMPLE_FIELDS = {
    "ATTITUDE": ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"),
    "LOCAL_POSITION_NED": ("x", "y", "z", "vx", "vy", "vz"),
    "ODOMETRY": ("x", "y", "z", "vx", "vy", "vz"),
    "HIGHRES_IMU": ("xgyro", "ygyro", "zgyro", "xacc", "yacc", "zacc"),
    "ACTUATOR_OUTPUT_STATUS": ("actuator",),
    "COLLISION": ("id", "threat_level", "horizontal_minimum_delta"),
}

# controller.py 由来: rev3390 の rad/s 指令オプトイン(bit16)
DCL_BODY_RATES_RADS_BIT = 16
ENCAP_RACE_STATUS = 1
ENCAP_TRACK_INFO = 2


def _fmt_fields(msg, names):
    out = []
    for n in names:
        v = getattr(msg, n, None)
        if v is None:
            continue
        if isinstance(v, float):
            out.append(f"{n}={v:.3f}")
        elif isinstance(v, (list, tuple)):
            out.append(f"{n}=[" + ",".join(f"{x:.2f}" for x in v[:4]) + "]")
        else:
            out.append(f"{n}={v}")
    return " ".join(out)


class Tally:
    """メッセージ型ごとの受信数・時間窓・最新メッセージ。"""

    def __init__(self):
        self.counts = defaultdict(int)
        self.first_t = {}
        self.last_t = {}
        self.last_msg = {}
        self.encap_subtypes = defaultdict(int)  # ENCAPSULATED_DATA の sub-type別

    def add(self, t, msg, now):
        self.counts[t] += 1
        self.last_msg[t] = msg
        self.last_t[t] = now
        self.first_t.setdefault(t, now)
        if t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw:
                self.encap_subtypes[raw[0]] += 1

    def report(self, title):
        print("\n" + "=" * 74)
        print(f"[{title}] 受信型 {len(self.counts)} 種")
        print("=" * 74)
        for t, c in sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0])):
            span = max(self.last_t[t] - self.first_t[t], 1e-6)
            hz = (c - 1) / span if c > 1 else 0.0
            tag = ("  <-- 状態推定テレメトリ" if t in STATE_ESTIMATE
                   else "  (基本)" if t in BASELINE else "  (その他)")
            print(f"  {t:<26} {c:6d}件  ~{hz:6.1f}Hz{tag}")
            if t in SAMPLE_FIELDS:
                s = _fmt_fields(self.last_msg[t], SAMPLE_FIELDS[t])
                if s:
                    print(f"      最新: {s}")
            if t == "ENCAPSULATED_DATA":
                subs = {ENCAP_RACE_STATUS: "race_status(1)",
                        ENCAP_TRACK_INFO: "track_info(2)"}
                desc = ", ".join(f"{subs.get(k, k)}×{v}"
                                 for k, v in sorted(self.encap_subtypes.items()))
                print(f"      sub-type: {desc}")
        return set(self.counts)


def collect(conn, mavutil, secs, tally, arm_conn_send=None, race_state=None):
    """secs 秒間メッセージを収集。arm_conn_send があれば毎ループ呼ぶ(飛行指令)。"""
    m = mavutil.mavlink
    t_end = time.time() + secs
    next_hb = next_ts = next_cmd = 0.0
    while time.time() < t_end:
        now = time.time()
        if now >= next_hb:                       # GCS ハートビート 2Hz
            next_hb = now + 0.5
            conn.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        if now >= next_ts:                       # TIMESYNC 10Hz(公式サンプル準拠)
            next_ts = now + 0.1
            conn.mav.timesync_send(int(time.time_ns()), 0)
        if arm_conn_send is not None and now >= next_cmd:  # 250Hz 指令
            next_cmd = now + 1.0 / 250.0
            arm_conn_send(now)
        msg = conn.recv_match(blocking=False)
        if msg is None:
            time.sleep(0.001)
            continue
        t = msg.get_type()
        if t == "BAD_DATA":
            continue
        tally.add(t, msg, now)
        if race_state is not None and t == "ENCAPSULATED_DATA":
            raw = bytes(msg.data)
            if raw and raw[0] == ENCAP_RACE_STATUS:
                (_, sim_boot, race_start, race_finish, agi, _) = struct.unpack_from(
                    "<BQqqIq", raw)
                started = race_start is not None and race_start > 0 and sim_boot >= race_start
                race_state["pin_released"] = bool(started)
                race_state["active_gate_index"] = int(agi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=12.0, help="受動収集の秒数")
    ap.add_argument("--fly", action="store_true",
                    help="ARM して穏やかな指令を送り、飛行中テレメトリを確認する")
    ap.add_argument("--fly-secs", type=float, default=10.0, help="飛行フェーズの秒数")
    ap.add_argument("--thrust", type=float, default=0.30, help="飛行フェーズの推力(0-1)")
    ap.add_argument("--mavlink-ip", type=str, default="0.0.0.0")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--no-relay", action="store_true")
    args = ap.parse_args()

    from pymavlink import mavutil

    relay_proc = None
    try:
        if not args.no_relay:
            from genesis_rl.dcl.client import spawn_win_relay
            relay_proc = spawn_win_relay(args.mavlink_port, args.video_port)

        conn = mavutil.mavlink_connection(f"udpin:{args.mavlink_ip}:{args.mavlink_port}")
        print(f"MAVLink: udpin:{args.mavlink_ip}:{args.mavlink_port} ハートビート待ち ...",
              flush=True)
        conn.wait_heartbeat()
        print(f"接続: system={conn.target_system} component={conn.target_component}",
              flush=True)

        # ---- フェーズA: 受動(ARM しない) ----
        passive = Tally()
        print(f"\n[フェーズA] 受動収集 {args.secs:.0f}s(ARM/指令なし) ...", flush=True)
        collect(conn, mavutil, args.secs, passive)
        passive_types = passive.report(f"受動 {args.secs:.0f}s")

        flying_types = set()
        if args.fly:
            # ---- フェーズB: ARM + 穏やかな飛行指令 ----
            print(f"\n[フェーズB] ARM して飛行時テレメトリを確認 "
                  f"({args.fly_secs:.0f}s, thrust={args.thrust}, rad/s指令) ...", flush=True)
            m = mavutil.mavlink
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                m.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            boot_ms = int(time.time() * 1000)
            type_mask = (m.ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE
                         | DCL_BODY_RATES_RADS_BIT)   # rev3390: 真の rad/s 解釈

            def send_cmd(now):
                conn.mav.set_attitude_target_send(
                    int(now * 1000) - boot_ms,
                    conn.target_system, conn.target_component,
                    type_mask, [1, 0, 0, 0],
                    0.0, 0.0, 0.0, float(args.thrust))  # ほぼホバー(rates=0)

            race_state = {"pin_released": False, "active_gate_index": 0}
            flying = Tally()
            collect(conn, mavutil, args.fly_secs, flying,
                    arm_conn_send=send_cmd, race_state=race_state)
            flying_types = flying.report(f"飛行中 {args.fly_secs:.0f}s")
            print(f"  レース pin_released={race_state['pin_released']} "
                  f"active_gate_index={race_state['active_gate_index']}", flush=True)

        # ---- 判定 ----
        all_types = passive_types | flying_types
        found = sorted(all_types & STATE_ESTIMATE)
        track_info = passive.encap_subtypes.get(ENCAP_TRACK_INFO, 0)
        print("\n" + "-" * 74)
        if found:
            print(f"✅ テレメトリ復元を確認: 状態推定メッセージ → {found}")
            print("   → ビジョン非依存(姿勢/位置/速度直読み)の航法が可能。")
        else:
            print("❌ 状態推定系(ATTITUDE/LOCAL_POSITION_NED/ODOMETRY)は受信できず。")
            print("   公式サンプルのコメント通り、これらは『最新版シム(VQ2)』で無効化。")
            print("   → 現在動作中はテレメトリ無効ビルド。VQ1レガシー版の起動が必要。")
        if args.fly:
            only_flying = sorted(flying_types - passive_types)
            print(f"\n[飛行で増えた型] {only_flying if only_flying else 'なし'}")
            if not only_flying:
                print("   → 『飛ばさないと取れない』テレメトリは無い(型セットは飛行で不変)。")
            else:
                print("   → 一部テレメトリは飛行中のみ出る。")
        print(f"[トラック情報(ゲート位置) ENCAPSULATED sub-type2]: "
              f"{'受信あり' if track_info else '受信なし(VQ2では nulled)'}")
        print("-" * 74, flush=True)

    except KeyboardInterrupt:
        print("\n中断しました。", flush=True)
    finally:
        if relay_proc is not None:
            try:
                relay_proc.terminate()
                relay_proc.wait(timeout=3.0)
            except Exception:
                try:
                    relay_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
