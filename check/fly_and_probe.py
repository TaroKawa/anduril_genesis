# -*- coding: utf-8 -*-
"""公式サンプルクライアントを *そのまま使って* ドローンを実際に飛ばし、
無効化されているテレメトリ(ATTITUDE / LOCAL_POSITION_NED / ODOMETRY /
トラック情報=ゲート位置)が飛行中に取れないかを実測する。

方針(「公式サンプルを使う」を厳守):
  - 接続/スレッド構成は公式 setup.setup_components() をそのまま使う。
  - 受信は公式 mavlink_rx.MAVLinkRX を継承し、各 on_* ハンドラに *カウンタだけ*
    足した InstrumentedMAVLinkRX に差し替える(ロジックは super() で公式のまま)。
  - 飛行指令は公式 controller.update_attitude_flight_control() をそのまま送る
    (controller.py で USE_RAD_PER_SEC_BODY_RATES=True: rev3390 の rad/s bit16 適用済み、
     PITCH_RATE=-0.3 で前進 / THRUST=0.6)。
  - WSL 対応: シムは Windows 127.0.0.1 へ送るため win_relay を自動起動し、
    公式 setup を server_ip="0.0.0.0" で呼ぶ(サンプル本体は無改変)。

実行(Windows側でシムを起動した状態で):
  UV_PROJECT_ENVIRONMENT=.venv-host uv run python check/fly_and_probe.py --fly-secs 12
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_HERE, _REPO):          # check/ (公式サンプル) と リポジトリルート の両方
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pymavlink import mavutil

# --- 公式サンプルのモジュール(無改変で import) ---
import setup as sample_setup
import controller as sample_controller
from mavlink_rx import MAVLinkRX, ENCAPSULATED_RACE_STATUS_MSG_ID, ENCAPSULATED_TRACK_INFO_MSG_ID

# VQ2 で公式に無効化されている = これが出れば「テレメトリ復元(VQ1)」の証拠
STATE_ESTIMATE = {"ATTITUDE", "LOCAL_POSITION_NED", "ODOMETRY", "TRACK_INFO(gates)"}

_LOCK = threading.Lock()
TALLY: dict[str, int] = defaultdict(int)
SAMPLE: dict[str, str] = {}
RACE = {"pin_released": False, "active_gate_index": 0}


def _bump(name, msg=None, fields=None):
    with _LOCK:
        TALLY[name] += 1
        if fields and msg is not None:
            parts = []
            for f in fields:
                v = getattr(msg, f, None)
                if isinstance(v, float):
                    parts.append(f"{f}={v:.3f}")
                elif isinstance(v, (list, tuple)):
                    parts.append(f"{f}=[" + ",".join(f"{x:.2f}" for x in v[:4]) + "]")
                elif v is not None:
                    parts.append(f"{f}={v}")
            SAMPLE[name] = " ".join(parts)


class InstrumentedMAVLinkRX(MAVLinkRX):
    """公式 MAVLinkRX を継承し、各ハンドラにカウンタを足すだけ(処理は super で公式のまま)。"""

    def on_heartbeat(self, msg):
        _bump("HEARTBEAT"); super().on_heartbeat(msg)

    def on_timesync(self, msg):
        _bump("TIMESYNC"); super().on_timesync(msg)

    def on_attitude(self, msg):
        _bump("ATTITUDE", msg, ("roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"))
        super().on_attitude(msg)

    def on_local_position_ned(self, msg):
        _bump("LOCAL_POSITION_NED", msg, ("x", "y", "z", "vx", "vy", "vz"))
        super().on_local_position_ned(msg)

    def on_odometry(self, msg):
        _bump("ODOMETRY", msg, ("x", "y", "z", "vx", "vy", "vz"))
        super().on_odometry(msg)

    def on_highres_imu(self, msg):
        _bump("HIGHRES_IMU", msg, ("xgyro", "ygyro", "zgyro", "xacc", "yacc", "zacc"))
        super().on_highres_imu(msg)

    def on_actuator_output_status(self, msg):
        _bump("ACTUATOR_OUTPUT_STATUS", msg, ("actuator",))
        super().on_actuator_output_status(msg)

    def on_collision(self, msg):
        _bump("COLLISION", msg, ("id", "threat_level", "horizontal_minimum_delta"))
        super().on_collision(msg)

    def on_encapsulated_data(self, msg):
        _bump("ENCAPSULATED_DATA"); super().on_encapsulated_data(msg)

    def on_race_status(self, msg):
        import struct
        raw = bytes(msg.data)
        (_, _sb, race_start, _rf, agi, _lg) = struct.unpack_from("<BQqqIq", raw)
        with _LOCK:
            RACE["active_gate_index"] = int(agi)
            RACE["pin_released"] = bool(race_start is not None and race_start > 0)
        _bump("ENCAP:race_status(1)")
        super().on_race_status(msg)

    def on_track_data(self, payload):
        _bump("TRACK_INFO(gates)")   # sub-type2 が組み上がって初めて呼ばれる
        super().on_track_data(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly-secs", type=float, default=12.0, help="飛行しながら計測する秒数")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--no-relay", action="store_true")
    ap.add_argument("--no-fly", action="store_true",
                    help="ARM/指令を送らず受動のみ(比較用)")
    args = ap.parse_args()

    relay_proc = None
    if not args.no_relay:
        from genesis_rl.dcl.client import spawn_win_relay
        relay_proc = spawn_win_relay(args.mavlink_port, args.video_port)

    # 公式 setup が生成する MAVLinkRX を計測版へ差し替え(サンプル本体は無改変)
    sample_setup.MAVLinkRX = InstrumentedMAVLinkRX

    boot_ms = int(time.time() * 1000)
    shared: dict = {}
    # 公式 setup_components をそのまま使用(server_ip を 0.0.0.0 にするだけ)
    components = sample_setup.setup_components(shared, boot_ms, "0.0.0.0", args.mavlink_port)
    sim_conn = components["sim_conn"]
    controller = components["controller"]

    # 公式 TimeSync は setup 内で start されない実装なので明示的に起動(faithful & 有効化)
    from timesync import TimeSync
    ts = TimeSync.create_timesync(sim_conn, shared)

    try:
        if not args.no_fly:
            print("Arming drone... (公式 controller.arm)", flush=True)
            controller.arm()
            print(f"飛行しながら {args.fly_secs:.0f}s 計測 "
                  f"(公式 update_attitude_flight_control: pitch={sample_controller.PITCH_RATE} "
                  f"thrust={sample_controller.THRUST} rad/s bit適用)...", flush=True)
        else:
            print(f"受動 {args.fly_secs:.0f}s 計測(ARM/指令なし)...", flush=True)

        t0 = time.time()
        next_cmd = next_hb = next_log = 0.0
        m = mavutil.mavlink
        while time.time() - t0 < args.fly_secs:
            now = time.time()
            if not args.no_fly and now >= next_cmd:          # 250Hz 公式姿勢指令
                next_cmd = now + 1.0 / sample_controller.CONTROL_HZ
                sample_controller.update_attitude_flight_control(sim_conn, boot_ms)
            if now >= next_hb:                                # GCS heartbeat 1Hz(リンク維持)
                next_hb = now + 1.0
                sim_conn.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            if now >= next_log:                               # 3秒ごとに生存ログ
                next_log = now + 3.0
                with _LOCK:
                    seen = dict(TALLY)
                print(f"  t={now - t0:4.1f}s pin={RACE['pin_released']} "
                      f"gate={RACE['active_gate_index']} 型={len(seen)} "
                      f"imu={seen.get('HIGHRES_IMU', 0)} "
                      f"att={seen.get('ATTITUDE', 0)} pos={seen.get('LOCAL_POSITION_NED', 0)} "
                      f"odom={seen.get('ODOMETRY', 0)}", flush=True)
            time.sleep(0.001)
    finally:
        # スレッド停止(公式 get_thread_for_join。TimeSync は None を返し得るのでガード)
        for key in ("ts_loop", "mavlink_rx", "vision_rx"):
            comp = components.get(key)
            if comp is not None:
                th = comp.get_thread_for_join()
                if th is not None:
                    th.join(timeout=1.0)
        th = ts.get_thread_for_join()
        if th is not None:
            th.join(timeout=1.0)
        if relay_proc is not None:
            try:
                relay_proc.terminate(); relay_proc.wait(timeout=3.0)
            except Exception:
                try:
                    relay_proc.kill()
                except Exception:
                    pass

    # ---- レポート ----
    with _LOCK:
        seen = dict(TALLY); samp = dict(SAMPLE)
    print("\n" + "=" * 74)
    print(f"受信テレメトリ型: {len(seen)} 種 / 飛行計測 {args.fly_secs:.0f}s "
          f"(pin_released={RACE['pin_released']}, gate={RACE['active_gate_index']})")
    print("=" * 74)
    for name, c in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])):
        tag = "  <-- 状態推定/トラック(VQ2で無効)" if name in STATE_ESTIMATE else ""
        print(f"  {name:<28} {c:6d}件{tag}")
        if name in samp and samp[name]:
            print(f"      最新: {samp[name]}")
    found = sorted(set(seen) & STATE_ESTIMATE)
    print("\n" + "-" * 74)
    if found:
        print(f"✅ 飛行中に無効化テレメトリを受信: {found}")
        print("   → VQ1レガシー(テレメトリ有効)版で稼働している。ビジョン非依存航法が可能。")
    else:
        print("❌ 実際に飛ばしても ATTITUDE / LOCAL_POSITION_NED / ODOMETRY / トラック情報は")
        print("   一切受信できず。公式サンプルのコメント通り『最新版シム=VQ2』では設定で無効。")
        print("   → 飛行の有無に関係なくビルドで無効。VQ1レガシー版の起動が必要。")
    print("-" * 74, flush=True)

    os._exit(0)   # 公式 RX/Vision は daemon=False。確実に終了させる


if __name__ == "__main__":
    main()
