# -*- coding: utf-8 -*-
"""SIM_RESET を送ってからトラックデータ(ゲート位置=GATE_INFO 相当)を捕捉する。

トラック情報は DATA_TRANSMISSION_HANDSHAKE(chunk 数を通知)→ ENCAPSULATED_DATA
sub-type2(チャンク本体)の順で、レース開始/リセット時に一度だけ送られる。
途中接続では拾えないため、SIM_RESET(31000)を送ってから受信し直す。

実行(WSL、Windows側でシム起動中):
  UV_PROJECT_ENVIRONMENT=.venv-host uv run python check/probe_track_reset.py --secs 15
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

MAVLINK_CMD_SIM_RESET = 31000
ENCAP_RACE_STATUS = 1
ENCAP_TRACK_INFO = 2


def reassemble_track(chunks, expected):
    """完成した transfer をゲート配列にデコード。返り値: [(gate_id, x, y, z, w, ...)]"""
    for tid, parts in chunks.items():
        if tid not in expected or len(parts) != expected[tid]:
            continue
        payload = b"".join(parts[i] for i in range(len(parts)))
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for _ in range(num_gates):
            g = struct.unpack_from("<Hfffffffff", payload)
            gates.append(g)
            payload = payload[38:]
        return num_gates, gates
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=15.0)
    ap.add_argument("--mavlink-ip", type=str, default="0.0.0.0")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--no-relay", action="store_true")
    ap.add_argument("--no-reset", action="store_true", help="RESET を送らず受信のみ")
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

        m = mavutil.mavlink
        # GCS ハートビート / TIMESYNC を少し送って生存を示す
        conn.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)

        if not args.no_reset:
            print("SIM_RESET(31000) 送信 → トラックデータ再送を誘発 ...", flush=True)
            conn.mav.command_long_send(
                conn.target_system, conn.target_component,
                MAVLINK_CMD_SIM_RESET, 0, 0, 0, 0, 0, 0, 0, 0)

        # 受信ループ
        track_chunks = defaultdict(dict)          # transfer_id -> {seq: bytes}
        expected = {}                             # transfer_id -> packet count
        counts = defaultdict(int)
        handshakes = 0
        track_result = None
        t_end = time.time() + args.secs
        next_hb = next_ts = 0.0
        while time.time() < t_end and track_result is None:
            now = time.time()
            if now >= next_hb:
                next_hb = now + 0.5
                conn.mav.heartbeat_send(m.MAV_TYPE_GCS, m.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            if now >= next_ts:
                next_ts = now + 0.1
                conn.mav.timesync_send(int(time.time_ns()), 0)

            msg = conn.recv_match(blocking=False)
            if msg is None:
                time.sleep(0.001)
                continue
            t = msg.get_type()
            if t == "BAD_DATA":
                continue
            counts[t] += 1

            if t == "DATA_TRANSMISSION_HANDSHAKE":
                handshakes += 1
                tid = msg.width
                track_chunks[tid] = {}
                expected[tid] = msg.packets
                print(f"  handshake: transfer_id={tid} packets={msg.packets}", flush=True)

            elif t == "ENCAPSULATED_DATA":
                raw = bytes(msg.data)
                if not raw:
                    continue
                if raw[0] == ENCAP_TRACK_INFO:
                    _, tid = struct.unpack_from("<BH", raw)
                    if tid in expected:
                        track_chunks[tid][msg.seqnr] = raw[3:]
                        if len(track_chunks[tid]) == expected[tid]:
                            track_result = reassemble_track(track_chunks, expected)

        # ---- 結果 ----
        print("\n" + "=" * 74)
        print(f"[受信サマリ {args.secs:.0f}s] 型 {len(counts)} 種 / handshake {handshakes} 回")
        print("=" * 74)
        for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {t:<30} {c:6d}件")

        print("\n" + "-" * 74)
        if track_result is not None:
            num_gates, gates = track_result
            print(f"✅ トラックデータ受信! ゲート数={num_gates}")
            allzero = True
            for g in gates[:8]:
                gid, x, y, z, qw, qx, qy, qz, w, h = g
                print(f"   gate {gid:2d}: NED=({x:8.2f},{y:8.2f},{z:8.2f}) "
                      f"q=({qw:.2f},{qx:.2f},{qy:.2f},{qz:.2f}) w={w:.2f} h={h:.2f}")
                if any(abs(v) > 1e-6 for v in (x, y, z, w, h)):
                    allzero = False
            if allzero:
                print("   ⚠ 値がすべて 0/null → ゲート位置は nulled(VQ2 相当)")
            else:
                print("   → ゲート位置が実値で入っている(VQ1 レガシー=トラック情報有効)")
        elif handshakes > 0:
            print("△ handshake は来たがチャンクが揃わず(時間切れ or 一部欠落)。--secs を延ばして再試行を。")
        else:
            print("❌ トラックデータの handshake すら来ない。")
            print("   → このビルドではトラック情報(GATE_INFO)は送信されていない(VQ2 相当)。")
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
