# -*- coding: utf-8 -*-
"""実シムからトラック情報(ゲート位置/向き/寸法)を受信してダンプする。

VQ1(テレメトリ有効)版でのみ `track_info`(ENCAPSULATED sub-type2 + DATA_TRANSMISSION_
HANDSHAKE)が流れる。VQ2 では null化されて来ない。受信できれば、実トラックに合わせた
Genesis コース(course.py の固定コース化)を作る土台になる(#3/#8 のギャップ解消)。

client.py の MavlinkIO がトラックを解析して shared["track"] に格納するので、それを待って
JSON に保存するだけ。win_relay は自動起動。

実行(Windows側で VQ1 シムを起動した状態で):
  UV_PROJECT_ENVIRONMENT=.venv-host uv run python check/dump_track.py --secs 20 --out runs/real_track.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=20.0, help="トラック受信を待つ秒数")
    ap.add_argument("--out", type=str, default="runs/real_track.json")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--no-relay", action="store_true")
    args = ap.parse_args()

    from genesis_rl.dcl.client import MavlinkIO, spawn_win_relay

    relay = None if args.no_relay else spawn_win_relay(args.mavlink_port, args.video_port)
    shared: dict = {}
    mav = MavlinkIO(shared, "0.0.0.0", args.mavlink_port)
    try:
        t_end = time.time() + args.secs
        print(f"{args.secs:.0f}秒 トラック情報を待ちます(VQ2では来ません)...", flush=True)
        while time.time() < t_end:
            mav.heartbeat_if_due()
            if shared.get("track"):
                break
            time.sleep(0.05)
        track = shared.get("track")
        if track:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(track, f, indent=2)
            print(f"✅ トラック受信: {track['num_gates']}ゲート → {args.out}", flush=True)
            for g in track["gates"][:5]:
                p = g["pos_ned"]
                print(f"  gate{g['gate_id']:2d}: NED=({p[0]:6.1f},{p[1]:6.1f},{p[2]:6.1f}) "
                      f"w={g['width']:.2f} h={g['height']:.2f}", flush=True)
            if track["num_gates"] > 5:
                print(f"  ... 他 {track['num_gates'] - 5} ゲート", flush=True)
        else:
            print("❌ トラック情報を受信できませんでした。", flush=True)
            print("   → 現行シムは VQ2(track_info は null化)。VQ1レガシー版で再実行してください。",
                  flush=True)
    finally:
        mav.close()
        if relay is not None:
            try:
                relay.terminate(); relay.wait(timeout=3.0)
            except Exception:
                try:
                    relay.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
