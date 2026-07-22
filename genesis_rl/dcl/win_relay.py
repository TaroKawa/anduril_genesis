# -*- coding: utf-8 -*-
"""Windows側UDPリレー(標準ライブラリのみ・Windows Pythonで実行)。

シム(DCL)はWindowsの127.0.0.1:14550(MAVLink)/5600(映像)へUDP送信するが、
WSL2(NATモード)のネイティブプロセスはWindowsループバック宛パケットを受信できず、
Docker DesktopのUDP転送は高レート映像で恒久的にフローを落とす(実測)。

そこで本スクリプトをWindows側で走らせ、WSLのIPへ直接転送する:
  映像    : 127.0.0.1:5600  → WSL:5600 (一方向)
  MAVLink : 127.0.0.1:14550 ⇄ WSL:14550 (双方向: シム側アドレスを記憶して往復)

通常は fly_dcl.py がWSLから python.exe 経由で自動起動する(手動起動も可):
  python win_relay.py --target <WSL_IP>
"""

import argparse
import select
import socket
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="WSL側のIPアドレス")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    args = ap.parse_args()

    mav_target = (args.target, args.mavlink_port)
    vid_target = (args.target, args.video_port)

    s_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    s_video.bind(("127.0.0.1", args.video_port))

    s_mav = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # シム側(受信+返信)
    s_mav.bind(("127.0.0.1", args.mavlink_port))

    s_fwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)   # WSLクライアント側(MAVLink)
    s_fwd.bind(("0.0.0.0", 0))

    # 映像転送用の送信ソケット。127.0.0.1にbindしたs_videoから外部IPへは
    # sendtoできない(ソースアドレス不一致)ため、必ず別ソケットで送る
    s_vfwd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s_vfwd.bind(("0.0.0.0", 0))

    print(f"[win_relay] video 127.0.0.1:{args.video_port} -> {vid_target[0]}:{vid_target[1]}",
          flush=True)
    print(f"[win_relay] mavlink 127.0.0.1:{args.mavlink_port} <-> {mav_target[0]}:{mav_target[1]}",
          flush=True)

    sim_addr = None
    n_vid = n_mav = 0
    last_report = time.time()
    socks = [s_video, s_mav, s_fwd]
    while True:
        ready, _, _ = select.select(socks, [], [], 1.0)
        for s in ready:
            try:
                data, addr = s.recvfrom(65536)
            except OSError:
                continue
            try:
                if s is s_video:
                    n_vid += 1
                    s_vfwd.sendto(data, vid_target)
                elif s is s_mav:
                    sim_addr = addr
                    s_fwd.sendto(data, mav_target)
                    n_mav += 1
                elif s is s_fwd and sim_addr is not None:
                    s_mav.sendto(data, sim_addr)
            except OSError:
                pass  # WSL側未起動などの一時エラーは握りつぶして継続
        if time.time() - last_report > 10.0:
            last_report = time.time()
            print(f"[win_relay] fwd video={n_vid} mav={n_mav}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
