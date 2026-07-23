"""本番シム(DCL)で学習方策を飛ばす — anduril_genesis 単体で完結・Docker不要。

  # Windows側でシミュレーターを起動してから(WSLで):
  UV_PROJECT_ENVIRONMENT=.venv-host uv run python -m genesis_rl.scripts.fly_dcl \
      --ckpt checkpoints_old_resnet/best_gates.pt

  ※ UV_PROJECT_ENVIRONMENT=.venv-host は、Dockerコンテナ(学習)が使う .venv を
    壊さずホスト専用venvを使うための指定。
  ※ シムはWindowsの127.0.0.1:14550/5600へUDP送信するため、本スクリプトが
    Windows側リレー(genesis_rl/dcl/win_relay.py)を python.exe interop で
    自動起動し、WSLのIPへ転送させる(--no-relayで無効化=手動起動)。

Ctrl+Cで終了。受信FPV映像(HUD付き)は --out のmp4へ逐次保存される。
チェックポイントの新旧アーキテクチャ(ResNet18+MLP / DINOv2+時系列)は自動判別。
ゲート検出は既定でYOLOX-x(YOLOX_outputs_x/yolox_x_custom/best_ckpt.pth)。緑の
十字HUDは検出したゲート中心。重みが無い場合は自動でHSV色検出にフォールバック。
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--mavlink-ip", type=str, default="0.0.0.0",
                    help="MAVLink待ち受けバインドIP(コンテナ内はDockerゲートウェイ経由なので0.0.0.0必須)")
    ap.add_argument("--mavlink-port", type=int, default=14550)
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--out", type=str, default="checkpoints/flight_dcl.mp4",
                    help="受信FPVの録画先mp4(''で無効)")
    ap.add_argument("--max-sec", type=float, default=0.0, help=">0なら指定秒数で自動終了")
    ap.add_argument("--no-reset-on-collision", action="store_true")
    ap.add_argument("--no-relay", action="store_true",
                    help="Windows側リレーの自動起動を無効化(手動起動する場合)")
    from ..dcl.client import DEFAULT_YOLOX_CKPT
    ap.add_argument("--gate-detector", choices=["yolox", "hsv"], default="yolox",
                    help="ゲート検出方式(既定=yolox。重み欠落時は自動でhsvへフォールバック)")
    ap.add_argument("--yolox-ckpt", type=str, default=DEFAULT_YOLOX_CKPT,
                    help="YOLOX-x 重み(best_ckpt.pth)のパス")
    ap.add_argument("--record-dir", type=str, default="",
                    help="指定すると1決定ごとに観測/行動/テレメトリを DIR/ に同期ログ"
                         "(steps.jsonl + frames/)。sim-to-sim分析・Phase2用。")
    ap.add_argument("--sysid", action="store_true",
                    help="方策を外し、既知のレートステップ列を送って開ループでプラントの"
                         "レートゲイン+符号を同定する(--record-dir と併用。--no-reset-on-collision推奨)")
    ap.add_argument("--gate-area-max", type=float, default=0.0,
                    help="rel_dist=1-bbox面積/この値。0で契約既定150000。実bboxはGenesis投影より"
                         "小さくrel_distが遠側に張り付くため、下げる(例25000)と接近で早く下がる")
    args = ap.parse_args()

    from ..dcl.client import run

    run(ckpt=args.ckpt, mavlink_ip=args.mavlink_ip, mavlink_port=args.mavlink_port,
        video_port=args.video_port, out_mp4=args.out or None, max_sec=args.max_sec,
        reset_on_collision=not args.no_reset_on_collision, relay=not args.no_relay,
        gate_detector=args.gate_detector, yolox_ckpt=args.yolox_ckpt,
        record_dir=args.record_dir or None, sysid=args.sysid,
        gate_area_max=(args.gate_area_max or None))


if __name__ == "__main__":
    main()
