# -*- coding: utf-8 -*-
"""録画済みFPV動画に YOLOX-x のゲート検出を重ねて可視化する(シム不要・オフライン)。

  UV_PROJECT_ENVIRONMENT=.venv-host uv run python -m genesis_rl.scripts.yolox_overlay \
      --in checkpoints/flight_dcl.mp4 --out checkpoints/yolox_overlay.mp4

緑の十字 = 検出したゲート中心(fly_dcl のHUDと同じ)。
緑の矩形 = YOLOX bbox。左上に frame/score/rel_dist を表示。
入力はどのFPV録画でもよい(640x360想定。他解像度でも動的に対応)。
"""

from __future__ import annotations

import argparse

import cv2
import imageio.v2 as imageio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="checkpoints/flight_dcl.mp4")
    ap.add_argument("--out", type=str, default="checkpoints/yolox_overlay.mp4")
    ap.add_argument("--yolox-ckpt", type=str, default=None,
                    help="YOLOX-x重み(省略時は client.DEFAULT_YOLOX_CKPT)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=0, help=">0 なら先頭N枚だけ処理")
    args = ap.parse_args()

    import torch

    from ..dcl.client import DEFAULT_YOLOX_CKPT
    from ..dcl.yolox_gate import GateYOLOX

    ckpt = args.yolox_ckpt or DEFAULT_YOLOX_CKPT
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det = GateYOLOX(ckpt, dev, conf_thre=args.conf, nms_thre=args.nms)

    reader = imageio.get_reader(args.inp)
    n = reader.count_frames()
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"input: {args.inp} ({n} frames) -> {args.out}", flush=True)

    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7)
    hits = 0
    try:
        for i in range(n):
            rgb = reader.get_data(i)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            d = det.detect(bgr, return_box=True)
            hud = bgr.copy()
            H, W = hud.shape[:2]
            if d["visible"]:
                hits += 1
                u, v = int(d["center"][0] * W), int(d["center"][1] * H)
                cv2.drawMarker(hud, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                if d.get("box"):
                    x1, y1, x2, y2 = (int(c) for c in d["box"])
                    cv2.rectangle(hud, (x1, y1), (x2, y2), (0, 255, 0), 2)
                txt = f"f={i} YOLOX score={d['score']:.2f} rel={d['rel_dist']:.2f}"
            else:
                txt = f"f={i} YOLOX no-det"
            cv2.putText(hud, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer.append_data(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB))
            if i % 200 == 0:
                print(f"  {i}/{n} (det {hits})", flush=True)
    finally:
        writer.close()
        reader.close()
    print(f"done: detected in {hits}/{n} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
